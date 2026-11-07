"""Retinexformer (Cai et al., ICCV 2023): one-stage Retinex-based transformer for low-light enhancement.

Self-contained port of `basicsr/models/archs/RetinexFormer_arch.py` from
https://github.com/caiyuanhao1998/Retinexformer (MIT licence, (c) Yuanhao Cai et al.).
Changes from the original: no basicsr / einops dependency (einops.rearrange replaced by reshape/permute),
unused helpers removed, type hints and comments added. Module and parameter names are unchanged, so the
official checkpoints load with `strict=True` (their files store the weights under the "params" key).

Pipeline (one stage):  illumination estimator -> lit-up image  I_lu = I * L_hat + I  ->
illumination-guided transformer (IGT, a U-shaped denoiser whose attention is modulated by the
illumination features) -> restored image.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper-parameters of the released LOL-v1 / LOL-v2 checkpoints (Options/RetinexFormer_LOL_*.yml)
LOL_CONFIG = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])


def trunc_normal_(tensor: torch.Tensor, mean=0.0, std=1.0, a=-2.0, b=2.0) -> torch.Tensor:
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lo, hi = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Illumination_Estimator(nn.Module):  # noqa: N801 - name kept for checkpoint compatibility
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        """img [b,3,h,w] -> (illumination features [b,c,h,w], light-up map [b,3,h,w])."""
        mean_c = img.mean(dim=1, keepdim=True)
        x_1 = self.conv1(torch.cat([img, mean_c], dim=1))
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(nn.Module):  # noqa: N801
    """Illumination-guided multi-head self-attention (attention over channels, so cost is linear in h*w)."""

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def _heads(self, t):  # 'b n (h d) -> b h n d'
        b, n, _ = t.shape
        return t.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)

    def forward(self, x_in, illu_fea_trans):
        """x_in, illu_fea_trans: [b,h,w,c] -> [b,h,w,c]"""
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp, k_inp, v_inp = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = self._heads(q_inp), self._heads(k_inp), self._heads(v_inp)
        illu_attn = self._heads(illu_fea_trans.flatten(1, 2))
        v = v * illu_attn
        q, k, v = q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)  # b,heads,d,hw
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale  # A = K^T Q  (d x d)
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):  # [b,h,w,c]
        return self.net(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)


class IGAB(nn.Module):
    """Illumination-guided attention block: num_blocks x (IG-MSA + pre-norm feed-forward), residual."""

    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.ModuleList([IG_MSA(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))])
             for _ in range(num_blocks)]
        )

    def forward(self, x, illu_fea):  # [b,c,h,w]
        x = x.permute(0, 2, 3, 1)
        illu = illu_fea.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class Denoiser(nn.Module):
    """U-shaped illumination-guided transformer (encoder -> bottleneck -> decoder, 2 levels)."""

    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=(2, 4, 4)):
        super().__init__()
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
            ]))
            dim_level *= 2

        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                     heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # unused in forward; kept as in original
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)
        fea_encoder, illu_fea_list = [], []
        for igab, fea_down, illu_down in self.encoder_layers:
            fea = igab(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = fea_down(fea)
            illu_fea = illu_down(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (fea_up, fusion, igab) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = igab(fea, illu_fea_list[self.level - 1 - i])

        return self.mapping(fea) + x


class RetinexFormer_Single_Stage(nn.Module):  # noqa: N801
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=(1, 1, 1)):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                 num_blocks=num_blocks)

    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img  # Retinex "light-up"
        return self.denoiser(input_img, illu_fea)


class RetinexFormer(nn.Module):
    """Input/output: [b,3,h,w] in [0,1]; h and w must be multiples of 4 (two 2x downsamplings)."""

    PAD_MULTIPLE = 4

    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=(1, 1, 1)):
        super().__init__()
        self.stage = stage
        self.body = nn.Sequential(*[
            RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat,
                                       level=2, num_blocks=list(num_blocks))
            for _ in range(stage)
        ])

    def forward(self, x):
        return self.body(x)


def build_lol_model() -> RetinexFormer:
    return RetinexFormer(**LOL_CONFIG)


def load_checkpoint(model: nn.Module, path, map_location="cpu") -> nn.Module:
    """Load an official ('params' key) or LUMEN fine-tuned ('model'/'params' or raw) checkpoint, strictly."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(state, dict):
        for k in ("params_ema", "params", "model"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    model.load_state_dict(state, strict=True)
    return model
