"""Zero-DCE++ (Li, Guo, Loy - TPAMI 2021): zero-reference deep curve estimation.

Layer names mirror the official implementation (github.com/Li-Chongyi/Zero-DCE_extension)
so the released pretrained snapshot can be loaded directly and then fine-tuned.
The network predicts a per-pixel curve parameter map A; the image is enhanced by
iterating the quadratic light-enhancement curve  x <- x + A * (x^2 - x).
Zero-reference losses mean it trains on low-light images only (no ground truth needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDN_Tem(nn.Module):
    """Depthwise-separable conv (depthwise 3x3 + pointwise 1x1)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEpp(nn.Module):
    def __init__(self, scale_factor: int = 1, n_filters: int = 32, n_iter: int = 8):
        super().__init__()
        self.scale_factor = scale_factor
        self.n_iter = n_iter
        f = n_filters
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = CSDN_Tem(3, f)
        self.e_conv2 = CSDN_Tem(f, f)
        self.e_conv3 = CSDN_Tem(f, f)
        self.e_conv4 = CSDN_Tem(f, f)
        self.e_conv5 = CSDN_Tem(f * 2, f)
        self.e_conv6 = CSDN_Tem(f * 2, f)
        self.e_conv7 = CSDN_Tem(f * 2, 3)

    def enhance(self, x, a):
        for _ in range(self.n_iter):
            x = x + a * (x * x - x)
        return x

    def forward(self, x):
        s = self.scale_factor
        x_down = x if s == 1 else F.interpolate(x, scale_factor=1 / s, mode="bilinear", align_corners=False)
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        a = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if s != 1:
            a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.enhance(x, a), a


# ----------------------------------------------------------------------------- losses
class ColorConstancyLoss(nn.Module):
    def forward(self, x):
        r, g, b = x.mean((2, 3), keepdim=True).split(1, dim=1)
        return torch.sqrt((r - g) ** 2 + (r - b) ** 2 + (b - g) ** 2 + 1e-8).mean()


class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = {
            "left": [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            "right": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            "up": [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            "down": [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        }
        for name, w in k.items():
            self.register_buffer(name, torch.tensor(w, dtype=torch.float32)[None, None])
        self.pool = nn.AvgPool2d(4)

    def forward(self, enhanced, original):
        o = self.pool(original.mean(1, keepdim=True))
        e = self.pool(enhanced.mean(1, keepdim=True))
        loss = 0
        for name in ("left", "right", "up", "down"):
            k = getattr(self, name)
            loss = loss + (F.conv2d(o, k, padding=1) - F.conv2d(e, k, padding=1)) ** 2
        return loss.mean()


class ExposureLoss(nn.Module):
    def __init__(self, patch: int = 16, level: float = 0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch)
        self.level = level

    def forward(self, x):
        return ((self.pool(x.mean(1, keepdim=True)) - self.level) ** 2).mean()


class TVLoss(nn.Module):
    def forward(self, a):
        h = (a[:, :, 1:, :] - a[:, :, :-1, :]).pow(2).mean()
        w = (a[:, :, :, 1:] - a[:, :, :, :-1]).pow(2).mean()
        return 2 * (h + w)


class ZeroDCELoss(nn.Module):
    """Weighted sum used by the official Zero-DCE++ training script."""

    def __init__(self, exposure_level: float = 0.6, w_tv=1600.0, w_spa=1.0, w_col=5.0, w_exp=10.0):
        super().__init__()
        self.col, self.spa = ColorConstancyLoss(), SpatialConsistencyLoss()
        self.exp, self.tv = ExposureLoss(16, exposure_level), TVLoss()
        self.w = (w_tv, w_spa, w_col, w_exp)

    def forward(self, original, enhanced, a):
        w_tv, w_spa, w_col, w_exp = self.w
        parts = {
            "tv": w_tv * self.tv(a),
            "spa": w_spa * self.spa(enhanced, original),
            "col": w_col * self.col(enhanced),
            "exp": w_exp * self.exp(enhanced),
        }
        return sum(parts.values()), {k: float(v) for k, v in parts.items()}
