"""Hybrid intelligence: classical illumination estimation + deep restoration + quality-aware fusion.

Pipeline
  1. Analyse the image and estimate illumination L (classical, guided filter).
  2. Candidates:
       C  - classical adaptive Retinex pipeline
       D  - Zero-DCE++ output; for extremely dark inputs the net is applied twice
       G  - Zero-DCE++ applied to the Retinex pre-brightened input (illumination-guided DL)
  3. Per-pixel quality weights (well-exposedness, local contrast, saturation, minus a
     noise penalty) decide how much each candidate contributes. Fusion is done in a
     Laplacian pyramid so there are no seams or halos.
  4. Illumination-aware edge-preserving denoise, colour correction and highlight
     roll-off guard against noise amplification, colour shifts and overexposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from lumen.classical import AdaptiveConfig, ClassicalEnhancer, analyze_image, select_params
from lumen.classical.contrast import gray_world, highlight_compress
from lumen.classical.filters import noise_adaptive_denoise
from lumen.classical.illumination import estimate_illumination, retinex_enhance
from lumen.deep import DeepEnhancer
from lumen.utils import luminance

HYBRID_COMPONENTS = ("classical_branch", "deep_branch", "guided_branch", "fusion", "post_denoise", "post_color")


# ----------------------------------------------------------------------------- fusion
def quality_weights(img: np.ndarray, noise_map: np.ndarray | None = None, sigma: float = 0.22) -> np.ndarray:
    """Mertens-style per-pixel quality: exposure x contrast x saturation (x noise penalty)."""
    y = luminance(img)
    exposure = np.exp(-((img - 0.5) ** 2) / (2 * sigma**2)).prod(axis=2)
    contrast = np.abs(cv2.Laplacian(y, cv2.CV_32F, ksize=3))
    contrast = cv2.GaussianBlur(contrast, (0, 0), 2) + 1e-3
    saturation = img.std(axis=2) + 1e-3
    w = exposure * contrast**0.5 * saturation**0.5
    if noise_map is not None:
        w = w / (1.0 + 20.0 * noise_map)
    return w.astype(np.float32) + 1e-12


def _local_noise(img: np.ndarray) -> np.ndarray:
    """Rough per-pixel noise energy: high-frequency residual on flat areas."""
    y = luminance(img)
    resid = np.abs(y - cv2.medianBlur((y * 255).astype(np.uint8), 3).astype(np.float32) / 255.0)
    grad = cv2.GaussianBlur(np.abs(cv2.Sobel(y, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(y, cv2.CV_32F, 0, 1)), (0, 0), 3)
    return cv2.GaussianBlur(resid, (0, 0), 3) / (grad + 0.05)


def _gauss_pyr(x, levels):
    pyr = [x]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _lap_pyr(x, levels):
    g = _gauss_pyr(x, levels)
    lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(levels - 1)]
    return lap + [g[-1]]


def pyramid_fusion(images: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    h, w = images[0].shape[:2]
    levels = int(np.clip(np.floor(np.log2(min(h, w))) - 3, 1, 6))
    wsum = np.sum(weights, axis=0)
    fused = None
    for img, wt in zip(images, weights):
        gw = _gauss_pyr((wt / wsum).astype(np.float32), levels)
        lp = _lap_pyr(img.astype(np.float32), levels)
        contrib = [l * g[..., None] for l, g in zip(lp, gw)]
        fused = contrib if fused is None else [f + c for f, c in zip(fused, contrib)]
    out = fused[-1]
    for lev in reversed(fused[:-1]):
        out = cv2.pyrUp(out, dstsize=(lev.shape[1], lev.shape[0])) + lev
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------- enhancer
@dataclass
class HybridOutput:
    image: np.ndarray
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)


class HybridEnhancer:
    def __init__(self, deep: DeepEnhancer | None = None, disable: set[str] | None = None,
                 post_config: AdaptiveConfig | None = None):
        self.deep = deep or DeepEnhancer()
        self.disable = set(disable or ())
        # The classical candidate uses the tuned configs/classical.yaml...
        self.classical = ClassicalEnhancer(disable={d for d in self.disable if d in ("adaptive",)})
        # ...but the hybrid's own post-processing keeps the hand-set defaults: the tuned constants were
        # optimised for the classical bilateral pipeline and over-smooth the fused result here
        # (eval15 LPIPS 0.327 -> 0.410). The hybrid gets its own tuning in Phase 3.
        self.post_config = post_config or AdaptiveConfig()

    def __call__(self, img: np.ndarray) -> HybridOutput:
        stats = analyze_image(img)
        params = select_params(stats, self.post_config)
        L = estimate_illumination(img, params.illum_radius, params.illum_eps)
        steps: dict[str, np.ndarray] = {"illumination": np.repeat(L[..., None], 3, -1)}

        candidates: dict[str, np.ndarray] = {}
        if "classical_branch" not in self.disable:
            candidates["classical"] = self.classical(img).image
        if "deep_branch" not in self.disable:
            iters = 2 if stats.category == "extremely_dark" else 1
            d, curve = self.deep(img, iterations=iters)
            candidates["deep"] = d
            steps["curve_map"] = np.repeat(np.clip(curve, 0, 1)[..., None], 3, -1)
        if "guided_branch" not in self.disable:
            # Classical illumination drives a mild pre-brightening, DL restores the rest.
            pre = retinex_enhance(img, L, float(np.sqrt(params.gamma)))
            candidates["guided"], _ = self.deep(pre)
        if not candidates:
            candidates["classical"] = self.classical(img).image
        steps.update({f"cand_{k}": v for k, v in candidates.items()})

        names, imgs = list(candidates), list(candidates.values())
        if len(imgs) == 1:
            fused, share = imgs[0], {names[0]: 1.0}
        elif "fusion" in self.disable:  # ablation: pick single best candidate by mean quality
            scores = [float(quality_weights(c).mean()) for c in imgs]
            best = int(np.argmax(scores))
            fused, share = imgs[best], {n: float(i == best) for i, n in enumerate(names)}
        else:
            weights = [quality_weights(c, _local_noise(c)) for c in imgs]
            total = np.sum(weights, axis=0)
            share = {n: round(float((w / total).mean()), 3) for n, w in zip(names, weights)}
            fused = pyramid_fusion(imgs, weights)
        steps["fused"] = fused

        x = fused
        if "post_denoise" not in self.disable:
            x = noise_adaptive_denoise(x, L, "guided", max(params.denoise_strength, 0.02), 4)
            steps["post_denoise"] = x
        if "post_color" not in self.disable:
            x = gray_world(x, params.wb_strength * 0.7)
            steps["post_color"] = x
        x = highlight_compress(x)

        info = {"stats": stats.as_dict(), "params": params.as_dict(), "fusion_share": share,
                "deep_pretrained": self.deep.pretrained}
        return HybridOutput(x.astype(np.float32), steps, info)
