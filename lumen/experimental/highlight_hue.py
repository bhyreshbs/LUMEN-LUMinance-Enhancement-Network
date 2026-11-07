"""Experimental highlight hue preservation (not used by any production mode).

Problem (see docs/HIGHLIGHT_HUE_STUDY.md): where the input is already (nearly) clipped - a bright window or a
light source in a dark room - Retinexformer's illumination division shifts the colour channels unequally, so
white turns orange or yellow and the region often gets darker. QGH's highlight step only rolls off luminance.

Fix: in a soft mask of near-clipped *input* pixels, take the colour from the input (its chromaticity is
reliable because the pixel is bright) and keep the model's brightness, with an optional floor so the
region is not darker than the input:

    mask  m  = feather(smoothstep(max_channel(x); start, full))
    colour u = (1 - s) * r / Y(r)  +  s * x / Y(x)        s = chroma_strength
    target   = u * max(Y(r), lum_keep * Y(x)),  scaled so no channel exceeds 1
    out      = r + m * (target - r)

Outside the mask the image is returned bit for bit. Nothing is learned and the model is not touched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from lumen.hybrid.quality_guided import smoothstep
from lumen.utils import luminance

_EPS = 1e-4


@dataclass(frozen=True)
class HighlightHueConfig:
    start: float = 0.88          # input max-channel ramp: no correction below this
    full: float = 0.97           # full correction above this
    sigma: float = 3.0           # mask feathering (px)
    chroma_strength: float = 1.0  # 0 = keep model colour, 1 = take the input colour
    lum_keep: float = 0.0        # output luminance in the mask >= lum_keep * input luminance
    min_coverage: float = 0.0005  # mean mask below this -> identity

    def as_dict(self):
        return asdict(self)


def highlight_hue_fix(r: np.ndarray, x: np.ndarray, cfg: HighlightHueConfig | None = None):
    """Return (corrected image, mask, info). r = model output, x = low-light input (same size, float RGB)."""
    cfg = cfg or HighlightHueConfig()
    m = smoothstep(x.max(axis=2), cfg.start, cfg.full)
    if cfg.sigma > 0:
        m = np.clip(cv2.GaussianBlur(m, (0, 0), cfg.sigma), 0.0, 1.0)
    cov = float(m.mean())
    if cov < cfg.min_coverage:
        return r, m, {"applied": False, "coverage": round(cov, 5)}
    yr, yx = luminance(r)[..., None], luminance(x)[..., None]
    u = (1 - cfg.chroma_strength) * (r / (yr + _EPS)) + cfg.chroma_strength * (x / (yx + _EPS))
    target = u * np.maximum(yr, cfg.lum_keep * yx)
    peak = target.max(axis=2, keepdims=True)
    target = np.where(peak > 1.0, target / peak, target)
    out = r + m[..., None] * (np.clip(target, 0.0, 1.0) - r)
    return np.clip(out, 0.0, 1.0).astype(np.float32), m, {"applied": True, "coverage": round(cov, 5)}
