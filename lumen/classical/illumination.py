"""Illumination estimation and Retinex decomposition  I = R * L."""

from __future__ import annotations

import cv2
import numpy as np

from lumen.classical.filters import guided_filter


def estimate_illumination(img: np.ndarray, radius: int = 15, eps: float = 1e-3) -> np.ndarray:
    """LIME-style illumination map: max-RGB initial estimate refined with a guided filter.

    The guided filter keeps illumination smooth inside regions but sharp at strong
    edges, which is what prevents halo artifacts in the subsequent Retinex step.
    """
    initial = img.max(axis=2).astype(np.float32)
    refined = guided_filter(initial, initial, radius, eps)
    return np.clip(refined, 1e-3, 1.0)


def decompose(img: np.ndarray, illumination: np.ndarray) -> np.ndarray:
    """Reflectance R = I / L (clipped)."""
    return np.clip(img / illumination[..., None], 0.0, 1.0)


def retinex_enhance(img: np.ndarray, illumination: np.ndarray, gamma: float) -> np.ndarray:
    """Illumination-adjusted Retinex: R * L^gamma  ==  I * L^(gamma - 1).

    gamma < 1 brightens dark regions much more than bright ones (the tone curve is
    applied to illumination only, so reflectance / texture is preserved).
    """
    gain = np.power(illumination, gamma - 1.0)[..., None]
    return np.clip(img * gain, 0.0, 1.0).astype(np.float32)


def multiscale_retinex(img: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    """Classic MSR (Jobson et al.) - kept as an alternative / comparison baseline."""
    h, w = img.shape[:2]
    scale = min(1.0, 256.0 / max(h, w))  # blur at low resolution for speed
    log_i = np.log1p(img * 255.0)
    out = np.zeros_like(img)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for s in sigmas:
        blur = cv2.GaussianBlur(small, (0, 0), max(s * scale, 0.5))
        blur = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
        out += log_i - np.log1p(blur * 255.0)
    out /= len(sigmas)
    lo, hi = np.percentile(out, (1, 99))
    return np.clip((out - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)
