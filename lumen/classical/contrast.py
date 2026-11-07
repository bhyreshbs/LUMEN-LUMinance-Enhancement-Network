"""Contrast (CLAHE) and colour correction."""

from __future__ import annotations

import cv2
import numpy as np

from lumen.utils import to_float, to_uint8


def clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the L channel of CIELAB so chroma is untouched."""
    lab = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return to_float(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def gray_world(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Gray-world white balance, blended by `strength` in [0, 1]."""
    means = img.reshape(-1, 3).mean(0) + 1e-6
    gains = means.mean() / means
    gains = 1.0 + strength * (gains - 1.0)
    return np.clip(img * gains, 0.0, 1.0).astype(np.float32)


def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return img
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 1.0)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


def highlight_compress(img: np.ndarray, knee: float = 0.85) -> np.ndarray:
    """Soft-knee roll-off above `knee` to prevent clipped / overexposed highlights."""
    out = img.copy()
    m = out > knee
    span = 1.0 - knee
    out[m] = knee + span * np.tanh((out[m] - knee) / span)
    return out


def percentile_stretch(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    a, b = np.percentile(img, (lo, hi))
    return np.clip((img - a) / (b - a + 1e-6), 0.0, 1.0).astype(np.float32)
