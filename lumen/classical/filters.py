"""Edge-preserving filters: guided filter (He et al., 2010) and bilateral filtering."""

from __future__ import annotations

import cv2
import numpy as np


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Gray-guide guided filter. `src` may be HxW or HxWxC (filtered per channel)."""
    guide = guide.astype(np.float32)
    if guide.ndim == 3:
        guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
    if src.ndim == 3:
        return np.stack([guided_filter(guide, src[..., c], radius, eps) for c in range(src.shape[2])], -1)
    src = src.astype(np.float32)
    mean_i, mean_p = _box(guide, radius), _box(src, radius)
    cov_ip = _box(guide * src, radius) - mean_i * mean_p
    var_i = _box(guide * guide, radius) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, radius) * guide + _box(b, radius)


def bilateral(img: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Bilateral filter on float images (sigma_color in [0,1] intensity units)."""
    return cv2.bilateralFilter(img.astype(np.float32), d, sigma_color, sigma_space)


def noise_adaptive_denoise(
    img: np.ndarray,
    illumination: np.ndarray,
    method: str = "bilateral",
    strength: float = 0.08,
    radius: int = 4,
) -> np.ndarray:
    """Denoise more where the scene was dark (noise is amplified most there).

    The blend mask is (1 - illumination), so bright, already-clean regions keep
    their original detail while dark regions get the filtered version.
    """
    if strength <= 0:
        return img
    if method == "guided":
        filtered = guided_filter(img, img, radius, strength**2)
    else:
        filtered = bilateral(img, d=2 * radius + 1, sigma_color=strength, sigma_space=radius * 1.5)
    mask = np.sqrt(np.clip(1.0 - illumination, 0.0, 1.0))[..., None]
    return np.clip(mask * filtered + (1.0 - mask) * img, 0.0, 1.0).astype(np.float32)
