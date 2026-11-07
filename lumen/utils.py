"""Image I/O and small helpers. All internal images are float32 RGB in [0, 1], HxWx3."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Accept gray, RGB or RGBA arrays and return 3-channel RGB."""
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        return img[..., :3]
    return img


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII Windows paths
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return to_float(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(str(path))


def luminance(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def limit_size(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is <= max_side (keeps latency bounded)."""
    h, w = img.shape[:2]
    s = max(h, w)
    if max_side <= 0 or s <= max_side:
        return img
    scale = max_side / s
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = (time.perf_counter() - t0) * 1000.0  # milliseconds
