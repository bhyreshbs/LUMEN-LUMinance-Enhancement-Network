"""Evaluation metrics required by P1.

Full-reference : PSNR, SSIM, LPIPS (need ground truth, e.g. LOL "high" images)
No-reference   : NIQE (via pyiqa)
Structure      : noise level, edge preservation, colour shift, overexposure ratio
Heavy models (LPIPS, NIQE) are created lazily and cached.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma

from lumen.utils import luminance


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(structural_similarity(gt, pred, data_range=1.0, channel_axis=-1))


def _torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(img: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float().to(_torch_device())


@lru_cache(maxsize=1)
def _lpips_model():
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).to(_torch_device()).eval()


def lpips_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    import torch

    with torch.inference_mode():
        return float(_lpips_model()(_to_tensor(pred) * 2 - 1, _to_tensor(gt) * 2 - 1).item())


@lru_cache(maxsize=1)
def _niqe_model():
    import pyiqa

    return pyiqa.create_metric("niqe", device=_torch_device())


def niqe(img: np.ndarray) -> float | None:
    """Lower is better. Returns None if pyiqa is unavailable."""
    try:
        import torch

        with torch.inference_mode():
            return float(_niqe_model()(_to_tensor(img)).item())
    except Exception:  # noqa: BLE001 - metric is optional
        return None


def noise_level(img: np.ndarray) -> float:
    """Estimated Gaussian noise sigma (lower = better noise suppression)."""
    return float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))


def _grad_mag(img: np.ndarray) -> np.ndarray:
    y = luminance(img)
    return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1))


def edge_preservation(pred: np.ndarray, ref: np.ndarray) -> float:
    """Pearson correlation of gradient magnitudes (1 = edges fully preserved).

    `ref` is the ground truth when available, otherwise the (normalised) input.
    """
    a, b = _grad_mag(pred).ravel(), _grad_mag(ref).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def colour_shift(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean CIELAB chroma (a*, b*) difference - a lightness-independent colour-shift measure."""
    lab_p = cv2.cvtColor(pred.astype(np.float32), cv2.COLOR_RGB2LAB)
    lab_r = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_RGB2LAB)
    return float(np.linalg.norm(lab_p[..., 1:] - lab_r[..., 1:], axis=-1).mean())


def overexposure_ratio(img: np.ndarray, thresh: float = 0.98) -> float:
    return float((img.max(axis=2) >= thresh).mean())
