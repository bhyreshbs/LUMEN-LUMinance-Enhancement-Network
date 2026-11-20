"""Noise reduction for very noisy photos: runs on the Natural and Highlight-safe outputs (not on Quick).

Problem (see docs/NOISE_REDUCTION.md and eval_P1_T66.md section 10.4): on very noisy inputs Retinexformer amplifies
the noise instead of removing it. On the real LOL-v2 pairs with input noise sigma 0.043-0.053 its outputs reach
SSIM 0.16 on average, and the QGH noise step barely helps.

Fix: when, and only when, the INPUT is very noisy, denoise the model output with colour non-local means whose
strength follows the output's own noise level:

    w   = smoothstep(sigma_in; gate_low, gate_high)       sigma_in = the input noise the app already measures
    h   = strength * sigma_out * 255                       sigma_out = noise estimate of the model output
    out = (1 - w) * r + w * NLM(r; h)

Below gate_low (every benchmark image, every real phone photo we have) the image is returned bit for bit: the step
cannot change low-noise results. Gates and strength were chosen on development data only.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from lumen.hybrid.quality_guided import smoothstep
from lumen.metrics.quality import noise_level


@dataclass(frozen=True)
class NoiseReductionConfig:
    gate_low: float = 0.030      # input noise sigma below which nothing changes (bit-exact)
    gate_high: float = 0.040     # full strength above this; smooth ramp in between
    strength: float = 0.8        # NLM filter strength h = strength * output noise sigma (in 0-255 units)
    template: int = 7            # NLM template window (px, odd)
    search: int = 21             # NLM search window (px, odd)

    def as_dict(self):
        return asdict(self)


def nlm_colour(img: np.ndarray, h: float, template: int = 7, search: int = 21) -> np.ndarray:
    """OpenCV colour non-local means (luma and chroma at the same strength) on a float RGB image in [0, 1]."""
    u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), None, float(h), float(h), template, search)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def reduce_noise(r: np.ndarray, input_sigma: float, cfg: NoiseReductionConfig | None = None):
    """Return (image, info). r = model output (float RGB); input_sigma = noise_level() of the low-light input."""
    cfg = cfg or NoiseReductionConfig()
    t0 = time.perf_counter()
    w = float(smoothstep(np.float32(input_sigma), cfg.gate_low, cfg.gate_high))
    info = {"applied": False, "input_sigma": round(float(input_sigma), 4), "weight": round(w, 3)}
    if w <= 0.0:
        return r, info
    sigma_out = noise_level(r)
    h = cfg.strength * sigma_out * 255.0
    d = nlm_colour(r, h, cfg.template, cfg.search)
    out = d if w >= 1.0 else ((1.0 - w) * r + w * d).astype(np.float32)
    info.update(applied=True, output_sigma=round(sigma_out, 4), h=round(h, 2),
                ms=round((time.perf_counter() - t0) * 1000, 1))
    return out, info


class NoiseReductionConfigError(ValueError):
    """The noise reduction configuration is missing or invalid."""


DEFAULT_NR_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "noise_reduction.yaml"


def nr_config_path() -> Path:
    return Path(os.environ.get("LUMEN_NR_CONFIG", DEFAULT_NR_CONFIG_PATH))


def load_nr_config(path: str | Path | None = None) -> NoiseReductionConfig:
    """The selected config: `path`, else $LUMEN_NR_CONFIG, else configs/noise_reduction.yaml. Never falls back to
    defaults silently: a missing or invalid file raises NoiseReductionConfigError."""
    import yaml

    p = Path(path) if path else nr_config_path()
    if not p.is_file():
        raise NoiseReductionConfigError(f"Noise reduction config not found at {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        params = data["params"]
        known = {f.name for f in fields(NoiseReductionConfig)}
        if set(params) - known:
            raise ValueError(f"unknown keys {sorted(set(params) - known)}")
        ints = {"template", "search"}
        cfg = NoiseReductionConfig(**{k: (int(v) if k in ints else float(v)) for k, v in params.items()})
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as e:
        raise NoiseReductionConfigError(f"Noise reduction config {p} is invalid: {e}") from e
    if not (0.0 < cfg.gate_low < cfg.gate_high < 1.0 and 0.0 < cfg.strength <= 5.0
            and cfg.template % 2 == 1 and cfg.search % 2 == 1 and 1 <= cfg.template <= cfg.search <= 51):
        raise NoiseReductionConfigError(f"Noise reduction config {p} has out-of-range values: {cfg}")
    return cfg
