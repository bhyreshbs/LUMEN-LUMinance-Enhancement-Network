"""Noise reduction for grainy photos: runs on the Natural and Highlight-safe outputs (not on Quick).

Problem (see docs/NOISE_REDUCTION.md and eval_P1_T66.md section 10.4): Retinexformer amplifies sensor noise instead of
removing it. On the real LOL-v2 pairs with input noise sigma 0.043-0.053 its outputs reach SSIM 0.16 on average; on
moderately noisy photos (sigma about 0.01-0.03, e.g. the app's coffee example) the grain stays clearly visible.

Two filters, chosen by the INPUT noise (the estimate the app already reports as "grain"):

    moderate grain (sigma_in from about 0.010): the detail-protecting noise-aware filter of docs/NOISE_STUDY.md
        (QGH-NA v2: two-scale grain/texture separation, texture relief, chroma only where luma grain was found)
    heavy grain (sigma_in from 0.030, full at 0.040): colour non-local means, h = strength * sigma_out * 255

    w_mod   = smoothstep(sigma_in; moderate_low, moderate_high)
    w_heavy = smoothstep(sigma_in; gate_low, gate_high)
    mid     = (1 - w_mod) * r + w_mod * NA(r)
    out     = (1 - w_heavy) * mid + w_heavy * NLM(r)

Below moderate_low (every benchmark image, every real phone photo we have) the image is returned bit for bit. Above
gate_high the output is exactly the heavy-grain filter. Without a `moderate` block only the heavy-grain filter exists.
Gates and strengths were chosen on development data only.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

from lumen.hybrid.quality_guided import smoothstep
from lumen.metrics.quality import noise_level


@dataclass(frozen=True)
class NoiseReductionConfig:
    gate_low: float = 0.030      # heavy grain: NLM starts at this input noise sigma
    gate_high: float = 0.040     # ... and is at full strength above this (smooth ramp in between)
    strength: float = 0.8        # NLM filter strength h = strength * output noise sigma (in 0-255 units)
    template: int = 7            # NLM template window (px, odd)
    search: int = 21             # NLM search window (px, odd)
    moderate_low: float = 0.0    # moderate grain: the noise-aware filter starts here (0 = no moderate filter)
    moderate_high: float = 0.0   # ... full above this
    moderate: dict = field(default_factory=dict)   # NoiseAwareConfig parameters (lumen/experimental/noise_aware.py)

    def as_dict(self):
        return asdict(self)

    @property
    def floor(self) -> float:
        """Input noise below which the image is returned bit for bit."""
        return self.moderate_low if self.moderate else self.gate_low


def nlm_colour(img: np.ndarray, h: float, template: int = 7, search: int = 21) -> np.ndarray:
    """OpenCV colour non-local means (luma and chroma at the same strength) on a float RGB image in [0, 1]."""
    u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), None, float(h), float(h), template, search)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _moderate_filter(r: np.ndarray, params: dict) -> np.ndarray:
    from lumen.experimental.noise_aware import NoiseAwareConfig, noise_aware_denoise

    return noise_aware_denoise(r, NoiseAwareConfig.from_dict(params))[0]


def reduce_noise(r: np.ndarray, input_sigma: float, cfg: NoiseReductionConfig | None = None):
    """Return (image, info). r = model output (float RGB); input_sigma = noise_level() of the low-light input."""
    cfg = cfg or NoiseReductionConfig()
    t0 = time.perf_counter()
    s = np.float32(input_sigma)
    w_heavy = float(smoothstep(s, cfg.gate_low, cfg.gate_high))
    w_mod = float(smoothstep(s, cfg.moderate_low, cfg.moderate_high)) if cfg.moderate else 0.0
    info = {"applied": False, "input_sigma": round(float(input_sigma), 4), "weight": round(w_heavy, 3),
            "moderate_weight": round(w_mod, 3)}
    if w_heavy <= 0.0 and w_mod <= 0.0:
        return r, info
    out = r
    if w_heavy < 1.0 and w_mod > 0.0:
        m = _moderate_filter(r, cfg.moderate)
        out = m if w_mod >= 1.0 else ((1.0 - w_mod) * r + w_mod * m).astype(np.float32)
    if w_heavy > 0.0:
        sigma_out = noise_level(r)
        h = cfg.strength * sigma_out * 255.0
        d = nlm_colour(r, h, cfg.template, cfg.search)
        out = d if w_heavy >= 1.0 else ((1.0 - w_heavy) * out + w_heavy * d).astype(np.float32)
        info.update(output_sigma=round(sigma_out, 4), h=round(h, 2))
    level = "heavy" if w_heavy >= 1.0 else "mixed" if w_heavy > 0.0 else "moderate"
    info.update(applied=True, level=level, ms=round((time.perf_counter() - t0) * 1000, 1))
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
        params = dict(data["params"])
        moderate = data.get("moderate") or {}
        known = {f.name for f in fields(NoiseReductionConfig)} - {"moderate"}
        if set(params) - known:
            raise ValueError(f"unknown keys {sorted(set(params) - known)}")
        ints = {"template", "search"}
        cfg = NoiseReductionConfig(**{k: (int(v) if k in ints else float(v)) for k, v in params.items()},
                                   moderate=dict(moderate))
        if moderate:
            from lumen.experimental.noise_aware import NoiseAwareConfig

            NoiseAwareConfig.from_dict(moderate)   # validates the moderate filter's parameters
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as e:
        raise NoiseReductionConfigError(f"Noise reduction config {p} is invalid: {e}") from e
    if cfg.moderate and not (0.0 < cfg.moderate_low < cfg.moderate_high <= cfg.gate_low):
        raise NoiseReductionConfigError(f"Noise reduction config {p}: need 0 < moderate_low < moderate_high <= gate_low")
    if not (0.0 < cfg.gate_low < cfg.gate_high < 1.0 and 0.0 < cfg.strength <= 5.0
            and cfg.template % 2 == 1 and cfg.search % 2 == 1 and 1 <= cfg.template <= cfg.search <= 51):
        raise NoiseReductionConfigError(f"Noise reduction config {p} has out-of-range values: {cfg}")
    return cfg
