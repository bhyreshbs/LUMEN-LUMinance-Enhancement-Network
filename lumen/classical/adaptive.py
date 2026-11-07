"""Image analysis and adaptive parameter selection (instead of fixed enhancement parameters)."""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.restoration import estimate_sigma

from lumen.utils import luminance


@dataclass
class ImageStats:
    mean_lum: float        # global brightness
    p5: float              # dark percentile
    p95: float             # bright percentile
    contrast: float        # std of luminance
    noise_sigma: float     # estimated Gaussian noise std (input domain)
    nonuniformity: float   # coefficient of variation of the blurred luminance
    colour_cast: float     # max relative deviation of channel means from gray

    @property
    def category(self) -> str:
        if self.mean_lum < 0.06:
            return "extremely_dark"
        if self.nonuniformity > 0.6:
            return "non_uniform"
        if self.mean_lum < 0.25:
            return "low_light"
        return "normal"

    def as_dict(self) -> dict:
        return {**{k: round(v, 4) for k, v in asdict(self).items()}, "category": self.category}


@dataclass
class EnhanceParams:
    gamma: float
    illum_radius: int
    illum_eps: float
    denoise_strength: float
    denoise_radius: int
    clahe_clip: float
    clahe_tile: int
    wb_strength: float
    saturation: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class AdaptiveConfig:
    """Tunable constants of select_params(). Defaults are the hand-set values; tuned values live in
    configs/classical.yaml (written by scripts/tune_classical.py)."""

    target_mean: float = 0.45          # brightness the illumination mean is mapped to
    gamma_min: float = 0.2             # strongest allowed brightening
    denoise_mult: float = 2.5          # bilateral sigma_color = mult * effective noise
    denoise_max: float = 0.25
    clahe_gain: float = 2.5            # clip = 1 + gain * (ref - contrast) / ref
    clahe_ref_contrast: float = 0.18
    clahe_min: float = 1.0
    clahe_max: float = 3.0
    clahe_noise_thresh: float = 0.05   # above this effective noise, CLAHE clip is capped
    clahe_noise_cap: float = 1.5
    clahe_tile: int = 8
    radius_base: float = 20.0          # illumination guided-filter radius = base - slope * nonuniformity
    radius_slope: float = 10.0
    wb_mult: float = 2.0               # white-balance strength = mult * colour cast
    sat_gain: float = 0.5              # saturation = 1 + gain * (1 - gamma)
    sat_max: float = 1.3

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveConfig":
        known = {f.name: f.type for f in fields(cls)}
        unknown = set(d) - set(known)
        if unknown:
            raise ValueError(f"Unknown classical config keys: {sorted(unknown)}")
        defaults = cls()
        return cls(**{k: type(getattr(defaults, k))(v) for k, v in d.items()})

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "classical.yaml"


def config_path() -> Path:
    return Path(os.environ.get("LUMEN_CLASSICAL_CONFIG", DEFAULT_CONFIG_PATH))


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime: float) -> AdaptiveConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AdaptiveConfig.from_dict(data.get("params", data))


def load_config(path: str | Path | None = None) -> AdaptiveConfig:
    """Load the tuned config (YAML with a `params:` mapping). Missing file -> built-in defaults."""
    p = Path(path) if path else config_path()
    if not p.is_file():
        if path:  # an explicitly requested file must exist
            raise FileNotFoundError(p)
        return AdaptiveConfig()
    return _load_cached(str(p), p.stat().st_mtime)


def save_config(config: AdaptiveConfig, path: str | Path, meta: dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"meta": meta or {}, "params": config.as_dict()}
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def analyze_image(img: np.ndarray) -> ImageStats:
    y = luminance(img)
    small = cv2.resize(y, (64, 64), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (0, 0), 6)
    means = img.reshape(-1, 3).mean(0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma = float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))
    sigma = 0.0 if not np.isfinite(sigma) else sigma  # flat images yield NaN
    return ImageStats(
        mean_lum=float(y.mean()),
        p5=float(np.percentile(y, 5)),
        p95=float(np.percentile(y, 95)),
        contrast=float(y.std()),
        noise_sigma=sigma,
        nonuniformity=float(blurred.std() / (blurred.mean() + 1e-6)),
        colour_cast=float(np.abs(means - means.mean()).max() / (means.mean() + 1e-6)),
    )


def select_params(stats: ImageStats, config: "AdaptiveConfig | None" = None) -> EnhanceParams:
    """Map image statistics to enhancement parameters. Constants come from `config`
    (default: configs/classical.yaml if present, else the built-in AdaptiveConfig defaults)."""
    c = config or load_config()
    m = float(np.clip(stats.mean_lum, 1e-3, 0.95))
    # output ~ L^gamma  =>  choose gamma so the mean illumination maps to target_mean
    if m < c.target_mean:
        gamma = float(np.clip(np.log(c.target_mean) / np.log(m), c.gamma_min, 1.0))
    else:
        gamma = 1.0

    # Noise is multiplied by roughly the brightening gain m^(gamma - 1)
    gain = m ** (gamma - 1.0)
    eff_noise = stats.noise_sigma * gain
    denoise = float(np.clip(c.denoise_mult * eff_noise, 0.0, c.denoise_max))

    # Low-contrast images get stronger CLAHE; noisy ones get less (CLAHE amplifies noise)
    clip = 1.0 + c.clahe_gain * (c.clahe_ref_contrast - stats.contrast) / c.clahe_ref_contrast
    clip = float(np.clip(clip, c.clahe_min, c.clahe_max))
    if eff_noise > c.clahe_noise_thresh:
        clip = min(clip, c.clahe_noise_cap)

    # Non-uniform lighting -> smaller illumination radius to follow local light changes
    radius = int(np.clip(round(c.radius_base - c.radius_slope * stats.nonuniformity), 5, 30))

    return EnhanceParams(
        gamma=gamma,
        illum_radius=radius,
        illum_eps=1e-3,
        denoise_strength=denoise,
        denoise_radius=3 if denoise < 0.1 else 5,
        clahe_clip=clip,
        clahe_tile=c.clahe_tile,
        wb_strength=float(np.clip(stats.colour_cast * c.wb_mult, 0.0, 1.0)),
        saturation=float(np.clip(1.0 + c.sat_gain * (1 - gamma), 1.0, c.sat_max)),
    )
