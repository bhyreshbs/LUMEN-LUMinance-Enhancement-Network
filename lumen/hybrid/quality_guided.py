"""Quality-guided hybrid: Retinexformer first, classical refinement only where it is justified.

Unlike the fusion hybrid (lumen/hybrid/fusion.py), which blends several full candidates, this mode
treats the deep model's output R as the answer and asks "where is R still wrong?":

  1. R = deep model(input)  - run exactly once
  2. Detectors on R (and the input) produce soft masks in [0, 1]:
       noise      residual high-frequency energy on flat areas of R
       colour     chromaticity drift of R w.r.t. the input's own chromaticity (only where the input
                  is bright enough for its chromaticity to be trustworthy)
       contrast   local contrast of R far below the input's (detail flattened by the model)
       highlight  near-clipped luminance / channels in R
  3. Each correction is a local OpenCV operation blended in by its feathered mask:
       noise -> guided-filter denoise, colour -> local chroma pull towards the input chromaticity,
       contrast -> guided base/detail boost, highlight -> soft-knee luminance roll-off.
  4. A refinement is applied only if its mask covers at least `min_coverage` of the image.
     If none is applied, R is returned unchanged (identity path).

All thresholds live in QualityGuidedConfig. Diagnostics record, per refinement, whether it ran and
how much of the image it touched.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

from lumen.classical.filters import guided_filter
from lumen.utils import luminance

QG_COMPONENTS = ("noise", "colour", "contrast", "highlight")
_EPS = 1e-4


@dataclass(frozen=True)
class QualityGuidedConfig:
    """Thresholds and strengths. Score ramps are (start, full): mask = smoothstep(score; start, full)."""

    enabled: tuple[str, ...] = QG_COMPONENTS  # refinements allowed to run (others are never computed)
    mask_sigma: float = 4.0                  # Gaussian feathering of every mask (px) - no visible seams
    min_coverage: float = 0.005              # mean mask value needed to activate a refinement

    # noise: high-frequency residual / local gradient (scale-free score)
    noise_ramp: tuple[float, float] = (0.06, 0.12)
    noise_radius: int = 3
    noise_eps: float = 4e-4                   # guided-filter eps (~ 0.02^2)

    # colour: L1 distance between local chromaticities (r, g, b sum to 1)
    colour_ramp: tuple[float, float] = (0.04, 0.08)
    colour_sigma: float = 8.0                 # blur for local chromaticity (suppresses input noise)
    colour_min_input: float = 0.03            # input luminance below this -> chromaticity unreliable
    colour_strength: float = 0.5              # fraction of the way towards the input chromaticity

    # contrast: 1 - (local CV of R / local CV of input)
    contrast_ramp: tuple[float, float] = (0.35, 0.6)
    contrast_sigma: float = 3.0
    contrast_min_cv: float = 0.08             # ignore flat input regions
    contrast_min_input: float = 0.03
    contrast_gain: float = 0.6                # detail boost factor
    contrast_radius: int = 4

    # highlight: luminance / max-channel ramp
    highlight_ramp: tuple[float, float] = (0.90, 0.98)
    highlight_knee: float = 0.80

    def __post_init__(self):
        """Reject values that would make the refinements meaningless or unsafe."""
        for name in ("noise_ramp", "colour_ramp", "contrast_ramp", "highlight_ramp"):
            ramp = getattr(self, name)
            if len(ramp) != 2 or not all(np.isfinite(ramp)) or not ramp[0] < ramp[1]:
                raise ValueError(f"{name} must be two finite numbers (start < full), got {ramp}")
        positive = ("noise_eps", "colour_sigma", "contrast_sigma", "noise_radius", "contrast_radius")
        for name in positive:
            v = getattr(self, name)
            if not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be > 0, got {v}")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be in [0, 1], got {self.min_coverage}")
        if self.mask_sigma < 0:
            raise ValueError(f"mask_sigma must be >= 0, got {self.mask_sigma}")
        if not 0.0 < self.highlight_knee < 1.0:
            raise ValueError(f"highlight_knee must be in (0, 1), got {self.highlight_knee}")
        bad = set(self.enabled) - set(QG_COMPONENTS)
        if bad:
            raise ValueError(f"Unknown refinements in 'enabled': {sorted(bad)}")

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "QualityGuidedConfig":
        if not isinstance(d, dict):
            raise ValueError(f"expected a mapping of parameters, got {type(d).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown quality-guided config keys: {sorted(unknown)}")
        defaults = cls()
        out = {}
        for k, v in d.items():
            ref = getattr(defaults, k)
            if k == "enabled":
                bad = set(v) - set(QG_COMPONENTS)
                if bad:
                    raise ValueError(f"Unknown refinements in 'enabled': {sorted(bad)}")
                out[k] = tuple(v)
            elif isinstance(ref, tuple):
                out[k] = tuple(float(x) for x in v)
            else:
                out[k] = type(ref)(v)
        return cls(**out)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QualityGuidedConfig":
        """Parse a YAML config. Any problem raises QGConfigError naming the file."""
        import yaml

        path = Path(path)
        if not path.is_file():
            raise QGConfigError(f"Quality-guided config not found: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
            raise QGConfigError(f"Quality-guided config {path} is not readable YAML: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("params", data), dict):
            raise QGConfigError(f"Quality-guided config {path} must contain a 'params:' mapping")
        try:
            return cls.from_dict(data.get("params", data))
        except (ValueError, TypeError) as e:
            raise QGConfigError(f"Quality-guided config {path} is invalid: {e}") from e


class QGConfigError(ValueError):
    """The calibrated quality-guided configuration is missing or invalid."""


DEFAULT_QG_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "quality_guided.yaml"


def qg_config_path() -> Path:
    return Path(os.environ.get("LUMEN_QG_CONFIG", DEFAULT_QG_CONFIG_PATH))


def load_qg_config(path: str | Path | None = None, allow_defaults: bool = False) -> QualityGuidedConfig:
    """Load the *calibrated* config: `path`, else $LUMEN_QG_CONFIG, else configs/quality_guided.yaml.

    Never falls back silently: a missing or invalid file raises QGConfigError. The built-in dataclass
    defaults are uncalibrated (they refine far too much), so they are returned only when the caller
    explicitly passes allow_defaults=True and no file exists (research / unit-test use).
    """
    p = Path(path) if path else qg_config_path()
    if not p.is_file() and allow_defaults:
        return QualityGuidedConfig()
    if not p.is_file():
        raise QGConfigError(
            f"Calibrated quality-guided config not found at {p}. Quality Guided mode needs it - restore "
            "configs/quality_guided.yaml or run scripts/calibrate_quality_guided.py."
        )
    return QualityGuidedConfig.from_yaml(p)


# ----------------------------------------------------------------------------- helpers
def smoothstep(x: np.ndarray, start: float, full: float) -> np.ndarray:
    t = np.clip((x - start) / max(full - start, 1e-6), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.astype(np.float32)
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma), 0.0, 1.0)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)


def _chromaticity(img: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Local chromaticity (r,g,b)/(r+g+b) of a blurred image, and the blurred channel sum."""
    b = _blur(img, sigma)
    s = b.sum(axis=2, keepdims=True)
    return b / (s + _EPS), s[..., 0]


# ----------------------------------------------------------------------------- detectors
def noise_score(r: np.ndarray) -> np.ndarray:
    """Residual high-frequency energy relative to local structure: high = noisy flat area."""
    y = luminance(r)
    y8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    resid = np.abs(y - cv2.medianBlur(y8, 3).astype(np.float32) / 255.0)
    grad = np.abs(cv2.Sobel(y, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(y, cv2.CV_32F, 0, 1))
    return _blur(resid, 3.0) / (_blur(grad, 3.0) + 0.05)


def colour_score(r: np.ndarray, inp: np.ndarray, cfg: QualityGuidedConfig) -> tuple[np.ndarray, np.ndarray]:
    """L1 chromaticity drift of R from the input, and the input-reliability weight."""
    c_r, _ = _chromaticity(r, cfg.colour_sigma)
    c_i, _ = _chromaticity(inp, cfg.colour_sigma)
    drift = np.abs(c_r - c_i).sum(axis=2)
    y_in = _blur(luminance(inp), cfg.colour_sigma)
    reliable = smoothstep(y_in, cfg.colour_min_input, 2 * cfg.colour_min_input)
    return drift, reliable


def _local_cv(y: np.ndarray, sigma: float) -> np.ndarray:
    mu = _blur(y, sigma)
    var = np.maximum(_blur(y * y, sigma) - mu * mu, 0.0)
    return np.sqrt(var) / (mu + 0.02)


def contrast_score(r: np.ndarray, inp: np.ndarray, cfg: QualityGuidedConfig) -> tuple[np.ndarray, np.ndarray]:
    """1 - CV(R)/CV(input): how much local contrast R lost. Valid only where the input has structure."""
    y_in = _blur(luminance(inp), 1.0)  # light blur so input noise is not counted as contrast
    cv_in = _local_cv(y_in, cfg.contrast_sigma)
    cv_r = _local_cv(luminance(r), cfg.contrast_sigma)
    loss = 1.0 - cv_r / (cv_in + 1e-3)
    valid = (cv_in > cfg.contrast_min_cv) & (_blur(y_in, cfg.contrast_sigma) > cfg.contrast_min_input)
    return loss.astype(np.float32), valid.astype(np.float32)


def highlight_score(r: np.ndarray) -> np.ndarray:
    return np.maximum(luminance(r), r.max(axis=2) - 0.02)


# ----------------------------------------------------------------------------- corrections
def denoise_correction(r, cfg):
    return np.clip(guided_filter(r, r, cfg.noise_radius, cfg.noise_eps), 0.0, 1.0)


def colour_correction(r, inp, cfg):
    """Move R's local chromaticity part-way towards the input's, keeping R's luminance."""
    c_r, _ = _chromaticity(r, cfg.colour_sigma)
    c_i, _ = _chromaticity(inp, cfg.colour_sigma)
    target = c_r + cfg.colour_strength * (c_i - c_r)
    gain = target / (c_r + _EPS)
    out = r * gain
    y0, y1 = luminance(r), luminance(out)
    out = out * (y0 / (y1 + _EPS))[..., None]  # restore luminance
    return np.clip(out, 0.0, 1.0)


def contrast_correction(r, cfg):
    y = luminance(r)
    base = guided_filter(y, y, cfg.contrast_radius, 1e-2)
    y_new = np.clip(y + cfg.contrast_gain * (y - base), 0.0, 1.0)
    return np.clip(r * (y_new / (y + _EPS))[..., None], 0.0, 1.0)


def highlight_correction(r, cfg):
    y = luminance(r)
    k = cfg.highlight_knee
    span = 1.0 - k
    y_new = np.where(y > k, k + span * np.tanh((y - k) / span), y)
    out = r * (y_new / (y + _EPS))[..., None]
    m = out.max(axis=2, keepdims=True)  # also tame single clipped channels
    out = np.where(m > 1.0, out / m, out)
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------- enhancer
@dataclass
class QualityGuidedOutput:
    image: np.ndarray
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def _blend(base: np.ndarray, corrected: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (base + mask[..., None] * (corrected - base)).astype(np.float32)


class QualityGuidedHybrid:
    def __init__(self, deep, config: QualityGuidedConfig | None = None, disable: set[str] | None = None):
        self.deep = deep
        self.cfg = config or load_qg_config()
        self.disable = set(disable or ())

    def refine(self, r: np.ndarray, inp: np.ndarray) -> QualityGuidedOutput:
        """Refine an already restored image `r` given the low-light input `inp` (no model call)."""
        t_start = time.perf_counter()
        cfg = self.cfg
        active = [c for c in QG_COMPONENTS if c in cfg.enabled and c not in self.disable]
        steps: dict[str, np.ndarray] = {"restored": r}
        diag: dict[str, dict] = {}
        out = r

        def record(name, mask, applied):
            diag[name] = {
                "active": bool(applied),
                "coverage": round(float(mask.mean()), 5),         # mean soft-mask value
                "area_frac": round(float((mask > 0.5).mean()), 5),  # pixels refined by > 50 %
            }
            steps[f"mask_{name}"] = np.repeat(mask[..., None], 3, -1)

        masks = {}
        if "noise" in active:
            masks["noise"] = feather(smoothstep(noise_score(r), *cfg.noise_ramp), cfg.mask_sigma)
        if "colour" in active:
            drift, reliable = colour_score(r, inp, cfg)
            masks["colour"] = feather(smoothstep(drift, *cfg.colour_ramp) * reliable, cfg.mask_sigma)
        if "contrast" in active:
            loss, valid = contrast_score(r, inp, cfg)
            m = smoothstep(loss, *cfg.contrast_ramp) * valid
            if "noise" in masks:
                m = m * (1.0 - masks["noise"])  # never sharpen where we just decided to denoise
            masks["contrast"] = feather(m, cfg.mask_sigma)

        corrections = {
            "noise": lambda x: denoise_correction(x, cfg),
            "colour": lambda x: colour_correction(x, inp, cfg),
            "contrast": lambda x: contrast_correction(x, cfg),
        }
        for name in ("noise", "colour", "contrast"):
            if name not in masks:
                continue
            m = masks[name]
            applied = m.mean() >= cfg.min_coverage
            if applied:
                out = _blend(out, corrections[name](out), m)
                steps[f"after_{name}"] = out
            record(name, m, applied)

        if "highlight" in active:  # last: catches overshoot from the contrast boost too
            m = feather(smoothstep(highlight_score(out), *cfg.highlight_ramp), cfg.mask_sigma)
            applied = m.mean() >= cfg.min_coverage
            if applied:
                out = _blend(out, highlight_correction(out, cfg), m)
                steps["after_highlight"] = out
            record("highlight", m, applied)

        identity = not any(d["active"] for d in diag.values())
        if identity:
            out = r  # bit-exact identity path
        out = np.clip(out, 0.0, 1.0).astype(np.float32) if not identity else r
        info = {"refinements": diag, "identity": identity,
                "applied": [k for k, d in diag.items() if d["active"]],
                "configured": list(cfg.enabled),
                "timing_ms": {"refine": round((time.perf_counter() - t_start) * 1000, 1)}}
        return QualityGuidedOutput(out, steps, info)

    def __call__(self, img: np.ndarray) -> QualityGuidedOutput:
        t0 = time.perf_counter()
        r, aux = self.deep(img)  # the only model call
        t1 = time.perf_counter()
        res = self.refine(r, img)
        res.info["timing_ms"] = {"model": round((t1 - t0) * 1000, 1), "refine": res.info["timing_ms"]["refine"]}
        res.intermediates[f"{self.deep.name}_map"] = np.repeat(aux[..., None], 3, -1)
        return res
