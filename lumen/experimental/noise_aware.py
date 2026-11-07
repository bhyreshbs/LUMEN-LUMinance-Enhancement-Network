"""Experimental noise-aware refinement for the Retinexformer output (research variant, not a default).

Why it exists (see docs/NOISE_STUDY.md): on noisy low-light inputs Retinexformer amplifies sensor noise
into fine luminance grain plus low-frequency chroma blotches, and the calibrated QGH noise refinement
removes little of it. Its detector threshold is one absolute constant, coverage-capped at 25 %, and
its guided filter uses a fixed eps = 0.04^2, which is below the variance of that noise, so the filter
treats the noise as structure.

This variant replaces the QGH *noise* refinement only (the highlight refinement is reused unchanged):

  1. Noise-level estimation: a local noise sigma map from the robust (low-quantile) magnitude of the
     Immerkaer Laplacian in 16 px blocks, separately for luma (Y) and chroma (Cr, Cb). A low quantile
     in each block means real texture and edges inflate the estimate less than they would a mean.
  2. Luma: an edge-aware filter whose strength follows the local sigma. "bilateral" uses a layered
     bilateral filter (several range sigmas interpolated per pixel); "guided" uses a self-guided filter
     with a per-pixel eps = (k * sigma)^2. An optional second, half-resolution pass removes the
     low-frequency blotches that Retinexformer's correlated noise leaves.
  3. Chroma: a guided filter over a large window. "colour_guided" uses a 3-channel guide (denoised Y
     plus lightly smoothed Cr, Cb), so colour edges survive even where their luma contrast is below
     the noise. "joint_guided" (luma-only guide) bleeds such edges; the unit test documents this.
  4. Detail protection: where the smoothed gradient is far above the local noise sigma (fine branches,
     text, object edges), luma filtering is reduced by `edge_protect`.
  5. Adaptivity: everything is blended in by smoothstep(sigma; floor, 2 * floor). Regions whose
     estimated noise is below the floor are left untouched. A small fraction of the removed luma
     residual (`detail_keep`) is added back so surfaces do not look plastic.

All parameters live in NoiseAwareConfig; they are selected on the dev sets by scripts/noise_study.py.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import norm

from lumen.hybrid.quality_guided import QualityGuidedConfig, QualityGuidedHybrid, QualityGuidedOutput, smoothstep

_LAP = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)  # Immerkaer (1996) noise operator
_EPS = 1e-6
LUMA_METHODS = ("bilateral", "guided", "nlm", "none")
CHROMA_METHODS = ("joint_guided", "colour_guided", "none")


@dataclass(frozen=True)
class NoiseAwareConfig:
    # noise-level estimation
    block: int = 16                 # block size (px) of the local sigma estimate
    quantile: float = 0.3           # low quantile of |Laplacian| per block (robust to texture/edges)
    # two-scale noise/texture separation (estimate_noise_map); texture_ratio = 0 disables it
    noise_ratio: float = 0.2        # sigma(half res) / sigma(full res) of the grain (dev: 0.17-0.37)
    texture_ratio: float = 0.65     # same ratio for natural texture (dev noise-free images: 0.46-0.81)
    # luma
    luma_method: str = "bilateral"
    luma_k: float = 3.0             # range sigma / eps scale relative to the estimated sigma
    luma_radius: int = 4
    coarse_k: float = 0.0           # > 0 enables the half-resolution pass (blotch removal)
    detail_keep: float = 0.15       # fraction of the removed luma residual added back (anti-plastic)
    # chroma
    chroma_method: str = "joint_guided"
    chroma_k: float = 2.0           # eps = (chroma_k * sigma_Y)^2 for the luma-guided chroma filter
    chroma_radius: int = 8
    # adaptivity (sigma floor, in [0, 1] intensity units): below it nothing changes
    luma_floor: float = 0.004
    chroma_floor: float = 0.002
    adaptive: bool = True           # False -> one global sigma per image (ablation)
    # detail protection: luma filtering is reduced by edge_protect * smoothstep(|grad Y| / sigma; ramp)
    edge_protect: float = 0.0
    detail_ramp: tuple = (8.0, 16.0)
    # ---- iteration 2 (v2) options. Every default reproduces v1 exactly. ----
    # image-level gate: return the input unchanged (bit-exact) when the image's noise-only sigma is below
    # image_gate, or when its scale ratio sigma(half)/sigma(full) says texture dominates (> texture_gate)
    image_gate: float = 0.0
    texture_gate: float = 0.0
    # chroma blend mask: "max" (v1: max of chroma-sigma ramp and luma mask) | "luma_noise" (luma noise-only
    # mask: chroma is filtered only where luma noise was detected, never on clean regions)
    chroma_mask: str = "max"
    # method-noise test: removed luma residual that is *structured* (strokes, grain lines: larger than the
    # noise and coherent) is put back: yd += residual_protect * structure * (y - yd)
    residual_protect: float = 0.0
    residual_ramp: tuple = (2.0, 4.0)
    # pixels outside every refinement mask keep the input value bit for bit (no YCbCr round-trip drift)
    exact_identity: bool = False
    # texture-adaptive strength: luma range sigma k * n is reduced to k * n / (1 + texture_relief * (T / n)^2),
    # T^2 = max(sigma_total^2 - n^2, 0) - dense texture under heavy noise (foliage) is not smeared
    texture_relief: float = 0.0

    def __post_init__(self):
        for name in ("detail_ramp", "residual_ramp"):
            r = tuple(float(v) for v in getattr(self, name))
            if len(r) != 2 or not r[0] < r[1]:
                raise ValueError(f"{name} must be (start, full) with start < full")
            object.__setattr__(self, name, r)
        if self.chroma_mask not in ("max", "luma_noise"):
            raise ValueError("chroma_mask must be 'max' or 'luma_noise'")
        if self.texture_relief < 0:
            raise ValueError("texture_relief must be >= 0")
        if not 0.0 <= self.residual_protect <= 1.0:
            raise ValueError("residual_protect must be in [0, 1]")
        if self.image_gate < 0 or not 0.0 <= self.texture_gate < 1.0:
            raise ValueError("image_gate must be >= 0 and texture_gate in [0, 1)")
        if not 0.0 <= self.edge_protect <= 1.0:
            raise ValueError("edge_protect must be in [0, 1]")
        if self.texture_ratio and not 0.0 < self.noise_ratio < self.texture_ratio < 1.0:
            raise ValueError("need 0 < noise_ratio < texture_ratio < 1 (or texture_ratio = 0)")
        if self.luma_method not in LUMA_METHODS:
            raise ValueError(f"luma_method must be one of {LUMA_METHODS}")
        if self.chroma_method not in CHROMA_METHODS:
            raise ValueError(f"chroma_method must be one of {CHROMA_METHODS}")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        for name in ("block", "luma_radius", "chroma_radius"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("luma_k", "chroma_k", "coarse_k", "luma_floor", "chroma_floor"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= self.detail_keep <= 1.0:
            raise ValueError("detail_keep must be in [0, 1]")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseAwareConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown noise-aware config keys: {sorted(unknown)}")
        defaults = cls()
        return cls(**{k: tuple(float(x) for x in v) if isinstance(getattr(defaults, k), tuple)
                      else type(getattr(defaults, k))(v) for k, v in d.items()})

    def as_yaml_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


DEFAULT_NA_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "experimental" / "noise_aware.yaml"


def load_na_config(path: str | Path | None = None) -> NoiseAwareConfig:
    """The dev-selected config (configs/experimental/noise_aware.yaml), else the dataclass defaults."""
    import yaml

    p = Path(path) if path else DEFAULT_NA_CONFIG_PATH
    if not p.is_file():
        if path:
            raise FileNotFoundError(p)
        return NoiseAwareConfig()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return NoiseAwareConfig.from_dict(data.get("params", data))


# ----------------------------------------------------------------------------- colour space
def rgb_to_ycc(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)  # BT.601, chroma offset 0.5


def ycc_to_rgb(ycc: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(ycc.astype(np.float32), cv2.COLOR_YCrCb2RGB), 0.0, 1.0)


# ----------------------------------------------------------------------------- noise estimation
def estimate_sigma_map(ch: np.ndarray, block: int = 16, quantile: float = 0.3) -> tuple[np.ndarray, float]:
    """Local noise sigma (HxW) and the image median, from block low-quantiles of |Laplacian|.

    For white Gaussian noise of std s, the Immerkaer operator output is N(0, 36 s^2), so the q-quantile
    of its magnitude is 6 s z_q with z_q = Phi^-1((1 + q) / 2). Blocks with texture or edges give
    larger values. The low quantile, a 3x3 median over blocks and smoothing all limit that bias.
    """
    ch = ch.astype(np.float32)
    h, w = ch.shape
    lap = np.abs(cv2.filter2D(ch, cv2.CV_32F, _LAP, borderType=cv2.BORDER_REFLECT))
    b = int(max(4, min(block, h // 2, w // 2)))
    bh, bw = max(1, h // b), max(1, w // b)
    tiles = lap[: bh * b, : bw * b].reshape(bh, b, bw, b).transpose(0, 2, 1, 3).reshape(bh, bw, -1)
    s = (np.quantile(tiles, quantile, axis=-1) / (6.0 * norm.ppf((1 + quantile) / 2))).astype(np.float32)
    if min(bh, bw) >= 3:
        s = cv2.medianBlur(s, 3)
    global_sigma = float(np.median(s))
    smap = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    smap = cv2.GaussianBlur(smap, (0, 0), b / 2)
    return np.maximum(smap, 0.0).astype(np.float32), global_sigma


def estimate_noise_map(y: np.ndarray, cfg: "NoiseAwareConfig") -> tuple[np.ndarray, float, dict]:
    """Noise-only sigma map: separate grain from texture using how the estimate scales.

    Grain is concentrated at the finest scale: after 2x downsampling its estimated sigma falls to
    a * sigma (a = noise_ratio ~ 0.2). Natural texture is close to scale-invariant: b * sigma
    (b = texture_ratio ~ 0.65). With s_f^2 = n^2 + T^2 and s_h^2 = a^2 n^2 + b^2 T^2,
        n^2 = (b^2 s_f^2 - s_h^2) / (b^2 - a^2).
    A noise-free but finely textured image therefore gets n ~ 0 and is left alone."""
    s_f, g_f = estimate_sigma_map(y, cfg.block, cfg.quantile)
    if not cfg.texture_ratio:
        return s_f, g_f, {"ratio": None, "total_map": s_f}
    down = cv2.pyrDown(y)
    s_h, g_h = estimate_sigma_map(down, max(4, cfg.block // 2), cfg.quantile)
    s_h = cv2.resize(s_h, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
    a2, b2 = cfg.noise_ratio ** 2, cfg.texture_ratio ** 2
    n = np.sqrt(np.clip((b2 * s_f ** 2 - s_h ** 2) / (b2 - a2), 0.0, None))
    n = np.minimum(n, s_f).astype(np.float32)
    g_n = float(np.sqrt(max((b2 * g_f ** 2 - g_h ** 2) / (b2 - a2), 0.0)))
    return n, min(g_n, g_f), {"ratio": round(g_h / max(g_f, 1e-6), 3), "sigma_total": round(g_f, 5),
                              "total_map": s_f}


# ----------------------------------------------------------------------------- filters
def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_eps_map(guide: np.ndarray, src: np.ndarray, r: int, eps: np.ndarray | float) -> np.ndarray:
    """Guided filter (He et al.) with a per-pixel regulariser eps (HxW or scalar)."""
    mi, mp = _box(guide, r), _box(src, r)
    cov = _box(guide * src, r) - mi * mp
    var = _box(guide * guide, r) - mi * mi
    a = cov / (var + eps + _EPS)
    bb = mp - a * mi
    return (_box(a, r) * guide + _box(bb, r)).astype(np.float32)


def colour_guided(guide: np.ndarray, src: np.ndarray, r: int, eps: np.ndarray) -> np.ndarray:
    """Colour guided filter (He et al., 3-channel guide): guide HxWx3, src HxW, eps HxWx3 (per-pixel
    diagonal regulariser per guide channel). Edges of any guide channel are preserved in the output."""
    mi = np.stack([_box(guide[..., k], r) for k in range(3)], -1)
    mp = _box(src, r)
    cov = np.stack([_box(guide[..., k] * src, r) for k in range(3)], -1) - mi * mp[..., None]
    s = {}
    for i in range(3):
        for j in range(i, 3):
            s[i, j] = _box(guide[..., i] * guide[..., j], r) - mi[..., i] * mi[..., j]
    for i in range(3):
        s[i, i] = s[i, i] + eps[..., i] + _EPS
    # closed-form inverse of the symmetric 3x3 covariance (much faster than batched np.linalg.solve)
    c00 = s[1, 1] * s[2, 2] - s[1, 2] ** 2
    c01 = s[0, 2] * s[1, 2] - s[0, 1] * s[2, 2]
    c02 = s[0, 1] * s[1, 2] - s[0, 2] * s[1, 1]
    c11 = s[0, 0] * s[2, 2] - s[0, 2] ** 2
    c12 = s[0, 1] * s[0, 2] - s[0, 0] * s[1, 2]
    c22 = s[0, 0] * s[1, 1] - s[0, 1] ** 2
    det = s[0, 0] * c00 + s[0, 1] * c01 + s[0, 2] * c02
    det = np.where(np.abs(det) < 1e-20, 1e-20, det)
    p0, p1, p2 = cov[..., 0], cov[..., 1], cov[..., 2]
    a = np.stack([c00 * p0 + c01 * p1 + c02 * p2, c01 * p0 + c11 * p1 + c12 * p2,
                  c02 * p0 + c12 * p1 + c22 * p2], -1) / det[..., None]
    b = mp - (a * mi).sum(-1)
    a_m = np.stack([_box(np.ascontiguousarray(a[..., k]), r) for k in range(3)], -1)
    return ((a_m * guide).sum(-1) + _box(b, r)).astype(np.float32)


def layered_bilateral(y: np.ndarray, sig: np.ndarray, radius: int) -> np.ndarray:
    """Bilateral filter whose range sigma varies per pixel: filter at a few geometric sigma levels and
    interpolate linearly in log-sigma."""
    lo, hi = float(np.percentile(sig, 2)), float(np.percentile(sig, 98))
    lo = max(lo, 1e-3)
    hi = max(hi, lo * 1.05)
    n = int(np.clip(np.ceil(np.log(hi / lo) / np.log(2.5)) + 1, 1, 3))  # one level per 2.5x, <= 3
    levels = np.geomspace(lo, hi, n) if n > 1 else np.array([np.sqrt(lo * hi)])
    d, ss = 2 * radius + 1, max(radius * 0.75, 1.0)
    outs = [cv2.bilateralFilter(y, d, float(s), ss) for s in levels]
    if n == 1:
        return outs[0]
    t = np.clip((np.log(np.maximum(sig, 1e-6)) - np.log(levels[0])) / (np.log(levels[-1]) - np.log(levels[0])),
                0, 1) * (n - 1)
    acc = np.zeros_like(y, dtype=np.float32)
    for i, o in enumerate(outs):  # hat-function weights: linear interpolation between adjacent levels
        acc += np.clip(1.0 - np.abs(t - i), 0.0, 1.0).astype(np.float32) * o
    return acc


def _luma_filter(y: np.ndarray, sig: np.ndarray, cfg: NoiseAwareConfig, k: float) -> np.ndarray:
    s = k * sig
    if cfg.luma_method == "bilateral":
        return layered_bilateral(y, s, cfg.luma_radius)
    if cfg.luma_method == "guided":
        return guided_eps_map(y, y, cfg.luma_radius, s * s)
    if cfg.luma_method == "nlm":  # single h per image (OpenCV NLM has no per-pixel strength)
        h = float(np.median(s)) * 255.0
        u8 = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
        return cv2.fastNlMeansDenoising(u8, None, max(h, 0.1), 7, 21).astype(np.float32) / 255.0
    return y


def detail_mask(y: np.ndarray, sig: np.ndarray, ramp: tuple[float, float]) -> np.ndarray:
    """1 where local structure is far above the noise floor: smoothed gradient energy / sigma.

    Measured on the coffee example vs a textured noise-free scene: noisy flat surfaces give a median
    ratio of ~3.7, fine branches ~10, so texture well above the noise is told apart from the grain."""
    ys = cv2.GaussianBlur(y, (0, 0), 1.0)
    gx, gy = cv2.Sobel(ys, cv2.CV_32F, 1, 0), cv2.Sobel(ys, cv2.CV_32F, 0, 1)
    mag = np.sqrt(cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 2.0))
    return smoothstep(mag / (sig + 1e-3), *ramp)


def residual_structure(e: np.ndarray, sig: np.ndarray, ramp: tuple[float, float]) -> np.ndarray:
    """Method-noise test (v2): 1 where the residual a filter removed looks like structure, not noise.

    Removed grain is about as large as the noise sigma and isotropic. A removed pen stroke or grain line
    is larger than the noise and oriented. Both are required: smoothstep(local RMS / sigma; ramp) times
    smoothstep(structure-tensor coherence; 0.35, 0.6). The coherence ramp is anchored on the median
    coherence of pure Retinexformer noise (0.37, coffee) and of real edges (0.85)."""
    rms = np.sqrt(cv2.GaussianBlur(e * e, (0, 0), 1.5))
    es = cv2.GaussianBlur(e, (0, 0), 0.7)
    gx, gy = cv2.Sobel(es, cv2.CV_32F, 1, 0), cv2.Sobel(es, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), 2.0) for a in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-12)
    return smoothstep(rms / (sig + 1e-3), *ramp) * smoothstep(coh, 0.35, 0.6)


# ----------------------------------------------------------------------------- main
def noise_aware_denoise(r: np.ndarray, cfg: NoiseAwareConfig | None = None) -> tuple[np.ndarray, dict, dict]:
    """Denoise an enhanced RGB image. Returns (image, maps for display, diagnostics)."""
    cfg = cfg or NoiseAwareConfig()
    t0 = time.perf_counter()
    ycc = rgb_to_ycc(r)
    y = np.ascontiguousarray(ycc[..., 0])
    sig_y, g_y, scale_info = estimate_noise_map(y, cfg)
    gate = None
    if cfg.image_gate > 0 and g_y < cfg.image_gate:
        gate = f"noise sigma {g_y:.4f} < image_gate {cfg.image_gate}"
    elif cfg.texture_gate > 0 and scale_info["ratio"] is not None and scale_info["ratio"] > cfg.texture_gate:
        gate = f"scale ratio {scale_info['ratio']:.3f} > texture_gate {cfg.texture_gate} (texture-dominated)"
    if gate:  # no correction justified: the input comes back bit for bit
        zero = np.zeros(y.shape, np.float32)
        info = {"gated": True, "gate": gate, "sigma_luma": round(g_y, 5), "scale_ratio": scale_info["ratio"],
                "coverage_luma": 0.0, "coverage_chroma": 0.0, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        return r, {"sigma_luma": sig_y, "mask_luma": zero, "mask_chroma": zero}, info
    sig_c = [estimate_sigma_map(np.ascontiguousarray(ycc[..., c]), cfg.block, cfg.quantile) for c in (1, 2)]
    if not cfg.adaptive:
        sig_y = np.full_like(sig_y, g_y)
        sig_c = [(np.full_like(sig_y, g), g) for _, g in sig_c]

    m_y = smoothstep(sig_y, cfg.luma_floor, 2 * cfg.luma_floor) if cfg.luma_floor > 0 else np.ones_like(sig_y)
    m_noise = m_y  # pure noise-level mask (chroma uses it; detail protection applies to luma only)
    if cfg.edge_protect > 0:
        m_y = m_y * (1.0 - cfg.edge_protect * detail_mask(y, sig_y, cfg.detail_ramp))
    out = ycc.copy()

    # luma
    yd = y
    if cfg.luma_method != "none" and cfg.luma_k > 0:
        k_map = cfg.luma_k
        if cfg.texture_relief > 0:
            t2 = np.maximum(scale_info["total_map"] ** 2 - sig_y ** 2, 0.0)
            k_map = cfg.luma_k / (1.0 + cfg.texture_relief * t2 / (sig_y ** 2 + 1e-8))
        yd = _luma_filter(y, sig_y, cfg, k_map)
        if cfg.coarse_k > 0 and min(y.shape) >= 32:
            down = cv2.pyrDown(yd)
            sd, _ = estimate_sigma_map(down, max(4, cfg.block // 2), cfg.quantile)
            coarse = cv2.pyrUp(down, dstsize=(y.shape[1], y.shape[0]))
            cd = cv2.pyrUp(_luma_filter(down, sd, cfg, cfg.coarse_k), dstsize=(y.shape[1], y.shape[0]))
            yd = yd - coarse + cd
        if cfg.residual_protect > 0:  # method-noise test: give back removed residual that looks like structure
            e = y - yd
            yd = yd + cfg.residual_protect * residual_structure(e, sig_y, cfg.residual_ramp) * e
        yd = yd + cfg.detail_keep * (y - yd)
        out[..., 0] = y + m_y * (yd - y)

    # chroma, steered by the denoised luma
    m_c = np.zeros_like(sig_y)
    if cfg.chroma_method != "none" and cfg.chroma_k > 0:
        guide = out[..., 0]
        eps = (cfg.chroma_k * np.maximum(sig_y, 1e-3)) ** 2
        if cfg.chroma_method == "colour_guided":
            # guide = (denoised Y, lightly smoothed Cr, Cb): chroma edges steer the filter as well as luma
            cs = [cv2.GaussianBlur(np.ascontiguousarray(ycc[..., c]), (0, 0), 1.5) for c in (1, 2)]
            guide3 = np.stack([guide, *cs], -1).astype(np.float32)
            eps3 = np.stack([eps] + [(cfg.chroma_k * np.maximum(s[0] / 2.5, 5e-4)) ** 2 for s in sig_c], -1)
        for j, c in enumerate((1, 2)):
            sc = sig_c[j][0]
            if cfg.chroma_mask == "luma_noise":
                mc = m_noise  # only where (texture-separated) luma noise was detected; clean regions untouched
            else:
                mc = smoothstep(sc, cfg.chroma_floor, 2 * cfg.chroma_floor) if cfg.chroma_floor > 0 \
                    else np.ones_like(sc)
                mc = np.maximum(mc, m_noise)  # chroma noise rides on luma noise: if luma is noisy, so is chroma
            if cfg.chroma_method == "colour_guided":
                cf = colour_guided(guide3, np.ascontiguousarray(ycc[..., c]), cfg.chroma_radius, eps3)
            else:
                cf = guided_eps_map(guide, np.ascontiguousarray(ycc[..., c]), cfg.chroma_radius, eps)
            out[..., c] = ycc[..., c] + mc * (cf - ycc[..., c])
            m_c = np.maximum(m_c, mc)

    img = ycc_to_rgb(out)
    if cfg.exact_identity:  # outside every mask the input is returned bit for bit (no colour round-trip drift)
        touched = (m_y > 0) | (m_c > 0)
        img = np.where(touched[..., None], img, r)
    info = {
        "gated": False,
        "sigma_luma": round(g_y, 5),
        "scale_ratio": scale_info["ratio"],
        "sigma_chroma": [round(g, 5) for _, g in sig_c],
        "coverage_luma": round(float(m_y.mean()), 4),
        "coverage_chroma": round(float(m_c.mean()), 4),
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    maps = {"sigma_luma": sig_y, "mask_luma": m_y, "mask_chroma": m_c}
    return img.astype(np.float32), maps, info


class NoiseAwareQGH:
    """QGH with the noise refinement replaced by noise_aware_denoise. The calibrated QGH highlight
    refinement runs afterwards, unchanged. The production QualityGuidedHybrid is not modified."""

    def __init__(self, deep, qg_config: QualityGuidedConfig, na_config: NoiseAwareConfig | None = None,
                 highlight: bool = True):
        self.deep = deep
        self.na = na_config or NoiseAwareConfig()
        keep = tuple(c for c in qg_config.enabled if c != "noise" and (highlight or c != "highlight"))
        self.qgh = QualityGuidedHybrid(deep, replace(qg_config, enabled=keep))

    def refine(self, r: np.ndarray, inp: np.ndarray) -> QualityGuidedOutput:
        t0 = time.perf_counter()
        d, maps, info = noise_aware_denoise(r, self.na)
        res = self.qgh.refine(d, inp)
        res.intermediates = {"restored": r, "after_noise_aware": d, **{f"na_{k}": v for k, v in maps.items()},
                             **{k: v for k, v in res.intermediates.items() if k != "restored"}}
        res.info["noise_aware"] = info
        res.info["timing_ms"] = {"refine": round((time.perf_counter() - t0) * 1000, 1)}
        return res

    def __call__(self, img: np.ndarray) -> QualityGuidedOutput:
        t0 = time.perf_counter()
        r, _ = self.deep(img)
        t1 = time.perf_counter()
        res = self.refine(r, img)
        res.info["timing_ms"]["model"] = round((t1 - t0) * 1000, 1)
        return res
