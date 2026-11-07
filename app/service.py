"""Application logic behind the LUMEN web UI (no web-framework imports, so it can be unit-tested directly).

User-facing modes
  standard        Retinexformer (recommended default)
  quality_guided  Retinexformer + conditional highlight / noise refinements (calibrated config required)
  classical       OpenCV pipeline (baseline)
Research methods (collapsed section in the UI) keep the fusion hybrid and Zero-DCE++ available for comparison.

Guarantees
  * A mode whose checkpoint or calibrated config is missing/invalid is reported unavailable with the reason;
    nothing ever runs on random weights or on uncalibrated quality-guided defaults.
  * Models and configs are loaded once (cached in the pipeline).
  * compare_all() runs Retinexformer exactly once and derives the quality-guided result from that output.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lumen import LumenPipeline
from lumen.classical.adaptive import config_path as classical_config_path
from lumen.deep import CheckpointError
from lumen.deep.checkpoints import verify as verify_checkpoint
from lumen.hybrid.fusion import HYBRID_COMPONENTS
from lumen.hybrid.quality_guided import QG_COMPONENTS, QGConfigError, QualityGuidedHybrid
from lumen.imageio import ImageValidationError, decode_image_file
from lumen.metrics import edge_preservation, niqe, noise_level, overexposure_ratio
from lumen.utils import to_uint8

log = logging.getLogger("lumen.app")

STANDARD, QUALITY_GUIDED, CLASSICAL = "standard", "quality_guided", "classical"
MODE_LABELS = {
    STANDARD: "Standard — Retinexformer",
    QUALITY_GUIDED: "Quality Guided — Retinexformer + conditional CV refinements",
    CLASSICAL: "Classical — OpenCV pipeline",
}
# research key -> (pipeline mode, deep model, label)
RESEARCH = {
    "hybrid_retinexformer": ("hybrid", "retinexformer", "Fusion hybrid — Retinexformer"),
    "hybrid_zerodce": ("hybrid", "zerodce", "Fusion hybrid — Zero-DCE++"),
    "deep_zerodce": ("deep", "zerodce", "Deep — Zero-DCE++ (zero-reference)"),
}
ZERODCE_NOTE = ("Zero-DCE++ weights are CC BY-NC 4.0 (academic, non-commercial) and are not shipped; "
                "run `python scripts/download_weights.py --only zerodce` locally to enable them.")


class ModeUnavailable(ValueError):
    """The requested mode cannot run here; the message explains why (shown to the user)."""


@dataclass
class ModeStatus:
    available: bool
    reason: str = ""


@dataclass
class EnhanceResult:
    mode: str
    label: str
    original: np.ndarray            # uint8 RGB (after validation / resizing)
    enhanced: np.ndarray            # uint8 RGB
    latency_ms: float
    timing_ms: dict = field(default_factory=dict)
    model: str | None = None
    checkpoint: str | None = None
    device: str = "cpu"
    fp16: bool | None = None
    refinements: dict | None = None  # quality-guided diagnostics
    identity: bool | None = None
    applied: list | None = None
    configured: list | None = None
    qg_config: str | None = None
    metrics: dict = field(default_factory=dict)
    note: str = ""
    info: dict = field(default_factory=dict)


def niqe_available() -> bool:
    return importlib.util.find_spec("pyiqa") is not None


class EnhancementService:
    def __init__(self, pipe: LumenPipeline | None = None, max_side: int = 1024, device: str | None = None,
                 require_verified: bool = True):
        """`require_verified`: Standard / Quality Guided run only if the loaded Retinexformer checkpoint matches
        the official SHA-256 (it may be supplied from outside the image, e.g. a mounted volume)."""
        self.pipe = pipe or LumenPipeline(max_side=max_side, device=device, deep_model="retinexformer")
        self.require_verified = require_verified
        self.checkpoint_check = "not checked"
        self._last_input: np.ndarray | None = None
        self.status = self.check()
        self._research_status: dict[str, ModeStatus] = {}

    # ------------------------------------------------------------------ availability
    def check(self) -> dict[str, ModeStatus]:
        """Load what each mode needs once, and record why a mode is unavailable."""
        st = {CLASSICAL: ModeStatus(True)}
        try:
            enh = self.pipe.get_deep("retinexformer")
            ok, msg = verify_checkpoint(enh.weights_path)
            st[STANDARD] = ModeStatus(True) if ok or not self.require_verified else ModeStatus(
                False, f"Retinexformer checkpoint rejected: {msg}")
            self.checkpoint_check = msg
        except CheckpointError as e:
            st[STANDARD] = ModeStatus(False, f"Retinexformer checkpoint unavailable: {e}")
        except Exception as e:  # noqa: BLE001 - surface any load failure as "unavailable"
            log.exception("Retinexformer failed to load")
            st[STANDARD] = ModeStatus(False, f"Retinexformer failed to load: {e}")
        try:
            self.pipe.get_qg_config()
            qg = ModeStatus(st[STANDARD].available, "" if st[STANDARD].available else st[STANDARD].reason)
        except QGConfigError as e:
            qg = ModeStatus(False, f"Quality Guided mode is disabled: {e}")
        st[QUALITY_GUIDED] = qg
        return st

    def warmup(self) -> None:
        """One tiny inference per available model so the first user request does not pay CUDA/kernel init."""
        if self.status[STANDARD].available:
            self.pipe.run(np.full((64, 64, 3), 0.1, np.float32), "deep", deep_model="retinexformer")

    def research_status(self, key: str) -> ModeStatus:
        if key not in self._research_status:
            _, model, _ = RESEARCH[key]
            try:
                self.pipe.get_deep(model)
                self._research_status[key] = ModeStatus(True)
            except CheckpointError as e:
                note = ZERODCE_NOTE if model == "zerodce" else str(e)
                self._research_status[key] = ModeStatus(False, note)
        return self._research_status[key]

    def available_modes(self) -> list[str]:
        return [m for m in (STANDARD, QUALITY_GUIDED, CLASSICAL) if self.status[m].available]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def load_input(image) -> np.ndarray:
        """Accept a file path (from the upload widget) or an array."""
        if image is None:
            raise ImageValidationError("Please upload an image first.")
        if isinstance(image, (str, Path)):
            return decode_image_file(image)
        return np.asarray(image)

    def _require(self, mode: str) -> None:
        if mode not in self.status:
            raise ModeUnavailable(f"Unknown mode '{mode}'. Choose one of: {', '.join(MODE_LABELS.values())}.")
        if not self.status[mode].available:
            raise ModeUnavailable(self.status[mode].reason)

    @staticmethod
    def _metrics(out: np.ndarray, inp: np.ndarray, latency_ms: float, compute_niqe: bool) -> dict:
        m = {"latency_ms": round(latency_ms, 1), "noise_sigma": round(noise_level(out), 4),
             "edge_preservation_vs_input": round(edge_preservation(out, inp), 4),
             "overexposed_percent": round(100 * overexposure_ratio(out), 2)}
        if compute_niqe:
            if not niqe_available():
                m["niqe"] = "unavailable (pyiqa not installed)"
            else:
                v = niqe(out)
                m["niqe"] = "unavailable" if v is None else round(v, 3)
        return m

    def _result(self, mode, label, r, compute_niqe, latency_ms=None, timing=None, note="") -> EnhanceResult:
        info = r.info if hasattr(r, "info") else {}
        out = r.output if hasattr(r, "output") else r.image
        lat = r.latency_ms if latency_ms is None else latency_ms
        res = EnhanceResult(
            mode=mode, label=label, original=to_uint8(self._last_input), enhanced=to_uint8(out),
            latency_ms=round(lat, 1), timing_ms=timing or {"total": round(lat, 1)},
            model=info.get("deep_model"), checkpoint=info.get("weights"), device=info.get("device", "cpu"),
            fp16=info.get("fp16"), metrics=self._metrics(out, self._last_input, lat, compute_niqe),
            note=note, info=dict(info),
        )
        if "refinements" in info:
            res.refinements, res.identity = info["refinements"], info["identity"]
            res.applied, res.configured = info["applied"], info.get("configured")
            res.qg_config = info.get("qg_config", self.pipe.qg_config_source)
        if mode == CLASSICAL:
            p = classical_config_path()
            res.checkpoint = None
            res.note = note or (f"Classical constants: {p.name} (tuned)" if p.is_file()
                                else "Classical constants: built-in hand-set defaults (configs/classical.yaml missing)")
        return res

    # ------------------------------------------------------------------ public API
    def enhance(self, image, mode: str = STANDARD, compute_niqe: bool = False) -> EnhanceResult:
        self._require(mode)
        arr = self.load_input(image)
        if mode == STANDARD:
            r = self.pipe.run(arr, "deep", deep_model="retinexformer")
            self._last_input = r.input
            return self._result(mode, MODE_LABELS[mode], r, compute_niqe,
                                timing={"model": round(r.latency_ms, 1), "total": round(r.latency_ms, 1)})
        if mode == QUALITY_GUIDED:
            r = self.pipe.run(arr, "quality_guided_hybrid", deep_model="retinexformer")
            self._last_input = r.input
            t = r.info.get("timing_ms", {})
            return self._result(mode, MODE_LABELS[mode], r, compute_niqe,
                                timing={**t, "total": round(r.latency_ms, 1)})
        r = self.pipe.run(arr, "classical")
        self._last_input = r.input
        return self._result(mode, MODE_LABELS[mode], r, compute_niqe)

    def compare_all(self, image, compute_niqe: bool = False) -> list[EnhanceResult]:
        """All available modes. Retinexformer runs once; Quality Guided reuses its output."""
        arr = self.load_input(image)
        results = []
        if self.status[STANDARD].available:
            std = self.pipe.run(arr, "deep", deep_model="retinexformer")
            self._last_input = std.input
            results.append(self._result(STANDARD, MODE_LABELS[STANDARD], std, compute_niqe,
                                        timing={"model": round(std.latency_ms, 1), "total": round(std.latency_ms, 1)}))
            if self.status[QUALITY_GUIDED].available:
                qgh = QualityGuidedHybrid(self.pipe.get_deep("retinexformer"), self.pipe.get_qg_config())
                t0 = time.perf_counter()
                q = qgh.refine(std.output, std.input)  # no second model call
                refine_ms = (time.perf_counter() - t0) * 1000
                q.info.update(self.pipe.deep_info(self.pipe.get_deep("retinexformer")),
                              qg_config=self.pipe.qg_config_source)
                total = std.latency_ms + refine_ms
                results.append(self._result(
                    QUALITY_GUIDED, MODE_LABELS[QUALITY_GUIDED], q, compute_niqe, latency_ms=total,
                    timing={"model": round(std.latency_ms, 1), "refine": round(refine_ms, 1), "total": round(total, 1)},
                    note="Model output shared with Standard (Retinexformer ran once)."))
        cls = self.pipe.run(arr, "classical")
        self._last_input = cls.input
        results.append(self._result(CLASSICAL, MODE_LABELS[CLASSICAL], cls, compute_niqe))
        return results

    def research(self, image, key: str, disabled=(), compute_niqe: bool = False) -> EnhanceResult:
        if key not in RESEARCH:
            raise ModeUnavailable(f"Unknown research method '{key}'.")
        st = self.research_status(key)
        if not st.available:
            raise ModeUnavailable(st.reason)
        mode, model, label = RESEARCH[key]
        valid = set(HYBRID_COMPONENTS) if mode == "hybrid" else set()
        bad = set(disabled or ()) - valid
        if bad:
            raise ModeUnavailable(f"Components {sorted(bad)} cannot be disabled for {label}.")
        r = self.pipe.run(self.load_input(image), mode, set(disabled or ()), deep_model=model)
        self._last_input = r.input
        return self._result(key, label, r, compute_niqe)


# ---------------------------------------------------------------------- presentation helpers
def _yes(b) -> str:
    return "yes" if b else "no"


def status_markdown(status: dict[str, ModeStatus]) -> str:
    lines = []
    for m in (STANDARD, QUALITY_GUIDED, CLASSICAL):
        s = status[m]
        lines.append(f"- **{MODE_LABELS[m]}**: " + ("available" if s.available else f"unavailable — {s.reason}"))
    return "\n".join(lines)


def details_markdown(res: EnhanceResult) -> str:
    t = res.timing_ms
    timing = ", ".join(f"{k} {v:.0f} ms" for k, v in t.items())
    rows = [("Mode", res.label), ("Processing time", timing)]
    if res.model:
        rows += [("Model", res.model), ("Checkpoint", f"`{res.checkpoint}`"), ("Device", res.device),
                 ("FP16", _yes(res.fp16))]
    else:
        rows += [("Model", "none (classical OpenCV pipeline)")]
    if res.qg_config:
        rows.append(("Calibrated config", f"`{Path(res.qg_config).name}`"))
    if res.note:
        rows.append(("Note", res.note))
    h, w = res.original.shape[:2]
    rows.append(("Processed size", f"{w}x{h}"))
    return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def diagnostics_markdown(res: EnhanceResult) -> str:
    if res.refinements is None:
        return "_Refinement diagnostics apply to Quality Guided mode only._"
    lines = ["| Refinement | Configured | Activated | Coverage | Area refined >50% |", "|---|---|---|---|---|"]
    for comp in QG_COMPONENTS:
        configured = comp in (res.configured or ())
        d = res.refinements.get(comp)
        if d is None:
            lines.append(f"| {comp} | {_yes(configured)} | - | - | - |")
        else:
            lines.append(f"| {comp} | {_yes(configured)} | **{_yes(d['active'])}** | {100 * d['coverage']:.2f}% | "
                         f"{100 * d['area_frac']:.2f}% |")
    names = {"highlight": "Highlight roll-off", "noise": "Noise refinement"}
    summary = [f"{names[c]}: **{_yes(c in (res.applied or []))}**" for c in ("highlight", "noise")]
    ident = ("**Identity path used** — no refinement was justified, the output is exactly the Retinexformer output."
             if res.identity else "Identity path not used — refinements were blended into the masked regions only.")
    return "\n".join(lines) + "\n\n" + " · ".join(summary) + "\n\n" + ident
