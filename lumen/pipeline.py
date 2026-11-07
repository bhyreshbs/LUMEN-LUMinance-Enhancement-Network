"""Single entry point used by the app, evaluation and tests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from lumen.imageio import DEFAULT_MAX_PIXELS, prepare_array
from lumen.utils import timer

MODES = ("classical", "deep", "hybrid", "quality_guided_hybrid")
QG_DEFAULT_MODEL = "retinexformer"  # quality-guided mode refines a supervised model's output


@dataclass
class PipelineResult:
    mode: str
    output: np.ndarray
    input: np.ndarray
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class LumenPipeline:
    """Runs the classical, deep or hybrid enhancer. Models are built lazily and reused."""

    def __init__(self, max_side: int = 1280, device: str | None = None, weights: str | None = None,
                 deep_model: str = "zerodce", qg_config=None, max_pixels: int = DEFAULT_MAX_PIXELS):
        self.max_side = max_side
        self.max_pixels = max_pixels  # inputs above this are rejected before any float conversion
        self.device = device
        self.weights = weights        # optional weights path for `deep_model`
        self.deep_model = deep_model  # deep model for mode="deep" and "hybrid" unless run(..., deep_model=...) overrides
        self._deep: dict = {}
        # QualityGuidedConfig for mode="quality_guided_hybrid". None -> the calibrated
        # configs/quality_guided.yaml, loaded once on first use; missing/invalid -> QGConfigError (never defaults)
        self.qg_config = qg_config
        self.qg_config_source = "explicit" if qg_config is not None else None
        self.hue_config = None  # HighlightHueConfig, loaded lazily on first use (highlight_hue=True)

    def get_deep(self, name: str | None = None):
        """Cached DeepEnhancer per registry name ("zerodce" | "retinexformer")."""
        name = name or self.deep_model
        if name not in self._deep:
            from lumen.deep import DeepEnhancer

            weights = self.weights if name == self.deep_model else None
            self._deep[name] = DeepEnhancer(name, weights, self.device)
        return self._deep[name]

    @property
    def deep(self):
        return self.get_deep()

    def get_qg_config(self):
        """The calibrated quality-guided config, loaded and validated once (raises QGConfigError)."""
        if self.qg_config is None:
            from lumen.hybrid.quality_guided import load_qg_config, qg_config_path

            self.qg_config = load_qg_config()
            self.qg_config_source = str(qg_config_path())
        return self.qg_config

    def get_hue_config(self):
        """The highlight hue config, loaded and validated once (raises HueConfigError)."""
        if self.hue_config is None:
            from lumen.hybrid.highlight_hue import load_hue_config

            self.hue_config = load_hue_config()
        return self.hue_config

    def validate(self, img) -> np.ndarray:
        """Validated float32 RGB image (no resizing). Raises ImageValidationError (a ValueError)."""
        return prepare_array(img, 0, self.max_pixels)

    def run(self, img, mode: str = "hybrid", disable: set[str] | None = None,
            deep_model: str | None = None, highlight_hue: bool = False) -> PipelineResult:
        """`highlight_hue=True` (modes "deep" and "quality_guided_hybrid" only) restores the colour of near-clipped
        input regions after the model, before QGH's highlight roll-off. Off by default, so the research and
        evaluation paths reproduce the frozen results exactly."""
        if highlight_hue and mode not in ("deep", "quality_guided_hybrid"):
            raise ValueError("highlight_hue applies to the deep and quality_guided_hybrid modes only")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        x = prepare_array(img, self.max_side, self.max_pixels)  # validates, then resizes before float
        t: dict[str, float] = {}
        with timer(t, "total"):
            if mode == "classical":
                from lumen.classical import ClassicalEnhancer

                r = ClassicalEnhancer(disable)(x)
                out, steps = r.image, r.intermediates
                info = {"stats": r.stats.as_dict(), "params": r.params.as_dict()}
            elif mode == "deep":
                hcfg = self.get_hue_config() if highlight_hue else None  # before the model call, like the QGH config
                enh = self.get_deep(deep_model)
                out, aux = enh(x)
                steps = {f"{enh.name}_map": np.repeat(aux[..., None], 3, -1)}
                info = self.deep_info(enh)
                if highlight_hue:
                    out, hinfo = self._hue(out, x, hcfg)
                    info["highlight_hue"] = hinfo
            elif mode == "quality_guided_hybrid":
                from lumen.hybrid import QualityGuidedHybrid

                cfg = self.get_qg_config()  # before the model call: a bad config must not cost inference
                hcfg = self.get_hue_config() if highlight_hue else None
                enh = self.get_deep(deep_model or QG_DEFAULT_MODEL)
                if not highlight_hue:
                    r = QualityGuidedHybrid(enh, cfg, disable)(x)
                    out, steps = r.image, r.intermediates
                    info = {**r.info, **self.deep_info(enh), "qg_config": self.qg_config_source}
                else:
                    # hue fix on the model output, THEN QGH (its highlight roll-off still runs afterwards)
                    t0 = time.perf_counter()
                    model_out, aux = enh(x)
                    t1 = time.perf_counter()
                    fixed, hinfo = self._hue(model_out, x, hcfg)
                    r = QualityGuidedHybrid(enh, cfg, disable).refine(fixed, x)
                    r.intermediates[f"{enh.name}_map"] = np.repeat(aux[..., None], 3, -1)
                    r.info["timing_ms"] = {"model": round((t1 - t0) * 1000, 1), "refine": r.info["timing_ms"]["refine"],
                                           "highlight_hue": hinfo["ms"]}
                    out, steps = r.image, r.intermediates
                    info = {**r.info, **self.deep_info(enh), "qg_config": self.qg_config_source, "highlight_hue": hinfo}
            else:
                from lumen.hybrid import HybridEnhancer

                # deep_model=None -> self.deep_model, which defaults to "zerodce"
                enh = self.get_deep(deep_model)
                r = HybridEnhancer(enh, disable)(x)
                out, info = r.image, {**r.info, **self.deep_info(enh)}
                # fusion.py names the deep branch's auxiliary map "curve_map" (a Zero-DCE++ term);
                # give it the model's own name so Retinexformer's light-up map is not mislabelled
                steps = {(f"{enh.name}_map" if k == "curve_map" else k): v for k, v in r.intermediates.items()}
        return PipelineResult(mode, out, x, steps, info, t["total"])

    @staticmethod
    def _hue(out, x, cfg):
        from lumen.hybrid.highlight_hue import highlight_hue_fix

        t0 = time.perf_counter()
        fixed, _, h = highlight_hue_fix(out, x, cfg)
        return fixed, {**h, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    @staticmethod
    def deep_info(enh) -> dict:
        """Metadata of the deep model that actually ran. Read *after* inference, because the
        enhancer may have fallen back from CUDA to CPU on out-of-memory."""
        return {
            "deep_model": enh.name,
            "deep_pretrained": enh.pretrained,
            "weights": enh.weights_path.name if enh.weights_path else None,
            "device": str(enh.device),
            "fp16": enh.fp16,
        }
