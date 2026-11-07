"""Classical-only enhancement pipeline with intermediate outputs and ablation switches."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lumen.classical.adaptive import AdaptiveConfig, EnhanceParams, ImageStats, analyze_image, select_params
from lumen.classical.contrast import adjust_saturation, clahe_lab, gray_world, highlight_compress
from lumen.classical.filters import noise_adaptive_denoise
from lumen.classical.illumination import estimate_illumination, retinex_enhance

# Components that can be switched off for ablation studies
COMPONENTS = ("retinex", "denoise", "clahe", "color", "adaptive")

# Used when the "adaptive" component is ablated: one fixed setting for every image
FIXED_PARAMS = EnhanceParams(
    gamma=0.45, illum_radius=15, illum_eps=1e-3, denoise_strength=0.08, denoise_radius=3,
    clahe_clip=2.0, clahe_tile=8, wb_strength=0.5, saturation=1.1,
)


@dataclass
class ClassicalOutput:
    image: np.ndarray
    illumination: np.ndarray
    stats: ImageStats
    params: EnhanceParams
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)


class ClassicalEnhancer:
    def __init__(self, disable: set[str] | None = None, config: AdaptiveConfig | None = None):
        self.disable = set(disable or ())
        self.config = config  # None -> configs/classical.yaml (or built-in defaults)

    def __call__(self, img: np.ndarray, stats: ImageStats | None = None) -> ClassicalOutput:
        """`stats` may be passed in when already computed (e.g. cached during tuning)."""
        stats = stats or analyze_image(img)
        params = FIXED_PARAMS if "adaptive" in self.disable else select_params(stats, self.config)
        steps: dict[str, np.ndarray] = {}

        L = estimate_illumination(img, params.illum_radius, params.illum_eps)
        steps["illumination"] = np.repeat(L[..., None], 3, -1)

        x = img
        if "retinex" not in self.disable:
            x = retinex_enhance(x, L, params.gamma)
            steps["retinex"] = x
        if "denoise" not in self.disable:
            x = noise_adaptive_denoise(x, L, "bilateral", params.denoise_strength, params.denoise_radius)
            steps["denoised"] = x
        if "clahe" not in self.disable:
            x = clahe_lab(x, params.clahe_clip, params.clahe_tile)
            steps["clahe"] = x
        if "color" not in self.disable:
            x = adjust_saturation(gray_world(x, params.wb_strength), params.saturation)
            steps["color"] = x
        x = highlight_compress(x)
        return ClassicalOutput(x.astype(np.float32), L, stats, params, steps)
