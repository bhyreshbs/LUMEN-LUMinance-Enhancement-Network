"""Experimental refinements under evaluation. Nothing here is used by a production mode.

See docs/NOISE_STUDY.md for the current study (noise-aware refinement).
"""

from lumen.experimental.noise_aware import (  # noqa: F401
    NoiseAwareConfig, NoiseAwareQGH, estimate_sigma_map, load_na_config, noise_aware_denoise,
)
