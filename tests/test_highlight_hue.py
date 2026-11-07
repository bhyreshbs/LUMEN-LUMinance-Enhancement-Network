"""Experimental highlight hue fix (lumen/experimental/highlight_hue.py): never used by production modes."""

import numpy as np

from lumen.experimental.highlight_hue import HighlightHueConfig, highlight_hue_fix


def _scene():
    x = np.full((64, 64, 3), 0.05, np.float32)
    x[20:40, 20:40] = (0.97, 0.96, 0.92)          # near-clipped white window in a dark room
    r = np.full((64, 64, 3), 0.4, np.float32)      # model output: brightened shadows ...
    r[20:40, 20:40] = (0.75, 0.48, 0.08)           # ... and an orange window
    return r, x


def test_window_hue_restored_and_outside_untouched():
    r, x = _scene()
    out, m, info = highlight_hue_fix(r, x, HighlightHueConfig(lum_keep=0.8))
    assert info["applied"]
    win = out[25:35, 25:35].reshape(-1, 3).mean(0)
    assert win.max() - win.min() < 0.1                  # near neutral again (was 0.67 apart)
    assert np.array_equal(out[:8], r[:8])               # far from the window: bit-exact
    assert out.min() >= 0 and out.max() <= 1


def test_identity_without_bright_input():
    r, x = _scene()
    x[:] = 0.2
    out, _, info = highlight_hue_fix(r, x)
    assert out is r and not info["applied"]


def test_lum_keep_prevents_darkening():
    r, x = _scene()
    dark = highlight_hue_fix(r, x, HighlightHueConfig(lum_keep=0.0))[0][25:35, 25:35].mean()
    kept = highlight_hue_fix(r, x, HighlightHueConfig(lum_keep=1.0))[0][25:35, 25:35].mean()
    assert kept > dark


# ---------------------------------------------------------------- integration (pipeline + service)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumen import LumenPipeline  # noqa: E402
from lumen.hybrid.highlight_hue import HueConfigError, load_hue_config  # noqa: E402


def test_production_config_loads_and_is_strict(tmp_path):
    cfg = load_hue_config()
    assert 0 < cfg.start < cfg.full <= 1
    with pytest.raises(HueConfigError):
        load_hue_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("params: {start: 0.9, full: 0.5}\n")
    with pytest.raises(HueConfigError):
        load_hue_config(bad)
    unk = tmp_path / "unk.yaml"
    unk.write_text("params: {start: 0.8, bogus: 1}\n")
    with pytest.raises(HueConfigError):
        load_hue_config(unk)


def _window_scene():
    x = np.full((64, 80, 3), 0.04, np.float32)
    x[16:40, 24:56] = (0.97, 0.96, 0.92)
    return x


def test_pipeline_flag_is_off_by_default_and_rejects_other_modes():
    pipe = LumenPipeline(max_side=128, deep_model="zerodce")
    x = _window_scene()
    base = pipe.run(x, "deep")
    again = pipe.run(x, "deep", highlight_hue=False)
    assert np.array_equal(base.output, again.output) and "highlight_hue" not in base.info
    with pytest.raises(ValueError):
        pipe.run(x, "classical", highlight_hue=True)       # Quick is never touched
    with pytest.raises(ValueError):
        pipe.run(x, "hybrid", highlight_hue=True)


def test_pipeline_applies_hue_before_qgh_and_leaves_clean_images_identical():
    pipe = LumenPipeline(max_side=128, deep_model="zerodce")
    x = _window_scene()
    on = pipe.run(x, "quality_guided_hybrid", highlight_hue=True)
    assert on.info["highlight_hue"]["applied"] and "highlight_hue" in on.info["timing_ms"]
    dark = np.full((64, 80, 3), 0.05, np.float32)           # nothing near-clipped: bit-identical to the old path
    a = pipe.run(dark, "quality_guided_hybrid", highlight_hue=True)
    b = pipe.run(dark, "quality_guided_hybrid")
    assert np.array_equal(a.output, b.output) and not a.info["highlight_hue"]["applied"]
    d1, d2 = pipe.run(dark, "deep", highlight_hue=True), pipe.run(dark, "deep")
    assert np.array_equal(d1.output, d2.output)
