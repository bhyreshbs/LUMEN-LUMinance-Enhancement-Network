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
