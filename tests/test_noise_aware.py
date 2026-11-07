"""Experimental noise-aware refinement (lumen/experimental). Production QGH must stay unaffected."""

import numpy as np
import pytest

from lumen.experimental.noise_aware import (
    NoiseAwareConfig, NoiseAwareQGH, estimate_sigma_map, layered_bilateral, noise_aware_denoise,
)
from lumen.hybrid.quality_guided import QualityGuidedConfig


def _scene(h=128, w=160):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.3 + 0.4 * (xx > w / 2)          # vertical edge
    img[..., 1] = 0.4 + 0.2 * np.sin(yy / 6.0)       # smooth texture
    img[..., 2] = 0.5
    return img


def test_sigma_estimate_white_noise():
    rng = np.random.default_rng(0)
    for s in (0.01, 0.03, 0.06):
        y = np.clip(0.5 + rng.normal(0, s, (256, 256)), 0, 1).astype(np.float32)
        _, g = estimate_sigma_map(y)
        assert abs(g - s) / s < 0.15


def test_reduces_noise_and_keeps_edge():
    rng = np.random.default_rng(1)
    clean = _scene()
    noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1).astype(np.float32)
    out, maps, info = noise_aware_denoise(noisy, NoiseAwareConfig(chroma_method="colour_guided"))
    assert out.shape == noisy.shape and out.dtype == np.float32
    assert np.abs(out - clean).mean() < 0.6 * np.abs(noisy - clean).mean()
    step = out[:, 82:86, 0].mean() - out[:, 74:78, 0].mean()  # colour edge (true step 0.4) survives
    assert step > 0.33
    assert info["coverage_luma"] > 0.9


def test_luma_guided_chroma_bleeds_low_luma_contrast_edges():
    """Documents the known limitation of the luma-only chroma guide (why colour_guided exists)."""
    rng = np.random.default_rng(1)
    clean = _scene()
    noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1).astype(np.float32)
    out, _, _ = noise_aware_denoise(noisy, NoiseAwareConfig(chroma_method="joint_guided"))
    assert out[:, 82:86, 0].mean() - out[:, 74:78, 0].mean() < 0.33


def test_clean_image_nearly_untouched():
    clean = _scene()
    out, _, info = noise_aware_denoise(clean, NoiseAwareConfig(luma_floor=0.008, chroma_floor=0.004))
    assert info["coverage_luma"] < 0.05
    assert np.abs(out - clean).max() < 0.02


def test_layered_bilateral_constant_sigma_matches_single():
    import cv2

    rng = np.random.default_rng(2)
    y = rng.random((64, 64)).astype(np.float32)
    a = layered_bilateral(y, np.full_like(y, 0.1), 3)
    b = cv2.bilateralFilter(y, 7, 0.1 * 1.0, 2.25)
    assert np.allclose(a, b, atol=0.02)


def test_config_validation():
    with pytest.raises(ValueError):
        NoiseAwareConfig(luma_method="median")
    with pytest.raises(ValueError):
        NoiseAwareConfig(detail_keep=1.5)
    with pytest.raises(ValueError):
        NoiseAwareConfig.from_dict({"bogus": 1})
    c = NoiseAwareConfig.from_dict({"luma_k": 2, "adaptive": False})
    assert c.luma_k == 2.0 and c.adaptive is False


def test_wrapper_does_not_touch_production_config():
    qg = QualityGuidedConfig(enabled=("noise", "highlight"))
    na = NoiseAwareQGH(None, qg)
    assert qg.enabled == ("noise", "highlight")          # frozen original unchanged
    assert na.qgh.cfg.enabled == ("highlight",)           # QGH noise refinement replaced, highlight kept
    rng = np.random.default_rng(3)
    x = np.clip(_scene() * 0.1 + rng.normal(0, 0.01, (128, 160, 3)), 0, 1).astype(np.float32)
    r = np.clip(x * 8, 0, 1).astype(np.float32)
    res = na.refine(r, x)
    assert res.image.shape == r.shape and "noise_aware" in res.info


def test_detail_protection_keeps_fine_lines():
    """Thin high-contrast lines on a mildly noisy background survive better with edge_protect."""
    rng = np.random.default_rng(4)
    clean = np.full((128, 128, 3), 0.5, np.float32)
    clean[:, ::6] = 0.15                                  # 1-px dark lines every 6 px (texture)
    noisy = np.clip(clean + rng.normal(0, 0.02, clean.shape), 0, 1).astype(np.float32)
    base = NoiseAwareConfig(luma_k=5.0, luma_floor=0.0, chroma_floor=0.0)
    off, _, _ = noise_aware_denoise(noisy, base)
    on, _, _ = noise_aware_denoise(noisy, NoiseAwareConfig(**{**base.as_dict(), "edge_protect": 1.0,
                                                              "detail_ramp": (4.0, 8.0)}))
    contrast = lambda im: im[:, 3::6, 1].mean() - im[:, ::6, 1].mean()  # noqa: E731
    assert contrast(on) > contrast(off)
    assert contrast(on) > 0.8 * contrast(clean)


# ---------------------------------------------------------------- v2 options (iteration 2)
def test_v2_defaults_reproduce_v1():
    rng = np.random.default_rng(5)
    noisy = np.clip(_scene() + rng.normal(0, 0.04, (128, 160, 3)), 0, 1).astype(np.float32)
    base = NoiseAwareConfig()
    a, _, _ = noise_aware_denoise(noisy, base)
    b, _, _ = noise_aware_denoise(noisy, NoiseAwareConfig(image_gate=0.0, texture_gate=0.0, chroma_mask="max",
                                                          residual_protect=0.0, exact_identity=False))
    assert np.array_equal(a, b)


def test_image_gate_returns_input_bit_exact():
    clean = _scene()
    out, maps, info = noise_aware_denoise(clean, NoiseAwareConfig(image_gate=0.01))
    assert info["gated"] and out is clean
    assert not maps["mask_luma"].any()


def test_exact_identity_outside_masks():
    rng = np.random.default_rng(6)
    img = _scene()
    img[:, :80] = np.clip(img[:, :80] + rng.normal(0, 0.05, img[:, :80].shape), 0, 1)  # noisy left half only
    img = img.astype(np.float32)
    out, maps, _ = noise_aware_denoise(img, NoiseAwareConfig(luma_floor=0.008, chroma_mask="luma_noise",
                                                             exact_identity=True))
    untouched = (maps["mask_luma"] == 0) & (maps["mask_chroma"] == 0)
    assert untouched.mean() > 0.2
    assert np.array_equal(out[untouched], img[untouched])


def test_luma_noise_chroma_mask_skips_clean_regions():
    clean = _scene()
    yy, xx = np.mgrid[0:128, 0:160]
    checker = (((yy // 2) + (xx // 2)) % 2).astype(np.float32) * 2 - 1  # fine, roughly isoluminant colour texture
    clean[..., 0] += 0.03 * checker
    clean[..., 2] -= 0.03 * (0.299 / 0.114) * checker
    clean = np.clip(clean, 0, 1).astype(np.float32)
    _, maps, _ = noise_aware_denoise(clean, NoiseAwareConfig(luma_floor=0.008, chroma_floor=0.0005,
                                                             chroma_mask="luma_noise"))
    assert maps["mask_chroma"].max() == 0  # v1 ("max") would filter chroma here via the chroma-sigma ramp
    _, maps_v1, _ = noise_aware_denoise(clean, NoiseAwareConfig(luma_floor=0.008, chroma_floor=0.0005))
    assert maps_v1["mask_chroma"].max() > 0


def test_residual_protection_keeps_strokes_on_noisy_page():
    """Faint thin strokes on a noisy page: the method-noise test gives back the removed strokes."""
    import cv2

    rng = np.random.default_rng(7)
    page = np.full((160, 200, 3), 0.8, np.float32)
    ink = np.zeros((160, 200), np.uint8)
    for yy in range(20, 160, 20):
        cv2.line(ink, (10, yy), (190, yy - 6), 255, 1, cv2.LINE_AA)
    page -= 0.25 * (ink.astype(np.float32) / 255)[..., None]
    noisy = np.clip(page + rng.normal(0, 0.02, page.shape), 0, 1).astype(np.float32)
    strong = NoiseAwareConfig(luma_k=5.0, luma_floor=0.0, chroma_floor=0.0)
    off, _, _ = noise_aware_denoise(noisy, strong)
    on, _, _ = noise_aware_denoise(noisy, NoiseAwareConfig(**{**strong.as_dict(), "residual_protect": 1.0,
                                                              "residual_ramp": (1.5, 3.0)}))
    stroke = ink > 128
    err = lambda im: np.abs(im[..., 1] - page[..., 1])[stroke].mean()  # noqa: E731
    assert err(on) < err(off)
