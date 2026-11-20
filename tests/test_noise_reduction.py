"""Noise reduction for very noisy inputs (lumen/hybrid/noise_reduction.py) and its wiring into the pipeline and service.

Uses a fake model, so every test runs without a checkpoint (CI included)."""

import sys
from pathlib import Path

import numpy as np
import pytest

import lumen.deep
from lumen import LumenPipeline
from lumen.hybrid.noise_reduction import NoiseReductionConfig, NoiseReductionConfigError, load_nr_config, reduce_noise
from lumen.hybrid.quality_guided import QualityGuidedHybrid
from lumen.metrics.quality import noise_level

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import service as S


class FakeDeep:
    """Brightens x4 like a low-light model, which also amplifies the input noise x4."""

    def __init__(self, model="retinexformer", weights=None, device=None, **_):
        self.name, self.pretrained, self.device, self.fp16 = model, True, "cpu", False
        self.weights_path = Path(f"{model}_fake.pth")

    def __call__(self, img, iterations=1):
        return np.clip(img * 4.0, 0, 1).astype(np.float32), np.zeros(img.shape[:2], np.float32)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(lumen.deep, "DeepEnhancer", FakeDeep)


def dark(h=96, w=128, noise=0.0, seed=0):
    """A smooth dark scene; `noise` is the std of added Gaussian sensor noise."""
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    img = np.stack([0.10 + 0.05 * xx, 0.08 + 0.05 * yy, 0.07 + 0.03 * xx], -1)
    img = img + np.random.default_rng(seed).normal(0, noise, img.shape) if noise else img
    return np.clip(img, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- the step itself
def test_production_config_loads_and_is_strict(tmp_path):
    cfg = load_nr_config()
    assert 0 < cfg.gate_low < cfg.gate_high and cfg.template % 2 == 1 and cfg.search % 2 == 1
    with pytest.raises(NoiseReductionConfigError):
        load_nr_config(tmp_path / "missing.yaml")
    for body in ("params: {gate_low: 0.05, gate_high: 0.03}\n",      # inverted gate
                 "params: {template: 6}\n",                           # even window
                 "params: {strength: 0.8, bogus: 1}\n",               # unknown key
                 "nonsense: true\n"):
        p = tmp_path / "bad.yaml"
        p.write_text(body)
        with pytest.raises(NoiseReductionConfigError):
            load_nr_config(p)


def test_below_the_gate_the_image_is_returned_bit_for_bit():
    r = dark(noise=0.05) * 4
    cfg = NoiseReductionConfig()
    for sigma in (0.0, 0.004, 0.0068, cfg.gate_low - 1e-4):          # every benchmark and phone photo sits here
        out, info = reduce_noise(r, sigma, cfg)
        assert out is r and not info["applied"] and info["weight"] == 0


def test_above_the_gate_noise_is_removed_and_the_ramp_blends():
    clean = np.clip(dark() * 4, 0, 1)
    r = np.clip(clean + np.random.default_rng(1).normal(0, 0.08, clean.shape), 0, 1).astype(np.float32)
    cfg = NoiseReductionConfig()
    full, info = reduce_noise(r, 0.05, cfg)
    assert info["applied"] and info["weight"] == 1 and info["h"] > 0
    assert np.abs(full - clean).mean() < 0.5 * np.abs(r - clean).mean()       # much closer to the clean image
    half, hinfo = reduce_noise(r, (cfg.gate_low + cfg.gate_high) / 2, cfg)
    assert 0 < hinfo["weight"] < 1
    np.testing.assert_allclose(half, (1 - hinfo["weight"]) * r + hinfo["weight"] * full, atol=1e-5)


def test_production_config_floor_moderate_band_and_heavy_band():
    cfg = load_nr_config()
    assert cfg.moderate and 0 < cfg.floor < cfg.gate_low
    clean = np.clip(dark() * 4, 0, 1)
    rng = np.random.default_rng(2)
    r = np.clip(clean + rng.normal(0, 0.035, clean.shape), 0, 1).astype(np.float32)
    for sigma in (0.0, 0.0043, 0.0068, cfg.floor - 1e-4):   # eval15 max 0.0068, phone photos max 0.0043
        out, info = reduce_noise(r, sigma, cfg)
        assert out is r and not info["applied"]
    mod, info = reduce_noise(r, 0.02, cfg)                     # moderate grain: the detail-protecting filter
    assert info["applied"] and info["level"] == "moderate" and info["weight"] == 0
    assert np.abs(mod - clean).mean() < np.abs(r - clean).mean()
    heavy, hinfo = reduce_noise(r, 0.05, cfg)                  # heavy grain: exactly the iteration-1 filter
    assert hinfo["level"] == "heavy" and np.array_equal(heavy, reduce_noise(r, 0.05, NoiseReductionConfig())[0])


def test_moderate_block_is_validated(tmp_path):
    p = tmp_path / "nr.yaml"
    p.write_text("params: {moderate_low: 0.05, moderate_high: 0.06}\nmoderate: {luma_k: 2.0}\n")   # above gate_low
    with pytest.raises(NoiseReductionConfigError):
        load_nr_config(p)
    p.write_text("params: {moderate_low: 0.01, moderate_high: 0.014}\nmoderate: {not_a_key: 1}\n")
    with pytest.raises(NoiseReductionConfigError):
        load_nr_config(p)


# ----------------------------------------------------------------------------- pipeline
def test_pipeline_flag_is_off_by_default_and_never_touches_quick(fake):
    pipe = LumenPipeline(max_side=256, deep_model="retinexformer")
    x = dark(noise=0.05)
    assert "noise_reduction" not in pipe.run(x, "deep").info
    with pytest.raises(ValueError):
        pipe.run(x, "classical", noise_reduction=True)
    with pytest.raises(ValueError):
        pipe.run(x, "hybrid", noise_reduction=True)


@pytest.mark.parametrize("mode", ["deep", "quality_guided_hybrid"])
@pytest.mark.parametrize("hue", [False, True])
def test_low_noise_input_is_bit_identical_with_the_step_on(fake, mode, hue):
    pipe = LumenPipeline(max_side=256, deep_model="retinexformer")
    x = dark(noise=0.003)
    assert noise_level(x) < pipe.get_nr_config().gate_low
    on = pipe.run(x, mode, highlight_hue=hue, noise_reduction=True)
    off = pipe.run(x, mode, highlight_hue=hue)
    assert np.array_equal(on.output, off.output) and on.info["noise_reduction"]["applied"] is False


def test_very_noisy_input_is_denoised_in_the_validated_order(fake):
    pipe = LumenPipeline(max_side=256, deep_model="retinexformer")
    x = dark(noise=0.05)
    assert noise_level(x) >= pipe.get_nr_config().gate_high
    nat = pipe.run(x, "deep", highlight_hue=True, noise_reduction=True)
    raw = pipe.run(x, "deep", highlight_hue=True)
    assert nat.info["noise_reduction"]["applied"] and noise_level(nat.output) < 0.5 * noise_level(raw.output)
    # Highlight-safe = QGH(NR(hue(model))): the same as refining the Natural output, which compare-all relies on
    hs = pipe.run(x, "quality_guided_hybrid", highlight_hue=True, noise_reduction=True)
    manual = QualityGuidedHybrid(None, pipe.get_qg_config()).refine(nat.output, nat.input).image
    assert np.array_equal(hs.output, manual)
    assert hs.info["noise_reduction"]["applied"] and "noise_reduction" in hs.info["timing_ms"]


# ----------------------------------------------------------------------------- service
def svc():
    return S.EnhancementService(LumenPipeline(max_side=256, deep_model="retinexformer"), require_verified=False)


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def test_service_enables_it_and_compare_all_shares_one_denoised_output(fake):
    s = svc()
    assert s.nr_enabled and s.nr_reason == ""
    res = {r.mode: r for r in s.compare_all(to_u8(dark(noise=0.05)))}
    for mode in (S.STANDARD, S.QUALITY_GUIDED):
        assert res[mode].info["noise_reduction"]["applied"]
    assert "noise_reduction" not in res[S.CLASSICAL].info                     # Quick is never touched
    single = s.enhance(to_u8(dark(noise=0.05)), S.QUALITY_GUIDED)
    assert np.array_equal(single.enhanced, res[S.QUALITY_GUIDED].enhanced)    # same result either way


def test_switch_off_and_bad_config_disable_only_this_step(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("LUMEN_NOISE_REDUCTION", "0")
    s = svc()
    assert not s.nr_enabled and "LUMEN_NOISE_REDUCTION=0" in s.nr_reason
    assert "noise_reduction" not in s.enhance(to_u8(dark(noise=0.05)), S.STANDARD).info
    monkeypatch.delenv("LUMEN_NOISE_REDUCTION")
    monkeypatch.setenv("LUMEN_NR_CONFIG", str(tmp_path / "missing.yaml"))
    s = svc()
    assert not s.nr_enabled and "Noise reduction is disabled" in s.nr_reason
    assert s.available_modes() == [S.STANDARD, S.QUALITY_GUIDED, S.CLASSICAL]  # the styles themselves still work
