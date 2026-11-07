"""Quality-guided hybrid: identity path, masks, localized corrections, config, model selection."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import lumen.deep
from lumen import MODES, LumenPipeline
from lumen.hybrid.quality_guided import (
    QG_COMPONENTS, QGConfigError, QualityGuidedConfig, QualityGuidedHybrid, colour_score, feather, highlight_score,
    load_qg_config, noise_score, smoothstep,
)
from lumen.utils import luminance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
H, W = 96, 128


def flat(v=0.45, h=H, w=W):
    return np.full((h, w, 3), v, np.float32)


def textured(h=H, w=W, seed=0):
    """Smooth, mid-tone, artifact-free image: nothing should be 'justified' here."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.45 + 0.08 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    return np.clip(np.stack([base, base * 0.97, base * 1.02], -1), 0, 1).astype(np.float32)


def low_input(r, gain=0.15):
    return (r * gain).astype(np.float32)


def run(r, inp=None, cfg=None, disable=None):
    inp = low_input(r) if inp is None else inp
    return QualityGuidedHybrid(None, cfg or QualityGuidedConfig(), disable).refine(r, inp)


# ----------------------------------------------------------------------------- identity path
def test_identity_when_nothing_enabled():
    r = textured()
    out = run(r, cfg=QualityGuidedConfig(enabled=()))
    assert out.image is r and out.info["identity"] and out.info["applied"] == []


def test_identity_when_no_problem_detected():
    r = textured()
    out = run(r)
    assert out.info["identity"], out.info["refinements"]
    assert np.array_equal(out.image, r)


def test_identity_when_all_refinements_disabled_by_ablation():
    r = flat(0.99)  # would trigger the highlight refinement
    out = run(r, disable=set(QG_COMPONENTS))
    assert out.info["identity"] and np.array_equal(out.image, r)


def test_min_coverage_gates_tiny_regions():
    r = textured()
    r[10:12, 10:12] = 1.0  # 4 px clipped: far below 0.5 % of the image
    out = run(r, cfg=QualityGuidedConfig(enabled=("highlight",)))
    assert not out.info["refinements"]["highlight"]["active"] and out.info["identity"]


# ----------------------------------------------------------------------------- masks
def test_smoothstep_and_feather_are_bounded_and_soft():
    x = np.linspace(-1, 2, 50, dtype=np.float32)
    s = smoothstep(x, 0.0, 1.0)
    assert s.min() == 0 and s.max() == 1 and np.all(np.diff(s) >= 0)
    hard = np.zeros((64, 64), np.float32)
    hard[:, 32:] = 1
    soft = feather(hard, 4.0)
    assert 0 <= soft.min() and soft.max() <= 1
    assert np.abs(np.diff(soft, axis=1)).max() < 0.2  # no hard seam


def test_highlight_mask_localised_to_clipped_region():
    r = textured()
    r[20:60, 30:80] = 0.995
    out = run(r, cfg=QualityGuidedConfig(enabled=("highlight",)))
    m = out.intermediates["mask_highlight"][..., 0]
    assert m[30:50, 40:70].mean() > 0.9 and m[80:, :20].max() < 1e-3
    d = out.info["refinements"]["highlight"]
    assert d["active"] and 0.1 < d["area_frac"] < 0.25


def test_noise_score_higher_on_noisy_flat_patch():
    r = textured()
    rng = np.random.default_rng(0)
    r[:, 64:] = np.clip(r[:, 64:] + rng.normal(0, 0.06, r[:, 64:].shape), 0, 1)
    s = noise_score(r)
    assert s[:, 80:].mean() > 3 * s[:, :48].mean()


def test_colour_mask_requires_reliable_input():
    r = textured()
    r[:, 64:, 0] *= 1.5  # R has a red cast on the right half that the input does not have
    inp = low_input(textured(), gain=0.5)
    drift, reliable = colour_score(r, inp, QualityGuidedConfig())
    assert drift[:, 90:].mean() > 3 * drift[:, :40].mean() and reliable.min() > 0.9
    dark_inp = low_input(textured(), gain=0.02)  # too dark: input chromaticity is not trustworthy
    _, unreliable = colour_score(r, dark_inp, QualityGuidedConfig())
    assert unreliable.max() < 0.1


# ----------------------------------------------------------------------------- localized corrections
def test_highlight_correction_only_inside_mask():
    r = textured()
    r[20:60, 30:80] = 0.995
    out = run(r, cfg=QualityGuidedConfig(enabled=("highlight",)))
    assert luminance(out.image)[30:50, 40:70].mean() < 0.99  # rolled off
    far = np.ones((H, W), bool)
    far[0:80, 14:96] = False  # > 4 sigma away from the patch
    assert np.allclose(out.image[far], r[far], atol=1e-6)  # untouched elsewhere


def test_noise_correction_reduces_local_noise_only_where_noisy():
    r = textured()
    rng = np.random.default_rng(1)
    r[:, 64:] = np.clip(r[:, 64:] + rng.normal(0, 0.06, r[:, 64:].shape), 0, 1)
    cfg = QualityGuidedConfig(enabled=("noise",), noise_ramp=(0.06, 0.12))
    out = run(r, cfg=cfg)
    assert out.info["refinements"]["noise"]["active"]
    assert out.image[:, 80:].std() < 0.8 * r[:, 80:].std()
    assert np.allclose(out.image[:, :40], r[:, :40], atol=1e-3)


def test_colour_correction_moves_towards_input_chromaticity():
    r = textured()
    r[:, 64:, 0] = np.clip(r[:, 64:, 0] * 1.5, 0, 1)
    inp = low_input(textured(), gain=0.5)
    out = run(r, inp, QualityGuidedConfig(enabled=("colour",)))
    assert out.info["refinements"]["colour"]["active"]
    red_ratio = lambda im: (im[:, 90:, 0] / im[:, 90:].sum(-1)).mean()  # noqa: E731
    assert red_ratio(out.image) < red_ratio(r)
    assert np.allclose(luminance(out.image)[:, 90:], luminance(r)[:, 90:], atol=0.02)  # luminance kept


@pytest.mark.parametrize("enabled", [QG_COMPONENTS, ("noise",), ("highlight",)])
def test_output_always_valid(enabled):
    rng = np.random.default_rng(3)
    r = np.clip(textured() + rng.normal(0, 0.05, (H, W, 3)), 0, 1).astype(np.float32)
    r[10:40, 10:40] = 1.0
    out = run(r, cfg=QualityGuidedConfig(enabled=enabled)).image
    assert out.shape == r.shape and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


# ----------------------------------------------------------------------------- config
def test_config_roundtrip_and_validation(tmp_path):
    import yaml

    cfg = replace(QualityGuidedConfig(), enabled=("noise",), noise_ramp=(0.05, 0.07), noise_eps=1e-3)
    p = tmp_path / "qg.yaml"
    p.write_text(yaml.safe_dump({"params": cfg.as_dict()}))
    assert QualityGuidedConfig.from_yaml(p) == cfg
    with pytest.raises(ValueError, match="Unknown quality-guided config keys"):
        QualityGuidedConfig.from_dict({"noise_rampp": [0, 1]})
    with pytest.raises(ValueError, match="Unknown refinements"):
        QualityGuidedConfig.from_dict({"enabled": ["sharpen"]})


def test_env_var_selects_config(tmp_path, monkeypatch):
    import yaml

    p = tmp_path / "qg.yaml"
    p.write_text(yaml.safe_dump({"params": {"enabled": ["highlight"]}}))
    monkeypatch.setenv("LUMEN_QG_CONFIG", str(p))
    assert load_qg_config().enabled == ("highlight",)
    # a missing calibrated config must never silently become the uncalibrated defaults
    monkeypatch.setenv("LUMEN_QG_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(QGConfigError, match="not found"):
        load_qg_config()
    assert load_qg_config(allow_defaults=True) == QualityGuidedConfig()  # explicit research opt-in only


def test_committed_config_is_valid_and_dev_calibrated():
    p = ROOT / "configs" / "quality_guided.yaml"
    if not p.is_file():
        pytest.skip("not calibrated yet")
    import yaml

    doc = yaml.safe_load(p.read_text())
    assert "eval15" not in doc["meta"]["dev"]
    cfg = load_qg_config(p)
    assert set(cfg.enabled) <= set(QG_COMPONENTS)


# ----------------------------------------------------------------------------- pipeline / model selection
class FakeDeep:
    built: list = []
    calls: list = []

    def __init__(self, model="zerodce", weights=None, device=None, **_):
        self.name, self.pretrained, self.device, self.fp16 = model, True, "cpu", False
        self.weights_path = Path(f"{model}_fake.pth")
        FakeDeep.built.append(model)

    def __call__(self, img, iterations=1):
        FakeDeep.calls.append(self.name)
        return np.clip(img * 4, 0, 1).astype(np.float32), np.zeros(img.shape[:2], np.float32)


@pytest.fixture
def fake_deep(monkeypatch):
    FakeDeep.built, FakeDeep.calls = [], []
    monkeypatch.setattr(lumen.deep, "DeepEnhancer", FakeDeep)
    return FakeDeep


def test_mode_registered():
    assert "quality_guided_hybrid" in MODES and MODES[:3] == ("classical", "deep", "hybrid")


def test_defaults_to_retinexformer_and_runs_model_once(fake_deep):
    pipe = LumenPipeline(qg_config=QualityGuidedConfig())
    r = pipe.run(textured() * 0.2, "quality_guided_hybrid")
    assert fake_deep.calls == ["retinexformer"]  # exactly one model call
    assert r.info["deep_model"] == "retinexformer" and r.info["weights"] == "retinexformer_fake.pth"
    assert {"refinements", "identity", "applied", "device", "fp16"} <= set(r.info)


def test_respects_requested_model(fake_deep):
    r = LumenPipeline(qg_config=QualityGuidedConfig()).run(textured() * 0.2, "quality_guided_hybrid",
                                                           deep_model="zerodce")
    assert fake_deep.calls == ["zerodce"] and r.info["deep_model"] == "zerodce"


def test_identity_through_pipeline_returns_model_output(fake_deep):
    img = textured() * 0.2
    pipe = LumenPipeline(qg_config=QualityGuidedConfig(enabled=()))
    q = pipe.run(img, "quality_guided_hybrid").output
    d = pipe.run(img, "deep", deep_model="retinexformer").output
    assert np.array_equal(q, d)


def test_ablation_disables_component(fake_deep):
    img = textured() * 0.25
    img[10:50, 10:60] = 0.3  # x4 -> clipped highlight in the fake model output
    pipe = LumenPipeline(qg_config=QualityGuidedConfig(enabled=("highlight",)))
    assert pipe.run(img, "quality_guided_hybrid").info["refinements"]["highlight"]["active"]
    r = pipe.run(img, "quality_guided_hybrid", {"highlight"})
    assert "highlight" not in r.info["refinements"] and r.info["identity"]


@pytest.mark.skipif(not (ROOT / "weights" / "retinexformer_LOL_v1.pth").is_file(), reason="no weights")
def test_real_model_output_valid():
    img = textured(37, 53) * 0.1
    r = LumenPipeline(device="cpu").run(img, "quality_guided_hybrid")
    assert r.output.shape == img.shape and r.output.dtype == np.float32
    assert 0 <= r.output.min() and r.output.max() <= 1 and r.info["deep_model"] == "retinexformer"


# ----------------------------------------------------------------------------- evaluator
def test_evaluator_includes_mode_and_ablations():
    from evaluate import ablation_configs, main_configs

    assert ("quality_guided_hybrid: retinexformer", "quality_guided_hybrid", set(), "retinexformer") in \
        main_configs(["retinexformer"])
    qg = [c for c in ablation_configs(["retinexformer"]) if c[1] == "quality_guided_hybrid"]
    assert [c[2] for c in qg] == [{c} for c in QG_COMPONENTS] and all(c[3] == "retinexformer" for c in qg)


def test_evaluator_writes_per_image_and_diagnostics(fake_deep, tmp_path):
    import csv
    import json

    import cv2

    from evaluate import evaluate, write_per_image

    for i in range(2):
        cv2.imwrite(str(tmp_path / f"{i}.png"), (textured(32, 48, i) * 0.2 * 255).astype(np.uint8))
    items = [(p, None) for p in sorted(tmp_path.glob("*.png"))]
    rows = []
    cfgs = [("deep: retinexformer", "deep", set(), "retinexformer"),
            ("quality_guided_hybrid: retinexformer", "quality_guided_hybrid", set(), "retinexformer")]
    pipe = LumenPipeline(qg_config=QualityGuidedConfig())
    evaluate(pipe, items, cfgs, False, tmp_path / "s", 0, False, per_image=rows)
    write_per_image(rows, tmp_path / "out")
    table = list(csv.DictReader(open(tmp_path / "out" / "per_image.csv", encoding="utf-8")))
    assert len(table) == 4 and {t["experiment"] for t in table} == {c[0] for c in cfgs}
    diag = json.loads((tmp_path / "out" / "qg_diagnostics.json").read_text())
    assert diag["quality_guided_hybrid: retinexformer"]["images"] == 2
