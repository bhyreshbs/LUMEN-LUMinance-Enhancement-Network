import sys
from pathlib import Path

import numpy as np
import pytest

import lumen.deep
from lumen import MODES, LumenPipeline
from lumen.hybrid.fusion import HYBRID_COMPONENTS
from lumen.classical import COMPONENTS, ClassicalEnhancer, analyze_image, select_params
from lumen.classical.filters import guided_filter
from lumen.metrics import edge_preservation, psnr, ssim


def synthetic_dark(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    scene = np.stack([0.5 + 0.4 * np.sin(8 * xx), 0.5 + 0.4 * np.cos(6 * yy), 0.5 + 0.3 * np.sin(5 * (xx + yy))], -1)
    scene[h // 3 : h // 2, w // 4 : w // 2] = 0.9  # a sharp-edged bright object
    illum = 0.03 + 0.15 * xx[..., None]  # dark and non-uniform
    return np.clip(scene * illum + rng.normal(0, 0.01, scene.shape), 0, 1).astype(np.float32), scene.astype(np.float32)


@pytest.fixture(scope="module")
def pipe():
    return LumenPipeline(device="cpu")


def test_adaptive_params_respond_to_brightness():
    dark, scene = synthetic_dark()
    p_dark = select_params(analyze_image(dark))
    p_bright = select_params(analyze_image(scene))
    assert p_dark.gamma < p_bright.gamma
    assert analyze_image(dark).category in ("extremely_dark", "non_uniform", "low_light")


def test_guided_filter_shape_and_range():
    x = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    y = guided_filter(x, x, 4, 1e-2)
    assert y.shape == x.shape and np.isfinite(y).all()


@pytest.mark.parametrize("mode", MODES)
def test_modes_brighten_and_stay_in_range(pipe, mode):
    dark, _ = synthetic_dark()
    r = pipe.run(dark, mode)
    assert r.output.shape == dark.shape and r.output.dtype == np.float32
    assert 0.0 <= r.output.min() and r.output.max() <= 1.0
    if mode != "deep" or pipe.deep.pretrained:  # untrained deep net is not guaranteed to brighten
        assert r.output.mean() > dark.mean() * 1.5
    assert r.latency_ms > 0


def test_classical_improves_psnr_over_input():
    dark, scene = synthetic_dark()
    out = ClassicalEnhancer()(dark).image
    assert psnr(out, scene) > psnr(dark, scene)
    assert ssim(out, scene) > ssim(dark, scene)
    assert edge_preservation(out, scene) > 0.5


@pytest.mark.parametrize("comp", COMPONENTS)
def test_classical_ablations_run(comp):
    dark, _ = synthetic_dark()
    assert ClassicalEnhancer({comp})(dark).image.shape == dark.shape


@pytest.mark.parametrize("comp", HYBRID_COMPONENTS)
def test_hybrid_ablations_run(pipe, comp):
    dark, _ = synthetic_dark()
    assert pipe.run(dark, "hybrid", {comp}).output.shape == dark.shape


def test_bad_input_rejected(pipe):
    with pytest.raises(ValueError):
        pipe.run(None)
    with pytest.raises(ValueError):
        pipe.run(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        pipe.run(np.zeros((32, 32, 3), np.uint8), mode="magic")


def test_gray_and_rgba_inputs(pipe):
    assert pipe.run(np.full((40, 40), 20, np.uint8), "classical").output.shape == (40, 40, 3)
    assert pipe.run(np.full((40, 40, 4), 20, np.uint8), "classical").output.shape == (40, 40, 3)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAVE_WEIGHTS = all((ROOT / "weights" / f).is_file() for f in ("zero_dce_pp.pth", "retinexformer_LOL_v1.pth"))


# ============================================================================= hybrid deep-model selection
class FakeDeep:
    """Stands in for DeepEnhancer: no model load, records which models were built and called."""

    built: list[str] = []
    calls: list[str] = []

    def __init__(self, model="zerodce", weights=None, device=None, **_):
        self.name = model
        self.pretrained = True
        self.weights_path = Path(f"{model}_fake.pth")
        self.device = "cpu"
        self.fp16 = False
        FakeDeep.built.append(model)

    def __call__(self, img, iterations=1):
        FakeDeep.calls.append(self.name)
        gain = 3.0 if self.name == "retinexformer" else 2.0  # distinguishable outputs
        return np.clip(img * gain, 0, 1).astype(np.float32), np.full(img.shape[:2], 0.5, np.float32)


@pytest.fixture
def fake_deep(monkeypatch):
    FakeDeep.built, FakeDeep.calls = [], []
    monkeypatch.setattr(lumen.deep, "DeepEnhancer", FakeDeep)  # LumenPipeline.get_deep imports it lazily
    return FakeDeep


def tiny_dark(h=48, w=72, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 0.15).astype(np.float32)


# ----------------------------------------------------------------------------- pipeline (mocked)
@pytest.mark.parametrize("model", ["retinexformer", "zerodce"])
def test_hybrid_uses_requested_model(fake_deep, model):
    r = LumenPipeline().run(tiny_dark(), "hybrid", deep_model=model)
    assert set(fake_deep.calls) == {model}  # only the requested model ran
    assert fake_deep.built == [model]       # and only it was loaded
    assert r.info["deep_model"] == model
    assert r.info["weights"] == f"{model}_fake.pth"
    assert r.info["device"] == "cpu" and r.info["fp16"] is False and r.info["deep_pretrained"] is True
    assert f"{model}_map" in r.intermediates and "curve_map" not in r.intermediates


def test_default_hybrid_uses_zerodce(fake_deep):
    r = LumenPipeline().run(tiny_dark(), "hybrid")
    assert set(fake_deep.calls) == {"zerodce"} and r.info["deep_model"] == "zerodce"


def test_hybrid_models_are_cached_and_shared_with_deep_mode(fake_deep):
    pipe = LumenPipeline()
    for _ in range(2):
        pipe.run(tiny_dark(), "hybrid", deep_model="retinexformer")
    pipe.run(tiny_dark(), "deep", deep_model="retinexformer")
    assert fake_deep.built == ["retinexformer"]  # lazy, built once, reused


@pytest.mark.parametrize("comp", HYBRID_COMPONENTS)
def test_hybrid_ablations_keep_selected_model(fake_deep, comp):
    r = LumenPipeline().run(tiny_dark(), "hybrid", {comp}, deep_model="retinexformer")
    assert r.info["deep_model"] == "retinexformer"
    assert "zerodce" not in fake_deep.built and "zerodce" not in fake_deep.calls


def test_hybrid_variants_differ_when_deep_branch_differs(fake_deep):
    pipe = LumenPipeline()
    a = pipe.run(tiny_dark(), "hybrid", deep_model="zerodce").output
    b = pipe.run(tiny_dark(), "hybrid", deep_model="retinexformer").output
    assert not np.allclose(a, b)


# ----------------------------------------------------------------------------- pipeline (real weights)
@pytest.mark.skipif(not HAVE_WEIGHTS, reason="run scripts/download_weights.py")
@pytest.mark.parametrize("model", ["zerodce", "retinexformer"])
def test_real_hybrid_variants_preserve_shape_dtype_range(model):
    img = tiny_dark(37, 53)
    r = LumenPipeline(device="cpu").run(img, "hybrid", deep_model=model)
    assert r.output.shape == img.shape and r.output.dtype == np.float32
    assert np.isfinite(r.output).all() and 0.0 <= r.output.min() and r.output.max() <= 1.0
    assert r.info["deep_model"] == model and r.info["weights"] and r.info["device"] == "cpu"


# ----------------------------------------------------------------------------- evaluator configs
def test_main_configs_cover_every_family_and_model():
    from evaluate import main_configs

    cfgs = main_configs(["zerodce", "retinexformer"])
    assert [(c[0], c[1], c[3]) for c in cfgs] == [
        ("classical", "classical", None),
        ("deep: zerodce", "deep", "zerodce"),
        ("deep: retinexformer", "deep", "retinexformer"),
        ("hybrid: zerodce", "hybrid", "zerodce"),
        ("hybrid: retinexformer", "hybrid", "retinexformer"),
        ("quality_guided_hybrid: zerodce", "quality_guided_hybrid", "zerodce"),
        ("quality_guided_hybrid: retinexformer", "quality_guided_hybrid", "retinexformer"),
    ]


def test_main_configs_respect_deep_models_selection():
    from evaluate import main_configs

    cfgs = main_configs(["retinexformer"])
    assert [c[0] for c in cfgs] == ["classical", "deep: retinexformer", "hybrid: retinexformer",
                                    "quality_guided_hybrid: retinexformer"]
    for label, mode, _, model in cfgs:  # the label always names the model that runs
        if mode != "classical":
            assert label.endswith(model)


def test_ablation_configs_preserve_selected_model():
    from evaluate import ablation_configs

    cfgs = ablation_configs(["retinexformer"])
    hybrid = [c for c in cfgs if c[1] == "hybrid"]
    assert len(hybrid) == len(HYBRID_COMPONENTS)
    assert all(c[3] == "retinexformer" and c[0].startswith("hybrid: retinexformer - ") for c in hybrid)
    assert all(c[3] is None for c in cfgs if c[1] == "classical")
    both = ablation_configs(["zerodce", "retinexformer"])
    assert {c[3] for c in both if c[1] == "hybrid"} == {"zerodce", "retinexformer"}
    assert len({c[0] for c in both}) == len(both)  # labels unique


def test_safe_name_has_no_windows_stream_separator():
    from evaluate import safe_name

    assert safe_name("hybrid: retinexformer - post_color") == "hybrid_retinexformer_-_post_color"
    assert ":" not in safe_name("deep: zerodce")


def test_evaluate_metadata_matches_each_experiment(fake_deep, tmp_path):
    import cv2

    from evaluate import evaluate, main_configs

    for i in range(2):
        cv2.imwrite(str(tmp_path / f"{i}.png"), (tiny_dark(32, 48, i) * 255).astype(np.uint8))
    items = [(p, None) for p in sorted(tmp_path.glob("*.png"))]
    res, meta = evaluate(LumenPipeline(), items, main_configs(["zerodce", "retinexformer"]), False,
                         tmp_path / "samples", 1, use_niqe=False)
    assert set(res) == set(meta)
    assert meta["classical"]["deep_model"] is None
    for model in ("zerodce", "retinexformer"):
        for family in ("deep", "hybrid"):
            m = meta[f"{family}: {model}"]
            assert m["deep_model"] == model and m["weights"] == f"{model}_fake.pth" and m["device"] == "cpu"
    assert not any(":" in p.name for p in (tmp_path / "samples").iterdir())


def test_evaluate_refuses_mislabelled_experiment(fake_deep, tmp_path):
    import cv2

    from evaluate import evaluate

    cv2.imwrite(str(tmp_path / "0.png"), (tiny_dark(32, 48) * 255).astype(np.uint8))

    class WrongModelPipe(LumenPipeline):  # simulates the old bug: hybrid ignores the requested model
        def run(self, img, mode="hybrid", disable=None, deep_model=None):
            return super().run(img, mode, disable, deep_model="zerodce")

    with pytest.raises(RuntimeError, match="requested retinexformer but zerodce ran"):
        evaluate(WrongModelPipe(), [(tmp_path / "0.png", None)],
                 [("hybrid: retinexformer", "hybrid", set(), "retinexformer")], False, tmp_path / "s", 0, False)
