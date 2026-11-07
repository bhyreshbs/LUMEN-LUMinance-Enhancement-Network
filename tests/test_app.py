"""Application-level tests: modes, input validation, missing config / checkpoints, diagnostics, reuse."""

import sys
from pathlib import Path

import numpy as np
import pytest

import lumen.deep
from lumen import LumenPipeline
from lumen.deep import CheckpointError, DeepEnhancer
from lumen.hybrid.quality_guided import QualityGuidedConfig
from lumen.imageio import ImageValidationError, decode_image_file, prepare_array

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import service as S  # noqa: E402


class FakeDeep:
    """Deterministic stand-in for DeepEnhancer that counts model calls."""

    calls: list = []
    missing: set = set()  # model names whose checkpoint "fails to load"

    def __init__(self, model="zerodce", weights=None, device=None, **_):
        if model in FakeDeep.missing:
            raise CheckpointError(f"No {model} checkpoint found (looked for: weights/{model}.pth).")
        self.name, self.pretrained, self.device, self.fp16 = model, True, "cpu", False
        self.weights_path = Path(f"{model}_fake.pth")

    def __call__(self, img, iterations=1):
        FakeDeep.calls.append(self.name)
        out = np.clip(img * 4.0, 0, 1).astype(np.float32)
        return out, np.zeros(img.shape[:2], np.float32)


@pytest.fixture
def fake(monkeypatch):
    FakeDeep.calls, FakeDeep.missing = [], set()
    monkeypatch.setattr(lumen.deep, "DeepEnhancer", FakeDeep)
    return FakeDeep


def svc(require_verified=False, **pipe_kw):
    # the fake model's "checkpoint" is not the official file, so hash verification is off unless tested
    return S.EnhancementService(LumenPipeline(max_side=256, deep_model="retinexformer", **pipe_kw),
                                require_verified=require_verified)


def scene(h=120, w=180, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    img = np.stack([0.5 + 0.3 * np.sin(6 * xx), 0.5 + 0.3 * np.cos(5 * yy), 0.4 + 0.2 * xx], -1)
    img[20:50, 30:90] = 0.98  # a bright object -> x4 clips -> highlight refinement can trigger
    return (np.clip(img * 0.25 + rng.normal(0, 0.005, img.shape), 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------- modes
@pytest.mark.parametrize("mode", [S.STANDARD, S.QUALITY_GUIDED, S.CLASSICAL])
def test_each_mode_produces_valid_output(fake, mode):
    s = svc()
    r = s.enhance(scene(), mode)
    assert r.mode == mode and r.label == S.MODE_LABELS[mode]
    assert r.enhanced.dtype == np.uint8 and r.enhanced.shape == r.original.shape
    assert r.latency_ms > 0 and "total" in r.timing_ms
    if mode == S.CLASSICAL:
        assert r.model is None and fake.calls == []
    else:
        assert r.model == "retinexformer" and r.checkpoint == "retinexformer_fake.pth"
        assert fake.calls == ["retinexformer"]  # exactly one model call per request


def test_default_mode_is_standard_and_all_modes_available(fake):
    s = svc()
    assert s.available_modes() == [S.STANDARD, S.QUALITY_GUIDED, S.CLASSICAL]
    assert s.available_modes()[0] == S.STANDARD


def test_quality_guided_diagnostics(fake):
    r = svc().enhance(scene(), S.QUALITY_GUIDED)
    assert set(r.refinements) == set(r.configured)  # only configured refinements are evaluated
    assert r.configured == list(QualityGuidedConfig.from_yaml(ROOT / "configs/quality_guided.yaml").enabled)
    assert r.identity == (not r.applied)
    assert {"model", "refine", "total"} <= set(r.timing_ms)
    md = S.diagnostics_markdown(r)
    assert "Highlight roll-off" in md and "Noise refinement" in md and "Identity path" in md
    for comp, d in r.refinements.items():
        assert f"{100 * d['coverage']:.2f}%" in md


def test_diagnostics_markdown_reports_identity():
    res = S.EnhanceResult(S.QUALITY_GUIDED, "QG", np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8),
                          1.0, refinements={"noise": {"active": False, "coverage": 0.0, "area_frac": 0.0}},
                          identity=True, applied=[], configured=["noise"])
    md = S.diagnostics_markdown(res)
    assert "Identity path used" in md and "Highlight roll-off: **no**" in md


def test_non_qg_modes_have_no_refinement_table(fake):
    md = S.diagnostics_markdown(svc().enhance(scene(), S.STANDARD))
    assert "Quality Guided mode only" in md


def test_unknown_mode_rejected(fake):
    with pytest.raises(S.ModeUnavailable, match="Unknown mode"):
        svc().enhance(scene(), "turbo")


def test_details_show_model_checkpoint_device_and_time(fake):
    md = S.details_markdown(svc().enhance(scene(), S.QUALITY_GUIDED))
    for needle in ("retinexformer", "retinexformer_fake.pth", "cpu", "model", "refine", "quality_guided.yaml"):
        assert needle in md


# ----------------------------------------------------------------------------- compare all / reuse
def test_compare_all_runs_retinexformer_exactly_once(fake):
    s = svc()
    results = s.compare_all(scene())
    assert [r.mode for r in results] == [S.STANDARD, S.QUALITY_GUIDED, S.CLASSICAL]
    assert fake.calls == ["retinexformer"]  # one call shared by Standard and Quality Guided
    std, qg = results[0], results[1]
    assert qg.timing_ms["model"] == std.timing_ms["model"]
    assert qg.timing_ms["total"] >= std.timing_ms["total"]


def test_compare_all_qg_equals_standalone_qg(fake):
    s = svc()
    cmp_qg = s.compare_all(scene())[1]
    single = s.enhance(scene(), S.QUALITY_GUIDED)
    assert np.array_equal(cmp_qg.enhanced, single.enhanced)
    assert cmp_qg.applied == single.applied and cmp_qg.refinements == single.refinements


def test_models_and_config_loaded_once(fake, monkeypatch):
    built = []
    orig = FakeDeep.__init__

    def counting_init(self, *a, **k):
        built.append(a[0] if a else k.get("model"))
        orig(self, *a, **k)

    monkeypatch.setattr(FakeDeep, "__init__", counting_init)
    import lumen.hybrid.quality_guided as qg

    loads = []
    real = qg.QualityGuidedConfig.from_yaml
    monkeypatch.setattr(qg.QualityGuidedConfig, "from_yaml", classmethod(lambda cls, p: loads.append(p) or real(p)))
    s = svc()
    for _ in range(3):
        s.enhance(scene(), S.STANDARD)
        s.enhance(scene(), S.QUALITY_GUIDED)
        s.compare_all(scene())
    assert built == ["retinexformer"] and len(loads) == 1


# ----------------------------------------------------------------------------- missing config
def test_missing_qg_config_disables_only_quality_guided(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_QG_CONFIG", str(tmp_path / "missing.yaml"))
    s = svc()
    assert not s.status[S.QUALITY_GUIDED].available
    assert "not found" in s.status[S.QUALITY_GUIDED].reason
    assert s.available_modes() == [S.STANDARD, S.CLASSICAL]
    with pytest.raises(S.ModeUnavailable, match="Quality Guided mode is disabled"):
        s.enhance(scene(), S.QUALITY_GUIDED)
    assert s.enhance(scene(), S.STANDARD).enhanced.shape[2] == 3  # other modes keep working
    assert [r.mode for r in s.compare_all(scene())] == [S.STANDARD, S.CLASSICAL]
    assert fake.calls.count("retinexformer") == 2  # never ran QGH on uncalibrated defaults


@pytest.mark.parametrize("content, match", [
    ("params: [1, 2", "not readable YAML"),
    ("params:\n  noise_rampp: [0.1, 0.2]\n", "Unknown quality-guided config keys"),
    ("params:\n  noise_ramp: [0.2, 0.1]\n", "start < full"),
    ("params:\n  enabled: [sharpen]\n", "Unknown refinements"),
    ("- just a list\n", "must contain a 'params:' mapping"),
])
def test_invalid_qg_config_disables_quality_guided(fake, tmp_path, monkeypatch, content, match):
    p = tmp_path / "bad.yaml"
    p.write_text(content)
    monkeypatch.setenv("LUMEN_QG_CONFIG", str(p))
    s = svc()
    assert not s.status[S.QUALITY_GUIDED].available
    assert match in s.status[S.QUALITY_GUIDED].reason and str(p) in s.status[S.QUALITY_GUIDED].reason
    assert s.status[S.STANDARD].available


# ----------------------------------------------------------------------------- missing checkpoints
def test_missing_retinexformer_disables_standard_and_quality_guided(fake):
    fake.missing = {"retinexformer"}
    s = svc()
    assert s.available_modes() == [S.CLASSICAL]
    assert "checkpoint" in s.status[S.STANDARD].reason.lower()
    for mode in (S.STANDARD, S.QUALITY_GUIDED):
        with pytest.raises(S.ModeUnavailable):
            s.enhance(scene(), mode)
    assert [r.mode for r in s.compare_all(scene())] == [S.CLASSICAL]
    assert fake.calls == []  # nothing ran on a model that failed to load


def test_missing_zerodce_makes_research_method_unavailable(fake):
    fake.missing = {"zerodce"}
    s = svc()
    st = s.research_status("deep_zerodce")
    assert not st.available and "CC BY-NC" in st.reason
    with pytest.raises(S.ModeUnavailable):
        s.research(scene(), "deep_zerodce")
    assert s.research(scene(), "hybrid_retinexformer", ["post_color"]).model == "retinexformer"


def test_research_rejects_invalid_ablation(fake):
    with pytest.raises(S.ModeUnavailable, match="cannot be disabled"):
        svc().research(scene(), "hybrid_retinexformer", ["noise"])


def test_real_enhancer_refuses_random_init(tmp_path, monkeypatch):
    from dataclasses import replace

    spec = replace(lumen.deep.REGISTRY["zerodce"], weights=[tmp_path / "nope.pth"])
    monkeypatch.setitem(lumen.deep.REGISTRY, "zerodce", spec)
    with pytest.raises(CheckpointError, match="No zerodce checkpoint"):
        DeepEnhancer("zerodce", device="cpu")
    e = DeepEnhancer("zerodce", device="cpu", allow_random_init=True)  # explicit opt-in only
    assert not e.pretrained


def test_corrupt_checkpoint_raises(tmp_path):
    bad = tmp_path / "corrupt.pth"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="could not be loaded"):
        DeepEnhancer("retinexformer", weights=bad, device="cpu")


# ----------------------------------------------------------------------------- invalid images
@pytest.mark.parametrize("img, match", [
    (None, "upload an image"),
    (np.zeros((8, 8, 3), np.uint8), "too small"),
    (np.zeros((32, 32, 5), np.uint8), "Unsupported image shape"),
    (np.zeros((4, 32, 32, 3), np.uint8), "Unsupported image shape"),
    (np.full((32, 32, 3), np.nan, np.float32), "NaN"),
    (np.full((32, 32, 3), -5, np.int32), "Integer pixel values"),
    (np.zeros((0, 0, 3), np.uint8), "empty"),
])
def test_invalid_arrays_rejected_with_clear_message(fake, img, match):
    with pytest.raises(ImageValidationError, match=match):
        svc().enhance(img, S.STANDARD)
    assert fake.calls == []


def test_oversized_image_rejected_before_float_allocation():
    big = np.lib.stride_tricks.as_strided(np.zeros(1, np.uint8), shape=(8000, 6000, 3), strides=(0, 0, 0))
    with pytest.raises(ImageValidationError, match="too large"):  # 48 MP > 40 MP; no 576 MB float array
        prepare_array(big, 1024)


def test_large_image_downscaled_in_native_dtype():
    img = np.random.default_rng(0).integers(0, 255, (3000, 4000, 3), dtype=np.uint8)
    out = prepare_array(img, 1024)
    assert out.shape == (768, 1024, 3) and out.dtype == np.float32


@pytest.mark.parametrize("arr", [np.full((40, 40), 30, np.uint8), np.full((40, 40, 4), 30, np.uint8),
                                 np.full((40, 40, 3), 3000, np.uint16), np.full((40, 40, 1), 30, np.uint8),
                                 np.ones((40, 40, 3), bool)])
def test_gray_rgba_16bit_and_bool_accepted(arr):
    out = prepare_array(arr, 0)
    assert out.shape == (40, 40, 3) and 0 <= out.min() and out.max() <= 1


def _save(tmp_path, arr, name, **kw):
    from PIL import Image

    p = tmp_path / name
    Image.fromarray(arr).save(p, **kw)
    return p


def test_decode_supported_formats(tmp_path):
    arr = scene()
    for name in ("a.jpg", "a.png", "a.webp", "a.bmp", "a.tif"):
        out = decode_image_file(_save(tmp_path, arr, name))
        assert out.shape == arr.shape and out.dtype == np.uint8


def test_decode_applies_exif_orientation(tmp_path):
    from PIL import Image

    arr = scene(60, 100)
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[274] = 6  # rotate 90 degrees clockwise on display
    p = tmp_path / "rot.jpg"
    im.save(p, exif=exif)
    assert decode_image_file(p).shape == (100, 60, 3)


def test_decode_16bit_png(tmp_path):
    from PIL import Image

    p = tmp_path / "g16.png"
    Image.fromarray(np.full((40, 40), 40000, np.uint16)).save(p)
    out = decode_image_file(p)
    assert out.dtype == np.uint16 and prepare_array(out).max() == pytest.approx(40000 / 65535, abs=1e-4)


@pytest.mark.parametrize("payload, match", [
    (b"", "empty"),
    (b"this is definitely not an image", "not a recognised image"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "not a recognised image|corrupted"),
])
def test_decode_rejects_bad_files(tmp_path, payload, match):
    p = tmp_path / "bad.png"
    p.write_bytes(payload)
    with pytest.raises(ImageValidationError, match=match):
        decode_image_file(p)


def test_decode_rejects_truncated_jpeg(tmp_path):
    p = _save(tmp_path, scene(), "full.jpg")
    t = tmp_path / "trunc.jpg"
    t.write_bytes(p.read_bytes()[:400])
    with pytest.raises(ImageValidationError, match="corrupted|truncated|not a recognised"):
        decode_image_file(t)


def test_decode_rejects_unsupported_format(tmp_path):
    p = _save(tmp_path, scene(), "a.gif")
    with pytest.raises(ImageValidationError, match="Unsupported image format 'GIF'"):
        decode_image_file(p)


def test_decode_rejects_oversized_by_header(tmp_path):
    from PIL import Image

    p = tmp_path / "wide.png"
    Image.new("L", (9000, 5000)).save(p)  # 45 MP, compresses to a tiny file
    with pytest.raises(ImageValidationError, match="too large"):
        decode_image_file(p)


def test_decompression_bomb_rejected(tmp_path, monkeypatch):
    from PIL import Image

    p = tmp_path / "bomb.png"
    Image.new("L", (3000, 3000)).save(p)  # 9 MP
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2_000_000)  # Pillow's guard now fires first
    with pytest.raises(ImageValidationError, match="decompression bomb"):
        decode_image_file(p, max_pixels=50_000_000)


def test_decode_rejects_oversized_file(tmp_path):
    p = _save(tmp_path, scene(), "a.png")
    with pytest.raises(ImageValidationError, match="MB"):
        decode_image_file(p, max_bytes=100)


def test_service_accepts_file_path(fake, tmp_path):
    r = svc().enhance(str(_save(tmp_path, scene(), "in.jpg")), S.STANDARD)
    assert r.enhanced.shape == scene().shape


# ----------------------------------------------------------------------------- checkpoint integrity
def test_unverified_checkpoint_disables_standard_and_qg(fake):
    s = svc(require_verified=True)  # fake checkpoint name is not an official file
    assert s.available_modes() == [S.CLASSICAL]
    assert "rejected" in s.status[S.STANDARD].reason and "not a known official checkpoint" in s.status[S.STANDARD].reason
    assert not s.status[S.QUALITY_GUIDED].available


def test_altered_checkpoint_is_rejected(fake, tmp_path, monkeypatch):
    bad = tmp_path / "retinexformer_LOL_v1.pth"
    bad.write_bytes(b"tampered")

    class AlteredFake(FakeDeep):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.weights_path = bad

    monkeypatch.setattr(lumen.deep, "DeepEnhancer", AlteredFake)
    s = svc(require_verified=True)
    assert not s.status[S.STANDARD].available and "SHA-256 mismatch" in s.status[S.STANDARD].reason


@pytest.mark.skipif(not (ROOT / "weights" / "retinexformer_LOL_v1.pth").is_file(), reason="no checkpoint")
def test_official_checkpoint_verifies():
    from lumen.deep.checkpoints import verify

    ok, msg = verify(ROOT / "weights" / "retinexformer_LOL_v1.pth")
    assert ok, msg


def test_hash_table_matches_download_script():
    import importlib.util

    from lumen.deep.checkpoints import KNOWN_SHA256

    spec = importlib.util.spec_from_file_location("dw", ROOT / "scripts" / "download_weights.py")
    dw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dw)
    assert {k: v[3] for k, v in dw.FILES.items()} == KNOWN_SHA256


# ----------------------------------------------------------------------------- deployment runtime
def test_runtime_threads_respect_env_and_quota(monkeypatch):
    import app as A

    monkeypatch.setenv("LUMEN_TORCH_THREADS", "3")
    assert A.runtime_threads() == 3
    monkeypatch.delenv("LUMEN_TORCH_THREADS")
    monkeypatch.setattr(A, "cpu_quota", lambda: 2)
    assert A.runtime_threads() <= 2
    monkeypatch.setattr(A, "cpu_quota", lambda: None)
    assert A.runtime_threads() >= 1
