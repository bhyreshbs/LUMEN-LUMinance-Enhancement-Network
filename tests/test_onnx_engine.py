"""The optional ONNX Runtime engine: opt-in, verified against the checkpoint, identical to PyTorch, falls back visibly."""

import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from lumen.deep import DeepEnhancer  # noqa: E402
from lumen.deep.onnx_engine import OnnxRetinexformer, OnnxUnavailable, onnx_paths  # noqa: E402

CKPT = ROOT / "weights" / "retinexformer_LOL_v1.pth"
ort = pytest.importorskip("onnxruntime")
needs_export = pytest.mark.skipif(not all(p.is_file() for p in (CKPT, *onnx_paths(CKPT))),
                                  reason="run scripts/export_onnx.py first (needs the checkpoint)")


def image(h=120, w=168, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 0.2).astype(np.float32)


def test_default_engine_is_pytorch(monkeypatch):
    monkeypatch.delenv("LUMEN_ENGINE", raising=False)
    if not CKPT.is_file():
        pytest.skip("checkpoint not present")
    e = DeepEnhancer("retinexformer", CKPT, "cpu")
    assert e.engine == "torch" and e.engine_note == ""


def test_unknown_engine_is_rejected():
    if not CKPT.is_file():
        pytest.skip("checkpoint not present")
    with pytest.raises(ValueError, match="LUMEN_ENGINE"):
        DeepEnhancer("retinexformer", CKPT, "cpu", engine="tensorrt")


@needs_export
def test_onnx_output_equals_pytorch_to_float_precision():
    pt = DeepEnhancer("retinexformer", CKPT, "cpu", engine="torch")
    ox = DeepEnhancer("retinexformer", CKPT, "cpu", engine="onnx")
    assert ox.engine == "onnx" and ox.engine_note == ""
    for h, w, seed in ((120, 168, 0), (203, 301, 1), (64, 64, 2)):     # includes sizes that need padding
        x = image(h, w, seed)
        a, aux_a = pt(x)
        b, aux_b = ox(x)
        assert a.shape == b.shape == (h, w, 3) and aux_a.shape == aux_b.shape == (h, w)
        assert np.abs(a - b).max() < 1e-4 and np.abs(aux_a - aux_b).max() < 1e-4


@needs_export
def test_tiled_inference_is_identical_too():
    pt = DeepEnhancer("retinexformer", CKPT, "cpu", engine="torch", tile=96, tile_overlap=16)
    ox = DeepEnhancer("retinexformer", CKPT, "cpu", engine="onnx", tile=96, tile_overlap=16)
    x = image(150, 220, 3)
    assert np.abs(pt(x)[0] - ox(x)[0]).max() < 1e-4


@needs_export
def test_checkpoint_hash_is_recorded_in_the_sidecar():
    from lumen.deep.checkpoints import KNOWN_SHA256

    side = json.loads(onnx_paths(CKPT)[1].read_text(encoding="utf-8"))
    assert side["source_sha256"] == KNOWN_SHA256["retinexformer_LOL_v1.pth"]
    assert side["max_abs_difference_vs_torch"] < 1e-4


@needs_export
@pytest.mark.parametrize("damage", ["wrong_checkpoint", "altered_onnx", "missing_sidecar", "missing_model"])
def test_a_bad_export_is_refused_and_falls_back_visibly(tmp_path, damage):
    d = tmp_path
    ck = d / CKPT.name
    shutil.copy(CKPT, ck)
    m, s = onnx_paths(ck)
    shutil.copy(onnx_paths(CKPT)[0], m)
    shutil.copy(onnx_paths(CKPT)[1], s)
    if damage == "wrong_checkpoint":
        meta = json.loads(s.read_text(encoding="utf-8")); meta["source_sha256"] = "0" * 64; s.write_text(json.dumps(meta))
    elif damage == "altered_onnx":
        m.write_bytes(m.read_bytes()[:-8] + b"tampered")
    elif damage == "missing_sidecar":
        s.unlink()
    else:
        m.unlink()
    with pytest.raises(OnnxUnavailable):
        OnnxRetinexformer(ck, 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e = DeepEnhancer("retinexformer", ck, "cpu", engine="onnx")
    assert e.engine == "torch" and "ONNX engine unavailable" in e.engine_note
    assert any("ONNX engine unavailable" in str(x.message) for x in w)
    assert np.abs(e(image())[0] - DeepEnhancer("retinexformer", ck, "cpu")(image())[0]).max() == 0   # still correct, just PyTorch


def test_engine_applies_to_retinexformer_on_cpu_only():
    if not CKPT.is_file():
        pytest.skip("checkpoint not present")
    e = DeepEnhancer("zerodce", allow_random_init=True, device="cpu", engine="onnx")
    assert e.engine == "torch" and "Retinexformer on CPU only" in e.engine_note


@needs_export
def test_pipeline_and_status_report_the_engine(monkeypatch):
    from fastapi.testclient import TestClient

    import server as SRV
    import service as S
    from lumen import LumenPipeline

    monkeypatch.setenv("LUMEN_ENGINE", "onnx")
    pipe = LumenPipeline(max_side=256, device="cpu", deep_model="retinexformer", weights=str(CKPT))
    r = pipe.run(image(96, 128), "deep", deep_model="retinexformer")
    assert r.info["engine"] == "onnx"
    st = TestClient(SRV.create_app(S.EnhancementService(pipe))).get("/api/status").json()["technical"]
    assert st["engine"] == "onnx" and st["engine_note"] == ""
