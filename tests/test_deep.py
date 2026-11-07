import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from lumen import LumenPipeline
from lumen.deep import REGISTRY, DeepEnhancer, available_models, pick_device
from lumen.deep.retinexformer import build_lol_model, load_checkpoint
from lumen.metrics import psnr

ROOT = Path(__file__).resolve().parents[1]
RF_V1 = ROOT / "weights" / "retinexformer_LOL_v1.pth"
needs_rf = pytest.mark.skipif(not RF_V1.is_file(), reason="run scripts/download_weights.py")


def dark(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 0.15).astype(np.float32)


def test_registry_lists_both_models():
    assert set(available_models()) == {"zerodce", "retinexformer"}
    assert REGISTRY["zerodce"].multi_pass and not REGISTRY["retinexformer"].multi_pass


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown deep model"):
        DeepEnhancer("unet9000", device="cpu")


def test_missing_explicit_weights_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeepEnhancer("zerodce", weights=tmp_path / "nope.pth", device="cpu")


def test_cuda_request_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="falling back to CPU"):
        assert pick_device("cuda").type == "cpu"


@needs_rf
def test_retinexformer_official_checkpoint_loads_strictly():
    model = load_checkpoint(build_lol_model(), RF_V1)
    assert sum(p.numel() for p in model.parameters()) == 1_605_701  # paper: 1.61M


@needs_rf
@pytest.mark.parametrize("shape", [(37, 53), (400, 600), (17, 129)])
def test_padding_handles_any_size(shape):
    enh = DeepEnhancer("retinexformer", device="cpu")
    out, aux = enh(dark(*shape))
    assert out.shape == (*shape, 3) and aux.shape == shape
    assert np.isfinite(out).all() and 0 <= out.min() and out.max() <= 1
    assert out.mean() > 0.075 * 1.5  # actually brightens


def test_tiling_is_seamless_for_local_model():
    img = dark(300, 520)
    whole = DeepEnhancer("zerodce", device="cpu", tile=4096)(img)[0]
    tiled = DeepEnhancer("zerodce", device="cpu", tile=160, tile_overlap=32)(img)[0]
    assert tiled.shape == whole.shape
    assert psnr(tiled, whole) > 45  # Zero-DCE++ is local, so tiles must agree


@needs_rf
def test_tiling_retinexformer_runs():
    out = DeepEnhancer("retinexformer", device="cpu", tile=256, tile_overlap=32)(dark(300, 700))[0]
    assert out.shape == (300, 700, 3) and np.isfinite(out).all()


def test_fp16_disabled_on_cpu():
    assert DeepEnhancer("zerodce", device="cpu", fp16=True).fp16 is False


def test_iterations_ignored_for_single_pass_models():
    if not RF_V1.is_file():
        pytest.skip("no weights")
    enh = DeepEnhancer("retinexformer", device="cpu")
    img = dark(32, 48)
    assert np.allclose(enh(img, iterations=3)[0], enh(img)[0])


@needs_rf
def test_pipeline_selects_deep_model():
    pipe = LumenPipeline(device="cpu")
    r = pipe.run(dark(40, 64), "deep", deep_model="retinexformer")
    assert r.info["deep_model"] == "retinexformer" and r.info["weights"]
    assert pipe.run(dark(40, 64), "deep").info["deep_model"] == "zerodce"


# ----------------------------------------------------------------------------- training helpers
sys.path.insert(0, str(ROOT / "scripts"))


def test_ssim_loss_identity():
    from train_retinexformer import _gauss, ssim_torch

    x = torch.rand(1, 3, 32, 32)
    assert abs(ssim_torch(x, x, _gauss()).item() - 1.0) < 1e-4
    assert ssim_torch(x, torch.rand(1, 3, 32, 32), _gauss()).item() < 0.5


def test_group_split_keeps_duplicates_on_one_side(tmp_path):
    import cv2

    from train_retinexformer import group_split

    rng = np.random.default_rng(0)
    pairs = []
    for i in range(30):
        base = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        for j in range(2):  # every scene twice, the second a near-duplicate
            img = np.clip(base.astype(int) + j, 0, 255).astype(np.uint8)
            p = tmp_path / f"{i}_{j}.png"
            cv2.imwrite(str(p), img)
            pairs.append((p, p))
    pairs.append((tmp_path / "0_0.png", tmp_path / "0_0.png"))  # an exact duplicate
    train, val, n_exact, n_groups = group_split(pairs, 0.2, seed=0, exact_tol=0.002, group_tol=0.01)
    assert n_exact >= 1 and n_groups == 30
    scene = lambda p: p[0].stem.split("_")[0]  # noqa: E731
    assert not ({scene(p) for p in train} & {scene(p) for p in val})
    assert len(val) > 0 and len(train) > len(val)
