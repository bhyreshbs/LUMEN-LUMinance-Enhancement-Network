from dataclasses import replace

import numpy as np
import pytest
import yaml

from lumen.classical import AdaptiveConfig, ClassicalEnhancer, analyze_image, load_config, save_config, select_params
from lumen.classical.adaptive import DEFAULT_CONFIG_PATH


def _stats():
    rng = np.random.default_rng(0)
    img = np.clip(rng.random((64, 64, 3)) * 0.1, 0, 1).astype(np.float32)
    return img, analyze_image(img)


def test_save_load_roundtrip(tmp_path):
    cfg = replace(AdaptiveConfig(), target_mean=0.52, clahe_tile=16, sat_max=1.45)
    path = tmp_path / "classical.yaml"
    save_config(cfg, path, meta={"note": "test"})
    doc = yaml.safe_load(path.read_text())
    assert doc["meta"]["note"] == "test" and doc["params"]["target_mean"] == 0.52
    loaded = load_config(path)
    assert loaded == cfg and isinstance(loaded.clahe_tile, int)


def test_env_var_selects_config_and_changes_params(tmp_path, monkeypatch):
    img, stats = _stats()
    path = tmp_path / "bright.yaml"
    save_config(replace(AdaptiveConfig(), target_mean=0.6), path)
    monkeypatch.setenv("LUMEN_CLASSICAL_CONFIG", str(path))
    assert load_config().target_mean == 0.6
    # higher target brightness -> stronger brightening (smaller gamma) than the defaults
    assert select_params(stats).gamma < select_params(stats, AdaptiveConfig()).gamma
    assert ClassicalEnhancer()(img).image.mean() > ClassicalEnhancer(config=AdaptiveConfig())(img).image.mean()


def test_missing_default_file_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_CLASSICAL_CONFIG", str(tmp_path / "does_not_exist.yaml"))
    assert load_config() == AdaptiveConfig()


def test_explicit_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump({"params": {"target_meen": 0.5}}))
    with pytest.raises(ValueError, match="target_meen"):
        load_config(path)


def test_edited_file_is_reloaded(tmp_path):
    path = tmp_path / "c.yaml"
    save_config(replace(AdaptiveConfig(), target_mean=0.4), path)
    assert load_config(path).target_mean == 0.4
    save_config(replace(AdaptiveConfig(), target_mean=0.5), path)
    import os
    import time

    os.utime(path, (time.time() + 5, time.time() + 5))  # guarantee a new mtime on coarse filesystems
    assert load_config(path).target_mean == 0.5


@pytest.mark.skipif(not DEFAULT_CONFIG_PATH.is_file(), reason="no tuned config committed yet")
def test_committed_config_is_valid():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert 0.2 < cfg.target_mean < 0.8 and cfg.clahe_tile in (4, 8, 16)
