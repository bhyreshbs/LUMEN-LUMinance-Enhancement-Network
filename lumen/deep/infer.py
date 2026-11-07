"""Deep-model registry and numpy inference wrapper.

    DeepEnhancer("zerodce")        # Zero-DCE++ (zero-reference, ~10k params)
    DeepEnhancer("retinexformer")  # Retinexformer (supervised, ~1.6M params)

Handles: reflect padding to a multiple of 8, tiled inference with feathered overlap for images whose
long side exceeds `tile` (default 1024), fp16 autocast on CUDA, CPU fallback when CUDA is missing,
and a retry with smaller tiles (then CPU) on CUDA out-of-memory.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"
PAD_MULTIPLE = 8


class CheckpointError(FileNotFoundError):
    """A required checkpoint is missing or cannot be loaded."""


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    build: Callable[[], nn.Module]
    load: Callable[[nn.Module, Path], None]
    weights: list[Path] = field(default_factory=list)  # first existing file wins
    weights_env: str = ""                               # env var that overrides the weights path
    multi_pass: bool = False                            # supports iterations > 1


def _build_zerodce():
    from lumen.deep.zero_dce import ZeroDCEpp

    return ZeroDCEpp()


def _load_zerodce(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state, strict=True)


def _build_retinexformer():
    from lumen.deep.retinexformer import build_lol_model

    return build_lol_model()


def _load_retinexformer(model, path):
    from lumen.deep.retinexformer import load_checkpoint

    load_checkpoint(model, path)


REGISTRY: dict[str, ModelSpec] = {
    "zerodce": ModelSpec(
        "zerodce", "Zero-DCE++ curve estimation (zero-reference)", _build_zerodce, _load_zerodce,
        [WEIGHTS / "zero_dce_pp_finetuned.pth", WEIGHTS / "zero_dce_pp.pth"],
        "LUMEN_ZERODCE_WEIGHTS", multi_pass=True,
    ),
    "retinexformer": ModelSpec(
        "retinexformer", "Retinexformer one-stage Retinex transformer (supervised, ICCV 2023)",
        _build_retinexformer, _load_retinexformer,
        # fine-tuned file is only written by scripts/train_retinexformer.py when it beats the official val PSNR
        [WEIGHTS / "retinexformer_finetuned.pth", WEIGHTS / "retinexformer_LOL_v1.pth"],
        "LUMEN_RETINEXFORMER_WEIGHTS",
    ),
}


def available_models() -> list[str]:
    return list(REGISTRY)


# ----------------------------------------------------------------------------- helpers
def pick_device(device: str | None = None) -> torch.device:
    device = device or os.environ.get("LUMEN_DEVICE")
    if device:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available - falling back to CPU.", stacklevel=3)
            return torch.device("cpu")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _forward_with_aux(model: nn.Module, name: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (restored, 3-channel auxiliary map): Zero-DCE++ curve map A, or Retinexformer light-up map."""
    if name == "zerodce":
        out, a = model(x)
        return out, a.abs()
    illu_map = None
    for stage in model.body:  # replicate RetinexFormer_Single_Stage.forward, keeping the first light-up map
        illu_fea, m = stage.estimator(x)
        illu_map = m if illu_map is None else illu_map
        x = stage.denoiser(x * m + x, illu_fea)
    return x, illu_map.abs()


def _ramp(n: int, overlap: int) -> np.ndarray:
    """1-D feathering weights: linear ramps of length `overlap` at both ends."""
    w = np.ones(n, np.float32)
    if overlap > 0:
        r = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
        k = min(overlap, n // 2)
        w[:k], w[n - k:] = r[:k], r[::-1][:k]
    return w


def _starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    s = list(range(0, size - tile, stride)) + [size - tile]
    return sorted(set(s))


# ----------------------------------------------------------------------------- enhancer
class DeepEnhancer:
    def __init__(
        self,
        model: str = "zerodce",
        weights: str | Path | None = None,
        device: str | None = None,
        fp16: bool | None = None,
        tile: int = 1024,
        tile_overlap: int = 64,
        allow_random_init: bool = False,
    ):
        """`allow_random_init=True` is for unit tests / architecture experiments only: by default a
        missing or unloadable checkpoint raises CheckpointError, so inference never runs on random
        weights."""
        # Backwards compatibility: DeepEnhancer(<path to weights>) used to mean Zero-DCE++ weights
        if model not in REGISTRY and Path(str(model)).suffix == ".pth":
            model, weights = "zerodce", model
        if model not in REGISTRY:
            raise ValueError(f"Unknown deep model '{model}'. Available: {available_models()}")
        self.spec = REGISTRY[model]
        self.name = model
        self.device = pick_device(device)
        self.fp16 = (self.device.type == "cuda") if fp16 is None else (fp16 and self.device.type == "cuda")
        self.tile, self.tile_overlap = tile, tile_overlap
        self.allow_random_init = allow_random_init
        self.model = self.spec.build().eval()
        self.weights_path = self._load(weights)
        self.model.to(self.device)

    @property
    def pretrained(self) -> bool:
        return self.weights_path is not None

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _load(self, weights) -> Path | None:
        env = os.environ.get(self.spec.weights_env) if self.spec.weights_env else None
        explicit = weights or env
        candidates = [Path(explicit)] if explicit else self.spec.weights
        for p in candidates:
            if p.is_file():
                try:
                    self.spec.load(self.model, p)
                except Exception as e:  # corrupt file, wrong architecture, truncated download, ...
                    raise CheckpointError(f"{self.name} checkpoint {p} could not be loaded: {e}") from e
                return p
        searched = ", ".join(str(c) for c in candidates)
        if explicit or not self.allow_random_init:
            raise CheckpointError(
                f"No {self.name} checkpoint found (looked for: {searched}). "
                "Run `python scripts/download_weights.py`."
            )
        warnings.warn(f"No {self.name} weights found - using RANDOM weights (allow_random_init=True).",
                      stacklevel=3)
        return None

    # -- core ---------------------------------------------------------------------------------
    def _run_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [1,3,h,w] on device. Reflect-pads to PAD_MULTIPLE, runs, crops back."""
        h, w = x.shape[-2:]
        ph, pw = (-h) % PAD_MULTIPLE, (-w) % PAD_MULTIPLE
        if ph or pw:
            mode = "reflect" if (ph < h and pw < w) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
            out, aux = _forward_with_aux(self.model, self.name, x)
        return out[..., :h, :w].float().clamp_(0, 1), aux[..., :h, :w].float()

    def _run_tiled(self, img: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        if max(h, w) <= tile:
            x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(self.device)
            o, a = self._run_tensor(x)
            return o[0].permute(1, 2, 0).cpu().numpy(), a[0].permute(1, 2, 0).cpu().numpy()

        th, tw = min(tile, h), min(tile, w)
        ov = min(self.tile_overlap, th // 4, tw // 4)
        out = np.zeros((h, w, 3), np.float32)
        aux = np.zeros((h, w, 3), np.float32)
        wsum = np.zeros((h, w, 1), np.float32)
        for y0 in _starts(h, th, th - ov):
            for x0 in _starts(w, tw, tw - ov):
                patch = img[y0:y0 + th, x0:x0 + tw]
                t = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))[None].to(self.device)
                o, a = self._run_tensor(t)
                wt = (_ramp(patch.shape[0], ov)[:, None] * _ramp(patch.shape[1], ov)[None, :])[..., None]
                out[y0:y0 + th, x0:x0 + tw] += o[0].permute(1, 2, 0).cpu().numpy() * wt
                aux[y0:y0 + th, x0:x0 + tw] += a[0].permute(1, 2, 0).cpu().numpy() * wt
                wsum[y0:y0 + th, x0:x0 + tw] += wt
        return out / wsum, aux / wsum

    @torch.inference_mode()
    def __call__(self, img: np.ndarray, iterations: int = 1) -> tuple[np.ndarray, np.ndarray]:
        if self.weights_path is None and not self.allow_random_init:  # defence in depth
            raise CheckpointError(f"{self.name} has no loaded checkpoint; refusing to run random weights.")
        """Enhance an HxWx3 float image in [0,1]. Returns (output, aux map HxW in [0,1]).

        `iterations` > 1 re-applies multi-pass models (Zero-DCE++) for extremely dark inputs;
        it is ignored by single-pass models.
        """
        x = np.ascontiguousarray(img, dtype=np.float32)
        n = max(1, iterations) if self.spec.multi_pass else 1
        aux = None
        for _ in range(n):
            x, a = self._run_with_fallback(x)
            aux = a if aux is None else aux
        m = aux.mean(axis=2)
        m = m / (np.percentile(m, 99.5) + 1e-6)  # normalise for display
        return np.clip(x, 0, 1).astype(np.float32), np.clip(m, 0, 1).astype(np.float32)

    def _run_with_fallback(self, img: np.ndarray):
        tile = self.tile
        while True:
            try:
                return self._run_tiled(img, tile)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if tile > 256:
                    tile //= 2
                    warnings.warn(f"CUDA OOM - retrying {self.name} with {tile}px tiles.", stacklevel=2)
                    continue
                warnings.warn(f"CUDA OOM at {tile}px tiles - falling back to CPU for {self.name}.", stacklevel=2)
                self.device, self.fp16 = torch.device("cpu"), False
                self.model.to(self.device)
                tile = self.tile
