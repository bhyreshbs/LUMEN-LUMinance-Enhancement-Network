"""Optional ONNX Runtime engine for Retinexformer on CPU (LUMEN_ENGINE=onnx).

The network is exported once by scripts/export_onnx.py into `<checkpoint>.onnx` next to the checkpoint, with a
sidecar `<checkpoint>.onnx.json` that records the SHA-256 of the checkpoint it came from and of the .onnx file itself.
The export is never committed or baked into an image (it is derived from weights whose redistribution rights are
uncertain): like the checkpoint, it is supplied at run time. The engine loads it only if

  * onnxruntime is installed,
  * the file and its sidecar exist,
  * the sidecar's checkpoint hash equals the SHA-256 of the checkpoint actually loaded, and
  * the file's own hash equals the sidecar's.

Anything else raises OnnxUnavailable with the reason, and DeepEnhancer falls back to PyTorch visibly (a warning, and
`engine` / `engine_note` in the run metadata), never silently. Output is the same function as the PyTorch path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lumen.deep.checkpoints import sha256

SIDECAR_FORMAT = 1


class OnnxUnavailable(RuntimeError):
    """The ONNX engine cannot be used; the message says why."""


def onnx_paths(weights: Path) -> tuple[Path, Path]:
    return weights.with_suffix(".onnx"), weights.with_suffix(".onnx.json")


class OnnxRetinexformer:
    """Runs the exported network: x [1,3,h,w] float32 (h, w multiples of 8) -> (restored, light-up map), both numpy."""

    def __init__(self, weights: Path, threads: int, checkpoint_sha256: str | None = None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise OnnxUnavailable("onnxruntime is not installed") from e
        model, side = onnx_paths(Path(weights))
        if not model.is_file() or not side.is_file():
            raise OnnxUnavailable(f"{model.name} / {side.name} not found next to the checkpoint (run scripts/export_onnx.py)")
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise OnnxUnavailable(f"{side.name} is not readable: {e}") from e
        if meta.get("format") != SIDECAR_FORMAT or meta.get("model") != "retinexformer":
            raise OnnxUnavailable(f"{side.name} is not a Retinexformer export of this format")
        ck = checkpoint_sha256 or sha256(weights)
        if meta.get("source_sha256") != ck:
            raise OnnxUnavailable(f"{model.name} was exported from a different checkpoint than {Path(weights).name}")
        if meta.get("onnx_sha256") != sha256(model):
            raise OnnxUnavailable(f"{model.name} does not match its recorded SHA-256 (altered or corrupt)")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001 - any load failure means "use PyTorch"
            raise OnnxUnavailable(f"onnxruntime could not load {model.name}: {e}") from e
        self.path = model
        self.sha256 = meta["onnx_sha256"]

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out, aux = self.session.run(None, {"x": np.ascontiguousarray(x, dtype=np.float32)})
        return out, aux
