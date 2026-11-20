"""Export the official Retinexformer checkpoint to ONNX for the optional CPU engine (LUMEN_ENGINE=onnx).

    python scripts/export_onnx.py                       # weights/retinexformer_LOL_v1.pth -> .onnx + .onnx.json beside it
    python scripts/export_onnx.py --weights path/to/retinexformer_LOL_v1.pth

The checkpoint's SHA-256 must be the official one. The export is checked against the PyTorch network (restored image and
light-up map) on random inputs of several sizes, and refused if they differ by more than --tol. The sidecar records the
checkpoint hash and the .onnx file's own hash, which the app verifies before using the file.

The result is derived from the checkpoint, whose redistribution rights are uncertain, so it is never committed and never
baked into an image: put it next to the checkpoint (for Cloud Run, upload both to the weights bucket).

Needs the export tools (requirements-dev.txt: onnx, onnxscript) and onnxruntime (requirements-onnx.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lumen.deep import DeepEnhancer  # noqa: E402
from lumen.deep.checkpoints import sha256, verify  # noqa: E402
from lumen.deep.infer import _forward_with_aux  # noqa: E402
from lumen.deep.onnx_engine import SIDECAR_FORMAT, onnx_paths  # noqa: E402

OPSET = 17


class Export(torch.nn.Module):
    """The same computation as lumen.deep.infer._forward_with_aux: restored image and the first light-up map."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        out, aux = _forward_with_aux(self.m, "retinexformer", x)
        return out, aux


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "retinexformer_LOL_v1.pth")
    ap.add_argument("--tol", type=float, default=1e-4, help="largest allowed absolute difference (images are in [0, 1])")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    ok, msg = verify(args.weights)
    if not ok:
        raise SystemExit(f"refusing to export: {msg}")
    enh = DeepEnhancer("retinexformer", args.weights, "cpu", engine="torch")
    model, side = onnx_paths(args.weights)
    net = Export(enh.model.eval()).eval()
    x = torch.rand(1, 3, 128, 192)
    torch.onnx.export(net, (x,), str(model), input_names=["x"], output_names=["out", "aux"], opset_version=OPSET, dynamo=False,
                      dynamic_axes={"x": {2: "h", 3: "w"}, "out": {2: "h", 3: "w"}, "aux": {2: "h", 3: "w"}})

    import onnxruntime as ort

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    worst = 0.0
    for h, w in ((64, 64), (200, 304), (400, 600 // 8 * 8), (512, 768)):
        h, w = h // 8 * 8, w // 8 * 8
        t = torch.rand(1, 3, h, w)
        with torch.inference_mode():
            ref_out, ref_aux = net(t)
        out, aux = sess.run(None, {"x": t.numpy()})
        d = max(float(np.abs(out - ref_out.numpy()).max()), float(np.abs(aux - ref_aux.numpy()).max()))
        worst = max(worst, d)
        print(f"  {w}x{h}: max |difference| {d:.2e}")
    if worst > args.tol:
        model.unlink(missing_ok=True)
        raise SystemExit(f"export differs from PyTorch by {worst:.2e} (> {args.tol:.0e}); not written")

    side.write_text(json.dumps({
        "format": SIDECAR_FORMAT, "model": "retinexformer", "opset": OPSET, "torch": torch.__version__, "onnxruntime": ort.__version__,
        "source_checkpoint": args.weights.name, "source_sha256": sha256(args.weights), "onnx_sha256": sha256(model),
        "max_abs_difference_vs_torch": worst,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {model} ({model.stat().st_size // 1024} KB) and {side.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
