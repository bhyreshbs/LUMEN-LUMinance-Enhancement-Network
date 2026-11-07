"""Latency / memory benchmark for the deep models (and optionally the full pipeline modes).

    python scripts/benchmark_latency.py                    # deep models, CPU + GPU, 600x400 and 1024 px
    python scripts/benchmark_latency.py --pipeline         # also classical / hybrid end-to-end
    python scripts/benchmark_latency.py --max-tile         # largest single (untiled) input that fits on the GPU

Timing excludes model loading: each case runs `--warmup` times before `--runs` timed runs
(CUDA is synchronised). Reports median and p95 in ms, plus peak GPU memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lumen import LumenPipeline  # noqa: E402
from lumen.deep import DeepEnhancer, available_models  # noqa: E402
from lumen.utils import read_image  # noqa: E402

SIZES = {"600x400 (LOL)": (600, 400), "1024x683": (1024, 683), "1920x1280": (1920, 1280)}


def sample_image(w, h) -> np.ndarray:
    src = sorted((ROOT / "data/LOL/eval15/low").glob("*.png"))
    img = read_image(src[0]) if src else np.random.default_rng(0).random((400, 600, 3), dtype=np.float32) * 0.2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def timed(fn, warmup, runs, cuda: bool):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    return float(np.median(ts)), float(np.percentile(ts, 95)), peak


@torch.inference_mode()
def max_untiled(model: str, device: str) -> int:
    """Largest long side (3:2 aspect) that runs in one piece on the GPU."""
    best = 0
    for side in (1024, 1280, 1536, 1792, 2048, 2560, 3072):
        enh = DeepEnhancer(model, device=device, tile=100_000)
        try:
            enh._run_tiled(sample_image(side, side * 2 // 3), 100_000)  # bypass the OOM fallback on purpose
            best = side
        except torch.OutOfMemoryError:
            break
        finally:
            del enh
            torch.cuda.empty_cache()
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=available_models(), choices=available_models())
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    ap.add_argument("--sizes", nargs="+", default=["600x400 (LOL)", "1024x683"], choices=list(SIZES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--pipeline", action="store_true", help="also time classical/hybrid end-to-end")
    ap.add_argument("--max-tile", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency.json")
    args = ap.parse_args()

    devices = [d for d in args.devices if d != "cuda" or torch.cuda.is_available()]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"GPU: {gpu}   torch {torch.__version__}   CPU threads: {torch.get_num_threads()}\n")
    rows = []
    for dev in devices:
        for m in args.models:
            enh = DeepEnhancer(m, device=dev)
            for label in args.sizes:
                img = sample_image(*SIZES[label])
                med, p95, peak = timed(lambda: enh(img), args.warmup, args.runs, dev == "cuda")
                rows.append({"what": f"deep: {m}", "device": dev, "fp16": enh.fp16, "size": label,
                             "median_ms": round(med, 1), "p95_ms": round(p95, 1),
                             "peak_gpu_mb": None if peak is None else round(peak, 1), "params": enh.n_params})
                print(f"{dev:4s} {m:14s} {label:14s} median {med:8.1f} ms  p95 {p95:8.1f} ms"
                      + (f"  peak GPU {peak:7.1f} MB" if peak is not None else ""))
            del enh
            if dev == "cuda":
                torch.cuda.empty_cache()
        if args.pipeline:
            pipe = LumenPipeline(device=dev)
            for mode in ("classical", "hybrid"):
                for label in args.sizes:
                    img = sample_image(*SIZES[label])
                    med, p95, _ = timed(lambda: pipe.run(img, mode), 1, max(3, args.runs // 2), dev == "cuda")
                    rows.append({"what": mode, "device": dev, "size": label, "median_ms": round(med, 1),
                                 "p95_ms": round(p95, 1)})
                    print(f"{dev:4s} {mode:14s} {label:14s} median {med:8.1f} ms  p95 {p95:8.1f} ms")

    result = {"gpu": gpu, "torch": torch.__version__, "rows": rows}
    if args.max_tile and torch.cuda.is_available():
        result["max_untiled_long_side_gpu"] = {m: max_untiled(m, "cuda") for m in args.models}
        print(f"\nlargest untiled input on GPU (long side, 3:2): {result['max_untiled_long_side_gpu']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
