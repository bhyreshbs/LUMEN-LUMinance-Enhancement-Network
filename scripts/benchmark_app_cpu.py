"""CPU latency and peak-RAM benchmark of the app's enhancement modes (the code path the web UI uses).

    python scripts/benchmark_app_cpu.py                       # 768 and 1024 px, all modes, 10 timed runs
    python scripts/benchmark_app_cpu.py --runs 20 --threads 4  # e.g. to emulate a 4-vCPU host

Every (mode, size) case runs in a fresh subprocess so its peak RSS is not polluted by other cases.
Per case: build the service (loads checkpoint + calibrated config), `--warmup` untimed runs, `--runs`
timed runs. Reports median / p95 latency of the service call (validation + resize + inference +
refinement + uint8 conversion; excludes HTTP and browser), RSS after loading, and peak RSS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("standard", "quality_guided", "classical", "compare_all")


def _peak_rss_mb() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)  # Windows
    if peak is None:  # Linux / macOS
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = ru * (1 if sys.platform == "darwin" else 1024)
    return peak / 2**20


def worker(mode: str, side: int, warmup: int, runs: int, threads: int) -> dict:
    os.environ["LUMEN_DEVICE"] = "cpu"
    import numpy as np
    import psutil
    import torch

    if threads:
        torch.set_num_threads(threads)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "app"))
    import cv2

    from service import EnhancementService

    from lumen import LumenPipeline
    from lumen.utils import read_image

    src = read_image(ROOT / "app" / "examples" / "coffee_dark.jpg")  # 600x400 synthetic low-light
    h = round(side * src.shape[0] / src.shape[1])
    img = (np.clip(cv2.resize(src, (side, h), interpolation=cv2.INTER_CUBIC), 0, 1) * 255).astype(np.uint8)

    t0 = time.perf_counter()
    svc = EnhancementService(LumenPipeline(max_side=side, device="cpu", deep_model="retinexformer"))
    load_s = time.perf_counter() - t0
    rss_loaded = psutil.Process().memory_info().rss / 2**20
    fn = (lambda: svc.compare_all(img)) if mode == "compare_all" else (lambda: svc.enhance(img, mode))
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return {"mode": mode, "size": f"{side}x{h}", "runs": runs, "threads": torch.get_num_threads(),
            "median_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1),
            "min_ms": round(min(ts), 1), "max_ms": round(max(ts), 1), "load_s": round(load_s, 2),
            "rss_after_load_mb": round(rss_loaded, 1), "peak_rss_mb": round(_peak_rss_mb(), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", type=int, default=[768, 1024])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = PyTorch default)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "app_cpu_benchmark.json")
    ap.add_argument("--worker", nargs=2, metavar=("MODE", "SIDE"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        print("RESULT " + json.dumps(worker(args.worker[0], int(args.worker[1]), args.warmup, args.runs,
                                            args.threads)), flush=True)
        return

    import platform

    import psutil

    rows = []
    print(f"CPU: {platform.processor() or platform.machine()}  logical cores: {psutil.cpu_count()}  "
          f"RAM: {psutil.virtual_memory().total / 2**30:.1f} GB  python {platform.python_version()}")
    for side in args.sizes:
        for mode in args.modes:
            cmd = [sys.executable, "-W", "ignore", __file__, "--worker", mode, str(side), "--warmup",
                   str(args.warmup), "--runs", str(args.runs), "--threads", str(args.threads)]
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
            if line is None:
                print(f"{mode} @ {side}: FAILED\n{out.stderr[-2000:]}")
                continue
            r = json.loads(line[7:])
            rows.append(r)
            print(f"{r['size']:>9s} {mode:15s} median {r['median_ms']:8.1f} ms  p95 {r['p95_ms']:8.1f} ms  "
                  f"RSS loaded {r['rss_after_load_mb']:7.1f} MB  peak {r['peak_rss_mb']:7.1f} MB  "
                  f"(threads {r['threads']}, load {r['load_s']} s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"platform": platform.platform(), "processor": platform.processor(),
                                    "logical_cores": psutil.cpu_count(), "rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
