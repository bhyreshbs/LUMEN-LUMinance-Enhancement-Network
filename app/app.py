"""LUMEN web app.  Run:  python app/app.py   ->  http://localhost:7860

Serves the restoration workspace (app/web/) and its JSON API (app/server.py) with Uvicorn. Restoration logic
lives in app/service.py; this file only configures the runtime and starts the server.

Environment: LUMEN_MAX_SIDE (default 1024), LUMEN_MAX_PIXELS (40e6), LUMEN_MAX_FILE_MB (20), LUMEN_DEVICE,
LUMEN_TORCH_THREADS, LUMEN_CUDNN, LUMEN_HOST (0.0.0.0), PORT (7860).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from server import create_app  # noqa: E402
from service import EnhancementService  # noqa: E402

log = logging.getLogger("lumen.app")


def cpu_quota() -> int | None:
    """CPUs granted by the container's cgroup quota (v2 `cpu.max`, v1 `cfs_quota_us`), or None if unlimited."""
    import math

    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
        return None
    except (OSError, ValueError):
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        return max(1, math.ceil(q / per)) if q > 0 else None
    except (OSError, ValueError):
        return None


def runtime_threads() -> int:
    """Threads for PyTorch/OpenCV: $LUMEN_TORCH_THREADS, else min(PyTorch default, CPU affinity, cgroup quota).
    PyTorch sizes its pool from the host's cores, so in a CPU-limited container it oversubscribes: measured in
    the LUMEN image with --cpus 4, Retinexformer 600x400 took 8.4 s with PyTorch's 8 threads vs 2.5 s with 4."""
    import torch

    env = os.environ.get("LUMEN_TORCH_THREADS")
    if env:
        return max(1, int(env))
    n = torch.get_num_threads()
    if hasattr(os, "sched_getaffinity"):
        n = min(n, len(os.sched_getaffinity(0)))
    q = cpu_quota()
    return max(1, min(n, q) if q else n)


def configure_runtime() -> None:
    """On CUDA, cuDNN builds a new execution plan for every new input size (~2.5 s on an RTX 3050 per new
    size, measured). Uploads almost always have new sizes, so the app disables cuDNN by default:
    new sizes then take ~0.1-0.2 s. Outputs differ by < 0.001 (eval15 PSNR 25.1539 vs 25.1517).
    Set LUMEN_CUDNN=1 to keep cuDNN (faster for repeated identical sizes). No effect on CPU."""
    import torch

    if torch.cuda.is_available() and os.environ.get("LUMEN_CUDNN", "0") != "1":
        torch.backends.cudnn.enabled = False
        log.info("cuDNN disabled for variable-size inputs (set LUMEN_CUDNN=1 to enable)")
    import cv2

    n = runtime_threads()
    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    log.info("CPU threads: %d (cgroup quota %s, LUMEN_TORCH_THREADS=%s)", n, cpu_quota(),
             os.environ.get("LUMEN_TORCH_THREADS", "unset"))


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    configure_runtime()
    service = EnhancementService(max_side=int(os.environ.get("LUMEN_MAX_SIDE", "1024")))
    service.warmup()
    log.info("checkpoint check: %s", service.checkpoint_check)
    for m, s in service.status.items():
        log.info("mode %-15s %s %s", m, "available" if s.available else "UNAVAILABLE", s.reason)
    host, port = os.environ.get("LUMEN_HOST", "0.0.0.0"), int(os.environ.get("PORT", 7860))
    log.info("LUMEN ready on http://%s:%d", "localhost" if host == "0.0.0.0" else host, port)
    uvicorn.run(create_app(service), host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
