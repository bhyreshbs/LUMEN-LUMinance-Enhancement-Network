"""Measure the production container: startup, model loading, per-mode latency and peak memory (cgroup).

    python scripts/benchmark_container.py --image lumen:phase7 --weights weights/retinexformer_LOL_v1.pth \
        --input bench_768x512.jpg --cpus 2 4 --memory 2560m

For each CPU limit and mode, a fresh container is started (so cgroup `memory.peak` is that mode's peak):
startup = `docker run` -> first HTTP 200; imports = container start -> runtime configured (log);
model load = runtime configured -> checkpoint verified (log; includes one warm-up inference).
Then `--warmup` + `--runs` requests through the public API; latency = server processing time reported by the
app (`timing_ms.total`) and the client round trip. Finally a concurrency test sends `--concurrent` requests at
once to one container and checks it was not OOM-killed. Needs Docker; HTTP via scripts/smoke_test_app.post_file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "MSYS_NO_PATHCONV": "1"}


def sh(*cmd, check=True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr}")
    return r.stdout.strip()


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]


def start(image, name, port, cpus, memory, weights) -> dict:
    sh("docker", "rm", "-f", name, check=False)
    t0 = time.perf_counter()
    sh("docker", "run", "-d", "--name", name, "-p", f"{port}:7860", "--cpus", str(cpus), "--memory", memory,
       "--mount", f"type=bind,source={weights},target=/models/retinexformer_LOL_v1.pth,readonly", image)
    url = f"http://127.0.0.1:{port}/"
    for _ in range(1200):
        try:
            urllib.request.urlopen(url, timeout=2)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    else:
        raise RuntimeError(f"{name} did not start:\n{sh('docker', 'logs', name, check=False)}")
    startup = time.perf_counter() - t0
    started = datetime.fromisoformat(sh("docker", "inspect", name, "--format", "{{.State.StartedAt}}")[:26])
    logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True, env=ENV)
    logs = logs.stdout + logs.stderr  # the app logs to stderr

    def ts(pattern):
        m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) INFO lumen\.app: " + pattern, logs)
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None

    t_rt, t_ck = ts("CPU threads"), ts("checkpoint check")
    threads = re.search(r"CPU threads: (\d+)", logs)
    return {"url": url, "startup_s": round(startup, 2),
            "imports_s": round((t_rt - started).total_seconds(), 2) if t_rt else None,
            "model_load_s": round((t_ck - t_rt).total_seconds(), 2) if t_rt and t_ck else None,
            "threads": int(threads.group(1)) if threads else None}


def mem_peak_mb(name) -> float:
    return int(sh("docker", "exec", name, "cat", "/sys/fs/cgroup/memory.peak")) / 2**20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="lumen:phase7")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "retinexformer_LOL_v1.pth")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--cpus", nargs="+", type=float, default=[2, 4])
    ap.add_argument("--memory", default="2560m")
    ap.add_argument("--modes", nargs="+", default=["standard", "quality_guided", "classical", "compare_all"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--concurrent", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "container_benchmark.json")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from smoke_test_app import post_file

    weights = str(args.weights.resolve())
    rows = []
    for cpus in args.cpus:
        for mode in args.modes:
            name = f"lumen-bench-{mode}-{cpus:g}"
            info = start(args.image, name, 7880, cpus, args.memory, weights)
            compare = mode == "compare_all"
            endpoint = info["url"] + ("api/compare" if compare else "api/restore")
            fields = {} if compare else {"style": mode}
            server, client = [], []
            for i in range(args.warmup + args.runs):
                t = time.perf_counter()
                code, out = post_file(endpoint, args.input, fields, timeout=300)
                rt = (time.perf_counter() - t) * 1000
                if code != 200:
                    raise RuntimeError(f"{mode}: HTTP {code} {out.get('error')}")
                if i >= args.warmup:
                    if compare:  # work actually done: one Retinexformer run + QGH refinement + classical
                        res = {r["style"]: r["details"]["timing_ms"] for r in out["results"]}
                        server.append(res["standard"]["total"] + res.get("quality_guided", {}).get("refine", 0)
                                      + res["classical"]["total"])
                    else:
                        server.append(out["details"]["timing_ms"]["total"])
                    client.append(rt)
            peak = mem_peak_mb(name)
            oom = sh("docker", "inspect", name, "--format", "{{.State.OOMKilled}}")
            sh("docker", "rm", "-f", name, check=False)
            row = {"cpus": cpus, "memory_limit": args.memory, "mode": mode, **{k: v for k, v in info.items() if k != "url"},
                   "server_median_ms": round(statistics.median(server), 1), "server_p95_ms": round(p95(server), 1),
                   "roundtrip_median_ms": round(statistics.median(client), 1), "roundtrip_p95_ms": round(p95(client), 1),
                   "peak_mem_mb": round(peak, 1), "oom_killed": oom == "true", "runs": args.runs}
            rows.append(row)
            print(f"cpus {cpus:g} {mode:15s} server median {row['server_median_ms']:7.0f} p95 {row['server_p95_ms']:7.0f} ms | "
                  f"round trip median {row['roundtrip_median_ms']:7.0f} p95 {row['roundtrip_p95_ms']:7.0f} ms | "
                  f"peak {row['peak_mem_mb']:6.0f} MB | startup {row['startup_s']} s (imports {row['imports_s']} s, "
                  f"model load {row['model_load_s']} s) threads {row['threads']} oom {row['oom_killed']}", flush=True)

    # concurrency: N simultaneous Standard requests against one container at the smallest CPU limit
    cpus = min(args.cpus)
    name = "lumen-bench-concurrency"
    info = start(args.image, name, 7881, cpus, args.memory, weights)
    errors, times = [], []

    def one():
        try:
            t = time.perf_counter()
            code, out = post_file(info["url"] + "api/restore", args.input, {"style": "standard"}, timeout=300)
            if code != 200:
                raise RuntimeError(f"HTTP {code} {out.get('error')}")
            times.append(time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e)[:200])

    threads = [threading.Thread(target=one) for _ in range(args.concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    conc = {"cpus": cpus, "memory_limit": args.memory, "concurrent_requests": args.concurrent, "errors": errors,
            "completion_s": sorted(round(t, 1) for t in times), "peak_mem_mb": round(mem_peak_mb(name), 1),
            "oom_killed": sh("docker", "inspect", name, "--format", "{{.State.OOMKilled}}") == "true"}
    sh("docker", "rm", "-f", name, check=False)
    print(f"concurrency: {args.concurrent} simultaneous Standard requests @ {cpus:g} CPUs / {args.memory}: "
          f"completed after {conc['completion_s']} s, errors {len(errors)}, peak {conc['peak_mem_mb']:.0f} MB, "
          f"OOM {conc['oom_killed']}")
    image_size = sh("docker", "image", "inspect", args.image, "--format", "{{.Size}}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"image": args.image, "image_bytes": int(image_size),
                                    "input": args.input.name, "rows": rows, "concurrency": conc}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
