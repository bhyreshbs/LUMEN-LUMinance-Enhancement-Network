"""Tune the classical branch's adaptive constants (AdaptiveConfig) with Optuna.

    python scripts/tune_classical.py --trials 150

Data policy: only data/LOL/our485 is used, split 90/10 (fixed seed) into tune / validation.
Optuna maximises  mean(PSNR) + 10 * mean(SSIM)  on the tune split. The top-K trials are then
re-scored on the held-out validation split and the best one on validation is saved to
configs/classical.yaml, which select_params() loads by default. eval15 is never touched.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import date
import multiprocessing as mp
from pathlib import Path

# One BLAS/OpenMP thread per process - the workers already use every core (must precede numpy import)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from lumen.classical import AdaptiveConfig, ClassicalEnhancer, analyze_image, save_config  # noqa: E402
from lumen.metrics import psnr, ssim  # noqa: E402
from lumen.utils import read_image  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# ----------------------------------------------------------------------------- worker side
def _worker(conn, pairs: list[tuple[str, str]]):
    """Persistent worker: loads only its own shard once, then scores every config it is sent."""
    import warnings

    warnings.filterwarnings("ignore")
    import cv2

    cv2.setNumThreads(1)  # one process per core; avoid oversubscription
    data = []
    for lo, hi in pairs:
        x, y = read_image(lo), read_image(hi)
        data.append((x, y, analyze_image(x)))  # stats do not depend on the config -> cache once
    conn.send("ready")
    while True:
        cfg_dict = conn.recv()
        if cfg_dict is None:
            break
        try:
            enh = ClassicalEnhancer(config=AdaptiveConfig.from_dict(cfg_dict))
            scores = []
            for x, y, st in data:
                o = enh(x, st).image
                scores.append((psnr(o, y), ssim(o, y)))
            conn.send(scores)
        except Exception as e:  # noqa: BLE001 - report to parent instead of hanging it
            conn.send(e)


class Scorer:
    """Image pairs are sharded across persistent processes; scoring a config fans out to all shards."""

    def __init__(self, pairs, workers: int):
        ctx = mp.get_context("spawn")
        workers = max(1, min(workers, len(pairs)))
        self.conns, self.procs = [], []
        for k in range(workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, pairs[k::workers]), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
        for c in self.conns:
            if c.recv() != "ready":
                raise RuntimeError("worker failed to start")

    def __call__(self, cfg: AdaptiveConfig) -> dict:
        d = cfg.as_dict()
        for c in self.conns:
            c.send(d)
        res = []
        for c in self.conns:
            part = c.recv()
            if isinstance(part, Exception):
                raise part
            res += part
        p, s = float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))
        return {"psnr": p, "ssim": s, "objective": p + 10.0 * s}

    def close(self):
        for c in self.conns:
            c.send(None)
        for proc in self.procs:
            proc.join(timeout=10)


# ----------------------------------------------------------------------------- search space
def suggest(trial) -> AdaptiveConfig:
    return AdaptiveConfig(
        target_mean=trial.suggest_float("target_mean", 0.30, 0.65),
        gamma_min=trial.suggest_float("gamma_min", 0.10, 0.50),
        denoise_mult=trial.suggest_float("denoise_mult", 0.0, 6.0),
        denoise_max=trial.suggest_float("denoise_max", 0.05, 0.40),
        clahe_gain=trial.suggest_float("clahe_gain", 0.0, 4.0),
        clahe_ref_contrast=trial.suggest_float("clahe_ref_contrast", 0.08, 0.30),
        clahe_max=trial.suggest_float("clahe_max", 1.0, 4.0),
        clahe_noise_thresh=trial.suggest_float("clahe_noise_thresh", 0.01, 0.15),
        clahe_noise_cap=trial.suggest_float("clahe_noise_cap", 1.0, 2.5),
        clahe_tile=trial.suggest_categorical("clahe_tile", [4, 8, 16]),
        radius_base=trial.suggest_float("radius_base", 8.0, 30.0),
        radius_slope=trial.suggest_float("radius_slope", 0.0, 15.0),
        wb_mult=trial.suggest_float("wb_mult", 0.0, 4.0),
        sat_gain=trial.suggest_float("sat_gain", 0.0, 1.0),
        sat_max=trial.suggest_float("sat_max", 1.0, 1.6),
    )


def split_pairs(low: Path, high: Path, val_frac: float, seed: int):
    lows = sorted(p for p in low.iterdir() if p.suffix.lower() in EXTS)
    pairs = [(str(p), str(high / p.name)) for p in lows if (high / p.name).is_file()]
    if len(pairs) < 20:
        raise SystemExit(f"Only {len(pairs)} pairs found in {low}")
    random.Random(seed).shuffle(pairs)
    n_val = max(1, round(len(pairs) * val_frac))
    return pairs[n_val:], pairs[:n_val]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", type=Path, default=ROOT / "data/LOL/our485/low")
    ap.add_argument("--high", type=Path, default=ROOT / "data/LOL/our485/high")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5, help="best tune-split trials re-checked on validation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=Path, default=ROOT / "configs/classical.yaml")
    args = ap.parse_args()

    if "eval15" in str(args.low) or "eval15" in str(args.high):
        raise SystemExit("Refusing to tune on eval15 - it is the held-out test set.")

    import optuna

    tune_pairs, val_pairs = split_pairs(args.low, args.high, args.val_frac, args.seed)
    print(f"tune: {len(tune_pairs)}  val: {len(val_pairs)}  workers: {args.workers}")

    tune = Scorer(tune_pairs, args.workers)
    t0 = time.perf_counter()
    default = AdaptiveConfig()
    base_tune = tune(default)
    print(f"defaults on tune split: PSNR {base_tune['psnr']:.3f}  SSIM {base_tune['ssim']:.4f}  "
          f"obj {base_tune['objective']:.3f}  ({time.perf_counter() - t0:.1f}s per trial)")

    def objective(trial):
        r = tune(suggest(trial))
        trial.set_user_attr("psnr", r["psnr"])
        trial.set_user_attr("ssim", r["ssim"])
        return r["objective"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed),
                                study_name="lumen-classical")
    study.enqueue_trial({k: v for k, v in default.as_dict().items() if k != "clahe_min"})  # start from defaults

    def log(study, trial):
        best = study.best_trial
        print(f"trial {trial.number + 1:3d}/{args.trials}  obj {trial.value:.3f}  "
              f"best {best.value:.3f} (PSNR {best.user_attrs['psnr']:.2f}, SSIM {best.user_attrs['ssim']:.4f})",
              flush=True)

    study.optimize(objective, n_trials=args.trials, callbacks=[log])
    tune.close()

    # Model selection on the held-out validation split (guards against overfitting the tune split)
    done = [t for t in study.trials if t.value is not None]
    top = sorted(done, key=lambda t: t.value, reverse=True)[: args.top_k]
    val = Scorer(val_pairs, args.workers)
    base_val = val(default)
    print(f"\ndefaults on val split: PSNR {base_val['psnr']:.3f}  SSIM {base_val['ssim']:.4f}  obj {base_val['objective']:.3f}")
    scored = []
    for t in top:
        cfg = AdaptiveConfig(**t.params)
        r = val(cfg)
        scored.append((r, cfg, t))
        print(f"  trial {t.number:3d}: tune obj {t.value:.3f} -> val PSNR {r['psnr']:.3f}  SSIM {r['ssim']:.4f}  obj {r['objective']:.3f}")
    val.close()

    best_r, best_cfg, best_t = max(scored, key=lambda x: x[0]["objective"])
    if best_r["objective"] <= base_val["objective"]:
        print("\nNo trial beats the defaults on validation - keeping defaults.")
        best_r, best_cfg = base_val, default

    meta = {
        "created": date.today().isoformat(),
        "tool": "scripts/tune_classical.py",
        "data": (args.low.resolve().relative_to(ROOT) if args.low.resolve().is_relative_to(ROOT) else args.low).as_posix(),
        "split": {"seed": args.seed, "val_frac": args.val_frac, "tune": len(tune_pairs), "val": len(val_pairs)},
        "trials": args.trials,
        "objective": "mean PSNR + 10 * mean SSIM",
        "selected_trial": None if best_cfg is default else best_t.number,
        "defaults": {"tune": _round(base_tune), "val": _round(base_val)},
        "tuned": {"tune": _round({"psnr": best_t.user_attrs["psnr"], "ssim": best_t.user_attrs["ssim"],
                                  "objective": best_t.value}) if best_cfg is not default else None,
                  "val": _round(best_r)},
    }
    save_config(best_cfg, args.out, meta)
    print(f"\nSaved {args.out}\nval: PSNR {base_val['psnr']:.3f} -> {best_r['psnr']:.3f}, "
          f"SSIM {base_val['ssim']:.4f} -> {best_r['ssim']:.4f}")


def _round(d: dict) -> dict:
    return {k: round(float(v), 4) for k, v in d.items()}


if __name__ == "__main__":
    main()
