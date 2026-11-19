"""Post-freeze statistical checks of the LUMEN results (no model, configuration or threshold is changed).

    python scripts/hardening_study.py all                 # everything below, then update eval_P1_T66.md
    python scripts/hardening_study.py bootstrap           # eval15: paired bootstrap CIs on the frozen per-image results
    python scripts/hardening_study.py lolv2               # LOL-v2-real Test with the matching LOL-v2-real checkpoint
    python scripts/hardening_study.py noref               # no-reference scores on unseen photos (+ data/noref/<set>/)
    python scripts/hardening_study.py dev                 # the 269-image dev split (LOL-v1 checkpoint)
    python scripts/hardening_study.py strata              # eval15, LOL-v2-real Test and dev by input noise and brightness (from the saved CSVs)
    python scripts/hardening_study.py report              # rebuild results/phase_f_hardening/summary.md and the eval section

What it does and why
  * Paired bootstrap (10,000 resamples) of per-image differences between methods, with the unit of resampling
    being a *scene*: images whose thumbnails are near-duplicates are one scene, so repeated shots cannot make a
    result look more certain than it is. The naive per-image interval is reported next to it.
  * An exact (or Monte-Carlo, for many scenes) paired sign-flip test, and win / tie / loss counts.
  * Every run re-measures the frozen eval15 rows first and checks them against results/phase5_eval15/per_image.csv.
  * Intervals are not corrected for the number of comparisons; with 15 images they show how little can be claimed.

Results: results/phase_f_hardening/ (summary.json, per-image CSVs, summary.md). The eval report section between
the phase-f markers in eval_P1_T66.md is generated from summary.json, so no number is typed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

OUT = ROOT / "results" / "phase_f_hardening"
FROZEN_CSV = ROOT / "results" / "phase5_eval15" / "per_image.csv"
EVAL_MD = ROOT / "eval_P1_T66.md"
N_RESAMPLES = 10_000
SEED = 20261119
START, END = "<!-- phase-f:start -->", "<!-- phase-f:end -->"

METRICS = ("psnr", "ssim", "lpips")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "niqe": False, "noise": False, "overexposed": False}
TIE = {"psnr": 0.005, "ssim": 5e-5, "lpips": 5e-5, "niqe": 0.005, "noise": 5e-5, "overexposed": 0.005}
DIGITS = {"psnr": 2, "ssim": 4, "lpips": 4, "niqe": 2, "noise": 4, "overexposed": 2}
LABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS", "niqe": "NIQE", "noise": "Noise σ", "overexposed": "Over-exposed %"}

# (key, display name, pipeline mode, highlight colour fix)
METHODS = [
    ("classical", "Classical", "classical", False),
    ("retinexformer", "Retinexformer", "deep", False),
    ("qgh", "Quality Guided (frozen)", "quality_guided_hybrid", False),
    ("natural", "Natural (app)", "deep", True),
    ("highlight_safe", "Highlight-safe (app)", "quality_guided_hybrid", True),
]
NAME = {k: n for k, n, _, _ in METHODS}
# (A, B) -> the difference A - B is reported
CONTRASTS = [("retinexformer", "qgh"), ("retinexformer", "classical"), ("qgh", "classical"),
             ("natural", "retinexformer"), ("highlight_safe", "qgh")]
FROZEN_NAMES = {"classical": "classical", "retinexformer": "deep: retinexformer", "qgh": "quality_guided_hybrid: retinexformer"}


# ------------------------------------------------------------------------------------------------ statistics
def scene_groups(fingerprints: np.ndarray, tol: float = 0.03) -> np.ndarray:
    """Connected components of 'thumbnails closer than tol' (mean absolute difference): one id per scene."""
    n = len(fingerprints)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.abs(fingerprints[i + 1:] - fingerprints[i]).mean(-1)
        for j in np.nonzero(d < tol)[0]:
            parent[find(i)] = find(i + 1 + int(j))
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def bootstrap_ci(diff: np.ndarray, groups: np.ndarray | None = None, n: int = N_RESAMPLES, seed: int = SEED):
    """95 % percentile interval of the mean paired difference, resampling whole groups (scenes) with replacement."""
    diff = np.asarray(diff, float)
    groups = np.arange(len(diff)) if groups is None else np.asarray(groups)
    inv = np.unique(groups, return_inverse=True)[1]
    k = inv.max() + 1
    sums = np.bincount(inv, weights=diff, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    idx = np.random.default_rng(seed).integers(0, k, size=(n, k))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_flip_p(diff: np.ndarray, groups: np.ndarray | None = None, seed: int = SEED, mc: int = 200_000) -> float:
    """Two-sided paired sign-flip (randomisation) test on per-scene mean differences: exact up to 20 scenes."""
    diff = np.asarray(diff, float)
    groups = np.arange(len(diff)) if groups is None else np.asarray(groups)
    inv = np.unique(groups, return_inverse=True)[1]
    units = np.bincount(inv, weights=diff) / np.bincount(inv)
    k, obs = len(units), abs(units.sum())
    if obs < 1e-12:
        return 1.0
    if k <= 20:
        signs = (((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) * 2 - 1).astype(float)
        return float((np.abs(signs @ units) >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(mc, k)) * 2 - 1
    return float(((np.abs(signs @ units) >= obs - 1e-12).sum() + 1) / (mc + 1))


def compare(a: np.ndarray, b: np.ndarray, metric: str, groups: np.ndarray | None) -> dict:
    """Paired difference a - b of one metric over the same images."""
    d = np.asarray(a, float) - np.asarray(b, float)
    tol = TIE[metric]
    better = (d > tol) if HIGHER_IS_BETTER[metric] else (d < -tol)
    worse = (d < -tol) if HIGHER_IS_BETTER[metric] else (d > tol)
    lo, hi = bootstrap_ci(d, groups)
    nlo, nhi = bootstrap_ci(d, None)
    return {"n": int(len(d)), "scenes": int(len(np.unique(groups))) if groups is not None else int(len(d)),
            "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "naive_ci_lo": nlo, "naive_ci_hi": nhi,
            "excludes_zero": bool(lo > 0 or hi < 0), "p": sign_flip_p(d, groups),
            "a_better": int(better.sum()), "ties": int(len(d) - better.sum() - worse.sum()), "a_worse": int(worse.sum())}


# ------------------------------------------------------------------------------------------------ data and runs
def fingerprints_of(paths: list[tuple[Path, Path]]) -> np.ndarray:
    from train_retinexformer import fingerprint

    return np.stack([np.concatenate([fingerprint(lo), fingerprint(hi)]) for lo, hi in paths])


def load_pairs_dir(low: Path, high: Path):
    from evaluate import pair

    return pair(low, high)


def load_pairs_list(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            lo, hi = line.split("\t")
            out.append((ROOT / lo, ROOT / hi))
    return out


def make_pipe(weights: Path):
    from lumen import LumenPipeline
    from lumen.deep.checkpoints import verify

    ok, msg = verify(weights)
    if not ok:
        raise SystemExit(f"checkpoint not usable: {msg}")
    os.environ.pop("LUMEN_RETINEXFORMER_WEIGHTS", None)
    return LumenPipeline(max_side=1280, deep_model="retinexformer", weights=str(weights)), weights.name


def run_methods(pipe, ckpt_name: str, pairs, with_gt: bool, methods=METHODS, niqe_on: bool = False, tag: str = ""):
    """One row per (image, method): the same metric functions and procedure as scripts/evaluate.py."""
    import cv2

    from lumen.metrics import colour_shift, edge_preservation, lpips_distance, niqe, noise_level, overexposure_ratio, psnr, ssim
    from lumen.utils import luminance, read_image

    first = read_image(pairs[0][0])
    for _, _, mode, hue in methods:                       # warm-up: model load, CUDA init
        pipe.run(first, mode, deep_model="retinexformer" if mode != "classical" else None, highlight_hue=hue)
    rows = []
    t0 = time.time()
    for i, (lo, hi) in enumerate(pairs):
        x = read_image(lo)
        gt = read_image(hi) if with_gt else None
        for key, name, mode, hue in methods:
            r = pipe.run(x, mode, deep_model="retinexformer" if mode != "classical" else None, highlight_hue=hue)
            if mode != "classical" and r.info.get("weights") != ckpt_name:
                raise RuntimeError(f"{name}: expected checkpoint {ckpt_name}, ran {r.info.get('weights')}")
            out = r.output
            row = {"image": Path(lo).name, "method": key, "noise": noise_level(out),
                   "overexposed": 100 * overexposure_ratio(out), "in_mean_lum": float(luminance(r.input).mean()),
                   "in_noise": noise_level(r.input)}
            if with_gt:
                g = gt if gt.shape == out.shape else cv2.resize(gt, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA)
                row.update(psnr=psnr(out, g), ssim=ssim(out, g), lpips=lpips_distance(out, g),
                           edge=edge_preservation(out, g), colour_shift=colour_shift(out, g))
            if niqe_on:
                row["niqe"] = niqe(out)
            rows.append(row)
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(pairs):
            print(f"  {tag} {i + 1}/{len(pairs)} images ({time.time() - t0:.0f}s)", flush=True)
    return rows


def table(rows, metric, keys=None):
    """{method: array over images}, rows ordered like the image list."""
    out = {}
    for r in rows:
        out.setdefault(r["method"], {})[r["image"]] = r[metric]
    images = sorted(next(iter(out.values())))
    return images, {k: np.array([v[im] for im in images], float) for k, v in out.items()}


def write_rows(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def contrasts_for(rows, groups_by_image, metrics=METRICS, pairs=CONTRASTS, keys=None):
    out = []
    have = {r["method"] for r in rows}
    for metric in metrics:
        images, t = table(rows, metric)
        groups = np.array([groups_by_image[im] for im in images]) if groups_by_image else None
        for a, b in pairs:
            if a in have and b in have:
                out.append({"a": a, "b": b, "metric": metric, **compare(t[a], t[b], metric, groups)})
    return out


def means_for(rows, metrics=METRICS):
    out = {}
    for metric in metrics:
        _, t = table(rows, metric)
        for k, v in t.items():
            out.setdefault(k, {})[metric] = float(v.mean())
    return out


def load_summary():
    p = OUT / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_summary(s):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")


def frozen_check(rows):
    """Re-measured eval15 rows against the frozen per_image.csv (the headline run)."""
    frozen = {}
    for r in csv.DictReader(open(FROZEN_CSV, encoding="utf-8")):
        frozen[(r["experiment"], r["image"])] = r
    worst = {m: 0.0 for m in METRICS}
    for r in rows:
        if r["method"] not in FROZEN_NAMES:
            continue
        f = frozen[(FROZEN_NAMES[r["method"]], r["image"])]
        for m in METRICS:
            worst[m] = max(worst[m], abs(float(f[m]) - r[m]))
    return worst


def eval15_pairs():
    return load_pairs_dir(ROOT / "data/LOL/eval15/low", ROOT / "data/LOL/eval15/high")


# ------------------------------------------------------------------------------------------------ 1. bootstrap on eval15
def cmd_bootstrap(args):
    s = load_summary()
    print("== eval15: re-measure the three frozen methods plus the app styles, then bootstrap")
    pairs = eval15_pairs()
    pipe, ck = make_pipe(ROOT / "weights/retinexformer_LOL_v1.pth")
    rows = run_methods(pipe, ck, pairs, True, tag="eval15")
    fp = fingerprints_of(pairs)
    groups = scene_groups(fp)
    by_image = {Path(lo).name: int(g) for (lo, _), g in zip(pairs, groups)}
    for r in rows:
        r["scene"] = by_image[r["image"]]
    write_rows(rows, "per_image_eval15.csv")
    worst = frozen_check(rows)
    print("   max |difference| from the frozen per-image results:", {k: round(v, 4) for k, v in worst.items()})
    from train_retinexformer import DATA, fingerprint, pair_dir

    o485 = pair_dir(DATA / "LOL/our485/low", DATA / "LOL/our485/high")
    dist = np.abs(fingerprints_of(o485)[None] - fp[:, None]).mean(-1).min(1)
    s["eval15"] = {
        "n": len(pairs), "scenes": int(len(set(groups))), "checkpoint": ck,
        "means": means_for(rows), "frozen_check_max_abs_diff": worst,
        "near_duplicate_in_training": {"tol_0.01": int((dist < 0.01).sum()), "tol_0.03": int((dist < 0.03).sum())},
        "contrasts": contrasts_for(rows, by_image),
    }
    save_summary(s)


# ------------------------------------------------------------------------------------------------ 2. LOL-v2-real Test
def cmd_lolv2(args):
    s = load_summary()
    print("== LOL-v2-real Test with the LOL-v2-real checkpoint")
    from train_retinexformer import DATA, pair_dir

    test = pair_dir(DATA / "LOL-v2/Real_captured/Test/Low", DATA / "LOL-v2/Real_captured/Test/Normal")
    train = pair_dir(DATA / "LOL-v2/Real_captured/Train/Low", DATA / "LOL-v2/Real_captured/Train/Normal")
    o485 = pair_dir(DATA / "LOL/our485/low", DATA / "LOL/our485/high")
    if args.limit:
        test = test[:args.limit]
    ft = fingerprints_of(test)
    dist_train = np.abs(fingerprints_of(train)[None] - ft[:, None]).mean(-1).min(1)
    dist_o485 = np.abs(fingerprints_of(o485)[None] - ft[:, None]).mean(-1).min(1)
    groups = scene_groups(ft)
    by_image = {Path(lo).name: int(g) for (lo, _), g in zip(test, groups)}
    pipe, ck = make_pipe(ROOT / "weights/retinexformer_LOL_v2_real.pth")
    rows = run_methods(pipe, ck, test, True, tag="v2-real test")
    for r in rows:
        r["scene"] = by_image[r["image"]]
    write_rows(rows, "per_image_lolv2_real_test.csv")
    s["lolv2_real_test"] = {
        "n": len(test), "scenes": int(len(set(groups))), "checkpoint": ck,
        "leakage": {"near_duplicate_in_v2_real_train": {"tol_0.01": int((dist_train < 0.01).sum()), "tol_0.03": int((dist_train < 0.03).sum()),
                                                        "tol_0.05": int((dist_train < 0.05).sum())},
                    "near_duplicate_in_lol_v1_train_tol_0.01": int((dist_o485 < 0.01).sum())},
        "means": means_for(rows), "contrasts": contrasts_for(rows, by_image),
    }
    save_summary(s)


# ------------------------------------------------------------------------------------------------ 3. no-reference
def cmd_noref(args):
    s = load_summary()
    print("== no-reference scores on unseen photos (no ground truth)")
    from evaluate import list_images
    from lumen.metrics import niqe as niqe_fn

    sets = {"unseen phone photos": list_images(ROOT / "data/unseen")}
    extra = ROOT / "data/noref"
    if extra.is_dir():
        for d in sorted(p for p in extra.iterdir() if p.is_dir()):
            sets[d.name] = list_images(d)
    pipe, ck = make_pipe(ROOT / "weights/retinexformer_LOL_v1.pth")
    methods = [m for m in METHODS if m[0] in ("classical", "natural", "highlight_safe")]
    probe = None
    out = {"checkpoint": ck, "sets": {}, "not_provided": [n for n in ("DICM", "LIME") if n not in sets]}
    for name, files in sets.items():
        pairs = [(f, f) for f in files]
        rows = run_methods(pipe, ck, pairs, False, methods=methods, niqe_on=True, tag=name)
        write_rows(rows, f"per_image_noref_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.csv")
        probe = probe or rows[0].get("niqe")
        same_input = [r for r in rows if r["method"] == "classical"]       # the input photo is the same for every method
        entry = {"n": len(files), "means": {}, "contrasts": [],
                 "input": {"noise_min": float(min(r["in_noise"] for r in same_input)),
                           "noise_median": float(np.median([r["in_noise"] for r in same_input])),
                           "noise_max": float(max(r["in_noise"] for r in same_input)),
                           "brightness_median": float(np.median([r["in_mean_lum"] for r in same_input]))}}
        for metric in ("niqe", "noise", "overexposed"):
            if any(r.get(metric) is None for r in rows):
                continue
            _, t = table(rows, metric)
            for k, v in t.items():
                entry["means"].setdefault(k, {})[metric] = float(v.mean())
            for a, b in (("natural", "classical"), ("highlight_safe", "classical"), ("highlight_safe", "natural")):
                entry["contrasts"].append({"a": a, "b": b, "metric": metric, **compare(t[a], t[b], metric, None)})
        out["sets"][name] = entry
    out["niqe_available"] = probe is not None
    s["noref"] = out
    save_summary(s)


# ------------------------------------------------------------------------------------------------ 4. dev split and strata
def cmd_dev(args):
    s = load_summary()
    print("== dev split (269 LOL-v2-real Train pairs unseen by the LOL-v1 checkpoint), LOL-v1 checkpoint")
    dev = load_pairs_list(ROOT / "data/splits/dev_lolv2_real_unseen.txt")
    if args.limit:
        dev = dev[:args.limit]
    pipe, ck = make_pipe(ROOT / "weights/retinexformer_LOL_v1.pth")
    methods = [m for m in METHODS if m[0] in ("classical", "retinexformer", "qgh")]
    rows = run_methods(pipe, ck, dev, True, methods=methods, tag="dev")
    gd = scene_groups(fingerprints_of(dev))
    by_image = {Path(lo).name: int(g) for (lo, _), g in zip(dev, gd)}
    for r in rows:
        r["scene"] = by_image[r["image"]]
    write_rows(rows, "per_image_dev.csv")
    s["dev"] = {"n": len(dev), "scenes": int(len(set(gd))), "checkpoint": ck, "means": means_for(rows),
                "contrasts": contrasts_for(rows, by_image, pairs=[("retinexformer", "qgh"), ("retinexformer", "classical"), ("qgh", "classical")])}
    save_summary(s)


# The dev split's noise histogram has two populations and nothing between 0.01 and 0.04, so the split sits in that gap.
# It was chosen after looking at the data: an exploratory split, not a pre-registered one.
NOISE_SPLIT = 0.025


def read_rows(path: Path):
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.append({k: (v if k in ("image", "method") else (float(v) if v not in ("", None) else None)) for k, v in r.items()})
    return out


def strata_for(rows):
    """Per-image CSV rows -> results by input-noise band, and by brightness within the low-noise band."""
    images, _ = table(rows, "psnr")
    first = {}
    for r in rows:
        first.setdefault(r["image"], r)
    noise = np.array([first[i]["in_noise"] for i in images])
    lum = np.array([first[i]["in_mean_lum"] for i in images])
    scene = np.array([first[i].get("scene", k) if first[i].get("scene") is not None else k for k, i in enumerate(images)])
    tabs = {m: table(rows, m)[1] for m in METRICS}
    pairs = {"retinexformer_minus_classical": ("retinexformer", "classical"), "qgh_minus_retinexformer": ("qgh", "retinexformer")}

    def block(mask):
        e = {"n": int(mask.sum()), "scenes": int(len(np.unique(scene[mask]))),
             "input_noise_range": [float(noise[mask].min()), float(noise[mask].max())],
             "input_brightness_range": [float(lum[mask].min()), float(lum[mask].max())],
             "means": {k: {m: float(tabs[m][k][mask].mean()) for m in METRICS} for k in ("classical", "retinexformer", "qgh")},
             "contrasts": []}
        for name, (a, b) in pairs.items():
            for m in METRICS:
                e["contrasts"].append({"a": a, "b": b, "metric": m, **compare(tabs[m][a][mask], tabs[m][b][mask], m, scene[mask])})
        return e

    out = {"noise_split": NOISE_SPLIT, "bands": {}, "brightness_within_low_noise": [],
           "corr_input_noise_vs_retinexformer_ssim": float(np.corrcoef(noise, tabs["ssim"]["retinexformer"])[0, 1]) if len(noise) > 2 else None,
           "corr_input_noise_vs_brightness": float(np.corrcoef(noise, lum)[0, 1]) if len(noise) > 2 else None}
    for name, mask in (("low", noise < NOISE_SPLIT), ("high", noise >= NOISE_SPLIT)):
        if mask.any():
            out["bands"][name] = block(mask)
    low = np.nonzero(noise < NOISE_SPLIT)[0]
    if len(low) >= 6:
        parts = np.array_split(low[np.argsort(lum[low], kind="stable")], 3)
        for label, idx in zip(("darkest third", "middle third", "brightest third"), parts):
            mask = np.zeros(len(images), bool)
            mask[idx] = True
            e = block(mask)
            e["label"] = label
            out["brightness_within_low_noise"].append(e)
    return out


def cmd_strata(args):
    s = load_summary()
    print("== strata, derived from the saved per-image results")
    out = {}
    for key, fname in (("eval15", "per_image_eval15.csv"), ("lolv2_real_test", "per_image_lolv2_real_test.csv"), ("dev", "per_image_dev.csv")):
        p = OUT / fname
        if not p.exists():
            print(f"   {fname} missing: run the step that produces it first")
            continue
        rows = [r for r in read_rows(p) if r["method"] in ("classical", "retinexformer", "qgh")]
        out[key] = strata_for(rows)
    s["strata"] = out
    save_summary(s)


# ------------------------------------------------------------------------------------------------ report
def fmt(v, metric):
    """A signed difference: one more decimal for PSNR, because the refinement effects are hundredths of a dB."""
    return f"{v:+.{DIGITS[metric] + (1 if metric == 'psnr' else 0)}f}"


def pvalue(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def contrast_rows(contrasts, metrics=METRICS, scope=None):
    lines = ["| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |",
             "|---|---|---|---|---|---|---|---|"]
    for c in contrasts:
        if c["metric"] not in metrics:
            continue
        m = c["metric"]
        arrow = "↑" if HIGHER_IS_BETTER[m] else "↓"
        lines.append(f"| {scope + ': ' if scope else ''}{NAME.get(c['a'], c['a'])} − {NAME.get(c['b'], c['b'])} | {LABEL[m]} {arrow} | {fmt(c['mean_diff'], m)} | "
                     f"[{fmt(c['ci_lo'], m)}, {fmt(c['ci_hi'], m)}] | [{fmt(c['naive_ci_lo'], m)}, {fmt(c['naive_ci_hi'], m)}] | "
                     f"{'yes' if c['excludes_zero'] else 'no'} | {pvalue(c['p'])} | {c['a_better']} / {c['ties']} / {c['a_worse']} |")
    return "\n".join(lines)


def means_table(means, keys, metrics=METRICS):
    lines = ["| Method | " + " | ".join(LABEL[m] for m in metrics) + " |", "|---|" + "---|" * len(metrics)]
    for k in keys:
        if k in means:
            lines.append(f"| {NAME.get(k, k)} | " + " | ".join(f"{means[k][m]:.{DIGITS[m]}f}" for m in metrics) + " |")
    return "\n".join(lines)


def verdict(c) -> str:
    """Plain words for one contrast, from its interval: 'better', 'worse' or 'not distinguishable'."""
    if not c["excludes_zero"]:
        return "not distinguishable"
    good = (c["mean_diff"] > 0) == HIGHER_IS_BETTER[c["metric"]]
    return "better" if good else "worse"


def find(contrasts, a, b, metric):
    return next(c for c in contrasts if c["a"] == a and c["b"] == b and c["metric"] == metric)


def band_means_table(bands, labels):
    lines = ["| Input noise | n (scenes) | Method | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|---|"]
    for key, title in labels:
        if key not in bands:
            continue
        e = bands[key]
        for i, k in enumerate(("classical", "retinexformer", "qgh")):
            first = f"{title}; observed {e['input_noise_range'][0]:.4f}–{e['input_noise_range'][1]:.4f}" if i == 0 else ""
            count = f"{e['n']} ({e['scenes']})" if i == 0 else ""
            lines.append(f"| {first} | {count} | {NAME[k]} | {e['means'][k]['psnr']:.2f} | {e['means'][k]['ssim']:.4f} | {e['means'][k]['lpips']:.4f} |")
    return "\n".join(lines)


def strata_section(name, st) -> list[str]:
    L = [f"#### {name}", ""]
    labels = (("low", f"low (σ < {st['noise_split']})"), ("high", f"high (σ ≥ {st['noise_split']})"))
    L += [band_means_table(st["bands"], labels), ""]
    for key, title in labels:
        if key in st["bands"]:
            L += [contrast_rows(st["bands"][key]["contrasts"], scope=title.split(" ")[0]), ""]
    if st["brightness_within_low_noise"]:
        L += ["Brightness within the low-noise group only (so that darkness is not confused with noise):", "",
              "| Third of input brightness | Input brightness | n | Classical PSNR | Retinexformer PSNR | Quality Guided PSNR | Retinexformer SSIM | Retinexformer LPIPS |",
              "|---|---|---|---|---|---|---|---|"]
        for e in st["brightness_within_low_noise"]:
            m = e["means"]
            L.append(f"| {e['label']} | {e['input_brightness_range'][0]:.3f}–{e['input_brightness_range'][1]:.3f} | {e['n']} | {m['classical']['psnr']:.2f} | "
                     f"{m['retinexformer']['psnr']:.2f} | {m['qgh']['psnr']:.2f} | {m['retinexformer']['ssim']:.4f} | {m['retinexformer']['lpips']:.4f} |")
        L.append("")
    return L


def summary_bullets(s) -> list[str]:
    """Plain-language headline of section 10, every number taken from the results."""
    out = ["### 10.0 In short", ""]
    e, v = s.get("eval15"), s.get("lolv2_real_test")
    if e and v:
        rc, vc = find(e["contrasts"], "retinexformer", "classical", "psnr"), find(v["contrasts"], "retinexformer", "classical", "psnr")
        out.append(f"- **Retinexformer against the classical pipeline is a solid result on both benchmarks:** {fmt(rc['mean_diff'], 'psnr')} dB on eval15 "
                   f"(95 % interval [{fmt(rc['ci_lo'], 'psnr')}, {fmt(rc['ci_hi'], 'psnr')}], better on {rc['a_better']} of {rc['n']} images) and "
                   f"{fmt(vc['mean_diff'], 'psnr')} dB on LOL-v2-real Test with its matching checkpoint ([{fmt(vc['ci_lo'], 'psnr')}, {fmt(vc['ci_hi'], 'psnr')}], "
                   f"{vc['a_better']} of {vc['n']} images, {vc['scenes']} scenes).")
        q = find(e["contrasts"], "retinexformer", "qgh", "psnr")
        qs, ql = find(e["contrasts"], "retinexformer", "qgh", "ssim"), find(e["contrasts"], "retinexformer", "qgh", "lpips")
        out.append(f"- **Quality Guided costs almost nothing on these low-noise sets:** on eval15 plain Retinexformer is {verdict(q)} on PSNR by "
                   f"{fmt(q['mean_diff'], 'psnr')} dB ([{fmt(q['ci_lo'], 'psnr')}, {fmt(q['ci_hi'], 'psnr')}]), while the SSIM difference is {verdict(qs)} and the LPIPS difference is {verdict(ql)}.")
    dev = (s.get("strata") or {}).get("dev")
    if dev and "high" in dev["bands"] and "low" in dev["bands"]:
        hi, lo = dev["bands"]["high"], dev["bands"]["low"]
        rp, rs = find(hi["contrasts"], "retinexformer", "classical", "psnr"), find(hi["contrasts"], "retinexformer", "classical", "ssim")
        gp = find(hi["contrasts"], "qgh", "retinexformer", "psnr")
        bench_max = max(s["strata"][k]["bands"]["low"]["input_noise_range"][1] for k in ("eval15", "lolv2_real_test") if k in s["strata"])
        out.append(f"- **Very noisy inputs are the weak spot.** Of the {lo['n'] + hi['n']} dev images, {hi['n']} have an input noise σ of "
                   f"{hi['input_noise_range'][0]:.3f}–{hi['input_noise_range'][1]:.3f}, against at most {bench_max:.3f} in any benchmark image. "
                   f"On those, Retinexformer against the classical pipeline is {verdict(rp)} on PSNR ({fmt(rp['mean_diff'], 'psnr')} dB) and {verdict(rs)} on SSIM ({fmt(rs['mean_diff'], 'ssim')}). "
                   f"Quality Guided is better than plain Retinexformer on {gp['a_better']} of {gp['n']} of them ({fmt(gp['mean_diff'], 'psnr')} dB PSNR).")
        out.append("- **No benchmark image and no unseen photo is that noisy**, so the headline numbers describe the low-noise regime only.")
    n = ((s.get("noref") or {}).get("sets") or {}).get("unseen phone photos")
    if n:
        c = next(c for c in n["contrasts"] if c["a"] == "highlight_safe" and c["b"] == "natural" and c["metric"] == "overexposed")
        out.append(f"- **On the 20 unseen photos (no ground truth)** Highlight-safe clips {fmt(-c['mean_diff'], 'overexposed').lstrip('+')} percentage points fewer pixels than Natural "
                   f"([{fmt(-c['ci_hi'], 'overexposed').lstrip('+')}, {fmt(-c['ci_lo'], 'overexposed').lstrip('+')}]; fewer on {c['a_better']} of {c['n']} photos); NIQE cannot tell the styles apart.")
    return out + [""]


def build_section(s) -> str:
    L = ["## 10. Post-freeze additions: statistical checks", "",
         "Everything in this section was added **after** the Phase 5 freeze. It changes no model, configuration, threshold or headline",
         "number: sections 4-6 stand as written. Source: `scripts/hardening_study.py`, results in `results/phase_f_hardening/`.",
         f"Intervals are 95 % paired bootstrap percentile intervals ({N_RESAMPLES:,} resamples, seed {SEED}). The resampling unit is a",
         "*scene* (images whose thumbnails are near-duplicates count as one), and the naive per-image interval is shown beside it. Intervals",
         "are not adjusted for the number of comparisons. The interval averages over images and resamples scenes; the sign-flip p-value tests the per-scene",
         "means, so for sets with scenes of unequal size the two can disagree near the boundary. A / tie / A worse count images, with ties below the reported precision.", ""]
    L += summary_bullets(s)
    e = s.get("eval15")
    if e:
        L += ["### 10.1 eval15: how far can 15 images be trusted?", "",
              f"{e['n']} images, {e['scenes']} scenes, checkpoint `{e['checkpoint']}`. The three frozen methods were measured again and reproduce the frozen "
              f"per-image results (largest per-image difference: {e['frozen_check_max_abs_diff']['psnr']:.4f} dB PSNR, {e['frozen_check_max_abs_diff']['ssim']:.4f} SSIM, "
              f"{e['frozen_check_max_abs_diff']['lpips']:.4f} LPIPS). "
              f"Near-duplicate neighbours in the Retinexformer training set (LOL-v1 our485): {e['near_duplicate_in_training']['tol_0.01']} of {e['n']} eval15 images "
              f"at a strict threshold, {e['near_duplicate_in_training']['tol_0.03']} at a looser one.", "",
              means_table(e["means"], [k for k, *_ in METHODS]), "", contrast_rows(e["contrasts"]), ""]
    v = s.get("lolv2_real_test")
    if v:
        lk = v["leakage"]
        L += ["### 10.2 LOL-v2-real Test with the LOL-v2-real checkpoint", "",
              f"{v['n']} images in {v['scenes']} scenes, checkpoint `{v['checkpoint']}` (not the LOL-v1 checkpoint: 90 of these images are copies of its training set). "
              f"Leakage check for *this* checkpoint: images with a near-duplicate in LOL-v2-real Train: {lk['near_duplicate_in_v2_real_train']['tol_0.01']} (strict), "
              f"{lk['near_duplicate_in_v2_real_train']['tol_0.03']} (loose), {lk['near_duplicate_in_v2_real_train']['tol_0.05']} (very loose: neighbouring shots). "
              "The Quality Guided thresholds were calibrated on LOL-v1-checkpoint outputs, so its rows here are a transfer test, not a re-calibration.", "",
              means_table(v["means"], [k for k, *_ in METHODS]), "", contrast_rows(v["contrasts"]), ""]
    n = s.get("noref")
    if n:
        L += ["### 10.3 No-reference scores (no ground truth)", ""]
        if not n["niqe_available"]:
            L += ["NIQE could not be computed on this machine (pyiqa unavailable); only noise and over-exposure are shown.", ""]
        for name, entry in n["sets"].items():
            L += [f"**{name}** ({entry['n']} photos). Lower is better for every column.", "",
                  means_table(entry["means"], ["classical", "natural", "highlight_safe"], metrics=("niqe", "noise", "overexposed")), "",
                  contrast_rows(entry["contrasts"], metrics=("niqe", "noise", "overexposed")), ""]
        if n["not_provided"]:
            L += [f"Not scored because no images were provided: {', '.join(n['not_provided'])}. Put them in `data/noref/<name>/` and rerun "
                  "`python scripts/hardening_study.py noref`.", ""]
        L += ["No ground truth exists for these photos, so these numbers say nothing about fidelity. NIQE rewards natural-looking statistics, not accuracy.", ""]
    st = s.get("strata")
    if st:
        L += ["### 10.4 Where is the model weakest? (input noise and brightness)", ""]
        dev = st.get("dev")
        if dev and "high" in dev["bands"] and "low" in dev["bands"]:
            hi, lo = dev["bands"]["high"], dev["bands"]["low"]
            rc = {m: find(hi["contrasts"], "retinexformer", "classical", m) for m in METRICS}
            rl = {m: find(lo["contrasts"], "retinexformer", "classical", m) for m in METRICS}
            L += [f"The dev split (LOL-v2-real Train without copies of the LOL-v1 checkpoint's training data) holds two populations of input noise "
                  f"(estimated σ): {lo['n']} images between {lo['input_noise_range'][0]:.4f} and {lo['input_noise_range'][1]:.4f}, and {hi['n']} images between "
                  f"{hi['input_noise_range'][0]:.4f} and {hi['input_noise_range'][1]:.4f}, with none in between. The split used here, σ = {dev['noise_split']}, "
                  "sits in that gap; it was chosen after looking at the data, so treat it as exploratory. "
                  f"Across the dev split, input noise and Retinexformer's SSIM correlate at {dev['corr_input_noise_vs_retinexformer_ssim']:+.2f}, "
                  f"and input noise and brightness at {dev['corr_input_noise_vs_brightness']:+.2f}, which is why brightness is examined only within the low-noise group.", "",
                  f"Retinexformer against Classical: in the **low-noise** group it is {verdict(rl['psnr'])} on PSNR, {verdict(rl['ssim'])} on SSIM and "
                  f"{verdict(rl['lpips'])} on LPIPS; in the **high-noise** group it is {verdict(rc['psnr'])} on PSNR, {verdict(rc['ssim'])} on SSIM and "
                  f"{verdict(rc['lpips'])} on LPIPS (95 % intervals by scene, tables below).", ""]
        ranges = []
        for key, nm in (("eval15", "eval15"), ("lolv2_real_test", "LOL-v2-real Test")):
            if key in st and "low" in st[key]["bands"]:
                b = st[key]["bands"]["low"]
                ranges.append(f"{nm}: {b['input_noise_range'][0]:.4f}–{b['input_noise_range'][1]:.4f}")
        un = (s.get("noref") or {}).get("sets", {}).get("unseen phone photos")
        if un:
            ranges.append(f"the 20 unseen phone photos: {un['input']['noise_min']:.4f}–{un['input']['noise_max']:.4f} (median {un['input']['noise_median']:.4f})")
        if ranges:
            L += ["Input noise (estimated σ) of the other evidence in this report: " + "; ".join(ranges) + ". "
                  "**No benchmark image and no unseen photo falls in the high-noise group**, so every headline number in this report describes the low-noise regime only.", ""]
        L += strata_section("eval15 (LOL-v1 checkpoint)", st["eval15"]) if "eval15" in st else []
        L += strata_section("LOL-v2-real Test (LOL-v2-real checkpoint)", st["lolv2_real_test"]) if "lolv2_real_test" in st else []
        L += strata_section("Dev split (LOL-v1 checkpoint)", st["dev"]) if dev else []
        d = s.get("dev")
        if d:
            L += [f"Dev split overall ({d['n']} images, {d['scenes']} scenes):", "", means_table(d["means"], ["classical", "retinexformer", "qgh"]), "",
                  contrast_rows(d["contrasts"]), ""]
    return "\n".join(L).rstrip() + "\n"


EVAL_JSON = ROOT / "app" / "web" / "static" / "data" / "evaluation.json"


def robustness_block(s) -> dict:
    """The 'how far can these numbers be trusted' table of the details page, built from summary.json only."""
    rows = []

    def add(dataset, a, b, metric, contrasts):
        c = find(contrasts, a, b, metric)
        short = lambda k: NAME[k].replace(" (frozen)", "")
        rows.append({"dataset": dataset, "comparison": f"{short(a)} − {short(b)}", "metric": LABEL[metric],
                     "difference": fmt(c["mean_diff"], metric), "interval": f"[{fmt(c['ci_lo'], metric)}, {fmt(c['ci_hi'], metric)}]",
                     "reading": verdict(c), "metric_key": metric, "higher_is_better": HIGHER_IS_BETTER[metric],
                     "mean": round(c["mean_diff"], 5), "lo": round(c["ci_lo"], 5), "hi": round(c["ci_hi"], 5),
                     "n": c["n"], "a_better": c["a_better"], "a_worse": c["a_worse"]})

    if "eval15" in s:
        for a, b, m in (("retinexformer", "classical", "psnr"), ("retinexformer", "classical", "ssim"), ("retinexformer", "qgh", "psnr"),
                        ("retinexformer", "qgh", "ssim"), ("retinexformer", "qgh", "lpips")):
            add(f"eval15 ({s['eval15']['n']} images)", a, b, m, s["eval15"]["contrasts"])
    if "lolv2_real_test" in s:
        for a, b, m in (("retinexformer", "classical", "psnr"), ("retinexformer", "qgh", "psnr")):
            add(f"LOL-v2-real Test ({s['lolv2_real_test']['n']} images)", a, b, m, s["lolv2_real_test"]["contrasts"])
    dev = (s.get("strata") or {}).get("dev")
    if dev:
        for band, label in (("low", "low-noise"), ("high", "high-noise")):
            if band in dev["bands"]:
                for m in ("psnr", "ssim"):
                    add(f"Dev split, {label} ({dev['bands'][band]['n']} images)", "retinexformer", "classical", m, dev["bands"][band]["contrasts"])
    caveat = ""
    if dev and "high" in dev["bands"] and "low" in dev["bands"]:
        hi = dev["bands"]["high"]
        caveat = (f"Every test set above, and the unseen photos, has low input noise (σ up to {max(b['bands']['low']['input_noise_range'][1] for b in (s['strata'][k] for k in ('eval15', 'lolv2_real_test') if k in s['strata'])):.3f}). "
                  f"The development split also contains {hi['n']} much noisier photos (σ about {hi['input_noise_range'][0]:.2f}–{hi['input_noise_range'][1]:.2f}); "
                  "on those the Natural style is not clearly better than the Quick one (and slightly worse on SSIM), so the headline numbers should not be read as a promise for very noisy photos. "
                  f"Highlight-safe did better than Natural on {find(hi['contrasts'], 'qgh', 'retinexformer', 'psnr')['a_better']} of those {hi['n']} photos.")
    return {"intro": f"95 % paired bootstrap intervals ({N_RESAMPLES:,} resamples), resampled by scene so that repeated shots of one scene count once. "
                     "A difference whose interval includes zero is called not distinguishable. The intervals are not adjusted for the number of comparisons. "
                     "LOL-v2-real Test is scored with the matching LOL-v2-real checkpoint; every other row uses the LOL-v1 checkpoint.",
            "rows": rows, "caveat": caveat,
            # the "In short" bullets of eval_P1_T66.md section 10.0, as plain text
            "summary": [re.sub(r"\*\*", "", line[2:]) for line in summary_bullets(s) if line.startswith("- ")],
            "noise_bands": noise_bands_block(s)}


def noise_bands_block(s) -> list[dict]:
    """Mean results per input-noise group on the dev split, for the Evidence page's noise chart."""
    dev = (s.get("strata") or {}).get("dev")
    if not dev:
        return []
    out = []
    for key, label in (("low", "Low-noise photos"), ("high", "Very noisy photos")):
        b = dev["bands"].get(key)
        if not b:
            continue
        rc = find(b["contrasts"], "retinexformer", "classical", "psnr")
        out.append({"key": key, "label": label, "n": b["n"], "noise_range": [round(v, 4) for v in b["input_noise_range"]],
                    "psnr": {k: round(b["means"][k]["psnr"], 3) for k in ("classical", "retinexformer", "qgh")},
                    "ssim": {k: round(b["means"][k]["ssim"], 4) for k in ("classical", "retinexformer", "qgh")},
                    "natural_vs_quick_psnr": {"mean": round(rc["mean_diff"], 3), "lo": round(rc["ci_lo"], 3), "hi": round(rc["ci_hi"], 3),
                                              "reading": verdict(rc)}})
    return out


def write_evaluation_json(block: dict) -> None:
    """Add or replace the "robustness" key of evaluation.json without reformatting the rest of the file."""
    text = EVAL_JSON.read_text(encoding="utf-8").rstrip()
    i = text.find(',\n  "robustness"')
    if i >= 0:
        text = text[:i] + "\n}"
    body = json.dumps(block, indent=2, ensure_ascii=False).replace("\n", "\n  ")
    EVAL_JSON.write_text(text[:-1].rstrip() + ',\n  "robustness": ' + body + "\n}\n", encoding="utf-8")


def cmd_report(args):
    s = load_summary()
    section = build_section(s)
    (OUT / "summary.md").write_text(section, encoding="utf-8")
    if all(k in s for k in ("eval15", "lolv2_real_test", "strata")):
        write_evaluation_json(robustness_block(s))
    text = EVAL_MD.read_text(encoding="utf-8")
    block = f"{START}\n{section}{END}\n"
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block
    EVAL_MD.write_text(text, encoding="utf-8")
    print(f"wrote {OUT / 'summary.md'} and the phase-f section of {EVAL_MD.name}")


COMMANDS = {"bootstrap": cmd_bootstrap, "lolv2": cmd_lolv2, "dev": cmd_dev, "noref": cmd_noref, "strata": cmd_strata, "report": cmd_report}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N images of the large sets")
    args = ap.parse_args()
    for c in (["bootstrap", "lolv2", "dev", "noref", "strata", "report"] if args.command == "all" else [args.command]):
        COMMANDS[c](args)


if __name__ == "__main__":
    main()
