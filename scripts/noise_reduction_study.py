"""Noise reduction for very noisy photos (research branch `research/noise-reduction`).

Question: can a denoising step after Retinexformer make Natural / Highlight-safe clearly better on very noisy
REAL photos, while leaving low-noise photos (the eval15 headline, real phone photos) unchanged?

Prior work: docs/NOISE_STUDY.md (QGH-NA v1/v2, mostly synthetic noise). The real-noise evidence there is only the
269-pair dev split, which was also used for selection, so nothing real was held out. This study fixes that first:

  python scripts/noise_reduction_study.py cache [--sets dev_real eval15 val_real unseen]
        run Retinexformer ONCE per image; store input, model output (float16) and ground truth on disk
        (results/noise_reduction/cache/, git-ignored)
  python scripts/noise_reduction_study.py split
        scene-level split of the 269 real dev pairs into TUNE and TEST halves, stratified by input noise
        (fixed seed) -> data/splits/noise_reduction_{tune,test}.txt. TEST is not looked at until a method is frozen.
  python scripts/noise_reduction_study.py baseline [--set tune]
        current styles, QGH-NA v2 and simple denoisers on the TUNE half, by noise group

Noise groups use the app's own input measurement (skimage estimate_sigma, lumen.metrics.noise_level). The dev
inputs fall into two populations with nothing between 0.0063 and 0.0413 (eval_P1_T66.md section 10.4), so the
group split at 0.025 is not sensitive to its exact value.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from lumen.metrics.quality import lpips_distance, noise_level, psnr, ssim
from lumen.utils import limit_size, read_image

OUT = ROOT / "results" / "noise_reduction"
CACHE = OUT / "cache"
SPLITS = ROOT / "data" / "splits"
MAX_SIDE = 768            # the deployed processing size
NOISY = 0.025             # input sigma separating the two dev populations (gap 0.0063 - 0.0413)
SEED = 20261120
DEFAULT_SETS = ("dev_real", "eval15", "val_real", "unseen")


# ============================================================================ data
def load_items(name: str) -> list[dict]:
    import noise_study as v1
    import noise_study_v2 as v2

    if name in ("dev_real", "dev_syn", "eval15", "dev_detail", "val_detail", "val_general", "val_real"):
        return v2.load_set(name)
    return v1.load_set(name)


def cmd_cache(args):
    from lumen.deep import DeepEnhancer

    enh = DeepEnhancer("retinexformer")
    print(f"model: {enh.name} on {enh.device}, engine {getattr(enh, 'engine', 'torch')}")
    for name in args.sets:
        d = CACHE / name
        d.mkdir(parents=True, exist_ok=True)
        items = load_items(name)
        index, t0 = [], time.perf_counter()
        for i, it in enumerate(items):
            f = d / f"{i:04d}.npz"
            x = limit_size(read_image(it["low"]), MAX_SIDE)
            rec = {"i": i, "id": str(it["id"]), "low": str(Path(it["low"]).relative_to(ROOT)).replace("\\", "/"),
                   "gt": str(Path(it["gt"]).relative_to(ROOT)).replace("\\", "/") if it.get("gt") else None,
                   "input_sigma": noise_level(x), "shape": list(x.shape)}
            if not f.exists():
                gt = read_image(it["gt"]) if it.get("gt") else None
                if gt is not None and gt.shape[:2] != x.shape[:2]:
                    gt = cv2.resize(gt, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
                r, _ = enh(x)
                np.savez_compressed(f, x=x.astype(np.float32), r=r.astype(np.float16),
                                    gt=(np.round(gt * 255).astype(np.uint8) if gt is not None else np.zeros(0, np.uint8)))
            index.append(rec)
        (d / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"  {name}: {len(index)} images, {time.perf_counter() - t0:.0f}s")


def load_cached(name: str, ids: set[str] | None = None) -> list[dict]:
    d = CACHE / name
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    out = []
    for rec in index:
        if ids is not None and rec["low"] not in ids:
            continue
        z = np.load(d / f"{rec['i']:04d}.npz")
        gt = z["gt"].astype(np.float32) / 255 if z["gt"].size else None
        out.append({**rec, "x": z["x"], "r": z["r"].astype(np.float32), "gt_img": gt})
    return out


# ============================================================================ split
def cmd_split(_args):
    from hardening_study import fingerprints_of, scene_groups

    index = json.loads((CACHE / "dev_real" / "index.json").read_text(encoding="utf-8"))
    pairs = [(ROOT / r["low"], ROOT / r["gt"]) for r in index]
    groups = scene_groups(fingerprints_of(pairs))
    noisy = np.array([r["input_sigma"] >= NOISY for r in index])
    rng = np.random.default_rng(SEED)
    tune = np.zeros(len(index), bool)
    for flag in (True, False):           # stratify: half of the noisy scenes, half of the low-noise scenes
        scenes = sorted({g for g, n in zip(groups, noisy) if n == flag})
        scenes = list(rng.permutation(scenes))
        pick = set(scenes[: len(scenes) // 2])
        tune |= np.array([g in pick and n == flag for g, n in zip(groups, noisy)])
    for nm, sel in (("tune", tune), ("test", ~tune)):
        lines = [f"{index[i]['low']}\t{index[i]['gt']}\t{'noisy' if noisy[i] else 'low_noise'}\tscene{groups[i]}\n"
                 for i in np.flatnonzero(sel)]
        (SPLITS / f"noise_reduction_{nm}.txt").write_text("".join(lines), encoding="utf-8")
        print(f"{nm}: {sel.sum()} images ({(sel & noisy).sum()} noisy, {(sel & ~noisy).sum()} low-noise), "
              f"{len(set(groups[sel]))} scenes")
    assert not (set(groups[tune]) & set(groups[~tune])), "a scene is in both halves"


def split_ids(nm: str) -> set[str]:
    return {ln.split("\t")[0] for ln in (SPLITS / f"noise_reduction_{nm}.txt").read_text(encoding="utf-8").splitlines() if ln}


# ============================================================================ methods
def base_methods():
    from lumen.experimental.noise_aware import NoiseAwareQGH, load_na_config
    from lumen.hybrid.highlight_hue import highlight_hue_fix, load_hue_config
    from lumen.hybrid.quality_guided import QualityGuidedHybrid, load_qg_config

    hue_cfg, qg_cfg = load_hue_config(), load_qg_config()
    qgh = QualityGuidedHybrid(None, qg_cfg)
    v2 = NoiseAwareQGH(None, qg_cfg, load_na_config(ROOT / "configs/experimental/noise_aware_v2.yaml"))

    def natural(it):
        return highlight_hue_fix(it["r"], it["x"], hue_cfg)[0]

    return {
        "natural": natural,                                               # production Natural
        "highlight_safe": lambda it: qgh.refine(natural(it), it["x"]).image,   # production Highlight-safe
        "qgh_na_v2": lambda it: v2.refine(natural(it), it["x"]).image,   # docs/NOISE_STUDY.md section 9
    }, natural


def nlm(img: np.ndarray, h: float, hc: float, tw: int = 7, sw: int = 21) -> np.ndarray:
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    out = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), None, h, hc, tw, sw)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def score(out: np.ndarray, gt: np.ndarray) -> dict:
    return {"psnr": psnr(out, gt), "ssim": ssim(out, gt), "lpips": lpips_distance(out, gt)}


def cmd_baseline(args):
    items = load_cached("dev_real", split_ids(args.set))
    methods, natural = base_methods()
    for h in (5, 10, 15):
        methods[f"nlm_h{h}"] = (lambda hh: lambda it: nlm(natural(it), hh, hh))(h)
    rows = []
    t0 = time.perf_counter()
    for it in items:
        grp = "noisy" if it["input_sigma"] >= NOISY else "low_noise"
        for m, fn in methods.items():
            t = time.perf_counter()
            out = fn(it)
            ms = (time.perf_counter() - t) * 1000
            rows.append({"id": it["low"], "group": grp, "method": m, **score(out, it["gt_img"]), "ms": ms})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"baseline_{args.set}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(items)} images, {time.perf_counter() - t0:.0f}s\n")
    print_table(rows)


def cmd_explore(args):
    """Strength grid for NLM after Natural, on the NOISY images of the TUNE half only (parallel NLM)."""
    from concurrent.futures import ThreadPoolExecutor

    items = [it for it in load_cached("dev_real", split_ids("tune")) if it["input_sigma"] >= NOISY]
    _, natural = base_methods()
    nat = [natural(it) for it in items]
    s_out = [noise_level(r) for r in nat]
    grid = {"natural": None}
    for h in (15, 20, 25):
        grid[f"fixed_h{h}"] = lambda i, h=h: (h, h)
    for k in (0.8, 1.0, 1.2, 1.4):
        for c in (1.0, 1.5):
            grid[f"adapt_k{k}_c{c}"] = lambda i, k=k, c=c: (k * s_out[i] * 255, c * k * s_out[i] * 255)
    rows = []
    with ThreadPoolExecutor(args.threads) as pool:
        for name, hp in grid.items():
            t0 = time.perf_counter()
            outs = list(nat) if hp is None else list(pool.map(lambda i: nlm(nat[i], *hp(i)), range(len(items))))
            for i, (it, out) in enumerate(zip(items, outs)):
                rows.append({"id": it["low"], "group": "noisy", "method": name, **score(out, it["gt_img"]),
                             "ms": (time.perf_counter() - t0) * 1000 / len(items)})
    (OUT / "explore_tune.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print_table(rows)
    base = {r["id"]: r for r in rows if r["method"] == "natural"}
    print("worst single image vs natural (dSSIM, dLPIPS):")
    for name in grid:
        if name == "natural":
            continue
        rs = [r for r in rows if r["method"] == name]
        print(f"  {name:<18} min dSSIM {min(r['ssim'] - base[r['id']]['ssim'] for r in rs):+.3f}"
              f"  max dLPIPS {max(r['lpips'] - base[r['id']]['lpips'] for r in rs):+.3f}")


VAL_SETS = ("test", "syn_noisy", "syn_clean", "val_detail", "val_general", "examples", "eval15", "val_real", "unseen")


def cmd_validate(args):
    """Run ONCE with the frozen configs/noise_reduction.yaml on held-out data. Production code path:
    Natural = hue(R); Natural+NR = reduce_noise(hue(R)); Highlight-safe = QGH(hue(R)); Highlight-safe+NR = QGH(NR(hue(R)))."""
    from concurrent.futures import ThreadPoolExecutor

    from hardening_study import compare

    from lumen.hybrid.highlight_hue import highlight_hue_fix, load_hue_config
    from lumen.hybrid.noise_reduction import load_nr_config, reduce_noise
    from lumen.hybrid.quality_guided import QualityGuidedHybrid, load_qg_config

    nr_cfg, hue_cfg = load_nr_config(), load_hue_config()
    qgh = QualityGuidedHybrid(None, load_qg_config())
    print("frozen config:", nr_cfg)

    def run(it):
        nat = highlight_hue_fix(it["r"], it["x"], hue_cfg)[0]
        nat_nr, info = reduce_noise(nat, it["input_sigma"], nr_cfg)
        hs = qgh.refine(nat, it["x"]).image
        hs_nr = hs if not info["applied"] else qgh.refine(nat_nr, it["x"]).image
        return {"natural": nat, "natural_nr": nat_nr, "highlight_safe": hs, "highlight_safe_nr": hs_nr}, info

    report, summary = [], {}
    for set_name in VAL_SETS:
        if set_name == "test":
            ids = split_ids("test")
            scene = {ln.split("\t")[0]: ln.split("\t")[3] for ln in
                     (SPLITS / "noise_reduction_test.txt").read_text(encoding="utf-8").splitlines() if ln}
            items = load_cached("dev_real", ids)
        else:
            items, scene = load_cached(set_name), {}
        rows = []
        for k in range(0, len(items), 16):
            chunk = items[k:k + 16]
            with ThreadPoolExecutor(args.threads) as pool:
                results = list(pool.map(run, chunk))
            for it, (outs, info) in zip(chunk, results):
                row = {"id": it["low"], "scene": scene.get(it["low"], it["low"]), "input_sigma": it["input_sigma"],
                       "noisy": it["input_sigma"] >= NOISY, "nr": info,
                       "identical_natural": bool(np.array_equal(outs["natural"], outs["natural_nr"])),
                       "identical_hs": bool(np.array_equal(outs["highlight_safe"], outs["highlight_safe_nr"]))}
                if it["gt_img"] is not None:
                    for m, out in outs.items():
                        row[m] = score(out, it["gt_img"])
                rows.append(row)
        summary[set_name] = rows
        report += summarise(set_name, rows)
        print(f"{set_name}: done ({len(rows)} images)")
    # paired bootstrap by scene on the real held-out noisy images (the claim) and the real low-noise ones (safety)
    stats = {}
    test = summary["test"]
    for grp, sel in (("test_noisy", [r for r in test if r["noisy"]]), ("test_low_noise", [r for r in test if not r["noisy"]])):
        g = np.array([r["scene"] for r in sel])
        for base, new in (("natural", "natural_nr"), ("highlight_safe", "highlight_safe_nr")):
            for metric in ("psnr", "ssim", "lpips"):
                stats[f"{grp}:{new}-{base}:{metric}"] = compare(np.array([r[new][metric] for r in sel]),
                                                                np.array([r[base][metric] for r in sel]), metric, g)
    report += ["", "## Paired bootstrap by scene, real held-out TEST half (A - B; 95 % interval; better/tie/worse)", "",
               "| Group | Comparison | Metric | Mean diff | 95 % interval | Zero excluded | Better / tie / worse |",
               "|---|---|---|---|---|---|---|"]
    for k, s in stats.items():
        grp, cmp_, metric = k.split(":")
        report.append(f"| {grp} (n={s['n']}, {s['scenes']} scenes) | {cmp_} | {metric} | {s['mean_diff']:+.4f} | "
                      f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | {'yes' if s['excludes_zero'] else 'no'} | "
                      f"{s['a_better']} / {s['ties']} / {s['a_worse']} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps({"config": nr_cfg.as_dict(), "sets": summary, "stats": stats},
                                                    indent=1, default=float), encoding="utf-8")
    (OUT / "validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def summarise(set_name: str, rows: list[dict]) -> list[str]:
    applied = [r for r in rows if r["nr"]["applied"]]
    lines = ["", f"## {set_name} (n={len(rows)}): noise reduction triggered on {len(applied)}; "
             f"bit-identical Natural {sum(r['identical_natural'] for r in rows)}/{len(rows)}, "
             f"Highlight-safe {sum(r['identical_hs'] for r in rows)}/{len(rows)}"]
    if applied:
        lines.append(f"NR time (ms, applied images): median {np.median([r['nr']['ms'] for r in applied]):.0f}, "
                     f"max {max(r['nr']['ms'] for r in applied):.0f}")
    if "natural" not in rows[0]:
        return lines + ["(no ground truth)"]
    lines += ["", "| Subset | Method | PSNR | SSIM | LPIPS | worst dSSIM vs base | worst dLPIPS vs base |", "|---|---|---|---|---|---|---|"]
    for sub_name, sub in (("all", rows), ("triggered", applied)):
        if not sub:
            continue
        for m, base in (("natural", None), ("natural_nr", "natural"), ("highlight_safe", None), ("highlight_safe_nr", "highlight_safe")):
            mean = lambda k: float(np.mean([r[m][k] for r in sub]))
            worst = ("", "") if base is None else (f"{min(r[m]['ssim'] - r[base]['ssim'] for r in sub):+.4f}",
                                                   f"{max(r[m]['lpips'] - r[base]['lpips'] for r in sub):+.4f}")
            lines.append(f"| {sub_name} (n={len(sub)}) | {m} | {mean('psnr'):.2f} | {mean('ssim'):.4f} | {mean('lpips'):.4f} | {worst[0]} | {worst[1]} |")
    return lines


def print_table(rows):
    groups = sorted({r["group"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for g in groups:
        n = len({r["id"] for r in rows if r["group"] == g})
        print(f"[{g}] n={n}")
        print(f"  {'method':<16}{'PSNR':>8}{'SSIM':>9}{'LPIPS':>9}{'ms':>8}")
        for m in methods:
            rs = [r for r in rows if r["group"] == g and r["method"] == m]
            mean = lambda k: float(np.mean([r[k] for r in rs]))
            print(f"  {m:<16}{mean('psnr'):8.2f}{mean('ssim'):9.4f}{mean('lpips'):9.4f}{mean('ms'):8.0f}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    sub.add_parser("split")
    b = sub.add_parser("baseline")
    b.add_argument("--set", default="tune", choices=["tune"])  # TEST is reserved for the frozen method
    x = sub.add_parser("explore")
    x.add_argument("--threads", type=int, default=12)
    v = sub.add_parser("validate")
    v.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    {"cache": cmd_cache, "split": cmd_split, "baseline": cmd_baseline, "explore": cmd_explore,
     "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    main()
