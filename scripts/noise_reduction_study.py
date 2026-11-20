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
    if name == "fresh":
        return [dict(id=Path(lo).parent.name + "/" + Path(lo).stem, low=ROOT / lo, gt=ROOT / hi, clean_low=None, tag=tag)
                for lo, hi, tag in v1._pairs(SPLITS / "noise_reduction_fresh.txt")]
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


FRESH = ROOT / "data" / "noise_reduction_fresh"


def cmd_build_fresh(_args):
    """A NEW held-out synthetic set for the moderate-noise extension (iteration 2): source photos and generated scenes
    that no earlier split used, new noise draws. Every scene has a noise-free and a noisy version."""
    import noise_study_v2 as v2s
    from train_retinexformer import fingerprint

    def names(split):
        return {Path(ln.split("\t")[1]).name for ln in (SPLITS / split).read_text(encoding="utf-8").splitlines() if ln}

    train = sorted((ROOT / "data/LOL-v2/Synthetic/Train/Normal").glob("*.png"))
    test = sorted((ROOT / "data/LOL-v2/Synthetic/Test/Normal").glob("*.png"))
    used = set()
    for f in ("noise_study_dev_syn.txt", "noise_study_v2_val_general.txt", "noise_study_v2_dev_detail.txt",
              "noise_study_v2_val_detail.txt", "highlight_hue_dev.txt", "highlight_hue_eval.txt"):
        used |= names(f)
    pool = [p for p in train if p.name not in used]
    ref = np.stack([fingerprint(p) for p in [t for t in train if t.name in used] + test])
    d = np.abs(np.stack([fingerprint(p) for p in pool])[:, None] - ref[None]).mean(-1).min(1)
    pool = [p for p, di in zip(pool, d) if di >= 0.01]
    energy = {p: v2s.texture_energy(p) for p in pool}
    textured = sorted(pool, key=lambda p: -energy[p])[:16]
    rest = [p for p in pool if p not in textured]
    rng = np.random.default_rng(20261121)
    general = list(rng.choice(rest, 50, replace=False))
    print(f"pool after exclusions: {len(pool)} photos; general 50, textured 16")

    r = np.random.default_rng(31)
    hw = [cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX | cv2.FONT_ITALIC]
    pr = [cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC]
    scenes = [(f"fresh_hand{i:02d}", "hand", v2s.handwriting(r, hw, cursive=0.5)) for i in range(10)]
    scenes += [(f"fresh_print{i:02d}", "print", v2s.handwriting(r, pr)) for i in range(6)]
    scenes += [(f"fresh_wood{i:02d}", "wood", v2s.wood(r)) for i in range(4)]
    scenes += [(f"fresh_fabric{i:02d}", "fabric", v2s.fabric(r)) for i in range(4)]
    scenes += [(f"fresh_tex_{p.stem}", "photo", v2s._load_rgb(p)) for p in textured]
    scenes += [(f"fresh_gen_{p.stem}", "general", v2s._load_rgb(p)) for p in general]
    lines = []
    v2s._pairs_for(scenes, FRESH, np.random.default_rng(4242), lines, (0.03, 0.18, 0.001, 0.012, 300, 6000))
    (SPLITS / "noise_reduction_fresh.txt").write_text("".join(lines), encoding="utf-8")
    print(f"fresh: {len(lines)} images ({len(scenes)} scenes x noise-free/noisy)")


def scale_ratio(img: np.ndarray) -> float:
    """sigma(half resolution) / sigma(full resolution) of the luma noise estimate: grain falls to about 0.2 after 2x
    downsampling, natural texture keeps about 0.6 (docs/NOISE_STUDY.md section 2). High = texture-dominated."""
    from lumen.experimental.noise_aware import estimate_sigma_map

    y = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    _, g_f = estimate_sigma_map(y, 16, 0.3)
    _, g_h = estimate_sigma_map(cv2.pyrDown(y), 8, 0.3)
    return g_h / max(g_f, 1e-6)


def cmd_explore_moderate(args):
    """Moderate-noise extension, DEV ONLY (dev_syn, dev_detail, TUNE half). Precompute NLM at a few strengths, then
    score gate x strength x texture-safeguard policies offline against fixed constraints."""
    import itertools
    from concurrent.futures import ThreadPoolExecutor

    import noise_study as v1
    import noise_study_v2 as v2s

    _, natural = base_methods()
    groups = []
    for it in load_cached("dev_syn"):
        groups.append((it, "syn_" + {str(i["id"]): i["tag"] for i in v1.load_set("dev_syn")}[str(it["id"])]))
    tags = {str(i["id"]): i["tag"] for i in v2s.load_set("dev_detail")}
    for it in load_cached("dev_detail"):
        groups.append((it, "detail_" + ("noisy" if "noisy" in tags[str(it["id"])] else "clean")))
    for it in load_cached("dev_real", split_ids("tune")):
        groups.append((it, "real_noisy" if it["input_sigma"] >= NOISY else "real_clean"))
    ks = (0.8, 1.2, 1.5)
    recs = []
    u8 = lambda im: np.clip(im * 255 + 0.5, 0, 255).astype(np.uint8)  # noqa: E731  (memory: ~300 images in RAM)
    f32 = lambda im: im.astype(np.float32) / 255  # noqa: E731

    def prep(pair):
        it, g = pair
        n = natural(it)
        so = noise_level(n)
        return it, g, n, so, scale_ratio(n), {k: nlm(n, k * so * 255, k * so * 255) for k in ks if it["input_sigma"] >= 0.008}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as pool:
        for it, g, n, so, ratio, outs in pool.map(prep, groups):
            base = score(n, it["gt_img"])
            rec = {"id": it["id"], "group": g, "sin": it["input_sigma"], "sout": so, "ratio": ratio, "base": base, "k": {}}
            for k, d in outs.items():   # blends are scored later for the weights a policy produces
                rec["k"][k] = u8(d)
            rec["nat"] = n.astype(np.float16)
            rec["gt"] = u8(it["gt_img"])
            recs.append(rec)
    print(f"prepared {len(recs)} dev images in {time.perf_counter() - t0:.0f}s")
    print("scale ratio by group (median):", {g: round(float(np.median([r['ratio'] for r in recs if r['group'] == g])), 3)
                                             for g in sorted({r['group'] for r in recs})})
    from lumen.hybrid.quality_guided import smoothstep

    cache = {}

    def blended_score(r, k, w):
        key = (r["id"], r["group"], k, round(w, 3))
        if key not in cache:
            nat = r["nat"].astype(np.float32)
            out = f32(r["k"][k]) if w >= 1 else (1 - w) * nat + w * f32(r["k"][k])
            cache[key] = score(out.astype(np.float32), f32(r["gt"]))
        return cache[key]

    results = []
    for gl, k, tex in itertools.product((0.010, 0.012, 0.015, 0.020, 0.030), ks, (None, (0.45, 0.60), (0.40, 0.55), (0.35, 0.50))):
        gh = gl + 0.01
        rows = []
        for r in recs:
            w = float(smoothstep(np.float32(r["sin"]), gl, gh))
            if tex is not None:
                w *= 1.0 - float(smoothstep(np.float32(r["ratio"]), *tex))
            s = r["base"] if w <= 0 or k not in r["k"] else blended_score(r, k, w)
            rows.append((r["group"], s["lpips"] - r["base"]["lpips"], s["ssim"] - r["base"]["ssim"], s["lpips"], w))
        by = lambda g: [x for x in rows if x[0] == g]  # noqa: E731
        res = {"gate": gl, "k": k, "tex": tex}
        for g in sorted({x[0] for x in rows}):
            xs = by(g)
            res[g] = {"mean_lpips": float(np.mean([x[3] for x in xs])), "worst_dlpips": float(max(x[1] for x in xs)),
                      "worst_dssim": float(min(x[2] for x in xs)), "changed": int(sum(x[4] > 0 for x in xs)), "n": len(xs)}
        results.append(res)
    (OUT / "explore_moderate.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    # constraints fixed before looking at the ranking
    def feasible(r):
        clean_ok = all(r[g]["worst_dlpips"] <= 0.005 for g in ("syn_clean", "detail_clean", "real_clean") if g in r)
        detail_ok = r.get("detail_noisy", {"worst_dlpips": 0})["worst_dlpips"] <= 0.02
        syn_ok = r["syn_noisy"]["worst_dlpips"] <= 0.05
        return clean_ok and detail_ok and syn_ok
    noisy_groups = ("syn_noisy", "detail_noisy", "real_noisy")
    obj = lambda r: float(np.mean([r[g]["mean_lpips"] for g in noisy_groups]))  # noqa: E731
    ranked = sorted(results, key=obj)
    print(f"\n{'gate':>6}{'k':>5}{'tex':>14}{'feasible':>9}{'obj':>8} | " + " | ".join(f"{g} meanLPIPS/worstdLPIPS/changed" for g in noisy_groups))
    for r in ranked[:25]:
        print(f"{r['gate']:6.3f}{r['k']:5.1f}{str(r['tex']):>14}{str(feasible(r)):>9}{obj(r):8.4f} | " +
              " | ".join(f"{r[g]['mean_lpips']:.4f}/{r[g]['worst_dlpips']:+.3f}/{r[g]['changed']}" for g in noisy_groups))
    cur = [r for r in results if r["gate"] == 0.030 and r["k"] == 0.8 and r["tex"] is None][0]
    print(f"\ncurrent (gate 0.030, k 0.8, no safeguard): obj {obj(cur):.4f} feasible {feasible(cur)}")
    best = [r for r in ranked if feasible(r)][:5]
    print("best feasible:", [(r["gate"], r["k"], r["tex"], round(obj(r), 4)) for r in best])
    for r in best[:1]:
        print("clean-group worst dLPIPS:", {g: r[g]["worst_dlpips"] for g in ("syn_clean", "detail_clean", "real_clean")})


VAL_SETS = ("fresh", "test", "syn_noisy", "syn_clean", "val_detail", "val_general", "examples", "eval15", "val_real", "unseen")


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
        elif set_name == "fresh":
            items = load_cached("fresh")
            tags = {ln.split("	")[0]: ln.split("	")[2] for ln in
                    (SPLITS / "noise_reduction_fresh.txt").read_text(encoding="utf-8").splitlines() if ln}
            scene = {it["low"]: Path(it["low"]).stem for it in items}       # noise-free + noisy twin = one scene
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
                       "tag": tags.get(it["low"], "") if set_name == "fresh" else "",
                       "identical_natural": bool(np.array_equal(outs["natural"], outs["natural_nr"])),
                       "identical_hs": bool(np.array_equal(outs["highlight_safe"], outs["highlight_safe_nr"]))}
                if it["gt_img"] is not None:
                    for m, out in outs.items():
                        row[m] = score(out, it["gt_img"])
                rows.append(row)
        summary[set_name] = rows
        report += summarise(set_name, rows)
        print(f"{set_name}: done ({len(rows)} images)")
    # paired bootstrap by scene: iteration 2's claim is the FRESH moderate band; the rest are safety / regression checks
    stats = {}
    test, fresh = summary["test"], summary["fresh"]
    floor = nr_cfg.floor
    for grp, sel in (("fresh_moderate_noisy", [r for r in fresh if r["tag"].endswith("noisy") and floor <= r["input_sigma"] < nr_cfg.gate_low]),
                     ("fresh_heavy_noisy", [r for r in fresh if r["tag"].endswith("noisy") and r["input_sigma"] >= nr_cfg.gate_low]),
                     ("fresh_detail_noisy", [r for r in fresh if r["tag"].endswith("noisy") and r["input_sigma"] >= floor
                                             and r["tag"].split("_")[0] in ("hand", "print", "wood", "fabric", "photo")]),
                     ("fresh_noise_free", [r for r in fresh if r["tag"].endswith("clean")]),
                     ("test_noisy", [r for r in test if r["noisy"]]), ("test_low_noise", [r for r in test if not r["noisy"]])):
        if not sel:
            continue
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
    sub.add_parser("build_fresh")
    em = sub.add_parser("explore_moderate")
    em.add_argument("--threads", type=int, default=12)
    v = sub.add_parser("validate")
    v.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    {"cache": cmd_cache, "split": cmd_split, "baseline": cmd_baseline, "explore": cmd_explore,
     "build_fresh": cmd_build_fresh, "explore_moderate": cmd_explore_moderate, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    main()
