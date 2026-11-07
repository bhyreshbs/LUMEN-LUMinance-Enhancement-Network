"""Highlight hue-preservation study (experimental branch). Production code, configs and results are untouched.

    python scripts/highlight_hue_study.py build     # high-contrast sets with ground truth
    python scripts/highlight_hue_study.py search    # DEV ONLY: pick start/full/chroma_strength/lum_keep
    python scripts/highlight_hue_study.py eval      # frozen config, held-out sets, run once -> results/highlight_hue/
    python scripts/highlight_hue_study.py figures

High-contrast sets: a normal-light LOL-v2 Synthetic photo (the ground truth) is darkened everywhere except
its bright regions, which keep (nearly) their original exposure: lin_in = lin_gt * (e + (1 - e) * m) with
m = feather(smoothstep(Y_gt; 0.7, 0.9)). That is a bright window / lamp in a dark room, with a known
correct colour. The dev set uses LOL-v2 Synthetic *Train* photos, the held-out set Synthetic *Test* photos.

Methods (identical inputs): standard (Retinexformer), qgh (production Highlight-safe), standard+hue, qgh+hue
(the hue fix applied to the model output, then the production QGH step).
Highlight colour error: mean CIELAB a*b* distance to the ground truth over pixels whose input has a channel >= 0.95.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import noise_study as v1  # noqa: E402
from lumen.experimental.highlight_hue import HighlightHueConfig, highlight_hue_fix  # noqa: E402
from lumen.hybrid.quality_guided import QualityGuidedHybrid, load_qg_config, smoothstep  # noqa: E402
from lumen.metrics.quality import lpips_distance, niqe, psnr, ssim  # noqa: E402
from lumen.utils import limit_size, luminance, read_image, write_image  # noqa: E402

DATA, SPLITS = v1.DATA, v1.SPLITS
HC = DATA / "highlight_hue"
OUT = ROOT / "results" / "highlight_hue"
CFG_PATH = ROOT / "configs" / "experimental" / "highlight_hue.yaml"


# ----------------------------------------------------------------------------- build
def make_hc(gt_u8, e, rng, noisy):
    gt = gt_u8.astype(np.float32) / 255.0
    m = np.clip(cv2.GaussianBlur(smoothstep(luminance(gt), 0.7, 0.9), (0, 0), 3), 0, 1)[..., None]
    lin = gt ** 2.2 * (e + (1 - e) * m)
    if noisy:
        lin = np.clip(rng.poisson(lin * 1500) / 1500 + rng.normal(0, 0.004, lin.shape), 0, 1)
    return (np.clip(lin ** (1 / 2.2), 0, 1) * 255 + 0.5).astype(np.uint8)


def build(_a):
    def pick(folder, n, seed):
        rng = np.random.default_rng(seed)
        files = sorted(folder.glob("*.png"))
        good = []
        for f in files:
            g = cv2.imread(str(f)); y = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) / 255.0
            if 0.015 < float((y > 0.85).mean()) < 0.35:  # a real bright region, but not a bright image
                good.append(f)
        idx = rng.permutation(len(good))[:n]
        return [good[i] for i in idx], rng

    for split, folder, n, seed in (("dev", DATA / "LOL-v2/Synthetic/Train/Normal", 50, 3),
                                   ("eval", DATA / "LOL-v2/Synthetic/Test/Normal", 60, 41)):
        files, rng = pick(folder, n, seed)
        lines = []
        for i, f in enumerate(files):
            gt = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            noisy = i % 3 == 2
            lo = HC / split / f"{f.stem}.jpg"
            lo.parent.mkdir(parents=True, exist_ok=True)
            low = make_hc(gt, float(rng.uniform(0.03, 0.12)), rng, noisy)
            cv2.imwrite(str(lo), cv2.cvtColor(low, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines.append(f"{v1.rel(lo)}\t{v1.rel(f)}\t{'hc_noisy' if noisy else 'hc'}\n")
        (SPLITS / f"highlight_hue_{split}.txt").write_text("".join(lines), encoding="utf-8")
        print(split, len(lines), "high-contrast images")


# ----------------------------------------------------------------------------- sets / metrics
def load_set(name):
    if name in ("hc_dev", "hc_eval"):
        f = SPLITS / f"highlight_hue_{name[3:]}.txt"
        return [dict(id=Path(lo).stem, low=ROOT / lo, gt=ROOT / hi, tag=t) for lo, hi, t in v1._pairs(f)]
    if name == "dev_real":
        return v1.load_set("dev_real")[::3]
    if name == "dev_syn":
        return v1.load_set("dev_syn")
    if name == "syn_noisy":
        return [{**i, "tag": "noisy"} for i in v1.load_set("syn_noisy")]
    if name == "syn_clean":
        return [{**i, "tag": "clean"} for i in v1.load_set("syn_clean")]
    if name in ("eval15", "unseen", "examples", "lolv2_syn"):
        return v1.load_set(name)
    raise ValueError(name)


def lab_ab(img):
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)[..., 1:]


def metrics(out, p, with_niqe):
    x, gt = p["x"], p["gt"]
    d = {}
    hl = x.max(axis=2) >= 0.95
    d["hl_frac"] = float(hl.mean())
    if gt is not None:
        d["psnr"], d["ssim"], d["lpips"] = psnr(out, gt), ssim(out, gt), lpips_distance(out, gt)
        if hl.sum() > 50:
            d["hl_dE"] = float(np.linalg.norm(lab_ab(out) - lab_ab(gt), axis=-1)[hl].mean())
            d["hl_Y_gap"] = float(abs(luminance(out)[hl].mean() - luminance(gt)[hl].mean()))
    elif hl.sum() > 50:
        d["hl_dE_vs_input"] = float(np.linalg.norm(lab_ab(out) - lab_ab(x), axis=-1)[hl].mean())
    if with_niqe:
        d["niqe"] = niqe(out)
    return d


def prepare(items, enh):
    out = []
    for it in items:
        x = limit_size(read_image(it["low"]), v1.MAX_SIDE)
        gt = None
        if it.get("gt"):
            gt = read_image(it["gt"])
            if gt.shape[:2] != x.shape[:2]:
                gt = cv2.resize(gt, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
        out.append({"id": it["id"], "tag": it["tag"], "x": x, "gt": gt, "r": enh(x)[0]})
    return out


def methods(cfg):
    qgh = QualityGuidedHybrid(None, load_qg_config())
    return {
        "standard": lambda p: p["r"],
        "qgh": lambda p: qgh.refine(p["r"], p["x"]).image,
        "standard+hue": lambda p: highlight_hue_fix(p["r"], p["x"], cfg)[0],
        "qgh+hue": lambda p: qgh.refine(highlight_hue_fix(p["r"], p["x"], cfg)[0], p["x"]).image,
    }


# ----------------------------------------------------------------------------- search (dev only)
GRID = {"start": [0.80, 0.88, 0.92], "full": [0.95, 0.98], "chroma_strength": [0.7, 1.0], "lum_keep": [0.0, 0.5, 0.8, 1.0]}


def search(args):
    from lumen.deep import DeepEnhancer

    enh = DeepEnhancer("retinexformer")
    sets = {n: prepare(load_set(n), enh) for n in ("hc_dev", "dev_real", "dev_syn")}
    qgh = QualityGuidedHybrid(None, load_qg_config())
    base = {}
    for n, P in sets.items():
        base[n] = []
        for p in P:
            o = qgh.refine(p["r"], p["x"]).image
            base[n].append((psnr(o, p["gt"]), ssim(o, p["gt"]), lpips_distance(o, p["gt"])))
    # cache QGH(R) per config is cheap; evaluate on the "qgh+hue" pipeline (what the Highlight-safe style would ship)
    log = []
    best = None
    for st, fu, cs, lk in product(*GRID.values()):
        if fu <= st:
            continue
        cfg = HighlightHueConfig(start=st, full=fu, chroma_strength=cs, lum_keep=lk)
        res = {}
        for n, P in sets.items():
            rows = []
            for p, b in zip(P, base[n]):
                o = qgh.refine(highlight_hue_fix(p["r"], p["x"], cfg)[0], p["x"]).image
                d = metrics(o, p, False)
                rows.append((d["psnr"] - b[0], d["ssim"] - b[1], d["lpips"] - b[2], d.get("hl_dE")))
            a = np.array([[v if v is not None else np.nan for v in r] for r in rows], dtype=float)
            res[n] = {"dpsnr": float(np.nanmean(a[:, 0])), "dssim": float(np.nanmean(a[:, 1])), "dlpips": float(np.nanmean(a[:, 2])),
                      "worst_dssim": float(np.nanmin(a[:, 1])), "hl_dE": float(np.nanmean(a[:, 3])) if not np.isnan(a[:, 3]).all() else None}
        # objective: lowest highlight colour error on the high-contrast dev set, subject to no regression elsewhere
        feas = all(res[n]["dpsnr"] >= -0.02 and res[n]["dssim"] >= -0.001 and res[n]["dlpips"] <= 0.001 for n in ("dev_real", "dev_syn")) \
            and res["hc_dev"]["dlpips"] <= 0.0 and res["hc_dev"]["dpsnr"] >= 0.0
        key = (not feas, res["hc_dev"]["hl_dE"] if res["hc_dev"]["hl_dE"] is not None else 1e9)
        log.append({"cfg": cfg.as_dict(), "feasible": feas, **res})
        print(f"{cfg}  hc dE {res['hc_dev']['hl_dE']:.2f} dPSNR {res['hc_dev']['dpsnr']:+.2f} dLPIPS {res['hc_dev']['dlpips']:+.4f} | "
              f"real dPSNR {res['dev_real']['dpsnr']:+.3f} syn dPSNR {res['dev_syn']['dpsnr']:+.3f} feasible={feas}", flush=True)
        if best is None or key < best[0]:
            best = (key, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_search.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    import yaml

    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(yaml.safe_dump({"meta": {"status": "experimental - branch experiment/highlight-hue, not used by any production mode",
                                                 "selected_on": "hc_dev (50 synthetic high-contrast LOL-v2 Train photos), dev_real, dev_syn; no held-out set used",
                                                 "objective": "min highlight a*b* error vs GT on hc_dev, subject to hc_dev dPSNR>=0 and dLPIPS<=0, and dev_real/dev_syn dPSNR>=-0.02, dSSIM>=-0.001, dLPIPS<=+0.001 (vs production QGH)"},
                                        "params": best[1].as_dict()}, sort_keys=False), encoding="utf-8")
    print("BEST", best[1], "feasible" if not best[0][0] else "INFEASIBLE")


def load_cfg():
    import yaml

    return HighlightHueConfig(**yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))["params"])


# ----------------------------------------------------------------------------- eval
EVAL_SETS = ("hc_eval", "eval15", "syn_noisy", "syn_clean", "lolv2_syn", "examples", "unseen")


def evaluate(args):
    from lumen.deep import DeepEnhancer

    enh, cfg = DeepEnhancer("retinexformer"), load_cfg()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        P = prepare(load_set(name), enh)
        ms, rows = methods(cfg), {}
        for m, fn in ms.items():
            per, ts = [], []
            for p in P:
                t0 = time.perf_counter(); o = fn(p); ts.append((time.perf_counter() - t0) * 1000)
                per.append({"id": p["id"], "tag": p["tag"], **metrics(o, p, with_niqe=True)})
            rows[m] = {"per_image": per, "ms": float(np.median(ts))}
        t0 = [];
        for p in P[:10]:
            s = time.perf_counter(); highlight_hue_fix(p["r"], p["x"], cfg); t0.append((time.perf_counter() - s) * 1000)
        rows["_hue_ms"] = float(np.median(t0))
        (OUT / f"eval_{name}.json").write_text(json.dumps({"set": name, "cfg": cfg.as_dict(), "methods": rows}, indent=1), encoding="utf-8")
        print(f"  {name}: done ({len(P)} images)", flush=True)


def report(_a):
    cols = [("psnr", "PSNR ↑", 2), ("ssim", "SSIM ↑", 4), ("lpips", "LPIPS ↓", 4), ("niqe", "NIQE ↓", 2), ("hl_dE", "Highlight a*b* err ↓", 2),
            ("hl_Y_gap", "Highlight Y gap ↓", 3), ("hl_dE_vs_input", "Highlight a*b* vs input ↓", 2)]
    lines = ["# Highlight hue study - held-out tables (scripts/highlight_hue_study.py)\n"]
    for f in sorted(OUT.glob("eval_*.json")):
        r = json.loads(f.read_text()); rows = r["methods"]
        std = {p["id"]: p for p in rows["standard"]["per_image"]}
        tags = sorted({p["tag"] for p in rows["standard"]["per_image"]})
        for tag in ([None] + (tags if len(tags) > 1 else [])):
            sel = lambda per: [p for p in per if tag is None or p["tag"] == tag]  # noqa: E731
            n = len(sel(rows["standard"]["per_image"]))
            keys = [c for c in cols if any(c[0] in p for p in rows["standard"]["per_image"])]
            lines += [f"\n## {r['set']}" + (f" - {tag}" if tag else "") + f" (n = {n})\n",
                      "| Method | " + " | ".join(c[1] for c in keys) + " | worst ΔSSIM vs standard | refine ms |", "|---|" + "---|" * (len(keys) + 2)]
            for m, v in rows.items():
                if m.startswith("_"):
                    continue
                per = sel(v["per_image"])
                cells = []
                for k, _, pr in keys:
                    vals = [p[k] for p in per if p.get(k) is not None]
                    cells.append(f"{np.mean(vals):.{pr}f}" if vals else "-")
                ds = [p["ssim"] - std[p["id"]]["ssim"] for p in per if "ssim" in p]
                lines.append(f"| {m} | " + " | ".join(cells) + f" | {min(ds):+.4f} |" .replace("| +", "| +") if ds else f"| {m} | " + " | ".join(cells) + " | - |")
                lines[-1] += f" {v['ms']:.0f} |"
        lines.append(f"\nHue fix alone: {rows['_hue_ms']:.0f} ms median.\n")
    (OUT / "eval_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "eval_tables.md")


# ----------------------------------------------------------------------------- figures
def figures(args):
    from lumen.deep import DeepEnhancer

    enh, cfg = DeepEnhancer("retinexformer"), load_cfg()
    ms = methods(cfg)
    items = [("hc_eval", None), ("hc_eval", None), ("hc_eval", None), ("examples", "rocket_night_highlights"), ("examples", "astronaut_uneven_light")]
    fig = OUT / "figures"
    for k, (name, iid) in enumerate(items):
        pool = load_set(name)
        it = next((i for i in pool if i["id"] == iid), pool[[3, 11, 27][k]] if iid is None else pool[0])
        p = prepare([it], enh)[0]
        ims = [p["x"], ms["qgh"](p), ms["qgh+hue"](p)] + ([p["gt"]] if p["gt"] is not None else [])
        names = ["Input", "Highlight-safe (production)", "Highlight-safe + hue fix"] + (["Ground truth"] if p["gt"] is not None else [])
        write_image(fig / f"{name}_{it['id']}.png", v1._panel(ims, names).astype(np.float32) / 255)
        print("wrote", fig / f"{name}_{it['id']}.png")
    # the user's photo (no ground truth) if present
    up = Path(r"C:\Users\Bhyresh BS\Music\test\bad-lighting.webp")
    if up.exists():
        x = limit_size(read_image(up), 768); p = {"x": x, "gt": None, "r": enh(x)[0]}
        write_image(fig / "user_bad_lighting.png", v1._panel([x, ms["standard"](p), ms["qgh+hue"](p)], ["Input", "Natural", "Highlight-safe + hue fix"]).astype(np.float32) / 255)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build"); sub.add_parser("search"); sub.add_parser("report"); sub.add_parser("figures")
    e = sub.add_parser("eval"); e.add_argument("--sets", nargs="+", default=list(EVAL_SETS))
    args = ap.parse_args()
    {"build": build, "search": search, "eval": evaluate, "report": report, "figures": figures}[args.cmd](args)


if __name__ == "__main__":
    main()
