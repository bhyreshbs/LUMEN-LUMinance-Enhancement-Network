"""Noise study, iteration 2 (QGH-NA v2): one controlled fix iteration. Experimental only.

    python scripts/noise_study_v2.py build     # fine-detail dev set + a NEW untouched validation set
    python scripts/noise_study_v2.py search    # DEV ONLY, starting from the frozen v1 config -> noise_aware_v2.yaml
    python scripts/noise_study_v2.py eval      # every method on identical validation inputs, once
    python scripts/noise_study_v2.py figures

What v2 is allowed to change (everything else is the frozen v1 config, configs/experimental/noise_aware.yaml):
  * image-level noise-vs-texture gate  -> exact identity when no correction is justified
  * chroma blend mask from *luma noise* only (no chroma filtering on clean regions)
  * method-noise test: removed residual that is structured (strokes, grain lines) is put back
  * exact identity outside the refinement masks (no YCbCr round-trip drift)

Data roles (iteration 1 sets are NOT used to claim anything for v2: their clean set was inspected to diagnose
the handwriting failure):
  dev   dev_real (269, as before), dev_syn (90, as before), dev_detail (NEW: handwriting, printed text,
        wood, fabric, scikit-image page/brick/grass, high-texture LOL-v2 Synthetic Train photos; each scene
        noise-free and noisy)
  val   val_detail (NEW, disjoint scenes / fonts / seeds / words: same categories, scikit-image
        text/gravel/stereo_motorcycle), val_general (NEW: 60 LOL-v2 Synthetic Train photos never used),
        val_real (LOL-v2 real Test pairs that are not copies of LOL-v1 or of the dev split), eval15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import noise_study as v1  # noqa: E402
from lumen.experimental.noise_aware import (  # noqa: E402
    NoiseAwareConfig, NoiseAwareQGH, estimate_sigma_map, load_na_config, noise_aware_denoise, rgb_to_ycc,
)
from lumen.hybrid.quality_guided import QualityGuidedHybrid, load_qg_config  # noqa: E402
from lumen.metrics.quality import lpips_distance, psnr, ssim  # noqa: E402
from lumen.utils import limit_size, read_image  # noqa: E402

DATA, SPLITS = v1.DATA, v1.SPLITS
NS2 = DATA / "noise_study_v2"
OUT = ROOT / "results" / "noise_study_v2"
V2_CONFIG = ROOT / "configs" / "experimental" / "noise_aware_v2.yaml"

WORDS = ("light dark evening river window letter morning garden silver quiet winter paper candle shadow "
         "harbour orchard lantern meadow copper journey thunder violet amber whisper station travel autumn "
         "marble forest hollow crimson velvet ember willow granite ribbon falcon cobalt saffron").split()


# ============================================================================ procedural scenes
def paper(h, w, rng):
    base = np.array(rng.choice([[0.93, 0.89, 0.78], [0.96, 0.95, 0.91], [0.88, 0.80, 0.62]]), np.float32)
    illum = cv2.GaussianBlur(rng.normal(0, 1, (h // 16 + 1, w // 16 + 1)).astype(np.float32), (0, 0), 2)
    illum = cv2.resize(illum, (w, h)) * 0.04
    fibre = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8) * 0.015
    return np.clip(base * (1 + illum[..., None]) + fibre[..., None], 0, 1)


def cursive_word(mask, x, y, rng, size):
    """Pseudo-cursive pen stroke: a trochoid (looping baseline curve) with per-letter heights/widths."""
    n = int(rng.integers(3, 9))
    pts, cx = [], float(x)
    for _ in range(n):
        a = size * rng.uniform(0.35, 0.6)                 # loop radius
        v = a * rng.uniform(0.3, 0.6)                    # forward speed (< a -> loops)
        hgt = size * (rng.uniform(1.6, 2.6) if rng.random() < 0.25 else rng.uniform(0.7, 1.1))  # ascenders
        for t in np.linspace(0, 2 * np.pi, 18, endpoint=False):
            pts.append((cx + v * t - a * np.sin(t), y - hgt * 0.5 * (1 - np.cos(t))))
        cx += v * 2 * np.pi
    pts = np.array(pts, np.float32)
    pts[:, 1] += np.cumsum(rng.normal(0, 0.15, len(pts)))  # slight baseline wander
    cv2.polylines(mask, [np.round(pts * 4).astype(np.int32)], False, 255, int(rng.choice([1, 1, 2])),
                  cv2.LINE_AA, shift=2)
    return cx - x


def handwriting(rng, fonts, h=384, w=512, cursive=0.0):
    img = paper(h, w, rng)
    ink = np.array(rng.choice([[0.25, 0.15, 0.08], [0.10, 0.15, 0.45], [0.08, 0.08, 0.08]]), np.float32)
    strength = rng.uniform(0.25, 0.9)  # faint to strong ink
    mask = np.zeros((h, w), np.uint8)  # putText needs 8-bit
    y = int(rng.uniform(28, 40))
    while y < h - 10:
        x = int(rng.uniform(8, 30))
        line_cursive = rng.random() < cursive
        while x < w - 60:
            if line_cursive:
                x += int(cursive_word(mask, x, y, rng, rng.uniform(7, 12)) + rng.uniform(10, 20))
                continue
            word = str(rng.choice(WORDS))
            font = int(rng.choice(fonts))
            scale, thick = rng.uniform(0.55, 1.1), int(rng.choice([1, 1, 2]))
            cv2.putText(mask, word, (x, y + int(rng.normal(0, 1.5))), font, scale, 255, thick, cv2.LINE_AA)
            x += int(cv2.getTextSize(word, font, scale, thick)[0][0] + rng.uniform(10, 22))
        y += int(rng.uniform(26, 40))
    mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.5) * strength
    return np.clip(img * (1 - mask[..., None]) + ink * mask[..., None], 0, 1)


def wood(rng, h=384, w=512):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warp = cv2.resize(rng.normal(0, 1, (h // 32 + 2, w // 32 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    x0, y0 = rng.uniform(-w, 2 * w), rng.uniform(-3 * h, -h)
    d = np.sqrt((xx - x0) ** 2 + ((yy - y0) * rng.uniform(0.15, 0.4)) ** 2) + 6 * warp
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * d / rng.uniform(7, 16))
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.6, sigmaY=6.0) * 0.08
    v = np.clip(0.55 * rings ** 2 + 0.45 + grain, 0, 1)
    c1, c2 = np.array([0.36, 0.20, 0.09], np.float32), np.array([0.78, 0.55, 0.32], np.float32)
    return np.clip(c1 + (c2 - c1) * v[..., None], 0, 1)


def fabric(rng, h=384, w=512):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p = rng.uniform(3.0, 6.0)
    warp_t = 0.5 + 0.5 * np.sin(2 * np.pi * xx / p)
    weft_t = 0.5 + 0.5 * np.sin(2 * np.pi * yy / p)
    over = ((np.floor(xx / p) + np.floor(yy / p)) % 2).astype(np.float32)
    v = over * warp_t + (1 - over) * weft_t
    v = v + cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.7) * 0.06
    c1 = rng.uniform(0.1, 0.5, 3).astype(np.float32)
    c2 = np.clip(c1 + rng.uniform(0.2, 0.4), 0, 1)
    return np.clip(c1 + (c2 - c1) * np.clip(v, 0, 1)[..., None], 0, 1)


def gray_tinted(img_u8, tint, size=512):
    g = img_u8.astype(np.float32) / 255.0
    g = cv2.resize(g, None, fx=min(1.0, size / max(g.shape)), fy=min(1.0, size / max(g.shape)),
                   interpolation=cv2.INTER_AREA)
    return np.clip(g[..., None] * np.array(tint, np.float32), 0, 1)


def texture_energy(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())


# ============================================================================ build
def _save(path: Path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92] if path.suffix == ".jpg" else [])


def _pairs_for(scenes, root: Path, rng, lines, ranges):
    """Each scene -> a noise-free and a noisy low-light version (same exposure)."""
    for name, cat, gt in scenes:
        gt_u8 = (np.clip(gt, 0, 1) * 255 + 0.5).astype(np.uint8)
        gt_path = root / "gt" / f"{name}.png"
        _save(gt_path, gt_u8)
        e_lo, e_hi, rn_lo, rn_hi, ph_lo, ph_hi = ranges
        p = dict(exposure=rng.uniform(e_lo, e_hi), side=rng.random() < 0.2, read_noise=rng.uniform(rn_lo, rn_hi),
                 photons=float(np.exp(rng.uniform(np.log(ph_lo), np.log(ph_hi)))))
        for tag, noisy in (("clean", False), ("noisy", True)):
            lo = root / tag / f"{name}.jpg"
            _save(lo, v1.simulate(gt_u8, **p, rng=rng, noisy=noisy))
            lines.append(f"{v1.rel(lo)}\t{v1.rel(gt_path)}\t{cat}_{tag}\n")


def _load_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def build(_args):
    from skimage import data as sk
    from train_retinexformer import fingerprint

    train = sorted((DATA / "LOL-v2/Synthetic/Train/Normal").glob("*.png"))
    used = {Path(ln.split("\t")[1]).name for ln in (SPLITS / "noise_study_dev_syn.txt").read_text().splitlines() if ln}
    test = sorted((DATA / "LOL-v2/Synthetic/Test/Normal").glob("*.png"))  # inspected in iteration 1
    pool = [p for p in train if p.name not in used]
    # near-duplicate guard: drop pool images within tol of the dev_syn images or of the (inspected) test set
    ref = np.stack([fingerprint(p) for p in [t for t in train if t.name in used] + test])
    fp = np.stack([fingerprint(p) for p in pool])
    d = np.abs(fp[:, None] - ref[None]).mean(-1).min(1)
    pool = [p for p, di in zip(pool, d) if di >= 0.01]
    energy = {p: texture_energy(p) for p in pool}
    textured = sorted(pool, key=lambda p: -energy[p])[:30]  # most textured -> foliage/fabric/wood-like photos
    rest = [p for p in pool if p not in textured]
    rng = np.random.default_rng(99)
    general_val = list(rng.choice(rest, 60, replace=False))

    def scenes(split):
        r = np.random.default_rng(11 if split == "dev" else 23)
        hw_fonts = ([cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX | cv2.FONT_ITALIC]
                    if split == "dev" else [cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC])
        pr_fonts = [cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_SIMPLEX] if split == "dev" else \
            [cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]
        out = [(f"{split}_hand{i:02d}", "hand", handwriting(r, hw_fonts, cursive=0.6)) for i in range(12)]
        out += [(f"{split}_print{i:02d}", "print", handwriting(r, pr_fonts)) for i in range(4)]
        out += [(f"{split}_wood{i:02d}", "wood", wood(r)) for i in range(4)]
        out += [(f"{split}_fabric{i:02d}", "fabric", fabric(r)) for i in range(4)]
        if split == "dev":
            out += [("dev_sk_page", "hand", gray_tinted(sk.page(), (0.95, 0.88, 0.72))),
                    ("dev_sk_brick", "texture", gray_tinted(sk.brick(), (0.85, 0.55, 0.45))),
                    ("dev_sk_grass", "foliage", gray_tinted(sk.grass(), (0.55, 0.85, 0.40)))]
            photos = textured[0::2]
        else:
            out += [("val_sk_text", "print", gray_tinted(sk.text(), (0.95, 0.92, 0.85))),
                    ("val_sk_gravel", "texture", gray_tinted(sk.gravel(), (0.80, 0.75, 0.65))),
                    ("val_sk_motorcycle", "photo", sk.stereo_motorcycle()[0].astype(np.float32) / 255)]
            photos = textured[1::2]
        out += [(f"{split}_photo_{p.stem}", "photo", _load_rgb(p)) for p in photos]
        return out

    for split, ranges, seed in (("dev", (0.04, 0.15, 0.002, 0.008, 500, 4000), 5),
                                ("val", (0.03, 0.18, 0.001, 0.010, 400, 5000), 77)):
        lines = []
        _pairs_for(scenes(split), NS2 / f"{split}_detail", np.random.default_rng(seed), lines, ranges)
        (SPLITS / f"noise_study_v2_{split}_detail.txt").write_text("".join(lines), encoding="utf-8")
        print(f"{split}_detail: {len(lines)} images")

    lines = []
    rng = np.random.default_rng(1234)
    for i, gt in enumerate(general_val):
        src = cv2.cvtColor(cv2.imread(str(gt)), cv2.COLOR_BGR2RGB)
        noisy = i % 2 == 0
        p = dict(exposure=rng.uniform(0.03, 0.18), side=rng.random() < 0.25, read_noise=rng.uniform(0.001, 0.01),
                 photons=float(np.exp(rng.uniform(np.log(400), np.log(5000)))))
        lo = NS2 / "val_general" / ("noisy" if noisy else "clean") / f"{gt.stem}.jpg"
        _save(lo, v1.simulate(src, **p, rng=rng, noisy=noisy))
        lines.append(f"{v1.rel(lo)}\t{v1.rel(gt)}\t{'general_noisy' if noisy else 'general_clean'}\n")
    (SPLITS / "noise_study_v2_val_general.txt").write_text("".join(lines), encoding="utf-8")
    print(f"val_general: {len(lines)} images")

    # real: LOL-v2 real Test pairs that are not (near-)copies of LOL-v1 our485/eval15 or of the dev split
    from train_retinexformer import pair_dir as pd_

    cand = pd_(DATA / "LOL-v2/Real_captured/Test/Low", DATA / "LOL-v2/Real_captured/Test/Normal")
    seen = pd_(DATA / "LOL/our485/low", DATA / "LOL/our485/high") + pd_(DATA / "LOL/eval15/low", DATA / "LOL/eval15/high")
    seen += [(ROOT / a, ROOT / b) for a, b in v1._pairs(SPLITS / "dev_lolv2_real_unseen.txt")]

    def fpp(pairs):
        return np.stack([np.concatenate([fingerprint(lo), fingerprint(hi)]) for lo, hi in pairs])

    dd = np.abs(fpp(cand)[:, None] - fpp(seen)[None]).mean(-1).min(1)
    keep = [p for p, di in zip(cand, dd) if di >= 0.01]
    (SPLITS / "noise_study_v2_val_real.txt").write_text(
        "".join(f"{v1.rel(lo)}\t{v1.rel(hi)}\treal\n" for lo, hi in keep), encoding="utf-8")
    print(f"val_real: {len(keep)} of {len(cand)} LOL-v2 real Test pairs are untouched")


# ============================================================================ sets
def load_set(name):
    if name in ("dev_real", "dev_syn", "eval15"):
        return v1.load_set(name)
    f = {"dev_detail": "noise_study_v2_dev_detail.txt", "val_detail": "noise_study_v2_val_detail.txt",
         "val_general": "noise_study_v2_val_general.txt", "val_real": "noise_study_v2_val_real.txt"}[name]
    return [dict(id=Path(lo).parent.name + "/" + Path(lo).stem, low=ROOT / lo, gt=ROOT / hi, clean_low=None, tag=tag)
            for lo, hi, tag in v1._pairs(SPLITS / f)]


def prepare_light(items, enh, desc=""):
    """Memory-light cache: model output float16, ground truth uint8, input not kept."""
    out = []
    t0 = time.perf_counter()
    for it in items:
        x = limit_size(read_image(it["low"]), v1.MAX_SIDE)
        gt = read_image(it["gt"])
        if gt.shape[:2] != x.shape[:2]:
            gt = cv2.resize(gt, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
        r, _ = enh(x)
        s0 = estimate_sigma_map(np.ascontiguousarray(rgb_to_ycc(r)[..., 0]))[1]
        out.append({"id": it["id"], "tag": it["tag"], "r16": r.astype(np.float16),
                    "gt8": (gt * 255 + 0.5).astype(np.uint8), "sigma0": s0})
    print(f"  {desc}: {len(out)} images, model {time.perf_counter() - t0:.0f}s", flush=True)
    return out


# ============================================================================ search (dev only)
CLEAN_LIKE = ("clean", "low_noise", "detail_clean")
NOISY_LIKE = ("real", "noisy", "detail_noisy")
TOL_MEAN = {"psnr": -0.02, "ssim": -0.001, "lpips": 0.001}     # clean-like groups, mean
TOL_WORST = {"ssim": -0.005, "lpips": 0.01}                     # clean-like groups, every image
TOL_DETAIL_NOISY_WORST = {"lpips": 0.02}                        # noisy fine-detail: no erasing
SEARCH_V2 = {
    "texture_relief": [16.0, 0.0, 4.0, 32.0, 64.0],
    "chroma_mask": ["luma_noise", "max"],
    "image_gate": [0.0, 0.006, 0.008, 0.010, 0.012, 0.016],
    "texture_gate": [0.0, 0.35, 0.40, 0.45, 0.50],
    "residual_protect": [0.0, 0.5, 0.75, 1.0],
    "residual_ramp": [(1.5, 3.0), (2.0, 4.0), (3.0, 6.0)],
    "luma_k": [1.0, 1.5, 2.0, 3.0],
    "luma_floor": [0.008, 0.010, 0.012],
}


def _group(tag, sigma0):
    g = ["detail_" + tag.rsplit("_", 1)[1]] if tag.endswith(("_clean", "_noisy")) and tag.split("_")[0] != "general" \
        else [tag]
    if sigma0 < v1.LOW_NOISE_SIGMA:
        g.append("low_noise")
    return g


def dev_eval(cfg, dev, base, pool):
    def one(it):
        r = it["r16"].astype(np.float32)
        gt = it["gt8"].astype(np.float32) / 255
        out, _, info = noise_aware_denoise(r, cfg)
        return psnr(out, gt), ssim(out, gt), out, gt, bool(np.array_equal(out, r))

    res = list(pool.map(one, dev))
    rows = []
    for (p, s, out, gt, ident), it, b in zip(res, dev, base):
        lp = b["lpips"] if ident else lpips_distance(out, gt)
        rows.append({"groups": _group(it["tag"], it["sigma0"]), "psnr": p - b["psnr"], "ssim": s - b["ssim"],
                     "lpips": lp - b["lpips"], "lpips_abs": lp, "identity": ident})
    groups = {}
    for r in rows:
        for g in r["groups"]:
            groups.setdefault(g, []).append(r)
    s = {g: {k: float(np.mean([v[k] for v in vs])) for k in ("psnr", "ssim", "lpips", "lpips_abs", "identity")}
         | {"n": len(vs)} for g, vs in groups.items()}
    viol, mag = [], 0.0
    for g in NOISY_LIKE:
        if g in s and (s[g]["psnr"] < 0 or s[g]["ssim"] < 0):
            viol.append(f"{g}: mean dPSNR/dSSIM < 0")
            mag += max(0, -s[g]["psnr"]) * 0.1 + max(0, -s[g]["ssim"]) * 5
    for g in CLEAN_LIKE:
        if g not in s:
            continue
        c = s[g]
        m = (max(0, TOL_MEAN["psnr"] - c["psnr"]) * 0.1 + max(0, TOL_MEAN["ssim"] - c["ssim"]) * 5
             + max(0, c["lpips"] - TOL_MEAN["lpips"]) * 5)
        worst_s = min(v["ssim"] for v in groups[g])
        worst_l = max(v["lpips"] for v in groups[g])
        m += max(0, TOL_WORST["ssim"] - worst_s) * 2 + max(0, worst_l - TOL_WORST["lpips"]) * 2
        if m > 0:
            viol.append(f"{g} degraded (worst dSSIM {worst_s:.4f}, worst dLPIPS {worst_l:.4f})")
            mag += m
    if "detail_noisy" in groups:
        worst_l = max(v["lpips"] for v in groups["detail_noisy"])
        if worst_l > TOL_DETAIL_NOISY_WORST["lpips"]:
            viol.append(f"detail_noisy worst dLPIPS {worst_l:.4f}")
            mag += (worst_l - TOL_DETAIL_NOISY_WORST["lpips"]) * 2
    objective = float(np.mean([r["lpips_abs"] for r in rows]))
    return {"objective": objective, "feasible": not viol, "violations": viol, "violation": mag, "groups": s}


def search(args):
    from lumen.deep import DeepEnhancer

    cv2.setNumThreads(1)
    enh = DeepEnhancer("retinexformer")
    dev = []
    for name in ("dev_real", "dev_syn", "dev_detail"):
        dev += prepare_light(load_set(name), enh, name)
    base = []
    for it in dev:
        r, gt = it["r16"].astype(np.float32), it["gt8"].astype(np.float32) / 255
        base.append({"psnr": psnr(r, gt), "ssim": ssim(r, gt), "lpips": lpips_distance(r, gt)})
    pool = ThreadPoolExecutor(args.workers)
    v1_cfg = load_na_config()
    log = []

    def score(c):
        t0 = time.perf_counter()
        r = dev_eval(c, dev, base, pool)
        log.append({"config": c.as_yaml_dict(), **r, "sec": round(time.perf_counter() - t0, 1)})
        return r["objective"] + (0 if r["feasible"] else 0.1 + 2 * r["violation"]), r

    r_v1 = score(v1_cfg)[1]
    print(f"v1 on new dev: objective {r_v1['objective']:.4f} feasible={r_v1['feasible']} {r_v1['violations']}")
    # start: v1 + exact identity + the image gates and texture relief found on dev in the first (stopped) pass
    cfg = replace(v1_cfg, chroma_mask="luma_noise", exact_identity=True, image_gate=0.016, texture_gate=0.35,
                  texture_relief=16.0)
    best_key, best_r = score(cfg)
    print(f"v2 start: objective {best_r['objective']:.4f} feasible={best_r['feasible']} {best_r['violations']}", flush=True)
    for sweep in range(args.sweeps):
        changed = False
        for field, values in SEARCH_V2.items():
            for v in values:
                if getattr(cfg, field) == (tuple(v) if isinstance(v, tuple) else v):
                    continue
                cand = replace(cfg, **{field: v})
                key, r = score(cand)
                if key < best_key - 1e-5:
                    best_key, best_r, cfg, changed = key, r, cand, True
            print(f"sweep {sweep} {field:16s} -> {getattr(cfg, field)!s:12s} objective {best_r['objective']:.4f} "
                  f"feasible={best_r['feasible']} {best_r['violations']}", flush=True)
        if not changed:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_search.json").write_text(json.dumps({"v1_on_new_dev": r_v1, "evaluations": log,
                                                     "best": {"config": cfg.as_yaml_dict(), **best_r}}, indent=1))
    import yaml

    meta = {"created": time.strftime("%Y-%m-%d"), "tool": "scripts/noise_study_v2.py search",
            "status": "experimental - not used by any production mode", "base": "configs/experimental/noise_aware.yaml (v1)",
            "dev": ["dev_lolv2_real_unseen (269)", "noise_study_dev_syn (90)", "noise_study_v2_dev_detail"],
            "objective": "min mean LPIPS on all dev; noisy groups mean dPSNR, dSSIM >= 0; clean-like groups "
                         f"(noise-free, low-noise, noise-free fine detail) mean tolerance {TOL_MEAN}, every image "
                         f"{TOL_WORST}; noisy fine detail every image dLPIPS <= {TOL_DETAIL_NOISY_WORST['lpips']}",
            "evaluations": len(log), "dev_result": best_r["groups"], "validation_sets_used_for_selection": "none"}
    V2_CONFIG.write_text(yaml.safe_dump({"meta": meta, "params": cfg.as_yaml_dict()}, sort_keys=False), encoding="utf-8")
    print(f"best {cfg}\n{best_r['groups']}\nwrote {V2_CONFIG}")


# ============================================================================ eval (validation, once)
VAL_SETS = ("val_detail", "val_general", "val_real", "eval15")


def evaluate(args):
    from lumen.deep import DeepEnhancer

    enh = DeepEnhancer("retinexformer")
    qg = load_qg_config()
    na1, na2 = NoiseAwareQGH(None, qg, load_na_config()), NoiseAwareQGH(None, qg, load_na_config(V2_CONFIG))
    qgh = QualityGuidedHybrid(None, qg)
    methods = {"input": lambda it: it["x"], "standard": lambda it: it["r"],
               "qgh": lambda it: qgh.refine(it["r"], it["x"]).image,
               "qgh_na_v1": lambda it: na1.refine(it["r"], it["x"]).image,
               "qgh_na_v2": lambda it: na2.refine(it["r"], it["x"]).image}
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        prepared = v1.prepare(load_set(name), enh, name)
        rows = {}
        for m, fn in methods.items():
            per, t_ms = [], []
            for it in prepared:
                t0 = time.perf_counter()
                out = fn(it)
                t_ms.append((time.perf_counter() - t0) * 1000)
                d = v1.image_metrics(out, it, full=False)
                d["identical_to_qgh_highlight"] = None
                per.append({"id": it["id"], "tag": it["tag"], **d})
            rows[m] = {"per_image": per, "refine_ms_median": float(np.median(t_ms))}
        # exact-identity bookkeeping: v2 == QGH highlight-only path, bit for bit?
        hl = QualityGuidedHybrid(None, replace(qg, enabled=tuple(c for c in qg.enabled if c != "noise")))
        for it, p in zip(prepared, rows["qgh_na_v2"]["per_image"]):
            p["identical_to_qgh_highlight"] = bool(np.array_equal(na2.refine(it["r"], it["x"]).image,
                                                                  hl.refine(it["r"], it["x"]).image))
        (OUT / f"val_{name}_results.json").write_text(json.dumps({"set": name, "methods": rows}, indent=1))
        print(f"  {name}: done", flush=True)


def report(_args):
    lines = ["# Noise study v2 - validation tables (scripts/noise_study_v2.py)\n"]
    cols = [("psnr", "PSNR ↑", 2), ("ssim", "SSIM ↑", 4), ("lpips", "LPIPS ↓", 4), ("edge", "Edge ↑", 4),
            ("colour_cast", "Cast ↓", 2), ("sigma_y_est", "σY est ↓", 4)]
    for f in sorted(OUT.glob("val_*_results.json")):
        r = json.loads(f.read_text())
        name, rows = r["set"], r["methods"]
        tags = sorted({p["tag"] for p in rows["standard"]["per_image"]})
        subsets = [("all", None)] + ([(t, t) for t in tags] if len(tags) > 1 else [])
        std = {p["id"]: p for p in rows["standard"]["per_image"]}
        for label, tag in subsets:
            sel = lambda per: [p for p in per if tag is None or p["tag"] == tag]  # noqa: E731
            n = len(sel(rows["standard"]["per_image"]))
            lines.append(f"\n## {name}" + ("" if tag is None else f" - {label}") + f" (n = {n})\n")
            lines.append("| Method | " + " | ".join(c[1] for c in cols) +
                         " | worst ΔSSIM vs std | worst ΔLPIPS vs std | images better / worse (LPIPS ±0.005) | Refine ms |")
            lines.append("|---|" + "---|" * (len(cols) + 4))
            for m, v in rows.items():
                per = sel(v["per_image"])
                cells = [f"{np.mean([p[k] for p in per]):.{prec}f}" for k, _, prec in cols]
                dl = [p["lpips"] - std[p["id"]]["lpips"] for p in per]
                ds = [p["ssim"] - std[p["id"]]["ssim"] for p in per]
                bw = f"{sum(x < -0.005 for x in dl)} / {sum(x > 0.005 for x in dl)}"
                lines.append(f"| {m} | " + " | ".join(cells) + f" | {min(ds):+.4f} | {max(dl):+.4f} | {bw} | "
                             f"{v['refine_ms_median']:.0f} |")
        ident = [p["identical_to_qgh_highlight"] for p in rows["qgh_na_v2"]["per_image"]]
        clean_ident = [p["identical_to_qgh_highlight"] for p in rows["qgh_na_v2"]["per_image"]
                       if p["tag"].endswith("clean")]
        lines.append(f"\nv2 bit-exact identical to the QGH highlight-only path: {sum(ident)} / {len(ident)} images"
                     + (f" (noise-free inputs: {sum(clean_ident)} / {len(clean_ident)})" if clean_ident else "") + "\n")
    (OUT / "val_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'val_tables.md'}")


def figures(args):
    from lumen.deep import DeepEnhancer

    enh = DeepEnhancer("retinexformer")
    qg = load_qg_config()
    na1, na2 = NoiseAwareQGH(None, qg, load_na_config()), NoiseAwareQGH(None, qg, load_na_config(V2_CONFIG))
    qgh = QualityGuidedHybrid(None, qg)
    picks = [("val_detail", args.ids)] if args.ids else []
    if not picks:
        items = load_set("val_detail")
        want = ["clean/val_hand00", "noisy/val_hand00", "noisy/val_wood00", "noisy/val_fabric00", "clean/val_sk_text",
                "noisy/val_sk_motorcycle"]
        picks = [("val_detail", want)]
    for set_name, ids in picks:
        items = {i["id"]: i for i in load_set(set_name)}
        for iid in ids:
            p = v1.prepare([items[iid]], enh, iid)[0]
            ims = [p["x"], p["r"], qgh.refine(p["r"], p["x"]).image, na1.refine(p["r"], p["x"]).image,
                   na2.refine(p["r"], p["x"]).image, p["gt"]]
            names = ["Input", "Standard", "QGH", "QGH-NA v1", "QGH-NA v2", "Ground truth"]
            full = v1._panel(ims, names)
            h, w = p["x"].shape[:2]
            zoom = v1._panel(ims, [n + " (zoom)" for n in names], (h // 3, w // 3, min(128, h // 2), min(128, w // 2)), 3)
            zoom = cv2.resize(zoom, (full.shape[1], round(zoom.shape[0] * full.shape[1] / zoom.shape[1])),
                              interpolation=cv2.INTER_NEAREST)
            fn = OUT / "figures" / f"{iid.replace('/', '_')}.png"
            from lumen.utils import write_image

            write_image(fn, np.concatenate([full, zoom], 0).astype(np.float32) / 255)
            print("wrote", fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("--sweeps", type=int, default=2)
    s.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("eval")
    e.add_argument("--sets", nargs="+", default=list(VAL_SETS))
    sub.add_parser("report")
    f = sub.add_parser("figures")
    f.add_argument("--ids", nargs="*")
    args = ap.parse_args()
    {"build": build, "search": search, "eval": evaluate, "report": report, "figures": figures}[args.cmd](args)


if __name__ == "__main__":
    main()
