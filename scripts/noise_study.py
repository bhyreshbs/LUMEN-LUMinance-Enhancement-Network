"""Noise-aware refinement study (experimental). Production modes, configs and Phase 5 results are untouched.

    python scripts/noise_study.py build     # synthetic noisy / clean sets -> data/noise_study, data/splits
    python scripts/noise_study.py search    # DEV ONLY: select parameters -> configs/experimental/noise_aware.yaml
    python scripts/noise_study.py eval      # frozen config, held-out sets, once -> results/noise_study/
    python scripts/noise_study.py figures   # before/after comparison images -> results/noise_study/figures
    python scripts/noise_study.py latency   # CPU refine latency at 600x400 / 768 / 1024 px

Data roles
  dev   dev_real : data/splits/dev_lolv2_real_unseen.txt (269 real LOL-v2 pairs unseen by the checkpoint;
                   the same split QGH was calibrated on)
        dev_syn  : 90 LOL-v2 *Synthetic Train* normal-light images darkened with the Poisson-Gaussian
                   sensor model of scripts/make_examples.py (45 noisy, 45 noise-free - the clean-image
                   guard against over-smoothing texture)
  eval  eval15   : LOL-v1 eval15 (headline set; never used for selection)
        syn_noisy / syn_clean : the 100 LOL-v2 *Synthetic Test* images darkened with / without noise
                   (different seed and ranges than dev). syn_noisy also has a noise-free twin, so
                   residual noise = output - Retinexformer(noise-free twin) is measured exactly.
        lolv2_syn : LOL-v2 Synthetic Test's own low/normal pairs (relatively clean low light)
        examples : the 4 app examples incl. coffee_dark.jpg, ground truth = scikit-image originals
        unseen   : data/unseen (20 real photos, no ground truth, no-reference metrics only)
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

from lumen.experimental.noise_aware import (  # noqa: E402
    DEFAULT_NA_CONFIG_PATH, NoiseAwareConfig, NoiseAwareQGH, estimate_sigma_map, load_na_config,
    noise_aware_denoise, rgb_to_ycc,
)
from lumen.hybrid.quality_guided import QualityGuidedHybrid, load_qg_config  # noqa: E402
from lumen.metrics.quality import (  # noqa: E402
    colour_shift, edge_preservation, lpips_distance, niqe, noise_level, psnr, ssim,
)
from lumen.utils import limit_size, read_image, to_uint8, write_image  # noqa: E402

DATA = ROOT / "data"
NS = DATA / "noise_study"
SPLITS = DATA / "splits"
OUT = ROOT / "results" / "noise_study"
MAX_SIDE = 1024  # the app service's processing limit


# ============================================================================ build
def simulate(img_u8, exposure, side, read_noise, photons, rng, noisy=True):
    """scripts/make_examples.py sensor model; noisy=False gives the noise-free twin (same exposure)."""
    lin = (img_u8.astype(np.float32) / 255.0) ** 2.2
    illum = np.tile(np.linspace(1.0, 0.15, lin.shape[1], dtype=np.float32), (lin.shape[0], 1))[..., None] \
        if side else 1.0
    lin = lin * exposure * illum
    if noisy:
        lin = np.clip(rng.poisson(lin * photons) / photons + rng.normal(0, read_noise, lin.shape), 0, 1)
    return (np.clip(lin ** (1 / 2.2), 0, 1) * 255 + 0.5).astype(np.uint8)


def _save_jpg(path: Path, rgb_u8):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def build(_args):
    rng = np.random.default_rng(2026)
    train = sorted((DATA / "LOL-v2/Synthetic/Train/Normal").glob("*.png"))
    test = sorted((DATA / "LOL-v2/Synthetic/Test/Normal").glob("*.png"))
    if not train or not test:
        raise SystemExit("Need data/LOL-v2/Synthetic (see data/README.md)")
    lines = []
    for i, gt in enumerate(rng.choice(train, 90, replace=False)):
        src = cv2.cvtColor(cv2.imread(str(gt)), cv2.COLOR_BGR2RGB)
        noisy = i % 2 == 0
        p = dict(exposure=rng.uniform(0.04, 0.15), side=rng.random() < 0.2, read_noise=rng.uniform(0.002, 0.008),
                 photons=float(np.exp(rng.uniform(np.log(500), np.log(4000)))))
        low = simulate(src, **p, rng=rng, noisy=noisy)
        out = NS / "dev_syn" / ("noisy" if noisy else "clean") / f"{gt.stem}.jpg"
        _save_jpg(out, low)
        lines.append(f"{rel(out)}\t{rel(gt)}\t{'noisy' if noisy else 'clean'}\n")
    (SPLITS / "noise_study_dev_syn.txt").write_text("".join(lines), encoding="utf-8")

    rng = np.random.default_rng(7)  # eval: separate seed and slightly wider ranges than dev
    noisy_l, clean_l = [], []
    for gt in test:
        src = cv2.cvtColor(cv2.imread(str(gt)), cv2.COLOR_BGR2RGB)
        p = dict(exposure=rng.uniform(0.03, 0.18), side=rng.random() < 0.25, read_noise=rng.uniform(0.001, 0.01),
                 photons=float(np.exp(rng.uniform(np.log(400), np.log(5000)))))
        n_out, c_out = NS / "eval_syn" / "noisy" / f"{gt.stem}.jpg", NS / "eval_syn" / "clean" / f"{gt.stem}.jpg"
        _save_jpg(n_out, simulate(src, **p, rng=rng, noisy=True))
        _save_jpg(c_out, simulate(src, **p, rng=rng, noisy=False))
        noisy_l.append(f"{rel(n_out)}\t{rel(gt)}\t{rel(c_out)}\n")
        clean_l.append(f"{rel(c_out)}\t{rel(gt)}\n")
    (SPLITS / "noise_study_eval_syn_noisy.txt").write_text("".join(noisy_l), encoding="utf-8")
    (SPLITS / "noise_study_eval_syn_clean.txt").write_text("".join(clean_l), encoding="utf-8")

    # noise-free twins of the app examples (same exposure as scripts/make_examples.py)
    import make_examples as me
    from skimage import data as skdata

    ex = []
    for name, loader, _, exposure, kind, *_ in me.EXAMPLES:
        src = getattr(skdata, loader)()
        lin = (src.astype(np.float32) / 255.0) ** 2.2 * exposure * me.illumination(*src.shape[:2], kind)
        clean = (np.clip(lin ** (1 / 2.2), 0, 1) * 255 + 0.5).astype(np.uint8)
        _save_jpg(NS / "examples_clean" / f"{name}.jpg", clean)
        gt_p = NS / "examples_gt" / f"{name}.png"
        gt_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(gt_p), cv2.cvtColor(src, cv2.COLOR_RGB2BGR))
        ex.append(f"app/examples/{name}.jpg\t{rel(gt_p)}\t{rel(NS / 'examples_clean' / f'{name}.jpg')}\n")
    (SPLITS / "noise_study_examples.txt").write_text("".join(ex), encoding="utf-8")
    print("built dev_syn (90), eval_syn noisy/clean (100 each), examples (4)")


# ============================================================================ sets
def _pairs(path: Path):
    return [ln.split("\t") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_set(name: str) -> list[dict]:
    """Items: {id, low, gt (or None), clean_low (or None), tag}."""
    items = []
    if name == "dev_real":
        for lo, hi in _pairs(SPLITS / "dev_lolv2_real_unseen.txt"):
            items.append(dict(id=Path(lo).stem, low=ROOT / lo, gt=ROOT / hi, clean_low=None, tag="real"))
    elif name == "dev_syn":
        for lo, hi, tag in _pairs(SPLITS / "noise_study_dev_syn.txt"):
            items.append(dict(id=Path(lo).stem, low=ROOT / lo, gt=ROOT / hi, clean_low=None, tag=tag))
    elif name == "eval15":
        for lo in sorted((DATA / "LOL/eval15/low").glob("*.png")):
            items.append(dict(id=lo.stem, low=lo, gt=DATA / "LOL/eval15/high" / lo.name, clean_low=None, tag="real"))
    elif name == "syn_noisy":
        for lo, hi, cl in _pairs(SPLITS / "noise_study_eval_syn_noisy.txt"):
            items.append(dict(id=Path(lo).stem, low=ROOT / lo, gt=ROOT / hi, clean_low=ROOT / cl, tag="noisy"))
    elif name == "syn_clean":
        for lo, hi in _pairs(SPLITS / "noise_study_eval_syn_clean.txt"):
            items.append(dict(id=Path(lo).stem, low=ROOT / lo, gt=ROOT / hi, clean_low=None, tag="clean"))
    elif name == "lolv2_syn":
        for lo in sorted((DATA / "LOL-v2/Synthetic/Test/Low").glob("*.png")):
            items.append(dict(id=lo.stem, low=lo, gt=DATA / "LOL-v2/Synthetic/Test/Normal" / lo.name,
                              clean_low=None, tag="clean"))
    elif name == "examples":
        for lo, hi, cl in _pairs(SPLITS / "noise_study_examples.txt"):
            items.append(dict(id=Path(lo).stem, low=ROOT / lo, gt=ROOT / hi, clean_low=ROOT / cl, tag="noisy"))
    elif name == "unseen":
        for lo in sorted((DATA / "unseen").glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            items.append(dict(id=lo.stem, low=lo, gt=None, clean_low=None, tag="real"))
    else:
        raise ValueError(name)
    return items


def prepare(items, enh, desc=""):
    """Load images, run Retinexformer once per image (and on the noise-free twin if any)."""
    out = []
    t0 = time.perf_counter()
    for it in items:
        x = limit_size(read_image(it["low"]), MAX_SIDE)
        gt = read_image(it["gt"]) if it["gt"] else None
        if gt is not None and gt.shape[:2] != x.shape[:2]:
            gt = cv2.resize(gt, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
        r, _ = enh(x)
        rc = enh(limit_size(read_image(it["clean_low"]), MAX_SIDE))[0] if it["clean_low"] else None
        out.append({**it, "x": x, "gt": gt, "r": r, "rc": rc})
    print(f"  {desc}: {len(out)} images, model {time.perf_counter() - t0:.0f}s")
    return out


# ============================================================================ methods
def make_methods(qg_cfg, na_cfg, enh, ablations=True):
    """name -> fn(item) -> output image. QGH objects are built with deep=None: refine() never calls it."""
    qgh = QualityGuidedHybrid(None, qg_cfg)
    na = NoiseAwareQGH(None, qg_cfg, na_cfg)
    m = {
        "input": lambda it: it["x"],
        "standard": lambda it: it["r"],
        "qgh": lambda it: qgh.refine(it["r"], it["x"]).image,
        "qgh_na": lambda it: na.refine(it["r"], it["x"]).image,
    }
    if not ablations:
        return m

    def na_variant(**kw):
        obj = NoiseAwareQGH(None, qg_cfg, replace(na_cfg, **kw))
        return lambda it: obj.refine(it["r"], it["x"]).image

    na_nohl = NoiseAwareQGH(None, qg_cfg, na_cfg, highlight=False)
    qgh_unc = QualityGuidedHybrid(None, replace(qg_cfg, noise_ramp=(-1.0, -0.999), enabled=("noise", "highlight")))

    def pre_denoise(it):
        # denoise the *input* (sqrt domain ~ variance-stabilised) before Retinexformer, then QGH highlight
        s = np.sqrt(np.clip(it["x"], 0, 1))
        d, _, _ = noise_aware_denoise(s, na_cfg)
        r2, _ = enh(np.clip(d, 0, 1) ** 2)
        return QualityGuidedHybrid(None, replace(qg_cfg, enabled=("highlight",))).refine(r2, it["x"]).image

    m.update({
        "ab: NA without highlight": lambda it: na_nohl.refine(it["r"], it["x"]).image,
        "ab: - chroma": na_variant(chroma_method="none"),
        "ab: chroma = luma-only guide": na_variant(chroma_method="joint_guided"),
        "ab: - luma (chroma only)": na_variant(luma_method="none"),
        "ab: - adaptivity (global sigma)": na_variant(adaptive=False),
        "ab: - clean-region floor": na_variant(luma_floor=0.0, chroma_floor=0.0),
        ("ab: + detail_keep 0.15" if na_cfg.detail_keep == 0 else "ab: - detail_keep"):
            na_variant(detail_keep=0.15 if na_cfg.detail_keep == 0 else 0.0),
        ("ab: + detail protection 0.75" if na_cfg.edge_protect == 0 else "ab: - detail protection"):
            na_variant(edge_protect=0.75 if na_cfg.edge_protect == 0 else 0.0),
        "ab: - two-scale noise/texture separation": na_variant(texture_ratio=0.0),
        ("ab: - coarse pass" if na_cfg.coarse_k > 0 else "ab: + coarse pass (k=2)"):
            na_variant(coarse_k=0.0 if na_cfg.coarse_k > 0 else 2.0),
        "ab: luma = guided": na_variant(luma_method="guided"),
        "ab: luma = NLM": na_variant(luma_method="nlm"),
        "ab: QGH noise uncapped (mask = 1)": lambda it: qgh_unc.refine(it["r"], it["x"]).image,
        "ab: pre-model denoise": pre_denoise,
    })
    return m


# ============================================================================ metrics
def _hp(x, s=4.0):
    return x - cv2.GaussianBlur(x, (0, 0), s)


def _blur(x, s=8.0):
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)


def image_metrics(out, it, full=True):
    d = {}
    gt, rc, r = it["gt"], it["rc"], it["r"]
    ycc = rgb_to_ycc(out)
    d["sigma_y_est"] = estimate_sigma_map(np.ascontiguousarray(ycc[..., 0]))[1]
    d["sigma_c_est"] = float(np.mean([estimate_sigma_map(np.ascontiguousarray(ycc[..., c]))[1] for c in (1, 2)]))
    if gt is not None:
        d["psnr"], d["ssim"] = psnr(out, gt), ssim(out, gt)
        d["lpips"] = lpips_distance(out, gt)
        d["edge"] = edge_preservation(out, gt)
        d["colour_cast"] = colour_shift(_blur(out), _blur(gt))  # low-frequency chroma bias (CIELAB a*b*)
    else:
        d["edge_vs_std"] = edge_preservation(out, r)
        d["colour_cast_vs_std"] = colour_shift(_blur(out), _blur(r))
        if full:
            d["niqe"] = niqe(out)
    if rc is not None:  # exact residual noise w.r.t. Retinexformer on the noise-free twin
        res = rgb_to_ycc(out) - rgb_to_ycc(rc)
        d["resid_y"] = float(_hp(res[..., 0]).std())
        d["resid_c"] = float(np.mean([_hp(res[..., c]).std() for c in (1, 2)]))
        d["psnr_vs_clean_r"] = psnr(out, rc)
        d["lpips_vs_clean_r"] = lpips_distance(out, rc)
    if full:
        d["sigma_skimage"] = noise_level(out)
    return d


def run_set(prepared, methods, full=True):
    rows = {}
    for name, fn in methods.items():
        per = []
        t_ms = []
        for it in prepared:
            t0 = time.perf_counter()
            out = fn(it)
            t_ms.append((time.perf_counter() - t0) * 1000)
            per.append({"id": it["id"], "tag": it["tag"], **image_metrics(out, it, full)})
        rows[name] = {"per_image": per, "refine_ms_median": float(np.median(t_ms))}
    return rows


def summarise(per, tag=None):
    sel = [p for p in per if tag is None or p["tag"] == tag]
    keys = [k for k in sel[0] if k not in ("id", "tag")] if sel else []
    return {k: float(np.mean([p[k] for p in sel if p.get(k) is not None])) for k in keys
            if any(p.get(k) is not None for p in sel)} | {"n": len(sel)}


# ============================================================================ search (dev only)
SEARCH_SPACE = {
    "texture_ratio": [0.65, 0.0, 0.55, 0.75],
    "luma_method": ["bilateral", "guided"],
    "luma_k": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "luma_radius": [2, 3, 4, 6],
    "coarse_k": [0.0, 1.0, 2.0, 3.0],
    "detail_keep": [0.0, 0.1, 0.2, 0.3],
    "edge_protect": [0.0, 0.5, 0.75, 1.0],
    "detail_ramp": [(4.0, 8.0), (6.0, 12.0), (8.0, 16.0), (12.0, 24.0)],
    "chroma_method": ["joint_guided", "colour_guided"],
    "chroma_k": [0.5, 1.0, 2.0, 4.0, 8.0],
    "chroma_radius": [4, 8, 12, 16],
    "luma_floor": [0.0, 0.004, 0.006, 0.008, 0.012],
    "chroma_floor": [0.0, 0.002, 0.004, 0.008],
}
# constraints (dev): the refinement may not lose PSNR/SSIM on noisy sets nor degrade noise-free images
CLEAN_TOL = {"psnr": -0.05, "ssim": -0.002, "lpips": 0.002}
WORST_TOL = {"psnr": -0.5, "ssim": -0.02}   # every single dev image
LOW_NOISE_SIGMA = 0.008                    # dev images whose standard output reads below this = "low noise"
START = NoiseAwareConfig(luma_k=2.0, edge_protect=0.75, luma_floor=0.008, chroma_floor=0.004,
                         chroma_method="colour_guided", chroma_radius=8)  # conservative start


def _dev_eval(cfg, dev, base, pool):
    def one(it):
        out, _, _ = noise_aware_denoise(it["r"], cfg)
        return {"psnr": psnr(out, it["gt"]), "ssim": ssim(out, it["gt"])}, out

    res = list(pool.map(one, dev))
    lp = [lpips_distance(o, it["gt"]) for (_, o), it in zip(res, dev)]
    deltas = [{"psnr": m["psnr"] - b["psnr"], "ssim": m["ssim"] - b["ssim"], "lpips": l - b["lpips"], "lpips_abs": l}
              for (m, _), l, b in zip(res, lp, base)]
    groups = {"real": [], "noisy": [], "clean": [], "low_noise": []}
    for d, it in zip(deltas, dev):
        groups[it["tag"]].append(d)
        if it["sigma0"] < LOW_NOISE_SIGMA:
            groups["low_noise"].append(d)
    s = {g: {k: float(np.mean([v[k] for v in vs])) for k in vs[0]} | {"n": len(vs)} for g, vs in groups.items() if vs}
    objective = float(np.mean([d["lpips_abs"] for d in deltas]))
    viol, mag = [], 0.0
    for g in ("real", "noisy"):
        mag += max(0.0, -s[g]["psnr"]) * 0.1 + max(0.0, -s[g]["ssim"]) * 5
        if s[g]["psnr"] < 0 or s[g]["ssim"] < 0:
            viol.append(f"{g}: mean dPSNR/dSSIM < 0")
    for g in ("clean", "low_noise"):
        c = s[g]
        m = (max(0.0, CLEAN_TOL["psnr"] - c["psnr"]) * 0.1 + max(0.0, CLEAN_TOL["ssim"] - c["ssim"]) * 5
             + max(0.0, c["lpips"] - CLEAN_TOL["lpips"]) * 5)
        if m > 0:
            viol.append(f"{g} degraded")
            mag += m
    worst_p, worst_s = min(d["psnr"] for d in deltas), min(d["ssim"] for d in deltas)
    m = max(0.0, WORST_TOL["psnr"] - worst_p) * 0.05 + max(0.0, WORST_TOL["ssim"] - worst_s) * 2
    if m > 0:
        viol.append(f"worst image dPSNR {worst_p:.2f} / dSSIM {worst_s:.3f}")
        mag += m
    return {"objective": objective, "feasible": not viol, "violations": viol, "violation": mag,
            "worst": {"psnr": worst_p, "ssim": worst_s}, "groups": s}


def search(args):
    from lumen.deep import DeepEnhancer

    cv2.setNumThreads(1)
    enh = DeepEnhancer("retinexformer")
    dev = prepare(load_set("dev_real"), enh, "dev_real") + prepare(load_set("dev_syn"), enh, "dev_syn")
    base = [{"psnr": psnr(it["r"], it["gt"]), "ssim": ssim(it["r"], it["gt"]),
             "lpips": lpips_distance(it["r"], it["gt"])} for it in dev]
    for it in dev:
        it["sigma0"] = estimate_sigma_map(np.ascontiguousarray(rgb_to_ycc(it["r"])[..., 0]))[1]
    pool = ThreadPoolExecutor(args.workers)
    cfg = START
    log = []

    def score(c):
        t0 = time.perf_counter()
        r = _dev_eval(c, dev, base, pool)
        key = r["objective"] + (0 if r["feasible"] else 0.1 + 2 * r["violation"])  # graded: moves towards feasible
        log.append({"config": c.as_dict(), **r, "sec": round(time.perf_counter() - t0, 1)})
        return key, r

    best_key, best_r = score(cfg)
    print(f"start objective {best_r['objective']:.4f} feasible={best_r['feasible']} {best_r['groups']}")
    for sweep in range(args.sweeps):
        changed = False
        for field, values in SEARCH_SPACE.items():
            for v in values:
                if getattr(cfg, field) == v:
                    continue
                cand = replace(cfg, **{field: v})
                key, r = score(cand)
                if key < best_key - 1e-5:
                    best_key, best_r, cfg, changed = key, r, cand, True
            print(f"sweep {sweep} {field:13s} -> {getattr(cfg, field)!s:9s} objective {best_r['objective']:.4f} "
                  f"feasible={best_r['feasible']}", flush=True)
        if not changed:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_search.json").write_text(json.dumps({"baseline_standard": {
        g: {k: float(np.mean([b[k] for b, it in zip(base, dev) if it["tag"] == g])) for k in base[0]}
        for g in ("real", "noisy", "clean")}, "evaluations": log, "best": {"config": cfg.as_yaml_dict(), **best_r}},
        indent=1), encoding="utf-8")
    import yaml

    DEFAULT_NA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = {"created": time.strftime("%Y-%m-%d"), "tool": "scripts/noise_study.py search", "status": "experimental",
            "dev": ["data/splits/dev_lolv2_real_unseen.txt", "data/splits/noise_study_dev_syn.txt"],
            "objective": "min mean LPIPS on all dev; mean dPSNR, dSSIM >= 0 on dev_real and dev_syn noisy; "
                         f"noise-free and low-noise dev tolerance {CLEAN_TOL}; every image {WORST_TOL}",
            "evaluations": len(log), "dev_result": best_r["groups"]}
    DEFAULT_NA_CONFIG_PATH.write_text(yaml.safe_dump({"meta": meta, "params": cfg.as_yaml_dict()}, sort_keys=False),
                                      encoding="utf-8")
    print(f"best {cfg}\n{best_r}\nwrote {DEFAULT_NA_CONFIG_PATH}")


# ============================================================================ eval (held-out, once)
EVAL_SETS = ("eval15", "syn_noisy", "syn_clean", "lolv2_syn", "examples", "unseen")
ABLATION_SETS = ("dev_real", "dev_syn", "eval15", "syn_noisy", "syn_clean")


def evaluate(args):
    from lumen.deep import DeepEnhancer

    enh = DeepEnhancer("retinexformer")
    qg_cfg, na_cfg = load_qg_config(), load_na_config()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"na_config": na_cfg.as_yaml_dict(), "qg_config": "configs/quality_guided.yaml", "sets": {}}
    for name in args.sets:
        items = load_set(name)
        if name.startswith("dev_") and args.dev_every > 1:
            items = items[:: args.dev_every]  # dev ablations on a fixed subsample (selection is already done)
        prepared = prepare(items, enh, name)
        methods = make_methods(qg_cfg, na_cfg, enh, ablations=name in ABLATION_SETS)
        rows = run_set(prepared, methods, full=name not in ("dev_real", "dev_syn"))
        results["sets"][name] = {
            m: {"all": summarise(v["per_image"]),
                **({t: summarise(v["per_image"], t) for t in ("noisy", "clean")} if name == "dev_syn" else {}),
                "refine_ms_median": v["refine_ms_median"], "per_image": v["per_image"]}
            for m, v in rows.items()}
        print(f"  {name}: done", flush=True)
        (OUT / f"{args.out_name}_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_report(results, args.out_name)


COLS = [("psnr", "PSNR ↑", 2), ("ssim", "SSIM ↑", 4), ("lpips", "LPIPS ↓", 4), ("edge", "Edge ↑", 4),
        ("colour_cast", "Cast ↓", 2), ("resid_y", "Resid. Y ↓", 4), ("resid_c", "Resid. C ↓", 4),
        ("psnr_vs_clean_r", "PSNR vs clean-R ↑", 2), ("sigma_y_est", "σY est ↓", 4), ("sigma_c_est", "σC est ↓", 4),
        ("niqe", "NIQE ↓", 2), ("edge_vs_std", "Edge vs std ↑", 4), ("colour_cast_vs_std", "Cast vs std ↓", 2)]


def write_report(results, out_name="eval"):
    lines = ["# Noise study - evaluation tables (generated by scripts/noise_study.py)\n"]
    for name, rows in results["sets"].items():
        for sub in ("all", "noisy", "clean"):
            first = next(iter(rows.values()))
            if sub not in first:
                continue
            keys = [c for c in COLS if c[0] in first[sub]]
            lines.append(f"\n## {name}" + ("" if sub == "all" else f" ({sub})") + f" - n = {first[sub]['n']}\n")
            lines.append("| Method | " + " | ".join(c[1] for c in keys) + " | Refine ms |")
            lines.append("|---|" + "---|" * (len(keys) + 1))
            for m, v in rows.items():
                cells = [f"{v[sub][k]:.{p}f}" if k in v[sub] else "-" for k, _, p in keys]
                lines.append(f"| {m} | " + " | ".join(cells) + f" | {v['refine_ms_median']:.0f} |")
    (OUT / f"{out_name}_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / f'{out_name}_tables.md'}")


def report(_args):
    """Merge per-set result files (eval --out-name eval_<set>) into eval_results.json / eval_tables.md."""
    merged = None
    order = list(EVAL_SETS) + ["dev_real", "dev_syn"]
    files = sorted(OUT.glob("eval_*_results.json"), key=lambda f: order.index(f.stem[5:-8])
                   if f.stem[5:-8] in order else 99)
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if merged is None:
            merged = {k: v for k, v in r.items() if k != "sets"} | {"sets": {}}
        merged["sets"].update(r["sets"])
    (OUT / "eval_results.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
    write_report(merged, "eval")


# ============================================================================ figures
def _label(img_u8, text):
    out = img_u8.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _panel(images, labels, crop=None, scale=1):
    tiles = []
    for im, lb in zip(images, labels):
        u = to_uint8(im)
        if crop:
            y0, x0, h, w = crop
            u = u[y0:y0 + h, x0:x0 + w]
            u = cv2.resize(u, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        tiles.append(_label(u, lb))
    return np.concatenate(tiles, 1)


FIGS = [  # set, id, crop (y, x, h, w) for the zoomed row
    ("examples", "coffee_dark", (150, 200, 180, 240)),
    ("examples", "chelsea_extremely_dark", (60, 120, 160, 200)),
    ("examples", "astronaut_uneven_light", (60, 300, 180, 200)),
    ("eval15", "1", (150, 250, 150, 200)),
    ("eval15", "748", (120, 200, 150, 200)),
    ("syn_clean", None, (100, 100, 150, 150)),
    ("unseen", "2", (150, 150, 180, 220)),
    ("unseen", "10", (150, 150, 180, 220)),
]


def figures(args):
    from lumen.deep import DeepEnhancer

    enh = DeepEnhancer("retinexformer")
    qg_cfg, na_cfg = load_qg_config(), load_na_config()
    methods = make_methods(qg_cfg, na_cfg, enh, ablations=False)
    fig_dir = OUT / "figures"
    for set_name, iid, crop in FIGS:
        items = load_set(set_name)
        it = next((i for i in items if i["id"] == iid), items[0])
        p = prepare([it], enh, f"{set_name}/{it['id']}")[0]
        outs = {m: methods[m](p) for m in ("input", "standard", "qgh", "qgh_na")}
        names = ["Input", "Standard (Retinexformer)", "QGH (calibrated)", "QGH-NA (new)"]
        ims = list(outs.values())
        if p["gt"] is not None:
            ims.append(p["gt"])
            names.append("Ground truth")
        full = _panel(ims, names)
        h, w = p["x"].shape[:2]
        y0, x0, ch, cw = crop
        y0, x0 = min(y0, h - ch), min(x0, w - cw)
        zoom = _panel(ims, [n + " (zoom)" for n in names], (max(y0, 0), max(x0, 0), min(ch, h), min(cw, w)), 2)
        if zoom.shape[1] != full.shape[1]:
            zoom = cv2.resize(zoom, (full.shape[1], round(zoom.shape[0] * full.shape[1] / zoom.shape[1])),
                              interpolation=cv2.INTER_NEAREST)
        sheet = np.concatenate([full, zoom], 0)
        write_image(fig_dir / f"{set_name}_{it['id']}.png", sheet.astype(np.float32) / 255.0)
        print("wrote", fig_dir / f"{set_name}_{it['id']}.png")
    # root-cause figure for coffee: QGH noise mask vs the noise-aware sigma map
    it = next(i for i in load_set("examples") if i["id"] == "coffee_dark")
    p = prepare([it], enh, "coffee masks")[0]
    q = QualityGuidedHybrid(None, qg_cfg).refine(p["r"], p["x"])
    _, maps, _ = noise_aware_denoise(p["r"], na_cfg)
    # high-passed so tone differences between the two model outputs are not counted as noise
    resid = _hp(rgb_to_ycc(p["r"])[..., 0] - rgb_to_ycc(p["rc"])[..., 0])
    resid = np.clip(np.sqrt(cv2.GaussianBlur(resid ** 2, (0, 0), 6)) / 0.1, 0, 1)
    sig = np.clip(maps["sigma_luma"] / 0.08, 0, 1)
    tiles = [p["r"], q.intermediates["mask_noise"], np.repeat(resid[..., None], 3, -1),
             np.repeat(sig[..., None], 3, -1), np.repeat(maps["mask_luma"][..., None], 3, -1)]
    write_image(fig_dir / "coffee_root_cause.png", _panel(tiles, [
        "Standard output", "QGH noise mask (calibrated)", "True residual noise (high-pass R - R_clean)",
        "Estimated luma sigma (NA)", "NA luma blend mask"]).astype(np.float32) / 255.0)
    print("wrote", fig_dir / "coffee_root_cause.png")


# ============================================================================ latency
def latency(args):
    qg_cfg, na_cfg = load_qg_config(), load_na_config()
    rows = []
    src = read_image(DATA / "unseen" / "1.jpg")  # 2560 px real photo
    sizes = {"600x400": (600, 400), "768 px": None, "1024 px": None}
    for threads in (args.threads,):
        cv2.setNumThreads(threads)
        for label, wh in sizes.items():
            img = cv2.resize(src, wh, interpolation=cv2.INTER_AREA) if wh else limit_size(src, int(label.split()[0]))
            r = np.clip(img * 4, 0, 1).astype(np.float32)  # stand-in for a model output (content-independent timing)
            qgh = QualityGuidedHybrid(None, qg_cfg)
            na = NoiseAwareQGH(None, qg_cfg, na_cfg)
            for name, fn in (("qgh refine", lambda: qgh.refine(r, img)), ("qgh_na refine", lambda: na.refine(r, img))):
                fn()
                ts = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    fn()
                    ts.append((time.perf_counter() - t0) * 1000)
                rows.append({"what": name, "size": f"{img.shape[1]}x{img.shape[0]}", "cv2_threads": threads,
                             "median_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1)})
                print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("--sweeps", type=int, default=2)
    s.add_argument("--workers", type=int, default=12)
    e = sub.add_parser("eval")
    e.add_argument("--sets", nargs="+", default=list(EVAL_SETS))
    e.add_argument("--dev-every", type=int, default=3)
    e.add_argument("--out-name", default="eval")
    sub.add_parser("figures")
    sub.add_parser("report")
    lt = sub.add_parser("latency")
    lt.add_argument("--threads", type=int, default=4)
    lt.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    {"build": build, "search": search, "eval": evaluate, "figures": figures, "latency": latency, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
