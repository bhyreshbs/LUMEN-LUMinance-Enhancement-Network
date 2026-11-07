"""Calibrate the quality-guided refinement thresholds on the dev split (never on eval15).

    python scripts/make_dev_split.py
    python scripts/calibrate_quality_guided.py          # -> configs/quality_guided.yaml

For every refinement (noise, colour, contrast, highlight) and a small grid of strengths:
  1. R = Retinexformer(low) once per dev image (cached); C = correction(R) everywhere.
  2. Per-pixel squared-error gain  g = |R - GT|^2 - |C - GT|^2  (positive = the correction helps;
     squared error is what PSNR measures), divided by the image's own squared error so every image
     counts equally - this matches the reported metric (mean of per-image PSNR).
  3. g is accumulated in bins of the detector score. The threshold t that maximises the total gain of
     "correct only pixels with score >= t" is chosen, subject to the refined area being at most
     `--max-coverage` of the image (refinement must stay local; a correction that only helps when
     applied almost everywhere is a global/domain adjustment, not a problem-region fix).
  4. A refinement whose best total gain is below `--min-rel-gain` of the dev squared error stays
     *disabled* - the detector does not identify pixels where the correction is justified.

Writes the config plus every gain curve (results/quality_guided_calibration.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lumen.deep import DeepEnhancer  # noqa: E402
from lumen.hybrid.quality_guided import (  # noqa: E402
    QG_COMPONENTS, QualityGuidedConfig, colour_correction, colour_score, contrast_correction, contrast_score,
    denoise_correction, highlight_correction, highlight_score, noise_score,
)
from lumen.utils import read_image  # noqa: E402

N_BINS = 200
RAMP_BINS = 4  # soft ramp spans this many score-quantile bins above the threshold
# Semantic floors: a detector's threshold may not drop below what its name claims to detect.
# "highlight" means near-clipped pixels, so it may not become a general darkening of bright areas.
MIN_THRESHOLD = {"highlight": 0.85}
INVALID = -1e9


def scores(comp, r, x, cfg):
    if comp == "noise":
        return noise_score(r)
    if comp == "colour":
        drift, reliable = colour_score(r, x, cfg)
        return np.where(reliable > 0.5, drift, INVALID)
    if comp == "contrast":
        loss, valid = contrast_score(r, x, cfg)
        return np.where(valid > 0.5, loss, INVALID)
    return highlight_score(r)


def corrected(comp, r, x, cfg):
    return {"noise": lambda: denoise_correction(r, cfg), "colour": lambda: colour_correction(r, x, cfg),
            "contrast": lambda: contrast_correction(r, cfg), "highlight": lambda: highlight_correction(r, cfg)}[comp]()


GRID = {  # strength settings tried per refinement (config field -> values)
    "noise": ("noise_eps", [1e-4, 4e-4, 1.6e-3]),
    "colour": ("colour_strength", [0.25, 0.5, 1.0]),
    "contrast": ("contrast_gain", [0.3, 0.6, 1.0]),
    "highlight": ("highlight_knee", [0.7, 0.8, 0.9]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-list", type=Path, default=ROOT / "data/splits/dev_lolv2_real_unseen.txt")
    ap.add_argument("--model", default="retinexformer")
    ap.add_argument("--min-rel-gain", type=float, default=1e-3,
                    help="minimum total gain as a fraction of dev squared error (1e-3 ~ 0.004 dB)")
    ap.add_argument("--max-coverage", type=float, default=0.25,
                    help="largest fraction of pixels a refinement may claim (locality constraint)")
    ap.add_argument("--out", type=Path, default=ROOT / "configs/quality_guided.yaml")
    ap.add_argument("--report", type=Path, default=ROOT / "results/quality_guided_calibration.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if "eval15" in str(args.pairs_list):
        raise SystemExit("Refusing to calibrate on eval15 - it is the held-out test set.")
    lines = [ln.split("\t") for ln in args.pairs_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if any("eval15" in a or "eval15" in b for a, b in lines):
        raise SystemExit("The pairs list contains eval15 images.")
    items = [(ROOT / a, ROOT / b) for a, b in lines][: args.limit or None]
    base_cfg = QualityGuidedConfig()

    enh = DeepEnhancer(args.model)
    cache = []
    for lo, hi in tqdm(items, desc=f"{args.model} on dev"):
        x, gt = read_image(lo), read_image(hi)
        r, _ = enh(x)
        cache.append((x.astype(np.float16), r.astype(np.float16), gt.astype(np.float16)))
    img_sse = [float(((r.astype(np.float32) - g.astype(np.float32)) ** 2).mean(2).sum()) for _, r, g in cache]
    n_px = sum(r.shape[0] * r.shape[1] for _, r, _ in cache)
    mean_psnr = float(np.mean([10 * np.log10(r.shape[0] * r.shape[1] / e) for (_, r, _), e in zip(cache, img_sse)]))
    base_sse = float(len(cache))  # gains are per-image relative, so the "total error" is 1 per image
    print(f"{len(cache)} dev images, {args.model} mean dev PSNR {mean_psnr:.3f} dB")

    report = {"dev": args.pairs_list.relative_to(ROOT).as_posix(), "images": len(cache), "model": args.model,
              "weights": enh.weights_path.name if enh.weights_path else None, "base_sse": base_sse, "components": {}}
    params, enabled = base_cfg.as_dict(), []

    for comp in QG_COMPONENTS:
        field_name, values = GRID[comp]
        pilot = np.concatenate([scores(comp, c[1].astype(np.float32), c[0].astype(np.float32), base_cfg).ravel()
                                for c in cache[:: max(1, len(cache) // 30)]])
        pilot = pilot[pilot > INVALID / 2]
        edges = np.unique(np.quantile(pilot, np.linspace(0, 1, N_BINS + 1)))
        best = None
        curves = {}
        for v in values:
            cfg = QualityGuidedConfig.from_dict({**{k: params[k] for k in ("colour_sigma", "colour_min_input",
                                                                         "contrast_sigma", "contrast_min_cv",
                                                                         "contrast_min_input")}, field_name: v})
            gain_hist = np.zeros(len(edges) - 1)
            cnt_hist = np.zeros(len(edges) - 1)
            for (xh, rh, gh), sse_i in zip(cache, img_sse):
                x, r, gt = xh.astype(np.float32), rh.astype(np.float32), gh.astype(np.float32)
                s = scores(comp, r, x, cfg).ravel()
                ok = s > INVALID / 2
                g = (((r - gt) ** 2).mean(2) - ((corrected(comp, r, x, cfg) - gt) ** 2).mean(2)).ravel() / sse_i
                sc = np.clip(s[ok], edges[0], edges[-1])
                gain_hist += np.histogram(sc, bins=edges, weights=g[ok])[0]
                cnt_hist += np.histogram(sc, bins=edges)[0]
            suffix = np.cumsum(gain_hist[::-1])[::-1]  # gain of correcting every pixel with score >= edges[k]
            cover = np.cumsum(cnt_hist[::-1])[::-1] / n_px
            floor = MIN_THRESHOLD.get(comp, -np.inf)
            allowed = np.where((cover <= args.max_coverage) & (edges[:-1] >= floor), suffix, -np.inf)
            k = int(np.argmax(allowed)) if np.isfinite(allowed).any() else len(suffix) - 1
            res = {"value": v, "best_gain": float(suffix[k]), "rel_gain": float(suffix[k] / base_sse),
                   "threshold": float(edges[k]), "full": float(edges[min(k + RAMP_BINS, len(edges) - 1)]),
                   "coverage": float(cnt_hist[k:].sum() / n_px), "gain_everywhere": float(suffix[0])}
            curves[str(v)] = {"edges": edges[:-1].round(6).tolist(), "gain": gain_hist.tolist(),
                              "count": cnt_hist.tolist(), **res}
            print(f"  {comp:9s} {field_name}={v:<7}  best mean rel. MSE gain {res["rel_gain"] * 100:+.3f}% at score>="
                  f"{res['threshold']:.4f} (covers {res['coverage'] * 100:.2f}% px); "
                  f"everywhere {res['gain_everywhere'] / base_sse * 100:+.2f}%")
            if best is None or res["best_gain"] > best["best_gain"]:
                best = res
        est_db = 10 * np.log10(1 / max(1 - best["rel_gain"], 1e-9))  # mean relative MSE drop -> approx dB
        use = best["rel_gain"] >= args.min_rel_gain
        report["components"][comp] = {"enabled": use, "field": field_name, "est_db": float(est_db), **best,
                                      "curves": curves}
        print(f"-> {comp}: {'ENABLED' if use else 'disabled'} ({field_name}={best['value']}, "
              f"est. {est_db:+.4f} dB mean)\n")
        if use:
            enabled.append(comp)
            params[field_name] = best["value"]
            params[f"{comp}_ramp"] = [best["threshold"], max(best["full"], best["threshold"] + 1e-4)]

    params["enabled"] = enabled
    cfg = QualityGuidedConfig.from_dict(params)  # validates
    meta = {"created": date.today().isoformat(), "tool": "scripts/calibrate_quality_guided.py",
            "dev": report["dev"], "images": len(cache), "model": args.model, "weights": report["weights"],
            "selection": "max total squared-error gain per refinement on dev; eval15 never used",
            "objective": f"mean per-image relative squared-error gain, coverage <= {args.max_coverage}",
            "dev_mean_psnr": round(mean_psnr, 4),
            "est_db_per_component": {c: round(v["est_db"], 4) for c, v in report["components"].items()},
            "enabled": enabled}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump({"meta": meta, "params": cfg.as_dict()}, sort_keys=False), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    print(f"enabled refinements: {enabled or 'none (identity)'}\nwrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
