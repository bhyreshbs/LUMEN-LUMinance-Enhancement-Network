"""Comparative evaluation + ablation study for P1.

Paired data (PSNR/SSIM/LPIPS available), e.g. LOL eval15:
    python scripts/evaluate.py --low data/LOL/eval15/low --high data/LOL/eval15/high \
        --team T01 --usns 1XX22CS001,1XX22CS002

Unseen images without ground truth (no-reference metrics only):
    python scripts/evaluate.py --unseen data/unseen --team T01 --usns ...

Writes results/<run>/metrics.json, results/<run>/samples/*.png and eval_P1_<team>.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252 and choke on the arrows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lumen import LumenPipeline  # noqa: E402
from lumen.deep import available_models  # noqa: E402
from lumen.classical import COMPONENTS as CLASSICAL_COMPONENTS  # noqa: E402
from lumen.hybrid.fusion import HYBRID_COMPONENTS  # noqa: E402
from lumen.hybrid.quality_guided import QG_COMPONENTS, QualityGuidedConfig  # noqa: E402
from lumen.metrics import (  # noqa: E402
    colour_shift, edge_preservation, lpips_distance, niqe, noise_level, overexposure_ratio, psnr, ssim,
)
from lumen.utils import read_image, write_image  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Each experiment is (label, mode, disabled components, deep model). The deep model is explicit for every
# deep *and* hybrid experiment, so a label always names the model family that produced the numbers.
def main_configs(deep_models: list[str]) -> list[tuple]:
    deep = [(f"deep: {m}", "deep", set(), m) for m in deep_models]
    hybrid = [(f"hybrid: {m}", "hybrid", set(), m) for m in deep_models]
    qg = [(f"quality_guided_hybrid: {m}", "quality_guided_hybrid", set(), m) for m in deep_models]
    return [("classical", "classical", set(), None), *deep, *hybrid, *qg]


def ablation_configs(deep_models: list[str]) -> list[tuple]:
    classical = [(f"classical - {c}", "classical", {c}, None) for c in CLASSICAL_COMPONENTS]
    hybrid = [(f"hybrid: {m} - {c}", "hybrid", {c}, m) for m in deep_models for c in HYBRID_COMPONENTS]
    qg = [(f"quality_guided_hybrid: {m} - {c}", "quality_guided_hybrid", {c}, m)
          for m in deep_models for c in QG_COMPONENTS]
    return classical + hybrid + qg


META_KEYS = ("deep_model", "weights", "device", "fp16", "deep_pretrained")


def safe_name(label: str) -> str:
    """File-name-safe label: ':' would create an NTFS alternate data stream on Windows."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")

# Column order follows the P1 problem statement's list of evaluation metrics
COLUMNS = [
    ("psnr", "PSNR ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓"), ("niqe", "NIQE ↓"),
    ("latency_ms", "Latency ms ↓"), ("noise", "Noise σ ↓"), ("edge", "Edge pres. ↑"),
    ("colour_shift", "Colour shift ↓"), ("overexposed", "Overexp. % ↓"),
]


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)


def pair(low_dir: Path, high_dir: Path) -> list[tuple[Path, Path]]:
    """Match by filename, falling back to the digits in the name (LOL-v2: low00690 <-> normal00690)."""
    highs = list_images(high_dir)
    by_name = {p.name: p for p in highs}
    by_num = {re.sub(r"\D", "", p.stem): p for p in highs}
    pairs = []
    for lo in list_images(low_dir):
        hi = by_name.get(lo.name) or by_num.get(re.sub(r"\D", "", lo.stem))
        if hi is not None:
            pairs.append((lo, hi))
    return pairs


def gt_mean_rescale(out: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Retinexformer's --GT_mean protocol: scale the output so its mean gray level matches the GT's.
    Uses the ground truth, so it is only for comparing with published numbers, never the headline."""
    import cv2

    m_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).mean()
    m_gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).mean()
    return np.clip(out * (m_gt / max(m_out, 1e-6)), 0, 1).astype(np.float32)


def evaluate(pipe, items, configs, with_gt: bool, sample_dir: Path, n_samples: int, use_niqe: bool,
             gt_mean: bool = False, per_image: list | None = None):
    """Returns (metrics per experiment, metadata per experiment). Metadata comes from the result
    `info` of the experiment's own runs, so checkpoint/device are those actually used.
    If `per_image` is a list, one dict per (experiment, image) is appended to it, including the
    quality-guided refinement diagnostics when present."""
    results, meta = {}, {}
    for name, mode, disable, deep_model in configs:
        pipe.run(read_image(items[0][0]), mode, disable, deep_model=deep_model)  # warm-up: model load, CUDA init
        rows = []
        for i, (lo, hi) in enumerate(tqdm(items, desc=name)):
            x = read_image(lo)
            r = pipe.run(x, mode, disable, deep_model=deep_model)
            out = r.output
            row = {"latency_ms": r.latency_ms, "noise": noise_level(out), "overexposed": 100 * overexposure_ratio(out)}
            if use_niqe:
                row["niqe"] = niqe(out)
            if with_gt:
                gt = read_image(hi)
                if gt.shape != out.shape:
                    import cv2

                    gt = cv2.resize(gt, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA)
                if gt_mean:
                    out = gt_mean_rescale(out, gt)
                row.update(psnr=psnr(out, gt), ssim=ssim(out, gt), lpips=lpips_distance(out, gt),
                           edge=edge_preservation(out, gt), colour_shift=colour_shift(out, gt))
            else:
                row["edge"] = edge_preservation(out, r.input)
            rows.append(row)
            if per_image is not None:
                rec = {"experiment": name, "image": Path(lo).name, **row}
                if "refinements" in r.info:
                    rec["applied"] = "+".join(r.info["applied"]) or "identity"
                    for comp, d in r.info["refinements"].items():
                        rec[f"{comp}_active"] = d["active"]
                        rec[f"{comp}_coverage"] = d["coverage"]
                        rec[f"{comp}_area_frac"] = d["area_frac"]
                per_image.append(rec)
            if i < n_samples:
                write_image(sample_dir / f"{lo.stem}__{safe_name(name)}.png", out)
        keys = rows[0].keys()
        results[name] = {k: float(np.mean([r[k] for r in rows if r[k] is not None] or [np.nan])) for k in keys}
        meta[name] = {"mode": mode, "disabled": sorted(disable), **{k: r.info.get(k) for k in META_KEYS}}
        if deep_model is not None and meta[name]["deep_model"] != deep_model:
            raise RuntimeError(f"{name}: requested {deep_model} but {meta[name]['deep_model']} ran")
    return results, meta


def write_per_image(rows: list[dict], out_dir: Path) -> None:
    """per_image.csv (every experiment x image) and qg_diagnostics.json (refinement summary)."""
    if not rows:
        return
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    diag = {}
    for r in rows:
        if "applied" not in r:
            continue
        d = diag.setdefault(r["experiment"], {"images": 0, "identity_images": 0, "activations": {}, "coverage": {}})
        d["images"] += 1
        d["identity_images"] += r["applied"] == "identity"
        for comp in QG_COMPONENTS:
            if f"{comp}_active" in r:
                d["activations"][comp] = d["activations"].get(comp, 0) + int(r[f"{comp}_active"])
                d["coverage"].setdefault(comp, []).append(r[f"{comp}_coverage"])
    for d in diag.values():
        d["mean_coverage"] = {c: round(float(np.mean(v)), 5) for c, v in d.pop("coverage").items()}
    if diag:
        (out_dir / "qg_diagnostics.json").write_text(json.dumps(diag, indent=2))


def meta_markdown(meta: dict) -> str:
    lines = ["| Experiment | Deep model | Checkpoint | Device | FP16 |", "|---|---|---|---|---|"]
    for name, m in meta.items():
        if m["deep_model"] is None:
            lines.append(f"| {name} | - | - | CPU (classical) | - |")
        else:
            ck = m["weights"] or "none (random init)"
            lines.append(f"| {name} | {m['deep_model']} | `{ck}` | {m['device']} | {m['fp16']} |")
    return "### Experiment metadata\n\n" + "\n".join(lines) + "\n"


def to_markdown(results: dict, title: str) -> str:
    cols = [c for c in COLUMNS if any(c[0] in v for v in results.values())]
    lines = [f"### {title}", "", "| Method | " + " | ".join(h for _, h in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, v in results.items():
        cells = [f"{v[k]:.4f}" if k in ("ssim", "lpips", "edge") else f"{v[k]:.2f}" if k in v else "-" for k, _ in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", type=Path)
    ap.add_argument("--high", type=Path)
    ap.add_argument("--unseen", type=Path, help="folder of unseen low-light images without ground truth")
    ap.add_argument("--pairs-list", type=Path,
                    help="text file with one 'low<TAB>high' pair per line (paths relative to the repo), "
                         "e.g. data/splits/dev_lolv2_real_unseen.txt; used instead of --low/--high")
    ap.add_argument("--qg-config", type=Path,
                    help="QualityGuidedConfig YAML (default: configs/quality_guided.yaml if present)")
    ap.add_argument("--team", default="T00")
    ap.add_argument("--usns", default="USN1,USN2,USN3", help="comma separated USNs (first line of eval file)")
    ap.add_argument("--run", default="latest")
    ap.add_argument("--eval-md", type=Path, help="where to write the eval markdown (default: eval_P1_<team>.md)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-niqe", action="store_true")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--deep-models", nargs="+", default=["zerodce", "retinexformer"], choices=available_models())
    ap.add_argument("--device", help="cuda | cpu (default: auto)")
    ap.add_argument("--gt-mean", action="store_true",
                    help="rescale outputs to the GT mean brightness (published-protocol comparison only)")
    args = ap.parse_args()

    if not (args.low and args.high) and not args.unseen and not args.pairs_list:
        ap.error("give --low and --high (paired), --pairs-list, and/or --unseen")

    out_dir = ROOT / "results" / args.run
    qg_cfg = QualityGuidedConfig.from_yaml(args.qg_config) if args.qg_config else None
    pipe = LumenPipeline(max_side=args.max_side, device=args.device, qg_config=qg_cfg)
    MAIN = main_configs(args.deep_models)
    main_names = [c[0] for c in MAIN]
    configs = MAIN + ([] if args.no_ablation else ablation_configs(args.deep_models))
    report, all_results, all_meta, per_image = [], {}, {}, []

    if args.pairs_list or (args.low and args.high):
        if args.pairs_list:
            items = [tuple(ROOT / q for q in line.split("\t")) for line in
                     args.pairs_list.read_text(encoding="utf-8").splitlines() if line.strip()]
            set_name = args.pairs_list.stem
        else:
            items, set_name = pair(args.low, args.high), args.low.parent.name
        if not items:
            raise SystemExit("No low/high pairs found.")
        items = items[: args.limit] if args.limit else items
        res, meta = evaluate(pipe, items, configs, True, out_dir / "samples", args.samples, not args.no_niqe,
                             args.gt_mean, per_image)
        all_results["paired"] = res
        all_meta.update(meta)
        report += [to_markdown({k: res[k] for k in main_names}, f"Comparison on {set_name} ({len(items)} images)")]
        if not args.no_ablation:
            report += [to_markdown(res, "Ablation study")]

    if args.unseen:
        items = [(p, None) for p in list_images(args.unseen)]
        items = items[: args.limit] if args.limit else items
        res, meta = evaluate(pipe, items, MAIN, False, out_dir / "unseen_samples", args.samples, not args.no_niqe,
                             per_image=per_image)
        all_results["unseen"] = res
        all_meta.update(meta)
        report += [to_markdown(res, f"Unseen images, no reference ({len(items)} images)")]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({**all_results, "meta": all_meta}, indent=2))
    write_per_image(per_image, out_dir)
    lines = []
    if args.gt_mean:
        lines.append("GT-mean rescaling: ON - uses ground truth, for comparison with published numbers only")
    weights_txt = "".join(line + "  " + chr(10) for line in lines)
    eval_md = args.eval_md or ROOT / f"eval_P1_{args.team}.md"
    eval_md.parent.mkdir(parents=True, exist_ok=True)
    eval_md.write_text(
        args.usns + "\n\n# LUMEN - P1 Evaluation Results\n\n"
        + weights_txt
        + "Edge preservation: gradient-magnitude correlation with ground truth (or input when unseen).\n\n"
        + "\n".join(report) + "\n" + meta_markdown(all_meta),
        encoding="utf-8",
    )
    print("\n".join(report))
    print(f"Wrote {eval_md} and {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
