"""Before/after check of the integrated highlight hue fix, through the real LumenPipeline.

    python scripts/highlight_hue_integration_check.py [--sets ...]   -> results/highlight_hue_integration/

Four variants on identical inputs: Natural before/after (deep, highlight_hue False/True) and Highlight-safe
before/after (quality_guided_hybrid, False/True). Reports PSNR, SSIM, LPIPS, NIQE, highlight colour error vs ground
truth, over-exposure, pipeline latency (median ms of PipelineResult.latency_ms) and how many outputs are
bit-identical before vs after. "Before" = highlight_hue=False = the frozen behaviour (Phase 5 numbers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

import highlight_hue_study as hs  # noqa: E402
from lumen import LumenPipeline  # noqa: E402
from lumen.metrics.quality import overexposure_ratio  # noqa: E402
from lumen.utils import limit_size, read_image  # noqa: E402

OUT = ROOT / "results" / "highlight_hue_integration"
SETS = ("eval15", "hc_eval", "lolv2_syn", "syn_noisy", "syn_clean", "examples", "unseen")
STRIDE = {"syn_noisy": 2, "syn_clean": 2}  # every 2nd image of the 100-image noisy/clean regression sets
VARIANTS = {"natural_before": ("deep", False), "natural_after": ("deep", True),
            "safe_before": ("quality_guided_hybrid", False), "safe_after": ("quality_guided_hybrid", True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    args = ap.parse_args()
    pipe = LumenPipeline(max_side=1280, deep_model="retinexformer")
    pipe.run(np.full((64, 64, 3), 0.1, np.float32), "deep")  # warm-up
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        items = hs.load_set(name)[:: STRIDE.get(name, 1)]
        rows = {v: [] for v in VARIANTS}
        same = {"natural": 0, "safe": 0}
        for it in items:
            x = limit_size(read_image(it["low"]), 1280)
            gt = None
            if it.get("gt"):
                import cv2

                gt = read_image(it["gt"])
                if gt.shape[:2] != x.shape[:2]:
                    gt = cv2.resize(gt, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
            outs = {}
            for v, (mode, flag) in VARIANTS.items():
                r = pipe.run(x, mode, deep_model="retinexformer", highlight_hue=flag)
                outs[v] = r.output
                p = {"x": r.input, "gt": gt}
                d = hs.metrics(r.output, p, with_niqe=True)
                d.update(id=it["id"], tag=it["tag"], over=overexposure_ratio(r.output) * 100, ms=r.latency_ms,
                         hue_applied=bool(r.info.get("highlight_hue", {}).get("applied")))
                rows[v].append(d)
            same["natural"] += int(np.array_equal(outs["natural_before"], outs["natural_after"]))
            same["safe"] += int(np.array_equal(outs["safe_before"], outs["safe_after"]))
        (OUT / f"{name}.json").write_text(json.dumps({"set": name, "n": len(items), "identical": same, "rows": rows}, indent=1))
        g = lambda v, k: (np.mean([r[k] for r in rows[v] if r.get(k) is not None]) if any(r.get(k) is not None for r in rows[v]) else float("nan"))  # noqa: E731
        print(f"== {name} (n={len(items)}); bit-identical before/after: Natural {same['natural']}/{len(items)}, Highlight-safe {same['safe']}/{len(items)}")
        for v in VARIANTS:
            print(f"  {v:15s} PSNR {g(v,'psnr'):6.2f} SSIM {g(v,'ssim'):.4f} LPIPS {g(v,'lpips'):.4f} NIQE {g(v,'niqe'):.2f} "
                  f"hlErr {g(v,'hl_dE'):6.2f} over% {g(v,'over'):.3f} ms {np.median([r['ms'] for r in rows[v]]):.0f}", flush=True)


if __name__ == "__main__":
    main()
