"""Before/after pairs of the bundled example photos for the landing page.

    python scripts/make_landing_assets.py            # -> app/web/static/img/landing/*_before.jpg, *_after.jpg, manifest.json

`<name>_before.jpg` is the example photo as the app processes it; `<name>_after.jpg` is the same photo restored by the
Natural style exactly as the app runs it (EnhancementService: official LOL-v1 checkpoint, SHA-256 verified, plus the
highlight colour fix). `manifest.json` records the values the pipeline reported for each photo, which is what the landing
page may say about them. Every file is checked against the 150 KB budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app")]

import cv2  # noqa: E402

OUT = ROOT / "app" / "web" / "static" / "img" / "landing"
EXAMPLES = ROOT / "app" / "examples"
TITLES = {"coffee_dark": "Café table", "chelsea_extremely_dark": "Very dark room",
          "astronaut_uneven_light": "Uneven light", "rocket_night_highlights": "Night lights"}
BUDGET = 150 * 1024


def write_jpeg(path: Path, rgb, quality: int) -> int:
    """Write RGB uint8 as JPEG at `quality`, stepping down until it fits the budget. Returns the quality used."""
    for q in range(quality, 59, -4):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        if ok and len(buf) <= BUDGET:
            path.write_bytes(buf.tobytes())
            return q
    raise SystemExit(f"{path.name} does not fit in {BUDGET // 1024} KB even at quality 60")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    import service as S
    from lumen import LumenPipeline

    svc = S.EnhancementService(LumenPipeline(max_side=1024, deep_model="retinexformer"))
    if not svc.status[S.STANDARD].available:
        raise SystemExit(f"Natural is unavailable: {svc.status[S.STANDARD].reason}")
    OUT.mkdir(parents=True, exist_ok=True)
    photos = []
    for src in sorted(EXAMPLES.glob("*.jpg")):
        seen = {}
        r = svc.enhance(src, S.STANDARD, progress=lambda stage, data: seen.__setitem__(stage, data))
        h, w = r.enhanced.shape[:2]
        qb = write_jpeg(OUT / f"{src.stem}_before.jpg", r.original, args.quality)
        qa = write_jpeg(OUT / f"{src.stem}_after.jpg", r.enhanced, args.quality)
        hue = (r.info.get("highlight_hue") or {})
        photos.append({
            "id": src.stem, "title": TITLES.get(src.stem, src.stem.replace("_", " ").title()), "width": w, "height": h,
            "before": f"/static/img/landing/{src.stem}_before.jpg", "after": f"/static/img/landing/{src.stem}_after.jpg",
            "reported": {"category": seen["analysed"]["category"], "mean_brightness": seen["analysed"]["mean_lum"],
                         "input_noise": seen["analysed"]["noise_sigma"], "restore_ms": seen["restored"]["ms"],
                         "highlight_colour_kept": bool(hue.get("applied"))},
            "jpeg_quality": {"before": qb, "after": qa},
        })
        print(f"{src.stem:26s} {w}x{h}  before q{qb} {(OUT / f'{src.stem}_before.jpg').stat().st_size // 1024} KB, "
              f"after q{qa} {(OUT / f'{src.stem}_after.jpg').stat().st_size // 1024} KB")
    enh = svc.pipe.get_deep("retinexformer")
    manifest = {"made_by": "scripts/make_landing_assets.py", "style": "Natural", "checkpoint": enh.weights_path.name,
                "device": str(enh.device), "photos": photos}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
