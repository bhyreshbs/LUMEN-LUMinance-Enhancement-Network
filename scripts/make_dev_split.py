"""Write a development split that the LOL-v1 Retinexformer checkpoint has never seen.

    python scripts/make_dev_split.py            # -> data/splits/dev_lolv2_real_unseen.txt

LOL-v2-real Train re-contains ~400 LOL-v1 our485 images (the checkpoint's training set) plus near-
duplicate shots. This keeps only pairs whose 16x16 thumbnails are not within `--tol` of any LOL-v1
our485 or eval15 image, so the split can be used to calibrate/select settings without touching eval15
and without measuring on images the model memorised. One "low<TAB>high" pair per line (relative paths).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from train_retinexformer import DATA, fingerprint, pair_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=DATA / "splits" / "dev_lolv2_real_unseen.txt")
    args = ap.parse_args()

    cand = pair_dir(DATA / "LOL-v2/Real_captured/Train/Low", DATA / "LOL-v2/Real_captured/Train/Normal")
    seen = pair_dir(DATA / "LOL/our485/low", DATA / "LOL/our485/high") + \
        pair_dir(DATA / "LOL/eval15/low", DATA / "LOL/eval15/high")
    if not cand or not seen:
        raise SystemExit("Need data/LOL and data/LOL-v2 (see data/README.md).")

    def fp(pairs):
        return np.stack([np.concatenate([fingerprint(lo), fingerprint(hi)]) for lo, hi in pairs])

    d = np.abs(fp(cand)[:, None] - fp(seen)[None]).mean(-1).min(1)
    keep = [p for p, di in zip(cand, d) if di >= args.tol]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    args.out.write_text("".join(f"{rel(lo)}\t{rel(hi)}\n" for lo, hi in keep), encoding="utf-8")
    print(f"{len(keep)} of {len(cand)} LOL-v2-real Train pairs kept -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
