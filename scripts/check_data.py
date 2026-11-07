"""Verify the dataset layout described in data/README.md.

    python scripts/check_data.py            # check everything that is present
    python scripts/check_data.py --require LOL/our485 LOL/eval15

For each paired split: counts low/high images, lists unmatched files, and checks that the images decode
and that each low/high pair has the same size. Exits 1 on problems (missing *required* splits,
unpaired files, unreadable images, size mismatches). Optional splits that are absent are only reported.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# name -> (low dir, high dir), relative to data/
PAIRED = {
    "LOL/our485": ("LOL/our485/low", "LOL/our485/high"),
    "LOL/eval15": ("LOL/eval15/low", "LOL/eval15/high"),
    "LOL-v2/Real/Train": ("LOL-v2/Real_captured/Train/Low", "LOL-v2/Real_captured/Train/Normal"),
    "LOL-v2/Real/Test": ("LOL-v2/Real_captured/Test/Low", "LOL-v2/Real_captured/Test/Normal"),
    "LOL-v2/Synthetic/Train": ("LOL-v2/Synthetic/Train/Low", "LOL-v2/Synthetic/Train/Normal"),
    "LOL-v2/Synthetic/Test": ("LOL-v2/Synthetic/Test/Low", "LOL-v2/Synthetic/Test/Normal"),
}
UNPAIRED = {"unseen": "unseen"}
DEFAULT_REQUIRED = ["LOL/our485", "LOL/eval15"]


def images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def key(p: Path) -> str:
    """Digits-only fallback key: LOL-v2 Real uses low00690 <-> normal00690."""
    digits = re.sub(r"\D", "", p.stem)
    return digits or p.stem


def match(lows: list[Path], highs: list[Path]):
    """Pair by identical file name first (LOL-v1, LOL-v2 Synthetic), else by digits (LOL-v2 Real)."""
    names = {p.name for p in highs}
    if sum(p.name in names for p in lows) >= len(lows) / 2:
        lk, hk = {p.name: p for p in lows}, {p.name: p for p in highs}
    else:
        lk, hk = {key(p): p for p in lows}, {key(p): p for p in highs}
    return lk, hk


def image_size(p: Path) -> tuple[int, int] | None:
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return None if img is None else (img.shape[1], img.shape[0])


def check_paired(name: str, low: Path, high: Path, required: bool) -> list[str]:
    problems: list[str] = []
    if not (low.is_dir() and high.is_dir()):
        msg = f"[{'MISSING' if required else 'absent '}] {name}: expected {low.relative_to(DATA)} and {high.relative_to(DATA)}"
        print(msg)
        return [msg] if required else []

    lows, highs = images(low), images(high)
    lk, hk = match(lows, highs)
    only_low, only_high = sorted(set(lk) - set(hk)), sorted(set(hk) - set(lk))
    pairs = [(lk[k], hk[k]) for k in sorted(set(lk) & set(hk))]

    sizes: Counter = Counter()
    unreadable, mismatched = [], []
    for lo, hi in pairs:
        s_lo, s_hi = image_size(lo), image_size(hi)
        if s_lo is None or s_hi is None:
            unreadable.append(lo.name if s_lo is None else hi.name)
            continue
        if s_lo != s_hi:
            mismatched.append(f"{lo.name} {s_lo} vs {hi.name} {s_hi}")
        sizes[s_lo] += 1

    size_txt = ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.most_common(3)) or "-"
    print(f"[ok     ] {name}: {len(pairs)} pairs  (low {len(lows)}, high {len(highs)})  sizes: {size_txt}")
    for label, items in (("low without high", only_low), ("high without low", only_high),
                         ("unreadable", unreadable), ("size mismatch", mismatched)):
        if items:
            shown = ", ".join(items[:5]) + (" ..." if len(items) > 5 else "")
            msg = f"          {label}: {len(items)} -> {shown}"
            print(msg)
            problems.append(f"{name}: {label} ({len(items)})")
    if not pairs:
        problems.append(f"{name}: no pairs")
    return problems


def check_unpaired(name: str, d: Path) -> list[str]:
    if not d.is_dir():
        print(f"[absent ] {name}: data/{name}/ (put your own unseen low-light photos here)")
        return []
    imgs = images(d)
    unreadable = [p.name for p in imgs if image_size(p) is None]
    print(f"[ok     ] {name}: {len(imgs)} images")
    if unreadable:
        print(f"          unreadable: {len(unreadable)} -> {', '.join(unreadable[:5])}")
        return [f"{name}: unreadable ({len(unreadable)})"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require", nargs="*", default=DEFAULT_REQUIRED, choices=list(PAIRED),
                    help="splits that must exist (default: LOL/our485 LOL/eval15)")
    args = ap.parse_args()

    print(f"Checking {DATA}\n")
    problems: list[str] = []
    for name, (lo, hi) in PAIRED.items():
        problems += check_paired(name, DATA / lo, DATA / hi, name in args.require)
    for name, rel in UNPAIRED.items():
        problems += check_unpaired(name, DATA / rel)

    if problems:
        print(f"\nFAILED - {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll present datasets look good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
