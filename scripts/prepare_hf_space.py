"""Assemble the Hugging Face Space folder (Docker SDK) from this repository.

    python scripts/prepare_hf_space.py            # -> dist/hf_space/  (then upload it - docs/DEPLOY_HF_SPACES.md)

Copies only what the web app needs at run time, plus the Space files in deploy/huggingface/. Excluded:
checkpoints, datasets, results, tests, docs and the experimental research code (lumen/experimental,
configs/experimental - not part of any production mode). The build fails if a checkpoint, a dataset image
folder or an .env file would end up in the Space.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf_space"

TREES = {"lumen": ("experimental",), "app": (), "configs": ("experimental",)}
FILES = ["requirements.txt", "THIRD_PARTY_NOTICES.md", "weights/README.md", "scripts/download_weights.py"]
SPACE = ROOT / "deploy" / "huggingface"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyi", ".pytest_cache")
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".onnx"}


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for tree, skip in TREES.items():
        def ignore(d, names, _skip=skip, _tree=tree):
            out = set(IGNORE(d, names))
            if Path(d).resolve() == (ROOT / _tree).resolve():
                out |= {n for n in names if n in _skip}
            return out
        shutil.copytree(ROOT / tree, OUT / tree, ignore=ignore)
    for f in FILES:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, OUT / f)
    for f in ("Dockerfile", "README.md", "start.py"):
        shutil.copy2(SPACE / f, OUT / f)

    files = [p for p in OUT.rglob("*") if p.is_file()]
    bad = [p for p in files if p.suffix.lower() in FORBIDDEN_SUFFIXES or p.name.startswith(".env")]
    if bad:
        shutil.rmtree(OUT)
        raise SystemExit(f"refusing to package: {[str(p.relative_to(ROOT)) for p in bad]}")
    size = sum(p.stat().st_size for p in files)
    big = sorted(files, key=lambda p: -p.stat().st_size)[:3]
    print(f"wrote {OUT} - {len(files)} files, {size / 1e6:.2f} MB")
    print("largest:", ", ".join(f"{p.relative_to(OUT).as_posix()} ({p.stat().st_size / 1e3:.0f} kB)" for p in big))
    return 0


if __name__ == "__main__":
    sys.exit(main())
