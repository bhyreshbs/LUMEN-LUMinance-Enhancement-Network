"""Download and verify pretrained weights into weights/.

    python scripts/download_weights.py                 # everything
    python scripts/download_weights.py --only zerodce  # one model family

Sources and licences (see weights/README.md):
  * Retinexformer - https://github.com/caiyuanhao1998/Retinexformer (ICCV 2023; code MIT). Official
    pretrained-weights Google Drive folder 1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV linked from its README; the
    weights carry no explicit licence and were trained on LOL (terms unstated). Redistribution is uncertain.
  * Zero-DCE++    - https://github.com/Li-Chongyi/Zero-DCE_extension (snapshot Epoch99.pth). Its README
    states: academic research only, CC BY-NC 4.0. Not shipped with the app; download it yourself.

Every file is checked against a pinned SHA-256 hash and must load strictly into the LUMEN architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
sys.path.insert(0, str(ROOT))

# Checkpoints the web app needs (Standard / Quality Guided modes). Zero-DCE++ is research-only.
RUNTIME = ("retinexformer_LOL_v1.pth",)

# file name -> (family, source kind, source, sha256)
FILES = {
    "zero_dce_pp.pth": (
        "zerodce", "url",
        "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/"
        "Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth",
        "ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84",
    ),
    "retinexformer_LOL_v1.pth": (
        "retinexformer", "gdrive", "1xDwQtTCj3tlAVCTJgYrzonBGVwqeOhKu",
        "9b4810dc51322fab70be8658ee0ad1df64a54aa814497b7849767baa2ff32cd1",
    ),
    "retinexformer_LOL_v2_real.pth": (
        "retinexformer", "gdrive", "1tChRwTfqhs-A67QzG8a9Lrx7qKB3m89K",
        "539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b",
    ),
    "retinexformer_LOL_v2_synthetic.pth": (
        "retinexformer", "gdrive", "1VPNTcVhzwSQ1PcSyYOuZIgrclITM6F_b",
        "2f793853c6b1ba0b4a3ff12b7b10f0f2c851eaefda1006437eb3f35229838c4a",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(kind: str, src: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    if kind == "url":
        urllib.request.urlretrieve(src, tmp)
    else:
        import gdown

        if gdown.download(id=src, output=str(tmp), quiet=False) is None:
            raise RuntimeError(f"Google Drive download failed for id {src}")
    tmp.replace(dest)


def verify_loads(name: str, family: str, path: Path) -> None:
    import torch

    if family == "zerodce":
        from lumen.deep.zero_dce import ZeroDCEpp

        ZeroDCEpp().load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    else:
        from lumen.deep.retinexformer import build_lol_model, load_checkpoint

        load_checkpoint(build_lol_model(), path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["zerodce", "retinexformer"])
    ap.add_argument("--runtime", action="store_true",
                    help=f"only the checkpoint(s) the web app needs: {', '.join(RUNTIME)}")
    ap.add_argument("--verify-only", action="store_true",
                    help="do not download; fail if a selected file is missing, altered or not loadable")
    args = ap.parse_args()
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    failed = []
    for name, (family, kind, src, digest) in FILES.items():
        if args.only and family != args.only:
            continue
        if args.runtime and name not in RUNTIME:
            continue
        dest = WEIGHTS / name
        try:
            if dest.exists() and sha256(dest) == digest:
                print(f"[ok]   {name} (already present)")
            elif args.verify_only:
                raise RuntimeError("missing" if not dest.exists() else "SHA-256 mismatch (file altered or corrupt)")
            else:
                print(f"[get]  {name}")
                fetch(kind, src, dest)
                if sha256(dest) != digest:
                    dest.unlink()
                    raise RuntimeError("SHA-256 mismatch - file removed")
            verify_loads(name, family, dest)
            print(f"[ok]   {name} verified ({dest.stat().st_size / 1e6:.2f} MB, strict load)")
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            print(f"[FAIL] {name}: {e}")
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
