"""Hugging Face Spaces entry point: fetch the Retinexformer checkpoint, verify it, then start the LUMEN app.

The checkpoint is not part of the Space (its redistribution rights are uncertain; see THIRD_PARTY_NOTICES.md).
It is fetched at container start from one of (first that is set):

  LUMEN_WEIGHTS_REPO  a *private* Hugging Face model repo, e.g. "your-name/lumen-weights", holding
                      retinexformer_LOL_v1.pth. Needs the Space secret HF_TOKEN (read access).
  LUMEN_WEIGHTS_URL   any direct download URL of the file.

The file must match the pinned SHA-256 (lumen/deep/checkpoints.py) or it is deleted. If nothing is
configured or the download fails, the app still starts and offers only the Quick (classical) style,
showing the reason - the same behaviour as the Docker image without a mounted checkpoint.
The token is never printed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent


def known_sha256() -> dict:
    # load lumen/deep/checkpoints.py directly: importing the lumen.deep package would import torch first
    spec = importlib.util.spec_from_file_location("lumen_checkpoints", APP / "lumen" / "deep" / "checkpoints.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KNOWN_SHA256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch() -> None:
    dest = Path(os.environ.get("LUMEN_RETINEXFORMER_WEIGHTS", "/home/user/models/retinexformer_LOL_v1.pth"))
    expected = known_sha256().get(dest.name)
    if dest.is_file() and expected and sha256(dest) == expected:
        print(f"[lumen] checkpoint present and verified: {dest}", flush=True)
        return
    repo, url = os.environ.get("LUMEN_WEIGHTS_REPO", "").strip(), os.environ.get("LUMEN_WEIGHTS_URL", "").strip()
    if repo:
        url = f"https://huggingface.co/{repo}/resolve/main/{dest.name}"
    if not url:
        print("[lumen] no LUMEN_WEIGHTS_REPO / LUMEN_WEIGHTS_URL set: starting with the Quick (classical) style only",
              flush=True)
        return
    headers = {"User-Agent": "lumen-space"}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token and url.startswith("https://huggingface.co/"):
        headers["Authorization"] = f"Bearer {token}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    where = repo or url.split("?")[0]
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        hint = " (check that HF_TOKEN is set and can read the repo)" if e.code in (401, 403, 404) and repo else ""
        print(f"[lumen] checkpoint download from {where} failed: HTTP {e.code}{hint}", flush=True)
        tmp.unlink(missing_ok=True)
        return
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"[lumen] checkpoint download from {where} failed: {type(e).__name__}", flush=True)
        tmp.unlink(missing_ok=True)
        return
    got = sha256(tmp)
    if expected and got != expected:
        print(f"[lumen] downloaded file rejected: SHA-256 {got[:12]}... != expected {expected[:12]}...", flush=True)
        tmp.unlink(missing_ok=True)
        return
    tmp.replace(dest)
    print(f"[lumen] checkpoint downloaded from {where} and verified ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    fetch()
    os.chdir(APP)
    os.execv(sys.executable, [sys.executable, str(APP / "app" / "app.py")])
