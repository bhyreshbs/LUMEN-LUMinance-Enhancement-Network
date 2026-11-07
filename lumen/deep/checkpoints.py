"""Known checkpoint hashes (single source of truth for the app's integrity check and download_weights.py)."""

from __future__ import annotations

import hashlib
from pathlib import Path

KNOWN_SHA256 = {
    "retinexformer_LOL_v1.pth": "9b4810dc51322fab70be8658ee0ad1df64a54aa814497b7849767baa2ff32cd1",
    "retinexformer_LOL_v2_real.pth": "539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b",
    "retinexformer_LOL_v2_synthetic.pth": "2f793853c6b1ba0b4a3ff12b7b10f0f2c851eaefda1006437eb3f35229838c4a",
    "zero_dce_pp.pth": "ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84",
}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str | Path) -> tuple[bool, str]:
    """(ok, message). Unknown file names cannot be verified and are reported as such (ok=False)."""
    p = Path(path)
    expected = KNOWN_SHA256.get(p.name)
    if expected is None:
        return False, f"{p.name} is not a known official checkpoint; its integrity cannot be verified"
    actual = sha256(p)
    if actual != expected:
        return False, f"{p} SHA-256 mismatch (expected {expected[:12]}..., got {actual[:12]}...)"
    return True, f"{p.name} SHA-256 verified"
