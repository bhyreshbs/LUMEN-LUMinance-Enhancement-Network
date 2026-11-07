"""End-to-end smoke test of a running LUMEN app (local process or container), using real file uploads.

    python scripts/smoke_test_app.py --url http://127.0.0.1:7860 --expect-modes standard quality_guided classical
    python scripts/smoke_test_app.py --url ... --expect-modes classical          # e.g. container without checkpoint

Checks: pages and every asset they reference load; the available styles match --expect-modes; every expected
style restores real uploads (PNG result, same size as the processed original); Compare All runs the neural
network once; invalid uploads get a clear 4xx error; NIQE availability matches --expect-niqe.
Standard library only. Exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE_IDS = ["standard", "quality_guided", "classical"]


def post_file(url: str, path: Path, fields: dict, timeout: float = 120):
    """multipart/form-data POST with one file; returns (status, parsed JSON)."""
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def png_size(data_url: str) -> tuple[int, int]:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "result is not a PNG"
    return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--expect-modes", nargs="+", default=STYLE_IDS, choices=STYLE_IDS)
    ap.add_argument("--expect-niqe", choices=["available", "unavailable"], default="unavailable")
    ap.add_argument("--images", nargs="+", type=Path, default=sorted((ROOT / "app" / "examples").glob("*.jpg")))
    ap.add_argument("--bad-files", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    base = args.url.rstrip("/")
    failures = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        if not cond:
            failures.append(msg)

    for page in ("/", "/details"):
        code, html = get(base + page)
        refs = set(re.findall(r'(?:src|href)="(/[^"#]*)"', html.decode("utf-8", "replace")))
        bad = [r for r in refs if get(base + r)[0] != 200]
        check(code == 200 and not bad, f"page {page} and its {len(refs)} assets load" + (f" (missing {bad})" if bad else ""))
    # Cloud Run reserves /healthz on *.run.app (Google's frontend answers 404), so fall back to /api/status there
    check(get(base + "/healthz")[0] == 200 or get(base + "/api/status")[0] == 200, "health endpoint (/healthz or /api/status)")

    status = json.loads(get(base + "/api/status")[1])
    offered = [s["id"] for s in status["styles"] if s["available"]]
    check(offered == args.expect_modes, f"available styles {offered}")
    niqe = status["technical"].get("niqe_available")
    check((args.expect_niqe == "available") == bool(niqe), f"NIQE {'available' if niqe else 'unavailable'}")

    for img in args.images:
        for style in args.expect_modes:
            t = time.perf_counter()
            code, d = post_file(base + "/api/restore", img, {"style": style})
            ok = code == 200 and png_size(d["restored"]) == (d["width"], d["height"])
            info = f"applied={d['details']['applied']} identity={d['details']['identity']}" if ok and style == "quality_guided" else ""
            check(ok, f"{style:15s} {img.name:30s} {time.perf_counter() - t:5.2f}s"
                      + (f" server {d['details']['timing_ms']} {info}" if ok else f" -> {code} {d.get('error')}"))

    if "standard" in args.expect_modes:
        code, d = post_file(base + "/api/compare", args.images[0], {})
        res = {r["style"]: r for r in d.get("results", [])}
        check(code == 200 and list(res) == args.expect_modes, f"Compare all returns {list(res)}")
        if "quality_guided" in res:
            check(res["standard"]["details"]["timing_ms"]["model"] == res["quality_guided"]["details"]["timing_ms"]["model"],
                  "Compare all reuses one neural-network run")

    for bad in args.bad_files:
        code, d = post_file(base + "/api/restore", bad, {"style": args.expect_modes[0]})
        check(400 <= code < 500 and d.get("error"), f"invalid upload {bad.name} -> {code}: {str(d.get('error'))[:120]}")

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
