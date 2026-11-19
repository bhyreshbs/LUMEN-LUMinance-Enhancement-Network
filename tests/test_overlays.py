"""'What changed' overlays: only refinements that really ran are reported, with their real masks and coverage."""

import base64
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import lumen.deep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import server as SRV  # noqa: E402
from test_app import FakeDeep  # noqa: E402
from test_stream import parse_sse  # noqa: E402
from test_web import client, jpeg_bytes, restore  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    FakeDeep.calls, FakeDeep.missing = [], set()
    monkeypatch.setattr(lumen.deep, "DeepEnhancer", FakeDeep)
    return FakeDeep


H, W = 192, 256
PATCH = (slice(20, 80), slice(150, 230))     # a bright window in a dark room


def room(window: bool) -> bytes:
    img = np.full((H, W, 3), 28, np.uint8)
    img[::2, ::2] += 6                         # a little texture
    if window:
        img[PATCH] = (252, 250, 244)
    return jpeg_bytes(img)


def decode_rgba(url: str) -> np.ndarray:
    assert url.startswith("data:image/png;base64,")
    arr = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert arr.ndim == 3 and arr.shape[2] == 4, "overlay must be a transparent PNG"
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)


def ids(payload) -> list[str]:
    return [o["id"] for o in payload["overlays"]]


# ------------------------------------------------------------------------------------------- which overlays appear
def test_nothing_is_reported_when_nothing_was_refined(fake):
    c = client()
    assert restore(c, "classical", room(True)).json()["overlays"] == []     # Quick has no refinement steps
    assert restore(c, "standard", room(False)).json()["overlays"] == []     # no bright spot -> the colour fix did not apply


def test_natural_reports_the_highlight_colour_fix_when_it_applied(fake):
    d = restore(client(), "standard", room(True)).json()
    assert ids(d) == ["highlight_hue"]
    assert d["overlays"][0]["coverage"] == d["details"]["highlight_hue"]["coverage"] > 0


def test_highlight_safe_reports_only_what_ran_and_with_the_real_coverage(fake):
    d = restore(client(), "quality_guided", room(True)).json()
    applied = set(d["details"]["applied"]) | {"highlight_hue"}
    assert ids(d) and set(ids(d)) <= applied
    assert ids(d) == [i for i in SRV.OVERLAYS if i in ids(d)]               # stable, documented order
    for o in d["overlays"]:
        truth = (d["details"]["highlight_hue"] if o["id"] == "highlight_hue" else d["details"]["refinements"][o["id"]])
        assert o["coverage"] == truth["coverage"] > 0, o["id"]


def test_a_refinement_that_did_not_run_is_never_reported(fake):
    d = restore(client(), "quality_guided", room(False)).json()
    ran = {k for k, v in d["details"]["refinements"].items() if v["active"]}
    assert set(ids(d)) <= ran and "highlight_hue" not in ids(d)


# ------------------------------------------------------------------------------------------- the overlay pictures
def test_overlay_is_a_tinted_transparent_png_that_follows_the_mask(fake):
    d = restore(client(), "standard", room(True)).json()
    o = d["overlays"][0]
    rgba = decode_rgba(o["image"])
    assert max(rgba.shape[:2]) <= SRV.OVERLAY_MAX_SIDE
    assert abs(rgba.shape[0] / rgba.shape[1] - H / W) < 0.02                 # same aspect as the photo
    assert re.fullmatch(r"#[0-9a-f]{6}", o["color"])
    tint = tuple(int(o["color"][i:i + 2], 16) for i in (1, 3, 5))
    seen = rgba[..., 3] > 0
    assert seen.any() and (rgba[..., :3][seen] == tint).all()                # colour = the tint, opacity = the mask
    assert rgba[..., 3].max() <= round(SRV.OVERLAY_OPACITY * 255)
    sy, sx = rgba.shape[0] / H, rgba.shape[1] / W
    inside = rgba[int(PATCH[0].start * sy):int(PATCH[0].stop * sy), int(PATCH[1].start * sx):int(PATCH[1].stop * sx), 3]
    alpha = rgba[..., 3].astype(float)
    outside = (alpha.sum() - inside.astype(float).sum()) / (alpha.size - inside.size)
    assert inside.mean() > 10 * max(outside, 1.0)                           # the window is where the overlay is


def test_overlay_labels_are_plain_language():
    jargon = ["retinexformer", "psnr", "ssim", "lpips", "niqe", "checkpoint", "qgh", "quality-guided", "quality guided",
              "fp16", "cuda", "ablation", "zero-dce", "transformer", "clahe", "sha-256", "mask"]
    for key, (label, tint) in SRV.OVERLAYS.items():
        assert not [j for j in jargon if j in label.lower()], (key, label)
        assert re.fullmatch(r"#[0-9a-f]{6}", tint)


# ------------------------------------------------------------------------------------------- every route agrees
def test_stream_and_plain_endpoints_report_identical_overlays(fake):
    c = client()
    data = room(True)
    plain = restore(c, "quality_guided", data).json()
    done = dict(parse_sse(c.post("/api/restore/stream", files={"file": ("p.jpg", data, "image/jpeg")},
                                 data={"style": "quality_guided"}).text))["done"]
    assert done["overlays"] == plain["overlays"] and plain["overlays"]


def test_compare_all_reports_overlays_for_each_style(fake):
    c = client()
    data = room(True)
    res = {r["style"]: r for r in c.post("/api/compare", files={"file": ("p.jpg", data, "image/jpeg")}).json()["results"]}
    assert res["classical"]["overlays"] == []
    assert "highlight_hue" in ids(res["standard"]) and "highlight_hue" in ids(res["quality_guided"])
    single = {s: restore(c, s, data).json() for s in ("standard", "quality_guided")}
    for s in single:                                                          # same overlays as the single-style call
        assert res[s]["overlays"] == single[s]["overlays"], s


def test_overlays_do_not_change_the_restored_pictures(fake):
    """The overlay data is extra: the restored picture is the same one the plain pipeline produces."""
    c = client()
    data = room(True)
    a = restore(c, "quality_guided", data).json()
    b = restore(c, "quality_guided", data).json()
    assert a["restored"] == b["restored"] and a["overlays"] == b["overlays"]    # deterministic
    assert json.dumps(a["details"]["refinements"], sort_keys=True) == json.dumps(b["details"]["refinements"], sort_keys=True)


# ------------------------------------------------------------------------------------------- the page and its script
def test_every_element_the_workspace_script_looks_up_exists_in_the_page():
    """A typo in an element id fails silently in the browser; this catches it here."""
    web = ROOT / "app" / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    js = (web / "static" / "js" / "workspace.js").read_text(encoding="utf-8")
    page_ids = set(re.findall(r'\bid="([^"]+)"', html))
    wanted = set(re.findall(r'\$\("([^"]+)"\)', js))
    assert wanted and not (wanted - page_ids), f"ids used by workspace.js but missing from index.html: {sorted(wanted - page_ids)}"


def test_workspace_page_has_a_heading_for_screen_readers():
    html = (ROOT / "app" / "web" / "index.html").read_text(encoding="utf-8")
    workspace = html.split('id="viewWorkspace"', 1)[1].split("</section>", 1)[0]
    assert "<h1" in workspace            # the visible h1 lives in the start screen, which is hidden once a photo is chosen
