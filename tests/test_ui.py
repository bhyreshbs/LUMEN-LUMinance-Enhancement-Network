"""The redesigned web UI: pages, 404, the light map that comes with every result, landing assets, fonts, and the rules
that keep the interface honest (plain language, evidence-backed hint)."""

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
WEB = ROOT / "app" / "web"
sys.path.insert(0, str(ROOT / "app"))
import server as SRV  # noqa: E402
from test_app import FakeDeep  # noqa: E402
from test_stream import parse_sse  # noqa: E402
from test_web import JARGON, client, jpeg_bytes, restore  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    FakeDeep.calls, FakeDeep.missing = [], set()
    monkeypatch.setattr(lumen.deep, "DeepEnhancer", FakeDeep)
    return FakeDeep


def decode(url: str, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), flags)


# ------------------------------------------------------------------------------------------- pages
@pytest.mark.parametrize("page", ["/", "/how", "/evidence", "/details"])
def test_every_page_and_its_assets_load(fake, page):
    c = client()
    r = c.get(page)
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "default-src 'self'" in r.headers["content-security-policy"]
    refs = set(re.findall(r'(?:src|href)="(/[^"#]*)"', r.text))
    assert any(x.endswith(".js") for x in refs) and "/static/css/lumen.css" in refs
    for ref in refs:
        assert c.get(ref).status_code == 200, f"{page} references missing {ref}"
    assert 'id="themeBtn"' in r.text and "/static/js/theme-boot.js" in refs   # every page has the shell and no theme flash


def test_unknown_pages_get_the_designed_404_and_the_api_keeps_json(fake):
    c = client()
    r = c.get("/no/such/page")
    assert r.status_code == 404 and "text/html" in r.headers["content-type"] and "in the dark" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    api = c.get("/api/nope")
    assert api.status_code == 404 and api.json()["code"] == "not_found"
    assert c.get("/static/nope.css").status_code == 404 and "text/html" not in c.get("/static/nope.css").headers["content-type"]
    assert c.get("/examples/nope.jpg").status_code == 404


def test_every_element_each_page_script_looks_up_exists():
    for page, script in (("index.html", "workspace.js"), ("evidence.html", "evidence.js"), ("details.html", "details.js")):
        html = (WEB / page).read_text(encoding="utf-8")
        js = (WEB / "static" / "js" / script).read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([^"]+)"', html))
        missing = set(re.findall(r'\$\("([^"]+)"\)', js)) - ids
        assert not missing, f"{script} looks up ids missing from {page}: {sorted(missing)}"
    landing = (WEB / "static" / "js" / "landing.js").read_text(encoding="utf-8")
    index = (WEB / "index.html").read_text(encoding="utf-8")
    for el in re.findall(r'getElementById\("([^"]+)"\)', landing):
        assert f'id="{el}"' in index, el


def test_the_workspace_scripts_stay_in_plain_language():
    """The workspace's other modules are held to the same no-jargon rule as workspace.js."""
    for name in ("landing.js", "develop.js", "shell.js"):
        js = (WEB / "static" / "js" / name).read_text(encoding="utf-8")
        strings = " ".join(a or b for a, b in re.findall(r"`([^`]*)`|\"([^\"]*)\"", js)).lower()
        strings = re.sub(r"#version 300 es.*?}", "", strings, flags=re.S)   # shader source is code, not text a user reads
        found = [j for j in JARGON if j in strings]
        assert not found, f"jargon {found} in {name}"


# ------------------------------------------------------------------------------------------- the light map
@pytest.mark.parametrize("style,kind", [("standard", "added"), ("quality_guided", "added"), ("classical", "scene")])
def test_every_result_carries_a_small_grayscale_light_map(fake, style, kind):
    d = restore(client(), style).json()
    light = d["maps"]["light"]
    assert light["kind"] == kind and light["label"] == SRV.LIGHT_MAP_LABELS[kind]
    assert light["image"].startswith("data:image/jpeg;base64,")
    assert len(light["image"]) <= SRV.LIGHT_MAP_MAX_BYTES              # under 60 KB as sent
    img = decode(light["image"])
    assert img.ndim == 2 and max(img.shape) <= SRV.LIGHT_MAP_MAX_SIDE   # grayscale, small
    assert abs(img.shape[0] / img.shape[1] - d["height"] / d["width"]) < 0.02


def test_a_large_noisy_map_still_fits_the_budget():
    rng = np.random.default_rng(0)
    url = SRV._light_map_url(rng.random((2000, 3000)).astype(np.float32))   # worst case for JPEG: pure noise
    assert len(url) <= SRV.LIGHT_MAP_MAX_BYTES and max(decode(url).shape) == SRV.LIGHT_MAP_MAX_SIDE


def test_light_map_is_the_same_on_every_route(fake):
    c = client()
    data = jpeg_bytes()
    plain = restore(c, "standard", data).json()
    done = dict(parse_sse(c.post("/api/restore/stream", files={"file": ("p.jpg", data, "image/jpeg")}, data={"style": "standard"}).text))["done"]
    res = {r["style"]: r for r in c.post("/api/compare", files={"file": ("p.jpg", data, "image/jpeg")}).json()["results"]}
    assert done["maps"] == plain["maps"] == res["standard"]["maps"]
    assert res["quality_guided"]["maps"]["light"]["kind"] == "added"     # shares the one model run
    assert res["classical"]["maps"]["light"]["kind"] == "scene"


# ------------------------------------------------------------------------------------------- landing assets
def test_landing_pairs_are_real_documented_and_within_budget():
    out = WEB / "static" / "img" / "landing"
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["made_by"] == "scripts/make_landing_assets.py" and (ROOT / m["made_by"]).is_file()
    examples = sorted(p.stem for p in (ROOT / "app" / "examples").glob("*.jpg"))
    assert sorted(p["id"] for p in m["photos"]) == examples
    for p in m["photos"]:
        for side in ("before", "after"):
            f = out / Path(p[side]).name
            assert f.is_file() and f.stat().st_size <= 150 * 1024, f
            img = cv2.imread(str(f))
            assert (img.shape[1], img.shape[0]) == (p["width"], p["height"])
        before = cv2.imread(str(out / f"{p['id']}_before.jpg")).astype(float)
        src = cv2.imread(str(ROOT / "app" / "examples" / f"{p['id']}.jpg")).astype(float)
        # the example itself, re-encoded at the manifest's JPEG quality (85, chosen for the landing LCP budget): about 35 dB;
        # a different photo would be far below 30
        assert 10 * np.log10(255 ** 2 / max(((before - src) ** 2).mean(), 1e-9)) > 30
        assert min(p["jpeg_quality"].values()) >= 80
        assert set(p["reported"]) >= {"category", "mean_brightness", "input_noise"}
    assert "make_landing_assets.py" in (ROOT / "app" / "examples" / "SOURCES.md").read_text(encoding="utf-8")


def test_fonts_are_self_hosted_licensed_and_within_budget():
    fonts = list((WEB / "static" / "fonts").glob("*.woff2"))
    assert fonts and sum(f.stat().st_size for f in fonts) <= 120 * 1024
    assert "SIL OPEN FONT LICENSE" in (WEB / "static" / "fonts" / "OFL.txt").read_text(encoding="utf-8").upper()
    assert "Instrument Serif" in (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    css = (WEB / "static" / "css" / "lumen.css").read_text(encoding="utf-8")
    assert "font-display: swap" in css and "url(\"../fonts/instrument-serif-regular.woff2\")" in css


# ------------------------------------------------------------------------------------------- honesty
SUMMARY = ROOT / "results" / "phase_f_hardening" / "summary.json"


@pytest.mark.skipif(not SUMMARY.is_file(), reason="run scripts/hardening_study.py first")
def test_noisy_photo_hint_is_backed_by_the_evidence():
    """The hint fires only inside the measured noisy group, and its sentence matches that group's numbers."""
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    hi = s["strata"]["dev"]["bands"]["high"]
    js = (WEB / "static" / "js" / "workspace.js").read_text(encoding="utf-8")
    threshold = float(re.search(r"const GRAINY = ([0-9.]+);", js).group(1))
    assert threshold >= round(hi["input_noise_range"][0], 3) - 1e-9      # never below the lowest measured noisy photo
    low_max = max(s["strata"][k]["bands"]["low"]["input_noise_range"][1] for k in ("eval15", "lolv2_real_test", "dev"))
    assert threshold > low_max                                         # and above every low-noise photo
    html = (WEB / "index.html").read_text(encoding="utf-8")
    n = int(re.search(r"In our tests on (\d+) photos this grainy", html).group(1))
    assert n == hi["n"]
    for metric in ("psnr", "ssim", "lpips"):
        c = next(c for c in hi["contrasts"] if c["a"] == "qgh" and c["b"] == "retinexformer" and c["metric"] == metric)
        assert c["a_better"] == n, metric                               # "on every one" is literally true


def test_workspace_never_fakes_progress():
    js = (WEB / "static" / "js" / "workspace.js").read_text(encoding="utf-8")
    assert "Math.random" not in js
    assert not re.search(r"progress\s*[+]=|width\s*=\s*`\$\{[^}]*percent", js)    # no invented progress bar


# ------------------------------------------------------------------------------------------- transport
def test_text_is_compressed_but_the_live_stream_and_images_are_not(fake):
    c = client()
    gz = {"Accept-Encoding": "gzip"}
    for path in ("/", "/static/css/lumen.css", "/static/js/workspace.js", "/static/data/evaluation.json"):
        assert c.get(path, headers=gz).headers.get("content-encoding") == "gzip", path
    assert c.get("/static/img/landing/coffee_dark_before.jpg", headers=gz).headers.get("content-encoding") is None
    font = c.get("/static/fonts/instrument-serif-regular.woff2", headers=gz)
    assert font.headers["content-type"] == "font/woff2" and font.headers.get("content-encoding") is None
    r = c.post("/api/restore/stream", files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")}, data={"style": "standard"}, headers=gz)
    assert r.headers.get("content-encoding") is None and parse_sse(r.text)[-1][0] == "done"     # never buffered by gzip
    plain = c.post("/api/restore", files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")}, data={"style": "standard"}, headers=gz)
    assert plain.headers.get("content-encoding") == "gzip" and plain.json()["restored"].startswith("data:image/png")
