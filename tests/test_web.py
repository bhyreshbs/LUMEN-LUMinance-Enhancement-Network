"""Web layer: pages, static assets, JSON API, validation, limits, privacy, evaluation data, determinism."""

import base64
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import lumen.deep
from lumen import LumenPipeline

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "app" / "web"
sys.path.insert(0, str(ROOT / "app"))
import server as SRV  # noqa: E402
import service as S  # noqa: E402
from test_app import FakeDeep, scene  # noqa: E402

JARGON = ["retinexformer", "psnr", "ssim", "lpips", "niqe", "checkpoint", "qgh", "quality-guided", "quality guided",
          "fp16", "cuda", "ablation", "zero-dce", "transformer", "clahe", "sha-256"]


@pytest.fixture
def fake(monkeypatch):
    FakeDeep.calls, FakeDeep.missing = [], set()
    monkeypatch.setattr(lumen.deep, "DeepEnhancer", FakeDeep)
    return FakeDeep


def client(require_verified=False):
    svc = S.EnhancementService(LumenPipeline(max_side=256, deep_model="retinexformer"), require_verified=require_verified)
    return TestClient(SRV.create_app(svc))


def jpeg_bytes(img=None) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(scene() if img is None else img, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def decode_data_url(url: str) -> np.ndarray:
    head, b64 = url.split(",", 1)
    arr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def restore(c, style="standard", data=None, name="photo.jpg"):
    return c.post("/api/restore", files={"file": (name, data if data is not None else jpeg_bytes(), "image/jpeg")},
                  data={"style": style})


# ------------------------------------------------------------------------------------------- pages & assets
def test_pages_and_every_referenced_asset_resolve(fake):
    c = client()
    for page in ("/", "/details"):
        r = c.get(page)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        refs = set(re.findall(r'(?:src|href)="(/[^"#]*)"', r.text))
        assert any(x.endswith(".js") for x in refs) and any(x.endswith(".css") for x in refs)
        for ref in refs:
            assert c.get(ref).status_code == 200, f"{page} references missing {ref}"


def test_frontend_uses_no_external_resources():
    for f in [*WEB.glob("*.html"), *(WEB / "static").rglob("*.js"), *(WEB / "static").rglob("*.css")]:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", text), f"external URL in {f.name}"
        assert "<style" not in text and " style=" not in text, f"inline style (blocked by CSP) in {f.name}"
        assert "<script>" not in text and "onclick=" not in text, f"inline script (blocked by CSP) in {f.name}"


def test_workspace_has_no_technical_jargon():
    """The main workspace must not show model, metric or research vocabulary."""
    visible = (WEB / "index.html").read_text(encoding="utf-8")
    visible = re.sub(r"<!--.*?-->", "", visible, flags=re.S)
    js = (WEB / "static" / "js" / "workspace.js").read_text(encoding="utf-8")
    js_strings = " ".join(re.findall(r"`([^`]*)`|\"([^\"]*)\"", js) and
                          [a or b for a, b in re.findall(r"`([^`]*)`|\"([^\"]*)\"", js)])
    styles = " ".join(f"{s['name']} {s['summary']}" for s in SRV.STYLES.values())
    for text, where in ((visible, "index.html"), (js_strings, "workspace.js strings"), (styles, "style names")):
        low = text.lower()
        found = [j for j in JARGON if j in low]
        assert not found, f"jargon {found} in {where}"


def test_details_page_holds_the_technical_sections():
    html = (WEB / "details.html").read_text(encoding="utf-8")
    for anchor in ('id="how"', 'id="last"', 'id="evaluation"', 'id="lab"', 'id="system"'):
        assert anchor in html


def test_security_headers(fake):
    r = client().get("/")
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["referrer-policy"] == "no-referrer"
    assert client().get("/api/status").headers["cache-control"] == "no-store"
    css = client().get("/static/css/lumen.css")
    assert css.headers["cache-control"] == "no-cache" and css.headers.get("etag")  # revalidated, never stale


# ------------------------------------------------------------------------------------------- status & examples
def test_status_lists_plain_language_styles(fake):
    st = client().get("/api/status").json()
    assert [s["id"] for s in st["styles"]] == ["standard", "quality_guided", "classical"]
    assert [s["name"] for s in st["styles"]] == ["Natural", "Highlight-safe", "Quick"]
    assert st["styles"][0]["recommended"] and all(s["available"] for s in st["styles"])
    assert {r["id"] for r in st["research"]} == set(S.RESEARCH)
    assert st["limits"]["max_file_mb"] == 20 and st["technical"]["model"].startswith("Retinexformer")


def test_examples_listed_and_served_and_no_traversal(fake):
    c = client()
    ex = c.get("/api/examples").json()
    assert len(ex) == 4 and all(c.get(e["url"]).status_code == 200 for e in ex)
    for bad in ("/examples/..%2F..%2Fweights%2FREADME.md", "/examples/nope.jpg", "/examples/../README.md"):
        assert c.get(bad).status_code == 404


# ------------------------------------------------------------------------------------------- restore
@pytest.mark.parametrize("style,name", [("standard", "Natural"), ("quality_guided", "Highlight-safe"), ("classical", "Quick")])
def test_restore_each_style(fake, style, name):
    r = restore(client(), style)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["style"] == style and d["name"] == name
    out, orig = decode_data_url(d["restored"]), decode_data_url(d["original"])
    assert out.shape == orig.shape == (d["height"], d["width"], 3)
    assert d["restored"].startswith("data:image/png;base64,")
    assert d["details"]["mode"] == style and "timing_ms" in d["details"]
    assert fake.calls == ([] if style == "classical" else ["retinexformer"])


def test_quality_guided_details_reach_the_details_page(fake):
    d = restore(client(), "quality_guided").json()["details"]
    assert set(d["refinements"]) == set(d["configured"]) and isinstance(d["identity"], bool)
    assert d["qg_config"] == "quality_guided.yaml"  # file name only, never a server path


@pytest.mark.parametrize("payload,name,status,needle", [
    (b"this is not an image", "x.jpg", 422, "not a recognised image"),
    (b"", "empty.png", 422, "empty"),
    (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01"
     b"\x00\x01\x00\x00\x02\x02D\x01\x00;", "a.gif", 422, "Unsupported image format"),
])
def test_invalid_uploads_get_clear_errors(fake, payload, name, status, needle):
    r = restore(client(), data=payload, name=name)
    assert r.status_code == status and needle in r.json()["error"]
    assert fake.calls == []


def test_missing_file_and_unknown_style(fake):
    c = client()
    assert c.post("/api/restore", data={"style": "standard"}).status_code == 400
    r = restore(c, style="turbo")
    assert r.status_code == 400 and "Unknown restoration style" in r.json()["error"]


def test_upload_size_limit(fake, monkeypatch):
    monkeypatch.setattr(SRV, "DEFAULT_MAX_FILE_BYTES", 1000)
    r = restore(client(), data=jpeg_bytes())
    assert r.status_code == 413 and "larger than" in r.json()["error"]


def test_busy_queue_returns_503(fake, monkeypatch):
    monkeypatch.setattr(SRV, "MAX_QUEUE", 0)
    r = restore(client())
    assert r.status_code == 503 and "busy" in r.json()["error"]


def test_unavailable_style_is_reported(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_QG_CONFIG", str(tmp_path / "missing.yaml"))
    c = client()
    st = {s["id"]: s for s in c.get("/api/status").json()["styles"]}
    assert not st["quality_guided"]["available"] and st["standard"]["available"]
    r = restore(c, "quality_guided")
    assert r.status_code == 409 and "disabled" in r.json()["error"]
    assert restore(c, "standard").status_code == 200


def test_missing_checkpoint_leaves_quick_only(fake):
    fake.missing = {"retinexformer"}
    c = client()
    assert [s["available"] for s in c.get("/api/status").json()["styles"]] == [False, False, True]
    assert restore(c, "standard").status_code == 409 and restore(c, "classical").status_code == 200
    assert fake.calls == []


def test_uploads_are_deleted_after_each_request(fake):
    tmp = Path(tempfile.gettempdir())
    before = set(tmp.glob("lumen_upload_*"))
    c = client()
    restore(c)
    restore(c, data=b"not an image", name="bad.jpg")
    assert set(tmp.glob("lumen_upload_*")) - before == set()


# ------------------------------------------------------------------------------------------- compare & research
def test_compare_runs_the_network_once(fake):
    r = client().post("/api/compare", files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")})
    res = r.json()["results"]
    assert [x["style"] for x in res] == ["standard", "quality_guided", "classical"]
    assert fake.calls == ["retinexformer"]


def test_research_endpoint(fake):
    c = client()
    r = c.post("/api/research", files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
               data={"method": "hybrid_retinexformer", "disabled": "post_color"})
    assert r.status_code == 200 and r.json()["details"]["model"] == "retinexformer"
    bad = c.post("/api/research", files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
                 data={"method": "hybrid_retinexformer", "disabled": "noise"})
    assert bad.status_code == 409
    assert c.post("/api/research", files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
                  data={"method": "nope"}).status_code == 409


# ------------------------------------------------------------------------------------------- evaluation data
def test_details_evaluation_numbers_match_the_report():
    ev = json.loads((WEB / "static" / "data" / "evaluation.json").read_text(encoding="utf-8"))
    report = (ROOT / "eval_P1_T66.md").read_text(encoding="utf-8")
    cols = ["psnr", "ssim", "lpips", "niqe", "latency_ms", "noise", "edge", "colour", "overexposed"]
    for m in ev["methods"]:
        line = next(ln for ln in report.splitlines() if ln.startswith(f"| {m['eval_label']} |"))
        cells = [c.strip().strip("*") for c in line.split("|")[2:-1]]
        row = dict(zip(cols, cells))
        for k in ("psnr", "ssim", "lpips", "niqe", "latency_ms", "overexposed"):
            assert float(row[k]) == pytest.approx(m[k]), (m["name"], k)


# ------------------------------------------------------------------------------------------- determinism
@pytest.mark.skipif(not (ROOT / "weights" / "retinexformer_LOL_v1.pth").is_file(), reason="checkpoint not present")
@pytest.mark.parametrize("style", ["standard", "quality_guided", "classical"])
def test_api_output_identical_to_direct_service_call(style):
    """The redesign must not change restorations: API output == service output, byte for byte, and repeatable."""
    svc = S.EnhancementService(LumenPipeline(max_side=256, device="cpu", deep_model="retinexformer"))
    c = TestClient(SRV.create_app(svc))
    data = (ROOT / "app" / "examples" / "coffee_dark.jpg").read_bytes()
    a = decode_data_url(restore(c, style, data).json()["restored"])
    b = decode_data_url(restore(c, style, data).json()["restored"])
    direct = svc.enhance(str(ROOT / "app" / "examples" / "coffee_dark.jpg"), style).enhanced
    assert np.array_equal(a, b) and np.array_equal(a, direct)


def test_app_no_longer_imports_gradio():
    code = ("import sys; sys.path[:0] = [r'%s', r'%s']; import app, server; "
            "print('gradio' in sys.modules)") % (ROOT, ROOT / "app")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
