"""Streamed restoration: /api/restore/stream reports real stage boundaries and never changes the result."""

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import lumen.deep
from lumen import LumenPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import server as SRV  # noqa: E402
import service as S  # noqa: E402
from test_app import FakeDeep, scene  # noqa: E402
from test_web import client, decode_data_url, jpeg_bytes, restore  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    FakeDeep.calls, FakeDeep.missing = [], set()
    monkeypatch.setattr(lumen.deep, "DeepEnhancer", FakeDeep)
    return FakeDeep


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """[(event, data)] from a Server-Sent Events body (comments such as keep-alives are skipped)."""
    out = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln and not ln.startswith(":")]
        if not lines:
            continue
        ev = next(ln[7:] for ln in lines if ln.startswith("event: "))
        data = next(ln[6:] for ln in lines if ln.startswith("data: "))
        out.append((ev, json.loads(data)))
    return out


def stream(c, style="standard", data=None, name="photo.jpg"):
    return c.post("/api/restore/stream", files={"file": (name, data if data is not None else jpeg_bytes(), "image/jpeg")},
                  data={"style": style})


def names(events):
    return [e for e, _ in events]


# ------------------------------------------------------------------------------------------- event order
@pytest.mark.parametrize("style,expected", [
    ("standard", ["decoded", "analysed", "restored", "refined", "done"]),
    ("quality_guided", ["decoded", "analysed", "restored", "refined", "done"]),
    ("classical", ["decoded", "analysed", "restored", "done"]),
])
def test_stage_order_per_style(fake, style, expected):
    r = stream(client(), style)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-store"
    ev = parse_sse(r.text)
    assert names(ev) == expected
    times = [d["t_ms"] for e, d in ev if e != "done"]
    assert times == sorted(times) and all(t >= 0 for t in times)
    assert ev[0][1]["plan"] == expected[:-1]          # the page lists exactly the steps that will happen


def test_stage_fields_are_real_measurements(fake):
    ev = dict(parse_sse(stream(client(), "quality_guided").text))
    h, w = scene().shape[:2]
    assert (ev["decoded"]["width"], ev["decoded"]["height"]) == (w, h)
    assert ev["decoded"]["source_width"] >= ev["decoded"]["width"]
    assert ev["analysed"]["category"] in {"extremely_dark", "non_uniform", "low_light", "normal"}
    assert 0.0 <= ev["analysed"]["mean_lum"] <= 1.0
    assert ev["restored"]["ms"] >= 0
    ref = ev["refined"]
    assert isinstance(ref["identity"], bool) and set(ref["applied"]) <= set(S.QG_COMPONENTS)
    assert set(ref["refinements"]) == set(ev["done"]["details"]["configured"])
    assert {"applied", "coverage"} <= set(ref["highlight_hue"])


def test_natural_reports_the_highlight_colour_step_and_quick_does_not(fake):
    nat = dict(parse_sse(stream(client(), "standard").text))
    assert set(nat["refined"]) >= {"highlight_hue", "t_ms"} and "refinements" not in nat["refined"]
    assert "refined" not in dict(parse_sse(stream(client(), "classical").text))


def test_no_refined_step_when_the_highlight_fix_is_switched_off(fake, monkeypatch):
    monkeypatch.setenv("LUMEN_HIGHLIGHT_HUE", "0")
    ev = parse_sse(stream(client(), "standard").text)
    assert names(ev) == ["decoded", "analysed", "restored", "done"]
    assert ev[0][1]["plan"] == ["decoded", "analysed", "restored"]


# ------------------------------------------------------------------------------------------- result unchanged
@pytest.mark.parametrize("style", ["standard", "quality_guided", "classical"])
def test_final_image_identical_to_the_plain_endpoint(fake, style):
    c = client()
    data = jpeg_bytes()
    plain = restore(c, style, data).json()
    done = dict(parse_sse(stream(c, style, data).text))["done"]
    for key in ("style", "name", "width", "height", "original", "restored"):
        assert done[key] == plain[key], key          # byte-identical PNG / JPEG data URLs
    assert np.array_equal(decode_data_url(done["restored"]), decode_data_url(plain["restored"]))
    assert done["details"]["applied"] == plain["details"]["applied"]


@pytest.mark.parametrize("mode,kw", [
    ("deep", {}), ("deep", {"highlight_hue": True}),
    ("quality_guided_hybrid", {}), ("quality_guided_hybrid", {"highlight_hue": True}),
    ("classical", {}), ("hybrid", {}),
])
def test_progress_observer_never_changes_the_output(fake, mode, kw):
    pipe = LumenPipeline(max_side=256, deep_model="retinexformer")
    img = scene()
    plain = pipe.run(img, mode, **kw).output
    seen = []
    watched = pipe.run(img, mode, progress=lambda s, d: seen.append(s), **kw).output
    assert np.array_equal(plain, watched)
    assert seen[:3] == ["decoded", "analysed", "restored"]


def test_a_failing_observer_does_not_break_the_restoration(fake):
    pipe = LumenPipeline(max_side=256, deep_model="retinexformer")
    img = scene()

    def boom(stage, data):
        raise RuntimeError("observer bug")

    assert np.array_equal(pipe.run(img, "deep").output, pipe.run(img, "deep", progress=boom).output)


def test_analysis_is_only_computed_when_observed(fake, monkeypatch):
    import lumen.classical

    calls = []
    real = lumen.classical.analyze_image
    monkeypatch.setattr(lumen.classical, "analyze_image", lambda x: calls.append(1) or real(x))
    pipe = LumenPipeline(max_side=256, deep_model="retinexformer")
    pipe.run(scene(), "deep")
    assert calls == []
    pipe.run(scene(), "deep", progress=lambda s, d: None)
    assert calls == [1]


# ------------------------------------------------------------------------------------------- errors
def test_bad_photo_arrives_as_an_error_event(fake):
    r = stream(client(), data=b"this is not an image", name="x.jpg")
    assert r.status_code == 200
    ev = parse_sse(r.text)
    assert names(ev)[-1] == "error" and "done" not in names(ev)
    err = ev[-1][1]
    assert err["status"] == 422 and err["code"] == "invalid_image" and "not a recognised image" in err["error"]
    assert fake.calls == []


def test_unavailable_style_arrives_as_an_error_event(fake):
    fake.missing = {"retinexformer"}
    c = client()
    err = parse_sse(stream(c, "standard").text)[-1]
    assert err[0] == "error" and err[1]["status"] == 409 and err[1]["code"] == "unavailable"
    assert names(parse_sse(stream(c, "classical").text))[-1] == "done"


def test_problems_before_the_work_starts_are_plain_json_errors(fake, monkeypatch):
    c = client()
    assert c.post("/api/restore/stream", data={"style": "standard"}).status_code == 400
    r = stream(c, "turbo")
    assert r.status_code == 400 and "Unknown restoration style" in r.json()["error"]
    monkeypatch.setattr(SRV, "DEFAULT_MAX_FILE_BYTES", 1000)
    assert stream(c).status_code == 413
    monkeypatch.undo()
    monkeypatch.setattr(SRV, "MAX_QUEUE", 0)
    r = stream(client())
    assert r.status_code == 503 and "busy" in r.json()["error"]


def test_uploads_are_deleted_after_a_stream(fake):
    tmp = Path(tempfile.gettempdir())
    before = set(tmp.glob("lumen_upload_*"))
    c = client()
    stream(c)
    stream(c, data=b"not an image", name="bad.jpg")
    deadline = time.time() + 2     # the done-callback runs just after the last event
    while set(tmp.glob("lumen_upload_*")) - before and time.time() < deadline:
        time.sleep(0.05)
    assert set(tmp.glob("lumen_upload_*")) - before == set()


# ------------------------------------------------------------------------------------------- queue
def test_a_second_photo_is_told_it_is_queued(fake, monkeypatch):
    """One restoration runs at a time: while the first is busy, the second reports `queued`, then completes."""
    started, release = threading.Event(), threading.Event()
    real = S.EnhancementService.enhance

    def slow(self, image, mode=S.STANDARD, compute_niqe=False, progress=None):
        started.set()
        assert release.wait(10)
        return real(self, image, mode, compute_niqe, progress)

    monkeypatch.setattr(S.EnhancementService, "enhance", slow)
    results = {}
    with client() as c:                         # one shared event loop, so the two requests really contend
        def go(key):
            results[key] = stream(c)
        a = threading.Thread(target=go, args=("a",))
        a.start()
        assert started.wait(10)
        b = threading.Thread(target=go, args=("b",))
        b.start()
        time.sleep(1.0)                         # b has reached the gate and is waiting
        release.set()
        a.join(20)
        b.join(20)
    ev_a, ev_b = parse_sse(results["a"].text), parse_sse(results["b"].text)
    assert "queued" not in names(ev_a) and names(ev_a)[-1] == "done"
    assert names(ev_b)[0] == "queued" and ev_b[0][1] == {"ahead": 1} and names(ev_b)[-1] == "done"
    assert SRV.MAX_QUEUE >= 1


def test_keepalive_comments_are_sent_while_waiting(fake, monkeypatch):
    monkeypatch.setattr(SRV, "KEEPALIVE_SECONDS", 0.05)
    started, release = threading.Event(), threading.Event()
    real = S.EnhancementService.enhance

    def slow(self, image, mode=S.STANDARD, compute_niqe=False, progress=None):
        started.set()
        assert release.wait(10)
        return real(self, image, mode, compute_niqe, progress)

    monkeypatch.setattr(S.EnhancementService, "enhance", slow)
    out = {}
    with client() as c:
        t = threading.Thread(target=lambda: out.setdefault("r", stream(c)))
        t.start()
        assert started.wait(10)
        time.sleep(0.4)
        release.set()
        t.join(20)
    assert ": keepalive" in out["r"].text and names(parse_sse(out["r"].text))[-1] == "done"
