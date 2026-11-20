"""HTTP layer of the LUMEN web app: a small JSON API plus the static frontend (app/web/).

The restoration itself is unchanged and lives in app/service.py (EnhancementService); this module only
translates HTTP to service calls:

  GET  /                    restoration workspace (app/web/index.html)
  GET  /details             technical details: your last photo, research lab, system status
  GET  /how                 how LUMEN works (pipeline, styles, privacy, limits)
  GET  /evidence            the evaluation, as charts generated from the results files
  GET  /api/status          user-facing styles + technical status
  GET  /api/examples        example photos
  POST /api/restore         multipart (file, style)            -> original + restored image (+ details)
  POST /api/restore/stream  same input; Server-Sent Events: queued?, decoded, analysed, restored, refined?, done
                            (done carries exactly the /api/restore payload) or error
  POST /api/compare        multipart (file)                    -> every available style, Retinexformer once
  POST /api/research        multipart (file, method, disabled)  -> research baselines (fusion hybrid, Zero-DCE++)
  GET  /healthz             liveness

Uploads are streamed to a private temporary file (size-limited) and deleted after the request; image
validation, decoding limits and checkpoint/config checks are those of the service layer. One restoration runs
at a time (the models are not re-entrant and peak RAM must stay bounded); up to MAX_QUEUE requests wait.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from lumen.imageio import DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_PIXELS, SUPPORTED_TEXT, ImageValidationError
from service import (
    CLASSICAL, QUALITY_GUIDED, RESEARCH, STANDARD, EnhanceResult, EnhancementService, ModeUnavailable, niqe_available,
)

log = logging.getLogger("lumen.app")
mimetypes.add_type("font/woff2", ".woff2")   # not in every platform's MIME table; fonts must not be sent as octet-stream
APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
EXAMPLES_DIR = APP_DIR / "examples"
MAX_QUEUE = int(os.environ.get("LUMEN_MAX_QUEUE", "16"))
KEEPALIVE_SECONDS = 10  # SSE comment sent while a queued request waits, so proxies keep the connection open

# User-facing names: plain language, no model or research terminology.
STYLES = {
    STANDARD: {"name": "Natural", "recommended": True,
               "summary": "The best overall restoration. Recommended for most photos."},
    QUALITY_GUIDED: {"name": "Highlight-safe", "recommended": False,
                     "summary": "Same restoration, with extra care for lamps, windows and white surfaces."},
    CLASSICAL: {"name": "Quick", "recommended": False,
                "summary": "Lightweight and fast, but less accurate on very dark photos."},
}
# "What changed" overlays: only refinements that really changed the photo are sent. Plain language, no model terms.
OVERLAYS = {
    "highlight_hue": ("Colour kept in bright spots", "#3dd68c"),
    "highlight": ("Bright areas softened", "#ff5fa2"),
    "noise": ("Grain reduced", "#4cc9f0"),
    "colour": ("Colour nudged back", "#f6b73c"),
    "contrast": ("Detail sharpened", "#b794f6"),
}
# "Look closer": the light map that comes with every result, small enough to always send.
LIGHT_MAP_MAX_SIDE = 384
LIGHT_MAP_MAX_BYTES = 60 * 1024    # the whole data URL as sent
LIGHT_MAP_LABELS = {"added": "Where light was added", "scene": "Light already in the photo"}
OVERLAY_MAX_SIDE = 512     # the masks are feathered, so a smaller picture scaled up by the browser loses nothing
OVERLAY_OPACITY = 0.55     # strong enough to see where, light enough that the photo stays readable underneath
EXAMPLE_TITLES = {
    "coffee_dark": "Café table", "chelsea_extremely_dark": "Very dark room",
    "astronaut_uneven_light": "Uneven light", "rocket_night_highlights": "Night with bright lights",
}


class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


# ---------------------------------------------------------------------------------------------- helpers
def _data_url(rgb: np.ndarray, fmt: str) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png" if fmt == "png" else ".jpg", bgr,
                           [] if fmt == "png" else [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ApiError(500, "The restored image could not be encoded.", "encode_failed")
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _overlay_url(mask: np.ndarray, tint: str) -> str:
    """Transparent PNG: the tint colour, with the refinement's soft mask as its opacity."""
    h, w = mask.shape
    scale = min(1.0, OVERLAY_MAX_SIDE / max(h, w))
    if scale < 1.0:
        mask = cv2.resize(mask, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    rgba = np.empty((*mask.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = (int(tint[i:i + 2], 16) for i in (5, 3, 1))  # BGR order for OpenCV
    rgba[..., 3] = np.clip(mask * OVERLAY_OPACITY * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise ApiError(500, "The change overlay could not be encoded.", "encode_failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _overlays(r: EnhanceResult) -> list[dict]:
    return [{"id": key, "label": label, "color": tint, "coverage": round(r.masks[key][1], 5),
             "image": _overlay_url(r.masks[key][0], tint)}
            for key, (label, tint) in OVERLAYS.items() if key in r.masks]


def _light_map_url(m: np.ndarray) -> str:
    """Grayscale JPEG, at most LIGHT_MAP_MAX_SIDE px and LIGHT_MAP_MAX_BYTES as a data URL (quality steps down if needed)."""
    h, w = m.shape
    scale = min(1.0, LIGHT_MAP_MAX_SIDE / max(h, w))
    if scale < 1.0:
        m = cv2.resize(m.astype(np.float32), (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(m, [2, 98])                     # stretched for display: relative to the rest of the photo
    m = (m - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(m)
    gray = np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8)
    url = ""
    for quality in (90, 80, 70, 60, 50, 40):
        ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ApiError(500, "The light map could not be encoded.", "encode_failed")
        url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        if len(url) <= LIGHT_MAP_MAX_BYTES:
            break
    return url


def _maps(r: EnhanceResult) -> dict:
    if r.light_map is None:
        return {}
    return {"light": {"kind": r.light_map_kind, "label": LIGHT_MAP_LABELS[r.light_map_kind], "image": _light_map_url(r.light_map)}}


def _details(r: EnhanceResult) -> dict:
    """Technical information for the Details page (never shown in the workspace)."""
    h, w = r.original.shape[:2]
    return {
        "mode": r.mode, "technical_label": r.label, "model": r.model, "checkpoint": r.checkpoint,
        "device": r.device, "fp16": r.fp16, "timing_ms": r.timing_ms, "processed_size": [w, h],
        "refinements": r.refinements, "identity": r.identity, "applied": r.applied, "configured": r.configured,
        "qg_config": Path(r.qg_config).name if r.qg_config else None, "metrics": r.metrics, "note": r.note,
        "highlight_hue": r.info.get("highlight_hue"),
    }


def _payload(r: EnhanceResult, style_key: str | None = None) -> dict:
    h, w = r.enhanced.shape[:2]
    style = STYLES.get(style_key or r.mode, {"name": r.label})
    return {"style": style_key or r.mode, "name": style["name"], "width": w, "height": h,
            "elapsed_ms": round(r.latency_ms, 1), "original": _data_url(r.original, "jpg"),
            "restored": _data_url(r.enhanced, "png"), "overlays": _overlays(r), "maps": _maps(r), "details": _details(r)}


def _json_default(o):
    """numpy scalars/arrays that end up in diagnostics."""
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'), default=_json_default)}\n\n"


async def _save_upload(upload: UploadFile | None) -> Path:
    if upload is None or not upload.filename:
        raise ApiError(400, "Please choose a photo to restore.", "no_file")
    limit = DEFAULT_MAX_FILE_BYTES
    suffix = "".join(c for c in Path(upload.filename).suffix[:10] if c.isalnum() or c == ".")
    fd, name = tempfile.mkstemp(prefix="lumen_upload_", suffix=suffix)
    path, total = Path(name), 0
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await upload.read(1 << 20):
                total += len(chunk)
                if total > limit:
                    raise ApiError(413, f"This photo is larger than {limit // 2**20} MB. Please choose a smaller file.",
                                   "too_large")
                f.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


# ---------------------------------------------------------------------------------------------- app
def create_app(service: EnhancementService) -> FastAPI:
    app = FastAPI(title="LUMEN", docs_url=None, redoc_url=None, openapi_url=None)
    # text (pages, CSS, JS, JSON results) is compressed; the live event stream must not be buffered, and images and fonts
    # are compressed already
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6,
                       exclude_content_types=("text/event-stream", "image/jpeg", "image/png", "image/webp", "font/woff2"))
    state = {"waiting": 0}
    gate = asyncio.Semaphore(1)

    async def run_exclusive(fn, *args):
        if state["waiting"] >= MAX_QUEUE:
            raise ApiError(503, "LUMEN is busy restoring other photos. Please try again in a moment.", "busy")
        state["waiting"] += 1
        try:
            async with gate:
                return await run_in_threadpool(fn, *args)
        finally:
            state["waiting"] -= 1

    async def with_upload(file: UploadFile | None, fn):
        path = await _save_upload(file)
        try:
            return await run_exclusive(fn, path)
        except ImageValidationError as e:
            raise ApiError(422, str(e), "invalid_image") from e
        except ModeUnavailable as e:
            raise ApiError(409, str(e), "unavailable") from e
        finally:
            path.unlink(missing_ok=True)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return JSONResponse({"error": e.message, "code": e.code}, status_code=e.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, e: StarletteHTTPException):
        path = request.url.path
        if e.status_code == 404 and request.method == "GET" and not path.startswith(("/api/", "/static/", "/examples/")):
            return FileResponse(WEB_DIR / "404.html", status_code=404, headers={"Cache-Control": "no-cache"})
        code = "not_found" if e.status_code == 404 else "http_error"
        return JSONResponse({"error": str(e.detail), "detail": e.detail, "code": code}, status_code=e.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, e: Exception):  # pragma: no cover - logged, generic message
        log.exception("Unexpected error")
        return JSONResponse({"error": "Something went wrong while restoring this photo. Please try another one.",
                             "code": "internal"}, status_code=500)

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        length = request.headers.get("content-length")
        if request.method == "POST" and length and length.isdigit() and int(length) > DEFAULT_MAX_FILE_BYTES + 2**16:
            return JSONResponse({"error": f"This photo is larger than {DEFAULT_MAX_FILE_BYTES // 2**20} MB. "
                                          "Please choose a smaller file.", "code": "too_large"}, status_code=413)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith("/static/"):
            # revalidate on every load (cheap 304 via ETag) so an update never leaves stale CSS/JS in browsers
            response.headers["Cache-Control"] = "no-cache"
        return response

    # -- pages -------------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/details", include_in_schema=False)
    async def details():
        return FileResponse(WEB_DIR / "details.html", headers={"Cache-Control": "no-cache"})

    @app.get("/how", include_in_schema=False)
    async def how():
        return FileResponse(WEB_DIR / "how.html", headers={"Cache-Control": "no-cache"})

    @app.get("/evidence", include_in_schema=False)
    async def evidence():
        return FileResponse(WEB_DIR / "evidence.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    examples = {p.stem: p for p in sorted(EXAMPLES_DIR.glob("*.jpg"))}

    @app.get("/examples/{name}", include_in_schema=False)
    async def example_file(name: str):
        p = examples.get(name.removesuffix(".jpg"))  # whitelist lookup: no path handling at all
        if p is None:
            raise ApiError(404, "Example not found.", "not_found")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    # -- API ----------------------------------------------------------------------------------
    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/api/examples")
    async def api_examples():
        return [{"id": k, "title": EXAMPLE_TITLES.get(k, k.replace("_", " ").title()), "url": f"/examples/{k}.jpg"}
                for k in examples]

    @app.get("/api/status")
    async def api_status():
        st = service.status
        styles = [{"id": m, **STYLES[m], "available": st[m].available, "reason": st[m].reason}
                  for m in (STANDARD, QUALITY_GUIDED, CLASSICAL)]
        research = [{"id": k, "label": v[2], "available": service.research_status(k).available,
                     "reason": service.research_status(k).reason} for k, v in RESEARCH.items()]
        technical = {"checkpoint_check": service.checkpoint_check, "max_side": service.pipe.max_side,
                     "niqe_available": niqe_available(),
                     "qg_config": Path(service.pipe.qg_config_source).name if service.pipe.qg_config_source else None,
                     "highlight_hue": {"enabled": service.hue_enabled, "reason": service.hue_reason}}
        if st[STANDARD].available:
            enh = service.pipe.get_deep("retinexformer")
            technical |= {"model": "Retinexformer (ICCV 2023)", "checkpoint": enh.weights_path.name,
                          "device": str(enh.device), "fp16": enh.fp16, "engine": getattr(enh, "engine", "torch"), "engine_note": getattr(enh, "engine_note", "")}
        return {"styles": styles, "research": research, "technical": technical,
                "limits": {"max_file_mb": DEFAULT_MAX_FILE_BYTES // 2**20, "max_megapixels": DEFAULT_MAX_PIXELS / 1e6,
                           "formats": SUPPORTED_TEXT}}

    @app.post("/api/restore")
    async def api_restore(file: UploadFile | None = File(None), style: str = Form(STANDARD)):
        if style not in STYLES:
            raise ApiError(400, "Unknown restoration style.", "bad_style")
        r = await with_upload(file, lambda p: service.enhance(p, style))
        return _payload(r, style)

    @app.post("/api/restore/stream")
    async def api_restore_stream(file: UploadFile | None = File(None), style: str = Form(STANDARD)):
        """Same restoration as /api/restore, reported as it happens. Problems found before the work starts (no file,
        unknown style, too large, queue full) are ordinary JSON errors; problems with the photo itself arrive as an
        `error` event, because the stream is already open by then."""
        if style not in STYLES:
            raise ApiError(400, "Unknown restoration style.", "bad_style")
        path = await _save_upload(file)
        if state["waiting"] >= MAX_QUEUE:
            path.unlink(missing_ok=True)
            raise ApiError(503, "LUMEN is busy restoring other photos. Please try again in a moment.", "busy")

        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        ahead = state["waiting"]          # photos running or waiting before this one
        state["waiting"] += 1             # counted from now, exactly like run_exclusive()
        counted = [True]

        def put(kind: str, data: dict | None) -> None:
            loop.call_soon_threadsafe(events.put_nowait, (kind, data))

        def release() -> None:
            if counted[0]:
                counted[0] = False
                state["waiting"] -= 1

        # the steps this style will report, so the page lists only the ones that really happen
        plan = ["decoded", "analysed", "restored"]
        if style == QUALITY_GUIDED or (style == STANDARD and service.hue_enabled):
            plan.append("refined")

        def job():
            t0 = time.perf_counter()

            def progress(stage: str, data: dict) -> None:
                extra = {"plan": plan} if stage == "decoded" else {}
                put(stage, {**data, **extra, "t_ms": round((time.perf_counter() - t0) * 1000)})

            return service.enhance(path, style, progress=progress)

        async def worker() -> None:
            try:
                if ahead:
                    put("queued", {"ahead": ahead})
                try:
                    async with gate:
                        result = await run_in_threadpool(job)
                finally:
                    release()
                put("done", await run_in_threadpool(_payload, result, style))
            except ImageValidationError as e:
                put("error", {"error": str(e), "code": "invalid_image", "status": 422})
            except ModeUnavailable as e:
                put("error", {"error": str(e), "code": "unavailable", "status": 409})
            except ApiError as e:
                put("error", {"error": e.message, "code": e.code, "status": e.status})
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - logged; the user sees a generic message
                log.exception("Unexpected error while streaming a restoration")
                put("error", {"error": "Something went wrong while restoring this photo. Please try another one.",
                              "code": "internal", "status": 500})
            finally:
                put("end", None)

        task = asyncio.create_task(worker())
        # Runs even if the task is cancelled before it starts: the upload is always deleted, the slot always freed.
        task.add_done_callback(lambda _t: (path.unlink(missing_ok=True), release()))

        async def stream():
            try:
                while True:
                    try:
                        kind, data = await asyncio.wait_for(events.get(), KEEPALIVE_SECONDS)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    if kind == "end":
                        break
                    yield _sse(kind, data)
            finally:
                if not task.done():       # the visitor left: stop waiting in the queue (a running model call finishes)
                    task.cancel()

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/api/compare")
    async def api_compare(file: UploadFile | None = File(None)):
        results = await with_upload(file, service.compare_all)
        return {"results": [_payload(r) for r in results]}

    @app.post("/api/research")
    async def api_research(file: UploadFile | None = File(None), method: str = Form(...), disabled: str = Form("")):
        parts = [d for d in disabled.split(",") if d]
        r = await with_upload(file, lambda p: service.research(p, method, parts))
        return _payload(r, method)

    return app
