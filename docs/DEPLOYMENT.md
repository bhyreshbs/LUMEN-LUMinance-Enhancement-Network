# LUMEN - Deployment guide

**Verification status (2026-11-06):** the production image was built from this Dockerfile and tested as a
running container on Docker Desktop 29.9.2 (WSL 2 backend, x86_64): all modes, Compare All, invalid uploads,
missing, tampered and altered checkpoints, missing and invalid QGH config, NIQE behaviour, concurrency,
resource use at 768 px, and the browser UI in headless Chrome. The redesigned UI (FastAPI + static frontend,
replacing Gradio) was re-verified in the container on the same date. See the *Verification log*. Not yet
verified: any hosting platform, any GPU, real mobile devices.

## What the container needs

| Item | Requirement |
|---|---|
| Image | `docker build -t lumen .` → **2.02 GB** (CPU PyTorch 2.14.1 + FastAPI/Uvicorn; was 2.32 GB with Gradio). Cold build 181 s on the test machine (Gradio image) |
| Checkpoint | `retinexformer_LOL_v1.pth` (6.48 MB) is **not in the image or the repository** (redistribution rights uncertain, see `weights/README.md`). Download it with `python scripts/download_weights.py --runtime` and mount it read-only at `/models` (`LUMEN_RETINEXFORMER_WEIGHTS=/models/retinexformer_LOL_v1.pth`). Its SHA-256 is verified at startup |
| Config | `configs/quality_guided.yaml` (calibrated) and `configs/classical.yaml` are baked into the image; the build fails if the QGH config is invalid |
| CPU / RAM | **2-4 vCPUs, 3 GB memory limit** recommended (measurements below). No GPU needed |
| Network | None at runtime (NIQE is not installed in the default image) |

```bash
python scripts/download_weights.py --runtime        # once: fetch + verify the checkpoint into weights/
docker build -t lumen .                             # --build-arg INSTALL_NIQE=1 adds the optional NIQE metric
docker run --rm -p 7860:7860 --cpus 2 --memory 3g -v "$PWD/weights:/models:ro" lumen
```

| Env var | Image default | Meaning |
|---|---|---|
| `LUMEN_MAX_SIDE` | 768 | processing size (longest side) |
| `LUMEN_TORCH_THREADS` | from the cgroup CPU quota | PyTorch/OpenCV threads; see *CPU threads* |
| `LUMEN_RETINEXFORMER_WEIGHTS` | `/models/retinexformer_LOL_v1.pth` | checkpoint location |
| `LUMEN_QG_CONFIG` | `configs/quality_guided.yaml` | calibrated Quality Guided config |
| `LUMEN_MAX_PIXELS`, `LUMEN_MAX_FILE_MB` | 40e6, 20 | upload limits |
| `PORT`, `LUMEN_HOST` | 7860, 0.0.0.0 | listen port and address |
| `LUMEN_MAX_QUEUE` | 16 | requests allowed to wait while one restoration runs; more get HTTP 503 "busy" |

## HTTP interface

| Endpoint | Purpose |
|---|---|
| `GET /` | restoration workspace |
| `GET /details` | *How it works*: technical details of the last restoration, evaluation, research lab, system status |
| `GET /healthz` | liveness (Docker `HEALTHCHECK`) |
| `GET /api/status` | styles with availability and reason, research methods, checkpoint check, limits |
| `POST /api/restore` | multipart `file`, `style` (`standard` / `quality_guided` / `classical`) → original (JPEG) + restored (PNG) data URLs and technical details |
| `POST /api/compare` | multipart `file` → every available style, with Retinexformer run once |
| `POST /api/research` | multipart `file`, `method`, `disabled` (ablation parts) → research baselines |

Errors are JSON `{"error", "code"}`: 400 (no file, unknown style), 409 (style unavailable), 413 (larger than
20 MB, also rejected from `Content-Length` before reading), 422 (invalid image, LUMEN's validation message),
503 (queue full). Every response carries a strict Content-Security-Policy (`'self'` only; no external scripts,
styles or connections), `nosniff`, `no-referrer` and a Permissions-Policy; API responses are `no-store`.

## Behaviour when something is wrong (verified in the container)

| Situation | Result |
|---|---|
| No checkpoint mounted | App starts. Only **Quick** (Classical) is offered; the other styles say "Not available right now", and the log and the *System* section of `/details` say `No retinexformer checkpoint found (looked for: /models/retinexformer_LOL_v1.pth)` |
| Corrupt checkpoint (fails to load) | Same: Standard and Quality Guided unavailable (`could not be loaded: ...`) |
| Altered checkpoint (loads, but not the official file) | Same: rejected by SHA-256 (`SHA-256 mismatch (expected 9b4810dc5132..., got ...)`). **Random or altered weights never run** |
| QGH config missing or invalid | **Only Quality Guided** is disabled, with the file path and the reason; Standard, Classical and Compare All keep working. **Never falls back to uncalibrated thresholds** |
| pyiqa not installed (default image) | `/details` reports the NIQE metric as not installed |
| Invalid upload | Clear message for: non-image file, empty file, truncated JPEG, > 40 MP, GIF, < 16 px (all six verified through real uploads) |

## Measured performance (container, 768x512 input, 2 warm-up + 10 timed requests per row)

Fresh container per row, so the peak memory is that mode's peak (cgroup `memory.peak`). "Server" is the
processing time reported by the app; "round trip" adds upload, encoding and HTTP. Host: Intel 12th-gen
(16 logical cores) laptop under Docker Desktop. Raw data: `results/container_benchmark.json`.

| CPUs | Mode | Server median | Server p95 | Round-trip median | Round-trip p95 | Peak memory |
|---|---|---|---|---|---|---|
| 2 | Standard | 6.33 s | 7.05 s | 6.82 s | 7.42 s | 2056 MB |
| 2 | Quality Guided | 5.67 s | 5.89 s | 6.16 s | 6.32 s | 1994 MB |
| 2 | Classical | 0.13 s | 0.15 s | 0.57 s | 0.68 s | 386 MB |
| 2 | Compare All (Retinexformer once) | 6.22 s | 7.66 s | 6.86 s | 8.83 s | 2560 MB* |
| 4 | Standard | 3.76 s | 3.91 s | 4.21 s | 4.35 s | 1941 MB |
| 4 | Quality Guided | 3.74 s | 4.09 s | 4.19 s | 4.52 s | 1907 MB |
| 4 | Classical | 0.11 s | 0.13 s | 0.55 s | 0.61 s | 379 MB |
| 4 | Compare All (Retinexformer once) | 3.86 s | 4.02 s | 4.51 s | 4.62 s | 1982 MB |

\* This run used a 2560 MB limit and touched it once; it was not OOM-killed. Two repeat runs of Compare All
(8 requests, 2 CPUs) peaked at 1959 MB (2560 MB limit) and 1931 MB (3 GB limit), with no swap and zero
`memory.events` max/oom. Hence the recommended **3 GB** limit.

**Re-measured after the UI redesign** (image `lumen:ui`, FastAPI, 2 CPUs, 3 GB limit, same protocol; a
768x512 resize of `app/examples/coffee_dark.jpg`, so the image content differs from the table above.
Raw data: `results/container_benchmark_ui.json`):

| CPUs | Mode | Server median | Server p95 | Round-trip median | Round-trip p95 | Peak memory |
|---|---|---|---|---|---|---|
| 2 | Standard | 5.86 s | 6.28 s | 5.98 s | 6.37 s | 1727 MB |
| 2 | Quality Guided | 5.57 s | 5.99 s | 5.67 s | 6.08 s | 1790 MB |
| 2 | Classical | 0.11 s | 0.14 s | 0.22 s | 0.27 s | 306 MB |
| 2 | Compare All (Retinexformer once) | 5.56 s | 5.81 s | 5.81 s | 6.07 s | 1800 MB |

The restoration itself is unchanged. The HTTP overhead (round trip minus server time) fell from ~0.45 s to
~0.1 s, and peak memory fell by ~250-300 MB without Gradio. Three simultaneous Standard requests completed at
6.3, 12.2 and 17.9 s, with no errors and a 1777 MB peak.

Startup (`docker run` → first HTTP 200): **4.7 s** (3 runs: 4.69-4.71 s) with the FastAPI app, against 6.6-7.0 s
with Gradio. Model loading + SHA-256 + one warm-up inference take ~0.3-0.5 s of that.

**Concurrency:** the app runs one restoration at a time (an asyncio semaphore, at most `LUMEN_MAX_QUEUE`=16 waiting), so memory does not
grow with concurrent users. Three simultaneous Standard requests at 2 CPUs / 2.5 GB completed at 6.9, 13.0 and
19.1 s, with no errors and a 1850 MB peak. **Queued users wait for the requests ahead of them**: under
concurrent load at 2 vCPUs, the ~10 s response target is met only for the first request in the queue.

### CPU threads
PyTorch sizes its thread pool from the host's cores, not the container's quota. With `--cpus 4` it used 8
threads, and Retinexformer at 600x400 took **8.4 s, against 2.5 s with 4 threads**. The app now sets PyTorch and
OpenCV threads to the cgroup CPU quota at startup (override: `LUMEN_TORCH_THREADS`).

### GPU (development laptop only)
RTX 3050 4 GB Laptop, not in the container: Retinexformer 600x400 144 ms, 1024x683 432 ms. The app disables
cuDNN on CUDA, because per-size plan building cost ~2.5 s for every new image size (`LUMEN_CUDNN=1` re-enables
it). The provided image is CPU-only; **no GPU deployment has been tested.**

## Verification log (2026-11-06)

| Check | Result |
|---|---|
| `docker build` (production Dockerfile) | OK, 181 s cold, 15 s cached rebuild; build-time checks: calibrated QGH config loads, app imports resolve |
| Image contents | `/app` = code, configs (incl. `quality_guided.yaml`), 4 example images, `download_weights.py`, `weights/README.md`; **no `.pth`/`.pt` files, no datasets, no results, no training checkpoints, no tests**; 67 Python packages (no pyiqa, lpips, torchvision, optuna, gdown, pytest); runs as non-root `appuser` |
| Smoke test, checkpoint mounted (`scripts/smoke_test_app.py`) | **All passed**: 3 modes x 4 example uploads, Compare All (Retinexformer ran once), NIQE disabled, 6 invalid uploads with LUMEN's messages |
| No checkpoint / corrupt / altered checkpoint | **Passed**: only Classical offered, reason shown and logged |
| QGH config missing / invalid | **Passed**: only Quality Guided disabled |
| Resource benchmark (`scripts/benchmark_container.py`) | Table above; no OOM kills |
| Issues found and fixed during verification | (1) thread oversubscription, 3.3x slower; (2) Gradio's generic "Could not preprocess input component" error for non-image, empty or truncated uploads, now LUMEN's specific messages; (3) **browser UI never loaded**: the upload widget was a `gr.Image` *subclass*, which Gradio serves as a custom component whose frontend assets (`/gradio_api/custom_component/<id>/client/{component,example}/{index.js,style.css}`) did not exist (404, *Failed to fetch dynamically imported module*). It is now a native `gr.Image` with LUMEN's decode-error mapping; a regression test checks every component is a Gradio built-in |
| Browser check, Gradio UI (headless Chrome 156, DevTools protocol) | Desktop 1366 px and emulated 390 px phone render with no horizontal overflow; real upload + Enhance shows the result; invalid upload shows LUMEN's error toast; no HTTP >= 400 responses and no console errors (superseded by the redesign below) |
| **UI redesign re-verification** (image `lumen:ui`) | Build OK; build-time checks also confirm `app/web/` pages and scripts are present. Image 2.02 GB: no Gradio, no httpx, no `.pth` files, non-root `appuser`; container files byte-identical to the source |
| Smoke test (new `scripts/smoke_test_app.py`, stdlib HTTP) | **All passed**: both pages + every referenced asset, `/healthz`, 3 styles x 4 examples, Compare All (Retinexformer ran once), NIQE unavailable, 6 invalid uploads -> 422 with LUMEN's messages |
| No checkpoint mounted (new UI) | **Passed**: only Quick offered; Natural request -> 409 with the reason |
| Browser E2E (headless Chrome 156 over CDP, 1440x900 / 820x1180 / 390x844) | **All checks passed** on all three: empty, HEIC rejection, ready, working + cancel, result, slider by keyboard and pointer, cached style switching (no repeat request), download (PNG byte-identical to the shown result), invalid upload alert (on screen on phones), TIFF (no preview -> placeholder, restores), details page, research lab, no horizontal overflow, images with alt text, named buttons. Console: only the intentional 422; no other HTTP errors |
| Upload deletion | No `lumen_upload_*` files left in the container after all tests |
| Hosting platform, GPU | **Not done** |

## Privacy note
Each upload is streamed to a private temporary file (`lumen_upload_*`, size-limited while streaming) and deleted
as soon as the request finishes, including on errors; results are returned in the response and never written
to disk. A test checks that no upload files remain, and none were found in the container after the full browser
and smoke tests. LUMEN does not log image contents. The *How it works* page keeps the last restoration's
file name and technical details in the browser's `sessionStorage` (cleared when the tab closes); no image data is stored there.

## Known limitations
- Latency and memory were measured on one laptop. Shared cloud vCPUs may be slower; re-run
  `scripts/benchmark_container.py` on the target.
- At 2 vCPUs a request takes ~6 s; queued concurrent users wait longer (see *Concurrency*).
- The checkpoint must be supplied by whoever deploys LUMEN (mounted volume); the image alone runs only Classical.
- The browser check used headless Chrome on the development machine; real phones and other browsers were not tested.
- Quality Guided is a safeguard (overexposure 0.82 % → 0.03 % on eval15), not a measured quality improvement
  (PSNR 25.12 vs 25.15).
