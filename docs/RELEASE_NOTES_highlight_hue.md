# Release notes: highlight hue fix (branch `integration/highlight-hue`, not merged, not deployed)

## What changes for users
White or bright areas that were already (nearly) clipped in the dark photo - a window, a lamp - no longer turn orange or
yellow in **Natural** and **Highlight-safe**. **Quick** is unchanged. Photos with no near-clipped pixels (most dark photos)
come out bit-for-bit as before.

## Trade-off to be aware of: more near-white pixels in some photos
Over-exposure = share of pixels whose brightest colour channel is >= 0.98.

| Set | Style | Before | After | Change |
|---|---|---|---|---|
| 20 unseen real photos | **Highlight-safe** | **0.86 %** | **1.10 %** | **+0.24 points (about 28 % more such pixels)** |
| 20 unseen real photos | Natural | 2.75 % | 2.73 % | -0.03 |
| 60 high-contrast test photos | Highlight-safe | 0.95 % | 0.09 % | -0.86 |
| 100 LOL-v2 low-light pairs | Highlight-safe | 1.06 % | 0.84 % | -0.22 |
| eval15 (15) | both | 0.82 % / 0.035 % | unchanged | 0 |

Why it goes up on the unseen photos: a bright window used to come back orange and darker, so it was *not* counted as clipped.
It now comes back near-white, as in the original photo, so those pixels are counted. This is the intended result, but it
does mean the fix is **not** a pure reduction of clipping. Where ground truth exists (high-contrast and LOL-v2 sets) the
over-exposed share goes down or stays flat and image scores improve. If zero added near-white pixels is a hard
requirement, the fix can be switched off (`LUMEN_HIGHLIGHT_HUE=0`) or tuned (`lum_keep` in `configs/highlight_hue.yaml`; a
lower value keeps the roll-off's dimming but was not evaluated).

## Final container test report (2026-11-07, image built from commit `ce1bbb4`)
Environment: Docker Desktop 29.9.2 (WSL 2), `--cpus 2 --memory 4g`, `PORT=8080`, `LUMEN_TORCH_THREADS=2`,
`LUMEN_MAX_SIDE=768`, checkpoint mounted read-only at `/models`.

| Check | Result |
|---|---|
| Image | 2.02 GB, linux/amd64, runs as **non-root `appuser` (uid 1000)**, `python app/app.py`; build-time checks passed; no `.pth` or `.env` in the image |
| Config in image | `/app/configs/highlight_hue.yaml` present and loaded: `start 0.8, full 0.95, sigma 3, chroma_strength 1, lum_keep 1` |
| Startup | first `/healthz` 200 after **5.1 s**; threads 2; checkpoint SHA-256 verified; all 3 modes available |
| `scripts/smoke_test_app.py` | **ALL PASSED**: pages and assets, health, styles, 3 styles x 4 examples, Compare All with one network run, invalid uploads |
| `/api/status` | styles available; `technical.highlight_hue = {enabled: true, reason: ""}` |
| Bright-window photo | Natural window RGB **0.966/0.953/0.879**, Highlight-safe **0.918/0.905/0.833** (pre-integration image: 0.752/0.488/0.084 and 0.751/0.487/0.083); Quick 0.881/0.901/0.925, unchanged. Details report `highlight_hue: {applied: true, coverage: 0.031}` |
| Compare All | styles standard, quality_guided, classical; every result bit-identical to the single-style call |
| PNG downloads | every restored payload decodes as PNG (15 / 15, plus Compare All) |
| `LUMEN_HIGHLIGHT_HUE=0` | `/api/status` shows `enabled: false, reason: switched off by LUMEN_HIGHLIGHT_HUE=0`; **15 / 15 outputs bit-identical to the pre-integration image built from `main`** |
| Hue ON vs pre-integration image | only the 2 bright-window outputs (Natural, Highlight-safe) change; the 4 app examples and all Quick outputs are identical |
| Config missing | log `Highlight hue fix is disabled: ... not found at /cfg/does_not_exist.yaml`; status shows the reason; 3 styles stay available; 15 / 15 outputs identical to the pre-integration behaviour |
| Config invalid: start > full | disabled with `has out-of-range values`; same outcome (15 / 15 identical) |
| Config invalid: unknown key | disabled with `unknown keys ['bogus_key']`; same outcome |
| Config invalid: broken YAML | disabled with the YAML parser message (multi-line); same outcome |
| Added time in the container | fix itself 34-39 ms with a clipped window, 6-12 ms with none, against 1.6-4.4 s model time |
| Memory | at most 1.58 GiB in `docker stats` samples (4 GiB limit); earlier cgroup measurements 1.7-2.6 GB; no OOM |
| Uploads / logs | 0 `lumen_upload_*` files left; no errors or tracebacks (only the intended config warnings in the failure tests) |
| Graceful shutdown | `docker stop` (SIGTERM) 1.3 s, exit code 0, not OOM-killed |

Test method note: an early failure-case run mistakenly tested only "file not found" because Git Bash rewrote the `/cfg/...`
path; the three invalid-content cases were rerun with path conversion disabled and old results deleted before comparing.

**Not verified:** a real Cloud Run deployment of this version (the live service still runs the previous image), the
Cloud Storage volume mount with this version, and real-world performance on photos with clipped highlights and ground truth.
