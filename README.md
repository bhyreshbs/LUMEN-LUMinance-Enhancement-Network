# LUMEN · LUMinance Enhancement Network

Low-light image restoration for **Computer Vision Hackathon 2026, Problem Statement P1**
("Robust Low-Light Image Restoration and Enhancement"), team **T66**. Private repository: no licence is
granted (see [Licensing](#licensing)).

LUMEN compares and serves three approaches:
- **Standard: Retinexformer** (ICCV 2023), the recommended mode.
- **Quality Guided:** Retinexformer plus classical refinements applied only where a calibrated detector finds a
  residual problem (noise, near-clipped highlights). If nothing is justified, the output is Retinexformer's,
  unchanged.
- **Classical:** an adaptive OpenCV pipeline (illumination estimation, Retinex, illumination-aware denoising,
  CLAHE, colour correction) with constants tuned on training data.

A fusion hybrid and Zero-DCE++ are kept as research baselines.

## Results (LOL-v1 eval15, 15 images, single run of a frozen configuration)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Over-exposed px ↓ |
|---|---|---|---|---|---|
| Classical (tuned) | 17.16 | 0.7241 | 0.2781 | 5.50 | 0.00 % |
| **Retinexformer** | **25.15** | **0.8441** | **0.1314** | **3.48** | 0.82 % |
| Fusion hybrid (research) | 18.89 | 0.7606 | 0.3042 | 5.39 | 0.00 % |
| **Quality Guided** | 25.12 | **0.8441** | **0.1314** | **3.48** | **0.03 %** |

Quality Guided reduces over-exposed pixels from 0.82 % to 0.03 %. Standalone Retinexformer stays slightly better
in PSNR, and the two are equivalent on SSIM, LPIPS and NIQE. These results are for eval15 only. Full report, with
ablations, per-image results and failure cases: [`eval_P1_T66.md`](eval_P1_T66.md).

| Document | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | requirements traceability, methods, training, ablation, failure analysis, limitations, ethics |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | system diagram and module map |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | container, checkpoint mount, measured CPU latency and RAM, verification log |
| [`weights/README.md`](weights/README.md), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | checkpoint provenance, licences, open questions |
| [`phases.md`](phases.md) | project phases: plan vs. outcome |

## Quick start (local)

```bash
# Python 3.12
py -3.12 -m venv .venv            # Windows   (Linux/macOS: python3.12 -m venv .venv)
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)

pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # or .../whl/cpu
pip install -r requirements.txt            # app runtime; requirements-dev.txt adds evaluation/training/tests

python scripts/download_weights.py --runtime   # official Retinexformer checkpoint -> weights/ (SHA-256 checked)
python app/app.py                              # http://localhost:7860
```

The checkpoint is **not included in this repository**: its redistribution rights are uncertain. The app verifies
its SHA-256 at startup. Without it, only Classical mode is offered.

## Using the app

The workspace at `/` is deliberately simple: **upload → choose a style → Restore → compare → Download**.
Drag-and-drop, paste and the file picker all work, as do the example photos. The before/after slider can be
dragged, or moved with the arrow keys. The restored image downloads as a PNG.

| Style in the app | Mode | Notes |
|---|---|---|
| **Natural** (recommended) | Standard: Retinexformer | |
| **Highlight-safe** | Quality Guided | |
| **Quick** | Classical | the only style offered without the checkpoint |

The workspace shows no model names, metrics or diagnostics. All of that is on **How it works** (`/details`):
- how each style works
- the technical details of your last restoration (model, checkpoint, device, timings, QGH refinements and
  coverage)
- the evaluation tables
- a research lab: compare all styles, the fusion hybrid with ablations, and Zero-DCE++ if its weights are present
- system status and limitations

The frontend is plain HTML, CSS and JavaScript in `app/web/`, with no build step and no external requests. It is
served by a small FastAPI layer (`app/server.py`) over the unchanged `app/service.py`.

## Docker (CPU)

```bash
python scripts/download_weights.py --runtime
docker build -t lumen .
docker run --rm -p 7860:7860 --cpus 2 --memory 3g -v "$PWD/weights:/models:ro" lumen
```

- **Image:** 2.02 GB, CPU-only, with the processing size capped at **768 px** in the container.
- **Checkpoint:** mounted read-only at `/models`, never baked into the image.
- **Measured on the test laptop, 768×512 input:** Standard takes **6.3 s median at 2 CPUs and 3.8 s at 4 CPUs**,
  with peak memory **≤ 2.1 GB** for single modes, hence the 3 GB limit. After the UI redesign (FastAPI, 2.02 GB
  image): Standard **5.9 s median at 2 CPUs**, peak ≤ 1.8 GB.
- **Hosting (prepared, not yet deployed):** Google Cloud Run, see [`docs/DEPLOY_CLOUD_RUN.md`](docs/DEPLOY_CLOUD_RUN.md)
  (setup and deploy scripts in `deploy/cloudrun/`, keyless GitHub Actions workflow in `.github/workflows/`), and
  Hugging Face Spaces, see [`docs/DEPLOY_HF_SPACES.md`](docs/DEPLOY_HF_SPACES.md). Both images were built and
  smoke-tested locally, the Cloud Run one with `PORT=8080`, 2 CPUs and 4 GiB. No GPU deployment has been tested.

## Reproducing the results

1. Datasets into `data/` (see [`data/README.md`](data/README.md)), then `python scripts/check_data.py`.
2. Optional: re-tune the classical constants (`our485` only, 90/10 split): `python scripts/tune_classical.py --trials 150`.
3. Optional: recalibrate Quality Guided on the dev split (never eval15):
   `python scripts/make_dev_split.py && python scripts/calibrate_quality_guided.py`.
4. Evaluate (writes `eval_P1_<team>.md` with the USNs on the first line, plus per-image CSV and diagnostics):
   ```bash
   python scripts/evaluate.py --low data/LOL/eval15/low --high data/LOL/eval15/high --deep-models retinexformer \
       --team T66 --usns 1RUA24CSE0093,1RUA24CSE0096,1RUA24CSE0258,1RUA24CSE0259 --run eval15
   ```
5. Tests: `pip install -r requirements-dev.txt && pytest -q` (167 tests, including the web API in `tests/test_web.py`).
6. Container checks: `python scripts/smoke_test_app.py --url http://127.0.0.1:7860` and
   `python scripts/benchmark_container.py --input <768x512 image>`.

## Python API

```python
from lumen import LumenPipeline
from lumen.utils import read_image, write_image

pipe = LumenPipeline(deep_model="retinexformer")
r = pipe.run(read_image("dark.jpg"), mode="deep")                    # Standard
q = pipe.run(read_image("dark.jpg"), mode="quality_guided_hybrid")   # Quality Guided
write_image("out.png", q.output)
print(q.latency_ms, q.info["applied"], q.info["refinements"])         # which refinements ran, coverage
```

Modes: `classical`, `deep`, `quality_guided_hybrid`, and `hybrid` (research). A missing or invalid checkpoint
raises `CheckpointError`, and a missing or invalid QGH config raises `QGConfigError`. Neither ever falls back silently.

## Project layout

```
lumen/      classical/ (adaptive, illumination, filters, contrast, pipeline)   deep/ (retinexformer, zero_dce,
            infer, checkpoints)   hybrid/ (quality_guided, fusion)   metrics/   imageio.py   pipeline.py
app/        app.py (entry point)   server.py (FastAPI: pages + JSON API)   service.py (modes, checks, Compare All)
            web/ (index.html, details.html, static/ css js data img)   examples/ (CC0 / public domain)
configs/    classical.yaml (tuned)   quality_guided.yaml (calibrated)   experimental/ (research only, unused by the app)
scripts/    data checks, tuning, calibration, training, evaluation, benchmarks, smoke test, checkpoint download,
            noise_study*.py (experimental research), prepare_hf_space.py
deploy/     cloudrun/ (setup.ps1, deploy.ps1)   huggingface/ (Space Dockerfile, start.py)
tests/      pytest suite
docs/       DESIGN, ARCHITECTURE, DEPLOYMENT, DEPLOY_CLOUD_RUN, DEPLOY_HF_SPACES, NOISE_STUDY (experimental)
data/       README + split lists only (datasets are not committed)
weights/    README only (checkpoints are not committed)
```

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `LUMEN_DEVICE` | auto | `cuda` / `cpu` |
| `LUMEN_MAX_SIDE` | 1024 local, 768 in Docker | processing size (longest side) |
| `LUMEN_MAX_PIXELS` / `LUMEN_MAX_FILE_MB` | 40e6 / 20 | upload limits (checked before decoding pixels) |
| `LUMEN_RETINEXFORMER_WEIGHTS` | `weights/...`, Docker `/models/...` | checkpoint path (SHA-256 verified) |
| `LUMEN_QG_CONFIG` | `configs/quality_guided.yaml` | calibrated Quality Guided config |
| `LUMEN_CLASSICAL_CONFIG` | `configs/classical.yaml` | tuned classical constants |
| `LUMEN_TORCH_THREADS` | cgroup CPU quota | PyTorch / OpenCV threads |
| `LUMEN_CUDNN` | 0 in the app | the app disables cuDNN on CUDA (per-size plan building cost ~2.5 s); `1` re-enables it |
| `PORT` / `LUMEN_HOST` | 7860 / 0.0.0.0 | app port and bind address |
| `LUMEN_MAX_QUEUE` | 16 | requests allowed to wait while one restoration runs; beyond that the API answers 503 "busy" |

## Submission checklist

- [ ] Live hosted application URL (not yet deployed)
- [x] Source code, architecture diagram, design document
- [x] `eval_P1_T66.md`: USNs on the first line, metrics in P1 order
- [x] Ablation study and failure-case analysis
- [x] No secrets, datasets, checkpoints or third-party PDFs committed

## Licensing
This repository is **private** and has **no licence file**: no reuse rights are granted. Third-party code,
weights, data and dependency licences are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), together
with the items that must be resolved before any public release:
- Zero-DCE++ re-implementation: academic, non-commercial terms.
- Retinexformer checkpoint: no explicit licence.
- pyiqa: PolyForm Noncommercial.

## Credits
Retinexformer: Y. Cai et al., ICCV 2023 (MIT code). Zero-DCE / Zero-DCE++: C. Li, C. Guo, C. C. Loy et al.,
CVPR 2020 / TPAMI 2021. LOL: C. Wei et al., BMVC 2018. LOL-v2: W. Yang et al., TIP 2021.
Guided filter: K. He, J. Sun, X. Tang, ECCV 2010. LIME: X. Guo et al., TIP 2017.
