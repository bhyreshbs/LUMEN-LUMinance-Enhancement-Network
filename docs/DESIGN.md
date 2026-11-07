# LUMEN - Design document

Problem statement P1 of Computer Vision Hackathon 2026: *Robust Low-Light Image Restoration and Enhancement*
(the organisers' PDF is not redistributed). Team T66. Evaluation results: [`eval_P1_T66.md`](../eval_P1_T66.md).
Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md). Deployment: [`DEPLOYMENT.md`](DEPLOYMENT.md).

## 1. Problem and goals
Restore severely under-exposed, noisy and colour-distorted images while preserving texture, edges and structure,
without over-exposure, halos or unnatural colours, across lighting conditions (extremely dark, non-uniform).
Deliver a usable application with adaptive enhancement, a comparison of classical, deep and hybrid approaches,
intermediate visualisations and failure analysis.

**Design principles adopted:**
- Measure everything on a held-out test set touched once.
- Never tune on the test set.
- Prefer an honest negative result to an unsupported claim.
- Never run a model on missing or unverified weights.

## 2. Requirements traceability

| P1 requirement | Implementation |
|---|---|
| Illumination estimation and decomposition | `lumen/classical/illumination.py`: max-RGB + guided-filter illumination map, Retinex decomposition |
| Retinex-based enhancement | `retinex_enhance` (illumination-adjusted, I·L^(γ−1)); `multiscale_retinex` (MSR baseline) |
| CLAHE | `lumen/classical/contrast.py` `clahe_lab` (CLAHE on L\*) |
| Edge-preserving denoising (bilateral / guided) | `lumen/classical/filters.py`; QGH noise refinement (guided filter) |
| Contrast and colour correction | gray-world WB, saturation, highlight roll-off (`contrast.py`) |
| Adaptive parameter selection | `lumen/classical/adaptive.py` (image statistics → parameters); constants tuned with Optuna on `our485` |
| Deep architecture | Retinexformer (`lumen/deep/retinexformer.py`, primary); Zero-DCE++ (`zero_dce.py`, baseline) |
| Fine-tune / adapt where data is available | `scripts/train_retinexformer.py` (supervised); tried and rejected on evidence (§5) |
| Generalisation across illumination | eval15 + 269-image dev split from another dataset; 20 unseen phone photos in the app; domain-shift failure documented (§8) |
| Image-quality-aware strategy | QGH detectors (noise, colour, contrast loss, highlight) decide where to refine; fusion hybrid uses quality weights |
| Integrate classical illumination with DL | fusion hybrid: guided branch (Retinex pre-brightening → DL); QGH: classical refinements on the DL output |
| Adaptive select / combine mechanism | fusion: quality-weighted Laplacian pyramid; QGH: per-region soft masks with a coverage gate and an identity path |
| Preserve edges, minimise noise amplification | edge-preserving filters; QGH never sharpens where it denoises; edge preservation reported |
| Extremely dark / non-uniform inputs | category detection, adaptive γ, illumination-radius adaptation; Zero-DCE++ double pass |
| Compare classical / DL / hybrid | `scripts/evaluate.py`, with the model labelled per experiment |
| Ablation | every component switchable (classical, fusion hybrid, QGH) |
| Over-exposure / colour shift / detail loss | over-exposure %, colour shift, edge-preservation metrics; QGH highlight roll-off |
| Unseen images | dev split unseen by the checkpoint; app accepts arbitrary uploads |
| Metrics: PSNR, SSIM, LPIPS, NIQE, latency, noise, edges | `lumen/metrics/quality.py` |

## 3. Architecture
See [`ARCHITECTURE.md`](ARCHITECTURE.md). In short: `lumen/pipeline.py` validates the input
(`lumen/imageio.py`) and dispatches to the classical pipeline, a deep model (`lumen/deep/`), the fusion
hybrid or the Quality-Guided Hybrid (`lumen/hybrid/`). The web app is a static HTML/CSS/JS frontend
(`app/web/`) served by a thin FastAPI layer (`app/server.py`) over `app/service.py`, which owns mode
availability, checkpoint verification and Compare All.

**UI design.** The workspace is reduced to the task: upload → style → restore → compare → download. It uses
plain-language style names (**Natural** = Standard, **Highlight-safe** = Quality Guided, **Quick** =
Classical) and contains no model names, metrics or diagnostics. Those appear only on the separate
*How it works* page (`/details`), together with the evaluation tables, the research lab (Compare All, fusion
hybrid with ablations, Zero-DCE++) and system status. The layout is responsive (desktop, tablet, phone with a
sticky action bar), follows light/dark system themes, and is keyboard-operable: the comparison slider is an
ARIA slider driven by the arrow keys.

## 4. Methods

### 4.1 Classical pipeline
- **Illumination:** L₀ = max(R, G, B), refined by a self-guided filter. Its radius shrinks for non-uniform lighting.
- **Retinex:** I' = I·L^(γ−1), with γ chosen so that the mean illumination maps to a target brightness.
- **Denoising:** a bilateral filter blended in by √(1 − L), so the dark (most amplified) regions are denoised most.
  Its strength is proportional to the estimated σ times the brightening gain.
- **CLAHE on L\*:** clip limit inversely proportional to contrast, capped when the effective noise is high.
- **Colour:** gray-world white balance weighted by the measured colour cast, then a mild saturation boost.
- **Tuning:** 16 constants searched with Optuna (150 trials, PSNR + 10·SSIM) on 437 `our485` images, and
  selected on 48 held-out `our485` images. Result: `configs/classical.yaml`.

### 4.2 Deep models
- **Retinexformer** (Cai et al., ICCV 2023): a one-stage Retinex transformer with 1.6 M parameters.
  - **Port:** dependency-free, with identical parameter names. Strict loading of the official checkpoints and
    bit-exact output parity with the upstream code were verified.
  - **Inference:** reflect padding to multiples of 8, feathered tiling above 1024 px, fp16 on CUDA, CPU fallback.
- **Zero-DCE++:** zero-reference curve estimation, kept as a baseline and research option.

### 4.3 Fusion hybrid (research baseline)
Three candidates (classical, deep, illumination-guided deep) are fused per pixel with Mertens-style weights
(exposure × √contrast × √saturation ÷ noise) in a Laplacian pyramid, followed by guided denoising and white
balance. It is retained as a documented negative result (§7).

### 4.4 Quality-Guided Hybrid (QGH)
Retinexformer runs once; its output R is treated as the answer. Classical refinement is applied only where a
detector shows a residual problem:
- **noise:** high-frequency residual ÷ local gradient → guided-filter denoise
- **colour:** chromaticity drift from the input's reliable regions → partial chroma pull
- **contrast:** local coefficient of variation lost relative to the input → base/detail boost
- **highlight:** near-clipped luminance → soft-knee roll-off

How the refinements are applied:
- Masks are soft and Gaussian-feathered, so there are no seams.
- A refinement runs only if its mask covers at least 0.5 % of the image. If none qualifies, R is returned
  bit-exact (the identity path).

**Calibration** (`scripts/calibrate_quality_guided.py`) on the dev split only:
- For each detector, choose the threshold that maximises the per-image relative squared-error gain against
  ground truth.
- Constraints: at most 25 % coverage, and a semantic floor for highlights (luminance ≥ 0.85).
- A refinement with no positive gain stays disabled. Outcome: noise and highlight enabled; colour and contrast disabled.
- An acceptance rule was fixed before eval15 was run (dev PSNR up; SSIM and LPIPS no more than 0.005 worse). Dev passed.
- The objective was revised twice, both times on dev only and both recorded:
  1. The first version favoured global smoothing, which a coverage cap now prevents.
  2. Highlight drifted down to luminance 0.62, which the 0.85 floor now prevents.

## 5. Data and training
- **Datasets:** LOL-v1 (`our485`, `eval15`) and LOL-v2 (Real, Synthetic); verified by `scripts/check_data.py`.
- **Leakage audit** (16×16 thumbnail fingerprints):
  - 91 / 100 LOL-v2-real Test images are copies of `our485`.
  - 398 `our485` images reappear in LOL-v2-real Train.
  - 16 `our485` images are near-duplicates of eval15 scenes.

  The training script drops all test duplicates and splits train/val by near-duplicate scene group.
  The dev split excludes `our485` and eval15 entirely.
- **Retinexformer fine-tuning:**
  - Setup: L1 + 0.2(1 − SSIM) + 0.05·LPIPS, AdamW with cosine schedule, AMP, 128–256 px crops, gradient accumulation.
  - A first run on LOL-v1 + LOL-v2 real + synthetic was stopped after 1 epoch, once its validation gain proved to
    be almost entirely on synthetic images (synthetic 15.6 → 20.4 dB, real 23.30 → 23.41 dB).
  - The real-only run (8 epochs) raised its validation PSNR from 22.54 to 23.61, but **lowered eval15 to 24.44**
    (from 25.15). The fine-tuned weights were therefore discarded and the official checkpoint is used.
- **Zero-DCE++** uses the official snapshot; it was not retrained.

## 6. Evaluation
Single run of the frozen configuration on eval15 (15 images). Full tables and the per-image breakdown are in
`eval_P1_T66.md`.

| Method | PSNR | SSIM | LPIPS | NIQE | Overexp. % | GPU latency |
|---|---|---|---|---|---|---|
| Classical (tuned) | 17.16 | 0.7241 | 0.2781 | 5.50 | 0.00 | 80 ms |
| **Retinexformer** | **25.15** | **0.8441** | **0.1314** | **3.48** | 0.82 | 148 ms |
| Fusion hybrid (Retinexformer) | 18.89 | 0.7606 | 0.3042 | 5.39 | 0.00 | 541 ms |
| **QGH (Retinexformer)** | 25.12 | **0.8441** | **0.1314** | **3.48** | **0.03** | 176 ms |

QGH reduces over-exposure from 0.82 % to 0.03 %. Standalone Retinexformer remains slightly better in PSNR
(+0.03 dB) and is equivalent on SSIM, LPIPS and NIQE.

## 7. Ablation study
- **Classical pipeline** (from 17.16 dB):
  - Retinex and denoising are essential (8.46 dB and 16.23 dB without them).
  - Adaptive parameters are worth +1.75 dB.
  - Removing colour correction *improves* eval15 (17.44 dB).
  - Removing CLAHE lowers PSNR but improves SSIM and LPIPS. So these two components are not uniformly beneficial.
- **Fusion hybrid** (18.89 dB):
  - Removing the classical branch gives 21.12 dB, the best variant.
  - Every variant stays at least 4 dB below Retinexformer. Weighting by quality (exposure, contrast, saturation)
    cannot tell a well-exposed but wrongly coloured candidate from a faithful one.
- **QGH:**
  - Highlight roll-off accounts for the full −0.03 dB and the full over-exposure reduction.
  - Noise refinement was the dev gain (+0.30 dB on dev) but activated on only 2/15 eval15 images, with no effect.

## 8. Failure-case analysis (eval15 images, inspected visually)
1. **Image 1, bright white surfaces:**
   - Retinexformer's worst PSNR (19.53): the output is brighter and cooler than the reference, and 11.5 % of
     pixels clip on white cabinet panels.
   - Root cause: a global exposure / white-balance mismatch on a high-albedo scene.
   - QGH roll-off removes almost all clipping (to 0.14 %) but cannot fix the global tone (−0.14 dB).
2. **Image 493, saturated textured scene:**
   - Lowest SSIM (0.699): less saturated colours and lifted shadows relative to the reference.
   - No QGH refinement activated, because the colour detector is disabled after calibration.
3. **Image 748, fusion hybrid colour cast:**
   - The hybrid adds a green/cyan cast and flattens contrast (17.64 dB against 28.37 dB). Root cause: fusion
     weights reward exposure and saturation, not fidelity.
   - Mitigation: QGH keeps the model output and only corrects detected regions.
4. **Domain shift (dev split, LOL-v2-real):**
   - Retinexformer drops to SSIM 0.33 / LPIPS 0.83 on noisier captures, and supervised fine-tuning did not
     transfer to eval15.
   - Root cause: training on a single capture setup.

## 9. Performance
- **GPU** (RTX 3050 Laptop, fp16): Retinexformer 144 ms at 600×400 and 432 ms at 1024×683; QGH adds about
  20–150 ms of CPU refinement.
- **Production container, 768×512, CPU:**

| CPUs | Standard median / p95 | Quality Guided | Classical | Compare All | Peak memory |
|---|---|---|---|---|---|
| 2 | 6.33 / 7.05 s | 5.67 / 5.89 s | 0.13 / 0.15 s | 6.22 / 7.66 s | ≤ 2.06 GB (single modes) |
| 4 | 3.76 / 3.91 s | 3.74 / 4.09 s | 0.11 / 0.13 s | 3.86 / 4.02 s | ≤ 1.98 GB |

Startup takes about 7 s (model load and SHA-256 check: 0.24–0.44 s). Details and method: `DEPLOYMENT.md`.

## 10. Deployment
- **Image:** CPU Docker image (2.32 GB) with a 768 px processing limit, one request at a time, and PyTorch and
  OpenCV threads matched to the container's CPU quota.
- **Checkpoint:** supplied by a read-only mount at `/models`, with its SHA-256 verified at startup. A missing,
  corrupt or altered checkpoint leaves only Classical available.
- **Calibrated QGH config:** baked into the image and validated at build and start. If it's missing or invalid,
  only Quality Guided is disabled.
- **Not yet verified:** no hosting provider and no GPU deployment.

## 11. Limitations
- eval15 has only 15 images from one capture setup, so differences of hundredths of a dB are not meaningful,
  and no generalisation beyond the evaluated data is claimed.
- Retinexformer degrades on noisier, differently exposed data (§8.4).
- The QGH colour and contrast detectors exist but are disabled, because calibration found no gain.
- CPU latency at 2 vCPUs is about 6–7 s per request, and queued users wait for the requests ahead of them.
- The UI was inspected in headless Chrome at desktop, tablet and phone sizes; real phones, other browsers and a
  hosted public URL are still outstanding.
- Licensing questions are open (see `THIRD_PARTY_NOTICES.md`).

## 12. Ethics
- **Privacy:** enhancement can reveal people or details in images captured without consent. LUMEN does not
  log or store image contents. Each upload is streamed to a private temporary file, which is deleted when the
  request finishes; results are returned in the response and never written to disk. A test checks the
  deletion.
- **Not evidence:** outputs are restorations, not forensic evidence. The models smooth noise and can alter
  colour and fine detail.
- **Licensing:** datasets, weights and the optional NIQE dependency carry research or non-commercial terms or no
  stated terms. LUMEN does not redistribute them, and documents what must be resolved before public release.
- **Honest reporting:** negative results are reported (the fusion hybrid, fine-tuning, the dev-only gains of QGH).

## 13. Team and contributions

| USN | Responsibilities |
|---|---|
| 1RUA24CSE0093 | *to be filled in by the team* |
| 1RUA24CSE0096 | *to be filled in by the team* |
| 1RUA24CSE0258 | *to be filled in by the team* |
| 1RUA24CSE0259 | *to be filled in by the team* |
