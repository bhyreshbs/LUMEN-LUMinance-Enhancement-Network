1RUA24CSE0093,1RUA24CSE0096,1RUA24CSE0258,1RUA24CSE0259

# LUMEN - Evaluation report, Problem Statement P1 (Team T66)

**Problem:** P1, Robust Low-Light Image Restoration and Enhancement. Source of every number: the single
Phase 5 run on LOL-v1 eval15 (`results/phase5_eval15/`: `eval15.md`, `metrics.json`, `per_image.csv`,
`qg_diagnostics.json`) and the Phase 5 dev run (`results/phase5_dev/metrics.json`). A Phase 7 regression run
reproduced the eval15 headline rows exactly.

**Update (after the Phase 5 freeze):** a small highlight colour fix was added to the Natural and Highlight-safe styles and the app was deployed on Google Cloud Run. The Phase 5 numbers in sections 4-6 are unchanged (they are the numbers of the frozen run and are reproduced exactly with the fix switched off). The effect of the new step, the trade-off it brings and the deployment are in **section 9**.

## 1. Objective
Restore severely under-exposed, noisy, colour-distorted images while preserving texture and edges and avoiding
over-exposure, halos and colour shifts. Compare classical, deep-learning and hybrid approaches with the P1
metrics: PSNR, SSIM, LPIPS, NIQE, latency, noise suppression and edge preservation.

## 2. Data and split methodology
| Set | Pairs | Role |
|---|---|---|
| LOL-v1 `our485` | 485 | Training set of the official Retinexformer checkpoint. LUMEN used it only to tune the classical constants (437 tune / 48 validation, seed 0) |
| LOL-v1 `eval15` | 15 | **Held-out test set.** Headline numbers. Never used for tuning, calibration or model selection; the final configuration was frozen before a single run |
| Dev split: LOL-v2-real Train minus copies and near-copies of `our485`/`eval15` | 269 | Calibration and acceptance of the Quality-Guided Hybrid thresholds (`data/splits/dev_lolv2_real_unseen.txt`) |

**Leakage found and handled:**
- 91 of the 100 LOL-v2-real *Test* images are exact copies of LOL-v1 `our485`, so LOL-v1-checkpoint results on LOL-v2-real Test are contaminated and **not reported** here.
- 398 `our485` images reappear in LOL-v2-real Train.
- 16 `our485` images are near-duplicates (neighbouring shots) of eval15 scenes.

The dev split removes all of these. Duplicates were found by 16x16-thumbnail fingerprints (mean absolute
difference < 0.01).

## 3. Evaluation procedure
- `scripts/evaluate.py` on eval15 at native 600x400 (no resizing). Every method uses the same images and metric code.
- **PSNR / SSIM** (scikit-image, data range 1); **LPIPS** (AlexNet); **NIQE** (pyiqa, lower is better).
- **Noise σ**: wavelet noise estimate of the output.
- **Edge preservation**: Pearson correlation of output and ground-truth gradient magnitudes.
- **Colour shift**: mean CIELAB a\*b\* distance to ground truth.
- **Over-exposure**: % of pixels whose maximum channel is ≥ 0.98.
- **Latency**: mean per-image processing time after one warm-up, on an RTX 3050 4 GB laptop GPU (fp16).
- No ground-truth-based rescaling ("GT-mean") is applied to any headline number.
- Checkpoint for all Retinexformer rows: official `retinexformer_LOL_v1.pth`, unmodified.

## 4. Results on eval15 (15 images)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms ↓ | Noise σ ↓ | Edge pres. ↑ | Colour shift ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Classical (tuned) | 17.16 | 0.7241 | 0.2781 | 5.50 | 79.89 | 0.01 | 0.8571 | 10.52 | 0.00 |
| **Deep: Retinexformer** | **25.15** | **0.8441** | **0.1314** | **3.48** | 147.93 | 0.00 | 0.9308 | **6.92** | 0.82 |
| Existing hybrid (fusion): Retinexformer | 18.89 | 0.7606 | 0.3042 | 5.39 | 541.27 | 0.00 | 0.8687 | 9.55 | **0.00** |
| **Quality-Guided Hybrid: Retinexformer** | 25.12 | **0.8441** | **0.1314** | **3.48** | 176.10 | 0.00 | **0.9309** | **6.92** | 0.03 |

Earlier-phase reference rows, same eval15 protocol:

| Method | PSNR | SSIM | LPIPS | NIQE |
|---|---|---|---|---|
| Classical, hand-set constants (before tuning) | 15.16 | 0.5327 | 0.3824 | 7.44 |
| Deep: Zero-DCE++ (zero-reference) | 15.14 | 0.5661 | 0.3421 | 7.94 |
| Existing hybrid: Zero-DCE++ | 16.80 | 0.7332 | 0.3268 | 6.20 |
| Retinexformer, fine-tuned on LOL-v1 + LOL-v2-real (rejected) | 24.44 | 0.8390 | 0.1315 | 3.46 |
| Retinexformer with GT-mean rescaling (paper protocol; comparison only) | 27.17 | 0.8485 | 0.1286 | - |

The published Retinexformer LOL-v1 result is 25.16 / 0.845; this reproduction gives 25.15 / 0.8441.

**Headline comparison:**
- **Over-exposure:** QGH reduces over-exposed pixels from **0.82 % to 0.03 %**.
- **PSNR:** standalone Retinexformer remains **slightly better, 25.15 vs 25.12 dB**.
- **SSIM, LPIPS, NIQE:** the two are **equivalent** at the reported precision (0.8441, 0.1314, 3.48).
- **Latency:** QGH costs +28 ms.

## 5. Ablation results (eval15)

**Classical pipeline**, each component removed in turn (baseline 17.16 / 0.7241 / 0.2781):

| Removed | PSNR | SSIM | LPIPS | Observation |
|---|---|---|---|---|
| Retinex enhancement | 8.46 | 0.2641 | 0.6775 | the core brightening step |
| Denoising | 16.23 | 0.4408 | 0.4705 | largest SSIM / LPIPS loss after Retinex |
| CLAHE | 16.85 | 0.7505 | 0.2528 | PSNR drops, but SSIM and LPIPS *improve* |
| Colour correction | 17.44 | 0.7279 | 0.2646 | *better* on all three: this component hurts on eval15 |
| Adaptive parameters (fixed instead) | 15.41 | 0.6194 | 0.3237 | adaptivity is worth +1.75 dB |

**Existing fusion hybrid (Retinexformer)**, baseline 18.89 / 0.7606 / 0.3042:

| Removed | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Classical branch | 21.12 | 0.7705 | 0.2773 |
| Deep branch | 17.69 | 0.7495 | 0.3132 |
| Guided branch | 19.69 | 0.7584 | 0.3278 |
| Fusion (best single candidate instead) | 18.24 | 0.7559 | 0.2936 |
| Post-denoise | 18.60 | 0.7012 | 0.2671 |
| Post-colour | 19.01 | 0.7617 | 0.2984 |

No variant of the fusion hybrid comes within 4 dB of standalone Retinexformer.

**Quality-Guided Hybrid**, baseline 25.12 / 0.8441 / 0.1314, over-exposure 0.03 %:

| Removed | PSNR | SSIM | LPIPS | NIQE | Overexp. % |
|---|---|---|---|---|---|
| Noise refinement | 25.12 | 0.8442 | 0.1312 | 3.47 | 0.03 |
| Highlight roll-off | 25.15 | 0.8439 | 0.1316 | 3.48 | 0.82 |
| Colour / contrast refinement | 25.12 | 0.8441 | 0.1314 | 3.48 | 0.03 |

The colour and contrast refinements are disabled by calibration, so removing them changes nothing. Highlight
roll-off accounts for the whole PSNR difference from Retinexformer (−0.03 dB) and the whole over-exposure
reduction.

**QGH on the dev split** (269 pairs, used for calibration and acceptance, not a test):

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Retinexformer | 15.66 | 0.3318 | 0.8317 |
| QGH | 15.96 | 0.3610 | 0.7991 |

The dev gain comes almost entirely from noise refinement (−noise: 15.67). It did **not** transfer to eval15,
where noise refinement activated on 2 of 15 images over 0.3 % of pixels.

## 6. Per-image observations (eval15)

| Image | Classical PSNR | Retinexformer PSNR / SSIM | Fusion hybrid PSNR | QGH PSNR | Retinexformer overexp. % | QGH refinement |
|---|---|---|---|---|---|---|
| 1 | 15.93 | 19.53 / 0.844 | 16.37 | 19.40 | 11.46 | highlight |
| 22 | 17.64 | 26.26 / 0.878 | 18.15 | 26.22 | 0.00 | highlight |
| 23 | 12.46 | 25.44 / 0.836 | 14.83 | 25.39 | 0.00 | highlight |
| 55 | 12.64 | 21.39 / 0.860 | 16.11 | 21.38 | 0.00 | highlight |
| 79 | 17.00 | 27.07 / 0.905 | 20.88 | 27.07 | 0.02 | none (identity) |
| 111 | 17.26 | 23.32 / 0.907 | 21.31 | 23.21 | 0.02 | highlight |
| 146 | 20.45 | 25.30 / 0.876 | 24.33 | 25.29 | 0.00 | noise |
| 179 | 14.44 | 28.92 / 0.862 | 17.80 | 28.90 | 0.00 | noise |
| 493 | 17.77 | 22.57 / 0.699 | 17.61 | 22.57 | 0.00 | none (identity) |
| 547 | 18.90 | 23.03 / 0.862 | 17.09 | 23.03 | 0.00 | none (identity) |
| 665 | 15.68 | 27.26 / 0.793 | 23.61 | 27.26 | 0.01 | none (identity) |
| 669 | 20.57 | 26.73 / 0.877 | 23.72 | 26.77 | 0.01 | highlight |
| 748 | 20.24 | 28.37 / 0.892 | 17.64 | 28.37 | 0.26 | none (identity) |
| 778 | 20.01 | 23.09 / 0.777 | 19.43 | 23.09 | 0.05 | none (identity) |
| 780 | 16.43 | 28.98 / 0.792 | 14.51 | 28.83 | 0.48 | highlight |

- Retinexformer beats the classical pipeline and the fusion hybrid on all 15 images. QGH matches or trails it
  by ≤ 0.15 dB on every image except 669, where QGH is 0.04 dB higher.
- QGH took the identity path (output exactly equal to Retinexformer's) on 6/15 images, applied highlight roll-off
  on 7, and noise refinement on 2.
- Largest QGH changes: image 1 (−0.14 dB, over-exposure 11.46 % → 0.14 %, LPIPS −0.003); image 780
  (−0.15 dB, 0.48 % → 0.02 %); image 669 (+0.04 dB). All other changes are within ±0.11 dB.

## 7. Limitations and failure cases
1. **Bright white surfaces (image 1):** Retinexformer's lowest PSNR (19.53). Its output is brighter and cooler
   than the ground truth, with 11.5 % of pixels clipped on white cabinet panels, while the reference is darker
   and warmer. QGH's roll-off removes almost all clipping but does not fix the global exposure and white
   balance, and PSNR drops slightly (−0.14 dB).
2. **Saturated, textured scene (image 493):** Retinexformer's lowest SSIM (0.699). Compared with the reference,
   colours are less saturated and shadows lifted. No refinement activated (identity path).
3. **Fusion hybrid colour cast (image 748):** blending classical, Zero-DCE-style and Retinexformer candidates adds a
   green/cyan cast and flattens contrast: 17.64 dB against 28.37 dB for Retinexformer. Mean-quality weighting
   (exposure, contrast, saturation) does not measure fidelity, so a well-exposed but wrongly-coloured candidate
   can win.
4. **Domain shift:** on the 269 noisier LOL-v2-real dev images, Retinexformer (trained on LOL-v1) reaches only
   SSIM 0.33 and LPIPS 0.83. Fine-tuning on LOL-v1 + LOL-v2-real improved its own validation PSNR
   (22.54 → 23.61) but lowered eval15 to 24.44, so it was discarded.
5. **Small test set:** eval15 has 15 images from one capture setup. Differences of a few hundredths of a dB
   (25.12 vs 25.15) are within what a different 15-image sample could change. No claim is made about other
   cameras, scenes or datasets. The 20 personal "unseen" photos have no ground truth and are not scored here.
6. **Metrics:** NIQE was computed with pyiqa defaults. Latency is GPU latency on one laptop; CPU and container
   figures are in `docs/DEPLOYMENT.md`.
7. **Bright windows and lamps (found after the freeze):** on a photo of a dim room with a bright window, the Natural and Highlight-safe outputs turned the white window **orange** (window RGB 0.96/0.95/0.89 became about 0.75/0.49/0.08). The eval15 images almost never contain clipped input, so this was invisible in the headline numbers. Cause: Retinexformer divides by an estimated illumination, and where the input is already clipped the colour channels are corrected unequally; LOL has no such scenes. QGH's roll-off only changes brightness. Addressed in section 9 (with a documented trade-off).

## 8. Conclusion
- **On this test set the best method is standalone Retinexformer:** 25.15 dB PSNR, 0.8441 SSIM, 0.1314 LPIPS,
  3.48 NIQE. That is +8 dB over the tuned classical pipeline and +6.3 dB over the fusion hybrid.
- **The Quality-Guided Hybrid** is equal on SSIM, LPIPS and NIQE at the reported precision (no significance
  test was run), 0.03 dB lower in PSNR,
  and cuts over-exposed pixels from 0.82 % to 0.03 %. It is a safeguard against clipping, not a quality
  improvement.
- **The fusion hybrid** is retained only as a documented negative result.
- These conclusions hold for LOL-v1 eval15 only.
- **After the freeze:** a colour fix for near-clipped input (section 9) leaves eval15 effectively unchanged (Natural 25.15 -> 25.16 dB, Highlight-safe 25.12 -> 25.12 dB) and fixes the orange-window failure. Its gains on pictures with bright windows are shown on synthetic high-contrast photos and LOL-v2 pairs; real photos with clipped highlights and ground truth were not available.

## 9. Update after the freeze: highlight colour fix and deployment

### 9.1 What changed
- **Colour fix** (`lumen/hybrid/highlight_hue.py`, `configs/highlight_hue.yaml`): in a soft mask of near-clipped input
  pixels (brightest channel from 0.80, full effect at 0.95), the colour is taken from the input and the model's brightness is
  kept (not darker than the input). Outside the mask the image is bit-identical.
- **Used in Natural and Highlight-safe only.** Quick is untouched. Order: model -> colour fix -> QGH (QGH is unchanged and
  still applies its highlight roll-off afterwards). Switch off with `LUMEN_HIGHLIGHT_HUE=0`.
- **Selection:** parameters and the order were chosen on **dev data only** (50 synthetic high-contrast LOL-v2 Train photos,
  90 real LOL-v2 pairs, 90 noisy/clean synthetic images). Order on the 50 dev images:

| Order | PSNR | SSIM | LPIPS | Highlight a*b* error | Highlight brightness gap vs GT | Over-exposed pixels |
|---|---|---|---|---|---|---|
| QGH only (before) | 14.71 | 0.641 | 0.370 | 63.1 | 0.305 | 1.12 % |
| colour fix -> QGH (**chosen**) | 17.76 | 0.691 | 0.298 | 3.84 | 0.055 | 0.125 % |
| QGH -> colour fix | 17.82 | 0.691 | 0.298 | 3.89 | 0.014 | 3.22 % |

The second order matches brightness slightly better but undoes QGH's protection (over-exposure back to 3.2 %), so it was not used.

### 9.2 Results, fix off vs on (same images, through the real pipeline)
`scripts/highlight_hue_integration_check.py`; "off" is the frozen behaviour. Natural / Highlight-safe. Bit-identical = outputs
that did not change at all.

| Set | PSNR off -> on | SSIM off -> on | LPIPS off -> on | NIQE off -> on | Bit-identical |
|---|---|---|---|---|---|
| eval15 (15, LOL-v1, the headline set) | 25.15 -> 25.16 / 25.12 -> 25.12 | 0.8441 -> 0.8441 | 0.1314 -> 0.1314 | 3.48 -> 3.48 | 14 / 15 |
| LOL-v2 synthetic low-light pairs (100) | 16.19 -> 17.10 / 16.51 -> **17.43** | 0.765 -> 0.781 / 0.770 -> 0.784 | 0.252 -> 0.229 / 0.252 -> 0.229 | 4.11 -> 4.10 / 4.15 -> 4.15 | 9 / 100 |
| High-contrast test photos (60, built by us) | 14.49 -> 16.86 / 14.96 -> **17.41** | 0.619 -> 0.661 / 0.626 -> 0.668 | 0.400 -> 0.339 / 0.391 -> 0.330 | 4.10 -> 4.00 / 4.07 -> 4.04 | 1 / 60 |
| Noisy synthetic (50) | unchanged | unchanged | unchanged | unchanged | 50 / 50 |
| Clean synthetic (50) | unchanged | unchanged | unchanged | unchanged | 50 / 50 |
| App examples (4) | unchanged | unchanged | unchanged | unchanged | 4 / 4 |
| Unseen real photos (20, no ground truth) | n/a | n/a | n/a | 3.82 -> 3.83 / 3.80 -> 3.80 | 4 / 20 |

Highlight colour error against ground truth (CIELAB a*b* distance, pixels with a near-clipped input): high-contrast photos
67.5 -> 4.5 (Natural), 67.0 -> 4.4 (Highlight-safe); LOL-v2 pairs 44.4 -> 9.7 and 43.6 -> 9.4. Highlight-safe over-exposure
is shown in 9.3. The high-contrast photos are synthetic and were built like the dev set (a normal photo darkened everywhere
except its bright regions), so they copy the failure by design and overstate real-world gain. On the user's own bright-window
photo the window returns to near white (Natural 0.966/0.953/0.879, Highlight-safe 0.918/0.905/0.833; before 0.75/0.49/0.08);
Quick is unchanged (0.881/0.901/0.925).

### 9.3 Trade-off: over-exposed pixels
Over-exposure = share of pixels whose brightest channel is >= 0.98.

| Set | Style | Off | On |
|---|---|---|---|
| 20 unseen real photos | **Highlight-safe** | **0.86 %** | **1.10 %** (+0.24 points) |
| 20 unseen real photos | Natural | 2.75 % | 2.73 % |
| 60 high-contrast photos | Highlight-safe | 0.95 % | 0.09 % |
| 100 LOL-v2 pairs | Highlight-safe | 1.06 % | 0.84 % |
| eval15 | Natural / Highlight-safe | 0.82 % / 0.035 % | unchanged |

On the unseen photos the share rises because a window that used to come back orange and darker (not counted as clipped) now
comes back near white, as in the original. Where ground truth exists the share falls or stays flat.

### 9.4 Latency of the fix
No extra model call. Laptop GPU, median per image: Natural +14 ms (no clipped pixels) to +80 ms (large clipped regions);
Highlight-safe +14 to +100 ms. In the Cloud Run container (2 vCPU): about 35-39 ms with a clipped window, 6-12 ms otherwise,
against 1.6-6.6 s of model time.

### 9.5 Tests and deployment
- **Tests:** 186 pass (180 earlier + 6 for the colour fix and its integration), on `main` at commit
  `7675876b9f469232cc9780b25014904fe57a5718`.
- **Container checks** (2 CPUs, 4 GiB, non-root): full smoke test passed; with `LUMEN_HIGHLIGHT_HUE=0` all 15 probe
  outputs are bit-identical to the pre-fix image; a missing or invalid fix config disables only the fix, with the reason shown
  in `/api/status`; Compare All and PNG downloads verified; graceful stop 1.3 s, exit code 0.
- **Live deployment:** Google Cloud Run, project `lumen-514004`, region `asia-south1`, revision `lumen-00003-wv5`, image
  tag `7675876` (digest `sha256:6c26bd73...`, the digest Cloud Build pushed for that commit). 2 vCPU, 4 GiB, concurrency 1,
  0-2 instances. URL: https://lumen-129721836906.asia-south1.run.app. The full smoke test, `/static/data/evaluation.json`
  (HTTP 200), the three styles, the colour fix, Compare All and PNG downloads all passed against the live URL; no errors
  in the logs. Cold start about 11 s (instance start to ready). The model file is served from a private bucket and is not in
  the repository.
- **Not verified:** the colour fix on real photos with clipped highlights *and* ground truth.
