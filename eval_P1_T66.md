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
   (22.54 → 23.61) but lowered eval15 to 24.44, so it was discarded. A post-freeze analysis (section 10.4) shows
   that this gap comes from the very noisy inputs in that split, not from the low-noise ones.
5. **Small test set:** eval15 has 15 images from one capture setup. Differences of a few hundredths of a dB
   (25.12 vs 25.15) are negligible in size; a paired bootstrap run after the freeze (section 10.1) finds the PSNR
   sign consistent but no SSIM or LPIPS difference. No claim is made about other
   cameras, scenes or datasets. The 20 personal "unseen" photos have no ground truth and are not scored in
   sections 4-6; section 10.3 scores them without a reference.
6. **Metrics:** NIQE was computed with pyiqa defaults. Latency is GPU latency on one laptop; CPU and container
   figures are in `docs/DEPLOYMENT.md`.
7. **Bright windows and lamps (found after the freeze):** on a photo of a dim room with a bright window, the Natural and Highlight-safe outputs turned the white window **orange** (window RGB 0.96/0.95/0.89 became about 0.75/0.49/0.08). The eval15 images almost never contain clipped input, so this was invisible in the headline numbers. Cause: Retinexformer divides by an estimated illumination, and where the input is already clipped the colour channels are corrected unequally; LOL has no such scenes. QGH's roll-off only changes brightness. Addressed in section 9 (with a documented trade-off).

## 8. Conclusion
- **On this test set the best method is standalone Retinexformer:** 25.15 dB PSNR, 0.8441 SSIM, 0.1314 LPIPS,
  3.48 NIQE. That is +8 dB over the tuned classical pipeline and +6.3 dB over the fusion hybrid.
- **The Quality-Guided Hybrid** is equal on SSIM, LPIPS and NIQE at the reported precision (a paired bootstrap
  after the freeze, section 10.1, finds no SSIM or LPIPS difference), 0.03 dB lower in PSNR,
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

<!-- phase-f:start -->
## 10. Post-freeze additions: statistical checks

Everything in this section was added **after** the Phase 5 freeze. It changes no model, configuration, threshold or headline
number: sections 4-6 stand as written. Source: `scripts/hardening_study.py`, results in `results/phase_f_hardening/`.
Intervals are 95 % paired bootstrap percentile intervals (10,000 resamples, seed 20261119). The resampling unit is a
*scene* (images whose thumbnails are near-duplicates count as one), and the naive per-image interval is shown beside it. Intervals
are not adjusted for the number of comparisons. The interval averages over images and resamples scenes; the sign-flip p-value tests the per-scene
means, so for sets with scenes of unequal size the two can disagree near the boundary. A / tie / A worse count images, with ties below the reported precision.

### 10.0 In short

- **Retinexformer against the classical pipeline is a solid result on both benchmarks:** +7.991 dB on eval15 (95 % interval [+6.182, +9.813], better on 15 of 15 images) and +4.398 dB on LOL-v2-real Test with its matching checkpoint ([+2.993, +5.609], 88 of 100 images, 43 scenes).
- **Quality Guided costs almost nothing on these low-noise sets:** on eval15 plain Retinexformer is better on PSNR by +0.032 dB ([+0.006, +0.060]), while the SSIM difference is not distinguishable and the LPIPS difference is not distinguishable.
- **Very noisy inputs are the weak spot.** Of the 269 dev images, 193 have an input noise σ of 0.041–0.053, against at most 0.015 in any benchmark image. On those, Retinexformer against the classical pipeline is not distinguishable on PSNR (-0.136 dB) and worse on SSIM (-0.0190). Quality Guided is better than plain Retinexformer on 193 of 193 of them (+0.418 dB PSNR).
- **No benchmark image and no unseen photo is that noisy**, so the headline numbers describe the low-noise regime only.
- **On the 20 unseen photos (no ground truth)** Highlight-safe clips 1.62 percentage points fewer pixels than Natural ([0.47, 3.05]; fewer on 20 of 20 photos); NIQE cannot tell the styles apart.

### 10.1 eval15: how far can 15 images be trusted?

15 images, 15 scenes, checkpoint `retinexformer_LOL_v1.pth`. The three frozen methods were measured again and reproduce the frozen per-image results (largest per-image difference: 0.0000 dB PSNR, 0.0000 SSIM, 0.0000 LPIPS). Near-duplicate neighbours in the Retinexformer training set (LOL-v1 our485): 6 of 15 eval15 images at a strict threshold, 8 at a looser one.

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Classical | 17.16 | 0.7241 | 0.2781 |
| Retinexformer | 25.15 | 0.8441 | 0.1314 |
| Quality Guided (frozen) | 25.12 | 0.8441 | 0.1314 |
| Natural (app) | 25.16 | 0.8441 | 0.1314 |
| Highlight-safe (app) | 25.12 | 0.8441 | 0.1314 |

| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |
|---|---|---|---|---|---|---|---|
| Retinexformer − Quality Guided (frozen) | PSNR (dB) ↑ | +0.032 | [+0.006, +0.060] | [+0.006, +0.060] | yes | 0.035 | 7 / 7 / 1 |
| Retinexformer − Classical | PSNR (dB) ↑ | +7.991 | [+6.182, +9.813] | [+6.182, +9.813] | yes | < 0.001 | 15 / 0 / 0 |
| Quality Guided (frozen) − Classical | PSNR (dB) ↑ | +7.959 | [+6.154, +9.769] | [+6.154, +9.769] | yes | < 0.001 | 15 / 0 / 0 |
| Natural (app) − Retinexformer | PSNR (dB) ↑ | +0.004 | [+0.000, +0.011] | [+0.000, +0.011] | no | 1.000 | 1 / 14 / 0 |
| Highlight-safe (app) − Quality Guided (frozen) | PSNR (dB) ↑ | +0.004 | [+0.000, +0.011] | [+0.000, +0.011] | no | 1.000 | 1 / 14 / 0 |
| Retinexformer − Quality Guided (frozen) | SSIM ↑ | -0.0001 | [-0.0005, +0.0003] | [-0.0005, +0.0003] | no | 0.855 | 5 / 7 / 3 |
| Retinexformer − Classical | SSIM ↑ | +0.1200 | [+0.0985, +0.1431] | [+0.0985, +0.1431] | yes | < 0.001 | 15 / 0 / 0 |
| Quality Guided (frozen) − Classical | SSIM ↑ | +0.1200 | [+0.0989, +0.1429] | [+0.0989, +0.1429] | yes | < 0.001 | 15 / 0 / 0 |
| Natural (app) − Retinexformer | SSIM ↑ | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | no | 1.000 | 1 / 14 / 0 |
| Highlight-safe (app) − Quality Guided (frozen) | SSIM ↑ | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | no | 1.000 | 1 / 14 / 0 |
| Retinexformer − Quality Guided (frozen) | LPIPS ↓ | +0.0000 | [-0.0004, +0.0005] | [-0.0004, +0.0005] | no | 0.984 | 3 / 9 / 3 |
| Retinexformer − Classical | LPIPS ↓ | -0.1467 | [-0.1860, -0.1120] | [-0.1860, -0.1120] | yes | < 0.001 | 15 / 0 / 0 |
| Quality Guided (frozen) − Classical | LPIPS ↓ | -0.1467 | [-0.1857, -0.1121] | [-0.1857, -0.1121] | yes | < 0.001 | 15 / 0 / 0 |
| Natural (app) − Retinexformer | LPIPS ↓ | +0.0000 | [+0.0000, +0.0001] | [+0.0000, +0.0001] | no | 1.000 | 0 / 14 / 1 |
| Highlight-safe (app) − Quality Guided (frozen) | LPIPS ↓ | +0.0000 | [+0.0000, +0.0001] | [+0.0000, +0.0001] | no | 1.000 | 0 / 14 / 1 |

### 10.2 LOL-v2-real Test with the LOL-v2-real checkpoint

100 images in 43 scenes, checkpoint `retinexformer_LOL_v2_real.pth` (not the LOL-v1 checkpoint: 90 of these images are copies of its training set). Leakage check for *this* checkpoint: images with a near-duplicate in LOL-v2-real Train: 0 (strict), 0 (loose), 28 (very loose: neighbouring shots). The Quality Guided thresholds were calibrated on LOL-v1-checkpoint outputs, so its rows here are a transfer test, not a re-calibration.

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Classical | 18.40 | 0.7394 | 0.2887 |
| Retinexformer | 22.80 | 0.8392 | 0.1708 |
| Quality Guided (frozen) | 22.78 | 0.8388 | 0.1720 |
| Natural (app) | 22.85 | 0.8392 | 0.1706 |
| Highlight-safe (app) | 22.84 | 0.8388 | 0.1718 |

| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |
|---|---|---|---|---|---|---|---|
| Retinexformer − Quality Guided (frozen) | PSNR (dB) ↑ | +0.011 | [+0.003, +0.020] | [+0.006, +0.017] | yes | 0.009 | 22 / 73 / 5 |
| Retinexformer − Classical | PSNR (dB) ↑ | +4.398 | [+2.993, +5.609] | [+3.643, +5.122] | yes | < 0.001 | 88 / 0 / 12 |
| Quality Guided (frozen) − Classical | PSNR (dB) ↑ | +4.387 | [+2.984, +5.596] | [+3.635, +5.112] | yes | < 0.001 | 88 / 0 / 12 |
| Natural (app) − Retinexformer | PSNR (dB) ↑ | +0.055 | [-0.002, +0.172] | [+0.008, +0.119] | no | 1.000 | 4 / 94 / 2 |
| Highlight-safe (app) − Quality Guided (frozen) | PSNR (dB) ↑ | +0.055 | [-0.002, +0.172] | [+0.008, +0.119] | no | 1.000 | 4 / 94 / 2 |
| Retinexformer − Quality Guided (frozen) | SSIM ↑ | +0.0004 | [-0.0000, +0.0012] | [+0.0000, +0.0011] | no | 0.145 | 16 / 70 / 14 |
| Retinexformer − Classical | SSIM ↑ | +0.0998 | [+0.0845, +0.1153] | [+0.0888, +0.1104] | yes | < 0.001 | 93 / 0 / 7 |
| Quality Guided (frozen) − Classical | SSIM ↑ | +0.0994 | [+0.0842, +0.1148] | [+0.0884, +0.1101] | yes | < 0.001 | 93 / 0 / 7 |
| Natural (app) − Retinexformer | SSIM ↑ | +0.0000 | [-0.0000, +0.0001] | [-0.0000, +0.0001] | no | 1.000 | 4 / 91 / 5 |
| Highlight-safe (app) − Quality Guided (frozen) | SSIM ↑ | +0.0000 | [-0.0000, +0.0001] | [-0.0000, +0.0001] | no | 1.000 | 4 / 91 / 5 |
| Retinexformer − Quality Guided (frozen) | LPIPS ↓ | -0.0012 | [-0.0037, -0.0000] | [-0.0028, -0.0001] | yes | 0.012 | 23 / 73 / 4 |
| Retinexformer − Classical | LPIPS ↓ | -0.1179 | [-0.1360, -0.0987] | [-0.1311, -0.1042] | yes | < 0.001 | 96 / 0 / 4 |
| Quality Guided (frozen) − Classical | LPIPS ↓ | -0.1167 | [-0.1347, -0.0977] | [-0.1297, -0.1032] | yes | < 0.001 | 96 / 0 / 4 |
| Natural (app) − Retinexformer | LPIPS ↓ | -0.0002 | [-0.0006, +0.0001] | [-0.0004, -0.0000] | no | 0.876 | 6 / 90 / 4 |
| Highlight-safe (app) − Quality Guided (frozen) | LPIPS ↓ | -0.0002 | [-0.0006, +0.0001] | [-0.0004, -0.0000] | no | 0.876 | 6 / 90 / 4 |

### 10.3 No-reference scores (no ground truth)

**unseen phone photos** (20 photos). Lower is better for every column.

| Method | NIQE | Noise σ | Over-exposed % |
|---|---|---|---|
| Classical | 3.90 | 0.0061 | 0.00 |
| Natural (app) | 3.83 | 0.0041 | 2.72 |
| Highlight-safe (app) | 3.80 | 0.0039 | 1.10 |

| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |
|---|---|---|---|---|---|---|---|
| Natural (app) − Classical | NIQE ↓ | -0.07 | [-0.38, +0.34] | [-0.38, +0.34] | no | 0.794 | 12 / 1 / 7 |
| Highlight-safe (app) − Classical | NIQE ↓ | -0.10 | [-0.32, +0.17] | [-0.32, +0.17] | no | 0.489 | 11 / 0 / 9 |
| Highlight-safe (app) − Natural (app) | NIQE ↓ | -0.03 | [-0.18, +0.09] | [-0.18, +0.09] | no | 0.957 | 5 / 4 / 11 |
| Natural (app) − Classical | Noise σ ↓ | -0.0020 | [-0.0039, -0.0005] | [-0.0039, -0.0005] | yes | 0.014 | 15 / 0 / 5 |
| Highlight-safe (app) − Classical | Noise σ ↓ | -0.0022 | [-0.0041, -0.0007] | [-0.0041, -0.0007] | yes | 0.005 | 15 / 1 / 4 |
| Highlight-safe (app) − Natural (app) | Noise σ ↓ | -0.0002 | [-0.0004, -0.0000] | [-0.0004, -0.0000] | yes | < 0.001 | 5 / 15 / 0 |
| Natural (app) − Classical | Over-exposed % ↓ | +2.72 | [+1.15, +4.55] | [+1.15, +4.55] | yes | < 0.001 | 0 / 0 / 20 |
| Highlight-safe (app) − Classical | Over-exposed % ↓ | +1.10 | [+0.35, +2.11] | [+0.35, +2.11] | yes | < 0.001 | 0 / 1 / 19 |
| Highlight-safe (app) − Natural (app) | Over-exposed % ↓ | -1.62 | [-3.05, -0.47] | [-3.05, -0.47] | yes | < 0.001 | 20 / 0 / 0 |

Not scored because no images were provided: DICM, LIME. Put them in `data/noref/<name>/` and rerun `python scripts/hardening_study.py noref`.

No ground truth exists for these photos, so these numbers say nothing about fidelity. NIQE rewards natural-looking statistics, not accuracy.

### 10.4 Where is the model weakest? (input noise and brightness)

The dev split (LOL-v2-real Train without copies of the LOL-v1 checkpoint's training data) holds two populations of input noise (estimated σ): 76 images between 0.0025 and 0.0063, and 193 images between 0.0413 and 0.0527, with none in between. The split used here, σ = 0.025, sits in that gap; it was chosen after looking at the data, so treat it as exploratory. Across the dev split, input noise and Retinexformer's SSIM correlate at -0.94, and input noise and brightness at +0.68, which is why brightness is examined only within the low-noise group.

Retinexformer against Classical: in the **low-noise** group it is better on PSNR, better on SSIM and better on LPIPS; in the **high-noise** group it is not distinguishable on PSNR, worse on SSIM and not distinguishable on LPIPS (95 % intervals by scene, tables below).

Input noise (estimated σ) of the other evidence in this report: eval15: 0.0036–0.0068; LOL-v2-real Test: 0.0038–0.0148; the 20 unseen phone photos: 0.0000–0.0043 (median 0.0008). **No benchmark image and no unseen photo falls in the high-noise group**, so every headline number in this report describes the low-noise regime only.

#### eval15 (LOL-v1 checkpoint)

| Input noise | n (scenes) | Method | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| low (σ < 0.025); observed 0.0036–0.0068 | 15 (15) | Classical | 17.16 | 0.7241 | 0.2781 |
|  |  | Retinexformer | 25.15 | 0.8441 | 0.1314 |
|  |  | Quality Guided (frozen) | 25.12 | 0.8441 | 0.1314 |

| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |
|---|---|---|---|---|---|---|---|
| low: Retinexformer − Classical | PSNR (dB) ↑ | +7.991 | [+6.182, +9.813] | [+6.182, +9.813] | yes | < 0.001 | 15 / 0 / 0 |
| low: Retinexformer − Classical | SSIM ↑ | +0.1200 | [+0.0985, +0.1431] | [+0.0985, +0.1431] | yes | < 0.001 | 15 / 0 / 0 |
| low: Retinexformer − Classical | LPIPS ↓ | -0.1467 | [-0.1860, -0.1120] | [-0.1860, -0.1120] | yes | < 0.001 | 15 / 0 / 0 |
| low: Quality Guided (frozen) − Retinexformer | PSNR (dB) ↑ | -0.032 | [-0.060, -0.006] | [-0.060, -0.006] | yes | 0.035 | 1 / 7 / 7 |
| low: Quality Guided (frozen) − Retinexformer | SSIM ↑ | +0.0001 | [-0.0003, +0.0005] | [-0.0003, +0.0005] | no | 0.855 | 3 / 7 / 5 |
| low: Quality Guided (frozen) − Retinexformer | LPIPS ↓ | -0.0000 | [-0.0005, +0.0004] | [-0.0005, +0.0004] | no | 0.984 | 3 / 9 / 3 |

Brightness within the low-noise group only (so that darkness is not confused with noise):

| Third of input brightness | Input brightness | n | Classical PSNR | Retinexformer PSNR | Quality Guided PSNR | Retinexformer SSIM | Retinexformer LPIPS |
|---|---|---|---|---|---|---|---|
| darkest third | 0.025–0.038 | 5 | 15.71 | 23.95 | 23.94 | 0.7931 | 0.1767 |
| middle third | 0.041–0.076 | 5 | 18.68 | 26.09 | 26.04 | 0.8660 | 0.1181 |
| brightest third | 0.076–0.097 | 5 | 17.09 | 25.42 | 25.38 | 0.8731 | 0.0993 |

#### LOL-v2-real Test (LOL-v2-real checkpoint)

| Input noise | n (scenes) | Method | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| low (σ < 0.025); observed 0.0038–0.0148 | 100 (43) | Classical | 18.40 | 0.7394 | 0.2887 |
|  |  | Retinexformer | 22.80 | 0.8392 | 0.1708 |
|  |  | Quality Guided (frozen) | 22.78 | 0.8388 | 0.1720 |

| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |
|---|---|---|---|---|---|---|---|
| low: Retinexformer − Classical | PSNR (dB) ↑ | +4.398 | [+2.993, +5.609] | [+3.643, +5.122] | yes | < 0.001 | 88 / 0 / 12 |
| low: Retinexformer − Classical | SSIM ↑ | +0.0998 | [+0.0845, +0.1153] | [+0.0888, +0.1104] | yes | < 0.001 | 93 / 0 / 7 |
| low: Retinexformer − Classical | LPIPS ↓ | -0.1179 | [-0.1360, -0.0987] | [-0.1311, -0.1042] | yes | < 0.001 | 96 / 0 / 4 |
| low: Quality Guided (frozen) − Retinexformer | PSNR (dB) ↑ | -0.011 | [-0.020, -0.003] | [-0.017, -0.006] | yes | 0.009 | 5 / 73 / 22 |
| low: Quality Guided (frozen) − Retinexformer | SSIM ↑ | -0.0004 | [-0.0012, +0.0000] | [-0.0011, -0.0000] | no | 0.145 | 14 / 70 / 16 |
| low: Quality Guided (frozen) − Retinexformer | LPIPS ↓ | +0.0012 | [+0.0000, +0.0037] | [+0.0001, +0.0028] | yes | 0.012 | 4 / 73 / 23 |

Brightness within the low-noise group only (so that darkness is not confused with noise):

| Third of input brightness | Input brightness | n | Classical PSNR | Retinexformer PSNR | Quality Guided PSNR | Retinexformer SSIM | Retinexformer LPIPS |
|---|---|---|---|---|---|---|---|
| darkest third | 0.013–0.041 | 34 | 18.36 | 22.94 | 22.93 | 0.8281 | 0.1974 |
| middle third | 0.041–0.059 | 33 | 19.13 | 23.06 | 23.04 | 0.8397 | 0.1580 |
| brightest third | 0.060–0.142 | 33 | 17.70 | 22.38 | 22.37 | 0.8502 | 0.1561 |

#### Dev split (LOL-v1 checkpoint)

| Input noise | n (scenes) | Method | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| low (σ < 0.025); observed 0.0025–0.0063 | 76 (65) | Classical | 14.42 | 0.5591 | 0.5214 |
|  |  | Retinexformer | 17.97 | 0.7354 | 0.3108 |
|  |  | Quality Guided (frozen) | 17.96 | 0.7353 | 0.3110 |
| high (σ ≥ 0.025); observed 0.0413–0.0527 | 193 (111) | Classical | 14.89 | 0.1919 | 1.0392 |
|  |  | Retinexformer | 14.75 | 0.1729 | 1.0369 |
|  |  | Quality Guided (frozen) | 15.17 | 0.2136 | 0.9913 |

| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |
|---|---|---|---|---|---|---|---|
| low: Retinexformer − Classical | PSNR (dB) ↑ | +3.547 | [+2.753, +4.336] | [+2.796, +4.281] | yes | < 0.001 | 67 / 0 / 9 |
| low: Retinexformer − Classical | SSIM ↑ | +0.1763 | [+0.1551, +0.1971] | [+0.1564, +0.1960] | yes | < 0.001 | 75 / 0 / 1 |
| low: Retinexformer − Classical | LPIPS ↓ | -0.2106 | [-0.2325, -0.1892] | [-0.2296, -0.1912] | yes | < 0.001 | 76 / 0 / 0 |
| low: Quality Guided (frozen) − Retinexformer | PSNR (dB) ↑ | -0.010 | [-0.020, -0.002] | [-0.020, -0.002] | yes | 0.023 | 1 / 63 / 12 |
| low: Quality Guided (frozen) − Retinexformer | SSIM ↑ | -0.0001 | [-0.0004, +0.0000] | [-0.0004, +0.0000] | no | 0.130 | 7 / 61 / 8 |
| low: Quality Guided (frozen) − Retinexformer | LPIPS ↓ | +0.0002 | [+0.0000, +0.0004] | [+0.0000, +0.0004] | yes | 0.109 | 7 / 61 / 8 |

| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |
|---|---|---|---|---|---|---|---|
| high: Retinexformer − Classical | PSNR (dB) ↑ | -0.136 | [-0.709, +0.368] | [-0.466, +0.191] | no | 0.587 | 130 / 0 / 63 |
| high: Retinexformer − Classical | SSIM ↑ | -0.0190 | [-0.0339, -0.0064] | [-0.0279, -0.0101] | yes | < 0.001 | 114 / 0 / 79 |
| high: Retinexformer − Classical | LPIPS ↓ | -0.0023 | [-0.0177, +0.0137] | [-0.0104, +0.0057] | no | 0.027 | 90 / 0 / 103 |
| high: Quality Guided (frozen) − Retinexformer | PSNR (dB) ↑ | +0.418 | [+0.364, +0.478] | [+0.393, +0.444] | yes | < 0.001 | 193 / 0 / 0 |
| high: Quality Guided (frozen) − Retinexformer | SSIM ↑ | +0.0408 | [+0.0385, +0.0432] | [+0.0390, +0.0425] | yes | < 0.001 | 193 / 0 / 0 |
| high: Quality Guided (frozen) − Retinexformer | LPIPS ↓ | -0.0456 | [-0.0505, -0.0413] | [-0.0482, -0.0430] | yes | < 0.001 | 193 / 0 / 0 |

Brightness within the low-noise group only (so that darkness is not confused with noise):

| Third of input brightness | Input brightness | n | Classical PSNR | Retinexformer PSNR | Quality Guided PSNR | Retinexformer SSIM | Retinexformer LPIPS |
|---|---|---|---|---|---|---|---|
| darkest third | 0.001–0.016 | 26 | 11.94 | 15.58 | 15.58 | 0.6492 | 0.4696 |
| middle third | 0.016–0.027 | 25 | 14.71 | 18.47 | 18.47 | 0.7492 | 0.2361 |
| brightest third | 0.027–0.059 | 25 | 16.70 | 19.94 | 19.92 | 0.8112 | 0.2202 |

Dev split overall (269 images, 176 scenes):

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Classical | 14.75 | 0.2956 | 0.8929 |
| Retinexformer | 15.66 | 0.3318 | 0.8317 |
| Quality Guided (frozen) | 15.96 | 0.3610 | 0.7991 |

| Comparison (A − B) | Metric | Mean difference | 95 % interval (by scene) | Naive interval (by image) | Zero excluded? | Sign-flip p | A better / tie / A worse |
|---|---|---|---|---|---|---|---|
| Retinexformer − Quality Guided (frozen) | PSNR (dB) ↑ | -0.297 | [-0.362, -0.241] | [-0.326, -0.267] | yes | < 0.001 | 12 / 63 / 194 |
| Retinexformer − Classical | PSNR (dB) ↑ | +0.904 | [+0.393, +1.406] | [+0.531, +1.277] | yes | < 0.001 | 197 / 0 / 72 |
| Quality Guided (frozen) − Classical | PSNR (dB) ↑ | +1.202 | [+0.677, +1.703] | [+0.830, +1.574] | yes | < 0.001 | 203 / 0 / 66 |
| Retinexformer − Quality Guided (frozen) | SSIM ↑ | -0.0292 | [-0.0324, -0.0257] | [-0.0317, -0.0266] | yes | < 0.001 | 8 / 61 / 200 |
| Retinexformer − Classical | SSIM ↑ | +0.0362 | [+0.0202, +0.0531] | [+0.0229, +0.0499] | yes | < 0.001 | 189 / 0 / 80 |
| Quality Guided (frozen) − Classical | SSIM ↑ | +0.0654 | [+0.0510, +0.0801] | [+0.0537, +0.0772] | yes | < 0.001 | 217 / 0 / 52 |
| Retinexformer − Quality Guided (frozen) | LPIPS ↓ | +0.0327 | [+0.0288, +0.0366] | [+0.0295, +0.0357] | yes | < 0.001 | 8 / 61 / 200 |
| Retinexformer − Classical | LPIPS ↓ | -0.0611 | [-0.0777, -0.0453] | [-0.0751, -0.0478] | yes | < 0.001 | 166 / 0 / 103 |
| Quality Guided (frozen) − Classical | LPIPS ↓ | -0.0938 | [-0.1077, -0.0803] | [-0.1055, -0.0826] | yes | < 0.001 | 234 / 0 / 35 |
<!-- phase-f:end -->
