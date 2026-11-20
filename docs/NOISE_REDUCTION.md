# LUMEN - Noise reduction for very noisy photos

**Status:** implemented on branch `research/noise-reduction` (Natural and Highlight-safe; not Quick). Selected on
development data only, frozen (commit `2075cd5`), then evaluated **once** on held-out data. The eval15 headline does
not change: every eval15 output is bit-identical.

Code: `lumen/hybrid/noise_reduction.py`. Config: `configs/noise_reduction.yaml`. Study: `scripts/noise_reduction_study.py`.
Results: `results/noise_reduction/validation.md` (local; regenerate with the commands below).

## 1. The problem

The project's weak spot (eval_P1_T66.md section 10.4): on very noisy inputs Retinexformer amplifies the noise instead of
removing it. On the real LOL-v2 pairs whose input noise sigma is 0.043-0.053, Natural reaches **SSIM 0.16-0.18 and LPIPS
about 1.0**, and the existing Quality-Guided noise step barely helps (SSIM about 0.2). Earlier noise work
(`docs/NOISE_STUDY.md`, QGH-NA v1/v2) was developed mostly on synthetic noise and had no held-out real-noise test.

## 2. Method

Order: model -> highlight hue fix -> **noise reduction** -> (Highlight-safe only) QGH refinement.

    w   = smoothstep(sigma_in; 0.030, 0.040)     sigma_in  = input noise, the same estimate the app shows as "grain"
    h   = 0.8 * sigma_out * 255                  sigma_out = noise estimate of the model output
    out = (1 - w) * r + w * NLM(r; h)            OpenCV colour non-local means, template 7, search 21

- **Below an input sigma of 0.030 nothing changes, bit for bit.** Every eval15 image (0.0036-0.0068), every LOL-v2-real
  Test image used here (max 0.0068) and every one of the 20 real phone photos (max 0.0043) is below it.
- The strength follows each image's own output noise, so it does not need retuning per noise level.
- `LUMEN_NOISE_REDUCTION=0` switches it off; a missing or invalid config disables only this step, with the reason in
  `/api/status` (`technical.noise_reduction`).

## 3. Protocol

| Data | Images | Role |
|---|---|---|
| Real dev pairs (LOL-v2-real Train minus copies of the checkpoint's training data, `dev_lolv2_real_unseen.txt`) | 269 | split by **scene** into TUNE (142: 107 very noisy) and TEST (127: 86 very noisy); fixed seed; no scene in both halves (`data/splits/noise_reduction_{tune,test}.txt`) |
| dev_syn (synthetic noisy and clean) | 90 | dev: choice of the gate |
| syn_noisy, syn_clean, val_detail, val_general, examples | 100 + 100 + 84 + 60 + 4 | held out (synthetic) |
| eval15, val_real (10 untouched LOL-v2-real Test pairs), 20 unseen phone photos | 45 | held out (real; must not change) |

- **Selection (TUNE + dev_syn only):** NLM strength by mean LPIPS on the very noisy tune images (fixed 15/20/25,
  adaptive 0.6-1.4 x output sigma). Gate from dev: every dev image with input sigma >= 0.03 improved (112 of 112), below
  0.03 the effect was mixed (about half improved, worst +0.15 LPIPS).
- **Frozen before any held-out run** (commit `2075cd5`), then `validate` was run once.
- **Caveat:** the real dev pairs were used earlier in the project (QGH calibration, noise study searches). The TEST half
  was not used for any choice in this study, but it is the same capture setup as TUNE. There is no other real noisy
  dataset with ground truth in this project.

## 4. Results (held out, run once)

**Real very noisy photos, TEST half (86 images, 56 scenes)** - paired bootstrap by scene, 10,000 resamples:

| Comparison | PSNR | SSIM | LPIPS | Better / tie / worse |
|---|---|---|---|---|
| Natural: with vs without | **+1.31 dB** [+1.12, +1.52] | **+0.478** [+0.449, +0.502] | **-0.520** [-0.540, -0.496] | 86 / 0 / 0 on every metric |
| Highlight-safe: with vs without | **+0.96 dB** [+0.81, +1.13] | **+0.438** [+0.409, +0.464] | **-0.477** [-0.496, -0.453] | 86 / 0 / 0 on every metric |

Means on those 86 images: Natural 14.25 dB / SSIM 0.185 / LPIPS 0.991 -> **15.56 / 0.663 / 0.471**.

**Everything that must not change (bit-identical, as designed):** eval15 15/15 (Natural 25.16, Highlight-safe 25.12),
val_real 10/10, the 20 real phone photos 20/20, syn_clean 100/100, and the 41 low-noise TEST images.

**Synthetic held-out sets (only images above the gate change):**

| Set | Triggered | Natural on triggered images, without -> with | Worst single image |
|---|---|---|---|
| syn_noisy | 27 / 100 | SSIM 0.339 -> 0.420, LPIPS 0.842 -> 0.681 | every one improved |
| val_general | 6 / 60 | SSIM 0.411 -> 0.482, LPIPS 0.746 -> 0.626 | one image LPIPS +0.028 |
| val_detail (text, fabric, foliage under noise) | 13 / 84 | PSNR 16.61 -> 17.12, LPIPS 0.671 -> 0.588, **SSIM 0.505 -> 0.490** | SSIM -0.129 |
| App examples | 1 / 4 (chelsea, 74 % strength) | SSIM 0.204 -> 0.443, LPIPS 1.348 -> 0.904 | - |

## 5. Limitations (stated plainly)

- **Fine texture is smoothed.** On noisy fine-detail images (handwriting, fabric) SSIM drops slightly on average even
  though LPIPS and PSNR improve. On the most extreme real noise the result looks soft and painterly: much closer to the
  reference than before, but not detail-preserving. The app's report says so when the step runs.
- **It only acts on very noisy inputs.** Most phone photos are already denoised in the camera (our 20 measure at most
  0.0043) and are not touched. Two of the four app examples (coffee 0.021, rocket 0.028) sit below the gate.
- **Real-noise evidence is one capture setup** (LOL-v2-real). No other real noisy pairs were available.
- **Latency:** about +0.25-0.45 s (+13-15 %) on very noisy photos only, in the production container (section 6).

## 6. Latency

Only very noisy photos pay anything; every other photo skips the step (the gate check costs one noise estimate of the
input, a few milliseconds).

**Production container** (the Cloud Run image built from this branch, `docker run --cpus 2 --memory 4g`, ONNX engine,
768 px cap, weights mounted read-only; `/api/restore`, best of 3, server-side timings):

| Photo | Model step | Noise reduction | Share of the total |
|---|---|---|---|
| very noisy real photo, 600x400 | 1.65 s | **+0.24 s** | about +13 % |
| the same, mirror-padded to 768x512 (deployed size) | 2.85 s | **+0.45 s** | about +15 % |
| low-noise photo (eval15 #179) | 1.71 s | 0 (skipped) | - |

The full smoke test passed in that container (all styles, compare-all reuses one network run). For comparison, on the
Windows development laptop with the process pinned to 2 logical cores the step cost +1.7 s (600x400) and +2.5 s (768 px):
OpenCV's NLM is much faster in the Linux container. A faster NLM window (5/15) was **not** the frozen setting and was not
validated.

## 7. Reproduce

```
python scripts/noise_reduction_study.py cache --sets dev_real eval15 val_real unseen dev_syn examples syn_noisy syn_clean val_detail val_general
python scripts/noise_reduction_study.py split        # writes data/splits/noise_reduction_{tune,test}.txt (already committed)
python scripts/noise_reduction_study.py baseline     # TUNE only
python scripts/noise_reduction_study.py explore      # TUNE only
python scripts/noise_reduction_study.py validate     # held out; run once with the frozen config
```
