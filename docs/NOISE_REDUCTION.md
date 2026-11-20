# LUMEN - Noise reduction for grainy photos

**Status:** implemented on branch `research/noise-reduction` (Natural and Highlight-safe; not Quick). Two iterations,
each selected on development data only, frozen in a commit, then evaluated **once** on held-out data it had never seen.
The eval15 headline does not change: every eval15 output is bit-identical.

Code: `lumen/hybrid/noise_reduction.py`. Config: `configs/noise_reduction.yaml`. Study: `scripts/noise_reduction_study.py`.
Results: `results/noise_reduction/validation.md` (local; regenerate with the commands in section 7).

## 1. The problem

Retinexformer amplifies sensor noise instead of removing it (eval_P1_T66.md section 10.4).
- **Heavy grain:** on the real LOL-v2 pairs whose input noise sigma is 0.043-0.053, Natural reaches SSIM 0.16-0.18 and
  LPIPS about 1.0; the existing Quality-Guided noise step barely helps.
- **Moderate grain:** on photos like the app's coffee example (input sigma 0.021) the colour speckle stays clearly
  visible. Iteration 1 did not act on these; iteration 2 does.

## 2. Method

Order: model -> highlight hue fix -> **noise reduction** -> (Highlight-safe only) QGH refinement. The filter is chosen by
the noise of the *input*, the same estimate the app shows as "grain":

| Input noise sigma | What runs | Why |
|---|---|---|
| below 0.010 | nothing: **bit-identical** | every eval15 image (max 0.0068), every real phone photo we have (max 0.0043), every clean dev image (max 0.009) |
| 0.010 - 0.030 (ramp in to 0.014) | **detail-protecting noise-aware filter** (QGH-NA v2 from `docs/NOISE_STUDY.md` section 9, parameters unchanged): grain/texture separated by how the estimate scales, texture relief, chroma filtered only where luma grain was found | NLM was rejected here on dev data: it erased text strokes (worst noisy-detail image +0.30 LPIPS) |
| 0.030 - 0.040 | blend towards NLM | - |
| above 0.040 | **colour non-local means**, h = 0.8 x the output's own noise sigma (x 255), template 7, search 21 | strongest on real heavy noise; exactly iteration 1 |

`LUMEN_NOISE_REDUCTION=0` switches it off; a missing or invalid config disables only this step, with the reason in
`/api/status` (`technical.noise_reduction`). The result info reports the `level` (moderate / mixed / heavy), which the
app uses for its wording ("reduced grain" / "removed heavy grain").

## 3. Protocol

| Data | Images | Role |
|---|---|---|
| Real dev pairs (LOL-v2-real Train minus copies of the checkpoint's training data) | 269 | split by **scene** into TUNE (142: 107 very noisy) and TEST (127: 86 very noisy), fixed seed, no scene in both halves (`data/splits/noise_reduction_{tune,test}.txt`) |
| dev_syn (90), dev_detail (84: text, wood, fabric, textured photos; noise-free and noisy) | 174 | dev |
| **fresh** (iteration 2's held-out set) | 180 | 90 new scenes, each noise-free and noisy: 50 unused LOL-v2 Synthetic photos, 16 unused textured photos, 24 newly generated text / print / wood / fabric images; new noise draws; near-duplicate-checked against every earlier split (`data/splits/noise_reduction_fresh.txt`) |
| syn_noisy, syn_clean, val_detail, val_general, examples | 348 | held out for iteration 1; regression checks afterwards |
| eval15, val_real (10 untouched LOL-v2-real Test pairs), 20 real phone photos | 45 | must not change |

- **Iteration 1** (heavy grain): NLM strength chosen on the very noisy TUNE images; gate where every dev image improved.
  Frozen at `2075cd5`, then validated once (TEST half + the synthetic sets).
- **Iteration 2** (moderate grain): gate and filter chosen on dev_syn, dev_detail and TUNE. Policies (gate x NLM
  strength x texture safeguard) were scored against constraints fixed beforehand: clean images untouched, no noisy
  fine-detail image worse than +0.02 LPIPS, no noisy synthetic image worse than +0.05. No lowered-gate NLM policy was
  feasible; the noise-aware filter was (dev: noisy detail LPIPS 0.377 -> 0.310, worst +0.008). Frozen at `68b8bba`, then
  validated once on **fresh**.
- **Caveats:** the noise-aware filter's own parameters were selected earlier on dev_detail and the real dev pairs
  (docs/NOISE_STUDY.md), so only fresh is new to it. The real dev pairs were used earlier in the project; the TEST half
  was not used for any choice here. Moderate-grain evidence is **synthetic** (Poisson-Gaussian sensor noise): the real
  noisy pairs in this project are all heavy (0.043-0.053), with none between 0.0063 and 0.041.

## 4. Results (held out, each run once)

Paired bootstrap by scene, 10,000 resamples, 95 % intervals. Natural with vs without noise reduction (Highlight-safe
gains slightly less on every row):

| Held-out group | PSNR | SSIM | LPIPS | Better / tie / worse (LPIPS) |
|---|---|---|---|---|
| **Real heavy grain**, TEST half (86 images, 56 scenes) | **+1.31 dB** [+1.12, +1.52] | **+0.478** [+0.449, +0.502] | **-0.520** [-0.540, -0.496] | 86 / 0 / 0 (every metric) |
| **Moderate grain**, fresh (50 noisy scenes, sigma 0.010-0.030) | **+0.26 dB** [+0.16, +0.38] | **+0.050** [+0.035, +0.066] | **-0.079** [-0.105, -0.054] | 39 / 5 / 6 |
| Heavy grain, fresh synthetic (25 scenes) | +1.17 dB [+0.74, +1.64] | +0.163 [+0.085, +0.246] | -0.308 [-0.395, -0.220] | 23 / 0 / 2 |
| Noisy fine detail, fresh (33 scenes: text, wood, fabric, textured photos) | +0.27 dB [+0.08, +0.53] | +0.008 [-0.012, +0.029] (not distinguishable) | -0.052 [-0.088, -0.021] | 23 / 4 / 6 |
| Noise-free twins, fresh (90) | -0.0005 dB | 0.0000 | +0.0002 | 89 identical, 1 negligibly changed |

Means on the real heavy-grain TEST images: Natural 14.25 dB / SSIM 0.185 / LPIPS 0.991 -> **15.56 / 0.663 / 0.471**.

**Unchanged, bit for bit:** eval15 15/15 (Natural 25.16, Highlight-safe 25.12), val_real 10/10, the 20 real phone photos
20/20, syn_clean 100/100, and the 41 low-noise TEST images.

**The app's four examples** (synthetic noise, with clean originals), Natural LPIPS / SSIM:

| Example | Input sigma | Level | Without | With |
|---|---|---|---|---|
| coffee_dark | 0.021 | moderate | 0.527 / 0.497 | **0.371 / 0.573** |
| astronaut_uneven_light | 0.019 | moderate | 0.586 / 0.498 | **0.432 / 0.560** |
| rocket_night_highlights | 0.028 | moderate | 0.788 / 0.250 | **0.541 / 0.464** |
| chelsea_extremely_dark | 0.037 | mixed | 1.348 / 0.204 | **0.717 / 0.543** |

**Regression check on the iteration-1 synthetic sets** (now also covering their moderate images; these sets were seen
before, so they are not a claim): syn_noisy 83 triggered, LPIPS 0.626 -> 0.438 (worst image +0.007); val_detail 42
triggered, LPIPS 0.361 -> 0.284 (worst +0.008); val_general 26 triggered, LPIPS 0.604 -> 0.453 (worst +0.065, the same
image the noise-aware filter already failed in docs/NOISE_STUDY.md).

## 5. Limitations (stated plainly)

- **Heavy grain: fine texture is smoothed.** NLM removes heavy grain but also fine texture: on fresh heavy-grain scenes
  SSIM drops on 9 of 25 (worst single image, a textured photo: +0.14 LPIPS, -0.14 SSIM). On extreme real noise the
  result looks soft and painterly: much closer to the reference than before, but not detail-preserving.
- **Moderate grain: reduced, not removed.** The colour speckle mostly goes; some low-frequency mottling remains (visible on
  the coffee saucer). 6 of the 50 fresh moderate scenes got slightly worse on LPIPS (worst +0.02).
- **Moderate-grain evidence is synthetic.** No real photos with moderate noise and ground truth were available.
- **Phone photos are mostly untouched**: cameras denoise in the device; our 20 measure at most 0.0043.
- **Latency** (production container, 2 CPUs): +0.14-0.18 s for moderate grain, +0.24-0.56 s for heavy grain, nothing
  for other photos (section 6).

## 6. Latency

Photos below the floor skip the step (the check costs one noise estimate of the input, a few milliseconds).

**Production container**, heavy grain (the Cloud Run image built from this branch, `docker run --cpus 2 --memory 4g`,
ONNX engine, 768 px cap, weights mounted read-only; `/api/restore`, best of 3, server-side timings):

| Photo | Model step | Noise reduction | Share of the total |
|---|---|---|---|
| very noisy real photo, 600x400 | 1.65 s | **+0.24 s** | about +13 % |
| the same, mirror-padded to 768x512 (deployed size) | 2.85 s | **+0.45 s** | about +15 % |
| low-noise photo (eval15 #179) | 1.71 s | 0 (skipped) | - |

**Production container**, moderate grain (iteration 2 image, same settings): the app's coffee example (input sigma
0.021) took **+0.14 s** for noise reduction at 600x400 and **+0.18 s** mirror-padded to 768x512 (deployed size). In that
run the heavy-grain filter took +0.56 s at 768x512 (0.45 s in the first run: timings vary by about 0.1 s). The full smoke
test passed in both containers (all styles; compare-all reuses one network run).

## 7. Reproduce

```
python scripts/noise_reduction_study.py cache --sets dev_real eval15 val_real unseen dev_syn examples syn_noisy syn_clean val_detail val_general dev_detail
python scripts/noise_reduction_study.py split            # data/splits/noise_reduction_{tune,test}.txt (committed)
python scripts/noise_reduction_study.py baseline         # TUNE only
python scripts/noise_reduction_study.py explore          # TUNE only (iteration 1)
python scripts/noise_reduction_study.py explore_moderate # dev only (iteration 2)
python scripts/noise_reduction_study.py build_fresh      # the fresh held-out set (split committed)
python scripts/noise_reduction_study.py cache --sets fresh
python scripts/noise_reduction_study.py validate         # held out; run once per frozen config
```
