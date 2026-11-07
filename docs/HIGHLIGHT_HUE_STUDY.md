# Highlight hue-preservation study (experimental branch `experiment/highlight-hue`)

**Status: experimental. Not integrated; production code, configs and deployment are unchanged.**

**Problem.** Where the input is already (nearly) clipped, such as a bright window or lamp in a dark room, Retinexformer's
illumination division moves the colour channels by different amounts, so white turns orange and the region gets darker
(measured on a user photo: window RGB 0.96/0.95/0.89 -> 0.75/0.48/0.08). QGH's highlight step only rolls off luminance.

**Fix** (`lumen/experimental/highlight_hue.py`). In a soft mask of near-clipped input pixels, take the colour from the
input and keep the model's brightness, with a floor so the region is not darker than the input. Outside the mask the
image is bit-identical. Nothing is learned; the model is untouched.

**Selection (dev only):** 48 settings on 50 synthetic high-contrast LOL-v2 Train photos, 90 real LOL-v2 pairs and 90
noisy/clean synthetic images. Chosen: `start 0.80, full 0.95, chroma_strength 1.0, lum_keep 1.0`
(`configs/experimental/highlight_hue.yaml`). Held-out sets were not used for selection.

## Held-out results (run once; Highlight-safe = production QGH)

| Set | Method | PSNR | SSIM | LPIPS | NIQE | Highlight a*b* error vs GT |
|---|---|---|---|---|---|---|
| hc_eval, 60 synthetic bright-window photos | Highlight-safe | 14.96 | 0.6258 | 0.3912 | 4.07 | 66.96 |
| | **+ hue fix** | **17.41** | **0.6676** | **0.3295** | 4.04 | **4.37** |
| lolv2_syn, 100 (low-light pairs) | Highlight-safe | 16.51 | 0.7698 | 0.2515 | 4.15 | 43.62 |
| | **+ hue fix** | **17.43** | **0.7838** | **0.2289** | 4.15 | **9.44** |
| eval15, 15 (LOL-v1) | Highlight-safe | 25.12 | 0.8441 | 0.1314 | 3.48 | 53.05 |
| | + hue fix | 25.12 | 0.8441 | 0.1314 | 3.48 | 54.72 |
| syn_noisy / syn_clean, 100 + 100 | both | identical to Highlight-safe (0.000 dB change) | | | | |
| app examples (4) | both | identical | | | | |

Natural style (Standard) behaves the same way: hc_eval 14.49 -> 16.86 dB, eval15 25.15 -> 25.16 dB.
Latency of the fix alone: 14-23 ms at 600x400, 68-76 ms on the larger high-contrast images, 93 ms on the unseen photos.
Where no input pixel is near-clipped (syn_clean, syn_noisy, examples) the output is exactly unchanged.

![Astronaut and rocket examples](figures/highlight_hue/examples_astronaut_uneven_light.png)

## Caveats (please read before deciding)
- **hc_eval is synthetic and built the same way as the dev set** (bright regions kept bright, the rest darkened), so it
  mirrors the failure by construction; the gain there is expected and overstates real-world gain. The real-image
  evidence is: lolv2_syn (+0.9 dB), eval15 (neutral), and the supplied bright-window photo (orange -> white).
- The "Highlight a*b* error" on eval15 slightly rises (53.05 -> 54.72): eval15 has almost no clipped input, so the
  metric covers very few pixels there.
- On the 20 unseen real photos there is no ground truth. Their colour distance to the *input* in clipped areas drops
  (30.0 -> 10.8), which shows consistency with the input, not correctness.
- With QGH after the fix, the highlight roll-off darkens the restored region again (luminance gap 0.059 vs 0.014 for
  Natural + fix). Applying the fix after QGH, or reducing the roll-off inside the mask, was not tested.
- Real photos with clipped highlights *and* ground truth were not available; this should be checked before release.

## Recommendation
Promising and low-risk: it never changes images without clipped input, adds about 15-90 ms, and fixes the orange-window
failure. **Do not integrate until reviewed.** If approved, integrate as a post-step of Natural and Highlight-safe with
its own unit tests, a re-run of the eval15 regression check (expect 25.15/25.12 unchanged), and a redeploy.
