# LUMEN UI redesign: design brief

**Status:** approved and implemented (branch `feat/ui-redesign`). The static prototypes used for approval were removed once the real UI replaced them; they remain in git history (tag `pre-cleanup-2026-11-20`). Decisions on the open questions (section 12): hint at σ ≥ 0.041; landing images regenerated
by `scripts/make_landing_assets.py`; the light map travels with every result (under 60 KB) and is drawn only when the drawer opens.

## 1. Concept: light returning to a dark photograph
Four rules hold the design together:
1. **Photos live on the darkroom.** Every photo sits on a near-black slab in both themes, because a photo is judged
   best against a neutral dark surround and because the product is about darkness. The page around it is night in the
   dark theme and warm paper in the light theme.
2. **Light is the only accent.** One warm lamplight colour, used only where light means something: the primary action,
   the active state, the part of a photo that was restored, the stage that has finished. Nothing else glows.
3. **Motion is exposure.** Things come up the way light does: fast open, long settle (`--ease-exposure`). Nothing
   bounces, spins or pulses. With reduced motion, everything is instant.
4. **Every number and sentence is a reported value.** Captions, chips, the pipeline strip and the report are built from
   fields the pipeline returned for this photo. No invented progress, no decorative statistics.

The **signature**: light in *space* on the landing (the visitor's light reveals the restored photo) and light in *time*
in the workspace (Develop). Both use the same real per-pixel change, so the first thing a visitor plays with is the
same idea they later use to inspect their own photo.

## 2. Information architecture
| Route | Purpose |
|---|---|
| `/` | Landing hero; becomes the workspace in place when a photo is chosen (no page load, as today) |
| `/how` | How it works: pipeline diagram, what each style does, what is and is not stored, limits |
| `/evidence` | Evaluation as designed charts (replaces the evaluation part of `/details`) |
| `/details` | Kept for technical detail: your last photo, research lab, system status |
| 404 | A real page in the same design (served for unknown non-API GET routes) |

Top bar on every page: wordmark (home), How it works, Evidence, theme toggle. Footer: privacy line, licences, Technical details.

## 3. Design tokens
**Colour** (all pairs used for text meet 4.5:1; checked by axe on every prototype screen):

| Token | Night (default) | Daylight paper | Use |
|---|---|---|---|
| `--bg` | `#0a0b0d` | `#f3f0e9` | page |
| `--surface` / `-2` / `-3` | `#131418` / `#1a1b20` / `#22232a` | `#fbfaf6` / `#eeeae1` / `#e5e0d5` | cards, hover, tracks |
| `--line` / `-2` | ink at 9 % / 17 % | ink at 11 % / 20 % | hairlines |
| `--ink` / `-2` / `-3` | `#ece8e0` / `#b9b3a8` / `#958f85` | `#1b1915` / `#4a453d` / `#686257` | text levels |
| `--light` | `#f4c77d` | `#e6a643` | the accent (fills, focus, active) |
| `--light-text` | `#f4c77d` | `#8a5300` | accent used as text |
| `--darkroom` | `#08090b` | `#111214` | photo surround, both themes |
| `--ok` / `--warn` | `#8fd3ad` / `#f2bb66` | `#1f7a52` / `#9a6200` | chip dots only, always with words |

**Type.** Display: *Instrument Serif* (SIL OFL), regular and italic, subset to Latin, **39 KB total** WOFF2, self-hosted,
`font-display: swap`. Used only for headlines, the wordmark and large values (the Develop amount). Everything else uses
the platform's system sans (Segoe UI Variable, SF, Roboto), so there is no other font download. Numbers use tabular figures.
Scale: display `clamp(46px, 6.1vw, 92px)/0.98`, h2 `clamp(26px, 2.4vw, 34px)`, lede 16-19 px, body 15 px, small 13-13.5 px.

**Space** on a 4 px base: 4, 8, 12, 16, 24, 32, 48, 64, 96; page gutter `clamp(16px, 4vw, 48px)`.
**Radii:** 8 (controls), 12 (buttons), 18 (cards), 26 (darkroom stage). **Elevation:** surfaces and hairlines; one shadow
for floating layers and one for photos ("a print on a table").
**Motion:** 120 ms micro, 220 ms UI, 420 ms reveal, 900 ms develop; `--ease-exposure: cubic-bezier(.16,1,.3,1)`,
`--ease-settle: cubic-bezier(.2,.7,.2,1)`. The landing light follows the pointer with a damped follow (no spring overshoot).

## 4. Component inventory
Top bar, theme toggle, buttons (lamplight / default / quiet), drop zone, example strip, darkroom stage, print, segmented
view switch (Before/after, Lens, Develop, All styles), toggle ("What changed"), zoom cluster, Develop control with live
histogram, lens, before/after slider (kept), pipeline strip (four stage nodes, queue state, cancel), style options,
diagnosis chips with gauge, report paragraph, "Look closer" drawer with maps, legend (kept), toasts, skeletons, empty, error,
offline and 404 states, evidence charts (forest plot, ablation bars, noise-band chart).

## 5. Signature interaction: Develop
For every pixel and channel, `now = (input + e) · gain^t − e` with `gain = (restored + e) / (input + e)`, `e = 1/255`,
gain clamped to [1/32, 64], and `t` from 0 to 100 %. This applies **the change LUMEN actually made, gradually, in
exposure space**: the light comes up where LUMEN added it and by how much, which a cross-fade or a slider cannot show.
Next to it, a live brightness histogram (original dashed, now filled, restored as a line) shows the distribution
spreading out of the dark end.
- Built on data the page already has (the processed original and the restored PNG); no new server work.
- WebGL2 fragment shader in the product; Canvas2D fallback on a down-scaled copy (the prototype uses Canvas2D and is
  smooth at 600 × 400). Reduced motion: no auto-play, scrub only. Keyboard: a real `<input type=range>` with a spoken value.
- Honest framing: the text says it applies "the light LUMEN added", not that it re-runs the model.

## 6. Landing scene (WebGL2)
Two textures per example (before and after, already generated). The fragment shader computes the same exposure blend
as Develop with `t` = a smooth falloff of distance to the light, so the reveal is a pool of real restoration, not a
mask. A faint spill lights the slab around the print. The light drifts slowly when idle and follows pointer or touch with
damping; four dots switch the example. The loop stops when the tab is hidden or the canvas is off-screen
(IntersectionObserver), and runs at most once per display frame. **Fallback** (no WebGL2, reduced motion, `Save-Data`):
a fixed composition with the light resting on the subject, the same caption, and no animation.
Art direction note: the bundled examples are 451-640 px wide, so the photo is shown as a *print* at no more than about
640 px rather than full-bleed. Upscaling them full-screen would look soft.

## 7. Workspace
- **Pipeline strip:** one node per real SSE event (Read, Measured the light, Restored, Checked bright areas), each
  with its measured time and a caption built from its fields. A running stage shows *elapsed* time, which is a fact, not
  a forecast. Queued shows the position. Cancel stays available. Screen readers hear each stage once (polite live region).
- **Chips** (every one maps to a field):

| Chip | Source field | Wording rule |
|---|---|---|
| Lighting | `analysed.category`, `analysed.mean_lum` | "Dark · 13 % brightness", with a gauge |
| Grain | `analysed.noise_sigma` | below 0.015 "Little grain"; 0.015-0.041 "Some grain"; 0.041 and above "Heavy grain" |
| Blown-out | `details.metrics.overexposed_percent` | "0.06 % blown-out after" |
| Time | `restored.ms` | "Restored in 5.9 s" |

- **Noisy-photo hint** under the style picker: see open question 1.
- **Lens:** a 192 px loupe showing the original inside the restored photo (or the reverse), following pointer or finger;
  keyboard moves it with arrow keys.
- **Look closer** drawer: the model's light-up map and the "What changed" masks, shown small and enlarged on demand. See open
  question 3.
- Plain words only: no model names or research terms (the existing jargon test stays and gains words if needed).

## 8. Evidence page
All numbers read from `evaluation.json`, extended by the existing generator (`scripts/hardening_study.py report`) with the
chart-ready fields it already computes. Charts in inline SVG: a forest plot of the differences with 95 % intervals; ablation
bars; results by input-noise band, which makes the high-noise limitation visible; then limitations, ethics, licences and
reproduce commands. A test will assert every plotted value equals the generated file.

## 9. Accessibility plan
WCAG 2.2 AA. axe 0 on every page and state, light and dark, phone and tablet (the prototypes already pass on all 12
screens). Full keyboard paths, including the slider, lens, Develop, zoom and drawer. Visible focus rings in lamplight.
Targets at least 44 px. `prefers-reduced-motion` and `forced-colors` honoured. The WebGL canvas is decorative and the
figure beside it carries the same message in text.

## 10. Performance budget
Landing LCP under 2.0 s on a mobile profile with 4x CPU slowdown; CLS under 0.05; JS under 120 KB gzipped (target: under
45 KB); fonts 39 KB; each landing image at most 150 KB (current pairs are 30-100 KB). Measured with Playwright traces.

## 11. Risks
| Risk | Mitigation |
|---|---|
| Low-resolution example photos | Shown as prints; no full-bleed upscaling |
| WebGL on weak phones | Lower internal resolution, damped frame rate, the static fallback |
| 3-6 s CPU wait | The pipeline strip makes real progress visible; nothing pretends |
| Payload size with maps | Small grayscale maps; budget at most +300 KB, measured in tests |
| Tests that assert today's markup | Updated only where the UI changes on purpose, each change listed; the CSP, external-resource, inline-style, identical-pixel, upload-deletion and plain-language tests are never weakened |

## 12. Open questions (need a decision before building)
1. **Threshold for the noisy-photo hint.** The brief asked for σ ≥ 0.025. The evidence (section 10.4 of the eval report)
   covers only σ ≤ 0.015 (benchmarks) and σ 0.041-0.053 (the noisy dev group); nothing in between was measured. Two of
   the four bundled examples sit in that gap (σ 0.028 and 0.037), so at 0.025 they would show a claim the data does not
   support. **Proposed:** show the hint only at σ ≥ 0.041.
2. **Leftover landing images.** An earlier session generated `app/web/static/img/landing/` and edited
   `app/examples/SOURCES.md`, citing `scripts/make_landing_assets.py`, which does not exist. The images check out (35-38 dB
   against a fresh Natural run, the difference being JPEG). **Proposed:** write that script, regenerate the images with it
   so the provenance is true, and keep the SOURCES.md wording.
3. **How maps reach the drawer.** The server is stateless, so "on demand" after the fact would mean restoring the photo a
   second time (another 3-6 s). **Proposed:** send the light-up map with every result as a small grayscale PNG (target
   under 60 KB) and decode it only when the drawer opens. The masks already arrive with the result.
