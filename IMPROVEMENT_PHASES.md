# LUMEN - Improvement phases (v2)

Goal: take the existing tool to the next level **without lowering anything that already works**. Every phase below
has a ready-to-paste prompt, an acceptance gate and a rollback rule. Work the phases in order; each one ends with a
green test run and your approval before the next starts.

## Decision record

| Idea | Decision | Reason |
|---|---|---|
| Hybrid of LUMEN with Zero-DCE++ | **Dropped** | LUMEN already measured it: Zero-DCE++ 15.14 dB, fusion hybrid with Zero-DCE++ 16.80 dB, against 25.15 dB for Retinexformer. Zero-DCE++ is also below the tuned classical pipeline (17.16 dB) and its code is academic-use only. |
| Another fusion or learned-fusion hybrid | Dropped | Every fusion variant stayed at least 4 dB below standalone Retinexformer (`eval_P1_T66.md` section 5). |
| Fine-tuning Retinexformer again | Dropped | Fine-tuned run raised validation PSNR but lowered eval15 to 24.44 dB (rejected earlier). |

The Zero-DCE++ code and the fusion hybrid stay in the repo as the documented research baselines. Nothing new is
built on them and nothing is added to Natural, Highlight-safe or Quick.

## Rules for every prompt (paste this block first in each session)

```
Project: LUMEN (C:\Users\Bhyresh BS\Documents\dev\LUMEN), team T66, P1 low-light restoration.
Rules for this session:
1. Read docs/DESIGN.md, docs/ARCHITECTURE.md and eval_P1_T66.md before changing anything.
2. LOL-v1 eval15 is a frozen test set: never tune, calibrate or select on it. Headline numbers stay as in
   eval_P1_T66.md; follow the evaluation protocol (LOL-v1 eval15 headline, GT-mean for comparison only,
   LOL-v2-real Test is contaminated for the LOL-v1 checkpoint).
3. Default outputs of Natural, Highlight-safe and Quick must not change unless the phase says so. If a change
   could alter pixels, prove it: bit-identical outputs, or eval15 PSNR/SSIM/LPIPS/NIQE within 0.01 dB / 0.001 /
   0.001 / 0.02.
4. Do not add Zero-DCE++ or fusion features to the app styles. Do not add dependencies without telling me why.
5. Keep the frontend dependency-free (plain HTML/CSS/JS, no build step, no external requests).
6. Run `pytest -q` before and after. Report failures with output; never hide skipped steps.
7. No commits, pushes or deployments unless I ask. Work on a new branch named in the phase.
8. If a phase gate fails, stop, report the numbers, and revert to the previous behaviour.
```

---

## Phase A - Make the documentation tell one true story
**Branch:** `docs/consistency` | **Risk:** none (docs only)

Findings from the audit: README says hosting is "prepared, not yet deployed" and DESIGN section 10 says "no hosting
provider", but `eval_P1_T66.md` section 9 says it is live on Cloud Run. Image size is 2.02 GB in one place and
2.32 GB in another. Test count is 167 vs 186. Highlight-hue study and release notes still say "not merged, not
deployed". (GPU model is consistent, RTX 3050 everywhere: it was wrongly listed here at first.)

```
Audit every markdown file (README.md, phases.md, docs/*.md, eval_P1_T66.md, weights/README.md, app/web
details page text and app/web/static/data/evaluation.json) for statements that contradict each other or the code.
Check at minimum: deployment status (Cloud Run is live per eval_P1_T66.md section 9), Docker image size, test count
(run pytest --collect-only to get the real number), GPU model, latency figures, which styles use the highlight
hue fix, and latest commit hash. Produce a table of every contradiction (file, line, claim, truth, source of
truth), then fix them so each fact has one authoritative home and other files link to it. Do not change any
measured number unless you re-measured it; if a number cannot be verified, mark it "not re-verified" instead of
guessing. Update phases.md Phase 9 from "prepared" to the real state. Keep eval_P1_T66.md line 1 (USNs) unchanged.
```

**Gate:** a grep for `not yet deployed|no hosting|167 tests|2.32 GB` returns only intentional historical mentions.

---

## Phase B - Show the work: streamed stages and a plain-English report
**Branch:** `feat/streamed-stages` | **Risk:** low (new endpoint; old one untouched)

Why: users wait about 6 s on CPU with no feedback beyond a spinner, and the workspace deliberately hides what was
done. A short, honest report of what the tool did is the biggest visible upgrade and costs no model quality.

```
Add a streaming restore endpoint without changing /api/restore. Implement POST /api/restore/stream returning
Server-Sent Events with these stages in order: "decoded" (size, orientation applied), "analysed" (image statistics
that already exist: category, mean brightness), "restored" (model finished, with elapsed ms), "refined"
(only for Highlight-safe / Natural: which refinements ran, from QGH diagnostics and highlight_hue info; say
"nothing needed fixing" on the identity path), "done" (same payload as /api/restore). Reuse the existing
exclusive-run lock in app/server.py so concurrency stays 1; if another request is running, emit a "queued" event
with the position. In app/web/static/js/workspace.js render a stage list with timings and a one-paragraph
plain-English summary built from the real diagnostics (no invented claims; every sentence must map to a field).
Fallback: if EventSource/fetch streaming fails, call /api/restore as today. Add tests in tests/test_web.py: event
order, error event on a bad file, identical final image bytes vs /api/restore, queued event under contention.
```

**Gate:** final PNG bytes identical to `/api/restore`; all existing web tests still pass; works at phone width.

---

## Phase C - Result inspection: maps, zoom, compare
**Branch:** `feat/inspection-ui` | **Risk:** low (frontend only plus read-only data already produced)

```
Improve the workspace result view using data the pipeline already returns (QGH masks, model map, intermediates):
1. A "What changed" toggle that overlays the QGH refinement masks (noise, highlight) and the highlight-hue mask on
   the result, with a legend and a coverage percentage per mask. Hide the toggle if no refinement ran.
2. Zoom and pan on the before/after slider (wheel, pinch, double-click to reset) without breaking the keyboard
   slider behaviour (ARIA slider, arrow keys).
3. A three-up compare (Natural / Highlight-safe / Quick) from the existing /api/compare, with the same zoom synced
   across panels.
4. Download options: PNG (existing), and a side-by-side PNG (before | after).
Keep model names and metrics off the workspace page; they stay on /details. Verify at desktop, tablet and phone
sizes in headless Chrome, check colour contrast in light and dark themes, and add tests for any new API fields.
```

**Gate:** no change to restored pixels; Lighthouse accessibility score does not drop; no new network requests.

---

## Phase D - Robust input handling and large photos
**Branch:** `feat/robust-input` | **Risk:** medium (touches the processing size cap)

Why: the container caps processing at 768 px, so a 12 MP phone photo comes back small. Users will notice.

```
Investigate and, if it passes the gate, improve large-image handling. Step 1 (measure only): on the Cloud Run
configuration (2 vCPU, 4 GiB) measure time and peak RAM for 768, 1024, 1536 px long side using the existing
feathered tiling in lumen/deep/infer.py. Step 2: propose the highest cap that keeps Standard under 10 s median and
peak memory under 3 GiB, and implement it behind an env var (default stays 768 until I approve). Step 3: return
the output at the original resolution by restoring a downscaled copy and transferring the correction back (for
example, apply the model's per-pixel gain to the full-resolution input), and compare against the plain-upscale
result on LOL-v2 pairs resized up, reporting PSNR/SSIM. Also verify EXIF orientation, 16-bit PNG, CMYK/palette
images, animated GIF/WebP (first frame), tiny images (< 32 px), and 0-byte or corrupt uploads each give a clear
user-facing error or a correct result. Add tests for each.
```

**Gate:** same-resolution outputs are bit-identical at the default cap; the new path is only enabled if it keeps the
10 s budget. Otherwise leave the cap and document the limit.

---

## Phase E - Speed
**Branch:** `perf/cpu-latency` | **Risk:** medium (could change pixels)

Current: Standard 6.3 s median on 2 CPUs, 3.8 s on 4. ONNX export was never done.

```
Benchmark CPU speed-ups for the Retinexformer forward pass at 768x512 on 2 and 4 threads, one at a time, each
against the current PyTorch fp32 baseline using scripts/benchmark_container.py methodology: (a) torch.inference_mode
plus channels_last, (b) torch.compile if it helps on CPU, (c) ONNX Runtime export with strict parity check,
(d) bf16/int8 only if parity holds. For each: median/p95 latency, peak RAM, and the output difference versus the
baseline on all 15 eval15 images (max abs error, PSNR between outputs, then eval15 PSNR/SSIM/LPIPS). Accept an
option only if it is at least 1.5x faster and eval15 metrics are within the rule-3 tolerance. Implement the best
accepted option behind LUMEN_ENGINE env var with the current engine as default, keep the checkpoint SHA-256
verification, and add a parity test. Also measure Cloud Run cold start and recommend min-instances based on cost.
```

**Gate:** 1.5x faster and metrics within tolerance, or no change shipped. Report even if the answer is "none qualify".

---

## Phase F - Stronger, more honest evaluation
**Branch:** `eval/hardening` | **Risk:** none to the tool (evaluation only)

Why: 15 images and differences of 0.03 dB are the weakest part of the submission.

```
Do not change any model or config. Extend evaluation only:
1. Paired bootstrap (10,000 resamples) confidence intervals for the per-image differences Retinexformer vs QGH vs
   Classical on eval15 for PSNR, SSIM, LPIPS; report CI and whether it excludes zero. State plainly that 15 images
   cannot support fine claims.
2. LOL-v2-real Test using ONLY the LOL-v2-real Retinexformer checkpoint (per the evaluation protocol), with the
   same metrics and bootstrap CIs.
3. No-reference scoring (NIQE, plus any metric already in requirements) on the 20 unseen phone photos and on any
   DICM/LIME images I place in data/, for Natural, Highlight-safe and Quick, clearly labelled "no ground truth".
4. A stratified view of eval15 and the dev split by input brightness and noise level (use the existing image
   statistics) to show where the model is weakest.
Write results to a new section in eval_P1_T66.md and results/ files, marked "post-freeze additions", without
touching the Phase 5 headline table. Update evaluation.json for the details page.
```

**Optional data task for me (not Claude):** shoot 30-50 real scenes on a tripod with a phone, a very dark exposure
and a properly exposed one, include some with bright windows or lamps. That gives real ground truth for the highlight
colour fix, which is the main unverified claim today. Then add:

```
Add my paired phone photos in data/real_pairs/ as a new held-out set (never used for tuning). Evaluate Natural,
Highlight-safe and Quick with the highlight hue fix off and on, report PSNR/SSIM/LPIPS and highlight a*b* error, and
state honestly if the fix does not help on real pairs.
```

**Gate:** headline table unchanged; every new number has a CI or an explicit "no ground truth" label.

---

## Phase G - Quality gate for generalisation (investigation, expected-small)
**Branch:** `research/checkpoint-routing` | **Risk:** high of no result; ship only on a clear win

Known weak spot: LOL-v1 checkpoint drops to SSIM 0.33 on noisier LOL-v2-real photos. A routing idea is to pick the
LOL-v1 or LOL-v2-real checkpoint per image. Caution: the dev split comes from LOL-v2-real Train, which the
LOL-v2-real checkpoint trained on, so calibrating there would be leakage.

```
Investigate only, do not ship. First check whether any held-out data exists that neither Retinexformer checkpoint
trained on (my phone pairs in data/real_pairs/ from Phase F, if present). If there is none with ground truth,
stop and report "no valid calibration data". If there is, compare: LOL-v1 checkpoint, LOL-v2-real checkpoint, and a
per-image router using existing image statistics (noise sigma, mean brightness), with the router thresholds fitted
on half the pairs and scored on the other half, with bootstrap CIs. Also record the licence status of the extra
checkpoint (see weights/README.md) and whether serving two checkpoints is acceptable. Recommend ship / do not ship.
```

**Gate:** ship only if the router beats the single checkpoint with a CI excluding zero on held-out real pairs and the
checkpoint licence is acceptable. Otherwise record it as a documented non-result.

---

## Phase H - Repository and release hygiene
**Branch:** `chore/hygiene` | **Risk:** low

```
1. Add a CI workflow that runs pytest (without the checkpoint) on pull requests and on main.
2. Scan the repo and git history for secrets or personal data (keys, project IDs that should not be public, local
   absolute paths, emails) and report; do not rewrite history without asking.
3. Add a LICENSE decision note: list what the repo can legally say given Retinexformer weights and LOL data terms
   (weights/README.md, THIRD_PARTY_NOTICES.md); propose wording; I decide.
4. Pin and document the exact dependency versions that the tests passed with; confirm the one-command local run
   and the Docker run from a clean clone in a temp directory.
5. Tag the release only after I approve.
```

**Gate:** clean-clone run succeeds; CI is green.

---

## Phase I - Demo and viva readiness
**Branch:** `docs/demo` | **Risk:** none

```
Create docs/DEMO.md: a 5-minute demo script on the live Cloud Run URL using 3 unseen images (one easy, one with a
bright window to show the highlight fix, one failure case such as eval15 image 1 or 493), with expected outputs and
timings. Include a viva Q&A for each team member's part covering: why Retinexformer over Zero-DCE++ (cite the
measured numbers), how test leakage was found and handled, why the fusion hybrid was kept as a negative result, what
QGH does and does not improve, what the highlight fix trade-off is (over-exposure +0.24 points on unseen photos),
the limits of 15 test images, and licensing. Also a fallback plan if Cloud Run cold-starts or the network fails
(local container command). Cross-check every number against eval_P1_T66.md and cite the line.
```

---

## Suggested order and effort

| Order | Phase | Value | Effort | Can break outputs? |
|---|---|---|---|---|
| 1 | A docs consistency | High (credibility) | Small | No |
| 2 | B streamed stages + report | High (UX, explainability) | Medium | No |
| 3 | C inspection UI | Medium-high | Medium | No |
| 4 | F evaluation hardening | High (rigour) | Medium | No |
| 5 | D large photos | Medium-high | Medium | Only behind env var |
| 6 | E speed | Medium | Medium | Only if gate passes |
| 7 | H hygiene | Medium | Small | No |
| 8 | I demo readiness | High at the end | Small | No |
| 9 | G routing | Low (likely no-go) | Large | Only if gate passes |
