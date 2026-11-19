# LUMEN - Project phases (plan vs. outcome)

Record of how the P1 solution was built: what each phase planned, what was delivered, and what changed.
Numbers are LOL-v1 eval15 unless stated otherwise. Full results: [`eval_P1_T66.md`](eval_P1_T66.md).

## Rubric drivers (from the hackathon instruction sheet)

| Criterion | What earns the top band | Where it is addressed |
|---|---|---|
| Evaluation metrics | Strong PSNR / SSIM / LPIPS / NIQE relative to other teams | Phases 2, 5 |
| Implementation | Works on unseen inputs, proper train/val split, modular code, bad input handled | Phases 1-6 |
| UI/UX + deployment | Public URL, < ~10 s, loading and error states, one-command run, no secrets | Phases 6-7 (hosting still open) |
| Documentation | Diagram matches code, complete design doc, reproducible README, limitations and ethics | Phase 8 |
| Demo and viva | Live demo on an unseen image; every member explains their part | Open |

## Phase status

| # | Phase | Status | Outcome |
|---|---|---|---|
| 0 | Setup: P1 extract, scaffold, environment | Done | Python 3.12 package `lumen/`, first (Gradio) app, tests, Dockerfile |
| 1 | Data, baseline, classical tuning | Done | LOL + LOL-v2 checked (`check_data.py`). Optuna (150 trials) on `our485` only: classical 15.16 → **17.16 dB** (val split 15.18 → 16.92) |
| 2 | Strong deep model | Done | Retinexformer ported (bit-exact with upstream, strict checkpoint load): **25.15 / 0.8441** (paper 25.16 / 0.845). Fine-tuning improved its validation score but lowered eval15 to 24.44, so it was **rejected**. Test-set leakage between LOL-v1 and LOL-v2 found and excluded |
| 3 | Hybrid model selection | Done | Fusion hybrid can use Zero-DCE++ or Retinexformer; the evaluator labels each experiment with the model that actually ran |
| 4 | Ablation of the fusion hybrid | Done | Best fusion variant 21.12 dB: it never approaches standalone Retinexformer (negative result, documented) |
| 5 | Quality-Guided Hybrid (QGH) | Done | Retinexformer + conditional refinements, thresholds calibrated on a 269-pair dev split, eval15 run once: 25.12 dB, equal SSIM/LPIPS/NIQE, over-exposure **0.82 % → 0.03 %** |
| 6 | Application integration | Done | Standard / Quality Guided / Classical modes, research section, strict config + checkpoint handling, image validation, QGH diagnostics, Compare All with one model run |
| 7 | Deployment verification | Done (local) | Production image built (2.32 GB with Gradio, 2.02 GB after the redesign in 7b) and smoke-tested; checkpoint mounted externally with SHA-256 check; CPU threads fixed (3.3x speed-up); 768 px limit; measured latency and RAM. **At that point no hosting provider or GPU deployment was verified (hosting: see Phase 9; GPU deployment is still untested)** |
| 7b | UI/UX redesign | Done (local) | Gradio replaced by a static HTML/CSS/JS frontend on FastAPI. Simple workspace (Natural / Highlight-safe / Quick, before/after slider, download); technical details, evaluation and research lab moved to *How it works*. Service, models, QGH and thresholds unchanged; API output byte-identical to direct service calls (test). Verified in headless Chrome at desktop, tablet and phone sizes |
| 8 | Documentation and repository preparation | Done | Evaluation report, design doc, deployment guide, licensing notes; committed to the private GitHub repository |
| 9 | Hosting and demo | **Hosting done; demo preparation open** | Deployed on Google Cloud Run (`docs/DEPLOY_CLOUD_RUN.md`: asia-south1, 2 vCPU / 4 GiB, checkpoint from a private bucket); the live smoke test, three styles, Compare All and PNG downloads passed (`eval_P1_T66.md` section 9.5). Hugging Face Spaces (`docs/DEPLOY_HF_SPACES.md`) is prepared and smoke-tested locally, **not deployed**. Still open: a public-URL check on a real phone, demo script and viva preparation (`IMPROVEMENT_PHASES.md` Phase I). Residual-noise study (experimental, not integrated): `docs/NOISE_STUDY.md` |
| 10 | Highlight colour fix (after the Phase 5 freeze) | Done, deployed | Natural and Highlight-safe keep the colour of near-clipped input regions, so bright windows no longer turn orange. eval15 essentially unchanged (1 of 15 outputs differs: Natural 25.15 → 25.16 dB, Highlight-safe 25.12 → 25.12 dB). Merged to `main`, live on Cloud Run. Validated on synthetic and LOL-v2 data only, no real clipped-highlight pairs with ground truth yet (`docs/HIGHLIGHT_HUE_STUDY.md`, `eval_P1_T66.md` section 9) |
| 11 | Streamed stages and plain-English report (`IMPROVEMENT_PHASES.md` Phase B) | Done locally, not deployed | New `POST /api/restore/stream` (Server-Sent Events) reports the real stages with timings; the workspace shows them live, then a short summary of what was done to that photo. Final image byte-identical to `/api/restore` (12 of 12 example/style pairs on the real model); the old endpoint is the fallback. 23 new tests (209 total), plus a real-browser check at desktop and phone width |

## Changes from the original plan
- **Deep model:** Retinexformer replaced Zero-DCE++ as the primary model (zero-reference training capped PSNR at about 15 dB).
- **Hybrid design:** the planned "hybrid v2" fusion was replaced by QGH, because ablations showed fusion degrades
  Retinexformer's output. Learned fusion was not pursued.
- **UI:** the Gradio app was first hardened (single service layer, clear errors, diagnostics), then replaced in
  the redesign by a FastAPI backend with a dependency-free HTML/CSS/JS frontend. Next.js was not needed: there is
  no build step and no external requests.
- **Speed:** ONNX export was **not done**. CPU latency was instead brought within budget by fixing thread
  oversubscription and capping processing at 768 px in the container.
- **Weights:** the checkpoint is **not redistributed** (licence uncertainty). It is mounted at run time and verified by SHA-256.

## Open items
1. Hosting is live; still open: check the public URL on a real phone and laptop and record the first-request latency.
2. Visual UI check in a real browser: desktop and mobile layout, slider, error toasts, loading states.
3. Licensing questions before any public release (see `THIRD_PARTY_NOTICES.md`).
4. Demo and viva preparation: a 3-minute script, a member-to-module ownership map, and failure cases to show
   (`eval_P1_T66.md` §7).
