---
title: LUMEN Low-Light Enhancement
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Restore low-light photos (Retinexformer + QGH)
---

# LUMEN - low-light image restoration

Upload a dark photo and LUMEN restores it in one of three styles:

- **Natural**: Retinexformer (ICCV 2023), a supervised transformer model.
- **Highlight-safe**: Retinexformer plus a calibrated quality-guided refinement that tames blown highlights.
- **Quick**: a classical, CPU-only Retinex/CLAHE pipeline.

*How it works* (`/details`) explains each step and shows the evaluation results.

**Privacy:** uploads are processed in memory and in a temporary file that is deleted when the request
finishes. Images are never stored or logged.

**Hardware:** this Space runs on the free 2 vCPU CPU tier. A restoration takes about 6-10 s, and requests are
processed one at a time, so you may wait briefly if others are using it. After a period without visitors the
Space sleeps; the first visit then takes a minute or two to wake it.

**Credits and licences:** see `THIRD_PARTY_NOTICES.md`. The Retinexformer checkpoint is not part of this
repository; the Space fetches it at start-up and verifies its SHA-256.
