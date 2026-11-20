# Release hygiene (Phase H)

Audit date 2026-11-20, branch `chore/hygiene`. Not legal advice. Nothing here rewrites history or tags a release;
those need the owner's decision (section 4).

## 1. Continuous integration

`.github/workflows/ci.yml` runs `pytest` on every pull request and on pushes to `main` (Python 3.12, CPU PyTorch,
the pinned `requirements-ci.txt`). No checkpoint exists in CI (they are not redistributable, `weights/README.md`), so a
test that fails *only* because a checkpoint file is absent is reported as a **skip with the reason**
(`tests/conftest.py`). Any other failure stays a failure, including a corrupt or hash-mismatched checkpoint.

Result from a clean clone, no weights, no `data/`, fresh Python 3.12 venv: **223 passed, 43 skipped, 0 failed**.
With the weights present (the development machine): 266 passed. The 43 skips are the checkpoint-dependent tests;
`pytest -rs` lists each one.

`requirements-ci.txt` deliberately excludes `requirements-niqe.txt` (`pyiqa` is PolyForm-Noncommercial and heavy).

## 2. Secrets and personal-data scan

Scope: the working tree and all 20 commits on all branches.

| Check | Result |
|---|---|
| API keys / tokens / private keys (AWS, Google, GitHub, Hugging Face, OpenAI-style, PEM) in tree and history | **none** |
| `.env`, key, credential, `.pth`, `.onnx`, `.pdf`, `.pptx` files ever committed | **none** (history is 7.8 MB) |
| Local absolute paths with the user's name | **2 in the tree, fixed here**: `scripts/highlight_hue_study.py` (now `LUMEN_USER_PHOTO`, default `data/user_photo/`) and `IMPROVEMENT_PHASES.md`. Neither was in a file that ships in the image |
| GCP project id `lumen-514004`, project number in the Cloud Run URL | present in `eval_P1_T66.md` and `README.md`. These are identifiers, not credentials, and the URL is the public demo address. Access control is IAM, not secrecy of the id. Remove them only if you do not want the project named |
| Author identity in git history | all 20 commits carry the owner's personal Gmail address as author. Every clone and the public GitHub page expose it |
| CI deploy | keyless (Workload Identity Federation); no key is stored in the repository or its settings |

**Decision (owner, 2026-11-20): keep the author email and the GCP project id in the public repository; no history rewrite.** Context: the author email in history cannot be removed without rewriting history, which changes
every commit hash and breaks anything that cites one (the report cites commits). If the repository goes public and
that address should not be exposed, the options are (a) accept it, (b) enable GitHub's "keep my email private" for
future commits only, or (c) rewrite history with `git filter-repo` before publishing. I have not done (c).

## 3. Licence decision note

What the repository can legally say, from `weights/README.md` and `THIRD_PARTY_NOTICES.md`:

| Item | Status | Consequence |
|---|---|---|
| LUMEN's own code | no licence file; by default no reuse rights are granted | choose a licence before publishing |
| Retinexformer architecture port | MIT (c) 2023 Yuanhao Cai, notice retained | may be redistributed with that notice |
| Retinexformer checkpoint (`.pth`) and its ONNX export (`.onnx`) | **no explicit licence; trained on LOL, whose terms are unstated.** The ONNX file is derived from the checkpoint, so it has the same status | never committed or baked into the image; operators supply both at run time. This stays true for the ONNX export |
| Zero-DCE++ re-implementation (`lumen/deep/zero_dce.py`) | upstream is academic research only, CC BY-NC 4.0 | an MIT licence would need an exception for this file, or the file removed. The dropped hybrid is the only user |
| `pyiqa` (NIQE) | PolyForm-Noncommercial | optional, not in the image or CI |
| Example images | CC0 / public domain | may ship |
| Instrument Serif font | SIL OFL 1.1 | may ship with its licence |
| LOL / LOL-v2 data | terms unstated, not included | `data/splits/*.txt` and calibrated configs are derived from them; low risk (names and thresholds only) |

**Proposed wording** (pick one; I have not added any licence file):

* **A. Stay private / academic (safest, matches the facts today).** README footer:
  *"LUMEN is a research prototype for the P1 low-light restoration task. Source code is shared for evaluation and
  review only; no licence is granted for reuse. The Retinexformer architecture is used under its MIT licence; the
  pretrained weights are not redistributed, and their licence has not been confirmed by the authors."*
* **B. MIT for LUMEN's own code, with exceptions.** Add `LICENSE` (MIT) and: *"Excludes
  `lumen/deep/zero_dce.py` and `scripts/train_zero_dce.py` (derived from Zero-DCE++, academic research only) and any
  model weights, which are not part of this repository and carry their own terms."* Prerequisite: remove or replace
  the Zero-DCE++ files, or accept that the exception is documented.
* **C. Before any public hosting claim:** ask the Retinexformer authors to confirm that their MIT licence covers
  the released weights. Until then the public demo serves a model whose weight licence is unconfirmed.

Recommendation: **A now**, move to **B** only after the Zero-DCE++ files are removed and the authors reply.

## 4. Dependencies and clean-clone runs

* Versions are pinned in `requirements.txt` (runtime), `requirements-onnx.txt`, `requirements-ci.txt` and
  `requirements-dev.txt`; `requirements.txt` states the Python version they were tested with (3.12.15).
  `requirements-ci.txt` is what the CI-equivalent clean clone above ran with.
* **Local run from a clean clone** (`python app/app.py`, no weights): starts, serves the page (HTTP 200), offers
  only Quick and states why Natural and Highlight-safe are unavailable.
* **Docker run from a clean clone:** `docker build` of a fresh clone succeeded; with the weights mounted read-only at
  `/models` the container offered all three styles and `scripts/smoke_test_app.py` passed (all modes, all four example
  images, compare-all runs the network once) for both `LUMEN_ENGINE=torch` (default) and `LUMEN_ENGINE=onnx`.
* **Release tag:** not created. Tag only after the owner approves (`git tag -a vX.Y.Z`, then `git push origin vX.Y.Z`).

## 5. Still open for the owner

1. Choose licence wording (section 3).
2. ~~Author email and project id in a public repository~~: decided, keep both (section 2).
3. Confirm the repository visibility on GitHub; `gh` is not installed here, so I could not read it. The notices file
   says private.
4. Approve a release tag.
