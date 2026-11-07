# Deploying LUMEN on Google Cloud Run

**Status (2026-11-07): prepared, not deployed.** Here is what has and has not been verified:
- The production image was built and smoke-tested locally the way Cloud Run runs it: `PORT=8080`, 2 CPUs,
  4 GiB, checkpoint mounted read-only at `/models`. See *Verification* at the end.
- **No Google Cloud project has been configured, and nothing has been deployed.** The steps below need your
  Google login, a project and billing.

## Architecture

```
Browser ──HTTPS──> Cloud Run service "lumen"   (asia-south1, gen2, 2 vCPU / 4 GiB, 0-2 instances)
                   └─ container: repository Dockerfile (python:3.12-slim, CPU PyTorch 2.14.1, FastAPI/Uvicorn)
                      app/app.py -> binds 0.0.0.0:$PORT (Cloud Run sets PORT=8080)
                      /models  <── Cloud Storage volume (read-only) ── private bucket <project>-lumen-weights
                                                                         └─ retinexformer_LOL_v1.pth (SHA-256 checked at start-up)
Images: Artifact Registry  asia-south1-docker.pkg.dev/<project>/lumen/lumen:<git sha>
Builds: Cloud Build (manual path) or GitHub Actions (keyless, Workload Identity Federation)
```

The container is the **unchanged production `Dockerfile`**: no new build. The checkpoint is supplied through a
Cloud Storage volume, exactly as the image expects a `/models` mount. That keeps it out of the image and the
repository; see *Model weights* below.

## Resource settings and why

| Setting | Value | Basis |
|---|---|---|
| Region | `asia-south1` (Mumbai) | requested; close to the users |
| CPU / memory | 2 vCPU / **4 GiB** | Measured peak at 768 px with 2 CPUs: 1.7-1.8 GB per style, 2.56 GB once in Compare All (`docs/DEPLOYMENT.md`). 4 GiB leaves at least 1.4 GiB headroom, and `/tmp` uploads (at most 20 MB) live in memory on Cloud Run |
| Concurrency | 1 | requested. The app already runs one restoration at a time internally. See the note below |
| Instances | min 0, max 2 | scale to zero = no idle cost; a cold start costs roughly 10-30 s on the first visit |
| Timeout | 300 s | a queued restoration takes about 6 s on 2 vCPU; 300 s is ample |
| Startup CPU boost | on | faster cold start (PyTorch import plus model load) |
| Execution environment | gen2 | required for Cloud Storage volume mounts |
| `LUMEN_TORCH_THREADS` | 2 | matches the 2 vCPUs (oversubscription made CPU inference 3.3x slower in tests) |
| `LUMEN_MAX_SIDE` | 768 | the measured latency budget (about 6 s at 2 vCPU; 1024 px takes about 9.7 s) |
| Architecture | linux/amd64 | Cloud Build and GitHub's ubuntu runners build x86_64 images |
| GPU | none | CPU only |

**Concurrency note:** with concurrency 1, every request needs its own instance slot, including page assets
and `/api/status`. While one visitor's restoration (about 6 s) is running, a second visitor is routed to the
second instance (a cold start) or waits in Cloud Run's queue. That is fine for a demo. If page loads feel slow
during use, redeploy with `-Concurrency 4`. Memory stays bounded either way, because the app serialises
restorations (three simultaneous requests peaked at 1.85 GB in tests).

## Prerequisites (your side)

1. **A Google account and a Google Cloud project** with **billing enabled**. Cloud Run needs billing even
   when usage stays inside the free tier.
2. **The Google Cloud CLI.** It is installed on this machine (SDK 592.0.0) but *not on PATH*, and no account
   is logged in.
3. **The checkpoint** at `weights\retinexformer_LOL_v1.pth`. It is present on this machine. Elsewhere:
   `pip install gdown==6.5.1` then `python scripts\download_weights.py --runtime`.
4. **An Owner or Editor role on the project**, for the one-time setup only.

## Deploy from a fresh Windows PowerShell terminal

```powershell
# 0. gcloud on PATH (this machine: installed per-user but not on PATH). New terminals: add it permanently via
#    System Properties > Environment Variables, or run this line in each session.
$env:Path += ";$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin"
gcloud --version

# 1. Log in (opens a browser)
gcloud auth login

# 2. Pick the project (list yours, or create one - project IDs are globally unique)
gcloud projects list
# gcloud projects create <your-project-id> --name "LUMEN"        # only if you need a new one
gcloud config set project <your-project-id>

# 3. Billing must be linked (console: https://console.cloud.google.com/billing/linkedaccount )
gcloud billing accounts list
# gcloud billing projects link <your-project-id> --billing-account <ACCOUNT_ID>

# 4. Get the code (skip if you already have it)
git clone https://github.com/bhyreshbs/LUMEN-LUMinance-Enhancement-Network.git
cd LUMEN-LUMinance-Enhancement-Network
#    ...and put the checkpoint at weights\retinexformer_LOL_v1.pth (prerequisite 3)

# 5. One-time setup: APIs, Artifact Registry, private weights bucket, least-privilege service accounts,
#    and keyless GitHub federation. Safe to re-run.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\deploy\cloudrun\setup.ps1 -ProjectId <your-project-id>

# 6. Build with Cloud Build, then deploy. -Public = open URL for the demo; omit it to require sign-in.
.\deploy\cloudrun\deploy.ps1 -ProjectId <your-project-id> -Public

# 7. Verify the live service end to end (all 3 styles x 4 examples, Compare All, invalid uploads)
python scripts\smoke_test_app.py --url https://<service-url> --expect-modes standard quality_guided classical
```

`deploy.ps1` prints the URL and whether each style is available (via `/api/status`: Google reserves `/healthz` on `*.run.app` and answers it with its own 404). **Natural and Highlight-safe must
show `available`.** If they don't, the checkpoint mount failed: see *Troubleshooting*.

Re-deploy after code changes: run step 6 again. Each deploy is tagged with the current git commit.

## IAM (least privilege, created by `setup.ps1`)

| Identity | Granted | Why |
|---|---|---|
| `lumen-run@` (runtime) | `roles/storage.objectViewer` on **the weights bucket only** | read the checkpoint; nothing else |
| `lumen-deployer@` (GitHub Actions) | `roles/artifactregistry.writer` on **the `lumen` repository only** | push images |
| | `roles/run.developer` (project) | deploy new revisions; **cannot** change who may invoke the service |
| | `roles/iam.serviceAccountUser` on **`lumen-run@` only** | deploy revisions that run as `lumen-run` |
| GitHub OIDC (WIF pool `github`, provider `github-oidc`) | `roles/iam.workloadIdentityUser` on `lumen-deployer@`, restricted by an attribute condition to `bhyreshbs/LUMEN-LUMinance-Enhancement-Network` | keyless login from that repository only |
| You (manual path) | Owner/Editor for setup; deploying needs `run.admin` for `-Public` | one-time public access, Cloud Build submission |

No service-account keys are created, and no credentials are stored in the repository or in GitHub.

## GitHub Actions (optional, after the first manual deploy)

`.github/workflows/deploy-cloudrun.yml` builds the image on GitHub's runner, pushes it to Artifact Registry and
deploys it with the same settings. It then reports the URL, `/healthz` and style availability in the run summary.

1. Run `setup.ps1` (without `-SkipGitHub`). It prints six values.
2. In GitHub, open **Settings -> Secrets and variables -> Actions -> Variables** and add `GCP_PROJECT_ID`,
   `GCP_REGION`, `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`, `GCP_RUN_SA` and `GCP_WEIGHTS_BUCKET`. They are
   identifiers, not secrets.
3. Do the first deploy manually with `-Public`; the deployer role cannot grant public access.
4. Release with `git push origin main:deploy`, or start **Actions -> Deploy to Cloud Run -> Run workflow**.
   Pushes to `main` do not deploy. Until `GCP_PROJECT_ID` is set, the workflow is skipped.

## Environment variables (set by `deploy.ps1` / the workflow)

| Variable | Value on Cloud Run | Meaning |
|---|---|---|
| `PORT` | 8080 (set by Cloud Run) | listen port; the app falls back to 7860 locally |
| `LUMEN_HOST` | 0.0.0.0 (image default) | bind address |
| `LUMEN_RETINEXFORMER_WEIGHTS` | `/models/retinexformer_LOL_v1.pth` | checkpoint path (bucket volume) |
| `LUMEN_TORCH_THREADS` | 2 | CPU threads |
| `LUMEN_MAX_SIDE` | 768 | processing size |
| `LUMEN_MAX_FILE_MB`, `LUMEN_MAX_PIXELS`, `LUMEN_MAX_QUEUE` | image defaults: 20, 40e6, 16 | upload and queue limits |

## Model weights: licensing limitation

The Retinexformer **code** is MIT-licensed. Its **pretrained checkpoint** carries no explicit licence and was
trained on LOL, whose terms are unstated. So redistribution is not established. The MIT code licence does not
automatically cover the weights.

Therefore the checkpoint is:
- **not** in the Git repository (`weights/*` is ignored);
- **not** in any container image (`.dockerignore`, `.gcloudignore`);
- held only in a **private** bucket that just the runtime service account can read.

A public Cloud Run URL still lets anyone *use* the model's outputs. Confirm that this is acceptable (hackathon
organisers, `THIRD_PARTY_NOTICES.md`) before deploying with `-Public`.

## Costs (approximate; verify at https://cloud.google.com/run/pricing and the pricing calculator)

| Item | Expectation |
|---|---|
| Cloud Run | Billed only while handling requests (min instances 0). One restoration is about 6 s x 2 vCPU = 12 vCPU-s plus 24 GiB-s. The monthly free tier (at the time of writing 180,000 vCPU-s, 360,000 GiB-s and 2 M requests per billing account) covers on the order of **10,000+ restorations per month**. Beyond that, roughly a few hundredths of a US cent per restoration |
| Artifact Registry | about 2 GB per image version (shared layers are stored once); first 0.5 GB/month free, then about US$0.10 per GB-month. Delete old tags to stay small |
| Cloud Build | about 5-10 min per build on the default machine; a monthly free allowance of build-minutes applies |
| Cloud Storage | a 6.5 MB object: effectively free |
| Network egress | each result is a PNG of a few hundred kB; negligible for demo traffic |

Set a **budget alert** (Billing -> Budgets & alerts) before making the URL public. `max-instances 2` caps the
worst-case spend rate.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `gcloud: command not found` | Run step 0 (PATH) in this terminal |
| `Billing is not enabled` (setup.ps1) | Link a billing account (step 3) |
| Cloud Build: `PERMISSION_DENIED` writing to Artifact Registry or logs | On newer projects builds run as the Compute Engine default SA. Grant it `roles/artifactregistry.writer` on the `lumen` repo and `roles/logging.logWriter` on the project (`gcloud projects describe <id> --format="value(projectNumber)"` gives `<number>-compute@developer.gserviceaccount.com`) |
| Natural/Highlight-safe `UNAVAILABLE`: `No retinexformer checkpoint found` | The volume is not mounted, or the object name is wrong: `gcloud storage ls gs://<project>-lumen-weights/` must list `retinexformer_LOL_v1.pth` |
| `SHA-256 mismatch` in logs | The bucket holds a different file. Re-run `setup.ps1`, which checks the hash before uploading |
| Revision fails: `Memory limit exceeded` | Raise `--memory` to 6Gi, or lower `LUMEN_MAX_SIDE` |
| HTTP 429 / long waits | All instances busy: raise `-MaxInstances` (a cost trade-off) or `-Concurrency` |
| First request slow (10-30 s) | A cold start at min-instances 0. Open the URL a minute before a demo, or set `--min-instances 1` (billed while idle) |
| Logs | `gcloud run services logs read lumen --region asia-south1 --limit 100` |

## Local development and Docker (unchanged)

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app\app.py                                   # http://localhost:7860
python -m pytest -q                                 # needs requirements-dev.txt
docker build -t lumen .
docker run --rm -p 8080:8080 -e PORT=8080 --cpus 2 --memory 4g -v "${PWD}\weights:/models:ro" lumen
```

## Verification (local, 2026-11-07)

Run on Docker Desktop 29.9.2 (WSL 2, x86_64) with the production `Dockerfile`, configured like Cloud Run:
`docker run -e PORT=8080 -e LUMEN_TORCH_THREADS=2 -e LUMEN_MAX_SIDE=768 --cpus 2 --memory 4g -v weights:/models:ro`.

| Check | Result |
|---|---|
| `docker build` | OK. Image `lumen:release` is 2.02 GB, linux/amd64, runs as non-root `appuser`, CMD `python app/app.py`. Build-time checks passed (QGH config valid, imports resolve). No `.pth`/`.pt`/`.env` file in the image, and no dev-only packages (pytest, lpips, pyiqa, gdown, optuna, torchvision) |
| Startup | First `/healthz` 200 after **5.1 s**. Logs: `CPU threads: 2 (cgroup quota 2)`, `retinexformer_LOL_v1.pth SHA-256 verified`, all 3 styles available, listening on **port 8080** from `$PORT` |
| `scripts/smoke_test_app.py --url http://127.0.0.1:8080` | **ALL PASSED**: both pages and every asset; `/healthz`; Natural (deep), Highlight-safe (QGH hybrid) and Quick (classical) on all 4 examples (Natural 1.8-5.2 s, Quick 0.05-0.3 s server time); Compare All with a single network run; NIQE reported as not installed; invalid uploads give 4xx with LUMEN's messages |
| Download payload | `POST /api/restore` returned a PNG of 600x400 (546 kB) that decodes. That data URL is exactly what the Download button saves |
| Invalid uploads | not an image -> 422 "not a recognised image ..."; empty file -> 422 "The uploaded file is empty." |
| Research fusion hybrid (`/api/research`, `hybrid_retinexformer`) | HTTP 200 |
| Memory under 2x Compare All | at most 1.21 GiB in `docker stats` samples (sampling misses short peaks; the earlier cgroup peak measurement was 1.7-2.6 GB) out of 4 GiB |
| Upload cleanup | 0 `lumen_upload_*` files left in `/tmp` after all requests |
| Logs | no errors, warnings or tracebacks |
| Graceful shutdown | `docker stop` (SIGTERM, 10 s grace, as on Cloud Run) exited in 1.4 s with **exit code 0**, not OOM-killed |
| No checkpoint mounted | starts; Natural and Highlight-safe `UNAVAILABLE: No retinexformer checkpoint found (looked for: /models/retinexformer_LOL_v1.pth)`; Quick works |

**Not verified:** an actual Cloud Run deployment, Cloud Build in your project, the Cloud Storage volume on
Cloud Run, and real cold-start times.
