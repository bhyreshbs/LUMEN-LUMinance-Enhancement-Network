# Deploying LUMEN on Hugging Face Spaces (free tier)

LUMEN runs on a free **CPU basic** Space (Docker SDK). No GPU and no paid plan are needed. The Space runs the
same application and production defaults as `Dockerfile` (Natural / Highlight-safe / Quick, 768 px limit). The
experimental noise-study code is **not** included.

## What is prepared in this repository

| File | Purpose |
|---|---|
| `deploy/huggingface/Dockerfile` | Space image: runs as uid 1000 (a Spaces requirement), 2 CPU threads, app on port 7860 |
| `deploy/huggingface/start.py` | At start-up: downloads the Retinexformer checkpoint, checks its SHA-256, then starts the app |
| `deploy/huggingface/README.md` | Space card (the YAML header tells Spaces: `sdk: docker`, `app_port: 7860`) |
| `scripts/prepare_hf_space.py` | Writes the folder to upload: `dist/hf_space/` (45 files, 0.55 MB; refuses to include checkpoints or `.env` files) |

**Verified locally (2026-11-07, Docker Desktop, `--cpus 2 --memory 3g`, i.e. free-tier-sized):**
- Image built and started.
- Checkpoint downloaded at start-up and SHA-256 verified.
- `scripts/smoke_test_app.py` passed every check: 3 styles x 4 examples, Compare All, invalid uploads.
- Natural took 3.4-4.6 s per example image on 2 CPUs.

Failure behaviour was checked too. With no checkpoint source, a wrong file (rejected by SHA-256) or a private
repo without a token (HTTP 401), the Space still starts. It offers only **Quick** and shows the reason.

**Not verified:** the real Hugging Face build and runtime. Spaces' shared CPUs may be slower than the laptop.

## Why the checkpoint is fetched separately

`retinexformer_LOL_v1.pth` (6.5 MB) carries no explicit licence, so its redistribution rights are uncertain
(`THIRD_PARTY_NOTICES.md`). It is therefore not in this repository and not in the Space files. Instead you put it in a
**private** Hugging Face model repo, and the Space downloads it at start-up with a read token.

## What you need to do

### 0. Decide the licensing question first
A public Space lets anyone use outputs of the Retinexformer weights, even though the file itself stays
private. Check with the hackathon organisers, or record your decision, before making the Space public.
This is an open item in `phases.md`.

### 1. Create a free Hugging Face account
Go to https://huggingface.co/join.

### 2. Put the checkpoint in a private model repo
1. Click **New -> Model**, name it `lumen-weights`, set visibility to **Private**, then **Create**.
2. Open **Files -> Add file -> Upload files** and upload `weights/retinexformer_LOL_v1.pth` from your machine.
   The file name must stay exactly `retinexformer_LOL_v1.pth`.

### 3. Create a read token for the Space
Go to **Settings -> Access Tokens -> Create new token**:
- Type: **Fine-grained**.
- Permission: **read** access to the repo `your-name/lumen-weights` only.
- Copy the token. Never paste it into any file in this repository.

### 4. Create the Space
1. Click **New -> Space**.
2. Name it `lumen`, choose SDK **Docker**, template **Blank** and hardware **CPU basic (free)**.
3. Set visibility to **Public** (after step 0).

### 5. Add the secret and the variable
In the Space, open **Settings -> Variables and secrets**:

| Kind | Name | Value |
|---|---|---|
| **Secret** | `HF_TOKEN` | the read token from step 3 |
| Variable | `LUMEN_WEIGHTS_REPO` | `your-name/lumen-weights` |

### 6. Upload the Space files
On your machine, from the repository root:
```bash
python scripts/prepare_hf_space.py                 # -> dist/hf_space/
pip install -U huggingface_hub
hf auth login                                      # paste a token with *write* access (a different token from step 3)
hf upload your-name/lumen dist/hf_space . --repo-type space
```
Older `huggingface_hub` versions name the commands `huggingface-cli login` and `huggingface-cli upload ...`.
Use the CLI (or the web uploader) rather than a plain `git push`: the Hub rejects binary files (the example
JPEGs) pushed without LFS/Xet, and the CLI handles that automatically.

### 7. Watch the build and check the result
1. Open the Space's **Logs** tab. The first build takes several minutes because it installs CPU PyTorch.
   When it is healthy you will see:
   ```
   [lumen] checkpoint downloaded from your-name/lumen-weights and verified (6.5 MB)
   ... mode standard        available
   ... LUMEN ready on http://localhost:7860
   ```
2. Open `https://your-name-lumen.hf.space` on a laptop **and** on a phone, and restore one of the examples.
3. Optional, from the repository root:
   ```bash
   python scripts/smoke_test_app.py --url https://your-name-lumen.hf.space --expect-modes standard quality_guided classical
   ```

### If something is wrong
| Log line | Fix |
|---|---|
| `no LUMEN_WEIGHTS_REPO / LUMEN_WEIGHTS_URL set` | Add the variable from step 5, then restart the Space |
| `failed: HTTP 401 (check that HF_TOKEN ...)` | The secret is missing, or the token cannot read the weights repo |
| `failed: HTTP 404` | Wrong repo name, or the file is not named `retinexformer_LOL_v1.pth` |
| `downloaded file rejected: SHA-256 ...` | The uploaded file is not the official checkpoint. Re-download it with `python scripts/download_weights.py --runtime` |

## Updating the Space
Re-run steps 6 and 7. `hf upload` replaces changed files, but files you deleted locally stay in the Space
until you delete them there (Files tab).

## Free-tier behaviour to expect (check Hugging Face's current documentation)
- **Sleeping:** a free Space sleeps after a period without visitors. The next visit restarts it (a minute or
  two). Open the Space a few minutes before a demo or viva.
- **Speed:** about 2 vCPU, so expect roughly 4-10 s per restoration. Requests run **one at a time**; a second
  visitor waits for the first. That is fine for a demo, but not for many simultaneous users.
- **Privacy:** unchanged from the Docker image. Uploads are deleted after each request and never logged.
