# LUMEN - Hugging Face Spaces image (Docker SDK, free "CPU basic" hardware: 2 vCPU, no GPU).
#
# Same application, same production defaults as the repository's Dockerfile. Differences, all for Spaces:
#   * runs as uid 1000 ("user"), which Spaces requires; app lives in /home/user/app
#   * the checkpoint is fetched at start-up by start.py (private HF model repo or URL, SHA-256 verified)
#     instead of being mounted as a volume - Spaces cannot mount volumes
#   * LUMEN_TORCH_THREADS=2 pinned to the free tier's 2 vCPUs (a shared host may not expose a cgroup quota)
# Built from the folder that scripts/prepare_hf_space.py writes (dist/hf_space/).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LUMEN_HOST=0.0.0.0 PORT=7860 \
    LUMEN_DEVICE=cpu LUMEN_MAX_SIDE=768 LUMEN_TORCH_THREADS=2 \
    LUMEN_RETINEXFORMER_WEIGHTS=/home/user/models/retinexformer_LOL_v1.pth \
    HOME=/home/user TORCH_HOME=/home/user/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

WORKDIR /home/user/app
COPY --chown=user requirements.txt ./
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY --chown=user lumen ./lumen
COPY --chown=user app ./app
COPY --chown=user configs ./configs
COPY --chown=user scripts ./scripts
COPY --chown=user weights ./weights
COPY --chown=user start.py THIRD_PARTY_NOTICES.md ./

# Build-time checks (as in the main Dockerfile): calibrated QGH config valid, app imports resolve, frontend present.
RUN python -c "from lumen.hybrid import load_qg_config; c = load_qg_config(); print('QG config OK:', c.enabled)" \
    && python -c "import sys; sys.path[:0] = ['app']; import service, server, app; print('app imports OK')" \
    && test -f app/web/index.html && test -f app/web/details.html && test -f app/web/static/js/workspace.js \
    && mkdir -p /home/user/models && chown -R user /home/user

USER user
EXPOSE 7860
CMD ["python", "start.py"]
