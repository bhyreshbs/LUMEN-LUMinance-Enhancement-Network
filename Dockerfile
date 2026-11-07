# LUMEN web app - CPU image (no GPU required; a GPU host needs a CUDA base image + CUDA torch wheel).
#
# The Retinexformer checkpoint is NOT in the image (its redistribution rights are uncertain - weights/README.md).
# Supply it at run time as a read-only volume; the app verifies its SHA-256 at startup:
#   python scripts/download_weights.py --runtime            # once, on the host -> weights/retinexformer_LOL_v1.pth
#   docker build -t lumen .                                 # add --build-arg INSTALL_NIQE=1 for the NIQE checkbox
#   docker run --rm -p 7860:7860 -v "$PWD/weights:/models:ro" lumen     # -> http://localhost:7860
# Without the volume the app still starts: only the "Quick" (classical) style is offered.
#
# LUMEN_MAX_SIDE=768: measured 5.8 s median with 2 CPU threads (1024 px: 9.7 s) - see docs/DEPLOYMENT.md.
# The build FAILS if the calibrated Quality Guided config is missing/invalid or a runtime import is missing.
FROM python:3.12-slim

ARG INSTALL_NIQE=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LUMEN_HOST=0.0.0.0 PORT=7860 \
    LUMEN_DEVICE=cpu LUMEN_MAX_SIDE=768 \
    LUMEN_RETINEXFORMER_WEIGHTS=/models/retinexformer_LOL_v1.pth \
    HOME=/home/appuser TORCH_HOME=/home/appuser/.cache/torch

# libglib2.0-0: required by OpenCV wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -d /home/appuser appuser \
    && mkdir -p /models

WORKDIR /app
COPY requirements.txt requirements-niqe.txt ./
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_NIQE" = "1" ]; then \
         pip install -r requirements-niqe.txt --extra-index-url https://download.pytorch.org/whl/cpu; fi

COPY lumen ./lumen
COPY app ./app
COPY configs ./configs
COPY scripts/download_weights.py ./scripts/download_weights.py
COPY weights/README.md ./weights/README.md

# Build-time checks: calibrated QGH config valid, every runtime import resolvable, frontend present.
RUN python -c "from lumen.hybrid import load_qg_config; c = load_qg_config(); print('QG config OK:', c.enabled)" \
    && python -c "import sys; sys.path[:0] = ['app']; import service, server, app; print('app imports OK')" \
    && test -f app/web/index.html && test -f app/web/details.html && test -f app/web/static/js/workspace.js \
    && chown -R appuser /app /home/appuser

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '7860'), timeout=5)"
CMD ["python", "app/app.py"]
