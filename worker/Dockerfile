FROM python:3.11-slim

# WeasyPrint system deps (Debian Trixie uses libgdk-pixbuf-2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev python3-pip python3-setuptools \
    python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY worker/requirements.txt /app/requirements.txt

# Install deps — CPU-only PyTorch (pinned in requirements.txt with
# --extra-index-url to the CPU wheel index; the explicit ==2.13.0+cpu pin
# prevents pip from re-resolving torch from the default PyPI index to the
# ~2.5GB GPU wheel during the requirements install).
RUN pip install --no-cache-dir -r requirements.txt

# HF cache — Koyeb volume mounts here so model persists across deploys
ENV HF_HOME=/app/hf_cache
ENV PREDICTABILITY_MODEL=gpt2
ENV SEMANTIC_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Copy both poc/ and worker/ — poc/ baked in as fallback
# Runtime git pull (if GIT_PAT set) overlays latest code on top
COPY poc/ /app/poc/
COPY worker/ /app/worker/

# Install git for runtime code pulls
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/worker

# entrypoint.sh checks volume for cached model before starting Celery
RUN chmod +x /app/worker/entrypoint.sh
ENTRYPOINT ["/app/worker/entrypoint.sh"]
