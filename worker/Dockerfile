FROM python:3.11-slim

# WeasyPrint system deps (Debian Trixie uses libgdk-pixbuf-2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev python3-pip python3-setuptools \
    python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY worker/requirements.txt /app/requirements.txt

# Install deps — use CPU-only PyTorch for smaller image
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# DeBERTa second-opinion checkpoint — BAKE into the image at build time. The worker's runtime
# cannot reach huggingface.co (egress blocked), so lazy-load and boot warm-download both fail.
# Saved OUTSIDE the HF volume mount (/app/hf_cache is a volume overlay) at /app/prebuilt/..., so
# it ships with the image and is always present. DRAFTPROOF_DEBERTA_MODEL points the loader here;
# from_pretrained on a local dir loads with local_files_only — zero runtime network.
RUN python3 -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
    import os; os.makedirs('/app/prebuilt/deberta-fakespot', exist_ok=True); \
    AutoModelForSequenceClassification.from_pretrained('fakespot-ai/roberta-base-ai-text-detection-v1').save_pretrained('/app/prebuilt/deberta-fakespot'); \
    AutoTokenizer.from_pretrained('fakespot-ai/roberta-base-ai-text-detection-v1').save_pretrained('/app/prebuilt/deberta-fakespot')"
ENV DRAFTPROOF_DEBERTA_MODEL=/app/prebuilt/deberta-fakespot

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
