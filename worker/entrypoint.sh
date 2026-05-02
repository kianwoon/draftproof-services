#!/usr/bin/env bash
set -e

MODEL="gpt2-medium"
CACHE_DIR="${HF_HOME}/hub"
MODEL_MARKER="${CACHE_DIR}/.model_ready"

echo "[entrypoint] HF_HOME=${HF_HOME}"
echo "[entrypoint] Cache dir: ${CACHE_DIR}"

# Ensure cache dir exists
mkdir -p "${CACHE_DIR}"

# Download model to volume if not already cached
if [ -f "${MODEL_MARKER}" ]; then
    echo "[entrypoint] Model ${MODEL} already cached on volume, skipping download"
else
    echo "[entrypoint] Downloading ${MODEL} to volume (first run)..."
    python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
        AutoTokenizer.from_pretrained('${MODEL}'); \
        AutoModelForCausalLM.from_pretrained('${MODEL}'); \
        print('Download complete')"
    touch "${MODEL_MARKER}"
    echo "[entrypoint] Model ${MODEL} cached and marker set"
fi

echo "[entrypoint] Starting Celery worker..."
exec celery -A app.celery_app worker --loglevel=info --concurrency=1 -Q scan,default
