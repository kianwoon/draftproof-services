#!/usr/bin/env bash
set -e

MODEL="${PREDICTABILITY_MODEL:-gpt2}"
CACHE_DIR="${HF_HOME}/hub"
# Model-specific marker so switching models triggers re-download
MODEL_MARKER="${CACHE_DIR}/.model_ready_${MODEL}"

echo "[entrypoint] ============================================"
echo "[entrypoint] DraftProof Worker Startup"
echo "[entrypoint] HF_HOME=${HF_HOME}"
echo "[entrypoint] Cache dir: ${CACHE_DIR}"
echo "[entrypoint] PREDICTABILITY_MODEL=${MODEL}"
echo "[entrypoint] Marker: ${MODEL_MARKER}"
echo "[entrypoint] ============================================"

# ── Pull latest poc/ code at runtime ──────────────────────────────
# poc/ is baked into the image as fallback. If GIT_PAT is set,
# we try to git pull the latest code on top (for fast code-only deploys).
# If git pull fails, we fall back to the baked-in copy.
CODE_DIR="/app/poc"
REPO_URL="${GIT_REPO_URL:-https://github.com/kianwoon/draftproof-services.git}"
REPO_BRANCH="${GIT_REPO_BRANCH:-main}"

if [ -n "${GIT_PAT}" ]; then
    AUTH_URL=$(echo "${REPO_URL}" | sed "s|https://|https://${GIT_PAT}@|")
    echo "[entrypoint] Attempting git pull for latest poc/ code..."
    if git clone --depth 1 --branch "${REPO_BRANCH}" "${AUTH_URL}" /tmp/draftproof-repo 2>/dev/null; then
        rm -rf "${CODE_DIR}"
        cp -a /tmp/draftproof-repo/poc "${CODE_DIR}"
        rm -rf /tmp/draftproof-repo
        CODE_SHA=$(cd "${CODE_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "[entrypoint] poc/ updated via git pull. SHA: ${CODE_SHA}"
    else
        echo "[entrypoint] Git pull failed — using baked-in poc/ code"
        rm -rf /tmp/draftproof-repo 2>/dev/null || true
    fi
else
    echo "[entrypoint] No GIT_PAT set — using baked-in poc/ code"
fi

# Ensure cache dir exists
mkdir -p "${CACHE_DIR}"

# Migrate old generic marker to model-specific marker
OLD_MARKER="${CACHE_DIR}/.model_ready"
if [ -f "${OLD_MARKER}" ] && [ ! -f "${MODEL_MARKER}" ]; then
    echo "[entrypoint] Migrating old marker to ${MODEL_MARKER}"
    mv "${OLD_MARKER}" "${MODEL_MARKER}"
fi

# Download model to volume if not already cached
if [ -f "${MODEL_MARKER}" ]; then
    echo "[entrypoint] Model ${MODEL} already cached on volume, skipping download"
else
    echo "[entrypoint] Downloading ${MODEL} to volume (first run)..."
    python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
        AutoModelForCausalLM.from_pretrained('${MODEL}'); \
        AutoTokenizer.from_pretrained('${MODEL}')"
    echo "[entrypoint] Model ${MODEL} cached and marker set"
    touch "${MODEL_MARKER}"
fi

# Preload model into memory before Celery starts (saves ~2s per first task)
echo "[entrypoint] Preloading ${MODEL} into memory..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained('${MODEL}', torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained('${MODEL}')
# Store in the scanner module so tasks reuse it
from predictability.scanner import PredictabilityScanner
s = PredictabilityScanner.__new__(PredictabilityScanner)
s._model = model
s._tokenizer = tokenizer
s._device = 'cpu'
PredictabilityScanner._shared = s
print('[preload] Preloaded model stored in scanner module cache')
"

echo "[entrypoint] Starting Celery worker..."
cd /app/worker
exec celery -A app.celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --pool=prefork \
    -Q default,scan
