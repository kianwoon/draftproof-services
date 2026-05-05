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
# The Docker image bakes in worker/ but NOT poc/.
# poc/ is cloned from GitHub so code-only deploys skip Docker rebuild.
CODE_DIR="/app/poc"
REPO_URL="${GIT_REPO_URL:-https://github.com/kianwoon/draftproof-services.git}"
REPO_BRANCH="${GIT_REPO_BRANCH:-main}"

if [ -n "${GIT_PAT}" ]; then
    # Inject PAT into URL for private repo access
    AUTH_URL="${REPO_URL/https:\/\//https:\/\/${GIT_PAT}@}"
else
    AUTH_URL="${REPO_URL}"
fi

if [ -d "${CODE_DIR}/.git" ]; then
    echo "[entrypoint] Pulling latest poc/ code from ${REPO_BRANCH}..."
    cd "${CODE_DIR}" && git fetch origin "${REPO_BRANCH}" && git reset --hard "origin/${REPO_BRANCH}"
else
    echo "[entrypoint] Cloning poc/ code from ${REPO_URL} (${REPO_BRANCH})..."
    rm -rf "${CODE_DIR}"
    git clone --depth 1 --branch "${REPO_BRANCH}" "${AUTH_URL}" /tmp/draftproof-repo
    cp -a /tmp/draftproof-repo/poc "${CODE_DIR}"
    rm -rf /tmp/draftproof-repo
fi

CODE_SHA=$(cd "${CODE_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "[entrypoint] poc/ code at SHA: ${CODE_SHA}"

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
