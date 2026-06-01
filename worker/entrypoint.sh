#!/usr/bin/env bash
set -e

MODEL="${PREDICTABILITY_MODEL:-gpt2}"
SEMANTIC_MODEL="${SEMANTIC_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
CELERY_WORKER_CONCURRENCY="${CELERY_WORKER_CONCURRENCY:-1}"
CACHE_DIR="${HF_HOME}/hub"
SAFE_MODEL="${MODEL//\//_}"
SAFE_SEMANTIC_MODEL="${SEMANTIC_MODEL//\//_}"
# Model-specific marker so switching models triggers re-download
MODEL_MARKER="${CACHE_DIR}/.model_ready_${SAFE_MODEL}"
SEMANTIC_MARKER="${CACHE_DIR}/.semantic_model_ready_${SAFE_SEMANTIC_MODEL}"

# Keep CPU inference deterministic across worker instances. Koyeb/container
# defaults can vary; the worker preload also calls torch.set_num_threads().
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${OMP_NUM_THREADS}}"

echo "[entrypoint] ============================================"
echo "[entrypoint] DraftProof Worker Startup"
echo "[entrypoint] HF_HOME=${HF_HOME}"
echo "[entrypoint] Cache dir: ${CACHE_DIR}"
echo "[entrypoint] PREDICTABILITY_MODEL=${MODEL}"
echo "[entrypoint] SEMANTIC_EMBEDDING_MODEL=${SEMANTIC_MODEL}"
echo "[entrypoint] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[entrypoint] MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "[entrypoint] TORCH_NUM_THREADS=${TORCH_NUM_THREADS}"
echo "[entrypoint] CELERY_WORKER_CONCURRENCY=${CELERY_WORKER_CONCURRENCY}"
echo "[entrypoint] Marker: ${MODEL_MARKER}"
echo "[entrypoint] Semantic marker: ${SEMANTIC_MARKER}"
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
        CODE_SHA=$(git -C /tmp/draftproof-repo rev-parse --short HEAD 2>/dev/null || echo "unknown")
        rm -rf "${CODE_DIR}"
        cp -a /tmp/draftproof-repo/poc "${CODE_DIR}"
        # Also overlay latest worker/app/ code for fast code deploys.
        # Important: /app/worker/app already exists in the image. Copying the
        # source directory onto that path would create /app/worker/app/app and
        # leave the baked tasks.py loaded by Celery. Replace the package so the
        # runtime SHA and imported worker code cannot diverge.
        rm -rf /app/worker/app
        mkdir -p /app/worker
        cp -a /tmp/draftproof-repo/worker/app /app/worker/app
        export DRAFTPROOF_RUNTIME_CODE_SHA="${CODE_SHA}"
        echo "${CODE_SHA}" > /app/runtime_code_sha
        echo "${CODE_SHA}" > /app/poc/.runtime_git_sha
        echo "${CODE_SHA}" > /app/worker/app/.runtime_git_sha
        rm -rf /tmp/draftproof-repo
        echo "[entrypoint] poc/ and worker/app/ updated via git pull. SHA: ${CODE_SHA}"
    else
        echo "[entrypoint] Git pull failed — using baked-in poc/ code"
        rm -rf /tmp/draftproof-repo 2>/dev/null || true
    fi
else
    echo "[entrypoint] No GIT_PAT set — using baked-in poc/ code"
fi

# ── Volume mount guard ────────────────────────────────────────────
# Verify HF_HOME is a real mounted volume, not a plain container dir.
# If the Koyeb volume is detached, models silently write to ephemeral
# storage and are re-downloaded on every redeploy.
CACHE_DEV=$(stat -c %d "${HF_HOME}" 2>/dev/null || echo "unknown")
ROOT_DEV=$(stat -c %d / 2>/dev/null || echo "root")
if [ "${CACHE_DEV}" = "${ROOT_DEV}" ] || [ "${CACHE_DEV}" = "unknown" ]; then
    echo "[entrypoint] ERROR: ${HF_HOME} is NOT a mounted volume (same device as /)"
    echo "[entrypoint]   Model cache will not persist across redeploys."
    echo "[entrypoint]   Ensure volume eeb30e19 is attached at /app/hf_cache in Koyeb."
    exit 1
fi
echo "[entrypoint] Volume check OK: ${HF_HOME} is a mounted volume (dev=${CACHE_DEV})"

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

if [ -f "${SEMANTIC_MARKER}" ]; then
    echo "[entrypoint] Semantic model ${SEMANTIC_MODEL} already cached on volume, skipping download"
else
    echo "[entrypoint] Downloading semantic model ${SEMANTIC_MODEL} to volume (first run)..."
    python3 -c "from sentence_transformers import SentenceTransformer; \
        SentenceTransformer('${SEMANTIC_MODEL}', cache_folder='${HF_HOME}')"
    echo "[entrypoint] Semantic model ${SEMANTIC_MODEL} cached and marker set"
    touch "${SEMANTIC_MARKER}"
fi

echo "[entrypoint] Model cache ready. Celery worker child will lazy-load cached scan models unless preload env flags are enabled."

echo "[entrypoint] Starting Celery worker..."
cd /app/worker
exec celery -A app.celery_app worker \
    --loglevel=info \
    --concurrency="${CELERY_WORKER_CONCURRENCY}" \
    --pool=prefork \
    -Q default,scan \
    --without-heartbeat \
    --without-gossip \
    --without-mingle
