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
    echo "[entrypoint] Model ${MODEL} already cached on volume (marker exists), skipping download"
else
    echo "[entrypoint] Downloading ${MODEL} to volume (first run)..."
    python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
        AutoTokenizer.from_pretrained('${MODEL}'); \
        AutoModelForCausalLM.from_pretrained('${MODEL}'); \
        print('Download complete')"
    touch "${MODEL_MARKER}"
    echo "[entrypoint] Model ${MODEL} cached and marker set"
fi

# Preload model into memory + run 10-sentence benchmark
echo "[entrypoint] Preloading ${MODEL} into memory..."
python3 << 'PYEOF'
import os, time, torch, logging
logging.basicConfig(level=logging.INFO, format="[preload] %(message)s")
log = logging.getLogger("preload")

model_name = os.environ.get("PREDICTABILITY_MODEL", "gpt2")
log.info("Loading %s ...", model_name)

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.monotonic()
tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
model.eval()
load_s = time.monotonic() - t0

params = sum(p.numel() for p in model.parameters())
size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
log.info("Model loaded: %s (%s params, %.0f MB) in %.1fs", model_name, f"{params:,}", size_mb, load_s)

# Log torch build config for performance debugging
log.info("Torch: version=%s build=%s MKL=%s OpenMP=%s threads=%d",
    torch.__version__,
    torch.version.debug or "release",
    torch.backends.mkl.is_available(),
    torch.backends.openmp.is_available(),
    torch.get_num_threads(),
)
log.info("Device: %s", model.device if hasattr(model, 'device') else 'cpu')

# 10-sentence benchmark
sentences = [
    "The implementation of these strategies serves as a practical model for addressing complex challenges in modern educational environments.",
    "Students benefit from hands-on experience when learning new techniques and approaches to problem-solving in real-world contexts.",
    "The data shows a clear trend toward increased adoption of digital tools across multiple sectors of the economy.",
    "Furthermore, the research indicates that early intervention leads to significantly better outcomes for most participants.",
    "This approach provides a comprehensive framework for understanding the complex dynamics at play in contemporary organizations.",
    "The results demonstrate the effectiveness of the proposed method across a range of different experimental conditions.",
    "It is important to note that these findings are consistent with previous research in this area of study.",
    "The study highlights the need for further investigation into the underlying mechanisms driving these observed patterns.",
    "In conclusion, the evidence strongly supports the hypothesis that targeted interventions can produce meaningful improvements.",
    "These findings have significant implications for policy makers and practitioners working in this rapidly evolving field.",
]

log.info("Running 10-sentence benchmark ...")
times = []
for s in sentences:
    inputs = tok(s, return_tensors="pt")
    t1 = time.monotonic()
    with torch.no_grad():
        model(inputs["input_ids"])
    times.append(time.monotonic() - t1)

avg_ms = sum(times) / len(times) * 1000
total_ms = sum(times) * 1000
log.info("Benchmark: %d sentences, total=%.0fms, avg=%.1fms/sent", len(times), total_ms, avg_ms)
if avg_ms > 500:
    log.warning("SLOW: avg %.0fms/sentence — consider smaller model or faster instance", avg_ms)
elif avg_ms > 200:
    log.info("MODERATE: avg %.0fms/sentence — acceptable for nano instance", avg_ms)
else:
    log.info("FAST: avg %.0fms/sentence — good performance", avg_ms)

# Store preloaded model in a module-level cache so scanner reuses it
import sys
sys.path.insert(0, "/app")
import poc.predictability.scanner as _sc
_sc._PRELOADED_MODEL = model
_sc._PRELOADED_TOKENIZER = tok
log.info("Preloaded model stored in scanner module cache")
PYEOF

echo "[entrypoint] Starting Celery worker..."
exec celery -A app.celery_app worker --loglevel=info --concurrency=1 -Q scan,default
