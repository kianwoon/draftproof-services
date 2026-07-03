"""Celery worker model preloading — opt-in warmup of scan models.

Extracted from tasks.py. The ``@worker_process_init.connect`` handler stays
here; importing this module from ``tasks.py`` registers the signal handler.
"""
import os
import time
import logging

from celery.signals import worker_process_init

logger = logging.getLogger(__name__)


def _configure_torch_threads(torch) -> int | None:
    """Apply explicit Torch CPU thread settings when configured."""
    raw_threads = (
        os.environ.get("TORCH_NUM_THREADS")
        or os.environ.get("OMP_NUM_THREADS")
        or os.environ.get("MKL_NUM_THREADS")
    )
    if not raw_threads:
        return None
    try:
        threads = int(raw_threads)
    except ValueError:
        logger.warning("Invalid TORCH_NUM_THREADS/OMP_NUM_THREADS value: %r", raw_threads)
        return None
    if threads <= 0:
        return None
    try:
        torch.set_num_threads(threads)
    except Exception:
        logger.warning("Failed to set torch num threads to %s", threads, exc_info=True)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except RuntimeError:
        # PyTorch can reject this after parallel work starts. Main inference
        # threads are still controlled by set_num_threads above.
        pass
    except Exception:
        logger.warning("Failed to set torch interop threads", exc_info=True)
    return threads


@worker_process_init.connect
def _preload_scan_models(**_kwargs):
    """Optionally warm scan models inside the Celery worker child process.

    Celery kills a child if this init signal blocks for too long, so expensive
    model warmup must stay opt-in. Normal tasks still lazy-load cached models on
    first use without tripping the worker startup watchdog.
    """
    enabled = os.environ.get("DRAFTPROOF_PRELOAD_PREDICTABILITY", "0").lower()
    if enabled not in {"0", "false", "no"}:
        try:
            _preload_predictability_model()
        except Exception:
            logger.warning("Failed to preload predictability model in worker child", exc_info=True)

    semantic_enabled = os.environ.get("DRAFTPROOF_PRELOAD_SEMANTIC", "0").lower()
    if semantic_enabled not in {"0", "false", "no"}:
        try:
            _preload_semantic_model()
        except Exception:
            logger.warning("Failed to preload semantic embedding model in worker child", exc_info=True)


def _preload_predictability_model():
    """Warm the GPT-2 scanner inside the Celery worker child process."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import predictability.scanner as scanner_module

        requested_threads = _configure_torch_threads(torch)
        model_name = scanner_module.resolve_predictability_model_name(
            os.environ.get("PREDICTABILITY_MODEL", "gpt2")
        )
        if scanner_module._PRELOADED_MODEL is not None:
            return
        logger.info("Preloading predictability model in worker child: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        scanner_module._PRELOADED_MODEL = model
        scanner_module._PRELOADED_TOKENIZER = tokenizer
        scanner_module._PRELOADED_MODEL_NAME = model_name
        try:
            mkl_available = bool(getattr(torch.backends, "mkl", None) and torch.backends.mkl.is_available())
            openmp_available = bool(getattr(torch.backends, "openmp", None) and torch.backends.openmp.is_available())
            logger.info(
                "[preload] Torch: version=%s MKL=%s OpenMP=%s threads=%s requested_threads=%s",
                torch.__version__,
                mkl_available,
                openmp_available,
                torch.get_num_threads(),
                requested_threads,
            )
            sample_sentences = [
                "People test the process before they rely on the final result.",
                "The reviewer checks the earlier step against the latest decision.",
                "A small condition can change once the system starts to respond.",
                "The task needs a visible process, not only a final result.",
                "During review, a participant may explain the goal indirectly.",
                "The facilitator can slow the task down at the decision stage.",
                "Writers need to connect the reason, constraint, and outcome.",
                "A draft shows the mistake before the final version does.",
                "A short change can alter how the reader understands the point.",
                "The working context makes technical decisions visible quickly.",
            ]
            encoded = tokenizer(
                sample_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(os.environ.get("DRAFTPROOF_PREDICTABILITY_MAX_TOKENS", "384")),
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            bench_t0 = time.monotonic()
            with torch.no_grad():
                model(**encoded)
            bench_ms = (time.monotonic() - bench_t0) * 1000
            logger.info(
                "[preload] Benchmark: %d sentences, total=%.0fms, avg=%.1fms/sent",
                len(sample_sentences),
                bench_ms,
                bench_ms / max(len(sample_sentences), 1),
            )
        except Exception:
            logger.warning("[preload] Benchmark failed", exc_info=True)
        logger.info("Predictability model preloaded in worker child: %s", model_name)
    except Exception:
        raise


def _preload_semantic_model():
    """Warm the sentence embedding scanner inside the Celery worker child process."""
    from sentence_transformers import SentenceTransformer
    import detect.semantic_shape as semantic_module

    model_name = semantic_module.resolve_semantic_embedding_model_name(
        os.environ.get("SEMANTIC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    if semantic_module._PRELOADED_EMBEDDER is not None:
        return
    logger.info("Preloading semantic embedding model in worker child: %s", model_name)
    embedder = SentenceTransformer(
        model_name,
        cache_folder=os.environ.get("HF_HOME"),
        local_files_only=True,
    )
    sample_sentences = [
        "Students practise the sectioning pattern before they cut.",
        "The stylist checks the guide length against the previous section.",
        "The lesson needs a visible process, not only a final result.",
        "Learners need to connect the angle, tension, and design line.",
    ]
    bench_t0 = time.monotonic()
    embedder.encode(sample_sentences, convert_to_numpy=True, normalize_embeddings=True)
    bench_ms = (time.monotonic() - bench_t0) * 1000
    semantic_module._PRELOADED_EMBEDDER = embedder
    semantic_module._PRELOADED_MODEL_NAME = model_name
    logger.info(
        "[preload] Semantic benchmark: %d sentences, total=%.0fms, avg=%.1fms/sent",
        len(sample_sentences),
        bench_ms,
        bench_ms / max(len(sample_sentences), 1),
    )
    logger.info("Semantic embedding model preloaded in worker child: %s", model_name)
