"""Lazy singleton loader + window inference for the off-the-shelf AI-text-detection checkpoint.

Loads from the existing worker HF volume (HF_HOME=/app/hf_cache). Tries ``local_files_only``
first, then a network fallback into the cache — same pattern as
``poc/predictability/scanner.py``. The model loads on first use (not at import) so worker
cold-start stays off the hot path; a one-time warm-download
(``poc/detect/_download_deberta.py``) pre-fills the volume.

Checkpoint-agnostic: the AI-class index is read from ``model.config.id2label`` rather than
hardcoded, so any 2-label (human/AI) sequence-classification checkpoint works without code
changes. The repo-id comes from ``DRAFTPROOF_DEBERTA_MODEL`` (set by entrypoint); the default
is the current research leading candidate and is confirmed/overridden by the Phase-0 SCoCESLE
gate before it touches real student text.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import List, Optional

logger = logging.getLogger(__name__)

# Process-level bounded LRU over per-window scores, keyed on sha256(window text) + model tag.
# A single rewrite fires many gate scans over near-identical documents, so the SAME window text is
# re-scored repeatedly; caching the final per-window score makes hits return identical values by
# construction (scores are unchanged). Mirrors poc/predictability/scanner.py's _SENTENCE_CACHE.
# Env cap DRAFTPROOF_DEBERTA_WINDOW_CACHE_MAX (default 4096; 0 disables). score_windows can be
# called concurrently, so get/put are guarded by _WINDOW_CACHE_LOCK.
_WINDOW_CACHE_LOCK = threading.RLock()
_WINDOW_CACHE: "OrderedDict[str, float]" = OrderedDict()


def _window_cache_max() -> int:
    try:
        value = int(os.environ.get("DRAFTPROOF_DEBERTA_WINDOW_CACHE_MAX", "4096"))
    except (TypeError, ValueError):
        return 4096
    return max(0, value)


def _window_cache_key(window: str, model_tag: str) -> str:
    return hashlib.sha256(f"{model_tag}\0{window}".encode("utf-8")).hexdigest()


def clear_window_cache() -> None:
    """Clear the process-local DeBERTa window-score cache (used by parity/timing tests)."""
    with _WINDOW_CACHE_LOCK:
        _WINDOW_CACHE.clear()

# Research leading candidate (poc/calibration/deberta_candidates.md). Confirmed or replaced by
# the Phase-0 SCoCESLE ESL-FPR gate (Task 0.3/0.4) before production use.
_DEFAULT_MODEL = "fakespot-ai/roberta-base-ai-text-detection-v1"

# Substrings (lowercased) in an id2label value that indicate the AI/fake class. Order-independent.
_AI_LABEL_HINTS = ("ai", "fake", "generated", "machine", "chatgpt", "synthetic", "llm", "robot")

_MODEL = None
_TOKENIZER = None
_AI_INDEX = None
_LOCK = threading.Lock()


def _model_name() -> str:
    return os.environ.get("DRAFTPROOF_DEBERTA_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def _resolve_ai_index(config) -> int:
    """Pick the index of the AI-like class from the checkpoint's id2label.

    Most detectors put AI at index 1, but not all — read the labels to be safe. Falls back to 1
    (the convention) if no label name matches an AI hint (e.g. bare {"0","1"} labels)."""
    id2label = {int(k): str(v).lower() for k, v in dict(getattr(config, "id2label", {}) or {}).items()}
    for idx, name in id2label.items():
        if any(hint in name for hint in _AI_LABEL_HINTS):
            return idx
    return 1


def _load():
    global _MODEL, _TOKENIZER, _AI_INDEX
    if _MODEL is not None:
        return
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = _model_name()
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(name, local_files_only=True)
    except (OSError, EnvironmentError):
        logger.warning("[deberta] %s not in HF cache; downloading into HF_HOME", name)
        cache = os.environ.get("HF_HOME")
        # NOTE: transformers from_pretrained uses `cache_dir` (NOT `cache_folder`, which is
        # the sentence-transformers convention and raises TypeError on AutoModel).
        tok = AutoTokenizer.from_pretrained(name, cache_dir=cache)
        mdl = AutoModelForSequenceClassification.from_pretrained(name, cache_dir=cache)
    mdl.eval()
    _AI_INDEX = _resolve_ai_index(mdl.config)
    _TOKENIZER, _MODEL = tok, mdl
    logger.info("[deberta] loaded %s (AI class index = %d)", name, _AI_INDEX)


def score_windows(windows: List[str]) -> Optional[List[float]]:
    """Return the AI-like probability per window, or ``None`` on any failure (fail-open).

    Fail-open is mandatory: this feeds an additive advisory field and must never break the scan.
    """
    if not windows:
        return []
    try:
        max_entries = _window_cache_max()
        model_tag = _model_name()
        results: List[Optional[float]] = [None] * len(windows)
        keys: List[Optional[str]] = [None] * len(windows)
        missing_indexes: List[int] = []
        hits = 0

        if max_entries > 0:
            keys = [_window_cache_key(w, model_tag) for w in windows]
            with _WINDOW_CACHE_LOCK:
                for index, key in enumerate(keys):
                    cached = _WINDOW_CACHE.get(key)
                    if cached is None:
                        missing_indexes.append(index)
                        continue
                    _WINDOW_CACHE.move_to_end(key)
                    results[index] = cached
                    hits += 1
        else:
            missing_indexes = list(range(len(windows)))

        if missing_indexes:
            with _LOCK:
                _load()
            import torch

            missing_windows = [windows[i] for i in missing_indexes]
            enc = _TOKENIZER(
                missing_windows,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            with torch.no_grad():
                logits = _MODEL(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, _AI_INDEX]
            scored = [float(p) for p in probs]
            with _WINDOW_CACHE_LOCK:
                for index, score in zip(missing_indexes, scored):
                    results[index] = score
                    if max_entries > 0 and keys[index] is not None:
                        _WINDOW_CACHE[keys[index]] = score
                        _WINDOW_CACHE.move_to_end(keys[index])
                while max_entries > 0 and len(_WINDOW_CACHE) > max_entries:
                    _WINDOW_CACHE.popitem(last=False)

        logger.info(
            "[deberta] window cache: %d hit(s), %d miss(es) over %d window(s)",
            hits, len(missing_indexes), len(windows),
        )
        return [r for r in results if r is not None]
    except Exception as e:  # noqa: BLE001 — fail-open: never block the primary scan
        logger.warning("[deberta] inference failed (fail-open): %s", e)
        return None
