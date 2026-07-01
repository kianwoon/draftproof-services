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

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

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
        tok = AutoTokenizer.from_pretrained(name, cache_folder=cache)
        mdl = AutoModelForSequenceClassification.from_pretrained(name, cache_folder=cache)
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
        with _LOCK:
            _load()
        import torch

        enc = _TOKENIZER(
            windows,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = _MODEL(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, _AI_INDEX]
        return [float(p) for p in probs]
    except Exception as e:  # noqa: BLE001 — fail-open: never block the primary scan
        logger.warning("[deberta] inference failed (fail-open): %s", e)
        return None
