"""Overlapping sentence-window construction + aggregation for the DeBERTa signal.

Pure Python — no ML. Tested independently of the model. Mirrors the windowing strategy
from the source design doc (sections 6-9)."""
from __future__ import annotations

import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text] if text else [])


def build_windows(sentences: List[str], size: int = 3, step: int = 1) -> List[str]:
    if not sentences:
        return []
    if len(sentences) <= size:
        return [" ".join(sentences)]
    windows: List[str] = []
    i = 0
    while i < len(sentences):
        windows.append(" ".join(sentences[i:i + size]))
        if i + size >= len(sentences):
            break
        i += step
    return windows


def aggregate(
    sentences: List[str],
    windows: List[str],
    window_probs: List[float],
    size: int = 3,
    step: int = 1,
) -> dict:
    """Map window probs back to sentences (mean of covering windows), then mean -> document.

    Coverage is re-derived by substring membership: sentence k is "covered" by a window if
    that window's text contains sentence k's text. This is robust to arbitrary size/step
    (the windowing that actually built `windows`), so callers do not need to thread size/step
    through — they just pass the windows they built and the per-window probabilities.
    """
    norm_sentences = [(s or "").strip() for s in sentences]
    coverage: List[List[float]] = [[] for _ in norm_sentences]
    for wi, w in enumerate(windows):
        if wi >= len(window_probs):
            break
        p = window_probs[wi]
        w_norm = (w or "").strip()
        for ki, s in enumerate(norm_sentences):
            if not s:
                continue
            # a sentence is covered if it appears verbatim inside the window text
            if s in w_norm:
                coverage[ki].append(p)
    sentence_scores = [
        (sum(c) / len(c)) if c else 0.0 for c in coverage
    ]
    doc = (sum(sentence_scores) / len(sentence_scores)) if sentence_scores else 0.0
    return {"document_score": doc, "sentence_scores": sentence_scores}
