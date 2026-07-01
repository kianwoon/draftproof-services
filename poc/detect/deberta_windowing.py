"""Overlapping sentence-window construction + aggregation for the DeBERTa signal.

Pure Python — no ML. Tested independently of the model. Mirrors the windowing strategy
from the source design doc (sections 6-9)."""
from __future__ import annotations

import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    # TODO: does not handle abbreviations/initials ("Dr.", "U.S.", "e.g.") — over-splits
    # on the period. The downstream composer may need to pre-segment for those.
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

    Coverage is derived by index arithmetic matching build_windows: window j starts at
    sentence j*step and covers [j*step, j*step+size). size/step are AUTHORITATIVE — callers
    must pass the same size/step used to build the windows. Uncovered sentences return None
    (not a silent 0.0) to surface coverage gaps; the document mean ignores None entries.
    """
    n = len(sentences)
    coverage: List[List[float]] = [[] for _ in range(n)]
    for j, prob in enumerate(window_probs):
        if j >= len(windows):
            break
        start = j * step
        for k in range(start, min(start + size, n)):
            coverage[k].append(prob)
    sentence_scores = [(sum(c) / len(c)) if c else None for c in coverage]
    valid = [s for s in sentence_scores if s is not None]
    doc = (sum(valid) / len(valid)) if valid else 0.0
    return {"document_score": doc, "sentence_scores": sentence_scores}
