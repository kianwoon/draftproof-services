"""Criterion: Sudden style shift between sections.

Detects when one section is significantly more AI-like than surrounding text:
human-like section → suddenly polished section, or high burstiness → low burstiness.

This helps detect partial AI insertion.
"""

import re
import math
from typing import Dict, List, Optional

from ._types import CriterionScore


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _sentence_predictability(sentence: str) -> float:
    """Rough heuristic for sentence predictability without GPT-2.

    Uses known AI-tell patterns: generic starters, formal hedging, abstract language.
    This is a fallback — the composite classifier should pass sentence_metrics when available.
    """
    text = sentence.lower()
    # NO-HARDCODE (L2): the baked AI-marker content-phrase list was removed (emptied to a
    # never-match sentinel, like the sibling criteria). Those transitions (furthermore,
    # moreover, in conclusion...) are common in formal/ESL human writing, so matching them
    # would false-flag humans. The real predictability signal comes from the GPT-2
    # sentence_metrics, which the production DetectionRunner always supplies; this non-model
    # fallback now abstains (returns 0.0) instead of guessing from content words.
    ai_markers: list[str] = []
    count = sum(1 for m in ai_markers if m in text)
    return min(1.0, count * 0.3)


def score(
    content: str,
    *,
    sentence_metrics: Optional[List[Dict]] = None,
    shift_threshold: float = 0.35,
    window_size: int = 5,
    **kwargs,
) -> CriterionScore:
    sentences = _split_sentences(content)
    if len(sentences) < window_size * 2:
        return CriterionScore(
            name="style_shift",
            value=0.0,
            label="low",
            details={"note": "insufficient sentences for shift analysis"},
        )

    # Get per-sentence predictability scores
    if sentence_metrics:
        scores = [m.get("predictability_risk", 0.0) for m in sentence_metrics]
    else:
        scores = [_sentence_predictability(s) for s in sentences]

    # Compute sliding-window averages and find maximum delta
    max_delta = 0.0
    max_delta_pos = 0
    for i in range(window_size, len(scores) - window_size + 1):
        left_avg = sum(scores[max(0, i - window_size):i]) / window_size
        right_avg = sum(scores[i:i + window_size]) / min(window_size, len(scores) - i)
        delta = abs(right_avg - left_avg)
        if delta > max_delta:
            max_delta = delta
            max_delta_pos = i

    value = min(1.0, max_delta / shift_threshold)

    if max_delta >= shift_threshold:
        label = "high"
    elif max_delta >= shift_threshold * 0.6:
        label = "medium"
    else:
        label = "low"

    flagged = []
    if max_delta >= shift_threshold * 0.5 and max_delta_pos < len(sentences):
        start = max(0, max_delta_pos - 2)
        end = min(len(sentences), max_delta_pos + 3)
        flagged = [sentences[j] for j in range(start, end)]

    return CriterionScore(
        name="style_shift",
        value=round(value, 4),
        label=label,
        details={
            "max_section_delta": round(max_delta, 4),
            "shift_position": max_delta_pos,
            "shift_threshold": shift_threshold,
        },
        flagged_excerpts=flagged,
    )
