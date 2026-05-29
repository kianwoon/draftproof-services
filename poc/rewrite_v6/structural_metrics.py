"""Cheap, text-only structural metrics that track the detector's dominant *movable* AI signals.

Empirically (poc/_probe_grounding_lever), a rewrite that drops the document AI score from 71 -> 43
moves these, not token predictability or generic_assertion_risk:
  - repeated_sentence_structure_risk  65 -> 0   (biggest lever)
  - qualifying_text_ai_density        83 -> 62  (hedging)
  - burstiness_risk                   45 -> 25  (sentence-length variation)

These proxies let the selector accept a candidate when it genuinely varies structure, cuts hedging,
and varies length -- forcing the writer to deliver the variation instead of a partway rewrite.
No model load; pure text.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean, pstdev

# Hedging / qualifying words that inflate qualifying_text_ai_density.
_HEDGES = frozenset({
    "may", "might", "could", "would", "should", "can", "perhaps", "possibly", "arguably",
    "often", "sometimes", "generally", "typically", "usually", "likely", "tends", "tend",
    "seems", "seem", "appears", "appear", "relatively", "somewhat", "rather", "fairly",
})
# Low-signal openers skipped when deriving a sentence's structural "frame".
_OPENER_STOPWORDS = frozenset({"the", "a", "an", "and", "but", "or", "so", "yet", "then", "thus"})

# A candidate must cut structural_penalty by at least this much vs the source to count as movement.
STRUCTURAL_MIN_GAIN = 0.12
# Sentence-length coefficient-of-variation at/above this reads as healthy human burstiness.
_HEALTHY_CV = 0.5


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if part.split()]


def _opening_frame(sentence: str) -> str:
    words = re.findall(r"[a-z']+", sentence.casefold())
    significant = [word for word in words if word not in _OPENER_STOPWORDS]
    return " ".join(significant[:2])


def repeated_opening_ratio(text: str) -> float:
    """Fraction of sentences whose opening frame is shared by another sentence (0 good, 1 bad)."""
    sentences = _sentences(text)
    if len(sentences) < 2:
        return 0.0
    frames = [_opening_frame(s) for s in sentences if _opening_frame(s)]
    if not frames:
        return 0.0
    counts = Counter(frames)
    repeated = sum(count for count in counts.values() if count >= 2)
    return repeated / len(sentences)


def hedge_density(text: str) -> float:
    """Hedging/qualifying words per sentence (0 good; >1 reads as AI-style qualification)."""
    sentences = _sentences(text)
    if not sentences:
        return 0.0
    total = sum(
        sum(1 for word in re.findall(r"[a-z']+", s.casefold()) if word in _HEDGES)
        for s in sentences
    )
    return total / len(sentences)


def length_cv(text: str) -> float:
    """Coefficient of variation of sentence word-counts (higher = burstier = more human)."""
    lengths = [len(s.split()) for s in _sentences(text)]
    if len(lengths) < 2:
        return 0.0
    average = mean(lengths)
    return (pstdev(lengths) / average) if average else 0.0


def structural_penalty(text: str) -> float:
    """Lower is more human: penalise repeated openings, hedging, and uniform sentence length."""
    uniformity = max(0.0, _HEALTHY_CV - length_cv(text))
    return repeated_opening_ratio(text) + 0.5 * hedge_density(text) + uniformity


def structural_metrics(text: str) -> dict[str, float]:
    return {
        "repeated_opening_ratio": round(repeated_opening_ratio(text), 3),
        "hedge_density": round(hedge_density(text), 3),
        "length_cv": round(length_cv(text), 3),
        "structural_penalty": round(structural_penalty(text), 3),
    }


def structural_improvement(source_text: str, candidate_text: str, *, min_gain: float = STRUCTURAL_MIN_GAIN) -> bool:
    """True when the candidate is meaningfully more human-structured than the source."""
    return structural_penalty(candidate_text) <= structural_penalty(source_text) - min_gain
