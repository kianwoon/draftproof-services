"""Criterion: Uniform paragraph structure.

AI writing often produces clean, balanced paragraphs with similar lengths
and claim → explanation → conclusion patterns.
"""

import re
import math
from typing import Dict, List

from ._types import CriterionScore


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split('\n\n') if p.strip()]


def score(
    content: str,
    *,
    uniformity_cv_threshold: float = 0.30,
    **kwargs,
) -> CriterionScore:
    paragraphs = _split_paragraphs(content)
    if len(paragraphs) < 3:
        return CriterionScore(
            name="paragraph_uniformity",
            value=0.0,
            label="low",
            details={"note": "fewer than 3 paragraphs"},
        )

    # Paragraph length variance (in words)
    lengths = [len(p.split()) for p in paragraphs]
    mean_l = sum(lengths) / len(lengths)
    if mean_l == 0:
        return CriterionScore(
            name="paragraph_uniformity",
            value=0.0,
            label="low",
            details={"avg_paragraph_words": 0},
        )

    variance = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
    cv = math.sqrt(variance) / mean_l  # coefficient of variation

    # Low CV = uniform = AI-like
    # cv > 0.8 is very varied (human, value=0), cv < 0.2 is very uniform (AI, value=1)
    value = max(0.0, min(1.0, 1.0 - (cv - 0.2) / 0.6))

    if cv < uniformity_cv_threshold:
        label = "high"
    elif cv < uniformity_cv_threshold * 1.5:
        label = "medium"
    else:
        label = "low"

    # Check for formulaic paragraph starters
    first_words = [p.split()[0].lower() for p in paragraphs if p.split()]
    fw_counts: Dict[str, int] = {}
    for w in first_words:
        fw_counts[w] = fw_counts.get(w, 0) + 1
    max_starter_ratio = max(fw_counts.values()) / len(first_words) if first_words else 0.0

    return CriterionScore(
        name="paragraph_uniformity",
        value=round(value, 4),
        label=label,
        details={
            "paragraph_count": len(paragraphs),
            "avg_paragraph_words": round(mean_l, 1),
            "paragraph_length_cv": round(cv, 4),
            "max_starter_ratio": round(max_starter_ratio, 4),
        },
    )
