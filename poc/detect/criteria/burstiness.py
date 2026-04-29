"""Criterion: Low burstiness (sentence-level predictability variance).

Human writing has uneven rhythm — some sentences are simple, some dense,
some short.  AI writing is often smoother and more uniform.

Burstiness = std(sentence_surprisal_scores).  Low burstiness = AI-like.
"""

import math
from typing import Dict, List, Optional

from ._types import CriterionScore


def score(
    content: str,
    *,
    sentence_metrics: Optional[List[Dict]] = None,
    burstiness_threshold: float = 0.35,
    min_word_count: int = 300,
    **kwargs,
) -> CriterionScore:
    word_count = len(content.split())

    if not sentence_metrics or len(sentence_metrics) < 3:
        return CriterionScore(
            name="low_burstiness",
            value=0.0,
            label="low",
            details={
                "burstiness": 0.0,
                "word_count": word_count,
                "note": "insufficient sentences for burstiness analysis",
            },
        )

    surprisals = [m.get("avg_surprisal", 3.0) for m in sentence_metrics if "avg_surprisal" in m]
    if len(surprisals) < 3:
        return CriterionScore(
            name="low_burstiness",
            value=0.0,
            label="low",
            details={"burstiness": 0.0, "word_count": word_count},
        )

    mean_s = sum(surprisals) / len(surprisals)
    variance = sum((s - mean_s) ** 2 for s in surprisals) / len(surprisals)
    std = math.sqrt(variance)

    # Low std = low burstiness = more AI-like
    # Normalise: std > 1.0 is very bursty (human, value=0), std < 0.2 is very flat (AI, value=1)
    burstiness_score = max(0.0, min(1.0, 1.0 - (std - 0.2) / 0.8))

    # Only flag if text is long enough for burstiness to be meaningful
    if word_count < min_word_count:
        label = "low"
        burstiness_score *= 0.5  # reduce confidence for short texts
    elif std < burstiness_threshold:
        label = "high"
    elif std < burstiness_threshold * 1.5:
        label = "medium"
    else:
        label = "low"

    return CriterionScore(
        name="low_burstiness",
        value=round(burstiness_score, 4),
        label=label,
        details={
            "surprisal_std": round(std, 4),
            "surprisal_mean": round(mean_s, 4),
            "word_count": word_count,
            "burstiness_threshold": burstiness_threshold,
        },
    )
