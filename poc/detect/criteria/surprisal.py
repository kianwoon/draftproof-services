"""Criterion: Low token surprisal.

AI-generated text often chooses statistically safe words, producing low
average surprisal under a reference language model (GPT-2).

Consumes pre-computed sentence-level metrics from PredictabilityScanner
via ``sentence_metrics`` kwarg to avoid double GPT-2 inference.
"""

import re
from typing import Dict, List, Optional

from ._types import CriterionScore


def score(
    content: str,
    *,
    sentence_metrics: Optional[List[Dict]] = None,
    surprisal_threshold: float = 2.5,
    **kwargs,
) -> CriterionScore:
    """Score low-surprisal criterion.

    Args:
        content: full text.
        sentence_metrics: list of per-sentence dicts with ``avg_surprisal`` key
            (from PredictabilityScanner output).  If absent, the criterion
            returns a zero-score placeholder.

    Returns:
        CriterionScore with value 0-1 where higher = more AI-like.
    """
    if not sentence_metrics:
        return CriterionScore(
            name="low_surprisal",
            value=0.0,
            label="low",
            details={"avg_surprisal": 0.0, "threshold": surprisal_threshold},
        )

    surprisals = [m.get("avg_surprisal", 5.0) for m in sentence_metrics if "avg_surprisal" in m]
    if not surprisals:
        return CriterionScore(
            name="low_surprisal",
            value=0.0,
            label="low",
            details={"avg_surprisal": 0.0, "threshold": surprisal_threshold},
        )

    avg_surprisal = sum(surprisals) / len(surprisals)
    low_count = sum(1 for s in surprisals if s < surprisal_threshold)
    low_ratio = low_count / len(surprisals)

    # Normalise: surprisal 5+ is very human-like (value=0), <1.5 is very AI-like (value=1)
    surprisal_score = max(0.0, min(1.0, 1.0 - (avg_surprisal - 1.5) / 3.5))

    if low_ratio >= 0.6 and avg_surprisal < surprisal_threshold:
        label = "high"
    elif low_ratio >= 0.35:
        label = "medium"
    else:
        label = "low"

    flagged = [
        m.get("sentence", "")
        for m in sentence_metrics
        if m.get("avg_surprisal", 99) < surprisal_threshold and m.get("sentence")
    ][:10]

    return CriterionScore(
        name="low_surprisal",
        value=round(surprisal_score, 4),
        label=label,
        details={
            "avg_surprisal": round(avg_surprisal, 4),
            "low_surprisal_ratio": round(low_ratio, 4),
            "threshold": surprisal_threshold,
        },
        flagged_excerpts=flagged,
    )
