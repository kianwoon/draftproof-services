"""Criterion: Polished but ungrounded.

Strong AI-generated pattern: text sounds fluent but has weak grounding.
Combination of high predictability + low specificity + low citation density.
"""

from typing import Dict, List, Optional

from ._types import CriterionScore


def score(
    content: str,
    *,
    predictability_value: float = 0.0,
    specificity_value: float = 0.0,
    citation_findings_count: int = 0,
    **kwargs,
) -> CriterionScore:
    """Score polished-but-ungrounded criterion.

    This is a meta-criterion that combines signals from other criteria.
    The composite classifier passes in the already-computed values.

    Args:
        predictability_value: from surprisal/topk criteria (0-1, higher=more AI-like).
        specificity_value: from specificity criterion (0-1, higher=more AI-like = LESS specific).
        citation_findings_count: number of citation issues found.
    """
    word_count = len(content.split())

    # Polished = high predictability (AI-like), ungrounded = low specificity
    polished = predictability_value >= 0.70
    ungrounded = specificity_value >= 0.60  # high value = low specificity

    if polished and ungrounded:
        value = min(1.0, (predictability_value + specificity_value) / 2)
        label = "high"
    elif polished or ungrounded:
        value = min(1.0, max(predictability_value, specificity_value) * 0.5)
        label = "medium" if value >= 0.35 else "low"
    else:
        value = 0.0
        label = "low"

    return CriterionScore(
        name="polished_but_ungrounded",
        value=round(value, 4),
        label=label,
        details={
            "predictability_value": round(predictability_value, 4),
            "specificity_value": round(specificity_value, 4),
            "citation_findings_count": citation_findings_count,
            "polished": polished,
            "ungrounded": ungrounded,
        },
    )
