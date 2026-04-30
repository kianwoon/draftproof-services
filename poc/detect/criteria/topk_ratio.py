"""Criterion: High top-k token prediction ratio.

If the actual tokens often appear in GPT-2's top-10 or top-50 predictions,
the phrasing is highly predictable — one of the strongest GPT-2-based signals.
"""

from typing import Dict, List, Optional

from ._types import CriterionScore


def score(
    content: str,
    *,
    sentence_metrics: Optional[List[Dict]] = None,
    top10_threshold: float = 0.65,
    top50_threshold: float = 0.80,
    **kwargs,
) -> CriterionScore:
    if not sentence_metrics:
        return CriterionScore(
            name="topk_predictability",
            value=0.0,
            label="low",
            details={"avg_top10": 0.0, "avg_top50": 0.0},
        )

    t10s = [m.get("top_10_ratio", 0.0) for m in sentence_metrics if "top_10_ratio" in m]
    t50s = [m.get("top_50_ratio", 0.0) for m in sentence_metrics if "top_50_ratio" in m]

    avg_top10 = sum(t10s) / max(len(t10s), 1)
    avg_top50 = sum(t50s) / max(len(t50s), 1)

    high_t10_count = sum(1 for t in t10s if t >= top10_threshold)
    high_t10_ratio = high_t10_count / max(len(t10s), 1)

    # Normalise top-10 ratio to 0-1 score
    value = min(1.0, max(0.0, avg_top10 / top10_threshold))

    if avg_top10 >= top10_threshold:
        label = "high"
    elif avg_top10 >= top10_threshold * 0.7:
        label = "medium"
    else:
        label = "low"

    flagged = [
        m.get("sentence", "")
        for m in sentence_metrics
        if m.get("top_10_ratio", 0) >= top10_threshold and m.get("sentence")
    ][:10]

    return CriterionScore(
        name="topk_predictability",
        value=round(value, 4),
        label=label,
        details={
            "avg_top10_ratio": round(avg_top10, 4),
            "avg_top50_ratio": round(avg_top50, 4),
            "high_t10_sentence_ratio": round(high_t10_ratio, 4),
        },
        flagged_excerpts=flagged,
    )
