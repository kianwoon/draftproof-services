"""Confidence bands and sample-size awareness for risk scores.

Prevents overconfident verdicts on short texts.
"""

from enum import Enum
from typing import Tuple


class ConfidenceLevel(Enum):
    INSUFFICIENT = "insufficient"   # < 150 words — no verdict
    LOW = "low"                     # 150–500 words
    MEDIUM = "medium"               # 500–1500 words
    HIGH = "high"                   # > 1500 words


def word_count_to_confidence(word_count: int) -> ConfidenceLevel:
    if word_count < 150:
        return ConfidenceLevel.INSUFFICIENT
    elif word_count < 500:
        return ConfidenceLevel.LOW
    elif word_count < 1500:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.HIGH


def confidence_description(level: ConfidenceLevel) -> str:
    descriptions = {
        ConfidenceLevel.INSUFFICIENT:
            "Text sample is too short for reliable analysis. "
            "Sentence-level notes only, no overall verdict.",
        ConfidenceLevel.LOW:
            "Short text sample. Scores are indicative but should not be "
            "treated as conclusive. Recommend manual review.",
        ConfidenceLevel.MEDIUM:
            "Moderate text length. Scores are reasonably reliable but "
            "should be corroborated with other evidence.",
        ConfidenceLevel.HIGH:
            "Adequate text length for style and predictability analysis.",
    }
    return descriptions[level]


def should_suppress_verdict(level: ConfidenceLevel) -> bool:
    """Whether to suppress the overall verdict due to insufficient data."""
    return level == ConfidenceLevel.INSUFFICIENT


def cap_verdict(level: ConfidenceLevel, risk: float) -> Tuple[float, str]:
    """Cap the verdict severity based on confidence.

    Returns (capped_risk, note).
    """
    if level == ConfidenceLevel.INSUFFICIENT:
        return 0.0, "Verdict suppressed — insufficient text"
    elif level == ConfidenceLevel.LOW and risk >= 0.60:
        return 0.55, "Capped from {:.2f} — low confidence due to short sample".format(risk)
    return risk, ""
