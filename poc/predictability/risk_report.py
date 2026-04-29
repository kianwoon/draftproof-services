"""DraftProof Risk Report — authorship / writing-process risk signals.

Not an AI detector. This module produces review-priority signals with
explainable findings. No determination of AI generation or misconduct.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ReviewPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceLevel(Enum):
    INSUFFICIENT = "insufficient"   # < 150 words
    LOW = "low"                     # 150–500 words
    MEDIUM = "medium"               # 500–1500 words
    HIGH = "high"                   # > 1500 words


@dataclass
class AxisScore:
    name: str
    score: float
    confidence: ConfidenceLevel
    reasons: List[str] = field(default_factory=list)


@dataclass
class RiskReport:
    # Multi-axis scores
    predictability_risk: AxisScore
    style_uniformity_risk: AxisScore
    structural_reuse_risk: Optional[AxisScore] = None
    source_grounding_risk: Optional[AxisScore] = None
    draft_evolution_risk: Optional[AxisScore] = None
    citation_integrity_risk: Optional[AxisScore] = None
    authorship_evidence_strength: Optional[AxisScore] = None

    # Overall
    overall_review_priority: ReviewPriority = ReviewPriority.MEDIUM
    review_reasons: List[str] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

    # Disclaimer
    not_a_determination: str = (
        "This is not proof of AI generation or misconduct. "
        "It identifies writing-process risk signals that may warrant review."
    )

    def to_dict(self) -> dict:
        def _axis(a: Optional[AxisScore]) -> Optional[dict]:
            if a is None:
                return None
            return {
                "score": round(a.score, 4),
                "confidence": a.confidence.value,
                "reasons": a.reasons,
            }

        return {
            "axes": {
                "predictability_risk": _axis(self.predictability_risk),
                "style_uniformity_risk": _axis(self.style_uniformity_risk),
                "structural_reuse_risk": _axis(self.structural_reuse_risk),
                "source_grounding_risk": _axis(self.source_grounding_risk),
                "draft_evolution_risk": _axis(self.draft_evolution_risk),
                "citation_integrity_risk": _axis(self.citation_integrity_risk),
                "authorship_evidence_strength": _axis(self.authorship_evidence_strength),
            },
            "overall_review_priority": self.overall_review_priority.value,
            "review_reasons": self.review_reasons,
            "confidence": self.confidence.value,
            "not_a_determination": self.not_a_determination,
        }


# ── Confidence helper ──────────────────────────────────────────────

def word_count_to_confidence(word_count: int) -> ConfidenceLevel:
    if word_count < 150:
        return ConfidenceLevel.INSUFFICIENT
    elif word_count < 500:
        return ConfidenceLevel.LOW
    elif word_count < 1500:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.HIGH


# ── Review priority logic ──────────────────────────────────────────

def compute_review_priority(
    predictability: float,
    style_uniformity: float,
    structural_reuse: Optional[float] = None,
    source_grounding: Optional[float] = None,
    draft_evolution: Optional[float] = None,
    citation_integrity: Optional[float] = None,
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
) -> tuple:
    """Rule-based review priority. Returns (priority, reasons).

    Does NOT average everything equally.
    High structural_reuse + high draft_evolution = automatic HIGH.
    """

    reasons = []

    # Signal collection
    if predictability >= 0.50:
        reasons.append("High sentence predictability")
    if style_uniformity >= 0.40:
        reasons.append("Uniform writing style")
    if structural_reuse is not None and structural_reuse >= 0.60:
        reasons.append("Structural similarity to another draft")
    if source_grounding is not None and source_grounding >= 0.50:
        reasons.append("Weak source grounding")
    if draft_evolution is not None and draft_evolution >= 0.60:
        reasons.append("Surface-level changes between drafts")
    if citation_integrity is not None and citation_integrity >= 0.50:
        reasons.append("Citation gaps or mismatches")

    # Priority determination — rule-based, not averaged
    is_high = False

    # Rule 1: structural reuse + surface evolution = HIGH
    if (structural_reuse is not None and structural_reuse >= 0.60
            and draft_evolution is not None and draft_evolution >= 0.60):
        is_high = True

    # Rule 2: weak source grounding + high predictability = HIGH
    if (source_grounding is not None and source_grounding >= 0.50
            and predictability >= 0.50):
        is_high = True

    # Rule 3: citation gaps + structural reuse = HIGH
    if (citation_integrity is not None and citation_integrity >= 0.50
            and structural_reuse is not None and structural_reuse >= 0.60):
        is_high = True

    # Rule 4: too many independent signals
    high_signals = sum([
        predictability >= 0.50,
        style_uniformity >= 0.40,
        structural_reuse is not None and structural_reuse >= 0.60,
        source_grounding is not None and source_grounding >= 0.50,
    ])
    if high_signals >= 3:
        is_high = True

    # Downgrade if insufficient sample
    if confidence == ConfidenceLevel.INSUFFICIENT:
        priority = ReviewPriority.LOW
        reasons.insert(0, "Text sample too short for reliable analysis")
    elif is_high:
        priority = ReviewPriority.HIGH
    elif len(reasons) >= 2:
        priority = ReviewPriority.MEDIUM
    else:
        priority = ReviewPriority.LOW

    # Override: if confidence is LOW, cap at MEDIUM
    if confidence == ConfidenceLevel.LOW and priority == ReviewPriority.HIGH:
        priority = ReviewPriority.MEDIUM
        reasons.append("Confidence is low due to short sample — recommend manual review")

    return priority, reasons
