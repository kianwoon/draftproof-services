"""Centralized threshold configuration for all detectors.

Every threshold that was previously scattered across detector files is now
in one place.  Domain profiles override specific fields as needed.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ThresholdConfig:
    """All tunable thresholds used by the detect pipeline.

    Import and override specific fields per domain:

        tc = ThresholdConfig()
        tc.predictability_high = 0.65   # stricter for this domain
    """

    # ── Predictability scanner ──────────────────────────────────────
    predictability_high: float = 0.60
    predictability_medium: float = 0.45
    predictability_review: float = 0.35

    # Short-text confidence floor
    short_text_word_floor: int = 250
    short_text_sentence_floor: int = 10

    # ── AI-generation likelihood (5-band) ────────────────────────────
    ai_likelihood_minimal: float = 0.20   # < 0.20 → minimal
    ai_likelihood_low: float = 0.40       # 0.20–0.40 → low
    ai_likelihood_moderate: float = 0.60  # 0.40–0.60 → moderate
    ai_likelihood_elevated: float = 0.75  # 0.60–0.75 → elevated
    ai_likelihood_high: float = 1.00      # 0.75+ → high

    # ── Similarity evidence strength ────────────────────────────────
    similarity_strong: float = 0.85
    similarity_moderate: float = 0.65

    # ── Post-processing ─────────────────────────────────────────────
    severity_gap: float = 0.10

    # ── Generic phrase density ──────────────────────────────────────
    generic_phrase_density_threshold: float = 0.30

    # ── Confidence assessment ───────────────────────────────────────
    confidence_word_low: int = 150      # below this → low confidence
    confidence_word_medium: int = 300   # below this → medium confidence

    # ── Predictability weights ──────────────────────────────────────
    predictability_weights: Dict[str, float] = field(default_factory=lambda: {
        "top_10_ratio": 0.30,
        "top_50_ratio": 0.30,
        "surprisal": 0.25,
        "generic_phrases": 0.15,
    })

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "ThresholdConfig":
        """Build from a dict, ignoring unknown keys."""
        if not d:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        # Handle nested dict fields separately
        flat = {}
        for k, v in d.items():
            if k == "predictability_weights" and isinstance(v, dict):
                flat[k] = v
            elif k in known:
                flat[k] = v
        return cls(**flat)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictability_high": self.predictability_high,
            "predictability_medium": self.predictability_medium,
            "predictability_review": self.predictability_review,
            "short_text_word_floor": self.short_text_word_floor,
            "short_text_sentence_floor": self.short_text_sentence_floor,
            "ai_likelihood_high": self.ai_likelihood_high,
            "ai_likelihood_elevated": self.ai_likelihood_elevated,
            "ai_likelihood_moderate": self.ai_likelihood_moderate,
            "ai_likelihood_low": self.ai_likelihood_low,
            "ai_likelihood_minimal": self.ai_likelihood_minimal,
            "similarity_strong": self.similarity_strong,
            "similarity_moderate": self.similarity_moderate,
            "severity_gap": self.severity_gap,
            "generic_phrase_density_threshold": self.generic_phrase_density_threshold,
            "confidence_word_low": self.confidence_word_low,
            "confidence_word_medium": self.confidence_word_medium,
            "predictability_weights": dict(self.predictability_weights),
        }
