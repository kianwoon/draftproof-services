"""DraftProof Detect — shared base classes for all detectors.

Every detector accepts content (text) and returns a DetectResult.
Aggregated results are wrapped in a DetectionReport.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from abc import ABC, abstractmethod


@dataclass
class Finding:
    """Single finding from any detector."""
    finding_type: str
    risk_level: str              # "critical" | "high" | "medium" | "low" | "review" | "info"
    evidence_strength: str       # "weak" | "moderate" | "strong" — how well supported
    detail: str                  # Human-readable explanation
    evidence: str                # The flagged text excerpt
    recommendation: str          # Suggested fix (free text)
    suggested_action_type: str   # Machine-readable action category
    location: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_category: str = ""    # "writing_quality" | "genericity" | "predictability" | "authorship_risk"
    actionability: str = ""      # "auto_rewrite_candidate" | "citation_repair" | "optional_structure_review" | "review_only" | "no_action" | "manual_required"


@dataclass
class DetectResult:
    """Unified result from any detector."""
    scanner: str                 # "predictability" | "ai_generation" | "similarity" | "citation"
    overall_risk: float          # 0.0–1.0 normalized
    confidence: str              # "low" | "medium" | "high"
    confidence_reason: str       # Why confidence is at this level
    risk_distribution: Dict[str, int] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    policy_message: str = ""     # Caveat the report should surface
    raw: Any = None              # Original detailed output for deep inspection

    # Detector provenance
    detector_version: str = "0.1.0"
    model_name: str = ""
    config_hash: str = ""

    # AI-generation classification fields (used by AIGenerationSignalDetector)
    classification_target: str = ""               # e.g. "ai_generated_content"
    likelihood_score: float = 0.0                 # 0.0–1.0 composite score
    feature_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class DetectionReport:
    """Aggregate report across all detectors."""
    overall_review_priority: str  # "low" | "medium" | "high"
    overall_risk: float           # 0.0–1.0 rule-based (not averaged)
    confidence: str               # "low" | "medium" | "high"
    scanner_results: List[DetectResult] = field(default_factory=list)
    top_findings: List[Finding] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    summary: str = ""
    postprocess_results: List = field(default_factory=list)  # List[PostProcessResult]
    # Rewrite handoff — detection owns the rewrite decision
    rewrite_decision: Any = None  # RewriteDecision (set after aggregation)
    actionability_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class RewriteDecision:
    """Detection tells the orchestrator whether rewriting is justified."""
    run_rewrite: bool
    reason: str                    # Human-readable explanation
    mode: str                      # "targeted" | "full" | "none"
    targets: List[str] = field(default_factory=list)   # Finding IDs to rewrite
    full_rewrite_allowed: bool = False
    allowed_actions: List[str] = field(default_factory=list)


class BaseDetector(ABC):
    """Protocol every detector must implement."""

    @abstractmethod
    def detect(self, content: str, **kwargs) -> DetectResult:
        """Run detection on content text. Returns unified DetectResult."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def detector_version(self) -> str:
        return "0.1.0"

    @property
    def policy_message(self) -> str:
        return ""

    def _assess_confidence(self, content: str, **kwargs) -> tuple:
        """Default confidence assessment. Override in subclasses.

        Returns:
            (confidence_level, reason)
        """
        word_count = len(content.split())
        if word_count < 50:
            return ("low", "Text sample has fewer than 50 words; analysis is unreliable.")
        if word_count < 150:
            return ("low", "Text sample has fewer than 150 words; sentence-level signals are unstable.")
        if word_count < 300:
            return ("medium", "Text sample is moderately sized; some signals may be underpowered.")
        return ("high", "Text sample is sufficient for reliable analysis.")


# ── Signal category classifier ────────────────────────────────────────

_SIGNAL_CATEGORY_MAP = {
    # Writing quality — grammar, fragments, surface issues
    "grammar_issue": "writing_quality",
    "fragment_sentence": "writing_quality",
    "spelling_issue": "writing_quality",
    "punctuation_issue": "writing_quality",
    # Genericity — vague, broad, template phrasing
    "generic_phrase": "genericity",
    "generic_policy_claim": "genericity",
    "broad_education_claim": "genericity",
    "formulaic_conclusion": "genericity",
    "template_personal_reflection": "genericity",
    "weak_source_grounding": "genericity",
    # Predictability — statistical token-level signals
    "high_predictability": "predictability",
    "medium_predictability": "predictability",
    "low_predictability": "predictability",
    "review_predictability": "predictability",
    "high_topk_predictability": "predictability",
    "style_shift": "predictability",
    "low_burstiness": "predictability",
    "repetitive_sentence_structure": "predictability",
    # Authorship risk — AI-likeness, unsupported claims, low draft evidence
    "high_ai_generation_likelihood": "authorship_risk",
    "medium_ai_generation_likelihood": "authorship_risk",
    "low_ai_generation_likelihood": "authorship_risk",
    "low_specificity": "genericity",
    "uncited_claim": "writing_quality",
    "missing_citation": "writing_quality",
    "broken_citation": "writing_quality",
    "similarity_overlap": "authorship_risk",
}


def classify_finding_category(finding_type: str) -> str:
    """Return signal category for a finding_type."""
    return _SIGNAL_CATEGORY_MAP.get(finding_type, "predictability")
