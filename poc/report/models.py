"""Report data models — tiers and dataclasses.

Extracted from report.py. These are the pure data structures shared across
the report builder, serializer, and external consumers.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


# ── Risk tiers ──────────────────────────────────────────────────────

class Tier(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CLEAN = "clean"


# ── AI policy context (Phase 1, docs/plans/policy_risk_external_review_response.md) ──
# ASSIGNMENT-level (not institution-wide) AI policy, optionally captured at scan
# submission. "unknown" is the default and the common case today -- nothing infers
# this from the writing. Frozen set, not a scoring input: never used to gate/allow
# anything, purely a presentation-selection input for a not-yet-built Phase 2.
AI_POLICY_VALUES = frozenset({
    "prohibited", "editing_only", "allowed_with_declaration", "collaboration_allowed", "unknown",
})

TIER_ORDER = [Tier.CRITICAL, Tier.HIGH, Tier.MEDIUM, Tier.LOW, Tier.CLEAN]
TIER_ICON = {
    Tier.CRITICAL: "[!]",
    Tier.HIGH: "[H]",
    Tier.MEDIUM: "[M]",
    Tier.LOW: "[L]",
    Tier.CLEAN: "[=]",
}

_RISK_LEVEL_TO_TIER = {
    "high": Tier.HIGH,
    "medium": Tier.MEDIUM,
    "low": Tier.LOW,
}


# ── Findings (unified across scanners) ──────────────────────────────

@dataclass
class Finding:
    """Single actionable finding from any scanner."""
    tier: Tier
    category: str
    scanner: str
    title: str
    detail: str
    evidence: str
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    finding_id: str = ""
    raw_risk: str = ""
    adjusted_risk: str = ""
    sentence_id: str = ""
    signal_category: str = ""   # "writing_quality" | "genericity" | "predictability" | "authorship_risk"


# ── Section summaries ───────────────────────────────────────────────

@dataclass
class PredictabilitySummary:
    overall_risk: float
    risk_distribution: Dict[str, int]
    sentences: List[Dict[str, Any]]
    style_shifts: List[Dict[str, Any]]
    generic_phrases_found: List[str]


@dataclass
class SimilaritySummary:
    overall_risk: str
    risk_distribution: Dict[str, int]
    matches: List[Dict[str, Any]]


@dataclass
class SemanticShapeSummary:
    model_name: str
    embedding_model_attached: bool
    sentence_count: int
    paragraph_count: int
    adjacent_similarity_mean: float
    adjacent_similarity_std: float
    paragraph_similarity_mean: float
    paragraph_similarity_std: float
    semantic_uniformity_risk: float
    discourse_regularity_risk: float
    semantic_drift_risk: float


@dataclass
class CitationSummary:
    citation_style: str
    in_text_count: int
    bib_entry_count: int
    findings: List[Dict[str, Any]]
    stats: Dict[str, int]


@dataclass
class RewriteSummary:
    original_risk: float
    original_top10: float
    final_risk: float
    final_top10: float
    passes_completed: int
    converged: bool
    convergence_reason: str
    improvement_risk: float
    improvement_top10: float
    pass_progression: List[Dict[str, float]]
    original_tier: str = ""
    original_findings: int = 0
    original_distribution: Dict[str, int] = None
    rewritten_tier: str = ""
    rewritten_findings: int = 0
    rewritten_distribution: Dict[str, int] = None
    sentence_comparison: List[Dict[str, Any]] = None
    # Post-rewrite detect verification
    post_detect_findings: int = 0
    post_detect_distribution: Dict[str, int] = None
    post_detect_improvement: int = 0
    detect_loops_used: int = 0
    detect_loop_history: List[Dict[str, Any]] = None
    reverted: bool = False
    revert_reason: str = ""
    # New multi-signal fields
    weighted_risk_original: float = 0.0
    weighted_risk_final: float = 0.0
    drift_score: float = 0.0
    protected_spans_count: int = 0
    regression_rejections: int = 0
    auto_fixable_count: int = 0
    manual_required_count: int = 0
    protected_count: int = 0
    manual_actions: List[Dict[str, str]] = None
    protected_actions: List[Dict[str, str]] = None
    # Fixability-aware scoring
    rewritable_risk_original: float = 0.0
    rewritable_risk_final: float = 0.0
    # Outcome classification
    outcome: str = ""  # "improved" | "partially_improved" | "floor_reached" | "manual_required" | "rejected_for_drift"
    floor_reasons: List[Dict[str, str]] = None
    # Voice & surface
    voice_preserved: bool = True
    voice_warnings: List[str] = None
    rewrite_surface_ratio: float = 1.0
    # Detect scan scores (same as shown in scan report)
    detect_ai_likelihood: float = 0.0      # AI Generation Likelihood from detect scan
    detect_writing_quality: float = 0.0    # Writing Quality Risk from detect scan


# ── Main report ─────────────────────────────────────────────────────

@dataclass
class DraftReport:
    """Structured report from a full DraftProof scan."""
    overall_tier: Tier
    finding_count: int
    findings_by_tier: Dict[str, List[Finding]]

    predictability: Optional[PredictabilitySummary] = None
    similarity: Optional[SimilaritySummary] = None
    semantic_shape: Optional[SemanticShapeSummary] = None
    citation: Optional[CitationSummary] = None
    rewrite: Optional[RewriteSummary] = None
    manual_actions: List[Dict[str, str]] = None

    scan_time_seconds: float = 0.0
    generated_at: str = ""
    original_text: str = ""
    rewritten_text: str = ""
    false_positives: Optional[List[Dict[str, str]]] = None
    raw_overall_tier: str = ""
    adjusted_overall_tier: str = ""
    # Cached canonical-sentence DeBERTa heatmap. Shared between the tile headline and the
    # Signal-highlights map so both use IDENTICAL per-sentence scores (no splitter mismatch).
    # Populated in build(); read in report_to_dict()._compute_deberta_heatmap. None when the
    # signal is off/unavailable (the map falls back to the legacy perplexity color path).
    deberta_heatmap: Optional[Dict[str, Any]] = None
    overall_tier_reason: str = ""
    rewrite_priority_tier: str = ""
    rewrite_priority_reason: str = ""
    rewrite_decision: Optional[Dict[str, Any]] = None
    actionability_distribution: Optional[Dict[str, int]] = None
    axis_scores: Optional[Dict[str, str]] = None
    reason_codes: List[str] = field(default_factory=list)
    authorship_concern_score: float = 0.0
    authorship_concern_confidence: str = "low"
    authorship_concern_signals: Optional[Dict[str, Any]] = None
    ai_risk_badge: Optional[Dict[str, Any]] = None
    paragraph_explanations: Optional[Dict[str, Any]] = None
    # Phase 1 (docs/plans/policy_risk_external_review_response.md): optional
    # assignment-level AI policy captured at scan submission. None when absent
    # (older reports, or the submitter didn't pick one) -- report_to_dict()
    # normalizes that to "unknown" in document_context, never fabricates a choice.
    ai_policy: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize the report to a JSON-ready dict."""
        # Late import to avoid a circular dependency (report.py imports models).
        from report.report import report_to_dict
        plan = report_to_dict(self)
        plan["original_text"] = self.original_text
        plan["rewritten_text"] = self.rewritten_text
        return plan
