"""Rewrite configuration, budget controls, outcomes, and floor detection."""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


# ── Rewrite outcome ──────────────────────────────────────────────────

class RewriteOutcome(Enum):
    IMPROVED = "improved"
    PARTIALLY_IMPROVED = "partially_improved"
    FLOOR_REACHED = "floor_reached"
    MANUAL_REQUIRED = "manual_required"
    REJECTED_FOR_DRIFT = "rejected_for_drift"
    SUGGESTION_ONLY = "suggestion_only"


# ── Rewrite surface ──────────────────────────────────────────────────

@dataclass
class RewriteSurface:
    total_chars: int
    protected_chars: int
    rewritable_chars: int
    rewrite_surface_ratio: float  # rewritable / total

    @property
    def is_mostly_protected(self) -> bool:
        return self.rewrite_surface_ratio < 0.40


# ── Floor detection ──────────────────────────────────────────────────

@dataclass
class FloorReason:
    finding_id: str
    reason_type: str   # "factual" | "protected" | "domain_vocab" | "citation_format" | "standard_phrase"
    explanation: str


# ── Budget ───────────────────────────────────────────────────────────

@dataclass
class RewriteBudget:
    """Limits on how much the rewrite can change."""
    max_passes: int = 2
    max_changed_sentence_ratio: float = 0.20
    max_changed_char_ratio: float = 0.15
    max_llm_tokens: int = 6000
    max_total_changed_sentence_ratio: float = 0.30
    max_total_changed_char_ratio: float = 0.25


# ── Config ───────────────────────────────────────────────────────────

@dataclass
class RewriteConfig:
    """Controls for the rewrite loop."""
    max_passes: int = 2
    max_detect_loops: int = 0
    target_top10: float = 0.50
    min_weighted_improvement: float = 1.0
    max_semantic_drift: float = 0.12
    min_improvement: float = 0.02
    rewrite_mode: str = "targeted"  # "targeted" | "paragraph" | "full"
    require_user_approval: bool = False
    suggestion_only: bool = False  # No LLM available → suggestion-only mode
    budget: RewriteBudget = field(default_factory=RewriteBudget)
    model: Optional[str] = None  # None → resolved by LLMGateway from LLM_MODEL env var
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # None → resolved by LLMGateway (OpenRouter default)
    llm_timeout_seconds: int = 20
    llm_max_retries: int = 1
    max_rewrite_seconds: int = 360
    max_llm_calls: int = 30
    max_auto_targets: int = 8
    max_density_passes: int = 8
    max_failed_targets: int = 4
    max_consecutive_failed_targets: int = 3


# ── Loop control ─────────────────────────────────────────────────────

@dataclass
class LoopDecision:
    """Result of should_continue check."""
    continue_loop: bool
    reason: str


def should_continue(
    prev_weighted_risk: float,
    current_weighted_risk: float,
    pass_no: int,
    config: RewriteConfig,
    semantic_drift: float = 0.0,
    high_risk_findings: int = 0,
) -> LoopDecision:
    """Adaptive loop decision based on multiple signals."""
    if pass_no >= config.max_passes:
        return LoopDecision(False, "max_passes_reached")

    if semantic_drift > config.max_semantic_drift:
        return LoopDecision(False, "semantic_drift_guard")

    if current_weighted_risk > prev_weighted_risk:
        return LoopDecision(False, "risk_regression")

    if high_risk_findings == 0 and current_weighted_risk <= 0:
        return LoopDecision(False, "target_reached")

    improvement = prev_weighted_risk - current_weighted_risk
    if improvement < config.min_weighted_improvement:
        return LoopDecision(False, "plateau")

    return LoopDecision(True, "continue")


# ── Floor detection helpers ──────────────────────────────────────────

def classify_floor(
    sentence: str,
    protected_coverage: float,
    domain_term_density: float = 0.0,
) -> str:
    """Classify why a sentence has reached its predictability floor."""
    import re
    number_count = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', sentence))
    date_count = len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', sentence))

    if number_count + date_count >= 3:
        return "factual"
    if protected_coverage > 0.4:
        return "protected"
    if domain_term_density > 0.25:
        return "domain_vocab"
    return "standard_phrase"


def compute_rewrite_surface(text: str, protected_spans: list) -> RewriteSurface:
    """Calculate how much of the text is actually rewritable."""
    total = len(text)
    protected_chars = sum(s.end_char - s.start_char for s in protected_spans)
    rewritable = max(total - protected_chars, 0)
    ratio = rewritable / max(total, 1)
    return RewriteSurface(
        total_chars=total,
        protected_chars=protected_chars,
        rewritable_chars=rewritable,
        rewrite_surface_ratio=ratio,
    )
