"""DraftProof Rewrite — content rewriting module.

Accepts content text + REQUIRED detection findings, produces rewritten text
with reduced predictability risk AND a rendered report.
"""

import sys
import os

_here = os.path.dirname(__file__)
_parent = os.path.join(_here, "..")
sys.path.insert(0, os.path.join(_parent, "rewriter"))
sys.path.insert(0, os.path.join(_parent, "predictability"))
sys.path.insert(0, _parent)

from rewriter import (
    RewritableSpan,
    RewriteResult,
    PassMetrics,
    MultiPassResult,
    compute_metrics,
    rewrite_text,
    multi_pass_rewrite,
)
from style_analyzer import StyleAnalyzer, StyleProfile
from .rewrite import RewriteModuleResult, run_rewrite, get_rewrite_summary
from .config import (
    RewriteConfig, RewriteBudget, LoopDecision, should_continue,
    RewriteOutcome, RewriteSurface, FloorReason, classify_floor,
    compute_rewrite_surface,
)
from .planner import (
    RewritePlanner, RewritePlan, RewriteAction,
    route_finding, FixabilityDecision,
    FINDING_ROUTING, EDIT_RADIUS, REWRITE_SCOPE,
    FIXABILITY_AUTO, FIXABILITY_PARTIAL, FIXABILITY_MANUAL, FIXABILITY_PROTECTED,
)
from .guards import (
    detect_protected_spans, check_semantic_drift, DriftCheck,
    RegressionMemory, mask_protected_spans,
    protected_spans_preserved, affected_region, transactional_apply, TransactionResult,
)
from .scorer import (
    weighted_finding_score, weighted_rewritable_risk,
    score_candidate, best_candidate, CandidateScore,
    RISK_WEIGHTS, FIXABILITY_WEIGHT,
)
from .voice import VoiceGuard, VoiceProfile, analyze_voice
from .parse_detect import DetectJSONContext, DetectJSONParser, findings_from_json

__all__ = [
    "RewritableSpan",
    "RewriteResult",
    "PassMetrics",
    "MultiPassResult",
    "RewriteModuleResult",
    "StyleAnalyzer",
    "StyleProfile",
    "compute_metrics",
    "rewrite_text",
    "multi_pass_rewrite",
    "run_rewrite",
    "get_rewrite_summary",
    # Config
    "RewriteConfig", "RewriteBudget", "LoopDecision", "should_continue",
    "RewriteOutcome", "RewriteSurface", "FloorReason",
    # Planner
    "RewritePlanner", "RewritePlan", "RewriteAction", "FixabilityDecision",
    # Guards
    "DriftCheck", "RegressionMemory", "TransactionResult",
    "transactional_apply", "affected_region",
    # Scorer
    "CandidateScore", "score_candidate", "best_candidate",
    "weighted_finding_score", "weighted_rewritable_risk",
    # Voice
    "VoiceGuard", "VoiceProfile", "analyze_voice",
    # Parse Detect JSON
    "DetectJSONContext", "DetectJSONParser", "findings_from_json",
]
