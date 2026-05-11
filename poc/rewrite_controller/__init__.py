"""Shared rewrite controller primitives."""

from .budget import (
    RewriteRunBudget,
    cap_phase_seconds_for_reserve,
    post_ai_search_reserve_seconds,
    resolve_global_rewrite_seconds,
)
from .quality_gate import evaluate_text_quality_regression
from .selector import CandidateLedger, build_candidate_record
from .ai_search_selection import (
    ai_search_candidate_rank,
    build_candidate_decision,
    classify_ai_search_candidate,
    CandidateDecision,
    detector_progress_rank,
)
from .formula_gap_orchestrator import (
    assemble_candidate_from_payload as assemble_formula_gap_candidate,
    block_portfolio_tasks as formula_gap_block_portfolio_tasks,
    budget_contract as formula_gap_budget_contract,
    extract_candidate_payload as extract_formula_gap_candidate_payload,
    formula_gap_candidate_prompt,
    formula_gap_plan,
    named_entity_inventory as formula_gap_named_entity_inventory,
    portfolio_families as formula_gap_portfolio_families,
)
from .eligible_span_density import (
    build_eligible_span_density_contract,
    compare_eligible_span_density,
)
from .segment_window_density import (
    SEGMENT_WINDOW_CONTROLLER_VERSION,
    assemble_segment_window_candidate,
    build_segment_density_windows,
    extract_segment_window_payload,
    is_canonical_fact_sentence as segment_window_is_canonical_fact_sentence,
    segment_patchwork_budget,
    segment_window_candidate_prompt,
    segment_window_tasks,
)

__all__ = [
    "CandidateLedger",
    "CandidateDecision",
    "RewriteRunBudget",
    "cap_phase_seconds_for_reserve",
    "post_ai_search_reserve_seconds",
    "resolve_global_rewrite_seconds",
    "ai_search_candidate_rank",
    "build_candidate_record",
    "build_candidate_decision",
    "classify_ai_search_candidate",
    "detector_progress_rank",
    "evaluate_text_quality_regression",
    "assemble_formula_gap_candidate",
    "extract_formula_gap_candidate_payload",
    "formula_gap_block_portfolio_tasks",
    "formula_gap_budget_contract",
    "formula_gap_candidate_prompt",
    "formula_gap_named_entity_inventory",
    "formula_gap_plan",
    "formula_gap_portfolio_families",
    "build_eligible_span_density_contract",
    "compare_eligible_span_density",
    "SEGMENT_WINDOW_CONTROLLER_VERSION",
    "assemble_segment_window_candidate",
    "build_segment_density_windows",
    "extract_segment_window_payload",
    "segment_patchwork_budget",
    "segment_window_candidate_prompt",
    "segment_window_is_canonical_fact_sentence",
    "segment_window_tasks",
]
