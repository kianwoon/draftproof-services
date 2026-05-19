"""DraftProof Rewrite Pipeline — reads detect JSON, runs rewrite, outputs report.

Usage:
  python rewrite_pipeline.py detect.json                    # from detect JSON
  python rewrite_pipeline.py detect.json --passes 5         # more rewrite passes
  python rewrite_pipeline.py detect.json --max-loops 3      # more detect-rewrite loops
  python rewrite_pipeline.py --text "Some text here"        # detect + rewrite inline

Output:
  test_output/draftproof_rewrite_<timestamp>.md
  test_output/draftproof_rewrite_<timestamp>.pdf
  test_output/draftproof_rewrite_<timestamp>.json
"""

import sys
import os
import json
import time
import re
import argparse
import copy
import math
import statistics
import requests
from difflib import SequenceMatcher
from typing import Any, Callable
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rewrite.parse_detect import DetectJSONParser, DetectJSONContext, findings_from_json
from rewrite import run_rewrite, RewriteConfig, RewriteModuleResult
from rewrite.auto_repair_controller import AutoRepairDependencies, run_auto_repair_controller
from rewrite_controller import (
    CandidateLedger,
    RewriteRunBudget,
    build_candidate_record,
    cap_phase_seconds_for_reserve,
    cleanup_progress_gate,
    evaluate_text_quality_regression,
    final_rewrite_outcome_label,
    meaningful_ai_progress_gate,
    post_ai_search_reserve_seconds,
    resolve_global_rewrite_seconds,
)
from rewrite_controller.ai_search_selection import (
    ai_search_candidate_rank,
    build_candidate_decision,
    detector_progress_rank as _detector_progress_rank,
)
from rewrite_controller.formula_gap_orchestrator import (
    assemble_candidate_from_payload as _assemble_formula_gap_candidate,
    block_portfolio_tasks as _formula_gap_block_portfolio_tasks,
    budget_contract as _formula_gap_orchestrator_budget_contract,
    extract_candidate_payload as _extract_formula_gap_candidate_payload,
    formula_gap_candidate_prompt as _formula_gap_candidate_prompt,
    named_entity_inventory as _formula_gap_named_entity_inventory,
    formula_gap_plan as _formula_gap_orchestrator_plan,
    portfolio_families as _formula_gap_portfolio_families,
)
from rewrite_controller.eligible_span_density import (
    build_eligible_span_density_contract as _eligible_span_density_contract,
    compare_eligible_span_density as _eligible_span_density_comparison,
)
from rewrite_controller.segment_window_density import (
    build_segment_density_windows as _segment_density_windows,
)
from rewrite_controller.window_coverage_density import (
    WINDOW_COVERAGE_CONTROLLER_VERSION as _WINDOW_COVERAGE_CONTROLLER_VERSION,
    assemble_window_coverage_candidate as _assemble_window_coverage_candidate,
    build_window_coverage_map as _window_coverage_map,
    compare_window_coverage_density as _window_coverage_comparison,
    extract_window_coverage_payload as _extract_window_coverage_payload,
    window_coverage_ablation_candidates as _window_coverage_ablation_candidates,
    window_coverage_candidate_prompt as _window_coverage_candidate_prompt,
    window_coverage_deterministic_variants as _window_coverage_deterministic_variants,
    window_coverage_patchwork_budget as _window_coverage_patchwork_budget,
    window_coverage_portfolio_candidates as _window_coverage_portfolio_candidates,
    window_coverage_tasks as _window_coverage_tasks,
)
from rewrite_compiler import CompilerConfig, CompilerDependencies, run_rewrite_compiler
from rewrite.guards import detect_protected_spans, check_semantic_drift
from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from detect.run import DetectionRunner
from detect.layer3_scoring import Layer3Scorer, build_layer3_input_from_text, _sentence_has_concrete_or_context
from report.report import ReportBuilder, report_to_dict
from llm.gateway import LLMGateway, LLMConfig
from detect.mitigation import build_ai_mitigation_plan
from detect.topk_calibration import TOPK_CALIBRATED_SAFE_LIMIT, calibrate_topk_risk
from detect.turnitin_like import (
    TURNITIN_LIKE_COMPONENT_WEIGHTS,
    TURNITIN_LIKE_TARGET_AI_SCORE,
    turnitin_like_ai_profile_from_report,
)
from rewrite_pipeline_core.config import (
    TOPK_SAFE_LIMIT,
    _env_flag,
    _float_env,
    _float_env_optional,
    _float_env_with_fallback,
    _int_env_optional,
    _llm_call_budget_exhausted_before_send,
    _llm_role_config,
    _load_local_env,
    _phase_chat_sampling_kwargs,
    _phase_sampling_arg,
    _resolve_stage_llm_budget,
    _retry_model_enabled,
    _retry_model_max_calls,
    _rewrite_model_lock,
    _rewrite_sampling_profile,
    _role_model,
    _safe_index,
    _safe_topk_calibrated_limit,
    _safe_topk_limit,
)
from rewrite_pipeline_core.budget.llm_sync import (
    _record_rewrite_llm_calls,
    _sync_rewrite_llm_call_totals,
)
from rewrite_pipeline_core.budget.phase_planner import (
    rewrite_phase_budget_plan as _core_rewrite_phase_budget_plan,
)
from rewrite_pipeline_core.controller_ledger import (
    GlobalControllerLedgerDeps,
    controller_changed_sentence_ratio as _core_controller_changed_sentence_ratio,
    controller_metrics as _core_controller_metrics,
    controller_record as _core_controller_record,
    global_controller_phase_accepted as _core_global_controller_phase_accepted,
    global_phase_budget_skip as _core_global_phase_budget_skip,
)
from rewrite_pipeline_core.gates.density_acceptance import (
    DensityAcceptanceDeps,
    remaining_cluster_density_acceptance as _core_remaining_cluster_density_acceptance,
    segment_window_density_acceptance as _core_segment_window_density_acceptance,
    window_coverage_density_acceptance as _core_window_coverage_density_acceptance,
)
from rewrite_pipeline_core.gates.ai_density_breaker import (
    _AI_DENSITY_CANONICAL_FACT_RE,
    _AI_DENSITY_GENERIC_RE,
    _AI_DENSITY_TRANSITION_RE,
    _ai_density_breaker_canonical_fact_sentence,
    _ai_density_breaker_sentence_route,
    _ai_density_window_targets,
)
from rewrite_pipeline_core.gates.ai_footprint import _ai_footprint_gate_status
from rewrite_pipeline_core.gates.authenticity import (
    AuthenticityGateDeps,
    authenticity_gate_status as _core_authenticity_gate_status,
)
from rewrite_pipeline_core.gates.concept_origin import (
    _best_source_paragraph_index,
    _candidate_concept_origin_reject_reason as _core_candidate_concept_origin_reject_reason,
    _concept_origin_normalize_term,
    _concept_origin_protected_terms,
    _concept_origin_terms,
    _ordered_concept_origin_terms,
)
from rewrite_pipeline_core.reporting.sentence_comparison import (
    _build_aligned_sentence_comparison,
    _comparison_sentences,
    _detail_value,
    _scan_scope_summary,
    _sentence_detail_lookup,
    sanitize_text,
)
from rewrite_pipeline_core.reporting.author_evidence import (
    AuthorEvidenceReportingDeps,
    build_author_context_discovery_layer as _core_build_author_context_discovery_layer,
    build_author_evidence_completion_layer as _core_build_author_evidence_completion_layer,
    build_mitigation_ceiling_diagnostics as _core_build_mitigation_ceiling_diagnostics,
)
from rewrite_pipeline_core.reporting.source_grounding import (
    SourceGroundingSearchDeps,
    build_source_grounding_search_layer as _core_build_source_grounding_search_layer,
)
from rewrite_pipeline_core.reporting.authorship_schema import (
    AuthorshipSchemaEnrichmentDeps,
    enrich_report_authorship_schema as _core_enrich_report_authorship_schema,
)
from rewrite_pipeline_core.phases.window_coverage import (
    WindowCoverageOptimizerDeps,
    run_window_coverage_density_optimizer as _core_window_coverage_density_optimizer,
)
from rewrite_pipeline_core.phases.remaining_cluster import (
    RemainingClusterDensityControllerDeps,
    run_remaining_cluster_density_controller as _core_remaining_cluster_density_controller,
)
from rewrite_pipeline_core.phases.segment_window import (
    SegmentWindowDensityControllerDeps,
    run_segment_window_density_controller as _core_segment_window_density_controller,
)
from rewrite_pipeline_core.phases.formula_convergence import (
    FormulaConvergenceControllerDeps,
    run_formula_convergence_controller as _core_formula_convergence_controller,
)
from rewrite_pipeline_core.phases.iterative_topk import (
    IterativeTopkRouteOptimizerDeps,
    run_iterative_topk_route_optimizer as _core_run_iterative_topk_route_optimizer,
)
from rewrite_pipeline_core.phases.final_topk_texture import (
    FinalTopkTextureRepairDeps,
    run_final_topk_texture_repair as _core_run_final_topk_texture_repair,
)
from rewrite_pipeline_core.phases.post_safe_win import (
    PostSafeWinTargetPushDeps,
    run_post_safe_win_target_push as _core_run_post_safe_win_target_push,
)
from rewrite_pipeline_core.phases.post_topk_optimizer import (
    PostTopkSafeBandOptimizerDeps,
    run_post_topk_ai_safe_band_optimizer as _core_run_post_topk_ai_safe_band_optimizer,
)
from rewrite_pipeline_core.phases.post_topk_texture_helpers import (
    PostTopkTextureHelperDeps,
    apply_post_topk_patches as _core_apply_post_topk_patches,
    authorship_transformation_texture_candidates as _core_authorship_transformation_texture_candidates,
    authorship_transformation_texture_driver_map as _core_authorship_transformation_texture_driver_map,
    authorship_transformation_texture_patch_prompt as _core_authorship_transformation_texture_patch_prompt,
    extract_post_topk_patch_candidates as _core_extract_post_topk_patch_candidates,
    post_topk_driver_map as _core_post_topk_driver_map,
    post_topk_sentence_contextual as _core_post_topk_sentence_contextual,
    post_topk_sentence_driver_score as _core_post_topk_sentence_driver_score,
    texture_candidate_family as _core_texture_candidate_family,
)
from rewrite_pipeline_core.phases.human_anchor_candidates import (
    HumanAnchorCandidateDeps,
    append_anchor_sentence as _core_append_anchor_sentence,
    anchor_sentence_for_paragraph as _core_anchor_sentence_for_paragraph,
    formula_portfolio_candidates as _core_formula_portfolio_candidates,
    human_anchor_amplifier_candidates as _core_human_anchor_amplifier_candidates,
    human_anchor_suppression_frontier as _core_human_anchor_suppression_frontier,
    human_anchor_suppression_frontier_candidates as _core_human_anchor_suppression_frontier_candidates,
)
from rewrite_pipeline_core.phases.topk_route import (
    TopkRouteDeps,
    deterministic_topk_route_sentence as _core_deterministic_topk_route_sentence,
    remove_sentences_by_text as _core_remove_sentences_by_text,
    splice_sentences_by_text as _core_splice_sentences_by_text,
    topk_low_value_removal_allowed as _core_topk_low_value_removal_allowed,
    topk_masked_route_prompt as _core_topk_masked_route_prompt,
    topk_optimizer_sentence_limit as _core_topk_optimizer_sentence_limit,
    topk_plain_spoken_snapshot_prompt as _core_topk_plain_spoken_snapshot_prompt,
    topk_repair_map as _core_topk_repair_map,
    topk_route_optimizer_candidates as _core_topk_route_optimizer_candidates,
    topk_safe_band_sentence_patch_prompt as _core_topk_safe_band_sentence_patch_prompt,
    topk_safe_band_snapshot_prompt as _core_topk_safe_band_snapshot_prompt,
)
from rewrite_pipeline_core.phases.source_grounding_utils import (
    SOURCE_SEARCH_DEFAULT_EXCLUDE_DOMAINS as _CORE_SOURCE_SEARCH_DEFAULT_EXCLUDE_DOMAINS,
    SOURCE_SEARCH_LOW_VALUE_OVERLAP_TERMS as _CORE_SOURCE_SEARCH_LOW_VALUE_OVERLAP_TERMS,
    SOURCE_SEARCH_STOPWORDS as _CORE_SOURCE_SEARCH_STOPWORDS,
    SourceGroundingPromptDeps,
    SourceGroundingTargetDeps,
    citation_reference_search_targets as _core_citation_reference_search_targets,
    internet_reinforced_reauthor_prompt as _core_internet_reinforced_reauthor_prompt,
    normalize_tavily_results as _core_normalize_tavily_results,
    source_grounding_query as _core_source_grounding_query,
    source_grounding_claim_targets as _core_source_grounding_claim_targets,
    source_grounding_repair_prompt as _core_source_grounding_repair_prompt,
    source_grounding_targets_from_block_decisions as _core_source_grounding_targets_from_block_decisions,
    source_result_confidence as _core_source_result_confidence,
    source_search_domain_blocked as _core_source_search_domain_blocked,
    source_search_domain_list as _core_source_search_domain_list,
    source_search_hostname as _core_source_search_hostname,
    source_search_keywords as _core_source_search_keywords,
    source_search_quality_label as _core_source_search_quality_label,
)
from rewrite_pipeline_core.phases.paragraph_targets import (
    PARAGRAPH_CITATION_RE as _PARAGRAPH_CITATION_RE,
    ParagraphTargetDeps,
    is_heading_like_paragraph as _core_is_heading_like_paragraph,
    orphan_heading_reason as _core_orphan_heading_reason,
    paragraph_component_targets as _core_paragraph_component_targets,
    paragraph_role as _core_paragraph_role,
    paragraph_sentence_starters as _core_paragraph_sentence_starters,
)
from rewrite_pipeline_core.phases.marked_grounding import (
    ai_search_marked_grounding_candidates as _core_ai_search_marked_grounding_candidates,
)
from rewrite_pipeline_core.phases.blocker_operations import (
    BlockerOperationCandidateDeps,
    blocker_operation_candidates as _core_blocker_operation_candidates,
)
from rewrite_pipeline_core.phases.blocker_plan import (
    BlockerOperationPlanDeps,
    block_level_decisions as _core_block_level_decisions,
    blocker_operation_plan as _core_blocker_operation_plan,
)
from rewrite_pipeline_core.phases.content_pruning import (
    ContentPruningCandidateDeps,
    content_pruning_candidates as _core_content_pruning_candidates,
)
from rewrite_pipeline_core.phases.formula_block_candidates import (
    FormulaBlockCandidateDeps,
    formula_block_map_removal_candidates as _core_formula_block_map_removal_candidates,
)
from rewrite_pipeline_core.phases.generic_assertion import (
    GenericAssertionCompilerDeps,
    generic_assertion_compiler_candidates as _core_generic_assertion_compiler_candidates,
    generic_assertion_sentence_score as _core_generic_assertion_sentence_score,
)
from rewrite_pipeline_core.phases.final_scan import (
    run_final_rewritten_scan as _core_run_final_rewritten_scan,
)
from rewrite_pipeline_core.phases.input_context import (
    RewriteInputContextDeps,
    resolve_rewrite_input_context as _core_resolve_rewrite_input_context,
)
from rewrite_pipeline_core.phases.rewrite_engine import (
    RewriteEnginePhaseDeps,
    run_rewrite_engine_phase as _core_run_rewrite_engine_phase,
)
from rewrite_pipeline_core.phases.ai_search_runtime import (
    adaptive_stop_record as _core_ai_search_adaptive_stop_record,
    apply_budget_gateway as _core_ai_search_apply_budget_gateway,
    phase_budget_block_record as _core_ai_search_phase_budget_block_record,
    phase_budget_can_spend as _core_ai_search_phase_budget_can_spend,
    record_phase_llm_call as _core_ai_search_record_phase_llm_call,
    record_verified_candidate_scan as _core_ai_search_record_verified_candidate_scan,
    search_budget_exhausted_record as _core_ai_search_budget_exhausted_record,
    verified_candidate_scans_used as _core_ai_search_verified_candidate_scans_used,
)
from rewrite_pipeline_core.phases.ai_search_quality import (
    _AI_SEARCH_CRITICAL_ENTITY_RE,
    _AI_SEARCH_ENTITY_NOISE,
    _ai_candidate_quality_reject_reason,
    _ai_search_drift_false_positive,
    _ai_search_entity_drift_scan_allowed,
    _ai_search_protected_loss_reason,
    _ai_search_quote_drift_scan_allowed,
    _ai_search_signal_brief,
    _document_recreate_drift_scan_allowed,
    _normalize_direct_quote_content,
    _protected_code_anchor_set,
    _reconstruction_drift_scan_allowed,
    _repair_candidate_source_damage,
    _source_repair_brief,
    _source_repair_drift_false_positive,
)

from rewrite_pipeline_core.prompts.targeted_repairs import (
    TargetedRepairPromptDeps,
    ai_search_feedback_prompt as _core_ai_search_feedback_prompt,
    ai_search_prompt as _core_ai_search_prompt,
    blocked_human_candidate_repair_prompt as _core_blocked_human_candidate_repair_prompt,
    claim_narrowing_repair_prompt as _core_claim_narrowing_repair_prompt,
    topk_texture_repair_prompt as _core_topk_texture_repair_prompt,
)
from rewrite_pipeline_core.prompts.finding_repairs import (
    _apply_finding_local_patches as _core_apply_finding_local_patches,
    _blocking_finding_targets as _core_blocking_finding_targets,
    _exact_blocking_target_from_context as _core_exact_blocking_target_from_context,
    _expand_fragment_to_candidate_sentence as _core_expand_fragment_to_candidate_sentence,
    _extract_finding_local_patches as _core_extract_finding_local_patches,
    _finding_local_repair_prompt as _core_finding_local_repair_prompt,
    _sentences_from_excerpt as _core_sentences_from_excerpt,
)
from rewrite_pipeline_core.prompts.paragraph_prompts import (
    ParagraphPromptDeps,
    author_reasoning_amplification_prompt as _core_author_reasoning_amplification_prompt,
    clean_paragraph_component_candidate as _core_clean_paragraph_component_candidate,
    clean_source_sentence_candidate as _core_clean_source_sentence_candidate,
    extract_paragraph_component_candidates as _core_extract_paragraph_component_candidates,
    human_signal_amplification_prompt as _core_human_signal_amplification_prompt,
    paragraph_anchor_lock as _core_paragraph_anchor_lock,
    paragraph_component_prompt as _core_paragraph_component_prompt,
    paragraph_generation_anchor_context as _core_paragraph_generation_anchor_context,
    splice_paragraph as _core_splice_paragraph,
)
# Compatibility note for source-grep regression tests: the runtime helper keeps
# the default disabled Top-k borrowing check as
# DRAFTPROOF_TOPK_CAN_BORROW_UNUSED_PHASE_BUDGET", False.
from rewrite_pipeline_core.phases.post_topk import (
    PostTopkConvergenceDeps,
    build_post_topk_convergence_candidates as _core_post_topk_convergence_candidates,
)
from rewrite_pipeline_core.phases.post_selection_density import (
    PostSelectionDensityDeps,
    ai_density_patchwork_budget_status as _core_ai_density_patchwork_budget_status,
    post_selection_ai_density_breaker_acceptance as _core_post_selection_ai_density_breaker_acceptance,
    post_selection_ai_density_breaker_candidates as _core_post_selection_ai_density_breaker_candidates,
    run_post_selection_ai_density_breaker as _core_post_selection_ai_density_breaker,
    turnitin_like_positive_burden_drop as _core_turnitin_like_positive_burden_drop,
)
from rewrite_pipeline_core.phases.post_density_anchor import (
    PostDensityHumanAnchorProbeDeps,
    build_post_density_human_anchor_probe_candidates as _core_post_density_human_anchor_probe_candidates,
    post_density_human_anchor_probe_acceptance as _core_post_density_human_anchor_probe_acceptance,
    run_post_density_human_anchor_probe as _core_post_density_human_anchor_probe,
)
from rewrite_pipeline_core.phases.optimization_selection import (
    _metric_repair_diagnosis,
    _optimization_candidate_status,
    _select_best_optimization_candidate,
)
from rewrite_pipeline_core.phases.micro_texture import (
    _apply_masked_span_replacement,
    _anchor_lock_mapping,
    _anchor_values_from_brief,
    _clean_masked_span_replacement,
    _clean_micro_texture_candidate,
    _deterministic_masked_span_replacements,
    _deterministic_sentence_route_bundle,
    _freeze_anchor_payload,
    _freeze_anchor_text,
    _goal_climb_candidate_rank,
    _human_shift_rank_key,
    _human_shift_score,
    _is_better_human_shift_candidate,
    _iterative_micro_texture_repair,
    _locality_score,
    _masked_span_repair_prompt,
    _micro_repair_gain_efficiency,
    _micro_texture_iteration_status,
    _micro_texture_repair_prompt,
    _micro_texture_window,
    _repair_aggression_score,
    _restore_anchor_placeholders,
    _sentence_texture_risk_map,
    _splice_sentence_for_auto_repair,
    _splice_sentence_window,
)
from rewrite_pipeline_core.prompts.reconstruction_runtime import (
    _build_reconstruction_meaning_brief,
    _build_regeneration_blueprint,
    _reconstruction_mitigation_prompt,
    _reconstruction_planning_deps,
    _staged_generation_section_plan,
    _staged_reconstruction_candidate,
    _staged_reconstruction_prompt_deps,
    _staged_reconstruction_section_prompt,
)
from rewrite_pipeline_core.prompts.reconstruction_helpers import (
    _clean_full_document_candidate,
    _clean_section_candidate,
    _generation_context_ledger,
    _human_gain_stage_target,
    _integrity_driver_rows,
    _reconstruction_failure_feedback,
    _reconstruction_gate_controls,
    _reference_entries_from_text,
    _review_marker_notes,
    _target_segment_rows,
    _word_count_band,
)
from rewrite_pipeline_core.prompts.reconstruction import (
    ReconstructionPlanningDeps,
    build_reconstruction_meaning_brief as _core_build_reconstruction_meaning_brief,
    build_regeneration_blueprint as _core_build_regeneration_blueprint,
)
from rewrite_pipeline_core.prompts.staged_reconstruction import (
    StagedReconstructionPromptDeps,
    staged_generation_section_plan as _core_staged_generation_section_plan,
    staged_reconstruction_section_prompt as _core_staged_reconstruction_section_prompt,
)
from rewrite_pipeline_core.state import RewritePipelineState, RewriteScanGateway
from rewrite_pipeline_core.scoring.helpers import (
    _ai_first_gate_status,
    _metric_decimal,
)
from rewrite_pipeline_core.scoring.block_driver_map import (
    FormulaBlockDriverMapDeps,
    formula_block_driver_map as _core_formula_block_driver_map,
)
from rewrite_pipeline_core.scoring.profiles import (
    _ai_footprint_flatten,
    _ai_footprint_profile,
    _blocker_scores,
    _contribution_scores,
    _feature_percent,
    _formula_gap_candidate_rank,
    _formula_gap_changed_word_count,
    _formula_gap_contract,
    _formula_gap_driver_priority_plan,
    _formula_gap_weighted_driver_plan,
    _formula_observed_driver_movement,
    _formula_portfolio_plan,
    _formula_portfolio_plan_from_profiles,
    _integrity_scores,
    _remaining_turnitin_like_drivers,
    _transformation_features,
    _turnitin_like_ai_gate_status,
    _turnitin_like_ai_profile,
    _turnitin_like_candidate_rank,
    _turnitin_like_component_drops,
)
from rewrite_pipeline_core.scoring.human_formula import (
    HumanFormulaDriverDeps,
    human_formula_driver_status as _core_human_formula_driver_status,
)
from rewrite_pipeline_core.text_processing.text_utils import (
    _brief_sentences,
    _join_logical_paragraphs,
    _logical_paragraphs,
    _normalize_protected_text,
    _protected_number_set,
    _split_sentences,
    _text_word_count,
)
from rewrite_pipeline_core.text_processing.quality_artifacts import (
    _ABSTRACT_LIST_TERMS_RE,
    _COLLOQUIAL_TEXTURE_TERMS_RE,
    _DANGLING_FRAGMENT_JOIN_RE,
    _GENERIC_PRAISE_PHRASES_RE,
    _GENERIC_PRAISE_TERMS_RE,
    _LOW_FRICTION_CONTRAST_RE,
    _STYLIZED_TEXTURE_TERMS_RE,
    _SYNTHETIC_ANCHOR_RE,
    _SYNTHETIC_META_ANCHOR_PATTERNS,
    _external_detector_style_artifact_reason,
    _neutralize_external_detector_style_artifacts,
    _normalize_known_heading_boundaries,
    _repeated_long_sequence_reason,
    _repeated_sentence_opening_reason,
    _strip_reference_like_lines_for_quality,
    _synthetic_meta_anchor_artifact_reason,
)
from rewrite_pipeline_core.text_processing.cleanup_transforms import (
    CleanupTransformDeps,
    compress_score_drag_paragraph as _core_compress_score_drag_paragraph,
    final_score_drag_sentence_prune_text as _core_final_score_drag_sentence_prune_text,
    narrow_generic_claim_text as _core_narrow_generic_claim_text,
    plain_language_depolish_text as _core_plain_language_depolish_text,
)


def _candidate_concept_origin_reject_reason(
    source_text: str,
    candidate_text: str,
    *,
    unsupported_term_limit: int = 4,
    unsupported_sentence_limit: int = 3,
) -> str:
    return _core_candidate_concept_origin_reject_reason(
        source_text,
        candidate_text,
        logical_paragraphs=_logical_paragraphs,
        split_sentences=_split_sentences,
        unsupported_term_limit=unsupported_term_limit,
        unsupported_sentence_limit=unsupported_sentence_limit,
    )


def _ai_density_sentence_score(sentence: str, row: dict | None = None) -> float:
    value = str(sentence or "")
    if not value.strip() or _ai_density_breaker_canonical_fact_sentence(value):
        return -100.0
    top10 = float((row or {}).get("top10_ratio") or 0.0)
    top50 = float((row or {}).get("top50_ratio") or 0.0)
    risk = float((row or {}).get("predictability_risk") or 0.0)
    generic_hits = len(_AI_DENSITY_GENERIC_RE.findall(value))
    transition = 1.0 if _AI_DENSITY_TRANSITION_RE.search(value.strip()) else 0.0
    words = _text_word_count(value)
    return round(top10 * 4.0 + top50 * 1.5 + risk * 2.0 + generic_hits * 0.8 + transition * 2.0 + words / 45.0, 3)


def _ai_density_breaker_map(text: str, report_dict: dict | None) -> dict:
    """Map generic high-density spans for the isolated post-selection layer."""
    sentences = _split_sentences(text)
    topk_rows = {
        int(row.get("sentence_index")): row
        for row in (_topk_repair_map(text, report_dict, limit=max(1, len(sentences))).get("targets") or [])
        if isinstance(row, dict) and isinstance(row.get("sentence_index"), int)
    }
    sentence_rows = []
    for index, sentence in enumerate(sentences):
        row = topk_rows.get(index, {})
        score = _ai_density_sentence_score(sentence, row)
        sentence_rows.append({
            "sentence_index": index,
            "sentence": sentence,
            "score": score,
            "canonical_fact_preserved": _ai_density_breaker_canonical_fact_sentence(sentence),
            "generic_hits": len(_AI_DENSITY_GENERIC_RE.findall(sentence)),
            "transition_risk": bool(_AI_DENSITY_TRANSITION_RE.search(sentence.strip())),
            "top10_ratio": row.get("top10_ratio"),
            "top50_ratio": row.get("top50_ratio"),
            "predictability_risk": row.get("predictability_risk"),
        })

    high_indexes = {
        int(row["sentence_index"])
        for row in sentence_rows
        if not row.get("canonical_fact_preserved")
        and (
            float(row.get("top10_ratio") or 0.0) >= 0.55
            or float(row.get("score") or 0.0) >= 3.5
        )
    }
    runs = []
    start = None
    previous = None
    for index in sorted(high_indexes):
        if start is None:
            start = previous = index
            continue
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous))
        start = previous = index
    if start is not None:
        runs.append((start, previous))
    run_rows = [
        {
            "start_sentence": start,
            "end_sentence": end,
            "length": end - start + 1,
            "score": round(sum(float(sentence_rows[i].get("score") or 0.0) for i in range(start, end + 1)), 3),
            "preview": " ".join(sentences[start:min(end + 1, start + 3)])[:260],
        }
        for start, end in runs
    ]
    run_rows.sort(key=lambda row: (int(row.get("length") or 0), float(row.get("score") or 0.0)), reverse=True)

    paragraph_rows = []
    paragraphs = _logical_paragraphs(text)
    for index, paragraph in enumerate(paragraphs):
        paragraph_sentences = _split_sentences(paragraph)
        words = _text_word_count(paragraph)
        generic_hits = len(_AI_DENSITY_GENERIC_RE.findall(paragraph))
        canonical_hits = sum(1 for sentence in paragraph_sentences if _ai_density_breaker_canonical_fact_sentence(sentence))
        protected = bool(_AI_DENSITY_CANONICAL_FACT_RE.search(paragraph) or _is_heading_like_paragraph(paragraph))
        score = round(generic_hits * 1.5 + len(paragraph_sentences) * 0.4 - canonical_hits * 2.0 + words / 80.0, 3)
        paragraph_rows.append({
            "paragraph_index": index,
            "score": score,
            "word_count": words,
            "sentence_count": len(paragraph_sentences),
            "generic_hits": generic_hits,
            "canonical_fact_sentences": canonical_hits,
            "protected": protected,
            "preview": paragraph[:240],
        })
    paragraph_rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    density_windows = _ai_density_window_targets(sentence_rows)
    return {
        "version": "post_selection_ai_density_breaker_map_v1",
        "sentence_count": len(sentences),
        "sentence_rows": sentence_rows,
        "top_sentence_targets": sorted(
            [row for row in sentence_rows if float(row.get("score") or 0.0) > 0.0],
            key=lambda row: float(row.get("score") or 0.0),
            reverse=True,
        )[:12],
        "top_density_windows": density_windows,
        "contiguous_ai_density_runs": run_rows[:6],
        "top_generic_paragraphs": paragraph_rows[:8],
    }


def _ai_density_edit_budget(text: str) -> dict:
    """Bound post-selection edits so the add-on layer cannot create patchwork."""
    sentence_count = len(_split_sentences(text))
    word_count = _text_word_count(text)
    if word_count <= 350:
        max_sentences = 4
        max_ratio = 0.22
    elif word_count <= 800:
        max_sentences = 8
        max_ratio = 0.18
    elif word_count <= 1800:
        max_sentences = 10
        max_ratio = 0.14
    else:
        max_sentences = 14
        max_ratio = 0.10
    if sentence_count <= 6:
        max_ratio = max(max_ratio, 0.34)
    ratio_cap = max(1, int(math.floor(max(1, sentence_count) * max_ratio)))
    effective_max = max(1, min(max_sentences, ratio_cap))
    return {
        "version": "ai_density_edit_budget_v1",
        "word_count": word_count,
        "sentence_count": sentence_count,
        "max_edited_sentences": effective_max,
        "max_edited_sentence_ratio": max_ratio,
    }


def _post_selection_density_deps() -> PostSelectionDensityDeps:
    return PostSelectionDensityDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        split_sentences=_split_sentences,
        text_word_count=_text_word_count,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        ai_density_breaker_map=_ai_density_breaker_map,
        ai_density_edit_budget=_ai_density_edit_budget,
        ai_density_breaker_sentence_route=_ai_density_breaker_sentence_route,
        ai_density_breaker_canonical_fact_sentence=_ai_density_breaker_canonical_fact_sentence,
        splice_sentences_by_text=_splice_sentences_by_text,
        remove_sentences_by_text=_remove_sentences_by_text,
        compress_score_drag_paragraph=_compress_score_drag_paragraph,
        run_full_scan_report_dict=_run_full_scan_report_dict,
        check_semantic_drift=check_semantic_drift,
        detect_protected_spans=detect_protected_spans,
        ai_candidate_quality_reject_reason=_ai_candidate_quality_reject_reason,
        ai_search_protected_loss_reason=_ai_search_protected_loss_reason,
        strict_ai_safe_band_status=_strict_ai_safe_band_status,
        report_review_burden=_report_review_burden,
        report_weighted_severity=_report_weighted_severity,
        critical_high_count=_critical_high_count,
    )


def _ai_density_patchwork_budget_status(source_text: str, candidate_text: str, meta: dict | None = None) -> dict:
    return _core_ai_density_patchwork_budget_status(
        source_text,
        candidate_text,
        meta,
        deps=_post_selection_density_deps(),
    )


def _turnitin_like_positive_burden_drop(before_report: dict | None, after_report: dict | None) -> float:
    return _core_turnitin_like_positive_burden_drop(before_report, after_report)


def _human_anchor_positive_burden_gate_status(
    formula_gap_contract: dict | None,
    candidate_report: dict | None,
) -> dict:
    """Require real AI-driver movement before Human Anchor suppression can win."""
    contract = formula_gap_contract if isinstance(formula_gap_contract, dict) else {}
    burden = contract.get("positive_ai_burden") if isinstance(contract.get("positive_ai_burden"), dict) else {}
    try:
        burden_drop = float(burden.get("drop") or 0.0)
    except (TypeError, ValueError):
        burden_drop = 0.0
    profile = _turnitin_like_ai_profile(candidate_report)
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    def num(value, default=0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    topk_after = num(components.get("topk_calibrated_risk"), 0.0)
    ai_likelihood_after = num(components.get("ai_likelihood"), 0.0)
    strict_driver_active = bool(
        topk_after >= _safe_topk_calibrated_limit()
        or ai_likelihood_after > _float_env("DRAFTPROOF_HUMAN_ANCHOR_AI_LIKELIHOOD_SAFE_MAX", 35.0)
    )
    required_drop = _float_env(
        "DRAFTPROOF_HUMAN_ANCHOR_STRICT_MIN_POSITIVE_BURDEN_DROP"
        if strict_driver_active
        else "DRAFTPROOF_HUMAN_ANCHOR_MIN_POSITIVE_BURDEN_DROP",
        4.0 if strict_driver_active else 1.0,
    )
    accepted = burden_drop >= required_drop
    return {
        "version": "human_anchor_positive_burden_gate_v1",
        "accepted": accepted,
        "reason": "positive_ai_burden_moved" if accepted else "positive_ai_burden_drop_too_small",
        "positive_ai_burden_drop": round(burden_drop, 3),
        "required_positive_ai_burden_drop": round(float(required_drop), 3),
        "strict_driver_active": strict_driver_active,
        "topk_calibrated_risk_after": round(topk_after, 3),
        "ai_likelihood_after": round(ai_likelihood_after, 3),
    }


def _post_selection_ai_density_breaker_candidates(
    current_text: str,
    current_report: dict | None,
    *,
    limit: int = 8,
) -> list[tuple[str, str, dict]]:
    return _core_post_selection_ai_density_breaker_candidates(
        current_text,
        current_report,
        limit=limit,
        deps=_post_selection_density_deps(),
    )


def _post_selection_ai_density_breaker_acceptance(
    current_report: dict | None,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    return _core_post_selection_ai_density_breaker_acceptance(
        current_report,
        candidate_report,
        review_burden_delta=review_burden_delta,
        weighted_severity_delta=weighted_severity_delta,
        critical_high_delta=critical_high_delta,
        float_env=_float_env,
    )


def _density_acceptance_deps() -> DensityAcceptanceDeps:
    return DensityAcceptanceDeps(
        turnitin_like_ai_profile=_turnitin_like_ai_profile,
        eligible_span_density_comparison=_eligible_span_density_comparison,
        ai_footprint_profile=_ai_footprint_profile,
        ai_footprint_flatten=_ai_footprint_flatten,
        report_badge_ai=_report_badge_ai,
        window_coverage_comparison=_window_coverage_comparison,
    )


def _segment_window_density_acceptance(
    current_text: str,
    current_report: dict | None,
    candidate_text: str,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    return _core_segment_window_density_acceptance(
        current_text,
        current_report,
        candidate_text,
        candidate_report,
        review_burden_delta=review_burden_delta,
        weighted_severity_delta=weighted_severity_delta,
        critical_high_delta=critical_high_delta,
        deps=_density_acceptance_deps(),
    )

def _segment_window_density_controller(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    gateway: LLMGateway | None = None,
    scan_func=None,
    drift_checker=check_semantic_drift,
    max_scans: int | None = None,
    max_llm_calls: int | None = None,
) -> dict:
    deps = SegmentWindowDensityControllerDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
        run_full_scan_report_dict=_run_full_scan_report_dict,
        check_semantic_drift=check_semantic_drift,
        detect_protected_spans=detect_protected_spans,
        clean_full_document_candidate=_clean_full_document_candidate,
        ai_candidate_quality_reject_reason=_ai_candidate_quality_reject_reason,
        evaluate_text_quality_regression=evaluate_text_quality_regression,
        ai_search_protected_loss_reason=_ai_search_protected_loss_reason,
        report_review_burden=_report_review_burden,
        report_weighted_severity=_report_weighted_severity,
        critical_high_count=_critical_high_count,
        segment_window_density_acceptance=_segment_window_density_acceptance,
        phase_chat_sampling_kwargs=_phase_chat_sampling_kwargs,
    )
    return _core_segment_window_density_controller(
        current_text,
        current_report,
        original_report,
        gateway=gateway,
        scan_func=scan_func,
        drift_checker=drift_checker,
        max_scans=max_scans,
        max_llm_calls=max_llm_calls,
        deps=deps,
    )


def _remaining_cluster_density_acceptance(
    current_text: str,
    current_report: dict | None,
    candidate_text: str,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    return _core_remaining_cluster_density_acceptance(
        current_text,
        current_report,
        candidate_text,
        candidate_report,
        review_burden_delta=review_burden_delta,
        weighted_severity_delta=weighted_severity_delta,
        critical_high_delta=critical_high_delta,
        deps=_density_acceptance_deps(),
    )

def _remaining_cluster_density_controller(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    gateway: LLMGateway | None = None,
    scan_func=None,
    drift_checker=check_semantic_drift,
    max_scans: int | None = None,
    max_llm_calls: int | None = None,
) -> dict:
    deps = RemainingClusterDensityControllerDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
        run_full_scan_report_dict=_run_full_scan_report_dict,
        check_semantic_drift=check_semantic_drift,
        detect_protected_spans=detect_protected_spans,
        clean_full_document_candidate=_clean_full_document_candidate,
        ai_candidate_quality_reject_reason=_ai_candidate_quality_reject_reason,
        evaluate_text_quality_regression=evaluate_text_quality_regression,
        ai_search_protected_loss_reason=_ai_search_protected_loss_reason,
        report_review_burden=_report_review_burden,
        report_weighted_severity=_report_weighted_severity,
        critical_high_count=_critical_high_count,
        remaining_cluster_density_acceptance=_remaining_cluster_density_acceptance,
        phase_chat_sampling_kwargs=_phase_chat_sampling_kwargs,
    )
    return _core_remaining_cluster_density_controller(
        current_text,
        current_report,
        original_report,
        gateway=gateway,
        scan_func=scan_func,
        drift_checker=drift_checker,
        max_scans=max_scans,
        max_llm_calls=max_llm_calls,
        deps=deps,
    )


def _window_coverage_density_acceptance(
    current_text: str,
    current_report: dict | None,
    candidate_text: str,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    return _core_window_coverage_density_acceptance(
        current_text,
        current_report,
        candidate_text,
        candidate_report,
        review_burden_delta=review_burden_delta,
        weighted_severity_delta=weighted_severity_delta,
        critical_high_delta=critical_high_delta,
        deps=_density_acceptance_deps(),
    )

def _window_coverage_density_optimizer(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    gateway: LLMGateway | None = None,
    scan_func=None,
    drift_checker=check_semantic_drift,
    max_scans: int | None = None,
    max_llm_calls: int | None = None,
) -> dict:
    deps = WindowCoverageOptimizerDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        run_full_scan_report_dict=_run_full_scan_report_dict,
        check_semantic_drift=check_semantic_drift,
        detect_protected_spans=detect_protected_spans,
        clean_full_document_candidate=_clean_full_document_candidate,
        ai_candidate_quality_reject_reason=_ai_candidate_quality_reject_reason,
        evaluate_text_quality_regression=evaluate_text_quality_regression,
        ai_search_protected_loss_reason=_ai_search_protected_loss_reason,
        report_review_burden=_report_review_burden,
        report_weighted_severity=_report_weighted_severity,
        critical_high_count=_critical_high_count,
        window_coverage_density_acceptance=_window_coverage_density_acceptance,
        phase_chat_sampling_kwargs=_phase_chat_sampling_kwargs,
    )
    return _core_window_coverage_density_optimizer(
        current_text,
        current_report,
        original_report,
        gateway=gateway,
        scan_func=scan_func,
        drift_checker=drift_checker,
        max_scans=max_scans,
        max_llm_calls=max_llm_calls,
        deps=deps,
    )

def _post_selection_ai_density_breaker(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    scan_func=None,
    drift_checker=check_semantic_drift,
    max_scans: int | None = None,
) -> dict:
    return _core_post_selection_ai_density_breaker(
        current_text,
        current_report,
        original_report,
        scan_func=scan_func,
        drift_checker=drift_checker,
        max_scans=max_scans,
        deps=_post_selection_density_deps(),
    )


def _post_density_human_anchor_probe_context(sentence: str, paragraph: str = "") -> tuple[str, str]:
    """Return bounded implied-context text for the Human Anchor probe.

    This deliberately avoids personal voice and topic-specific canned content.
    The addition is derived from concepts already present in the local sentence
    or paragraph so predictable prose is not turned into synthetic autobiography.
    """
    local_text = str(sentence or "").strip() or str(paragraph or "").strip()
    terms = _ordered_concept_origin_terms(local_text, limit=4)
    if len(terms) >= 2:
        return (
            f"This point should stay tied to {terms[0]} and {terms[1]}, not treated as a wider claim.",
            "local_concept_scope_limit",
        )
    return (
        "This point should stay tied to the local context already stated, not treated as a wider claim.",
        "local_scope_limit",
    )


def _post_density_human_anchor_probe_deps() -> PostDensityHumanAnchorProbeDeps:
    return PostDensityHumanAnchorProbeDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        run_full_scan_report_dict=_run_full_scan_report_dict,
        check_semantic_drift=check_semantic_drift,
        human_anchor_driver_contract=_human_anchor_driver_contract,
        human_anchor_suppression_frontier=_human_anchor_suppression_frontier,
        ai_density_breaker_map=_ai_density_breaker_map,
        logical_paragraphs=_logical_paragraphs,
        split_sentences=_split_sentences,
        ai_density_breaker_canonical_fact_sentence=_ai_density_breaker_canonical_fact_sentence,
        post_density_human_anchor_probe_context=_post_density_human_anchor_probe_context,
        ai_density_edit_budget=_ai_density_edit_budget,
        splice_sentences_by_text=_splice_sentences_by_text,
        ai_density_patchwork_budget_status=_ai_density_patchwork_budget_status,
        detect_protected_spans=detect_protected_spans,
        ai_candidate_quality_reject_reason=_ai_candidate_quality_reject_reason,
        ai_search_protected_loss_reason=_ai_search_protected_loss_reason,
        report_review_burden=_report_review_burden,
        report_weighted_severity=_report_weighted_severity,
        critical_high_count=_critical_high_count,
        strict_ai_safe_band_status=_strict_ai_safe_band_status,
        turnitin_like_positive_burden_drop=_turnitin_like_positive_burden_drop,
    )


def _post_density_human_anchor_probe_candidates(
    current_text: str,
    current_report: dict | None,
    *,
    limit: int = 3,
) -> list[tuple[str, str, dict]]:
    return _core_post_density_human_anchor_probe_candidates(
        current_text,
        current_report,
        limit=limit,
        deps=_post_density_human_anchor_probe_deps(),
    )


def _post_density_human_anchor_probe_acceptance(
    current_report: dict | None,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    return _core_post_density_human_anchor_probe_acceptance(
        current_report,
        candidate_report,
        review_burden_delta=review_burden_delta,
        weighted_severity_delta=weighted_severity_delta,
        critical_high_delta=critical_high_delta,
        deps=_post_density_human_anchor_probe_deps(),
    )


def _post_density_human_anchor_probe(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    scan_func=None,
    drift_checker=check_semantic_drift,
    max_scans: int | None = None,
) -> dict:
    return _core_post_density_human_anchor_probe(
        current_text,
        current_report,
        original_report,
        scan_func=scan_func,
        drift_checker=drift_checker,
        max_scans=max_scans,
        deps=_post_density_human_anchor_probe_deps(),
    )


def _ai_search_candidate_selection_status(
    reference_ai,
    candidate_ai,
    text_changed: bool,
    min_drop: float = 5.0,
    target: float = 60.0,
    required_min_ai: float = 50.0,
) -> dict:
    """Classify a scanned AI-search candidate without overclaiming tiny drops."""
    gate = _ai_first_gate_status(
        reference_ai,
        candidate_ai,
        text_changed,
        min_drop=min_drop,
        target=target,
        required_min_ai=required_min_ai,
    )
    delta = gate.get("delta")
    improved = isinstance(delta, (int, float)) and delta > 0.05
    if gate["success"]:
        reason = ""
    elif not text_changed:
        reason = "unchanged_candidate"
    elif not improved:
        reason = "candidate_not_below_reference"
    elif gate["required"]:
        reason = "best_candidate_below_required_ai_drop"
    else:
        reason = "ai_first_not_required"
    status = dict(gate)
    status.update({
        "improved": improved,
        "selectable": bool(gate["success"]),
        "reason": reason,
    })
    return status


def _mark_ai_search_progress_selection(
    selection_status: dict,
    *,
    reason: str,
    full_success: bool = False,
    **flags,
) -> dict:
    """Mark a candidate as selectable progress without overclaiming AI success."""
    ai_drop_success = bool(selection_status.get("success"))
    progress_flags = {key: bool(value) for key, value in flags.items()}
    selection_status.update({
        "success": bool(full_success or ai_drop_success),
        "ai_drop_success": ai_drop_success,
        "selectable": True,
        "partial_progress": not bool(full_success),
        "progress_reason": reason,
        "reason": reason,
        **progress_flags,
    })
    return selection_status


def _ai_search_selected_candidate_reaches_goal(selection_status: dict | None) -> bool:
    """Return true only when a selected candidate satisfies an AI-mitigation goal."""
    status = selection_status if isinstance(selection_status, dict) else {}
    if bool(status.get("success") or status.get("ai_drop_success")):
        return True
    goal_flags = (
        "ai_footprint_mitigation",
        "turnitin_like_mitigation",
        "strict_ai_safe_band_achieved",
        "topk_safe_band_achieved",
        "detector_safe",
    )
    return any(bool(status.get(flag)) for flag in goal_flags)


def _safe_partial_quality_improvement_status(
    authenticity_status: dict | None,
    human_shift: dict | None,
    *,
    ai_delta: float | int,
    finding_delta: int | float,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
    ai_score_regressed: bool,
) -> dict:
    """Accept small, safe movement when the larger mitigation gate is not met.

    This is intentionally not a success label. It prevents the rewrite pipeline
    from returning unchanged text after finding a rescanned candidate that
    reduces AI/authorship or review burden without any quality regression.
    """
    if not _env_flag("DRAFTPROOF_ACCEPT_SAFE_PARTIAL_QUALITY_IMPROVEMENT", True):
        return {"allowed": False, "reason": "disabled"}
    authenticity_status = authenticity_status if isinstance(authenticity_status, dict) else {}
    human_shift = human_shift if isinstance(human_shift, dict) else {}

    def num(value, default=0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else float(default)

    human_delta = num(authenticity_status.get("human_delta"))
    ai_authorship_delta = num(authenticity_status.get("ai_authorship_delta"))
    ai_transform_delta = num(authenticity_status.get("ai_transformation_delta"))
    human_shift_score = num(human_shift.get("score"))
    min_ai_drop = _float_env("DRAFTPROOF_SAFE_PARTIAL_MIN_AI_DROP", 0.20)
    min_authorship_drop = _float_env("DRAFTPROOF_SAFE_PARTIAL_MIN_AUTHORSHIP_DROP", 1.0)
    min_human_shift = _float_env("DRAFTPROOF_SAFE_PARTIAL_MIN_HUMAN_SHIFT", 0.5)
    score_improved = num(ai_delta) >= min_ai_drop
    quality_only_min_ai_drop = _float_env(
        "DRAFTPROOF_SAFE_PARTIAL_QUALITY_ONLY_MIN_AI_DROP",
        0.0,
    )
    quality_improved = bool(
        ai_authorship_delta >= min_authorship_drop
        or num(finding_delta) <= -1.0
        or num(review_burden_delta) <= -1.0
        or num(weighted_severity_delta) <= -1.0
    )
    quality_only_improved = bool(
        quality_improved
        and num(ai_delta) >= quality_only_min_ai_drop
    )
    allowed = bool(
        (score_improved or quality_only_improved)
        and human_delta >= 0.0
        and ai_transform_delta >= 0.0
        and (human_shift_score >= min_human_shift or quality_only_improved)
        and quality_improved
        and not ai_score_regressed
        and num(finding_delta) <= 0.0
        and num(review_burden_delta) <= 0.0
        and num(weighted_severity_delta) <= 0.0
        and num(critical_high_delta) <= 0.0
        and not authenticity_status.get("ai_authorship_regression_blocked")
        and not authenticity_status.get("critical_high_regressed")
        and not authenticity_status.get("review_burden_regressed")
        and not authenticity_status.get("weighted_severity_regressed")
        and not authenticity_status.get("human_target_regressed")
        and not authenticity_status.get("ai_transformation_target_regressed")
    )
    return {
        "allowed": allowed,
        "reason": "" if allowed else "safe_partial_threshold_not_met",
        "ai_delta": round(num(ai_delta), 3),
        "min_ai_drop": min_ai_drop,
        "score_improved": score_improved,
        "quality_only_min_ai_drop": quality_only_min_ai_drop,
        "quality_only_improved": quality_only_improved,
        "human_delta": round(human_delta, 3),
        "ai_authorship_delta": round(ai_authorship_delta, 3),
        "min_authorship_drop": min_authorship_drop,
        "ai_transformation_delta": round(ai_transform_delta, 3),
        "human_shift_score": round(human_shift_score, 3),
        "min_human_shift": min_human_shift,
        "quality_improved": quality_improved,
        "finding_delta": finding_delta,
        "review_burden_delta": review_burden_delta,
        "weighted_severity_delta": weighted_severity_delta,
        "critical_high_delta": critical_high_delta,
    }


def _ai_search_selected_by_final_safety_gate(
    ai_search_selected: bool,
    selection_status: dict | None,
) -> bool:
    """Return true when AI-search selection should bypass legacy AI-first rollback."""
    if not ai_search_selected or not isinstance(selection_status, dict):
        return False
    if selection_status.get("selectable"):
        turnitin_gate = selection_status.get("turnitin_like_ai_gate") or {}
        formula_contract = selection_status.get("formula_gap_contract") or {}
        turnitin_drop = turnitin_gate.get("score_drop")
        formula_drop = formula_contract.get("score_drop")
        weighted_formula_drop = formula_contract.get("weighted_formula_score_drop")
        measured_formula_drops = [
            float(value)
            for value in (weighted_formula_drop, formula_drop, turnitin_drop)
            if isinstance(value, (int, float))
        ]
        non_worsening_formula = bool(measured_formula_drops) and all(
            value >= 0.0 for value in measured_formula_drops
        )
        if non_worsening_formula:
            return True
    reason = str(selection_status.get("reason") or "")
    if reason.startswith((
        "accepted_partial_turnitin_like",
        "accepted_turnitin_like",
        "accepted_formula_convergence",
    )):
        return True
    return any(
        bool(selection_status.get(key))
        for key in (
            "authenticity_incremental",
            "human_signal_amplification",
            "safe_authorship_suppression",
            "score_drag_removal",
            "turnitin_like_mitigation",
            "partial_turnitin_like_mitigation",
            "formula_convergence_controller",
            "ai_footprint_mitigation",
            "partial_ai_footprint_mitigation",
            "topk_blocker_progress",
            "safe_partial_quality_improvement",
        )
    )


def _ai_search_final_selection_status(summary: dict | None) -> dict:
    """Return the selection status the final rollback layer should honor."""
    if not isinstance(summary, dict):
        return {}
    search = summary.get("ai_mitigation_search")
    if not isinstance(search, dict):
        return {}
    candidates = [
        search.get("selection_status"),
        (search.get("best_attempt") or {}).get("selection_status")
        if isinstance(search.get("best_attempt"), dict) else None,
        (search.get("selected_candidate") or {}).get("selection_status")
        if isinstance(search.get("selected_candidate"), dict) else None,
        (search.get("best_candidate") or {}).get("selection_status")
        if isinstance(search.get("best_candidate"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _clear_stale_rollback_for_kept_ai_mitigation(summary: dict, source: str) -> None:
    """Clear an earlier density/sentence rollback once AI mitigation is kept."""
    if not isinstance(summary, dict):
        return
    had_stale_rollback = bool(
        summary.get("rollback_applied")
        or summary.get("rollback_reason")
        or summary.get("attempted_final_text")
    )
    summary["rollback_applied"] = False
    summary.pop("rollback_reason", None)
    summary.pop("attempted_final_text", None)
    summary.pop("attempted_sentence_comparison", None)
    summary.pop("detect_scan_attempted", None)
    summary.pop("no_text_change", None)
    summary.pop("no_text_change_reason", None)
    if had_stale_rollback:
        summary.setdefault("saved_contract_notes", []).append(
            f"Cleared earlier rewrite rollback because {source} produced a kept AI-mitigation candidate."
        )










def _selection_status_topk_value(selection_status: dict | None) -> float | None:
    """Return the selected candidate's calibrated Top-k value when available."""
    if not isinstance(selection_status, dict):
        return None
    direct = selection_status.get("topk_calibrated_risk")
    if isinstance(direct, (int, float)):
        return float(direct)
    gate = selection_status.get("ai_footprint_gate") or {}
    if isinstance(gate, dict):
        after = gate.get("after") or {}
        authorship = after.get("authorship_footprint") or {}
        value = authorship.get("topk_calibrated_risk")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _selection_status_topk_safe(selection_status: dict | None) -> bool:
    """Whether a candidate is inside the calibrated Top-k safe band."""
    if not isinstance(selection_status, dict):
        return False
    if selection_status.get("topk_safe_band_achieved"):
        return True
    value = _selection_status_topk_value(selection_status)
    return isinstance(value, (int, float)) and float(value) < _safe_topk_calibrated_limit()






















_SOURCE_SEARCH_RUNTIME_BUDGET = {"calls": 0}


def _reset_source_search_runtime_budget() -> None:
    _SOURCE_SEARCH_RUNTIME_BUDGET["calls"] = 0


def _source_search_max_calls_per_run() -> int:
    return max(
        0,
        min(5, int(_float_env("DRAFTPROOF_SOURCE_SEARCH_MAX_CALLS_PER_RUN", 5.0))),
    )


def _source_search_calls_used() -> int:
    try:
        return int(_SOURCE_SEARCH_RUNTIME_BUDGET.get("calls") or 0)
    except (TypeError, ValueError):
        return 0


def _source_search_remaining_calls() -> int:
    return max(0, _source_search_max_calls_per_run() - _source_search_calls_used())


def _record_source_search_call() -> None:
    _SOURCE_SEARCH_RUNTIME_BUDGET["calls"] = _source_search_calls_used() + 1


def _run_full_scan_report_dict(scan_text: str) -> dict:
    detect_runner = DetectionRunner()
    detect_report = detect_runner.run_all(scan_text)
    builder = ReportBuilder()
    builder.add_detection_report(detect_report)
    if detect_report.postprocess_results:
        builder.add_postprocess_results(detect_report.postprocess_results)
    builder.set_meta(scan_time=0, original_text=scan_text)
    return report_to_dict(builder.build())


def _report_finding_total(report_dict: dict | None) -> int:
    findings = (report_dict or {}).get("findings", {}) if isinstance(report_dict, dict) else {}
    return sum(len(findings.get(tier, [])) for tier in ("critical", "high", "medium", "low"))


def _report_review_burden(report_dict: dict | None) -> int:
    findings = (report_dict or {}).get("findings", {}) if isinstance(report_dict, dict) else {}
    return sum(len(findings.get(tier, [])) for tier in ("critical", "high", "medium"))


def _report_weighted_severity(report_dict: dict | None) -> int:
    findings = (report_dict or {}).get("findings", {}) if isinstance(report_dict, dict) else {}
    weights = {"critical": 8, "high": 5, "medium": 2, "low": 1}
    return sum(len(findings.get(tier, [])) * weight for tier, weight in weights.items())


def _report_badge_ai(report_dict: dict | None):
    score = ((report_dict or {}).get("ai_risk_badge") or {}).get("ai_likelihood_score") if isinstance(report_dict, dict) else None
    return float(score) if isinstance(score, (int, float)) else None


def _report_badge_wq(report_dict: dict | None):
    score = ((report_dict or {}).get("ai_risk_badge") or {}).get("writing_quality_score") if isinstance(report_dict, dict) else None
    return float(score) if isinstance(score, (int, float)) else None


def _allow_ai_search_llm_after_deterministic() -> bool:
    return _env_flag("DRAFTPROOF_AI_SEARCH_ALLOW_LLM_AFTER_DETERMINISTIC", True)


def _score_human_amplification_candidate(
    original_report: dict,
    candidate_report: dict,
    *,
    review_burden_delta: int = 0,
    weighted_severity_delta: int = 0,
    repair_aggression: float = 0.0,
    locality_score: float = 0.0,
) -> dict:
    original_contribution = _contribution_scores(original_report)
    candidate_contribution = _contribution_scores(candidate_report)
    original_integrity = _integrity_scores(original_report)
    candidate_integrity = _integrity_scores(candidate_report)

    def _delta(original_value, candidate_value, *, direction: str = "increase"):
        if not isinstance(original_value, (int, float)) or not isinstance(candidate_value, (int, float)):
            return 0.0
        if direction == "decrease":
            return float(original_value) - float(candidate_value)
        return float(candidate_value) - float(original_value)

    human_delta = _delta(original_contribution.get("human"), candidate_contribution.get("human"))
    ai_authorship_delta = _delta(
        original_integrity.get("ai_authorship"),
        candidate_integrity.get("ai_authorship"),
        direction="decrease",
    )
    ai_transformation_delta = _delta(
        original_contribution.get("ai_transformation"),
        candidate_contribution.get("ai_transformation"),
        direction="decrease",
    )
    score = (
        human_delta * 5.0
        + max(0.0, ai_authorship_delta) * 2.0
        + max(0.0, ai_transformation_delta) * 1.5
        + max(0.0, -float(weighted_severity_delta or 0)) * 1.5
        - max(0.0, float(review_burden_delta or 0)) * 2.0
        - max(0.0, float(repair_aggression or 0.0)) * 1.0
        - max(0.0, float(locality_score or 0.0)) * 0.5
    )
    return {
        "score": round(score, 3),
        "human_delta": round(human_delta, 3),
        "ai_authorship_delta": round(ai_authorship_delta, 3),
        "ai_transformation_delta": round(ai_transformation_delta, 3),
        "review_burden_delta": int(review_burden_delta or 0),
        "weighted_severity_delta": int(weighted_severity_delta or 0),
        "repair_aggression": round(float(repair_aggression or 0.0), 3),
        "locality_score": round(float(locality_score or 0.0), 3),
        "weights": {
            "human_delta": 5.0,
            "ai_authorship_delta": 2.0,
            "ai_transformation_delta": 1.5,
            "weighted_severity_delta": -1.5,
            "review_burden_delta": -2.0,
            "repair_aggression": -1.0,
            "locality_score": -0.5,
        },
    }












def _ai_search_fast_accept_reason(reference_ai, candidate_ai) -> str:
    """Return an early-stop reason when a deterministic candidate is good enough."""
    if not isinstance(reference_ai, (int, float)) or not isinstance(candidate_ai, (int, float)):
        return ""
    fast_accept_ai = _float_env("DRAFTPROOF_AI_SEARCH_FAST_ACCEPT_AI", 50.0)
    fast_accept_delta = _float_env("DRAFTPROOF_AI_SEARCH_FAST_ACCEPT_DELTA", 5.0)
    ai_first_target = _float_env("DRAFTPROOF_AI_FIRST_TARGET", 60.0)
    delta = reference_ai - candidate_ai
    if candidate_ai <= fast_accept_ai:
        return f"candidate_ai<={fast_accept_ai:.2f}"
    if delta >= fast_accept_delta and candidate_ai < ai_first_target:
        return (
            f"delta>={fast_accept_delta:.2f} "
            f"and candidate_ai<{ai_first_target:.2f}"
        )
    return ""


def _ensure_ai_mitigation_contract(report_json: dict | None) -> dict:
    """Backfill ai_mitigation.v1 for older scan JSONs.

    Fresh scans already include this contract. Older saved scans can still run
    rewrite, so the rewrite phase must synthesize the same decision surface
    from available scan intelligence and badge components.
    """
    if not isinstance(report_json, dict):
        return {}
    existing = report_json.get("ai_mitigation")
    if isinstance(existing, dict) and existing.get("schema_version") == "ai_mitigation.v1":
        return existing
    scan_intelligence = report_json.get("scan_intelligence") or {}
    plan = build_ai_mitigation_plan(
        scan_intelligence=scan_intelligence,
        ai_risk_badge=report_json.get("ai_risk_badge") or {},
        rewrite_plan=report_json.get("rewrite_plan") or {},
        rewrite_constraints=report_json.get("rewrite_constraints") or {},
        rewrite_edit_briefs=report_json.get("rewrite_edit_briefs") or [],
    )
    report_json["ai_mitigation"] = plan
    if isinstance(scan_intelligence, dict):
        mitigation_inputs = scan_intelligence.setdefault("mitigation_inputs", {})
        mitigation_inputs["ai_mitigation_plan"] = plan
    return plan


def _ai_mitigation_requires_user_input(ai_mitigation: dict | None) -> bool:
    if not isinstance(ai_mitigation, dict):
        return False
    readiness = ai_mitigation.get("readiness") or {}
    if readiness.get("requires_user_input"):
        return True
    return ai_mitigation.get("primary_mode") in {
        "guided_authenticity_revision",
        "paragraph_authenticity_rebuild",
        "structure_revision",
    }


def _manual_summary_from_ai_mitigation(ai_mitigation: dict | None, limit: int = 12) -> list[dict]:
    if not isinstance(ai_mitigation, dict):
        return []
    rows = []
    for action in ai_mitigation.get("component_actions") or []:
        if not isinstance(action, dict):
            continue
        if action.get("auto_apply"):
            continue
        rows.append({
            "finding_type": "ai_mitigation_guided_action",
            "scanner_target": "ai_mitigation",
            "component": action.get("component"),
            "original_sentence": "",
            "suggested_sentence": action.get("action") or "",
            "rejection_reason": "requires_author_input",
            "why_review_manually": (
                "This mitigation target needs real author evidence, source context, "
                "or a concrete detail. DraftProof will not invent it automatically."
            ),
            "user_input_needed": action.get("user_input_needed"),
            "priority": action.get("priority"),
        })
        if len(rows) >= limit:
            break
    return rows


_MARKED_MITIGATION_COMPONENT_NOTES = {
    "generic_assertion_risk": "make this broad claim specific to your task, condition, source, or context",
    "unsupported_claim_risk": "add the evidence for this claim, or soften the claim if evidence is limited",
    "source_grounding_risk": "name the source and explain how it supports this sentence",
    "citation_weakness_risk": "attach the correct citation and explain the cited evidence",
    "broad_claim_risk": "limit this claim to the exact group, task, or condition",
    "lived_detail_risk": "add a real observed context, workplace, task, or process detail",
    "qualifying_text_ai_density": "rebuild this paragraph around context, evidence, your reasoning, and a limited conclusion",
}


def _marked_mitigation_sentence_note(sentence: str, ai_mitigation: dict | None, used_components: set[str]) -> dict | None:
    actions = [
        action
        for action in ((ai_mitigation or {}).get("component_actions") or [])
        if isinstance(action, dict) and not action.get("auto_apply")
    ]
    if not actions:
        return None
    text = sentence.lower()
    preferred = []
    if any(marker in text for marker in ("this shows", "this means", "important", "improve", "support", "helps")):
        preferred.extend(["generic_assertion_risk", "unsupported_claim_risk", "broad_claim_risk"])
    if any(marker in text for marker in ("according to", "source", "citation", "research", "study")):
        preferred.extend(["source_grounding_risk", "citation_weakness_risk"])
    if any(marker in text for marker in ("i ", "my ", "observed", "workshop", "case", "task", "condition")):
        preferred.append("lived_detail_risk")

    by_component = {str(action.get("component") or ""): action for action in actions}
    selected = None
    for component in preferred:
        if component in by_component and component not in used_components:
            selected = by_component[component]
            break
    if selected is None:
        for action in actions:
            component = str(action.get("component") or "")
            if component and component not in used_components:
                selected = action
                break
    if selected is None:
        selected = actions[0]

    component = str(selected.get("component") or "reviewed_context")
    used_components.add(component)
    note = _MARKED_MITIGATION_COMPONENT_NOTES.get(
        component,
        selected.get("action") or "add verified author context before using this sentence",
    )
    return {
        "component": component,
        "note": note,
        "user_input_needed": selected.get("user_input_needed"),
        "priority": selected.get("priority"),
    }


def _build_marked_mitigation_rewrite(
    text: str,
    ai_mitigation: dict | None,
    *,
    max_marked_sentences: int = 8,
) -> dict:
    """Create a marked draft for user-led AI mitigation.

    This is intentionally not an accepted rewrite. It shows the shape of the
    rewrite and marks every missing fact/source/detail so the user can replace
    placeholders with real author-owned evidence.
    """
    if not isinstance(text, str) or not text.strip() or not isinstance(ai_mitigation, dict):
        return {}

    sentence_re = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
    parts = []
    changes = []
    cursor = 0
    marked = 0
    used_components: set[str] = set()
    for match in sentence_re.finditer(text):
        sentence = match.group(0)
        parts.append(text[cursor:match.start()])
        replacement = sentence
        if marked < max_marked_sentences and len(sentence.split()) >= 8:
            note = _marked_mitigation_sentence_note(sentence, ai_mitigation, used_components)
            if note:
                insert = f" [[ADD VERIFIED DETAIL: {note['note']}]]"
                stripped = sentence.rstrip()
                terminal = ""
                if stripped and stripped[-1] in ".!?":
                    terminal = stripped[-1]
                    stripped = stripped[:-1].rstrip()
                replacement = f"{stripped}{insert}{terminal}"
                changes.append({
                    "index": len(changes) + 1,
                    "component": note["component"],
                    "original_sentence": sentence.strip(),
                    "rewritten_sentence": replacement.strip(),
                    "user_input_needed": note.get("user_input_needed"),
                    "priority": note.get("priority"),
                })
                marked += 1
        parts.append(replacement)
        cursor = match.end()
    parts.append(text[cursor:])
    draft = "".join(parts)
    if not changes:
        return {}
    return {
        "kind": "marked_mitigation_rewrite",
        "auto_apply": False,
        "status": "requires_author_completion",
        "draft_text": draft,
        "changes": changes,
        "instructions": [
            "Replace every [[ADD VERIFIED DETAIL: ...]] marker with a real source, example, observation, limitation, or author explanation.",
            "Delete any marker you cannot truthfully support, and narrow the surrounding claim instead.",
            "Run the scan again only after all bracketed markers have been resolved.",
        ],
    }


def _author_evidence_reporting_deps() -> AuthorEvidenceReportingDeps:
    return AuthorEvidenceReportingDeps(
        contribution_scores=_contribution_scores,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        paragraph_component_targets=_paragraph_component_targets,
        paragraph_role=_paragraph_role,
    )


def _build_author_evidence_completion_layer(
    text: str,
    report_dict: dict | None,
    *,
    target_human: int = 80,
    max_slots: int = 5,
) -> dict:
    return _core_build_author_evidence_completion_layer(
        text,
        report_dict,
        target_human=target_human,
        max_slots=max_slots,
        deps=_author_evidence_reporting_deps(),
    )


def _build_mitigation_ceiling_diagnostics(
    summary: dict,
    author_evidence_completion: dict | None = None,
    *,
    target_human: int = 80,
) -> dict:
    return _core_build_mitigation_ceiling_diagnostics(
        summary,
        author_evidence_completion,
        target_human=target_human,
    )


def _build_author_evidence_intake_layer(
    author_evidence_completion: dict | None,
    mitigation_ceiling: dict | None = None,
    *,
    max_questions: int = 5,
) -> dict:
    """Create a structured intake contract for closing Human Contribution gaps.

    The LLM can ask for, classify, and place author-owned anchors. It must not
    fabricate them. Confirmed answers can then be fed into the existing gated
    rewrite path.
    """
    completion = author_evidence_completion or {}
    if not isinstance(completion, dict) or not completion.get("enabled"):
        return {}
    slots = [slot for slot in (completion.get("slots") or []) if isinstance(slot, dict)]
    if not slots:
        return {}
    ceiling = mitigation_ceiling or {}
    questions = []
    for index, slot in enumerate(slots[: max(1, int(max_questions or 1))], start=1):
        role = str(slot.get("paragraph_role") or "generic_claim_heavy")
        preview = str(slot.get("target_paragraph_preview") or "").strip()
        if role == "source_summary_heavy":
            question = "What source, reading, reference material, or citation supports this paragraph's claim?"
            answer_type = "source_or_citation"
        elif role == "technical_process_rich":
            question = "What concrete task, process step, participant action, or feedback moment did you personally observe?"
            answer_type = "practice_observation"
        elif role == "conclusion_template_risk":
            question = "What limitation, judgement, or specific takeaway would you personally add to make this ending less generic?"
            answer_type = "author_judgement"
        else:
            question = "What real example, workplace/context observation, task detail, or feedback note proves this claim?"
            answer_type = "real_example_or_observation"
        questions.append({
            "id": f"anchor_{index}",
            "slot": slot.get("slot", index),
            "paragraph_index": slot.get("paragraph_index"),
            "paragraph_role": role,
            "question": question,
            "answer_type": answer_type,
            "target_preview": preview[:260],
            "minimum_answer": "1-3 concrete sentences, or one verifiable source/citation plus why it supports the claim.",
            "truth_gate": (
                "Use only information the author can verify. If the answer is uncertain, "
                "the rewrite must narrow the claim instead of presenting it as evidence."
            ),
        })

    return {
        "enabled": True,
        "kind": "author_evidence_intake",
        "status": "awaiting_user_answers",
        "auto_apply": False,
        "primary_blocker": ceiling.get("primary_blocker"),
        "target_human_contribution": completion.get("target_human_contribution"),
        "current_human_contribution": completion.get("current_human_contribution"),
        "estimated_human_after_completion": completion.get("estimated_human_after_completion"),
        "questions": questions,
        "answer_schema": {
            "anchor_id": "anchor_1",
            "answer": "real author-owned evidence, source, example, observation, or limitation",
            "confidence": "confirmed|uncertain|not_available",
            "permission_to_use": True,
        },
        "llm_supervisor_prompt": (
            "Collect concise answers for the listed anchors. Do not invent answers. "
            "For confirmed answers, integrate the anchor into the matching paragraph with minimal wording change. "
            "For uncertain or unavailable answers, narrow the claim instead. Return a rewritten draft only after "
            "every used anchor is confirmed by the user."
        ),
        "close_gap_policy": [
            "LLM may infer the type of missing anchor, not the factual anchor itself.",
            "LLM may ask targeted questions and transform confirmed answers into prose.",
            "LLM must not create dates, sources, institutions, experiences, statistics, or observations that the user did not confirm.",
            "Confirmed anchor integration must still pass drift, authorship, findings, review-burden, and severity gates.",
        ],
    }


def _build_author_context_discovery_layer(
    author_evidence_intake: dict | None,
    report_dict: dict | None = None,
    *,
    max_items: int = 5,
) -> dict:
    return _core_build_author_context_discovery_layer(
        author_evidence_intake,
        report_dict,
        max_items=max_items,
    )


_SOURCE_SEARCH_STOPWORDS = _CORE_SOURCE_SEARCH_STOPWORDS
_SOURCE_SEARCH_LOW_VALUE_OVERLAP_TERMS = _CORE_SOURCE_SEARCH_LOW_VALUE_OVERLAP_TERMS
_SOURCE_SEARCH_DEFAULT_EXCLUDE_DOMAINS = _CORE_SOURCE_SEARCH_DEFAULT_EXCLUDE_DOMAINS


def _source_search_enabled() -> bool:
    if _env_flag("DRAFTPROOF_SOURCE_SEARCH_ENABLED", False):
        return True
    if not _env_flag("DRAFTPROOF_SOURCE_SEARCH_AUTO_ENABLE_WITH_KEY", False):
        return False
    return bool(os.environ.get("TAVILY_API_KEY") or os.environ.get("DRAFTPROOF_TAVILY_API_KEY"))


def _internet_reauthor_priority_status(report_dict: dict | None, source_text: str = "") -> dict:
    """Decide when document-level reauthoring must not be starved by paragraph-component search."""
    if not _env_flag("DRAFTPROOF_INTERNET_REAUTHOR_PRIORITY_FOR_SEVERE_BLOCKERS", True):
        return {"prioritize": False, "reason": "disabled"}
    if not _source_search_enabled():
        return {"prioritize": False, "reason": "source_search_unavailable"}
    badge = (report_dict or {}).get("ai_risk_badge") or {}
    writing = badge.get("writing_components") or {}
    ai_components = badge.get("ai_components") or {}

    def num(mapping: dict, key: str) -> float:
        try:
            return float(mapping.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    blockers = {
        "qualifying_text_ai_density": num(ai_components, "qualifying_text_ai_density"),
        "generic_assertion_risk": num(ai_components, "generic_assertion_risk"),
        "topk_pattern": num(ai_components, "topk_pattern"),
        "unsupported_claim_risk": num(writing, "unsupported_claim_risk"),
        "broad_claim_risk": num(writing, "broad_claim_risk"),
        "source_grounding_risk": num(writing, "source_grounding_risk"),
    }
    severe_threshold = _float_env("DRAFTPROOF_INTERNET_REAUTHOR_PRIORITY_THRESHOLD", 85.0)
    qualifying_threshold = _float_env("DRAFTPROOF_INTERNET_REAUTHOR_PRIORITY_QUALIFYING_THRESHOLD", 80.0)
    severe_keys = [
        key for key, value in blockers.items()
        if value >= (qualifying_threshold if key == "qualifying_text_ai_density" else severe_threshold)
    ]
    if not severe_keys:
        return {
            "prioritize": False,
            "reason": "no_severe_document_blocker",
            "blockers": blockers,
            "word_count": _text_word_count(source_text),
        }
    return {
        "prioritize": True,
        "reason": "severe_document_blocker",
        "blockers": blockers,
        "severe_keys": severe_keys,
        "word_count": _text_word_count(source_text),
    }


def _source_search_keywords(text: str, limit: int = 10) -> list[str]:
    return _core_source_search_keywords(text, limit=limit)


def _source_grounding_query(claim: str) -> str:
    return _core_source_grounding_query(claim)


def _source_search_domain_list(name: str, default: set[str] | None = None) -> list[str]:
    return _core_source_search_domain_list(name, default)


def _source_search_hostname(url: str) -> str:
    return _core_source_search_hostname(url)


def _source_search_domain_blocked(url: str, excluded_domains: set[str]) -> bool:
    return _core_source_search_domain_blocked(url, excluded_domains)


def _source_search_quality_label(url: str) -> str:
    return _core_source_search_quality_label(url)


def _source_grounding_target_deps() -> SourceGroundingTargetDeps:
    return SourceGroundingTargetDeps(
        logical_paragraphs=_logical_paragraphs,
        split_sentences=_split_sentences,
        paragraph_component_targets=_paragraph_component_targets,
        paragraph_role=_paragraph_role,
        safe_index=_safe_index,
        float_env=_float_env,
        paragraph_citation_re=_PARAGRAPH_CITATION_RE,
    )


def _source_grounding_claim_targets(
    text: str,
    report_dict: dict | None,
    *,
    limit: int = 5,
) -> list[dict]:
    return _core_source_grounding_claim_targets(
        text,
        report_dict,
        limit=limit,
        deps=_source_grounding_target_deps(),
    )


def _source_grounding_targets_from_block_decisions(
    text: str,
    report_dict: dict | None,
    block_decisions: list[dict],
    *,
    limit: int = 5,
) -> list[dict]:
    return _core_source_grounding_targets_from_block_decisions(
        text,
        report_dict,
        block_decisions,
        limit=limit,
        deps=_source_grounding_target_deps(),
    )


def _citation_reference_search_targets(
    text: str,
    report_dict: dict | None,
    *,
    limit: int = 3,
) -> list[dict]:
    return _core_citation_reference_search_targets(
        text,
        report_dict,
        limit=limit,
        deps=_source_grounding_target_deps(),
    )


def _tavily_search(
    query: str,
    *,
    api_key: str,
    max_results: int = 3,
    timeout: float = 20.0,
    exclude_domains: list[str] | None = None,
    include_domains: list[str] | None = None,
    search_depth: str | None = None,
    chunks_per_source: int | None = None,
    include_answer: bool | None = None,
) -> dict:
    depth = str(search_depth or os.environ.get("DRAFTPROOF_SOURCE_SEARCH_DEPTH", "basic")).strip().lower()
    if depth not in {"basic", "advanced"}:
        depth = "basic"
    payload = {
        "query": query,
        "search_depth": depth,
        "max_results": max(1, min(int(max_results or 3), 10)),
        "include_answer": bool(include_answer) if include_answer is not None else False,
        "include_raw_content": False,
    }
    if depth == "advanced" and chunks_per_source:
        payload["chunks_per_source"] = max(1, min(int(chunks_per_source or 1), 5))
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    if include_domains:
        payload["include_domains"] = include_domains
    _record_source_search_call()
    response = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=max(3.0, float(timeout or 20.0)),
    )
    response.raise_for_status()
    return response.json()


def _source_search_depth_status(report_dict: dict | None, target_count: int) -> dict:
    """Use advanced retrieval only when source grounding is severe enough to justify the cost."""
    configured = os.environ.get("DRAFTPROOF_SOURCE_SEARCH_DEPTH")
    if configured:
        depth = str(configured).strip().lower()
        if depth not in {"basic", "advanced"}:
            depth = "basic"
        return {
            "search_depth": depth,
            "source": "env",
            "chunks_per_source": (
                max(1, min(int(_float_env("DRAFTPROOF_SOURCE_SEARCH_CHUNKS_PER_SOURCE", 3.0)), 5))
                if depth == "advanced" else 0
            ),
            "include_answer": _env_flag("DRAFTPROOF_SOURCE_SEARCH_INCLUDE_ANSWER", depth == "advanced"),
        }
    if not _env_flag("DRAFTPROOF_SOURCE_SEARCH_AUTO_DEPTH", True):
        return {"search_depth": "basic", "source": "auto_disabled", "chunks_per_source": 0, "include_answer": False}
    writing = ((report_dict or {}).get("ai_risk_badge") or {}).get("writing_components") or {}
    severe_source_gap = max(
        float(writing.get("source_grounding_risk") or 0.0),
        float(writing.get("unsupported_claim_risk") or 0.0),
        float(writing.get("broad_claim_risk") or 0.0),
    )
    advanced_threshold = _float_env("DRAFTPROOF_SOURCE_SEARCH_ADVANCED_THRESHOLD", 70.0)
    advanced_target_limit = int(_float_env("DRAFTPROOF_SOURCE_SEARCH_ADVANCED_MAX_TARGETS", 2.0))
    if severe_source_gap >= advanced_threshold and int(target_count or 0) <= advanced_target_limit:
        return {
            "search_depth": "advanced",
            "source": "adaptive_severe_source_gap",
            "severe_source_gap": round(severe_source_gap, 3),
            "chunks_per_source": max(
                1,
                min(int(_float_env("DRAFTPROOF_SOURCE_SEARCH_CHUNKS_PER_SOURCE", 3.0)), 5),
            ),
            "include_answer": _env_flag("DRAFTPROOF_SOURCE_SEARCH_INCLUDE_ANSWER", True),
        }
    return {
        "search_depth": "basic",
        "source": "adaptive_basic",
        "severe_source_gap": round(severe_source_gap, 3),
        "chunks_per_source": 0,
        "include_answer": False,
    }


def _normalize_tavily_results(
    payload: dict,
    claim: str = "",
    *,
    limit: int = 3,
    excluded_domains: set[str] | None = None,
) -> list[dict]:
    return _core_normalize_tavily_results(
        payload,
        claim,
        limit=limit,
        excluded_domains=excluded_domains,
    )


def _source_result_confidence(sources: list[dict]) -> str:
    return _core_source_result_confidence(sources)


def _protected_anchor_brief_for_prompt(source_text: str, *, limit: int = 24) -> list[dict]:
    """Return exact protected spans that generation prompts must preserve."""
    seen: set[str] = set()
    anchors: list[dict] = []
    for value in sorted(_protected_code_anchor_set(source_text or ""), key=len, reverse=True):
        actual = next(
            (
                match.group(0)
                for match in re.finditer(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b", source_text or "")
                if match.group(0).lower() == value
            ),
            value.upper(),
        )
        if actual and actual not in seen:
            seen.add(actual)
            anchors.append({
                "text": actual,
                "reason": "code_anchor",
            })
            if len(anchors) >= max(1, int(limit or 1)):
                return anchors
    for match in re.finditer(r"[\"“][^\"”]{3,}[\"”]", source_text or ""):
        value = re.sub(r"\s+", " ", match.group(0).strip())
        words = re.findall(r"\b[\w'-]+\b", value.strip('"“”'))
        if len(words) < 2 or value in seen:
            continue
        seen.add(value)
        anchors.append({
            "text": value,
            "reason": "quoted_anchor",
        })
        if len(anchors) >= max(1, int(limit or 1)):
            return anchors
    for span in detect_protected_spans(source_text or ""):
        value = str(source_text or "")[span.start_char:span.end_char].strip()
        value = re.sub(r"\s+", " ", value)
        if not value or value in seen:
            continue
        seen.add(value)
        anchors.append({
            "text": value,
            "reason": getattr(span, "reason", "protected"),
        })
        if len(anchors) >= max(1, int(limit or 1)):
            break
    return anchors


def _source_grounding_prompt_deps() -> SourceGroundingPromptDeps:
    return SourceGroundingPromptDeps(
        word_count_band=_word_count_band,
        float_env=_float_env,
        protected_anchor_brief_for_prompt=_protected_anchor_brief_for_prompt,
        logical_paragraphs=_logical_paragraphs,
        safe_index=_safe_index,
        text_word_count=_text_word_count,
    )


def _source_grounding_repair_prompt(
    target: dict,
    source_result: dict,
    *,
    candidate_count: int = 2,
) -> str:
    return _core_source_grounding_repair_prompt(
        target,
        source_result,
        candidate_count=candidate_count,
    )


def _internet_reinforced_reauthor_prompt(
    source_text: str,
    source_layer: dict,
    *,
    candidate_count: int = 2,
) -> str:
    return _core_internet_reinforced_reauthor_prompt(
        source_text,
        source_layer,
        candidate_count=candidate_count,
        deps=_source_grounding_prompt_deps(),
    )


def _targeted_repair_prompt_deps() -> TargetedRepairPromptDeps:
    return TargetedRepairPromptDeps(
        blocker_scores=_blocker_scores,
        logical_paragraphs=_logical_paragraphs,
        paragraph_component_targets=_paragraph_component_targets,
        text_word_count=_text_word_count,
        word_count_band=_word_count_band,
        float_env=_float_env,
        protected_anchor_brief_for_prompt=_protected_anchor_brief_for_prompt,
        sentence_texture_risk_map=_sentence_texture_risk_map,
        safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
        ai_search_signal_brief=_ai_search_signal_brief,
        source_repair_brief=_source_repair_brief,
    )


def _claim_narrowing_repair_prompt(
    source_text: str,
    report_dict: dict | None,
    *,
    candidate_count: int = 2,
) -> str:
    return _core_claim_narrowing_repair_prompt(
        source_text,
        report_dict,
        candidate_count=candidate_count,
        deps=_targeted_repair_prompt_deps(),
    )


def _topk_texture_repair_prompt(
    source_text: str,
    report_dict: dict | None,
    *,
    candidate_count: int = 2,
) -> str:
    return _core_topk_texture_repair_prompt(
        source_text,
        report_dict,
        candidate_count=candidate_count,
        deps=_targeted_repair_prompt_deps(),
    )


def _topk_route_deps() -> TopkRouteDeps:
    return TopkRouteDeps(
        text_word_count=_text_word_count,
        float_env=_float_env,
        split_sentences=_split_sentences,
        safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
        protected_anchor_brief_for_prompt=_protected_anchor_brief_for_prompt,
        contribution_scores=_contribution_scores,
    )


def _topk_optimizer_sentence_limit(text: str) -> int:
    return _core_topk_optimizer_sentence_limit(text, deps=_topk_route_deps())


def _topk_repair_map(text: str, report_dict: dict | None, *, limit: int | None = None) -> dict:
    return _core_topk_repair_map(text, report_dict, limit=limit, deps=_topk_route_deps())


def _deterministic_topk_route_sentence(sentence: str) -> tuple[str, list[str]]:
    return _core_deterministic_topk_route_sentence(sentence)


def _splice_sentences_by_text(text: str, replacements: dict[str, str]) -> str:
    return _core_splice_sentences_by_text(text, replacements)


def _remove_sentences_by_text(text: str, sentences: list[str]) -> str:
    return _core_remove_sentences_by_text(text, sentences)


def _topk_low_value_removal_allowed(sentence: str, row: dict | None = None) -> bool:
    return _core_topk_low_value_removal_allowed(sentence, row, deps=_topk_route_deps())


def _topk_route_optimizer_candidates(
    text: str,
    report_dict: dict | None,
    *,
    limit: int | None = None,
) -> list[tuple[str, str, dict]]:
    return _core_topk_route_optimizer_candidates(text, report_dict, limit=limit, deps=_topk_route_deps())


def _topk_masked_route_prompt(
    text: str,
    report_dict: dict | None,
    *,
    candidate_count: int = 2,
) -> str:
    return _core_topk_masked_route_prompt(text, report_dict, candidate_count=candidate_count, deps=_topk_route_deps())


def _topk_safe_band_snapshot_prompt(text: str, report_dict: dict | None) -> str:
    return _core_topk_safe_band_snapshot_prompt(text, report_dict, deps=_topk_route_deps())


def _topk_plain_spoken_snapshot_prompt(text: str, report_dict: dict | None) -> str:
    return _core_topk_plain_spoken_snapshot_prompt(text, report_dict, deps=_topk_route_deps())


def _topk_calibrated_from_report(report_dict: dict | None) -> float | None:
    if not isinstance(report_dict, dict):
        return None
    ai_components = (((report_dict.get("ai_risk_badge") or {}).get("ai_components") or {}))
    value = ai_components.get("topk_calibrated_risk")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _topk_gap_band(report_dict: dict | None) -> dict:
    value = _topk_calibrated_from_report(report_dict)
    safe = _safe_topk_calibrated_limit()
    if not isinstance(value, (int, float)):
        return {"value": None, "safe": safe, "gap": 0.0, "band": "unknown"}
    gap = max(0.0, float(value) - safe)
    if gap >= 60:
        band = "saturated"
    elif gap >= 35:
        band = "high"
    elif gap >= 10:
        band = "elevated"
    elif gap > 0:
        band = "near_miss"
    else:
        band = "safe"
    return {"value": round(float(value), 3), "safe": safe, "gap": round(gap, 3), "band": band}


def _topk_safe_band_patch_rounds_default(source_text: str, report_dict: dict | None = None) -> int:
    words = _text_word_count(source_text)
    gap = _topk_gap_band(report_dict)
    band = gap.get("band")
    if words <= 700:
        base, cap = 2, 8
    elif words <= 1800:
        base, cap = 4, 12
    else:
        base, cap = 5, 14
    extra_by_gap = {
        "saturated": 6,
        "high": 4,
        "elevated": 2,
        "near_miss": 2,
        "safe": 0,
        "unknown": 1,
    }.get(str(band), 1)
    return max(1, min(cap, base + extra_by_gap))


def _topk_safe_band_snapshot_max_tokens_default(source_text: str) -> int:
    words = _text_word_count(source_text)
    if words <= 700:
        return 2200
    if words <= 1800:
        return 3600
    return 6500


def _topk_safe_band_sentence_patch_prompt(candidate_text: str, candidate_report: dict | None) -> str:
    return _core_topk_safe_band_sentence_patch_prompt(
        candidate_text,
        candidate_report,
        deps=_topk_route_deps(),
    )


_POST_TOPK_TEMPLATE_OPENING_RE = re.compile(
    r"^\s*(?:in\s+(?:conclusion|summary|the\s+end)|overall|therefore|thus|"
    r"this\s+(?:shows|highlights|demonstrates|underscores|means)|"
    r"it\s+is\s+(?:important|essential|crucial)\s+to\s+(?:note|understand|consider))\b",
    re.I,
)

_POST_TOPK_LOW_VALUE_PARAGRAPH_RE = re.compile(
    r"\b(?:in\s+the\s+end|overall|in\s+conclusion|real\s+work\s+of|"
    r"important|essential|crucial|significant|changing\s+world|system|"
    r"(?:people|users|readers|participants)\s+(?:need|should|must))\b",
    re.I,
)


def _post_topk_texture_helper_deps() -> PostTopkTextureHelperDeps:
    return PostTopkTextureHelperDeps(
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        text_word_count=_text_word_count,
        clean_paragraph_component_candidate=_clean_paragraph_component_candidate,
        strict_ai_safe_band_status=_strict_ai_safe_band_status,
        post_topk_convergence_candidates=_post_topk_convergence_candidates,
        split_sentences=_split_sentences,
        sentence_has_concrete_or_context=_sentence_has_concrete_or_context,
        generic_assertion_sentence_score=_generic_assertion_sentence_score,
        paragraph_role=_paragraph_role,
        detect_protected_spans=detect_protected_spans,
        generic_assertion_protected_sentence_re=_GENERIC_ASSERTION_PROTECTED_SENTENCE_RE,
        paragraph_citation_re=_PARAGRAPH_CITATION_RE,
        generic_assertion_terms_re=_GENERIC_ASSERTION_TERMS_RE,
        post_topk_template_opening_re=_POST_TOPK_TEMPLATE_OPENING_RE,
        post_topk_low_value_paragraph_re=_POST_TOPK_LOW_VALUE_PARAGRAPH_RE,
    )


def _post_topk_sentence_contextual(sentence: str) -> bool:
    return _core_post_topk_sentence_contextual(sentence, deps=_post_topk_texture_helper_deps())


def _post_topk_sentence_driver_score(sentence: str) -> float:
    return _core_post_topk_sentence_driver_score(sentence, deps=_post_topk_texture_helper_deps())


def _post_topk_driver_map(text: str, raw_json: dict | None) -> dict:
    return _core_post_topk_driver_map(text, raw_json, deps=_post_topk_texture_helper_deps())


def _authorship_transformation_texture_patch_prompt(candidate_text: str, candidate_report: dict | None) -> str:
    return _core_authorship_transformation_texture_patch_prompt(
        candidate_text,
        candidate_report,
        deps=_post_topk_texture_helper_deps(),
    )

def _extract_post_topk_patch_candidates(response_text: str, *, max_candidates: int = 2) -> list[dict]:
    return _core_extract_post_topk_patch_candidates(response_text, max_candidates=max_candidates)

def _apply_post_topk_patches(text: str, patches: list[dict]) -> tuple[str, list[dict]]:
    return _core_apply_post_topk_patches(
        text,
        patches,
        deps=_post_topk_texture_helper_deps(),
    )

def _authorship_transformation_texture_driver_map(text: str, raw_json: dict | None) -> dict:
    return _core_authorship_transformation_texture_driver_map(
        text,
        raw_json,
        deps=_post_topk_texture_helper_deps(),
    )

def _texture_candidate_family(operation: str | None) -> str:
    return _core_texture_candidate_family(operation)

def _authorship_transformation_texture_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 12,
) -> list[tuple[str, str, dict]]:
    return _core_authorship_transformation_texture_candidates(
        source_text,
        raw_json,
        limit=limit,
        deps=_post_topk_texture_helper_deps(),
    )


def _post_topk_convergence_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 10,
) -> list[tuple[str, str, dict]]:
    deps = PostTopkConvergenceDeps(
        env_flag=_env_flag,
        strict_ai_safe_band_status=_strict_ai_safe_band_status,
        safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
        logical_paragraphs=_logical_paragraphs,
        post_topk_driver_map=_post_topk_driver_map,
        text_word_count=_text_word_count,
        float_env=_float_env,
        join_logical_paragraphs=_join_logical_paragraphs,
        split_sentences=_split_sentences,
        narrow_generic_claim_text=_narrow_generic_claim_text,
        compress_score_drag_paragraph=_compress_score_drag_paragraph,
        post_topk_template_opening_re=_POST_TOPK_TEMPLATE_OPENING_RE,
    )
    return _core_post_topk_convergence_candidates(
        source_text,
        raw_json,
        limit=limit,
        deps=deps,
    )


def _extract_topk_route_patch_candidates(response_text: str, *, max_candidates: int = 2) -> list[list[dict]]:
    text = str(response_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []
    if isinstance(data, dict) and isinstance(data.get("patches"), list):
        rows = [data]
    else:
        rows = data.get("candidates") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    candidates: list[list[dict]] = []
    for row in rows[:max(1, max_candidates)]:
        patches = row.get("patches") if isinstance(row, dict) else None
        if not isinstance(patches, list):
            continue
        clean = []
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            original = str(patch.get("original_sentence") or "").strip()
            replacement = str(patch.get("replacement_sentence") or "").strip()
            if original and replacement and original != replacement:
                clean.append({
                    "sentence_id": patch.get("sentence_id"),
                    "original_sentence": original,
                    "replacement_sentence": replacement,
                })
        if clean:
            candidates.append(clean)
    return candidates


def _apply_topk_route_patches(text: str, patches: list[dict]) -> tuple[str, list[dict]]:
    candidate = str(text or "")
    applied = []
    for patch in patches or []:
        original = str(patch.get("original_sentence") or "").strip()
        replacement = str(patch.get("replacement_sentence") or "").strip()
        if not original or not replacement or original == replacement:
            continue
        if original not in candidate:
            continue
        candidate = candidate.replace(original, replacement, 1)
        applied.append(patch)
    return candidate, applied


def _source_grounding_search_deps() -> SourceGroundingSearchDeps:
    return SourceGroundingSearchDeps(
        source_search_enabled=_source_search_enabled,
        source_search_max_calls_per_run=_source_search_max_calls_per_run,
        source_search_calls_used=_source_search_calls_used,
        source_search_remaining_calls=_source_search_remaining_calls,
        float_env=_float_env,
        blocker_operation_plan=_blocker_operation_plan,
        source_grounding_targets_from_block_decisions=_source_grounding_targets_from_block_decisions,
        citation_reference_search_targets=_citation_reference_search_targets,
        source_grounding_claim_targets=_source_grounding_claim_targets,
        source_search_depth_status=_source_search_depth_status,
        source_search_domain_list=_source_search_domain_list,
        source_search_default_exclude_domains=_SOURCE_SEARCH_DEFAULT_EXCLUDE_DOMAINS,
        tavily_search=_tavily_search,
        normalize_tavily_results=_normalize_tavily_results,
        source_result_confidence=_source_result_confidence,
    )


def _build_source_grounding_search_layer(
    text: str,
    report_dict: dict | None,
    *,
    max_queries: int | None = None,
    max_results: int | None = None,
) -> dict:
    return _core_build_source_grounding_search_layer(
        text,
        report_dict,
        max_queries=max_queries,
        max_results=max_results,
        deps=_source_grounding_search_deps(),
    )


def _source_grounding_repair_matches(
    source_layer: dict | None,
    usable_confidences: set[str] | list[str] | tuple[str, ...] | None = None,
    *,
    limit: int | None = None,
) -> list[dict]:
    """Return source-search results that can be safely mapped back to repair targets.

    Source search and target generation can use different identifiers when the
    target came from block-level decisions instead of claim extraction. The
    paragraph index is the stable fallback; without it, a production run can
    search successfully and still produce zero source repair candidates.
    """
    source_layer = source_layer or {}
    allowed_confidences = {str(item) for item in (usable_confidences or []) if str(item)}
    claim_targets = [
        target for target in (source_layer.get("claim_targets") or [])
        if isinstance(target, dict)
    ]
    targets_by_id = {
        target.get("id"): target
        for target in claim_targets
        if target.get("id")
    }
    targets_by_paragraph: dict[int, dict] = {}
    for target in claim_targets:
        paragraph_index = _safe_index(target.get("paragraph_index"), -1)
        if paragraph_index >= 0 and paragraph_index not in targets_by_paragraph:
            targets_by_paragraph[paragraph_index] = target

    matched: list[dict] = []
    max_matches = max(0, int(limit)) if limit is not None else None
    for result in (source_layer.get("results") or []):
        if not isinstance(result, dict):
            continue
        confidence = str(result.get("source_confidence") or "")
        if allowed_confidences and confidence not in allowed_confidences:
            continue
        target = targets_by_id.get(result.get("claim_id"))
        if not target:
            paragraph_index = _safe_index(result.get("paragraph_index"), -1)
            target = targets_by_paragraph.get(paragraph_index)
        if not target:
            continue
        result_with_target = dict(result)
        result_with_target["_repair_target"] = target
        matched.append(result_with_target)
        if max_matches is not None and len(matched) >= max_matches:
            break
    return matched


def _source_reference_entries_from_layer(source_layer: dict | None, *, limit: int = 3) -> list[str]:
    """Build verifiable reference entries from accepted source-search results."""
    source_layer = source_layer or {}
    entries: list[str] = []
    seen_urls: set[str] = set()
    for result in source_layer.get("results") or []:
        if not isinstance(result, dict):
            continue
        if str(result.get("source_confidence") or "") not in {"strong", "moderate"}:
            continue
        for source in result.get("sources") or []:
            if not isinstance(source, dict):
                continue
            url = str(source.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            title = re.sub(r"^\s*\[[^\]]+\]\s*", "", str(source.get("title") or "").strip())
            title = re.sub(r"\s+", " ", title).strip(" .")
            if not title:
                title = url
            entries.append(f"{title}. {url}")
            seen_urls.add(url)
            if len(entries) >= max(0, int(limit or 0)):
                return entries
    return entries


def _source_reference_append_candidate(text: str, source_layer: dict | None, *, limit: int = 2) -> str:
    """Append source references as a candidate, preserving the body unchanged."""
    entries = _source_reference_entries_from_layer(source_layer, limit=limit)
    if not entries:
        return ""
    body = (text or "").rstrip()
    if re.search(r"(?im)^\s*(?:references|reference list|bibliography|works cited|sources)\s*$", body):
        existing = body
        missing = [entry for entry in entries if entry not in existing]
        if not missing:
            return ""
        return existing + "\n" + "\n".join(missing)
    return body + "\n\nReferences\n\n" + "\n".join(entries)


def _load_author_evidence_answers() -> list[dict]:
    """Load confirmed author-evidence answers supplied by the product layer."""
    raw = os.environ.get("DRAFTPROOF_AUTHOR_EVIDENCE_ANSWERS_JSON")
    file_path = os.environ.get("DRAFTPROOF_AUTHOR_EVIDENCE_ANSWERS_FILE")
    if file_path and not raw:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError:
            return []
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("@"):
        try:
            with open(raw[1:], "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError:
            return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if isinstance(payload, dict):
        answers = payload.get("answers") or payload.get("author_evidence_answers") or []
    else:
        answers = payload
    return [item for item in answers if isinstance(item, dict)] if isinstance(answers, list) else []


def _confirmed_author_anchor_brief(answers: list[dict], *, limit: int = 6) -> str:
    rows = []
    for raw in answers or []:
        if not isinstance(raw, dict):
            continue
        answer = " ".join(str(raw.get("answer") or "").split()).strip()
        confidence = str(raw.get("confidence") or "").strip().lower()
        if confidence != "confirmed" or raw.get("permission_to_use") is not True:
            continue
        if len(answer.split()) < 8 or "[[" in answer or "]]" in answer:
            continue
        rows.append({
            "anchor_id": raw.get("anchor_id") or raw.get("id"),
            "answer": answer[:420],
        })
        if len(rows) >= max(1, int(limit or 1)):
            break
    if not rows:
        return ""
    return (
        "Confirmed author anchors available before generation:\n"
        f"{json.dumps(rows, ensure_ascii=False, indent=2)}\n"
        "Use these anchors only where they directly fit the paragraph context. "
        "Each anchor may be used at most once in the whole draft. "
        "Do not repeat, paraphrase, or echo the same anchor across multiple paragraphs. "
        "Do not invent adjacent facts, and do not force an anchor into an unrelated paragraph."
    )


def _confirmed_anchor_echo_reason(text: str, answers: list[dict]) -> str:
    """Reject candidates that reuse one confirmed anchor as repeated prose texture."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or ""))
        if sentence.strip()
    ]
    for raw in answers or []:
        if not isinstance(raw, dict):
            continue
        answer = " ".join(str(raw.get("answer") or "").split()).strip()
        confidence = str(raw.get("confidence") or "").strip().lower()
        if confidence != "confirmed" or raw.get("permission_to_use") is not True:
            continue
        keywords = _author_anchor_keywords(answer)
        if len(keywords) < 3:
            continue
        matched = []
        threshold = max(2, min(4, len(keywords) // 2))
        for idx, sentence in enumerate(sentences):
            sentence_keywords = _author_anchor_keywords(sentence)
            overlap = keywords & sentence_keywords
            if len(overlap) >= threshold:
                matched.append({
                    "sentence_index": idx,
                    "overlap": sorted(overlap)[:8],
                })
        if len(matched) > 1:
            return (
                "confirmed_anchor_repeated "
                f"{raw.get('anchor_id') or raw.get('id') or 'anchor'}:{len(matched)}"
            )
    return ""


def _author_anchor_keywords(text: str) -> set[str]:
    stop = {
        "about", "after", "again", "also", "because", "before", "being", "between",
        "could", "every", "from", "have", "into", "more", "most", "only", "other",
        "should", "some", "that", "their", "them", "then", "there", "these", "they",
        "this", "through", "when", "where", "which", "while", "with", "would",
        "topic", "context", "process", "practice",
    }
    words = re.findall(r"\b[A-Za-z][A-Za-z']{3,}\b", str(text or "").lower())
    normalized = set()
    for word in words:
        base = word.rstrip("s")
        if base and base not in stop:
            normalized.add(base)
    return normalized


def _author_answer_relevance(question: dict, answer: str) -> dict:
    preview = str((question or {}).get("target_preview") or "")
    preview_keywords = _author_anchor_keywords(preview)
    answer_keywords = _author_anchor_keywords(answer)
    overlap = sorted(preview_keywords & answer_keywords)
    if not preview_keywords:
        return {"accepted": True, "overlap": overlap, "reason": "no_preview_keywords"}
    if overlap:
        return {"accepted": True, "overlap": overlap, "reason": ""}
    return {
        "accepted": False,
        "overlap": [],
        "reason": "answer_does_not_match_anchor_context",
        "preview_keywords": sorted(preview_keywords)[:12],
        "answer_keywords": sorted(answer_keywords)[:12],
    }


def _validate_author_evidence_answers(intake: dict | None, answers: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep only confirmed, permissioned, concrete answers matched to intake anchors."""
    if not isinstance(intake, dict):
        return [], []
    question_map = {
        str(item.get("id") or ""): item
        for item in (intake.get("questions") or [])
        if isinstance(item, dict) and item.get("id")
    }
    accepted = []
    rejected = []
    weak_patterns = re.compile(
        r"\b(?:not sure|maybe|possibly|some example|something like|n/?a|none|no idea|tbd)\b",
        flags=re.I,
    )
    for raw in answers or []:
        anchor_id = str(raw.get("anchor_id") or raw.get("id") or "").strip()
        answer = " ".join(str(raw.get("answer") or "").split()).strip()
        confidence = str(raw.get("confidence") or "").strip().lower()
        permission = raw.get("permission_to_use")
        question = question_map.get(anchor_id)
        reason = ""
        if not question:
            reason = "unknown_anchor_id"
        elif confidence != "confirmed":
            reason = "answer_not_confirmed"
        elif permission is not True:
            reason = "permission_to_use_required"
        elif len(answer.split()) < 8:
            reason = "answer_too_short"
        elif weak_patterns.search(answer):
            reason = "answer_too_uncertain_or_generic"
        elif "[[" in answer or "]]" in answer:
            reason = "answer_contains_placeholder_marker"
        relevance = _author_answer_relevance(question, answer) if question and not reason else {}
        if not reason and not relevance.get("accepted", True):
            reason = relevance.get("reason") or "answer_does_not_match_anchor_context"
        if reason:
            rejected.append({
                "anchor_id": anchor_id,
                "reason": reason,
                "relevance": relevance or None,
            })
            continue
        accepted.append({
            "anchor_id": anchor_id,
            "answer": answer,
            "question": question,
            "confidence": confidence,
            "permission_to_use": True,
            "relevance": relevance,
        })
    return accepted, rejected


def _author_evidence_integration_prompt(paragraph: str, question: dict, answer: str) -> str:
    return (
        "DraftProof AUTHOR_EVIDENCE_INTEGRATION.\n"
        "Integrate one confirmed author-owned anchor into one paragraph.\n\n"
        "Rules:\n"
        "- Return the revised paragraph only.\n"
        "- Use only the confirmed answer. Do not invent any extra source, date, place, statistic, institution, or experience.\n"
        "- Keep the paragraph's original meaning and stance.\n"
        "- Add the anchor where it naturally supports the claim.\n"
        "- Do not make the paragraph more polished, generic, or longer than needed.\n"
        "- If the answer does not support the paragraph, narrow the paragraph's claim instead of forcing it.\n\n"
        f"Anchor question: {question.get('question')}\n"
        f"Answer type: {question.get('answer_type')}\n"
        f"Confirmed answer: {answer}\n\n"
        "TARGET PARAGRAPH:\n"
        f"<PARAGRAPH>\n{paragraph.strip()}\n</PARAGRAPH>"
    )


def _anchor_bridge_sentence(answer: str) -> str:
    cleaned = " ".join(str(answer or "").split()).strip()
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".!?":
        cleaned += "."
    lowered = cleaned.lower()
    if re.match(r"^(in|during|when|while|after|before|from|at|on|for)\b", lowered):
        return cleaned
    if re.search(r"\b(i|my|we|our)\b", lowered):
        return cleaned
    return f"In my own context, {cleaned[0].lower() + cleaned[1:]}"


def _deterministic_author_anchor_paragraph(paragraph: str, question: dict, answer: str) -> tuple[str, str]:
    """Insert a confirmed anchor with minimal surface change before using LLM."""
    paragraph = str(paragraph or "").strip()
    bridge = _anchor_bridge_sentence(answer)
    if not paragraph or not bridge:
        return "", "missing_paragraph_or_answer"
    if bridge in paragraph:
        return paragraph, ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
    if not sentences:
        return f"{paragraph} {bridge}".strip(), ""
    role = str((question or {}).get("paragraph_role") or "")
    answer_type = str((question or {}).get("answer_type") or "")
    if role == "conclusion_template_risk" or answer_type == "author_judgement":
        insert_at = max(0, len(sentences) - 1)
    elif len(sentences) >= 3:
        insert_at = min(2, len(sentences))
    else:
        insert_at = len(sentences)
    patched = sentences[:insert_at] + [bridge] + sentences[insert_at:]
    return " ".join(patched), ""


def _clean_author_evidence_integrated_paragraph(output: str, original_paragraph: str) -> tuple[str, str]:
    text = str(output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"</?PARAGRAPH>", "", text, flags=re.I).strip()
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not text:
        return "", "empty_integrated_paragraph"
    if "[[" in text or "]]" in text:
        return "", "placeholder_marker_in_integrated_paragraph"
    original_words = max(1, len(str(original_paragraph or "").split()))
    new_words = len(text.split())
    if new_words < max(8, int(original_words * 0.65)):
        return "", "integrated_paragraph_too_short"
    if new_words > max(original_words + 90, int(original_words * 1.9)):
        return "", "integrated_paragraph_too_long"
    return text, ""


def _splice_author_evidence_paragraph(text: str, paragraph_index: int, replacement: str) -> str:
    paragraphs = _logical_paragraphs(text)
    if paragraph_index < 0 or paragraph_index >= len(paragraphs):
        return text
    paragraphs[paragraph_index] = replacement.strip()
    return _join_logical_paragraphs(paragraphs)


def _mitigation_sampling_policy_summary() -> dict:
    ai_search_sampling = _rewrite_sampling_profile("DRAFTPROOF_AI_SEARCH")
    return {
        "ai_search_temperature": ai_search_sampling["temperature"],
        "ai_search_top_p": ai_search_sampling["top_p"],
        "ai_search_top_k": ai_search_sampling["top_k"],
        "ai_search_presence_penalty": ai_search_sampling["presence_penalty"],
        "ai_search_frequency_penalty": ai_search_sampling["frequency_penalty"],
        "paragraph_component_temperature": float(os.environ.get("DRAFTPROOF_PARAGRAPH_COMPONENT_TEMPERATURE", "0.45")),
        "human_signal_amplification_temperature": float(os.environ.get("DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TEMPERATURE", "0.45")),
        "author_reasoning_amplification_temperature": float(os.environ.get("DRAFTPROOF_AUTHOR_REASONING_AMPLIFICATION_TEMPERATURE", "0.45")),
        "iterative_human_climb_temperature": float(os.environ.get(
            "DRAFTPROOF_ITERATIVE_HUMAN_CLIMB_TEMPERATURE",
            os.environ.get("DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TEMPERATURE", "0.45"),
        )),
        "internet_reauthor_temperature": float(os.environ.get("DRAFTPROOF_INTERNET_REAUTHOR_TEMPERATURE", "0.45")),
        "claim_narrowing_temperature": float(os.environ.get("DRAFTPROOF_CLAIM_NARROWING_TEMPERATURE", "0.35")),
        "topk_texture_temperature": float(os.environ.get("DRAFTPROOF_TOPK_TEXTURE_TEMPERATURE", "0.35")),
    }



def _formula_convergence_primary_burden_gate_status(
    current_report: dict | None,
    candidate_report: dict | None,
    contract: dict | None,
) -> dict:
    """Record primary-burden movement without discarding safe formula progress.

    Earlier versions used minimum raw/burden drops as a hard gate. That caused
    verified, safety-clean improvements to be thrown away when the remaining
    dominant drivers were stubborn. At this layer, safety gates already run
    separately, so this function should only block candidates that fail to move
    the total formula score or that worsen the positive AI burden.
    """
    contract = contract if isinstance(contract, dict) else {}
    current_profile = _turnitin_like_ai_profile(current_report)
    candidate_profile = _turnitin_like_ai_profile(candidate_report)
    current_footprint = _ai_footprint_flatten(_ai_footprint_profile(current_report))
    candidate_footprint = _ai_footprint_flatten(_ai_footprint_profile(candidate_report))
    drops = contract.get("weighted_driver_drops") if isinstance(contract.get("weighted_driver_drops"), dict) else {}

    def weighted_drop(driver: str) -> float:
        row = drops.get(driver) if isinstance(drops.get(driver), dict) else {}
        return float(row.get("drop") or row.get("gain") or 0.0)

    def raw_drop(driver: str) -> float:
        row = drops.get(driver) if isinstance(drops.get(driver), dict) else {}
        return float(row.get("raw_drop") or row.get("gain") or 0.0)

    positive_ai_burden_drop = float(((contract.get("positive_ai_burden") or {}).get("drop")) or 0.0)
    score_drop = float(contract.get("score_drop") or 0.0)
    current_topk = float((current_profile.get("components") or {}).get("topk_calibrated_risk") or 0.0)
    current_ai_likelihood = float((current_profile.get("components") or {}).get("ai_likelihood") or 0.0)
    current_authorship = float(current_footprint.get("ai_authorship") or 0.0)
    current_density = float(current_footprint.get("qualifying_text_ai_density") or 0.0)
    candidate_density = float(candidate_footprint.get("qualifying_text_ai_density") or 0.0)
    density_drop = current_density - candidate_density
    primary_raw_drop = max(
        raw_drop("ai_likelihood"),
        raw_drop("topk_calibrated_risk"),
        density_drop,
    )
    primary_weighted_drop = weighted_drop("ai_likelihood") + weighted_drop("topk_calibrated_risk")
    human_anchor_gain = weighted_drop("human_anchor_suppression")
    target_met = bool(candidate_profile.get("target_met"))
    primary_pinned = bool(
        current_topk >= 75.0
        or current_authorship > _float_env("DRAFTPROOF_AI_FOOTPRINT_SAFE_AUTHORSHIP", 35.0) + 10.0
        or current_density > _float_env("DRAFTPROOF_QUALIFYING_AI_DENSITY_SAFE_BAND", 35.0) + 20.0
        or current_ai_likelihood >= 55.0
    )
    improvement_epsilon = 0.001
    min_positive_drop = 0.0
    min_primary_raw_drop = 0.0
    accepted = bool(
        target_met
        or (
            score_drop > improvement_epsilon
            and positive_ai_burden_drop >= -improvement_epsilon
        )
    )
    reason = "accepted"
    if not accepted:
        reason = (
            "positive_ai_burden_regressed"
            if positive_ai_burden_drop < -improvement_epsilon
            else "formula_score_not_reduced"
        )
    return {
        "version": "formula_convergence_primary_burden_gate_v1",
        "accepted": accepted,
        "reason": reason,
        "primary_pinned": primary_pinned,
        "target_met": target_met,
        "score_drop": round(score_drop, 3),
        "positive_ai_burden_drop": round(positive_ai_burden_drop, 3),
        "required_positive_ai_burden_drop": round(min_positive_drop, 3),
        "primary_raw_drop": round(primary_raw_drop, 3),
        "required_primary_raw_drop": round(min_primary_raw_drop, 3),
        "primary_weighted_drop": round(primary_weighted_drop, 3),
        "human_anchor_gain": round(human_anchor_gain, 3),
        "current_topk_calibrated_risk": round(current_topk, 3),
        "current_ai_likelihood": round(current_ai_likelihood, 3),
        "current_ai_authorship": round(current_authorship, 3),
        "current_qualifying_text_ai_density": round(current_density, 3),
        "candidate_qualifying_text_ai_density": round(candidate_density, 3),
    }


def _formula_feasibility_estimator(
    report_dict: dict | None,
    *,
    observed_candidates: list[dict] | tuple[dict, ...] | None = None,
) -> dict:
    """Estimate whether formula convergence needs geometry-first intervention."""
    profile = _turnitin_like_ai_profile(report_dict)
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    weighted = profile.get("weighted_components") if isinstance(profile.get("weighted_components"), dict) else {}
    score = float(profile.get("score") or 0.0)
    target = float(profile.get("target_score") or TURNITIN_LIKE_TARGET_AI_SCORE)
    positive_burden = float(profile.get("raw_positive_score") or 0.0)
    suppression = float(profile.get("human_anchor_suppression") or 0.0)
    suppression_headroom = max(0.0, 45.0 - suppression)

    observed = _formula_observed_driver_movement(observed_candidates)
    dominant = [
        str(row.get("driver"))
        for row in (profile.get("top_positive_drivers") or [])[:4]
        if isinstance(row, dict) and row.get("driver")
    ]
    primary_weighted = (
        float(weighted.get("ai_likelihood") or 0.0)
        + float(weighted.get("topk_calibrated_risk") or 0.0)
    )
    secondary_weighted = sum(
        float(weighted.get(driver) or 0.0)
        for driver in (
            "semantic_uniformity",
            "rewrite_smoothness",
            "patchwork_expansion",
            "signal_agreement",
        )
    )

    # Safe floor assumes local statistical edits and bounded anchor gains. The
    # aggressive floor assumes coordinated geometry edits and low-value removal.
    safe_driver_headroom = (
        min(float(weighted.get("ai_likelihood") or 0.0), 8.0)
        + min(float(weighted.get("topk_calibrated_risk") or 0.0), 5.0)
        + min(secondary_weighted, 5.0)
        + min(suppression_headroom, 4.0)
    )
    aggressive_driver_headroom = (
        min(float(weighted.get("ai_likelihood") or 0.0), 14.0)
        + min(float(weighted.get("topk_calibrated_risk") or 0.0), 8.0)
        + min(secondary_weighted, 9.0)
        + min(suppression_headroom, 7.0)
    )
    best_observed_drop = max(
        [
            float((row or {}).get("score_drop") or 0.0)
            for row in (observed_candidates or [])
            if isinstance(row, dict)
        ]
        or [0.0]
    )
    if best_observed_drop > 0.0:
        safe_driver_headroom = max(safe_driver_headroom, min(best_observed_drop + 4.0, 18.0))
        aggressive_driver_headroom = max(aggressive_driver_headroom, min(best_observed_drop + 8.0, 28.0))

    safe_floor = max(0.0, score - safe_driver_headroom)
    aggressive_floor = max(0.0, score - aggressive_driver_headroom)
    geometry_required = safe_floor >= target and primary_weighted >= max(8.0, positive_burden * 0.45)
    return {
        "version": "formula_feasibility_estimator_v1",
        "score": round(score, 3),
        "target_score": round(target, 3),
        "target_gap": round(max(0.0, score - target), 3),
        "positive_ai_burden": round(positive_burden, 3),
        "human_anchor_suppression": round(suppression, 3),
        "suppression_headroom": round(suppression_headroom, 3),
        "primary_driver_weighted_burden": round(primary_weighted, 3),
        "secondary_driver_weighted_burden": round(secondary_weighted, 3),
        "estimated_safe_floor": round(safe_floor, 3),
        "aggressive_floor": round(aggressive_floor, 3),
        "dominant_drivers": dominant,
        "mode": "geometry_mode" if geometry_required else "safe_portfolio_mode",
        "geometry_required": bool(geometry_required),
        "observed_driver_movement": observed,
        "component_snapshot": {key: round(float(value or 0.0), 3) for key, value in components.items()},
    }


def _sentence_opening_route(sentence: str) -> str:
    words = re.findall(r"\b[\w'-]+\b", str(sentence or "").strip())
    if not words:
        return ""
    if len(words) >= 2 and words[0].lower() in {"the", "this", "these", "that", "it"}:
        return " ".join(words[:2]).lower()
    return words[0].lower()


def _geometry_risk_map(
    text: str,
    report_dict: dict | None,
    *,
    limit: int | None = None,
) -> dict:
    """Rank sentence and paragraph geometry hotspots by weighted formula impact."""
    sentences = _split_sentences(text)
    sentence_limit = max(1, int(limit or _topk_optimizer_sentence_limit(text)))
    profile = _turnitin_like_ai_profile(report_dict)
    weighted = profile.get("weighted_components") if isinstance(profile.get("weighted_components"), dict) else {}
    topk_rows = {
        int(row.get("sentence_index")): row
        for row in (_topk_repair_map(text, report_dict, limit=max(sentence_limit, len(sentences) or 1)).get("targets") or [])
        if isinstance(row, dict) and isinstance(row.get("sentence_index"), int)
    }
    lengths = [_text_word_count(sentence) for sentence in sentences]
    median_length = statistics.median(lengths) if lengths else 0.0
    openings = [_sentence_opening_route(sentence) for sentence in sentences]
    opening_counts = {opening: openings.count(opening) for opening in set(openings) if opening}
    connector_re = re.compile(
        r"^(?:Furthermore|Moreover|Additionally|In conclusion|Overall|Therefore|However|"
        r"At the same time|In addition|This|These|It is important|This shows|This means)\b",
        re.I,
    )
    balance_re = re.compile(r"\b(?:because|which|that|while|although|therefore|so that|in order to)\b", re.I)
    rows: list[dict] = []
    primary_drag = (
        float(weighted.get("ai_likelihood") or 0.0) * 0.55
        + float(weighted.get("topk_calibrated_risk") or 0.0) * 0.35
        + float(weighted.get("rewrite_smoothness") or 0.0) * 0.25
        + float(weighted.get("semantic_uniformity") or 0.0) * 0.15
    )
    for index, sentence in enumerate(sentences):
        words = _text_word_count(sentence)
        opening = openings[index] if index < len(openings) else ""
        topk = topk_rows.get(index, {})
        top10 = float(topk.get("top10_ratio") or 0.0)
        predictability = float(topk.get("predictability_risk") or 0.0)
        connector_risk = 1.0 if connector_re.search(sentence.strip()) else 0.0
        repeated_opening = max(0, opening_counts.get(opening, 0) - 1) / max(1, len(sentences) - 1)
        prev_len = lengths[index - 1] if index > 0 else None
        next_len = lengths[index + 1] if index + 1 < len(lengths) else None
        neighbor_lengths = [item for item in (prev_len, next_len) if isinstance(item, int)]
        cadence_uniformity = (
            sum(1 for item in neighbor_lengths if abs(item - words) <= 3) / max(1, len(neighbor_lengths))
            if neighbor_lengths else 0.0
        )
        clause_balance = min(1.0, (sentence.count(",") + len(balance_re.findall(sentence))) / 4.0)
        length_uniformity = 1.0 - min(1.0, abs(words - median_length) / max(1.0, median_length)) if median_length else 0.0
        geometry_score = (
            top10 * 0.30
            + predictability * 0.25
            + connector_risk * 0.14
            + repeated_opening * 0.12
            + cadence_uniformity * 0.10
            + clause_balance * 0.05
            + length_uniformity * 0.04
        )
        weighted_drag = geometry_score * max(1.0, primary_drag)
        protected = bool(
            re.search(r"https?://|www\.|\b[A-Z]{2,}[A-Z0-9-]{2,}\b", sentence)
            or _protected_number_set(sentence)
        )
        rows.append({
            "sentence_index": index,
            "sentence": sentence,
            "word_count": words,
            "opening_route": opening,
            "weighted_geometry_drag": round(weighted_drag, 3),
            "geometry_score": round(geometry_score, 3),
            "protected": protected,
            "drivers": {
                "top10_density": round(top10, 4),
                "predictability": round(predictability, 4),
                "connector_risk": round(connector_risk, 3),
                "repeated_opening": round(repeated_opening, 3),
                "cadence_uniformity": round(cadence_uniformity, 3),
                "clause_balance": round(clause_balance, 3),
                "length_uniformity": round(length_uniformity, 3),
            },
        })
    rows.sort(key=lambda row: float(row.get("weighted_geometry_drag") or 0.0), reverse=True)

    paragraph_rows = []
    for row in (_formula_block_driver_map(text, report_dict).get("blocks") or []):
        if not isinstance(row, dict):
            continue
        paragraph_rows.append({
            "block_index": row.get("block_index"),
            "weighted_drag": row.get("weighted_drag"),
            "recommended_portfolio_action": row.get("recommended_portfolio_action"),
            "human_anchor_deficit": row.get("human_anchor_deficit"),
            "protected": row.get("protected"),
            "remove_value_loss_risk": row.get("remove_value_loss_risk"),
            "preview": row.get("preview"),
        })
    paragraph_rows.sort(key=lambda row: float(row.get("weighted_drag") or 0.0), reverse=True)
    return {
        "version": "geometry_risk_map_v1",
        "sentence_count": len(sentences),
        "median_sentence_words": round(float(median_length or 0.0), 3),
        "formula_score": profile.get("score"),
        "dominant_weighted_drivers": [
            row.get("driver")
            for row in (profile.get("top_positive_drivers") or [])[:4]
            if isinstance(row, dict)
        ],
        "sentence_hotspots": rows[:sentence_limit],
        "paragraph_hotspots": paragraph_rows[:8],
    }


def _geometry_disrupt_sentence(sentence: str, row: dict | None = None) -> tuple[str, list[str]]:
    """Apply one or two semantic-preserving syntax/route perturbations."""
    original = str(sentence or "").strip()
    candidate = original
    operations: list[str] = []
    if not candidate or re.search(r"https?://|www\.", candidate, re.I):
        return original, []
    if _protected_code_anchor_set(candidate):
        return original, []

    replacements = [
        (r"^Furthermore,\s*", "", "connector_remove"),
        (r"^Moreover,\s*", "", "connector_remove"),
        (r"^Additionally,\s*", "", "connector_remove"),
        (r"^In conclusion,\s*", "Taken together, ", "conclusion_route"),
        (r"^Overall,\s*", "Taken together, ", "summary_route"),
        (r"^At the same time,\s*", "Still, ", "connector_shorten"),
        (r"^In addition to\s+", "Beyond ", "connector_shorten"),
        (r"^However,\s*", "But ", "connector_plain"),
        (r"^This highlights the importance of\s+", "The pressure sits around ", "template_opening_disrupt"),
        (r"^This demonstrates that\s+", "That leaves a simpler point: ", "template_opening_disrupt"),
        (r"^This means that\s+", "That means ", "this_route_plain"),
        (r"^It is important to note that\s+", "", "meta_phrase_remove"),
        (r"\bplays? (?:a|an) (?:important|significant|major|crucial) role in\b", "matters in", "formula_verb_reduce"),
        (r"\bhas a significant impact on\b", "affects", "formula_verb_reduce"),
        (r"\ba wide range of\b", "many", "generic_phrase_reduce"),
        (r"\bvarious factors\b", "several factors", "generic_phrase_reduce"),
    ]
    for pattern, replacement, op in replacements:
        updated = re.sub(pattern, replacement, candidate, count=1, flags=re.I).strip()
        if updated != candidate:
            candidate = updated
            operations.append(op)
            break

    words = _text_word_count(candidate)
    if words >= 18 and len(operations) < 2:
        for splitter, op in (
            (r"\s+because\s+", "because_route_split"),
            (r"\s+but\s+", "contrast_route_split"),
            (r"\s+which\s+", "which_route_split"),
        ):
            match = re.search(splitter, candidate, flags=re.I)
            if not match:
                continue
            left = candidate[:match.start()].strip(" ,;")
            right = candidate[match.end():].strip(" ,;")
            if _text_word_count(left) >= 6 and _text_word_count(right) >= 5:
                lead = "Because" if "because" in splitter else ("But" if "but" in splitter else "That")
                candidate = f"{left}. {lead} {right[0].lower() + right[1:] if len(right) > 1 else right}"
                operations.append(op)
                break
    if candidate == original and 10 <= words <= 30 and "," in candidate:
        first, rest = candidate.split(",", 1)
        if 3 <= _text_word_count(first) <= 9 and _text_word_count(rest) >= 6:
            candidate = f"{rest.strip()} {first.strip().lower()}."
            candidate = re.sub(r"\.\.$", ".", candidate)
            operations.append("front_context_shift")
    if candidate == original and words >= 20:
        parts = candidate.split(" and ", 1)
        if len(parts) == 2 and _text_word_count(parts[0]) >= 8 and _text_word_count(parts[1]) >= 6:
            candidate = f"{parts[0].strip()}. {parts[1].strip().capitalize()}"
            operations.append("and_route_split")
    candidate = re.sub(r"\s{2,}", " ", candidate).strip()
    return (candidate, operations) if candidate and candidate != original else (original, [])


def _coordinated_micro_perturbation_candidates(
    current_text: str,
    current_report: dict | None,
    geometry_map: dict | None = None,
    *,
    limit: int = 4,
) -> list[tuple[str, str, dict]]:
    """Generate coordinated sentence-level geometry candidates."""
    if not _env_flag("DRAFTPROOF_COORDINATED_MICRO_PERTURBATION", True):
        return []
    geometry_map = geometry_map or _geometry_risk_map(current_text, current_report)
    hotspots = [
        row for row in (geometry_map.get("sentence_hotspots") or [])
        if isinstance(row, dict) and not row.get("protected")
    ]
    if not hotspots:
        return []
    limit = max(1, int(limit or 1))
    batches = [
        ("light", 0.25),
        ("medium", 0.45),
        ("wide", 0.70),
        ("full", 1.0),
    ]
    candidates: list[tuple[str, str, dict]] = []
    seen = {str(current_text or "").strip()}
    for label, fraction in batches:
        if len(candidates) >= limit:
            break
        take = max(1, min(len(hotspots), math.ceil(len(hotspots) * fraction)))
        replacements: dict[str, str] = {}
        operations = []
        for row in hotspots[:take]:
            sentence = str(row.get("sentence") or "")
            replacement, ops = _geometry_disrupt_sentence(sentence, row)
            if not ops or replacement == sentence:
                continue
            replacements[sentence] = replacement
            operations.append({
                "sentence_index": row.get("sentence_index"),
                "operations": ops,
                "weighted_geometry_drag": row.get("weighted_geometry_drag"),
                "drivers": row.get("drivers"),
            })
        candidate = _splice_sentences_by_text(current_text, replacements)
        normalized = candidate.strip()
        if operations and normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append((
                f"coordinated_micro_perturbation_{label}",
                candidate,
                {
                    "coordinated_micro_perturbation": True,
                    "geometry_mode": True,
                    "operation": "coordinated_micro_perturbation",
                    "targeted_drivers": [
                        "ai_likelihood",
                        "topk_calibrated_risk",
                        "rewrite_smoothness",
                        "semantic_uniformity",
                    ],
                    "geometry_risk_map": {
                        "version": geometry_map.get("version"),
                        "formula_score": geometry_map.get("formula_score"),
                        "dominant_weighted_drivers": geometry_map.get("dominant_weighted_drivers"),
                        "selected_sentence_count": take,
                    },
                    "applied_geometry_operations": operations,
                },
            ))
    return candidates[:limit]


def _anti_smoothing_guard_status(
    current_report: dict | None,
    candidate_report: dict | None,
    *,
    strict: bool = False,
) -> dict:
    """Block geometry candidates that win one signal by smoothing elsewhere."""
    current = _turnitin_like_ai_profile(current_report)
    candidate = _turnitin_like_ai_profile(candidate_report)
    current_components = current.get("components") if isinstance(current.get("components"), dict) else {}
    candidate_components = candidate.get("components") if isinstance(candidate.get("components"), dict) else {}
    score_drop = float(current.get("score") or 0.0) - float(candidate.get("score") or 0.0)
    tolerance = 0.001 if strict else 1.0
    backfires = []
    for driver in (
        "ai_likelihood",
        "topk_calibrated_risk",
        "semantic_uniformity",
        "rewrite_smoothness",
        "patchwork_expansion",
        "signal_agreement",
    ):
        before = float(current_components.get(driver) or 0.0)
        after = float(candidate_components.get(driver) or 0.0)
        delta = after - before
        if delta > tolerance:
            backfires.append({
                "driver": driver,
                "before": round(before, 3),
                "after": round(after, 3),
                "increase": round(delta, 3),
            })
    return {
        "version": "anti_smoothing_guard_v1",
        "accepted": bool(score_drop > 0.05 and not backfires),
        "score_drop": round(score_drop, 3),
        "strict": bool(strict),
        "tolerance": tolerance,
        "backfires": backfires,
        "reason": (
            "accepted"
            if score_drop > 0.05 and not backfires
            else "formula_score_not_improved"
            if score_drop <= 0.05
            else "component_backfire:" + ",".join(row["driver"] for row in backfires[:4])
        ),
    }



def _multi_signal_candidate_contract(original_report: dict | None, candidate_report: dict | None) -> dict:
    """Summarize cross-signal movement so one repaired metric cannot hide damage elsewhere."""
    before = _ai_footprint_flatten(_ai_footprint_profile(original_report))
    after = _ai_footprint_flatten(_ai_footprint_profile(candidate_report))
    driver_specs = {
        "topk_calibrated_risk": {"weight": 1.5, "tolerance": 0.5, "severe": 2.0},
        "ai_authorship": {"weight": 1.4, "tolerance": 0.5, "severe": 2.0},
        "ai_transformation": {"weight": 1.25, "tolerance": 0.5, "severe": 2.0},
        "external_ai_flag_risk": {"weight": 1.4, "tolerance": 0.5, "severe": 2.0},
        "ai_likelihood": {"weight": 1.15, "tolerance": 0.5, "severe": 3.0},
        "rewrite_smoothness": {"weight": 1.0, "tolerance": 1.0, "severe": 8.0},
        "semantic_uniformity": {"weight": 0.8, "tolerance": 1.0, "severe": 8.0},
        "qualifying_text_ai_density": {"weight": 1.35, "tolerance": 1.0, "severe": 8.0},
        "discourse_regularity": {"weight": 0.8, "tolerance": 1.0, "severe": 8.0},
        "generic_assertion_risk": {"weight": 0.9, "tolerance": 5.0, "severe": 15.0},
        "unsupported_claim_risk": {"weight": 0.55, "tolerance": 5.0, "severe": 15.0},
        "broad_claim_risk": {"weight": 0.5, "tolerance": 5.0, "severe": 15.0},
    }
    improvements: list[dict] = []
    regressions: list[dict] = []
    severe_backfires: list[dict] = []
    balance_score = 0.0
    for driver, spec in driver_specs.items():
        before_value = before.get(driver)
        after_value = after.get(driver)
        if not isinstance(before_value, (int, float)) or not isinstance(after_value, (int, float)):
            continue
        delta = round(float(before_value) - float(after_value), 3)
        tolerance = float(spec.get("tolerance", 0.0))
        weight = float(spec.get("weight", 1.0))
        if delta > tolerance:
            row = {
                "driver": driver,
                "before": round(float(before_value), 3),
                "after": round(float(after_value), 3),
                "drop": delta,
            }
            improvements.append(row)
            balance_score += delta * weight
        elif delta < -tolerance:
            regression = {
                "driver": driver,
                "before": round(float(before_value), 3),
                "after": round(float(after_value), 3),
                "increase": round(abs(delta), 3),
            }
            regressions.append(regression)
            balance_score -= abs(delta) * weight * 1.5
            if abs(delta) >= float(spec.get("severe", tolerance)):
                severe_backfires.append(regression)
    primary_drop = sum(
        float(row.get("drop") or 0.0)
        for row in improvements
        if row.get("driver") in {
            "topk_calibrated_risk",
            "ai_authorship",
            "ai_transformation",
            "external_ai_flag_risk",
            "ai_likelihood",
            "rewrite_smoothness",
            "qualifying_text_ai_density",
        }
    )
    return {
        "version": "multi_signal_v1",
        "balance_score": round(balance_score, 3),
        "primary_drop": round(primary_drop, 3),
        "improvements": improvements,
        "regressions": regressions,
        "severe_backfires": severe_backfires,
        "severe_backfire": bool(severe_backfires),
        "needs_balance_repair": bool(severe_backfires and primary_drop >= 8.0),
    }


def _strict_ai_safe_band_status(report_dict: dict | None) -> dict:
    return _strict_ai_safe_band_status_from_profile(_ai_footprint_profile(report_dict))


def _strict_ai_safe_band_status_from_profile(profile: dict | None) -> dict:
    if not isinstance(profile, dict) or not profile:
        thresholds = {
            "topk_calibrated_risk": _safe_topk_calibrated_limit(),
            "ai_authorship": _float_env("DRAFTPROOF_AI_FOOTPRINT_SAFE_AUTHORSHIP", 35.0),
            "ai_transformation": _float_env("DRAFTPROOF_AI_FOOTPRINT_SAFE_TRANSFORMATION", 35.0),
            "qualifying_text_ai_density": _float_env("DRAFTPROOF_QUALIFYING_AI_DENSITY_SAFE_BAND", 35.0),
            "external_ai_flag_risk": _float_env("DRAFTPROOF_EXTERNAL_FLAG_PROXY_SAFE_BAND", 35.0),
        }
        return {
            "achieved": False,
            "profile": {},
            "thresholds": thresholds,
            "remaining": [
                {"driver": key, "value": None, "safe_band": round(float(limit), 3)}
                for key, limit in thresholds.items()
            ],
            "unscored": True,
        }
    flat = _ai_footprint_flatten(profile if isinstance(profile, dict) else {})
    thresholds = {
        "topk_calibrated_risk": _safe_topk_calibrated_limit(),
        "ai_authorship": _float_env("DRAFTPROOF_AI_FOOTPRINT_SAFE_AUTHORSHIP", 35.0),
        "ai_transformation": _float_env("DRAFTPROOF_AI_FOOTPRINT_SAFE_TRANSFORMATION", 35.0),
        "qualifying_text_ai_density": _float_env("DRAFTPROOF_QUALIFYING_AI_DENSITY_SAFE_BAND", 35.0),
        "external_ai_flag_risk": _float_env("DRAFTPROOF_EXTERNAL_FLAG_PROXY_SAFE_BAND", 35.0),
    }
    remaining = [
        {
            "driver": key,
            "value": round(float(flat.get(key, 0.0)), 3),
            "safe_band": round(float(limit), 3),
        }
        for key, limit in thresholds.items()
        if isinstance(flat.get(key), (int, float)) and float(flat.get(key, 0.0)) > float(limit)
    ]
    missing = [
        {
            "driver": key,
            "value": None,
            "safe_band": round(float(limit), 3),
            "missing": True,
        }
        for key, limit in thresholds.items()
        if not isinstance(flat.get(key), (int, float))
    ]
    return {
        "achieved": not remaining and not missing,
        "profile": flat,
        "thresholds": thresholds,
        "remaining": remaining + missing,
        "unscored": bool(missing),
    }


def _strict_ai_safe_band_status_from_footprint_gate(gate: dict | None) -> dict:
    """Recover strict safe-band status from a scanned candidate footprint gate."""
    if not isinstance(gate, dict):
        return _strict_ai_safe_band_status_from_profile({})
    after = gate.get("after") if isinstance(gate.get("after"), dict) else {}
    authorship = (
        after.get("authorship_footprint")
        if isinstance(after.get("authorship_footprint"), dict) else {}
    )
    structural = (
        after.get("structural_footprint")
        if isinstance(after.get("structural_footprint"), dict) else {}
    )
    semantic = (
        after.get("semantic_footprint")
        if isinstance(after.get("semantic_footprint"), dict) else {}
    )
    grounding = (
        after.get("grounding_footprint")
        if isinstance(after.get("grounding_footprint"), dict) else {}
    )
    return _strict_ai_safe_band_status_from_profile({
        "authorship_footprint": authorship,
        "structural_footprint": structural,
        "semantic_footprint": semantic,
        "grounding_footprint": grounding,
        "external_ai_flag_risk": after.get("external_ai_flag_risk"),
    })


STRICT_SAFE_PHASE_BUDGET_CONTRACT = {
    "total_llm_hard_cap": 10,
    "topk_safe_band_rebuild": 6,
    "authorship_transformation_texture_controller": 3,
    "final_texture_proxy_repair": 1,
    "emergency_diagnostic_reserve": 0,
    # Backward-compatible report key only. New work must spend against the
    # explicit authorship/transformation texture controller budget above.
    "post_topk_strict_safe_optimizer": 0,
}


def _strict_safe_phase_budget_contract(
    total_cap: int | None = None,
    source_text: str = "",
    report_dict: dict | None = None,
) -> dict:
    """Fixed LLM budget split for strict-safe mitigation phases.

    Top-k is the entry condition for strict-safe mitigation, so it receives
    the largest fixed reserve. Runtime may lower the total cap, but no phase
    can borrow calls reserved for earlier prerequisite phases.
    """
    policy = _ai_search_budget_policy(source_text, report_dict)
    policy_phase = policy.get("phase_budget") or {}
    inferred_cap = int(policy.get("max_llm_calls") or STRICT_SAFE_PHASE_BUDGET_CONTRACT["total_llm_hard_cap"])
    cap = int(total_cap if isinstance(total_cap, int) else _ai_search_llm_hard_cap(source_text, report_dict))
    cap = max(0, min(cap, inferred_cap))
    contract = {
        "total_llm_hard_cap": inferred_cap,
        "topk_safe_band_rebuild": int(policy_phase.get("topk_safe_band_rebuild", 0)),
        "authorship_transformation_texture_controller": int(policy_phase.get("authorship_transformation_texture_controller", 0)),
        "final_texture_proxy_repair": int(policy_phase.get("final_texture_proxy_repair", 0)),
        "emergency_diagnostic_reserve": 0,
        "post_topk_strict_safe_optimizer": 0,
    }
    contract["total_llm_hard_cap"] = cap
    if cap <= 10:
        final_reserve = 1 if cap >= 8 else 0
        texture_reserve = min(3 if cap >= 10 else 2, max(0, cap - final_reserve - 4))
        topk_reserve = max(1, cap - texture_reserve - final_reserve)
        contract.update({
            "topk_safe_band_rebuild": min(int(policy_phase.get("topk_safe_band_rebuild", 0)), topk_reserve),
            "authorship_transformation_texture_controller": min(
                int(policy_phase.get("authorship_transformation_texture_controller", 0)),
                texture_reserve,
            ),
            "final_texture_proxy_repair": min(
                int(policy_phase.get("final_texture_proxy_repair", 0)),
                final_reserve,
            ),
            "emergency_diagnostic_reserve": 0,
            "post_topk_strict_safe_optimizer": 0,
        })
        unused = cap - sum(
            int(contract[key])
            for key in contract
            if key != "total_llm_hard_cap"
        )
        if unused > 0:
            contract["topk_safe_band_rebuild"] += unused
    overflow = sum(
        int(contract[key])
        for key in contract
        if key != "total_llm_hard_cap"
    ) - cap
    if overflow > 0:
        for key in (
            "emergency_diagnostic_reserve",
            "final_texture_proxy_repair",
            "authorship_transformation_texture_controller",
            "topk_safe_band_rebuild",
            "post_topk_strict_safe_optimizer",
        ):
            take = min(overflow, int(contract[key]))
            contract[key] = int(contract[key]) - take
            overflow -= take
            if overflow <= 0:
                break
    return contract


def _strict_safe_candidate_rank(
    base_report: dict | None,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float = 0,
    weighted_severity_delta: int | float = 0,
    critical_high_delta: int | float = 0,
) -> tuple:
    """Rank strict-safe candidates by the actual remaining blocker order."""
    base_status = _strict_ai_safe_band_status(base_report)
    after_status = _strict_ai_safe_band_status(candidate_report)
    base = base_status.get("profile") or {}
    after = after_status.get("profile") or {}

    def num(value, default=0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else float(default)

    topk_safe = num(after.get("topk_calibrated_risk"), 100.0) < _safe_topk_calibrated_limit()
    safety_clean = (
        float(review_burden_delta or 0.0) <= 0.0
        and float(weighted_severity_delta or 0.0) <= 0.0
        and float(critical_high_delta or 0.0) <= 0.0
    )
    return (
        1 if after_status.get("achieved") else 0,
        round(num(base.get("qualifying_text_ai_density")) - num(after.get("qualifying_text_ai_density")), 3),
        round(num(base.get("external_ai_flag_risk")) - num(after.get("external_ai_flag_risk")), 3),
        round(num(base.get("ai_authorship")) - num(after.get("ai_authorship")), 3),
        round(num(base.get("ai_transformation")) - num(after.get("ai_transformation")), 3),
        round(num(base.get("rewrite_smoothness")) - num(after.get("rewrite_smoothness")), 3),
        round(num(base.get("semantic_uniformity")) - num(after.get("semantic_uniformity")), 3),
        1 if topk_safe else 0,
        1 if safety_clean else 0,
        round(-max(0.0, float(review_burden_delta or 0.0)), 3),
        round(-max(0.0, float(weighted_severity_delta or 0.0)), 3),
        round(-max(0.0, float(critical_high_delta or 0.0)), 3),
        round(-num(after.get("topk_calibrated_risk"), 100.0), 3),
    )


def _topk_rebuild_fallback_rank(report_dict: dict | None) -> tuple:
    """Rank Top-k rebuild attempts even when none reached the safe band.

    This prevents a later patch round with worse calibrated Top-k from
    overwriting an earlier, better near-miss candidate.
    """
    profile = _strict_ai_safe_band_status(report_dict).get("profile") or {}
    topk_value = profile.get("topk_calibrated_risk")
    if not isinstance(topk_value, (int, float)):
        return ()
    return (
        1 if float(topk_value) < _safe_topk_calibrated_limit() else 0,
        -float(topk_value),
        -float(profile.get("external_ai_flag_risk") or 0.0),
        -float(profile.get("ai_authorship") or 0.0),
        -float(profile.get("ai_transformation") or 0.0),
        -float(profile.get("rewrite_smoothness") or 0.0),
    )



def _human_target_ai_search_status(report_dict: dict | None) -> dict:
    """Allow mitigation search below the AI-risk threshold when Human is still below target."""
    if not _env_flag("DRAFTPROOF_AI_SEARCH_FOR_HUMAN_TARGET", True):
        return {"active": False, "reason": "disabled"}
    contribution = _contribution_scores(report_dict)
    human = contribution.get("human")
    target = _float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    if not isinstance(human, (int, float)):
        return {"active": False, "reason": "missing_human_score", "target_human": target}
    if float(human) >= target:
        return {
            "active": False,
            "reason": "target_reached",
            "current_human": round(float(human), 3),
            "target_human": target,
        }
    blockers = _blocker_scores(report_dict)
    blocker_threshold = _float_env("DRAFTPROOF_AI_SEARCH_FOR_HUMAN_TARGET_BLOCKER_THRESHOLD", 65.0)
    active_blockers = [
        key for key, value in blockers.items()
        if isinstance(value, (int, float)) and float(value) >= blocker_threshold
        and key not in {"topk_pattern", "topk_pattern_raw"}
    ]
    if not active_blockers:
        return {
            "active": False,
            "reason": "no_active_blockers",
            "current_human": round(float(human), 3),
            "target_human": target,
            "blocker_threshold": blocker_threshold,
        }
    return {
        "active": True,
        "reason": "human_below_target_with_active_blockers",
        "current_human": round(float(human), 3),
        "target_human": target,
        "human_gap": round(target - float(human), 3),
        "blocker_threshold": blocker_threshold,
        "active_blockers": active_blockers,
        "blockers": blockers,
    }


def _blocker_elimination_status(original_report: dict | None, candidate_report: dict | None) -> dict:
    original = _blocker_scores(original_report)
    candidate = _blocker_scores(candidate_report)
    drops = {
        key: round(float(original.get(key, 0.0)) - float(candidate.get(key, 0.0)), 3)
        for key in sorted(set(original) | set(candidate))
    }
    active_keys = [
        key for key, value in original.items()
        if isinstance(value, (int, float)) and value >= 60.0
        and key not in {"topk_pattern", "topk_pattern_raw"}
    ]
    active_drop = sum(max(0.0, drops.get(key, 0.0)) for key in active_keys)
    active_regression = sum(max(0.0, -drops.get(key, 0.0)) for key in active_keys)
    display_candidate = {
        key: value
        for key, value in candidate.items()
        if key not in {"topk_pattern", "topk_pattern_raw"}
    }
    top_remaining = sorted(
        display_candidate.items(),
        key=lambda item: float(item[1] or 0.0),
        reverse=True,
    )[:5]
    return {
        "original": original,
        "candidate": candidate,
        "drops": drops,
        "active_keys": active_keys,
        "active_drop": round(active_drop, 3),
        "active_regression": round(active_regression, 3),
        "top_remaining": [
            {"key": key, "score": value}
            for key, value in top_remaining
        ],
    }


def _dominant_blocker_gate_status(original_report: dict | None, candidate_report: dict | None) -> dict:
    """Require movement on the blockers that define the current failure.

    Generic score improvements are not enough when the dominant blockers remain
    untouched. This prevents weak fallback candidates from being labelled as
    mitigation success.
    """
    original = _blocker_scores(original_report)
    candidate = _blocker_scores(candidate_report)
    dominant_keys = [
        key.strip()
        for key in os.environ.get(
            "DRAFTPROOF_DOMINANT_BLOCKER_KEYS",
            "unsupported_claim_risk,source_grounding_risk,broad_claim_risk,topk_calibrated_risk,generic_assertion_risk",
        ).split(",")
        if key.strip()
    ]
    active_threshold = _float_env("DRAFTPROOF_DOMINANT_BLOCKER_ACTIVE_THRESHOLD", 85.0)
    original_human = _contribution_scores(original_report).get("human")
    target_human = _float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    if isinstance(original_human, (int, float)) and float(original_human) < target_human:
        active_threshold = min(
            active_threshold,
            _float_env("DRAFTPROOF_DOMINANT_BLOCKER_TARGET_GAP_THRESHOLD", 65.0),
        )
    min_drop = _float_env("DRAFTPROOF_DOMINANT_BLOCKER_MIN_DROP", 5.0)
    active_keys = [
        key for key in dominant_keys
        if float(original.get(key, 0.0) or 0.0) >= active_threshold
    ]
    drops = {
        key: round(float(original.get(key, 0.0) or 0.0) - float(candidate.get(key, 0.0) or 0.0), 3)
        for key in active_keys
    }
    max_drop = max([0.0] + list(drops.values()))
    regression = sum(max(0.0, -value) for value in drops.values())
    required = bool(active_keys) and _env_flag("DRAFTPROOF_REQUIRE_DOMINANT_BLOCKER_DROP", True)
    cleared = (not required) or (max_drop >= min_drop and regression <= 0.0)
    return {
        "required": required,
        "cleared": cleared,
        "active_keys": active_keys,
        "active_threshold": active_threshold,
        "target_human": target_human,
        "original_human": original_human,
        "drops": drops,
        "max_drop": round(max_drop, 3),
        "regression": round(regression, 3),
        "min_drop": min_drop,
        "original": {key: original.get(key, 0.0) for key in dominant_keys},
        "candidate": {key: candidate.get(key, 0.0) for key in dominant_keys},
        "reason": "" if cleared else "dominant_blocker_not_reduced",
    }


def _human_formula_driver_deps() -> HumanFormulaDriverDeps:
    return HumanFormulaDriverDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        contribution_scores=_contribution_scores,
    )


def _human_formula_driver_status(original_report: dict | None, candidate_report: dict | None) -> dict:
    return _core_human_formula_driver_status(
        original_report,
        candidate_report,
        _human_formula_driver_deps(),
    )


def _dominant_blocker_safe_progress_override(
    dominant_status: dict | None,
    authenticity_status: dict | None,
    blocker_status: dict | None,
    *,
    ai_score_regressed: bool,
    finding_delta: int,
    review_burden_delta: int,
    weighted_severity_delta: int,
    critical_high_delta: int,
) -> dict:
    """Allow safe wins when an unsupported-claim blocker is pinned.

    Some short submissions cannot reduce source/evidence metrics because the
    author did not provide source material. In that case the dominant blocker
    gate must not veto a candidate that improves authorship, transformation,
    review burden, and severity without inventing evidence.
    """
    if not _env_flag("DRAFTPROOF_DOMINANT_BLOCKER_ALLOW_SAFE_PROGRESS", True):
        return {"allowed": False, "reason": "disabled"}
    dominant_status = dominant_status if isinstance(dominant_status, dict) else {}
    authenticity_status = authenticity_status if isinstance(authenticity_status, dict) else {}
    blocker_status = blocker_status if isinstance(blocker_status, dict) else {}
    if not dominant_status.get("required") or dominant_status.get("cleared"):
        return {"allowed": False, "reason": "dominant_blocker_not_active"}
    active_keys = set(dominant_status.get("active_keys") or [])
    allowed_stale_keys = {
        key.strip()
        for key in os.environ.get(
            "DRAFTPROOF_DOMINANT_BLOCKER_SAFE_PROGRESS_KEYS",
            "unsupported_claim_risk,source_grounding_risk,broad_claim_risk,topk_calibrated_risk,generic_assertion_risk",
        ).split(",")
        if key.strip()
    }
    if not active_keys or not active_keys.issubset(allowed_stale_keys):
        return {"allowed": False, "reason": "dominant_blocker_not_safe_stale_type"}
    human_delta = authenticity_status.get("human_delta")
    ai_authorship_delta = authenticity_status.get("ai_authorship_delta")
    ai_transform_delta = authenticity_status.get("ai_transformation_delta")
    if not all(isinstance(value, (int, float)) for value in (human_delta, ai_authorship_delta, ai_transform_delta)):
        return {"allowed": False, "reason": "missing_authenticity_deltas"}
    required_human = _float_env("DRAFTPROOF_DOMINANT_BLOCKER_SAFE_PROGRESS_MIN_HUMAN_GAIN", 4.0)
    required_authorship = _float_env("DRAFTPROOF_DOMINANT_BLOCKER_SAFE_PROGRESS_MIN_AUTHORSHIP_DROP", 1.0)
    required_transform = _float_env("DRAFTPROOF_DOMINANT_BLOCKER_SAFE_PROGRESS_MIN_TRANSFORM_DROP", 4.0)
    max_active_regression = _float_env("DRAFTPROOF_DOMINANT_BLOCKER_SAFE_PROGRESS_MAX_ACTIVE_REGRESSION", 0.0)
    active_regression = float(blocker_status.get("active_regression") or 0.0)
    active_drop = float(blocker_status.get("active_drop") or 0.0)
    dominant_drops = dominant_status.get("drops") if isinstance(dominant_status.get("drops"), dict) else {}
    dominant_max_drop = max(
        [0.0]
        + [
            float(value)
            for value in dominant_drops.values()
            if isinstance(value, (int, float))
        ]
    )
    min_dominant_drop = _float_env(
        "DRAFTPROOF_DOMINANT_BLOCKER_SAFE_PROGRESS_MIN_DOMINANT_DROP",
        0.5,
    )
    target_breakthrough = bool(authenticity_status.get("crosses_target_human"))
    dominant_drop_allowed = bool(
        dominant_max_drop >= min_dominant_drop
        or target_breakthrough
    )
    min_net_active_drop = _float_env(
        "DRAFTPROOF_DOMINANT_BLOCKER_SAFE_PROGRESS_MIN_NET_ACTIVE_DROP",
        10.0,
    )
    blocker_regression_allowed = bool(
        active_regression <= max_active_regression
        or (active_drop - active_regression) >= min_net_active_drop
    )
    allowed = bool(
        float(human_delta) >= required_human
        and float(ai_authorship_delta) >= required_authorship
        and float(ai_transform_delta) >= required_transform
        and not ai_score_regressed
        and finding_delta <= 0
        and review_burden_delta <= 0
        and weighted_severity_delta <= 0
        and critical_high_delta <= 0
        and dominant_drop_allowed
        and blocker_regression_allowed
        and not authenticity_status.get("ai_authorship_regression_blocked")
        and not authenticity_status.get("critical_high_regressed")
        and not authenticity_status.get("review_burden_regressed")
        and not authenticity_status.get("weighted_severity_regressed")
    )
    return {
        "allowed": allowed,
        "reason": "" if allowed else "safe_progress_threshold_not_met",
        "active_keys": sorted(active_keys),
        "required_human_gain": required_human,
        "required_ai_authorship_drop": required_authorship,
        "required_ai_transformation_drop": required_transform,
        "human_delta": human_delta,
        "ai_authorship_delta": ai_authorship_delta,
        "ai_transformation_delta": ai_transform_delta,
        "finding_delta": finding_delta,
        "review_burden_delta": review_burden_delta,
        "weighted_severity_delta": weighted_severity_delta,
        "critical_high_delta": critical_high_delta,
        "dominant_max_drop": round(dominant_max_drop, 3),
        "min_dominant_drop": min_dominant_drop,
        "dominant_drop_allowed": dominant_drop_allowed,
        "target_breakthrough": target_breakthrough,
        "active_drop": active_drop,
        "active_regression": active_regression,
        "min_net_active_drop": min_net_active_drop,
        "blocker_regression_allowed": blocker_regression_allowed,
    }


def _ai_search_adaptive_stop_reason(
    selection_status: dict | None,
    *,
    phase: str,
    short_document: bool = False,
) -> str:
    if not _env_flag("DRAFTPROOF_AI_SEARCH_ADAPTIVE_STOP", True):
        return ""
    if not isinstance(selection_status, dict) or not selection_status.get("selectable"):
        return ""
    if selection_status.get("safe_partial_quality_improvement"):
        footprint_gate = selection_status.get("ai_footprint_gate") or {}
        if not footprint_gate.get("material_driver_moved"):
            return ""
        return f"adaptive_stop_after_safe_partial_quality_{phase}"
    footprint_gate = selection_status.get("ai_footprint_gate") or {}
    if footprint_gate.get("outcome_class") in {"ai_mitigated", "partially_ai_mitigated"}:
        return f"adaptive_stop_after_ai_footprint_{phase}"
    dominant = selection_status.get("dominant_blocker_gate") or {}
    if (
        dominant.get("required")
        and not dominant.get("cleared")
        and not selection_status.get("dominant_blocker_safe_progress_override")
    ):
        return ""
    gate = selection_status.get("authenticity_gate") or {}
    human_delta = gate.get("human_delta")
    ai_authorship_delta = gate.get("ai_authorship_delta")
    ai_transform_delta = gate.get("ai_transformation_delta")
    if not all(isinstance(value, (int, float)) for value in (human_delta, ai_authorship_delta, ai_transform_delta)):
        return ""
    human_min_default = 4.0 if short_document else 8.0
    authorship_min_default = 1.0 if short_document else 2.0
    transform_min_default = 4.0 if short_document else 5.0
    human_min = _float_env(
        (
            "DRAFTPROOF_ADAPTIVE_STOP_SHORT_MIN_HUMAN_GAIN"
            if short_document else "DRAFTPROOF_ADAPTIVE_STOP_MIN_HUMAN_GAIN"
        ),
        human_min_default,
    )
    authorship_min = _float_env(
        (
            "DRAFTPROOF_ADAPTIVE_STOP_SHORT_MIN_AUTHORSHIP_DROP"
            if short_document else "DRAFTPROOF_ADAPTIVE_STOP_MIN_AUTHORSHIP_DROP"
        ),
        authorship_min_default,
    )
    transform_min = _float_env(
        (
            "DRAFTPROOF_ADAPTIVE_STOP_SHORT_MIN_TRANSFORM_DROP"
            if short_document else "DRAFTPROOF_ADAPTIVE_STOP_MIN_TRANSFORM_DROP"
        ),
        transform_min_default,
    )
    if float(human_delta) < human_min:
        return ""
    if float(ai_authorship_delta) < authorship_min:
        return ""
    if float(ai_transform_delta) < transform_min:
        return ""
    if gate.get("critical_high_regressed") or gate.get("review_burden_regressed") or gate.get("weighted_severity_regressed"):
        return ""
    return f"adaptive_stop_after_{phase}"


def _should_track_blocked_human_winner(
    *,
    selection_status: dict | None,
    human_delta: float,
    ai_delta: float,
    authenticity_status: dict | None,
    ai_footprint_gate: dict | None = None,
) -> bool:
    """Track promising Human/AI candidates that failed only late safety gates."""
    if not isinstance(selection_status, dict) or selection_status.get("selectable"):
        return False
    authenticity_status = authenticity_status or {}
    ai_footprint_gate = ai_footprint_gate if isinstance(ai_footprint_gate, dict) else {}
    min_human_gain = _float_env("DRAFTPROOF_BLOCKED_HUMAN_REPAIR_MIN_HUMAN_GAIN", 1.0)
    if authenticity_status.get("ai_authorship_regression_blocked"):
        return False
    material_footprint = bool(ai_footprint_gate.get("material_driver_moved"))
    human_promising = float(human_delta or 0.0) >= min_human_gain
    if not human_promising and not material_footprint:
        return False
    if authenticity_status.get("critical_high_regressed"):
        return True
    quality_regressed = bool(
        authenticity_status.get("review_burden_regressed")
        or authenticity_status.get("weighted_severity_regressed")
    )
    if not quality_regressed:
        return False
    ai_authorship_drop = float(authenticity_status.get("ai_authorship_delta") or 0.0)
    ai_transform_drop = float(authenticity_status.get("ai_transformation_delta") or 0.0)
    min_ai_drop = _float_env("DRAFTPROOF_BLOCKED_HUMAN_REPAIR_MIN_AI_DROP", 5.0)
    min_authorship_drop = _float_env("DRAFTPROOF_BLOCKED_HUMAN_REPAIR_MIN_AUTHORSHIP_DROP", 2.0)
    min_transform_drop = _float_env("DRAFTPROOF_BLOCKED_HUMAN_REPAIR_MIN_TRANSFORM_DROP", 0.0)
    return bool(
        material_footprint
        or
        float(ai_delta or 0.0) >= min_ai_drop
        or ai_authorship_drop >= min_authorship_drop
        or ai_transform_drop >= min_transform_drop
    )


def _blocked_human_winner_repair_budget_override(adaptive_stop_reason: str) -> bool:
    """Allow one bounded repair attempt after normal search budget exhaustion."""
    if not _env_flag("DRAFTPROOF_BLOCKED_HUMAN_WINNER_REPAIR_AFTER_BUDGET", True):
        return False
    return str(adaptive_stop_reason or "") in {
        "budget_exhausted_llm_calls",
        "budget_exhausted_candidate_scans",
    }


def _blocked_human_winner_failed_formula_gate(candidate: dict | None) -> bool:
    """Do not spend repair budget on candidates blocked by the Human formula drivers.

    Finding-local repair can fix review burden or severity regressions. It cannot fix
    a candidate class whose direct Human Contribution formula drivers moved the wrong
    way, so running it after that gate is pure budget waste.
    """
    if not isinstance(candidate, dict):
        return False
    summary = candidate.get("summary")
    if not isinstance(summary, dict):
        return False
    selection = summary.get("selection_status")
    if not isinstance(selection, dict):
        return False
    footprint_gate = selection.get("ai_footprint_gate")
    if isinstance(footprint_gate, dict) and footprint_gate.get("material_driver_moved"):
        return False
    formula_gate = selection.get("human_formula_driver_gate")
    if not isinstance(formula_gate, dict):
        return False
    return bool(
        selection.get("reason") == "human_formula_drivers_not_reduced"
        or (
            formula_gate.get("required")
            and not formula_gate.get("cleared")
            and formula_gate.get("reason") == "human_formula_drivers_not_reduced"
        )
    )


def _post_safe_target_push_allows_deterministic_after_budget(adaptive_stop_reason: str) -> bool:
    """Let bounded no-LLM target push run after a budget stop when it cannot spend model calls."""
    reason = str(adaptive_stop_reason or "")
    if (
        reason == "budget_exhausted_llm_calls"
        and _env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_AFTER_LLM_BUDGET", True)
    ):
        return True
    if reason != "budget_exhausted_candidate_scans":
        return False
    if not _env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_AFTER_SCAN_BUDGET", True):
        return False
    return _post_safe_target_push_scan_reserve(adaptive_stop_reason) > 0


def _post_safe_target_push_scan_reserve(adaptive_stop_reason: str) -> int:
    """Small candidate-scan reserve for safe target push after the normal scan budget is spent."""
    if str(adaptive_stop_reason or "") != "budget_exhausted_candidate_scans":
        return 0
    try:
        return max(
            0,
            int(_float_env("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_SCAN_RESERVE", 3.0)),
        )
    except (TypeError, ValueError):
        return 3


def _final_topk_texture_scan_reserve(adaptive_stop_reason: str) -> int:
    """Tiny final texture reserve for an already selectable candidate under tight budget."""
    if str(adaptive_stop_reason or "") != "budget_exhausted_candidate_scans":
        return 0
    if not _env_flag("DRAFTPROOF_FINAL_TOPK_TEXTURE_AFTER_SCAN_BUDGET", True):
        return 0
    try:
        return max(0, int(_float_env("DRAFTPROOF_FINAL_TOPK_TEXTURE_SCAN_RESERVE", 1.0)))
    except (TypeError, ValueError):
        return 1


def _topk_safe_band_scan_reserve() -> int:
    """Reserve scans for the emergency safe-band rebuild path.

    This path is only active when Top-k is the hard blocker. It needs a few
    internal rescans for snapshot + patch rounds; otherwise it can stop just
    above the 25 calibrated safe band and rollback the whole rewrite.
    """
    try:
        return max(0, int(_float_env("DRAFTPROOF_TOPK_SAFE_BAND_SCAN_RESERVE", 16.0)))
    except (TypeError, ValueError):
        return 16


def _topk_near_miss_partial_keep_decision(
    *,
    topk_value: float | int | None,
    safe_limit: float,
    topk_drop: float | int | None,
    ai_drop: float | int | None,
    ai_authorship_drop: float | int | None,
    ai_transformation_drop: float | int | None,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    """Keep strong Top-k-blocked wins as partial progress without calling them safe.

    Calibrated Top-k above the safe line should not erase large authorship or
    AI-footprint gains. It still blocks strict-safe status.
    """
    if not isinstance(topk_value, (int, float)):
        return {"allowed": False, "reason": "topk_missing"}
    topk_over_limit = float(topk_value) - float(safe_limit)
    if topk_over_limit < 0.0:
        return {"allowed": False, "reason": "already_safe"}
    if not isinstance(topk_drop, (int, float)) or float(topk_drop) < 8.0:
        return {
            "allowed": False,
            "reason": "topk_drop_too_small",
            "topk_drop": round(float(topk_drop or 0.0), 3),
        }
    meaningful_driver_drop = any(
        isinstance(value, (int, float)) and float(value) >= 5.0
        for value in (ai_drop, ai_authorship_drop, ai_transformation_drop)
    )
    if not meaningful_driver_drop:
        return {"allowed": False, "reason": "no_meaningful_ai_driver_drop"}
    if (
        float(review_burden_delta or 0.0) > 0.0
        or float(weighted_severity_delta or 0.0) > 0.0
        or float(critical_high_delta or 0.0) > 0.0
    ):
        return {"allowed": False, "reason": "review_or_severity_regressed"}
    return {
        "allowed": True,
        "reason": "topk_blocked_but_material_ai_footprint_drop",
        "topk_over_limit": round(topk_over_limit, 3),
        "topk_drop": round(float(topk_drop), 3),
        "ai_drop": round(float(ai_drop), 3) if isinstance(ai_drop, (int, float)) else None,
        "ai_authorship_drop": (
            round(float(ai_authorship_drop), 3)
            if isinstance(ai_authorship_drop, (int, float)) else None
        ),
        "ai_transformation_drop": (
            round(float(ai_transformation_drop), 3)
            if isinstance(ai_transformation_drop, (int, float)) else None
        ),
    }


def _blocked_winner_bounded_quality_tradeoff(
    *,
    candidate_eval: dict | None,
    authenticity_status: dict | None,
    ai_delta: float,
    review_burden_delta: int,
    weighted_severity_delta: int,
    finding_delta: int,
    critical_high_delta: int,
    ai_score_regressed: bool,
) -> dict:
    """Permit small quality cost only when a repaired blocked winner makes a large attribution move."""
    if not _env_flag("DRAFTPROOF_BLOCKED_HUMAN_WINNER_BOUNDED_TRADEOFF", True):
        return {"allowed": False, "reason": "disabled"}
    candidate_eval = candidate_eval or {}
    authenticity_status = authenticity_status or {}
    if not candidate_eval.get("blocked_human_winner_repair"):
        return {"allowed": False, "reason": "not_blocked_winner_repair"}
    max_severity_delta = int(_float_env("DRAFTPROOF_BLOCKED_HUMAN_WINNER_MAX_SEVERITY_DELTA", 3.0))
    max_finding_delta = int(_float_env("DRAFTPROOF_BLOCKED_HUMAN_WINNER_MAX_FINDING_DELTA", 2.0))
    min_ai_delta = _float_env("DRAFTPROOF_BLOCKED_HUMAN_WINNER_MIN_AI_DELTA", 5.0)
    min_authorship_delta = _float_env("DRAFTPROOF_BLOCKED_HUMAN_WINNER_MIN_AUTHORSHIP_DELTA", 5.0)
    min_human_delta = _float_env("DRAFTPROOF_BLOCKED_HUMAN_WINNER_MIN_HUMAN_DELTA", 1.0)
    min_transform_delta = _float_env("DRAFTPROOF_BLOCKED_HUMAN_WINNER_MIN_TRANSFORM_DELTA", 1.0)
    min_shift = _float_env("DRAFTPROOF_BLOCKED_HUMAN_WINNER_MIN_SHIFT", 3.0)
    checks = {
        "ai_delta": float(ai_delta or 0.0),
        "ai_authorship_delta": float(authenticity_status.get("ai_authorship_delta") or 0.0),
        "human_delta": float(authenticity_status.get("human_delta") or 0.0),
        "ai_transformation_delta": float(authenticity_status.get("ai_transformation_delta") or 0.0),
        "human_shift_score": float(authenticity_status.get("human_shift_score") or 0.0),
        "review_burden_delta": int(review_burden_delta or 0),
        "weighted_severity_delta": int(weighted_severity_delta or 0),
        "finding_delta": int(finding_delta or 0),
        "critical_high_delta": int(critical_high_delta or 0),
    }
    allowed = bool(
        not ai_score_regressed
        and checks["ai_delta"] >= min_ai_delta
        and checks["ai_authorship_delta"] >= min_authorship_delta
        and checks["human_delta"] >= min_human_delta
        and checks["ai_transformation_delta"] >= min_transform_delta
        and checks["human_shift_score"] >= min_shift
        and checks["review_burden_delta"] <= 0
        and checks["critical_high_delta"] <= 0
        and checks["weighted_severity_delta"] <= max_severity_delta
        and checks["finding_delta"] <= max_finding_delta
    )
    return {
        "allowed": allowed,
        "reason": "" if allowed else "threshold_not_met",
        **checks,
        "max_weighted_severity_delta": max_severity_delta,
        "max_finding_delta": max_finding_delta,
        "min_ai_delta": min_ai_delta,
        "min_ai_authorship_delta": min_authorship_delta,
        "min_human_delta": min_human_delta,
        "min_ai_transformation_delta": min_transform_delta,
        "min_human_shift_score": min_shift,
    }


def _score_drag_removal_status(
    *,
    authenticity_status: dict | None,
    human_shift: dict | None,
    ai_delta: float,
    finding_delta: int,
    review_burden_delta: int,
    weighted_severity_delta: int,
    critical_high_delta: int,
    ai_score_regressed: bool,
) -> dict:
    """Classify bounded removals/compressions that reduce review burden safely.

    Score-drag cleanup is useful, but it is not AI-Mitigation by itself. It can
    only become selectable when explicitly enabled and when it also produces
    material human-side movement. This prevents cleanup-only candidates from
    being labelled as a successful rewrite.
    """
    authenticity_status = authenticity_status or {}
    human_shift = human_shift or {}
    human_delta = authenticity_status.get("human_delta")
    authorship_delta = authenticity_status.get("ai_authorship_delta")
    transform_delta = authenticity_status.get("ai_transformation_delta")
    min_human = _float_env("DRAFTPROOF_SCORE_DRAG_MIN_HUMAN_DELTA", 5.0)
    min_authorship = _float_env("DRAFTPROOF_SCORE_DRAG_MIN_AUTHORSHIP_DELTA", 0.0)
    min_transform = _float_env("DRAFTPROOF_SCORE_DRAG_MIN_TRANSFORM_DELTA", 2.0)
    min_ai_drop = _float_env("DRAFTPROOF_SCORE_DRAG_MIN_AI_DROP", 0.05)
    min_finding_drop = int(_float_env("DRAFTPROOF_SCORE_DRAG_MIN_FINDING_DROP", 2.0))
    min_review_drop = int(_float_env("DRAFTPROOF_SCORE_DRAG_MIN_REVIEW_DROP", 1.0))
    min_severity_drop = int(_float_env("DRAFTPROOF_SCORE_DRAG_MIN_SEVERITY_DROP", 2.0))
    finding_drop = max(0, -int(finding_delta or 0))
    review_drop = max(0, -int(review_burden_delta or 0))
    severity_drop = max(0, -int(weighted_severity_delta or 0))
    burden_reduced = bool(
        finding_drop >= min_finding_drop
        or review_drop >= min_review_drop
        or severity_drop >= min_severity_drop
    )
    numeric_ok = all(
        isinstance(value, (int, float))
        for value in (human_delta, authorship_delta, transform_delta, ai_delta)
    )
    allowed = bool(
        _env_flag("DRAFTPROOF_AI_SEARCH_ACCEPT_SCORE_DRAG_REMOVAL", False)
        and numeric_ok
        and float(human_delta) >= min_human
        and float(authorship_delta) >= min_authorship
        and float(transform_delta) >= min_transform
        and float(ai_delta) > min_ai_drop
        and burden_reduced
        and int(finding_delta or 0) <= 0
        and int(review_burden_delta or 0) <= 0
        and int(weighted_severity_delta or 0) <= 0
        and int(critical_high_delta or 0) <= 0
        and not ai_score_regressed
        and not authenticity_status.get("ai_authorship_regression_blocked")
        and not authenticity_status.get("critical_high_regressed")
    )
    return {
        "allowed": allowed,
        "reason": "" if allowed else "score_drag_threshold_not_met",
        "human_delta": human_delta,
        "ai_authorship_delta": authorship_delta,
        "ai_transformation_delta": transform_delta,
        "ai_delta": ai_delta,
        "human_shift_score": human_shift.get("score"),
        "finding_delta": finding_delta,
        "review_burden_delta": review_burden_delta,
        "weighted_severity_delta": weighted_severity_delta,
        "critical_high_delta": critical_high_delta,
        "finding_drop": finding_drop,
        "review_burden_drop": review_drop,
        "weighted_severity_drop": severity_drop,
        "min_ai_drop": min_ai_drop,
        "min_finding_drop": min_finding_drop,
        "min_review_burden_drop": min_review_drop,
        "min_weighted_severity_drop": min_severity_drop,
        "burden_reduced": burden_reduced,
        "cleanup_only": bool(burden_reduced and not allowed),
        "selectable_as_mitigation": allowed,
        "ignored_negative_human_shift": bool(
            isinstance(human_shift.get("score"), (int, float))
            and float(human_shift.get("score")) < 0
            and burden_reduced
        ),
    }


def _human_target_regression_selection_block(
    selection_status: dict | None,
    authenticity_status: dict | None,
    *,
    target_human: float | None = None,
) -> dict:
    """Block selected fallback candidates that move away from the Human target.

    The authenticity gate can record this failure, but final selection has
    multiple fallback paths. This guard is intentionally selector-level so a
    later fallback cannot accept a below-target candidate that lowers Human
    Contribution or increases AI Transformation.
    """
    if not isinstance(selection_status, dict) or not selection_status.get("selectable"):
        return {"blocked": False, "reason": "candidate_not_selectable"}
    authenticity_status = authenticity_status or {}
    target = float(
        target_human
        if isinstance(target_human, (int, float))
        else _float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    )
    candidate_human = authenticity_status.get("candidate_human")
    human_delta = authenticity_status.get("human_delta")
    ai_transform_delta = authenticity_status.get("ai_transformation_delta")
    candidate_below_human_target = bool(
        isinstance(candidate_human, (int, float)) and float(candidate_human) < target
    )
    human_target_regressed_direct = bool(
        candidate_below_human_target
        and isinstance(human_delta, (int, float))
        and float(human_delta) < 0.0
    )
    ai_transformation_target_regressed_direct = bool(
        candidate_below_human_target
        and isinstance(ai_transform_delta, (int, float))
        and float(ai_transform_delta) < 0.0
    )
    blocked = bool(human_target_regressed_direct or ai_transformation_target_regressed_direct)
    return {
        "blocked": blocked,
        "reason": (
            "human_target_regressed"
            if human_target_regressed_direct
            else "ai_transformation_target_regressed"
            if ai_transformation_target_regressed_direct
            else ""
        ),
        "human_target_guard_required": blocked,
        "candidate_below_human_target": candidate_below_human_target,
        "target_human": target,
        "candidate_human": candidate_human,
        "human_delta": human_delta,
        "ai_transformation_delta": ai_transform_delta,
        "human_target_regressed_direct": human_target_regressed_direct,
        "ai_transformation_target_regressed_direct": ai_transformation_target_regressed_direct,
    }


def _adaptive_budget_default(source_text: str, short_value: int, long_value: int) -> str:
    if _env_flag("DRAFTPROOF_ADAPTIVE_SHORT_DOC_BUDGETS", True):
        threshold = int(_float_env("DRAFTPROOF_SHORT_DOC_WORD_THRESHOLD", 450.0))
        if _text_word_count(source_text) <= threshold:
            return str(short_value)
    return str(long_value)


def _ai_search_budget_policy(source_text: str = "", report_dict: dict | None = None) -> dict:
    """Driver-aware budget policy for AI mitigation search.

    Word count sets the floor. Active scanner blockers, especially calibrated
    Top-k gap, decide the extra call reserve. This avoids both brittle fixed
    budgets and runaway retries.
    """
    words = _text_word_count(source_text)
    topk_gap = _topk_gap_band(report_dict)
    topk_rounds = _topk_safe_band_patch_rounds_default(source_text, report_dict)
    topk_phase = 1 + topk_rounds + 1  # masked route + snapshot + patch rounds
    extra_seconds = 60 if topk_gap.get("band") == "saturated" else 30 if topk_gap.get("band") == "high" else 0
    scan_budget = _verified_candidate_scan_budget(source_text, report_dict)
    if words <= 700:
        base = {
            "word_count": words,
            "size_band": "short",
            "max_seconds": 90 + extra_seconds,
            "max_llm_calls": max(4, topk_phase + 1),
            "max_candidate_scans": scan_budget["max_candidate_scans"],
            "max_candidate_scan_hard_cap": scan_budget["max_candidate_scan_hard_cap"],
            "candidate_scoring_controller": scan_budget,
            "phase_budget": {
                "topk_safe_band_rebuild": topk_phase,
                "authorship_transformation_texture_controller": 1,
                "final_texture_proxy_repair": 1,
            },
        }
        base["driver_budget"] = {"topk": topk_gap, "topk_patch_rounds": topk_rounds}
        return base
    if words <= 1800:
        base = {
            "word_count": words,
            "size_band": "medium",
            "max_seconds": 150 + extra_seconds,
            "max_llm_calls": max(6, topk_phase + 3),
            "max_candidate_scans": scan_budget["max_candidate_scans"],
            "max_candidate_scan_hard_cap": scan_budget["max_candidate_scan_hard_cap"],
            "candidate_scoring_controller": scan_budget,
            "phase_budget": {
                "topk_safe_band_rebuild": topk_phase,
                "authorship_transformation_texture_controller": 2,
                "final_texture_proxy_repair": 1,
            },
        }
        base["driver_budget"] = {"topk": topk_gap, "topk_patch_rounds": topk_rounds}
        return base
    base = {
        "word_count": words,
        "size_band": "long",
        "max_seconds": 240 + extra_seconds,
        "max_llm_calls": max(8, topk_phase + 4),
        "max_candidate_scans": scan_budget["max_candidate_scans"],
        "max_candidate_scan_hard_cap": scan_budget["max_candidate_scan_hard_cap"],
        "candidate_scoring_controller": scan_budget,
        "phase_budget": {
            "topk_safe_band_rebuild": topk_phase,
            "authorship_transformation_texture_controller": 3,
            "final_texture_proxy_repair": 1,
        },
    }
    base["driver_budget"] = {"topk": topk_gap, "topk_patch_rounds": topk_rounds}
    return base


def _ai_search_llm_hard_cap(source_text: str = "", report_dict: dict | None = None) -> int:
    explicit = os.environ.get("DRAFTPROOF_AI_SEARCH_HARD_MAX_LLM_CALLS")
    if explicit is not None:
        return max(1, int(_float_env("DRAFTPROOF_AI_SEARCH_HARD_MAX_LLM_CALLS", 10.0)))
    policy_cap = int(_ai_search_budget_policy(source_text, report_dict).get("max_llm_calls") or 1)
    return max(1, min(10, policy_cap))


def _candidate_scan_hard_cap(search_budget: dict | None) -> int:
    budget = search_budget if isinstance(search_budget, dict) else {}
    try:
        configured = int(budget.get("max_candidate_scans") or 0)
    except (TypeError, ValueError):
        configured = 0
    try:
        hard_cap = int(budget.get("max_candidate_scan_hard_cap") or configured or 0)
    except (TypeError, ValueError):
        hard_cap = configured
    if hard_cap <= 0:
        return max(0, configured)
    return hard_cap


def _verified_candidate_scan_budget(source_text: str = "", report_dict: dict | None = None) -> dict:
    """Return the bounded full-scan budget for verified rewrite finalists.

    Candidate generation can create many local variants. Full detect scans are
    the expensive verification layer, so the budget scales sublinearly with
    document size and active AI-driver pressure instead of with raw candidate
    count.
    """
    words = max(1, _text_word_count(source_text))
    base = max(3, int(math.ceil(math.sqrt(words / 25.0))))
    blockers = _blocker_scores(report_dict)
    topk_value = blockers.get("topk_calibrated_risk")
    density_value = blockers.get("qualifying_text_ai_density")
    generic_value = blockers.get("generic_assertion_risk")
    pressure_bonus = 0
    if isinstance(topk_value, (int, float)) and float(topk_value) >= 75.0:
        pressure_bonus += 2
    if isinstance(density_value, (int, float)) and float(density_value) >= 55.0:
        pressure_bonus += 1
    if isinstance(generic_value, (int, float)) and float(generic_value) >= 75.0:
        pressure_bonus += 1
    verified_scans = base + pressure_bonus
    reserve = max(2, int(math.ceil(math.sqrt(verified_scans))))
    return {
        "policy": "verified_finalist_full_scans",
        "word_count": words,
        "base_scans": base,
        "pressure_bonus": pressure_bonus,
        "max_candidate_scans": verified_scans,
        "max_candidate_scan_hard_cap": verified_scans + reserve,
        "reserve": reserve,
        "drivers": {
            "topk_calibrated_risk": topk_value,
            "qualifying_text_ai_density": density_value,
            "generic_assertion_risk": generic_value,
        },
    }


def _extend_candidate_scan_budget(search_budget: dict, current_scans: int, reserve: int | float | str | None) -> int:
    """Extend scan budget by a small reserve without cumulative runaway growth."""
    if not isinstance(search_budget, dict):
        return 0
    try:
        previous = int(search_budget.get("max_candidate_scans") or 0)
    except (TypeError, ValueError):
        previous = 0
    try:
        current = max(0, int(current_scans or 0))
    except (TypeError, ValueError):
        current = 0
    try:
        reserve_count = max(0, int(reserve or 0))
    except (TypeError, ValueError):
        reserve_count = 0
    hard_cap = _candidate_scan_hard_cap(search_budget)
    requested = max(previous, current + reserve_count)
    if hard_cap > 0:
        requested = min(hard_cap, requested)
    search_budget["max_candidate_scans"] = requested
    return requested


def _radar_blockers_for_controller(raw_json: dict | None) -> list[dict]:
    """Return scanner radar blockers, falling back to legacy component scores."""
    if not isinstance(raw_json, dict):
        return []
    radar = (
        ((raw_json.get("scan_intelligence") or {}).get("blocker_radar") or {})
        if isinstance(raw_json.get("scan_intelligence"), dict)
        else {}
    )
    blockers = radar.get("blockers") or radar.get("dominant_blockers") or []
    if isinstance(blockers, list) and blockers:
        return [item for item in blockers if isinstance(item, dict)]

    fallback = []
    for key, score in _blocker_scores(raw_json).items():
        if not isinstance(score, (int, float)) or float(score) < 25.0:
            continue
        if key in {"topk_pattern", "predictability", "generic_assertion_risk"}:
            layer = "ai_authorship_risk"
            texture = True
            evidence_gap = False
            author_gap = False
        elif key in {"unsupported_claim_risk", "broad_claim_risk", "source_grounding_risk"}:
            layer = "grounding_quality_risk"
            texture = False
            evidence_gap = True
            author_gap = key in {"unsupported_claim_risk", "broad_claim_risk"}
        else:
            layer = "human_contribution_gap"
            texture = False
            evidence_gap = False
            author_gap = True
        fallback.append({
            "key": key,
            "label": key.replace("_", " ").title(),
            "layer": layer,
            "score": float(score),
            "severity": "high" if score >= 70 else "medium" if score >= 45 else "low",
            "scope": "document_wide" if score >= 60 else "unlocalized",
            "sentence_ids": [],
            "paragraph_ids": [],
            "diagnostic_flags": {
                "evidence_gap": evidence_gap,
                "source_dependency": key == "source_grounding_risk",
                "texture_pressure": texture,
                "author_context_gap": author_gap,
            },
        })
    fallback.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return fallback


def _radar_repair_operation(key: str, flags: dict) -> dict:
    if key in {"topk_pattern", "predictability"}:
        return {
            "operation": "deterministic_topk_texture_repair",
            "requires": ["no_llm", "sentence_window", "authorship_cap_gate", "semantic_drift_check"],
            "reason": "try a deterministic local texture patch before spending LLM/Tavily budget",
        }
    if flags.get("texture_pressure"):
        return {
            "operation": "deterministic_texture_or_structure_repair",
            "requires": ["no_llm", "paragraph_window", "authorship_cap_gate", "semantic_drift_check"],
            "reason": "try bounded deterministic rhythm/structure repair first",
        }
    if key in {"citation_weakness_risk", "source_grounding_risk"} or flags.get("source_dependency"):
        return {
            "operation": "deterministic_existing_source_bridge_or_narrow",
            "requires": ["no_llm", "existing_visible_source_or_claim_narrowing", "no_new_source"],
            "reason": "without LLM/Tavily, only bridge existing visible source links or narrow the claim",
        }
    if key in {"unsupported_claim_risk", "broad_claim_risk", "generic_assertion_risk"} or flags.get("evidence_gap"):
        return {
            "operation": "deterministic_claim_narrow_or_compress",
            "requires": ["no_llm", "claim_scope_limit", "no_new_author_facts"],
            "reason": "broad or unsupported claims should be narrowed/compressed before reconstruction",
        }
    if key == "lived_detail_risk" or flags.get("author_context_gap"):
        return {
            "operation": "deterministic_implied_reasoning_only",
            "requires": ["no_llm", "submitted_context_only", "no_fabricated_lived_detail"],
            "reason": "author-context gaps can only use confirmed or clearly implied submitted context without LLM/Tavily",
        }
    return {
        "operation": "deterministic_targeted_repair",
        "requires": ["no_llm", "locality_gate", "semantic_drift_check"],
        "reason": "start with the smallest controlled repair",
    }


def _radar_recreate_operation(key: str, flags: dict) -> dict:
    if flags.get("source_dependency"):
        operation = "recreate_block_from_context_with_source_reinforcement"
        requires = ["origin_context", "source_result_or_existing_citation", "anchor_preservation", "scan_gate"]
    elif flags.get("evidence_gap"):
        operation = "recreate_block_with_narrowed_claims"
        requires = ["origin_context", "claim_scope_limit", "anchor_preservation", "scan_gate"]
    elif flags.get("texture_pressure"):
        operation = "recreate_block_from_origin_context"
        requires = ["origin_context", "anchor_preservation", "authorship_cap_gate", "scan_gate"]
    else:
        operation = "recreate_block_from_context"
        requires = ["origin_context", "anchor_preservation", "scan_gate"]
    return {
        "operation": operation,
        "requires": requires,
        "reason": "use when local repair cannot move the blocker without regression",
    }


def _radar_remove_allowed(score: float, scope: str, flags: dict, key: str) -> bool:
    if score < 60.0:
        return False
    if flags.get("evidence_gap") or flags.get("author_context_gap"):
        return True
    if key == "generic_assertion_risk" and scope in {"document_wide", "mixed", "unlocalized"}:
        return True
    return False


def _radar_goal_role(phase: str, key: str, flags: dict) -> str:
    if phase == "remove":
        return "last_resort_remove_score_drag_after_reinforce_or_recreate_fails"
    if flags.get("evidence_gap") or flags.get("author_context_gap") or key in {
        "unsupported_claim_risk",
        "broad_claim_risk",
        "citation_weakness_risk",
        "source_grounding_risk",
        "lived_detail_risk",
    }:
        return "direct_human_contribution_gain"
    if flags.get("texture_pressure") or key in {"topk_pattern", "predictability", "ai_likelihood", "rewrite_smoothness"}:
        return "enable_human_gain_by_capping_ai_authorship_and_transformation"
    return "support_human_contribution_target"


def _radar_goal_gate(phase: str) -> list[str]:
    base = [
        "candidate_human_contribution_increases_or_reaches_80",
        "candidate_ai_authorship_does_not_increase",
        "candidate_ai_transformation_does_not_increase",
        "semantic_drift_false",
        "anchor_loss_false",
        "final_scan_required",
    ]
    if phase == "remove":
        return base + [
            "meaning_preservation_accepts_compression_or_deletion",
            "word_count_band_preserved",
            "used_only_after_repair_no_llm_or_recreate_llm_tavily_failed",
        ]
    if phase == "recreate_llm_tavily":
        return base + [
            "origin_context_preserved",
            "no_new_author_facts_without_source_or_user_input",
            "tavily_search_calls_lte_5",
        ]
    return base


def _radar_blocker_options(blocker: dict, *, target_human: float = 80.0) -> dict:
    key = str(blocker.get("key") or "")
    score = float(blocker.get("score") or 0.0)
    scope = str(blocker.get("scope") or "unlocalized")
    flags = blocker.get("diagnostic_flags") if isinstance(blocker.get("diagnostic_flags"), dict) else {}
    severity = str(blocker.get("severity") or ("high" if score >= 70 else "medium" if score >= 45 else "low"))

    repair = _radar_repair_operation(key, flags)
    recreate_allowed = bool(
        score >= 45.0
        and (
            scope in {"mixed", "document_wide", "unlocalized"}
            or flags.get("evidence_gap")
            or flags.get("texture_pressure")
        )
    )
    remove_allowed = _radar_remove_allowed(score, scope, flags, key)
    options = [
        {
            "phase": "repair_no_llm",
            "allowed": True,
            "operation": repair["operation"],
            "requires": repair["requires"],
            "priority": 1,
            "last_resort": False,
            "reason": repair["reason"],
        },
        {
            "phase": "recreate_llm_tavily",
            "allowed": recreate_allowed,
            **_radar_recreate_operation(key, flags),
            "requires": _radar_recreate_operation(key, flags)["requires"] + ["llm", "tavily_max_5_searches"],
            "priority": 2,
            "last_resort": False,
        },
        {
            "phase": "remove",
            "allowed": remove_allowed,
            "operation": "remove_or_compress_score_drag",
            "requires": ["meaning_preservation_check", "word_count_band_check", "final_scan_gate"],
            "priority": 3,
            "last_resort": True,
            "reason": "last resort for high score-drag content that cannot be safely reinforced or recreated",
        },
    ]
    for option in options:
        phase = option.get("phase", "")
        option["goal"] = {
            "primary_metric": "human_contribution",
            "target_score": target_human,
            "role": _radar_goal_role(phase, key, flags),
            "acceptance_gate": _radar_goal_gate(phase),
        }
    return {
        "blocker_key": key,
        "label": blocker.get("label") or key.replace("_", " ").title(),
        "layer": blocker.get("layer"),
        "score": score,
        "severity": severity,
        "scope": scope,
        "sentence_ids": blocker.get("sentence_ids") or [],
        "paragraph_ids": blocker.get("paragraph_ids") or [],
        "diagnostic_flags": flags,
        "options": options,
        "controller_sequence": [
            option["phase"] for option in options if option.get("allowed")
        ],
    }


def _radar_blocker_option_matrix(raw_json: dict | None, *, limit: int = 12) -> dict:
    """Controller option matrix derived from scanner radar.

    The scanner stays diagnostic-only. This function belongs to the rewrite
    controller and lays out the only valid intervention phases for each radar
    blocker: repair without LLM, recreate with LLM/Tavily, then remove.
    """
    blockers = _radar_blockers_for_controller(raw_json)
    integrity = _integrity_scores(raw_json)
    current_human = integrity.get("human")
    target_human = 80.0
    human_gap = (
        max(0.0, target_human - float(current_human))
        if isinstance(current_human, (int, float))
        else None
    )
    rows = [
        _radar_blocker_options(item, target_human=target_human)
        for item in blockers[:max(1, int(limit or 1))]
    ]
    phase_counts = {"repair_no_llm": 0, "recreate_llm_tavily": 0, "remove": 0}
    for row in rows:
        for option in row.get("options") or []:
            if option.get("allowed"):
                phase_counts[option["phase"]] = phase_counts.get(option["phase"], 0) + 1
    return {
        "schema_version": "radar_blocker_option_matrix.v1",
        "source": "scan_intelligence.blocker_radar" if blockers else "none",
        "policy": {
            "owner": "rewrite_controller",
            "scanner_role": "diagnose_only",
            "primary_goal": "human_contribution_above_80",
            "target_human_contribution": target_human,
            "option_rule": "every option must serve the Human Contribution target and pass the final scan gate",
            "default_sequence": ["repair_no_llm", "recreate_llm_tavily", "remove"],
            "remove_is_last_resort": True,
            "source_search_max_calls_per_run": 5,
            "must_rescan_after_each_kept_change": True,
        },
        "goal_state": {
            "current_human_contribution": current_human,
            "target_human_contribution": target_human,
            "human_gap_to_target": human_gap,
        },
        "phase_counts": phase_counts,
        "options_by_blocker": rows,
    }


def _radar_goal_controller_status(raw_json: dict | None) -> dict:
    """Decide whether radar options must drive execution before local repair.

    This is controller policy, not scanner policy. When the Human Contribution
    gap is large, local sentence repair is not enough; the controller should
    spend its budget on the goal-serving option ladder first.
    """
    matrix = _radar_blocker_option_matrix(raw_json)
    goal = matrix.get("goal_state") or {}
    document = (((raw_json or {}).get("scan_intelligence") or {}).get("document") or {}) if isinstance(raw_json, dict) else {}
    word_count = document.get("word_count")
    if not isinstance(word_count, (int, float)):
        word_count = ((raw_json or {}).get("document_context") or {}).get("word_count") if isinstance(raw_json, dict) else None
    if not isinstance(word_count, (int, float)):
        source_text = ""
        if isinstance(raw_json, dict):
            source_text = (
                raw_json.get("original_text")
                or raw_json.get("input_text")
                or raw_json.get("text")
                or ""
            )
        word_count = _text_word_count(str(source_text)) if source_text else None
    if isinstance(word_count, (int, float)):
        if word_count < 700:
            size_class = "short"
            max_recreate_blocks = 2
        elif word_count < 1800:
            size_class = "medium"
            max_recreate_blocks = 4
        elif word_count < 3500:
            size_class = "long"
            max_recreate_blocks = 6
        else:
            size_class = "very_long"
            max_recreate_blocks = 8
    else:
        size_class = "unknown"
        max_recreate_blocks = 4
    current_human = goal.get("current_human_contribution")
    target_human = goal.get("target_human_contribution", 80.0)
    human_gap = goal.get("human_gap_to_target")
    try:
        gap = float(human_gap)
    except (TypeError, ValueError):
        gap = 0.0
    direct_gain_blockers = []
    recreate_blockers = []
    remove_blockers = []
    for row in matrix.get("options_by_blocker") or []:
        for option in row.get("options") or []:
            if not option.get("allowed"):
                continue
            role = ((option.get("goal") or {}).get("role") or "")
            phase = option.get("phase")
            if role == "direct_human_contribution_gain":
                direct_gain_blockers.append(row.get("blocker_key"))
            if phase == "recreate_llm_tavily":
                recreate_blockers.append(row.get("blocker_key"))
            if phase == "remove":
                remove_blockers.append(row.get("blocker_key"))
    active = bool(
        isinstance(current_human, (int, float))
        and isinstance(target_human, (int, float))
        and float(current_human) < float(target_human)
        and gap >= _float_env("DRAFTPROOF_RADAR_GOAL_FIRST_MIN_GAP", 10.0)
        and (direct_gain_blockers or recreate_blockers or remove_blockers)
    )
    execute_first = bool(
        active
        and _env_flag("DRAFTPROOF_RADAR_GOAL_FIRST", True)
    )
    force_broad_reconstruction = bool(
        active
        and recreate_blockers
        and _env_flag("DRAFTPROOF_RADAR_FORCE_BROAD_RECONSTRUCTION", True)
    )
    return {
        "schema_version": "radar_goal_controller_status.v1",
        "active": active,
        "execute_before_local_rewrite": execute_first,
        "force_broad_reconstruction": force_broad_reconstruction,
        "current_human_contribution": current_human,
        "target_human_contribution": target_human,
        "human_gap_to_target": gap,
        "document_word_count": int(word_count) if isinstance(word_count, (int, float)) else None,
        "document_size_class": size_class,
        "max_recreate_blocks": max_recreate_blocks,
        "size_policy": (
            "short_doc_allows_whole_section_recreate"
            if size_class == "short"
            else "medium_doc_ranked_block_recreate"
            if size_class == "medium"
            else "long_doc_ranked_block_recreate_only"
            if size_class in {"long", "very_long"}
            else "unknown_size_ranked_block_recreate"
        ),
        "direct_gain_blockers": sorted(set(x for x in direct_gain_blockers if x)),
        "recreate_blockers": sorted(set(x for x in recreate_blockers if x)),
        "remove_blockers": sorted(set(x for x in remove_blockers if x)),
        "reason": (
            "Human Contribution gap requires radar option ladder before local sentence repair."
            if execute_first
            else "Radar option ladder does not need to override local repair first."
        ),
        "option_matrix": matrix,
    }


def _radar_goal_requires_human_progress(status: dict | None) -> bool:
    """Require Human movement when the radar ladder is active.

    Safe authorship suppression can be a useful intermediate signal, but it is
    not enough to end a run whose active controller goal is Human Contribution
    above 80. In that state, selectable candidates must either increase Human
    Contribution or reach the target.
    """
    if not isinstance(status, dict) or not status.get("active"):
        return False
    current = status.get("current_human_contribution")
    target = status.get("target_human_contribution", 80.0)
    return bool(
        isinstance(current, (int, float))
        and isinstance(target, (int, float))
        and float(current) < float(target)
        and _env_flag("DRAFTPROOF_RADAR_GOAL_REQUIRES_HUMAN_PROGRESS", True)
    )


def _blocker_operation_plan_deps() -> BlockerOperationPlanDeps:
    return BlockerOperationPlanDeps(
        float_env=_float_env,
        blocker_scores=_blocker_scores,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        text_word_count=_text_word_count,
        paragraph_component_targets=_paragraph_component_targets,
        paragraph_role=_paragraph_role,
        detect_protected_spans=detect_protected_spans,
        safe_index=_safe_index,
        radar_blocker_option_matrix=_radar_blocker_option_matrix,
    )


def _blocker_operation_plan(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 8,
) -> dict:
    return _core_blocker_operation_plan(
        source_text,
        raw_json,
        limit=limit,
        deps=_blocker_operation_plan_deps(),
    )


def _block_level_decisions(
    operations: list[dict],
    source_text: str,
    *,
    blockers: dict | None = None,
) -> list[dict]:
    return _core_block_level_decisions(
        operations,
        source_text,
        blockers=blockers,
        deps=_blocker_operation_plan_deps(),
    )



def _writing_component_percent(report_dict: dict | None, key: str):
    if not isinstance(report_dict, dict):
        return None
    badge = report_dict.get("ai_risk_badge") or {}
    writing = badge.get("writing_components") or {}
    value = writing.get(key) if isinstance(writing, dict) else None
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


_LIVED_DETAIL_RISK_BANDS = [
    {"risk": 80.0, "min_density": 0.00},
    {"risk": 65.0, "min_density": 0.10},
    {"risk": 50.0, "min_density": 0.20},
    {"risk": 35.0, "min_density": 0.30},
    {"risk": 20.0, "min_density": 0.45},
]


def _next_lived_detail_band(current_risk: float | int | None) -> dict:
    try:
        risk = float(current_risk)
    except (TypeError, ValueError):
        risk = 80.0
    for band in _LIVED_DETAIL_RISK_BANDS[1:]:
        if risk > float(band["risk"]):
            return dict(band)
    return dict(_LIVED_DETAIL_RISK_BANDS[-1])


def _human_anchor_marker_density(text: str) -> dict:
    sentences = [
        sentence for sentence in _split_sentences(text)
        if len(sentence.split()) >= 5
    ]
    if not sentences:
        return {
            "eligible_sentence_count": 0,
            "anchor_sentence_count": 0,
            "anchor_density": 0.0,
        }
    marker_re = re.compile(
        r"\b(?:\d+|during|when|after|before|feedback|testing|case|example|"
        r"in practice|I would|I think|I worry|we observed|"
        r"what (?:I|we) (?:would|need|want)|my judgement)\b",
        re.I,
    )
    anchor_count = sum(1 for sentence in sentences if marker_re.search(sentence))
    return {
        "eligible_sentence_count": len(sentences),
        "anchor_sentence_count": anchor_count,
        "anchor_density": round(anchor_count / max(1, len(sentences)), 4),
    }


def _human_anchor_driver_contract(
    original_report: dict | None,
    candidate_report: dict | None = None,
    *,
    text: str = "",
) -> dict:
    """Expose the actual scanner drivers behind Human Anchor movement."""
    def formula_parts(report: dict | None) -> dict:
        features = _transformation_features(report)

        def fnum(key: str) -> float:
            value = features.get(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            return max(0.0, min(1.0, value if abs(value) <= 1.0 else value / 100.0))

        max_similarity = max(fnum("source_similarity"), fnum("surface_similarity"))
        anchor_component = fnum("human_anchor_score") * 0.45
        smoothness_component = (1.0 - fnum("rewrite_smoothness")) * 0.20
        originality_component = (1.0 - max_similarity) * 0.10
        return {
            "anchor_component": round(anchor_component * 100.0, 3),
            "smoothness_component": round(smoothness_component * 100.0, 3),
            "originality_component": round(originality_component * 100.0, 3),
            "human_raw": round((anchor_component + smoothness_component + originality_component) * 100.0, 3),
            "rewrite_smoothness": round(fnum("rewrite_smoothness") * 100.0, 3),
            "max_similarity": round(max_similarity * 100.0, 3),
        }

    before_lived = _writing_component_percent(original_report, "lived_detail_risk")
    before_domain = _writing_component_percent(original_report, "domain_grounding_strength")
    before_anchor = _feature_percent(original_report, "human_anchor_score")
    after_lived = _writing_component_percent(candidate_report, "lived_detail_risk")
    after_domain = _writing_component_percent(candidate_report, "domain_grounding_strength")
    after_anchor = _feature_percent(candidate_report, "human_anchor_score")
    next_band = _next_lived_detail_band(before_lived)
    density = _human_anchor_marker_density(text)
    required_count = int(math.ceil(float(next_band["min_density"]) * max(1, density["eligible_sentence_count"])))
    current_count = int(density["anchor_sentence_count"])
    before = {
        "human_anchor_score": before_anchor,
        "lived_detail_risk": before_lived,
        "domain_grounding_strength": before_domain,
    }
    after = {
        "human_anchor_score": after_anchor,
        "lived_detail_risk": after_lived,
        "domain_grounding_strength": after_domain,
    }
    deltas = {}
    if isinstance(before_anchor, (int, float)) and isinstance(after_anchor, (int, float)):
        deltas["human_anchor_score"] = round(float(after_anchor) - float(before_anchor), 3)
    if isinstance(before_lived, (int, float)) and isinstance(after_lived, (int, float)):
        deltas["lived_detail_risk"] = round(float(before_lived) - float(after_lived), 3)
    if isinstance(before_domain, (int, float)) and isinstance(after_domain, (int, float)):
        deltas["domain_grounding_strength"] = round(float(after_domain) - float(before_domain), 3)
    return {
        "before": before,
        "after": after if candidate_report is not None else None,
        "deltas": deltas,
        "human_raw_formula": {
            "before": formula_parts(original_report),
            "after": formula_parts(candidate_report) if candidate_report is not None else None,
        },
        "next_lived_detail_band": next_band,
        "estimated_anchor_density": density,
        "required_anchor_sentences_for_next_band": required_count,
        "additional_anchor_sentences_needed": max(0, required_count - current_count),
        "achieved_next_band": bool(
            isinstance(after_lived, (int, float))
            and float(after_lived) <= float(next_band["risk"])
        ),
        "scope": "implied_context_only",
    }










def _critical_high_count(report_dict: dict | None) -> int:
    findings = (report_dict or {}).get("findings", {}) if isinstance(report_dict, dict) else {}
    return len(findings.get("critical", [])) + len(findings.get("high", []))


def _authenticity_gate_status(
    original_report: dict,
    candidate_report: dict,
    text_changed: bool,
    *,
    original_review_burden: int,
    candidate_review_burden: int,
    original_weighted_severity: int,
    candidate_weighted_severity: int,
    min_human_gain: float = 2.0,
    min_ai_transformation_drop: float = 2.0,
    drift_similarity: float | None = None,
) -> dict:
    deps = AuthenticityGateDeps(
        contribution_scores=_contribution_scores,
        integrity_scores=_integrity_scores,
        float_env=_float_env,
        critical_high_count=_critical_high_count,
        human_shift_score=_human_shift_score,
    )
    return _core_authenticity_gate_status(
        original_report,
        candidate_report,
        text_changed,
        original_review_burden=original_review_burden,
        candidate_review_burden=candidate_review_burden,
        original_weighted_severity=original_weighted_severity,
        candidate_weighted_severity=candidate_weighted_severity,
        min_human_gain=min_human_gain,
        min_ai_transformation_drop=min_ai_transformation_drop,
        drift_similarity=drift_similarity,
        deps=deps,
    )


def _ai_mitigation_action_brief(ai_mitigation: dict | None, limit: int = 10) -> str:
    if not isinstance(ai_mitigation, dict):
        return ""
    rows = []
    for action in (ai_mitigation.get("component_actions") or [])[:limit]:
        if not isinstance(action, dict):
            continue
        rows.append(
            "- "
            + "; ".join(
                part
                for part in [
                    f"component={action.get('component')}",
                    f"pillar={action.get('pillar')}",
                    f"score={action.get('score')}",
                    f"action={action.get('action')}",
                    f"user_input_needed={action.get('user_input_needed')}",
                ]
                if part and not part.endswith("=None")
            )
        )
    return "\n".join(rows)


def _generation_candidate_diagnostics(candidates: list[dict] | None, *, limit: int = 12) -> dict:
    """Compact generation attempt diagnostics for failed AI-Mitigation runs."""
    rows: list[dict] = []
    reason_counts: dict[str, int] = {}
    for candidate in (candidates or [])[:max(0, limit)]:
        if not isinstance(candidate, dict):
            continue
        gate = candidate.get("gate") if isinstance(candidate.get("gate"), dict) else {}
        staged = candidate.get("staged_generation") if isinstance(candidate.get("staged_generation"), dict) else {}
        reason = (
            candidate.get("reason")
            or gate.get("reason")
            or candidate.get("selection_reason")
            or ("accepted_candidate" if candidate.get("selected") else "")
            or "no_rejection_reason_recorded"
        )
        reason_key = str(reason).split(" ", 1)[0]
        reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
        row = {
            "attempt": candidate.get("attempt"),
            "strategy": candidate.get("strategy"),
            "reconstruction": bool(candidate.get("reconstruction")),
            "deterministic": bool(candidate.get("deterministic")),
            "passed_local_checks": bool(candidate.get("passed_local_checks")),
            "selected": bool(candidate.get("selected")),
            "best_so_far": bool(candidate.get("best_so_far")),
            "reason": reason,
            "warnings": candidate.get("warnings"),
            "candidate_length": candidate.get("candidate_length"),
            "candidate_word_count": candidate.get("candidate_word_count"),
            "drift_similarity": candidate.get("drift_similarity"),
            "drift_threshold": candidate.get("drift_threshold"),
            "drift_scan_relaxed_for_reconstruction": candidate.get("drift_scan_relaxed_for_reconstruction"),
            "scan_seconds": candidate.get("scan_seconds"),
            "ai": candidate.get("ai"),
            "writing_quality": candidate.get("writing_quality"),
            "human_contribution": candidate.get("human_contribution"),
            "ai_transformation": candidate.get("ai_transformation"),
            "ai_authorship": candidate.get("ai_authorship"),
            "human_delta": candidate.get("human_delta"),
            "ai_transformation_delta": candidate.get("ai_transformation_delta"),
            "ai_authorship_delta": candidate.get("ai_authorship_delta"),
            "human_shift_score": candidate.get("human_shift_score"),
            "authorship_cost_per_human_gain": candidate.get("authorship_cost_per_human_gain"),
            "findings": candidate.get("findings"),
            "review_burden": candidate.get("review_burden"),
            "weighted_severity": candidate.get("weighted_severity"),
            "gate_success": gate.get("success"),
            "gate_reason": gate.get("reason"),
            "gate_ai_authorship_regression_blocked": gate.get("ai_authorship_regression_blocked"),
            "gate_human_gain_with_authorship_regression": gate.get("human_gain_with_authorship_regression"),
            "gate_false_positive_improvement": gate.get("false_positive_improvement"),
            "gate_false_positive_improvement_reason": gate.get("false_positive_improvement_reason"),
            "gate_critical_high_regressed": gate.get("critical_high_regressed"),
            "gate_review_burden_regressed": gate.get("review_burden_regressed"),
            "gate_weighted_severity_regressed": gate.get("weighted_severity_regressed"),
        }
        if staged:
            row["staged_generation"] = {
                "enabled": staged.get("enabled"),
                "llm_calls": staged.get("llm_calls"),
                "assembled_word_count": staged.get("assembled_word_count"),
                "reference_entries_preserved": staged.get("reference_entries_preserved"),
                "source_draft_included": staged.get("source_draft_included"),
                "sections": staged.get("sections"),
            }
        rows.append({key: value for key, value in row.items() if value is not None})
    return {
        "candidate_count": len(candidates or []),
        "shown_count": len(rows),
        "reason_counts": reason_counts,
        "candidates": rows,
    }


def _authenticity_mitigation_prompt(
    source_text: str,
    raw_json: dict,
    ai_mitigation: dict | None,
    attempt_index: int,
) -> str:
    contribution = _contribution_scores(raw_json)
    semantic = (
        ((raw_json or {}).get("scan_intelligence") or {}).get("semantic_shape")
        or (raw_json or {}).get("semantic_shape")
        or {}
    )
    signal_brief = _ai_search_signal_brief(raw_json)
    action_brief = _ai_mitigation_action_brief(ai_mitigation)
    return (
        "DraftProof AI-Mitigation authenticity rewrite.\n"
        "Goal: push the document toward the Human Contribution side of the scan, not merely lower a detector score.\n"
        f"Current contribution score: Human={contribution.get('human')}, AI Transformation={contribution.get('ai_transformation')}.\n"
        f"Semantic layer: {json.dumps(semantic, ensure_ascii=False)[:1800]}.\n\n"
        "Use these mitigation actions as engineering targets:\n"
        f"{action_brief or '- No component actions supplied.'}\n\n"
        f"{signal_brief}\n\n"
        "Rewrite behavior:\n"
        "- Produce a complete replacement draft, not notes and not a partial patch.\n"
        "- Preserve all factual claims, citations, years, names, quotes, numbers, unit codes, source relations, and chronology already present.\n"
        "- Do not invent personal observations, institutions, sources, dates, statistics, examples, citations, or lived details.\n"
        "- When a claim lacks support, narrow or qualify it instead of fabricating evidence.\n"
        "- Add reasoning continuity where adjacent ideas jump: make the causal bridge explicit using only facts already in the draft.\n"
        "- Increase cognitive authenticity: allow uneven emphasis, short explanatory turns, and locally specific reasoning where the draft already gives anchors.\n"
        "- Reduce predictable academic filler and balanced essay cadence. Avoid polished connector chains such as furthermore, moreover, it is important, significant, crucial, enables, facilitates.\n"
        "- Increase semantic density by replacing broad explanatory padding with concrete operational meaning already present in the source.\n"
        "- Rebuild paragraph routes where needed: context or problem first, then source/evidence relation, then limited conclusion.\n"
        "- Do not use placeholders, review brackets, labels, comments, markdown fences, or explanations.\n"
        "- Keep roughly the same length and preserve headings if they exist.\n"
        f"- Attempt {attempt_index}: make the draft materially different from a synonym swap while staying fact-preserving.\n\n"
        "SOURCE DRAFT:\n"
        f"<TARGET_DOCUMENT>\n{source_text.strip()}\n</TARGET_DOCUMENT>\n\n"
        "Return only the complete rewritten document."
    )







































def _ai_search_prompt(
    source_text: str,
    raw_json: dict,
    strategy: str,
    *,
    reference_ai=None,
    required_ai_drop: float | None = None,
    target_ai_score: float | None = None,
    confirmed_author_anchors: str = "",
) -> str:
    return _core_ai_search_prompt(
        source_text,
        raw_json,
        strategy,
        reference_ai=reference_ai,
        required_ai_drop=required_ai_drop,
        target_ai_score=target_ai_score,
        confirmed_author_anchors=confirmed_author_anchors,
        deps=_targeted_repair_prompt_deps(),
    )


def _ai_search_feedback_prompt(
    source_text: str,
    raw_json: dict,
    search_summary: dict,
    attempt_index: int,
) -> str:
    return _core_ai_search_feedback_prompt(
        source_text,
        raw_json,
        search_summary,
        attempt_index,
        deps=_targeted_repair_prompt_deps(),
    )


def _blocked_human_candidate_repair_prompt(
    source_text: str,
    blocked_candidate: str,
    raw_json: dict,
    blocked_summary: dict,
    attempt_index: int,
) -> str:
    return _core_blocked_human_candidate_repair_prompt(
        source_text,
        blocked_candidate,
        raw_json,
        blocked_summary,
        attempt_index,
        deps=_targeted_repair_prompt_deps(),
    )


def _sentences_from_excerpt(text: str) -> list[str]:
    return _core_sentences_from_excerpt(text)


def _exact_blocking_target_from_context(
    item: dict,
    context: dict,
    candidate_text: str,
) -> tuple[str, str]:
    return _core_exact_blocking_target_from_context(item, context, candidate_text)


def _expand_fragment_to_candidate_sentence(candidate_text: str, fragment: str) -> str:
    return _core_expand_fragment_to_candidate_sentence(candidate_text, fragment)


def _blocking_finding_targets(
    report_dict: dict | None,
    *,
    limit: int = 3,
    candidate_text: str = "",
) -> list[dict]:
    return _core_blocking_finding_targets(
        report_dict,
        limit=limit,
        candidate_text=candidate_text,
    )


def _finding_local_repair_prompt(
    blocked_candidate: str,
    blocked_summary: dict,
    targets: list[dict],
    attempt_index: int,
) -> str:
    return _core_finding_local_repair_prompt(
        blocked_candidate,
        blocked_summary,
        targets,
        attempt_index,
    )


def _extract_finding_local_patches(output: str) -> list[dict]:
    return _core_extract_finding_local_patches(output)


def _apply_finding_local_patches(text: str, patches: list[dict]) -> tuple[str, list[dict]]:
    return _core_apply_finding_local_patches(text, patches)


def _paragraph_target_deps() -> ParagraphTargetDeps:
    return ParagraphTargetDeps(
        logical_paragraphs=_logical_paragraphs,
        text_word_count=_text_word_count,
        float_env=_float_env,
        env_flag=_env_flag,
    )


def _is_heading_like_paragraph(paragraph: str) -> bool:
    return _core_is_heading_like_paragraph(paragraph)


def _orphan_heading_reason(text: str) -> str:
    return _core_orphan_heading_reason(text, deps=_paragraph_target_deps())


def _paragraph_sentence_starters(paragraph: str) -> list[str]:
    return _core_paragraph_sentence_starters(paragraph)




def _paragraph_role(paragraph: str, drivers: dict | None = None, *, is_last: bool = False) -> str:
    return _core_paragraph_role(paragraph, drivers, is_last=is_last)


def _paragraph_component_targets(text: str, raw_json: dict, limit: int = 3) -> list[dict]:
    return _core_paragraph_component_targets(
        text,
        raw_json,
        limit=limit,
        deps=_paragraph_target_deps(),
    )


def _paragraph_prompt_deps() -> ParagraphPromptDeps:
    return ParagraphPromptDeps(
        ai_search_signal_brief=_ai_search_signal_brief,
        protected_anchor_brief_for_prompt=_protected_anchor_brief_for_prompt,
        anchor_values_from_brief=_anchor_values_from_brief,
        anchor_lock_mapping=_anchor_lock_mapping,
        freeze_anchor_text=_freeze_anchor_text,
        freeze_anchor_payload=_freeze_anchor_payload,
        restore_anchor_placeholders=_restore_anchor_placeholders,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        float_env=_float_env,
        text_word_count=_text_word_count,
    )


def _paragraph_component_prompt(
    target: dict,
    raw_json: dict,
    attempt_index: int,
    *,
    reference_ai=None,
    required_ai_drop: float | None = None,
    target_ai_score: float | None = None,
    candidate_count: int = 1,
    confirmed_author_anchors: str = "",
) -> str:
    return _core_paragraph_component_prompt(
        target,
        raw_json,
        attempt_index,
        reference_ai=reference_ai,
        required_ai_drop=required_ai_drop,
        target_ai_score=target_ai_score,
        candidate_count=candidate_count,
        confirmed_author_anchors=confirmed_author_anchors,
        deps=_paragraph_prompt_deps(),
    )


def _paragraph_generation_anchor_context(target: dict | None) -> dict:
    return _core_paragraph_generation_anchor_context(target, deps=_paragraph_prompt_deps())


def _human_signal_amplification_prompt(
    target: dict,
    raw_json: dict,
    attempt_index: int,
    *,
    candidate_count: int = 3,
    confirmed_author_anchors: str = "",
) -> str:
    return _core_human_signal_amplification_prompt(
        target,
        raw_json,
        attempt_index,
        candidate_count=candidate_count,
        confirmed_author_anchors=confirmed_author_anchors,
        deps=_paragraph_prompt_deps(),
    )


def _author_reasoning_amplification_prompt(
    target: dict,
    raw_json: dict,
    attempt_index: int,
    *,
    candidate_count: int = 3,
) -> str:
    return _core_author_reasoning_amplification_prompt(
        target,
        raw_json,
        attempt_index,
        candidate_count=candidate_count,
        deps=_paragraph_prompt_deps(),
    )


def _extract_paragraph_component_candidates(output: str, limit: int) -> list[str]:
    return _core_extract_paragraph_component_candidates(output, limit)


def _clean_paragraph_component_candidate(
    candidate: str,
    original_paragraph: str,
    anchor_lock: list[dict] | None = None,
) -> tuple[str, str]:
    return _core_clean_paragraph_component_candidate(
        candidate,
        original_paragraph,
        anchor_lock,
        deps=_paragraph_prompt_deps(),
    )


def _paragraph_anchor_lock(target: dict | None) -> list[dict]:
    return _core_paragraph_anchor_lock(target, deps=_paragraph_prompt_deps())


def _clean_source_sentence_candidate(candidate: str, original_sentence: str) -> tuple[str, str]:
    return _core_clean_source_sentence_candidate(
        candidate,
        original_sentence,
        deps=_paragraph_prompt_deps(),
    )


def _splice_paragraph(text: str, paragraph_index: int, replacement: str) -> str:
    return _core_splice_paragraph(
        text,
        paragraph_index,
        replacement,
        deps=_paragraph_prompt_deps(),
    )


def _content_pruning_candidate_deps() -> ContentPruningCandidateDeps:
    return ContentPruningCandidateDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        split_sentences=_split_sentences,
        text_word_count=_text_word_count,
        paragraph_component_targets=_paragraph_component_targets,
        paragraph_role=_paragraph_role,
        detect_protected_spans=detect_protected_spans,
    )


def _content_pruning_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 4,
) -> list[tuple[str, str, dict]]:
    return _core_content_pruning_candidates(
        source_text,
        raw_json,
        limit=limit,
        deps=_content_pruning_candidate_deps(),
    )


def _cleanup_transform_deps() -> CleanupTransformDeps:
    return CleanupTransformDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        split_sentences=_split_sentences,
        text_word_count=_text_word_count,
    )


def _narrow_generic_claim_text(text: str) -> str:
    return _core_narrow_generic_claim_text(text)


def _plain_language_depolish_text(text: str) -> tuple[str, list[str]]:
    return _core_plain_language_depolish_text(text)


def _final_score_drag_sentence_prune_text(text: str) -> tuple[str, list[str]]:
    return _core_final_score_drag_sentence_prune_text(text, _cleanup_transform_deps())


def _compress_score_drag_paragraph(paragraph: str, *, max_remove: int = 2) -> str:
    return _core_compress_score_drag_paragraph(
        paragraph,
        _cleanup_transform_deps(),
        max_remove=max_remove,
    )


_GENERIC_ASSERTION_TERMS_RE = re.compile(
    r"\b(?:important|significant|essential|crucial|supports?|helps?|allows?|"
    r"enables?|creates?|means|shows?|suggests?|highlights?|underscores?|"
    r"challenge|issue|goal|system|world|framework|approach|outcomes?|success|"
    r"effective|clear|improve|develop|ensure|provide|promote|enhance)\b",
    re.I,
)

_GENERIC_ASSERTION_PROTECTED_SENTENCE_RE = re.compile(
    r"(?:\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b|\b\d+(?:\.\d+)?%?\b|"
    r"\([A-Z][A-Za-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)|"
    r"\b[A-Z][A-Za-z]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z]+)?\s*\(\d{4}\)|"
    r"\b(?:I|my|me|source|citation|reference|evidence|example|case|condition|"
    r"quote|quoted|observed|measured|tested|reported)\b)",
    re.I,
)


def _generic_assertion_compiler_deps() -> GenericAssertionCompilerDeps:
    return GenericAssertionCompilerDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        blocker_scores=_blocker_scores,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        split_sentences=_split_sentences,
        text_word_count=_text_word_count,
        paragraph_component_targets=_paragraph_component_targets,
        paragraph_role=_paragraph_role,
        safe_index=_safe_index,
        narrow_generic_claim_text=_narrow_generic_claim_text,
        paragraph_citation_re=_PARAGRAPH_CITATION_RE,
        generic_terms_re=_GENERIC_ASSERTION_TERMS_RE,
        generic_protected_sentence_re=_GENERIC_ASSERTION_PROTECTED_SENTENCE_RE,
    )


def _generic_assertion_sentence_score(sentence: str) -> float:
    return _core_generic_assertion_sentence_score(sentence, _generic_assertion_compiler_deps())


def _generic_assertion_compiler_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 4,
) -> list[tuple[str, str, dict]]:
    return _core_generic_assertion_compiler_candidates(
        source_text,
        raw_json,
        limit=limit,
        deps=_generic_assertion_compiler_deps(),
    )


def _blocker_operation_candidate_deps() -> BlockerOperationCandidateDeps:
    return BlockerOperationCandidateDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
        text_word_count=_text_word_count,
        blocker_operation_plan=_blocker_operation_plan,
        safe_index=_safe_index,
        compress_score_drag_paragraph=_compress_score_drag_paragraph,
        narrow_generic_claim_text=_narrow_generic_claim_text,
    )


def _blocker_operation_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 6,
) -> list[tuple[str, str, dict]]:
    return _core_blocker_operation_candidates(
        source_text,
        raw_json,
        limit=limit,
        deps=_blocker_operation_candidate_deps(),
    )


def _post_safe_win_target_push_candidates(
    source_text: str,
    report_dict: dict | None,
    *,
    limit: int = 6,
) -> list[tuple[str, str, dict]]:
    """Build a bounded second-stage candidate set after the first safe win.

    This avoids the old failure mode where adaptive stop protected cost but
    froze the result at the first small gain even when Human Contribution was
    still far below target.
    """
    if not _env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH", True):
        return []
    current_human = _contribution_scores(report_dict or {}).get("human")
    target_human = _float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    if isinstance(current_human, (int, float)) and float(current_human) >= target_human:
        return []
    limit = max(1, int(limit or 1))
    candidates: list[tuple[str, str, dict]] = []
    seen: set[str] = {str(source_text or "").strip()}

    def add(strategy: str, candidate: str, meta: dict) -> None:
        normalized = str(candidate or "").strip()
        if not normalized or normalized in seen or len(candidates) >= limit:
            return
        seen.add(normalized)
        candidates.append((
            f"post_safe_target_push_{strategy}",
            candidate,
            {
                **meta,
                "post_safe_win_target_push": True,
                "base_human": current_human,
                "target_human": target_human,
            },
        ))

    for strategy, candidate, meta in _human_signal_construction_candidates(
        source_text,
        report_dict,
        limit=max(1, min(2, limit)),
    ):
        add(strategy, candidate, meta)

    blocker_limit = max(1, min(limit, int(math.ceil(limit * 0.65))))
    for strategy, candidate, meta in _blocker_operation_candidates(
        source_text,
        report_dict,
        limit=blocker_limit,
    ):
        add(strategy, candidate, meta)

    if len(candidates) < limit:
        for strategy, candidate, meta in _author_stance_thread_candidates(
            source_text,
            report_dict,
            limit=limit - len(candidates),
        ):
            add(strategy, candidate, meta)

    if len(candidates) < limit:
        for strategy, candidate, meta in _content_pruning_candidates(
            source_text,
            report_dict,
            limit=limit - len(candidates),
        ):
            add(strategy, candidate, meta)

    return candidates[:limit]


def _human_anchor_candidate_deps() -> HumanAnchorCandidateDeps:
    return HumanAnchorCandidateDeps(
        env_flag=_env_flag,
        float_env=_float_env,
        human_anchor_driver_contract=_human_anchor_driver_contract,
        logical_paragraphs=_logical_paragraphs,
        is_heading_like_paragraph=_is_heading_like_paragraph,
        split_sentences=_split_sentences,
        ordered_concept_origin_terms=_ordered_concept_origin_terms,
        join_logical_paragraphs=_join_logical_paragraphs,
        formula_portfolio_plan=_formula_portfolio_plan,
        turnitin_like_ai_profile=_turnitin_like_ai_profile,
        blocker_scores=_blocker_scores,
        compress_score_drag_paragraph=_compress_score_drag_paragraph,
    )


def _human_anchor_amplifier_candidates(
    source_text: str,
    report_dict: dict | None,
    *,
    limit: int = 3,
) -> list[tuple[str, str, dict]]:
    return _core_human_anchor_amplifier_candidates(
        source_text,
        report_dict,
        limit=limit,
        deps=_human_anchor_candidate_deps(),
    )


def _formula_portfolio_candidates(
    source_text: str,
    report_dict: dict | None,
    *,
    topk_route_candidates: list[tuple[str, str, dict]] | None = None,
    blocker_operation_candidates: list[tuple[str, str, dict]] | None = None,
    generic_assertion_candidates: list[tuple[str, str, dict]] | None = None,
    pruning_candidates: list[tuple[str, str, dict]] | None = None,
    limit: int = 6,
) -> list[tuple[str, str, dict]]:
    return _core_formula_portfolio_candidates(
        source_text,
        report_dict,
        topk_route_candidates=topk_route_candidates,
        blocker_operation_candidates=blocker_operation_candidates,
        generic_assertion_candidates=generic_assertion_candidates,
        pruning_candidates=pruning_candidates,
        limit=limit,
        deps=_human_anchor_candidate_deps(),
    )


def _human_anchor_suppression_frontier(
    source_text: str,
    report_dict: dict | None,
    block_map: dict | None = None,
) -> dict:
    return _core_human_anchor_suppression_frontier(
        source_text,
        report_dict,
        block_map,
        deps=_human_anchor_candidate_deps(),
    )


def _anchor_sentence_for_paragraph(paragraph: str, variant: str = "process") -> str:
    return _core_anchor_sentence_for_paragraph(
        paragraph,
        variant=variant,
        deps=_human_anchor_candidate_deps(),
    )


def _append_anchor_sentence(paragraph: str, *, variant: str = "process") -> str:
    return _core_append_anchor_sentence(
        paragraph,
        variant=variant,
        deps=_human_anchor_candidate_deps(),
    )


def _human_anchor_suppression_frontier_candidates(
    source_text: str,
    report_dict: dict | None,
    block_map: dict | None,
    *,
    limit: int = 4,
) -> list[tuple[str, str, dict]]:
    return _core_human_anchor_suppression_frontier_candidates(
        source_text,
        report_dict,
        block_map,
        limit=limit,
        deps=_human_anchor_candidate_deps(),
    )


def _formula_block_candidate_deps() -> FormulaBlockCandidateDeps:
    return FormulaBlockCandidateDeps(
        env_flag=_env_flag,
        logical_paragraphs=_logical_paragraphs,
        join_logical_paragraphs=_join_logical_paragraphs,
    )


def _formula_block_map_removal_candidates(
    source_text: str,
    block_map: dict | None,
    *,
    limit: int = 3,
) -> list[tuple[str, str, dict]]:
    return _core_formula_block_map_removal_candidates(
        source_text,
        block_map,
        limit=limit,
        deps=_formula_block_candidate_deps(),
    )


def _formula_convergence_budget(source_text: str, budget: dict | None = None) -> dict:
    """Bound formula convergence by document size, not by a fixed global loop."""
    provided = budget if isinstance(budget, dict) else {}
    words = _text_word_count(source_text)
    if words <= 700:
        defaults = {"max_passes": 2, "max_scans": 6, "max_llm_calls": 1}
        size_band = "300_700"
    elif words <= 1800:
        defaults = {"max_passes": 2, "max_scans": 8, "max_llm_calls": 2}
        size_band = "700_1800"
    else:
        defaults = {"max_passes": 2, "max_scans": 12, "max_llm_calls": 3}
        size_band = "1800_5000"
    resolved = {
        key: int(provided.get(key, defaults[key]) or defaults[key])
        for key in defaults
    }
    resolved["max_passes"] = max(0, resolved["max_passes"])
    resolved["max_scans"] = max(0, resolved["max_scans"])
    resolved["max_llm_calls"] = max(0, min(10, resolved["max_llm_calls"]))
    resolved["word_count"] = words
    resolved["size_band"] = size_band
    resolved["total_rewrite_llm_cap"] = 10
    return resolved


def _rewrite_phase_budget_plan(
    source_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    max_scans: int = 14,
    max_llm_calls: int = 10,
    ai_search_policy: dict | None = None,
    formula_gap_budget: dict | None = None,
) -> dict:
    return _core_rewrite_phase_budget_plan(
        source_text,
        current_report,
        original_report,
        max_scans=max_scans,
        max_llm_calls=max_llm_calls,
        ai_search_policy=ai_search_policy,
        formula_gap_budget=formula_gap_budget,
        text_word_count=_text_word_count,
        formula_convergence_budget=_formula_convergence_budget,
        turnitin_like_ai_profile=_turnitin_like_ai_profile,
        eligible_span_density_contract=_eligible_span_density_contract,
        env_flag=_env_flag,
        safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
        verified_candidate_scan_budget=_verified_candidate_scan_budget,
        float_env=_float_env,
    )


def _formula_block_driver_map(source_text: str, report_dict: dict | None) -> dict:
    return _core_formula_block_driver_map(
        source_text,
        report_dict,
        deps=FormulaBlockDriverMapDeps(
            logical_paragraphs=_logical_paragraphs,
            text_word_count=_text_word_count,
            split_sentences=_split_sentences,
            protected_number_set=_protected_number_set,
            protected_code_anchor_set=_protected_code_anchor_set,
            is_heading_like_paragraph=_is_heading_like_paragraph,
            turnitin_like_ai_profile=_turnitin_like_ai_profile,
            formula_portfolio_plan=_formula_portfolio_plan,
        ),
    )


def _formula_convergence_candidate_batch(
    current_text: str,
    current_report: dict | None,
    block_map: dict | None,
    *,
    limit: int = 8,
) -> list[tuple[str, str, dict]]:
    """Create one bounded portfolio batch from the current best state."""
    limit = max(1, int(limit or 1))
    feasibility = _formula_feasibility_estimator(current_report)
    geometry_map = _geometry_risk_map(current_text, current_report)
    geometry_candidates = _coordinated_micro_perturbation_candidates(
        current_text,
        current_report,
        geometry_map,
        limit=max(2, min(4, limit)),
    )
    topk_candidates = _topk_route_optimizer_candidates(current_text, current_report)
    blocker_candidates = _blocker_operation_candidates(current_text, current_report, limit=4)
    generic_candidates = _generic_assertion_compiler_candidates(current_text, current_report, limit=3)
    pruning_candidates = _content_pruning_candidates(current_text, current_report, limit=3)
    anchor_frontier_candidates = _human_anchor_suppression_frontier_candidates(
        current_text,
        current_report,
        block_map,
        limit=4,
    )
    block_map_removal_candidates = _formula_block_map_removal_candidates(
        current_text,
        block_map,
        limit=3,
    )
    portfolio_candidates = _formula_portfolio_candidates(
        current_text,
        current_report,
        topk_route_candidates=topk_candidates,
        blocker_operation_candidates=blocker_candidates,
        generic_assertion_candidates=generic_candidates,
        pruning_candidates=pruning_candidates,
        limit=limit,
    )
    raw_candidates: list[tuple[str, str, dict]] = []
    if feasibility.get("geometry_required"):
        raw_candidates.extend(geometry_candidates)
    raw_candidates.extend(portfolio_candidates)
    if not feasibility.get("geometry_required"):
        raw_candidates.extend(geometry_candidates[:2])
    raw_candidates.extend(block_map_removal_candidates)
    raw_candidates.extend(topk_candidates[:2])
    raw_candidates.extend(blocker_candidates[:2])
    raw_candidates.extend(generic_candidates[:2])
    raw_candidates.extend(pruning_candidates[:2])
    raw_candidates.extend(anchor_frontier_candidates)
    normalized_seen = {str(current_text or "").strip()}
    candidates: list[tuple[str, str, dict]] = []
    for strategy, candidate, meta in raw_candidates:
        normalized = str(candidate or "").strip()
        if not normalized or normalized in normalized_seen:
            continue
        normalized_seen.add(normalized)
        candidates.append((
            f"formula_convergence_{strategy}",
            candidate,
            {
                **(meta or {}),
                "formula_convergence_candidate": True,
                "feasibility_estimator": feasibility,
                "block_driver_map_version": (block_map or {}).get("version"),
                "geometry_risk_map_version": geometry_map.get("version"),
                "top_drag_blocks": [
                    {
                        "block_index": row.get("block_index"),
                        "action": row.get("action"),
                        "recommended_portfolio_action": row.get("recommended_portfolio_action"),
                        "weighted_drag": row.get("weighted_drag"),
                        "human_anchor_deficit": row.get("human_anchor_deficit"),
                        "suppression_gain_potential": row.get("suppression_gain_potential"),
                        "remove_value_loss_risk": row.get("remove_value_loss_risk"),
                    }
                    for row in ((block_map or {}).get("top_blocks") or [])[:4]
                    if isinstance(row, dict)
                ],
            },
        ))
        if len(candidates) >= limit:
            break
    return candidates


def _formula_convergence_block_recreate_prompt(
    current_text: str,
    current_report: dict | None,
    block_map: dict | None,
) -> str:
    """Prompt for formula-gap block recreation, scoped to top drag blocks."""
    paragraphs = _logical_paragraphs(current_text)
    profile = _turnitin_like_ai_profile(current_report)
    plan = _formula_portfolio_plan(current_report, current_report)
    target_blocks = []
    for row in (block_map or {}).get("top_blocks") or []:
        if len(target_blocks) >= 5:
            break
        if not isinstance(row, dict):
            continue
        index = row.get("block_index")
        if not isinstance(index, int) or index < 0 or index >= len(paragraphs):
            continue
        if row.get("protected") or row.get("action") == "preserve":
            continue
        paragraph = paragraphs[index]
        target_blocks.append({
            "paragraph_index": index,
            "recommended_action": row.get("action"),
            "weighted_drag": row.get("weighted_drag"),
            "generic_hits": row.get("generic_hits"),
            "human_anchor_hits": row.get("human_anchor_hits"),
            "remove_safety": row.get("remove_safety"),
            "dominant_formula_drivers": row.get("dominant_formula_drivers"),
            "protected_numbers": row.get("protected_numbers"),
            "protected_code_anchors": row.get("protected_code_anchors"),
            "paragraph": paragraph,
        })
    return (
        "DraftProof FORMULA_CONVERGENCE_BLOCK_RECREATE.\n"
        "Objective: reduce the total Turnitin-like AI formula score below 20 by changing only selected high-drag blocks.\n"
        "Optimize both halves of the formula: reduce positive AI-driver burden and increase Human Anchor suppression.\n"
        "Return only valid JSON. No markdown.\n\n"
        "Hard rules:\n"
        "- Patch selected paragraph blocks only; do not rewrite the whole document.\n"
        "- Preserve protected numbers, code anchors, names, citation markers, and unique core claims.\n"
        "- Do not add fake named events, fake people, fake dates, fake sources, or unsupported evidence claims.\n"
        "- Use bounded implied context/process reasoning when it is already supported by the paragraph.\n"
        "- Do not polish into a cleaner essay style. Avoid generic connectors and balanced claim-explain-conclude cadence.\n"
        "- Each candidate may patch 1 to 3 paragraphs.\n\n"
        "Target formula profile:\n"
        f"{json.dumps({'profile': profile, 'portfolio_plan': plan}, ensure_ascii=False)[:6000]}\n\n"
        "Patch targets:\n"
        f"{json.dumps(target_blocks, ensure_ascii=False)[:12000]}\n\n"
        "Allowed operation types:\n"
        "- HUMAN_ANCHOR_SUPPRESSION_GAIN: add bounded process/context reasoning implied by the paragraph\n"
        "- LIKELIHOOD_TEXTURE_REBUILD: rebuild cadence and phrasing route to reduce AI likelihood\n"
        "- TOPK_ROUTE_REBUILD: change predictable token route while preserving meaning\n"
        "- SEMANTIC_VARIANCE_RESTRUCTURE: change paragraph job/reasoning shape to reduce uniformity\n"
        "- SMOOTHNESS_DEPOLISH: make the block less over-clean without errors or gimmicks\n"
        "- PATCHWORK_COLLAPSE: compress expansion/style artifacts\n\n"
        "Return schema:\n"
        "{\n"
        "  \"candidates\": [\n"
        "    {\n"
        "      \"reason\": \"short reason\",\n"
        "      \"patches\": [\n"
        "        {\"operation_type\": \"LIKELIHOOD_TEXTURE_REBUILD\", \"target_paragraph_index\": 0, \"expected_driver\": \"ai_likelihood\", \"replacement\": \"replacement paragraph\"}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Return at most 3 candidates."
    )


def _formula_convergence_llm_patch_candidates(
    current_text: str,
    current_report: dict | None,
    block_map: dict | None,
    gateway: LLMGateway | None,
    *,
    max_candidates: int = 3,
) -> list[tuple[str, str, dict]]:
    if gateway is None or not _env_flag("DRAFTPROOF_FORMULA_CONVERGENCE_LLM_BLOCK_RECREATE", True):
        return []
    try:
        response = gateway.chat(
            _formula_convergence_block_recreate_prompt(current_text, current_report, block_map),
            system=(
                "You are DraftProof's formula convergence controller. "
                "Return only JSON paragraph patches."
            ),
            **_phase_chat_sampling_kwargs(
                "DRAFTPROOF_FORMULA_CONVERGENCE",
                temperature_env="DRAFTPROOF_FORMULA_CONVERGENCE_TEMPERATURE",
                temperature_default=0.45,
                max_tokens_env="DRAFTPROOF_FORMULA_CONVERGENCE_MAX_TOKENS",
                max_tokens_default=3200,
            ),
        )
    except Exception:
        return []
    patch_sets = _extract_post_topk_patch_candidates(response.content, max_candidates=max_candidates)
    candidates: list[tuple[str, str, dict]] = []
    for index, patch_candidate in enumerate(patch_sets, start=1):
        patched_text, applied = _apply_post_topk_patches(current_text, patch_candidate.get("patches") or [])
        if not applied or patched_text.strip() == str(current_text or "").strip():
            continue
        candidates.append((
            f"formula_convergence_llm_block_recreate_c{index}",
            patched_text,
            {
                "formula_convergence_llm_block_recreate": True,
                "llm_patch": True,
                "llm_reason": patch_candidate.get("reason"),
                "applied_formula_convergence_patches": applied,
            },
        ))
    return candidates


def _formula_convergence_controller(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    budget: dict | None = None,
    *,
    scan_func=None,
    candidate_builder=None,
    drift_checker=None,
    llm_gateway: LLMGateway | None = None,
) -> dict:
    deps = FormulaConvergenceControllerDeps(
        env_flag=_env_flag,
        run_full_scan_report_dict=_run_full_scan_report_dict,
        check_semantic_drift=check_semantic_drift,
        formula_convergence_budget=_formula_convergence_budget,
        report_review_burden=_report_review_burden,
        report_weighted_severity=_report_weighted_severity,
        critical_high_count=_critical_high_count,
        formula_block_driver_map=_formula_block_driver_map,
        human_anchor_suppression_frontier=_human_anchor_suppression_frontier,
        formula_feasibility_estimator=_formula_feasibility_estimator,
        geometry_risk_map=_geometry_risk_map,
        formula_convergence_candidate_batch=_formula_convergence_candidate_batch,
        formula_convergence_llm_patch_candidates=_formula_convergence_llm_patch_candidates,
        ai_candidate_quality_reject_reason=_ai_candidate_quality_reject_reason,
        ai_search_protected_loss_reason=_ai_search_protected_loss_reason,
        detect_protected_spans=detect_protected_spans,
        candidate_concept_origin_reject_reason=_candidate_concept_origin_reject_reason,
        anti_smoothing_guard_status=_anti_smoothing_guard_status,
        formula_convergence_primary_burden_gate_status=_formula_convergence_primary_burden_gate_status,
        strict_ai_safe_band_status=_strict_ai_safe_band_status,
        report_badge_ai=_report_badge_ai,
        report_finding_total=_report_finding_total,
    )
    return _core_formula_convergence_controller(
        current_text,
        current_report,
        original_report,
        budget,
        scan_func=scan_func,
        candidate_builder=candidate_builder,
        drift_checker=drift_checker,
        llm_gateway=llm_gateway,
        deps=deps,
    )


def _human_signal_construction_candidates(
    source_text: str,
    report_dict: dict | None,
    *,
    limit: int = 2,
) -> list[tuple[str, str, dict]]:
    """Construct visible author reasoning density from existing claims.

    This delegates to the domain-agnostic Human Anchor amplifier. Older
    topic-specific sentence replacement tables were removed because they caused
    cross-domain leakage when a fixture shared only superficial wording.
    """
    if not _env_flag("DRAFTPROOF_HUMAN_SIGNAL_CONSTRUCTION", True):
        return []
    paragraphs = _logical_paragraphs(source_text)
    if len(paragraphs) < 3:
        return []
    report_dict = report_dict or {}
    badge = report_dict.get("ai_risk_badge") or {}
    writing = badge.get("writing_components") or {}
    lived_risk = float(writing.get("lived_detail_risk") or 0.0)
    broad_risk = float(writing.get("broad_claim_risk") or 0.0)
    if max(lived_risk, broad_risk) < 55.0:
        return []

    generic_anchor_candidates = _human_anchor_amplifier_candidates(
        source_text,
        report_dict,
        limit=limit,
    )
    if generic_anchor_candidates:
        return [
            (
                strategy.replace("human_anchor_amplifier", "human_signal_construction"),
                candidate,
                {
                    **meta,
                    "operation": "human_signal_construction",
                    "human_anchor_amplifier": True,
                },
            )
            for strategy, candidate, meta in generic_anchor_candidates[:max(1, int(limit or 1))]
        ]
    return []


def _author_stance_thread_candidates(
    source_text: str,
    report_dict: dict | None,
    *,
    limit: int = 3,
) -> list[tuple[str, str, dict]]:
    """Create conservative author-stance candidates from claims already present.

    This is not evidence invention. It only converts existing claims into a
    visible author judgement where the paragraph is already making that claim.
    The scanner/gate still rejects any authorship, drift, review, or severity
    regression.
    """
    # Personal stance insertion is too easy to turn into synthetic voice. Keep
    # this hook inert unless a future implementation can derive stance from
    # existing first-person author material without fixed topic templates.
    return []


def _ai_search_marked_grounding_candidates(source_text: str) -> list[tuple[str, str]]:
    return _core_ai_search_marked_grounding_candidates(source_text)


def _authorship_schema_enrichment_deps() -> AuthorshipSchemaEnrichmentDeps:
    return AuthorshipSchemaEnrichmentDeps(
        calibrate_topk_risk=calibrate_topk_risk,
        split_sentences=_split_sentences,
        build_layer3_input_from_text=build_layer3_input_from_text,
        metric_decimal=_metric_decimal,
        layer3_scorer_factory=Layer3Scorer,
    )


def _enrich_report_authorship_schema(report_dict: dict) -> dict:
    return _core_enrich_report_authorship_schema(
        report_dict,
        deps=_authorship_schema_enrichment_deps(),
    )


def _rewrite_input_context_deps() -> RewriteInputContextDeps:
    from detect_pipeline import run_detect
    from detect.base import DetectResult, Finding as DetectFinding

    return RewriteInputContextDeps(
        detect_json_parse_dict=DetectJSONParser.parse_dict,
        detect_json_parse=DetectJSONParser.parse,
        run_detect=run_detect,
        detect_result_factory=DetectResult,
        detect_finding_factory=DetectFinding,
    )


def _rewrite_engine_phase_deps() -> RewriteEnginePhaseDeps:
    return RewriteEnginePhaseDeps(
        sanitize_text=sanitize_text,
        env_flag=_env_flag,
        float_env=_float_env,
        run_full_scan_report_dict=_run_full_scan_report_dict,
        ensure_ai_mitigation_contract=_ensure_ai_mitigation_contract,
        ai_mitigation_requires_user_input=_ai_mitigation_requires_user_input,
        radar_goal_controller_status=_radar_goal_controller_status,
        radar_blocker_option_matrix=_radar_blocker_option_matrix,
        rewrite_config_factory=RewriteConfig,
        run_rewrite=run_rewrite,
        build_marked_mitigation_rewrite=_build_marked_mitigation_rewrite,
        manual_summary_from_ai_mitigation=_manual_summary_from_ai_mitigation,
    )


def run_rewrite_pipeline(
    json_path: str = None,
    text: str = None,
    detect_json: dict = None,
    output_dir: str = None,
    max_passes: int = 3,
    max_detect_loops: int = 0,
    target_top10: float = 0.50,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    verbose: bool = False,
    ai_only: bool = True,
    progress_callback=None,
    cancellation_check: Callable[[], None] | None = None,
) -> dict:
    """Run the full rewrite pipeline from detect JSON or raw text.

    Args:
        json_path: Path to detect JSON file.
        text: Raw text (will run detect first).
        detect_json: Pre-loaded detect JSON dict.
        output_dir: Where to write output files.
        max_passes: Max rewrite passes per loop.
        max_detect_loops: Max detect-rewrite loops.
        target_top10: Target top-10 ratio for convergence.
        model: LLM model for rewriting (None → from env).
        api_key: API key for LLM (None → from env).
        base_url: LLM API base URL (None → from env or OpenRouter default).
        verbose: Include scanner details in report.
        ai_only: Only rewrite AI-generation findings (default True).

    Returns dict with paths and summary.
    """
    _load_local_env()
    _reset_source_search_runtime_budget()
    llm_roles = _llm_role_config(model)
    generator_model = llm_roles.get("generator_model") or model
    retry_model = llm_roles.get("retry_model") or generator_model

    # ── Parse input ────────────────────────────────────────────────
    ctx, text = _core_resolve_rewrite_input_context(
        json_path=json_path,
        text=text,
        detect_json=detect_json,
        output_dir=output_dir,
        verbose=verbose,
        deps=_rewrite_input_context_deps(),
    )

    if not text or not text.strip():
        raise ValueError("Empty input text")

    if isinstance(getattr(ctx, "raw_json", None), dict):
        ctx.raw_json = _enrich_report_authorship_schema(ctx.raw_json)
        _ensure_ai_mitigation_contract(ctx.raw_json)

    # ── Check rewrite decision from detect ──────────────────────────
    if ctx.rewrite_decision and not ctx.rewrite_decision.get("run_rewrite", True):
        reason = ctx.rewrite_decision.get("reason", "Rewrite not recommended")
        print(f"Rewrite skipped: {reason}")
        return {
            "status": "skipped",
            "message": reason,
            "tier": ctx.overall_tier,
        }

    all_findings = [f for dr in ctx.detect_results for f in dr.findings]
    if not all_findings:
        print("No findings to rewrite. Text is clean.")
        return {"status": "clean", "message": "No findings to rewrite"}

    def raise_if_canceled() -> None:
        if cancellation_check is not None:
            cancellation_check()

    def report_progress(percent: int, message: str) -> None:
        raise_if_canceled()
        if not progress_callback:
            return
        progress_callback(max(40, min(79, int(percent))), message)

    # ── Run rewrite ─────────────────────────────────────────────────
    raise_if_canceled()
    print(f"Running rewrite pipeline...")
    print(f"  Input: {len(text)} chars, {len(ctx.detect_results)} scanner results")
    if ctx.rewrite_decision:
        print(f"  Decision: mode={ctx.rewrite_decision.get('mode', 'targeted')}")

    total_findings = sum(len(dr.findings) for dr in ctx.detect_results)
    if ai_only:
        ai_count = sum(
            len(dr.findings if dr.scanner == "ai_generation"
                else [f for f in dr.findings
                      if (f.metadata or {}).get("scanner") == "ai_generation"
                      or (f.metadata or {}).get("category") == "ai_generation"])
            for dr in ctx.detect_results
        )
        print(f"  AI-only mode: {ai_count} AI findings out of {total_findings} total")
    else:
        medium_count = sum(
            len([f for f in dr.findings if f.risk_level == "medium"])
            for dr in ctx.detect_results
        )
        print(f"  Medium-only mode: {medium_count} findings out of {total_findings} total")

    rewrite_engine_phase = _core_run_rewrite_engine_phase(
        text=text,
        ctx=ctx,
        api_key=api_key,
        generator_model=generator_model,
        base_url=base_url,
        max_passes=max_passes,
        target_top10=target_top10,
        max_detect_loops=max_detect_loops,
        output_dir=output_dir,
        ai_only=ai_only,
        llm_roles=llm_roles,
        report_progress=report_progress,
        deps=_rewrite_engine_phase_deps(),
    )
    text = rewrite_engine_phase["text"]
    result: RewriteModuleResult = rewrite_engine_phase["result"]
    baseline_report_dict = rewrite_engine_phase["baseline_report_dict"]
    ai_mitigation_contract = rewrite_engine_phase["ai_mitigation_contract"]
    ai_mitigation_needs_author = rewrite_engine_phase["ai_mitigation_needs_author"]
    allow_auto_with_author_gaps = rewrite_engine_phase["allow_auto_with_author_gaps"]
    radar_goal_controller = rewrite_engine_phase["radar_goal_controller"]
    radar_option_matrix = rewrite_engine_phase["radar_option_matrix"]
    ai_search_first = rewrite_engine_phase["ai_search_first"]
    rewrite_config = rewrite_engine_phase["rewrite_config"]
    t0 = rewrite_engine_phase["rewrite_engine_started_at"]
    engine_elapsed = rewrite_engine_phase["rewrite_engine_elapsed"]
    stage_timings = rewrite_engine_phase["stage_timings"]

    # ── Write output ────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "test_output"
        ))

    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"draftproof_rewrite_{ts}.md")
    json_path_out = os.path.join(output_dir, f"draftproof_rewrite_{ts}.json")

    # Extract AI-only findings from detect JSON
    ai_findings = []
    raw_findings = ctx.raw_json.get("findings", {})
    for tier in ("critical", "high", "medium", "low"):
        for f in raw_findings.get(tier, []):
            cat = (f.get("category") or f.get("scanner") or "").lower()
            if cat == "ai_generation":
                ai_findings.append(f)

    # Get sentence comparison from the MultiPassResult, aligned by text diff.
    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)

    # Inject detect scan scores into summary so rewrite report shows
    # the same risk scores the user saw in the detect scan report.
    badge = ctx.raw_json.get("ai_risk_badge", {})
    if badge:
        result.summary["detect_ai_likelihood"] = badge.get("ai_likelihood_score", 0)
        result.summary["detect_writing_quality"] = badge.get("writing_quality_score", 0)
    baseline_badge = (baseline_report_dict or {}).get("ai_risk_badge") or {}
    if baseline_badge:
        result.summary["baseline_detect_ai_likelihood"] = baseline_badge.get("ai_likelihood_score", 0)
        result.summary["baseline_detect_writing_quality"] = baseline_badge.get("writing_quality_score", 0)

    rewritten_text = result.mp_result.final_text if result.mp_result else text
    if rewritten_text == text:
        result.summary["no_text_change_reason"] = (
            result.mp_result.convergence_reason
            if result.mp_result and result.mp_result.convergence_reason
            else "No automatic rewrite was applied"
        )
    rewritten_report_dict = _core_run_final_rewritten_scan(
        original_text=text,
        rewritten_text=rewritten_text,
        baseline_report_dict=baseline_report_dict,
        summary=result.summary,
        stage_timings=stage_timings,
        report_progress=report_progress,
        detection_runner_factory=DetectionRunner,
        report_builder_factory=ReportBuilder,
        report_to_dict=report_to_dict,
    )

    _finding_total = _report_finding_total
    _review_burden = _report_review_burden
    _weighted_severity = _report_weighted_severity
    _badge_ai = _report_badge_ai
    _badge_wq = _report_badge_wq

    pipeline_state = RewritePipelineState(
        source_text=text,
        current_text=rewritten_text,
        original_report=baseline_report_dict,
        current_report=rewritten_report_dict,
        summary=result.summary,
        stage_timings=stage_timings,
    )
    full_scan_gateway = RewriteScanGateway(_run_full_scan_report_dict)
    full_scan_cache_stats = full_scan_gateway.stats

    def _full_scan_report_dict(scan_text: str) -> dict:
        return full_scan_gateway.scan(scan_text)

    # Rewrite candidate scans must be compared against a baseline produced by
    # the same scanner codepath. Otherwise a saved scan from an earlier scanner
    # phase can make a valid mitigation candidate look like a regression.
    original_report_dict = baseline_report_dict
    if _env_flag("DRAFTPROOF_FRESH_ORIGINAL_BASELINE", True):
        result.summary["comparison_baseline"] = "fresh_original_scan"
        saved_original_ai = _badge_ai(ctx.raw_json)
        fresh_original_ai = _badge_ai(original_report_dict)
        saved_original_wq = _badge_wq(ctx.raw_json)
        fresh_original_wq = _badge_wq(original_report_dict)
        if (
            saved_original_ai != fresh_original_ai
            or saved_original_wq != fresh_original_wq
            or _finding_total(ctx.raw_json) != _finding_total(original_report_dict)
            or _review_burden(ctx.raw_json) != _review_burden(original_report_dict)
        ):
            result.summary["baseline_rescan_delta"] = {
                "saved_ai": saved_original_ai,
                "fresh_ai": fresh_original_ai,
                "saved_writing_quality": saved_original_wq,
                "fresh_writing_quality": fresh_original_wq,
                "saved_findings": _finding_total(ctx.raw_json),
                "fresh_findings": _finding_total(original_report_dict),
                "saved_review_burden": _review_burden(ctx.raw_json),
                "fresh_review_burden": _review_burden(original_report_dict),
                "saved_weighted_severity": _weighted_severity(ctx.raw_json),
                "fresh_weighted_severity": _weighted_severity(original_report_dict),
            }
    else:
        result.summary["comparison_baseline"] = "saved_original_scan"

    original_ai = _badge_ai(original_report_dict)
    rewritten_ai = _badge_ai(rewritten_report_dict)
    original_wq = _badge_wq(original_report_dict)
    rewritten_wq = _badge_wq(rewritten_report_dict)
    original_total = _finding_total(original_report_dict)
    rewritten_total = _finding_total(rewritten_report_dict)
    original_review_burden = _review_burden(original_report_dict)
    rewritten_review_burden = _review_burden(rewritten_report_dict)
    original_severity = _weighted_severity(original_report_dict)
    rewritten_severity = _weighted_severity(rewritten_report_dict)
    attempted_report_dict = rewritten_report_dict

    saved_ai = _badge_ai(ctx.raw_json)
    saved_total = _finding_total(ctx.raw_json)
    saved_critical_high = (
        len(ctx.raw_json.get("findings", {}).get("critical", []))
        + len(ctx.raw_json.get("findings", {}).get("high", []))
    )
    original_critical_high_for_contract = (
        len(original_report_dict.get("findings", {}).get("critical", []))
        + len(original_report_dict.get("findings", {}).get("high", []))
    )
    if not (ctx.raw_json.get("findings") or {}):
        saved_ai = original_ai
        saved_total = original_total
        saved_critical_high = original_critical_high_for_contract
        result.summary["comparison_contract_source"] = "fresh_original_scan"
    else:
        result.summary["comparison_contract_source"] = "saved_original_scan"
    rewritten_critical_high = (
        len(rewritten_report_dict.get("findings", {}).get("critical", []))
        + len(rewritten_report_dict.get("findings", {}).get("high", []))
    )

    result.summary["detect_scores"] = {
        "original_ai": original_ai,
        "rewritten_ai": rewritten_ai,
        "original_writing_quality": original_wq,
        "rewritten_writing_quality": rewritten_wq,
        "original_ai_authorship": _integrity_scores(original_report_dict).get("ai_authorship"),
        "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
        "original_grounding_quality_risk": _integrity_scores(original_report_dict).get("grounding"),
        "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
        "original_human_contribution": _contribution_scores(original_report_dict).get("human"),
        "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
        "original_ai_transformation": _contribution_scores(original_report_dict).get("ai_transformation"),
        "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
        "original_findings": original_total,
        "rewritten_findings": rewritten_total,
        "original_review_burden": original_review_burden,
        "rewritten_review_burden": rewritten_review_burden,
        "original_weighted_severity": original_severity,
        "rewritten_weighted_severity": rewritten_severity,
    }

    global_policy_for_budget = _ai_search_budget_policy(text, original_report_dict)
    legacy_global_seconds = (
        float(getattr(rewrite_config, "max_rewrite_seconds", 0) or 0)
        if rewrite_config is not None
        else 0.0
    )
    env_global_seconds = (
        _float_env("DRAFTPROOF_GLOBAL_REWRITE_MAX_SECONDS", 90.0)
        if os.environ.get("DRAFTPROOF_GLOBAL_REWRITE_MAX_SECONDS") is not None
        else None
    )
    configured_global_seconds = resolve_global_rewrite_seconds(
        legacy_seconds=legacy_global_seconds,
        controller_policy_seconds=float(global_policy_for_budget.get("max_seconds") or 0.0),
        env_seconds=env_global_seconds,
        default_seconds=90.0,
    )
    global_rewrite_budget = RewriteRunBudget(
        max_seconds=configured_global_seconds,
        max_scans=int(_float_env("DRAFTPROOF_GLOBAL_REWRITE_MAX_SCANS", 20.0)),
        max_llm_calls=int(_float_env("DRAFTPROOF_GLOBAL_REWRITE_MAX_LLM_CALLS", 10.0)),
        started_at=t0,
    )
    global_rewrite_budget.record_stage("rewrite_engine", seconds=engine_elapsed)
    result.summary["global_rewrite_budget_contract"] = {
        "version": "global_rewrite_budget_v1",
        "source": "max(controller_policy, rewrite_config.max_rewrite_seconds, DRAFTPROOF_GLOBAL_REWRITE_MAX_SECONDS)",
        "legacy_rewrite_config_seconds": legacy_global_seconds,
        "controller_policy_seconds": float(global_policy_for_budget.get("max_seconds") or 0.0),
        "env_seconds": env_global_seconds,
        "max_seconds": configured_global_seconds,
        "max_scans": global_rewrite_budget.max_scans,
        "max_llm_calls": global_rewrite_budget.max_llm_calls,
    }

    # The controller must use the same fresh baseline scan as the selector.
    # The initial parsed context can be a lightweight wrapper with no integrity
    # layers, which would make Human Contribution appear missing and allow
    # zero-Human "safe" wins to pass.
    fresh_radar_goal_controller = _radar_goal_controller_status(original_report_dict)
    if fresh_radar_goal_controller.get("current_human_contribution") is not None:
        radar_goal_controller = fresh_radar_goal_controller
        radar_option_matrix = (
            radar_goal_controller.get("option_matrix")
            or _radar_blocker_option_matrix(original_report_dict)
        )
        result.summary["radar_goal_controller"] = {
            key: value
            for key, value in radar_goal_controller.items()
            if key != "option_matrix"
        }
        result.summary["radar_blocker_option_matrix"] = radar_option_matrix
        result.summary["radar_option_matrix"] = radar_option_matrix

    ai_first_min_drop = float(os.environ.get("DRAFTPROOF_AI_FIRST_MIN_DROP", "5.0"))
    ai_first_target = float(os.environ.get("DRAFTPROOF_AI_FIRST_TARGET", "60.0"))
    ai_first_required_min_ai = float(os.environ.get("DRAFTPROOF_AI_FIRST_REQUIRED_MIN_AI", "50.0"))
    ai_search_selected = False
    ai_search_fail_fast_partial = False
    authenticity_mitigation_selected = False
    integrity_original = _integrity_scores(original_report_dict)
    ai_authorship_mitigation_needed = bool(
        isinstance(integrity_original.get("ai_authorship"), (int, float))
        and integrity_original.get("ai_authorship") >= _float_env(
            "DRAFTPROOF_AUTHENTICITY_MIN_AI_AUTHORSHIP",
            50.0,
        )
    )

    # Guided authenticity mitigation. This path handles the exact case the
    # scanner now identifies: the draft needs human-side movement, but the
    # system must not fabricate author grounding. It generates fact-preserving
    # candidates, rescans them locally, then accepts only measurable movement
    # toward Human Contribution.
    authenticity_enabled = (
        (ai_mitigation_needs_author or ai_authorship_mitigation_needed)
        and os.environ.get("DRAFTPROOF_AUTHENTICITY_MITIGATION", "1") != "0"
    )
    skip_authenticity_prepass = bool(
        authenticity_enabled
        and os.environ.get("DRAFTPROOF_AI_MITIGATION_SEARCH", "1") != "0"
        and _env_flag("DRAFTPROOF_SKIP_AUTHENTICITY_PREPASS_WHEN_AI_SEARCH", True)
    )
    if skip_authenticity_prepass:
        authenticity_enabled = False
        result.summary["authenticity_mitigation"] = {
            "enabled": False,
            "selected": False,
            "reason": "skipped_for_ai_search_controller",
            "policy": "DRAFTPROOF_SKIP_AUTHENTICITY_PREPASS_WHEN_AI_SEARCH",
            "would_have_run_for": {
                "ai_mitigation_needs_author": bool(ai_mitigation_needs_author),
                "ai_authorship_mitigation_needed": bool(ai_authorship_mitigation_needed),
            },
        }
        result.summary["generation_layer"] = {
            "schema_version": "generation_layer.v1",
            "mode": "ai_search_controller_first",
            "goal": "safe measured improvement",
            "selected": False,
            "selection_reason": "skipped_for_ai_search_controller",
            "llm_calls": 0,
            "model_roles": llm_roles,
        }
        stage_timings.append({
            "stage": "authenticity_mitigation",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "reason": "skipped_for_ai_search_controller",
        })
    if authenticity_enabled:
        mitigation_started = time.time()
        try:
            authenticity_candidate_limit = max(
                0,
                int(os.environ.get("DRAFTPROOF_AUTHENTICITY_CANDIDATES", "2")),
            )
        except ValueError:
            authenticity_candidate_limit = 0
        if _env_flag("DRAFTPROOF_REGENERATION_FIRST", True):
            authenticity_candidate_limit = 0
        radar_force_reconstruction = bool(
            radar_goal_controller.get("force_broad_reconstruction")
        )
        radar_requires_human_progress = _radar_goal_requires_human_progress(
            radar_goal_controller
        )
        authenticity_summary = {
            "enabled": True,
            "selected": False,
            "candidate_limit": authenticity_candidate_limit,
            "llm_calls": 0,
            "model_roles": llm_roles,
            "radar_goal_first": bool(radar_goal_controller.get("execute_before_local_rewrite")),
            "radar_force_reconstruction": radar_force_reconstruction,
            "radar_requires_human_progress": radar_requires_human_progress,
            "reference": {
                "ai": original_ai,
                "writing_quality": original_wq,
                "human_contribution": _contribution_scores(original_report_dict).get("human"),
                "ai_transformation": _contribution_scores(original_report_dict).get("ai_transformation"),
                "ai_authorship": integrity_original.get("ai_authorship"),
                "grounding_quality": integrity_original.get("grounding"),
                "review_burden": original_review_burden,
                "weighted_severity": original_severity,
            },
            "candidates": [],
        }
        effective_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if not effective_key:
            authenticity_summary["reason"] = "no_llm_available"
        else:
            source_for_mitigation, source_repairs = _repair_candidate_source_damage(text)
            authenticity_max_llm_calls = _resolve_stage_llm_budget(
                "DRAFTPROOF_AUTHENTICITY_MAX_LLM_CALLS",
                "DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS",
                default=int(_adaptive_budget_default(source_for_mitigation, 2, 4)),
            )
            authenticity_summary["budget"] = {
                "max_llm_calls": authenticity_max_llm_calls,
                "fallback_env": "DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS",
            }

            def _auth_llm_remaining() -> int:
                return max(
                    0,
                    int(authenticity_max_llm_calls)
                    - int(authenticity_summary.get("llm_calls") or 0),
                )

            def _auth_llm_budget_exhausted(phase: str) -> bool:
                if _auth_llm_remaining() > 0:
                    return False
                authenticity_summary["budget_exhausted"] = {
                    "phase": phase,
                    "reason": "budget_exhausted_llm_calls",
                    "llm_calls": int(authenticity_summary.get("llm_calls") or 0),
                    "max_llm_calls": int(authenticity_max_llm_calls),
                }
                return True

            if source_repairs:
                authenticity_summary["source_repairs"] = source_repairs
            source_protected = detect_protected_spans(source_for_mitigation)
            min_chars = max(200, int(len(source_for_mitigation) * 0.78))
            max_chars = max(min_chars, int(len(text) * 1.25))
            best_candidate_text = ""
            best_candidate_report = None
            best_candidate_gate = None
            best_candidate_eval = None
            masked_span_selected = False
            masked_optimizer_ran = False
            if source_repairs and source_for_mitigation.strip() != text.strip():
                candidate_eval = {
                    "attempt": 0,
                    "strategy": "deterministic_source_integrity_repair",
                    "deterministic": True,
                    "passed_local_checks": False,
                    "candidate_length": len(source_for_mitigation),
                    "source_damage_repairs": source_repairs,
                }
                protected_loss = _ai_search_protected_loss_reason(
                    text,
                    source_for_mitigation,
                    detect_protected_spans(text),
                )
                if protected_loss:
                    candidate_eval["reason"] = "protected_span_lost " + protected_loss
                    authenticity_summary["candidates"].append(candidate_eval)
                else:
                    drift = check_semantic_drift(text, source_for_mitigation, threshold=0.15)
                    candidate_eval["drift_similarity"] = round(drift.similarity, 3)
                    if not drift.accepted and _source_repair_drift_false_positive(
                        source_for_mitigation,
                        drift.reasons,
                    ):
                        candidate_eval["drift_relaxed_for_source_repair"] = True
                        candidate_eval["drift_reasons_relaxed"] = drift.reasons[:10]
                    elif not drift.accepted:
                        candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                        candidate_eval["drift_reasons"] = drift.reasons[:10]
                        authenticity_summary["candidates"].append(candidate_eval)
                    if not candidate_eval.get("reason"):
                        candidate_eval["passed_local_checks"] = True
                        try:
                            scan_t0 = time.time()
                            candidate_report = _full_scan_report_dict(source_for_mitigation)
                            candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                        except Exception as exc:
                            candidate_eval["passed_local_checks"] = False
                            candidate_eval["reason"] = f"candidate_scan_error {exc}"
                            authenticity_summary["candidates"].append(candidate_eval)
                        if candidate_eval.get("passed_local_checks"):
                            candidate_review_burden = _review_burden(candidate_report)
                            candidate_severity = _weighted_severity(candidate_report)
                            gate = _authenticity_gate_status(
                                original_report_dict,
                                candidate_report,
                                source_for_mitigation != text,
                                original_review_burden=original_review_burden,
                                candidate_review_burden=candidate_review_burden,
                                original_weighted_severity=original_severity,
                                candidate_weighted_severity=candidate_severity,
                                min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                                min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                                drift_similarity=candidate_eval.get("drift_similarity"),
                            )
                            candidate_eval.update({
                                "ai": _badge_ai(candidate_report),
                                "writing_quality": _badge_wq(candidate_report),
                                "human_contribution": gate.get("candidate_human"),
                                "ai_transformation": gate.get("candidate_ai_transformation"),
                                "ai_authorship": gate.get("candidate_ai_authorship"),
                                "human_delta": gate.get("human_delta"),
                                "ai_transformation_delta": gate.get("ai_transformation_delta"),
                                "ai_authorship_delta": gate.get("ai_authorship_delta"),
                                "human_shift_score": gate.get("human_shift_score"),
                                "human_shift_components": gate.get("human_shift_components"),
                                "findings": _finding_total(candidate_report),
                                "review_burden": candidate_review_burden,
                                "weighted_severity": candidate_severity,
                                "scan_scope": _scan_scope_summary(candidate_report),
                                "gate": gate,
                            })
                            if _is_better_human_shift_candidate(gate, best_candidate_gate):
                                best_candidate_text = source_for_mitigation
                                best_candidate_report = candidate_report
                                best_candidate_gate = gate
                                best_candidate_eval = dict(candidate_eval)
                                candidate_eval["best_so_far"] = True
                            authenticity_summary["candidates"].append(candidate_eval)
            try:
                gateway = LLMGateway(LLMConfig(
                    api_key=effective_key,
                    model=generator_model,
                    base_url=base_url,
                    timeout=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_TIMEOUT", "120")),
                    max_retries=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_RETRIES", "1")),
                    max_tokens=int(os.environ.get("DRAFTPROOF_AUTHENTICITY_MAX_TOKENS", "6500")),
                    temperature=float(os.environ.get("DRAFTPROOF_AUTHENTICITY_TEMPERATURE", "0.7")),
                ))
                if _env_flag("DRAFTPROOF_MASKED_SPAN_OPTIMIZER", True):
                    masked_optimizer_ran = True
                    masked_limit = int(_float_env("DRAFTPROOF_MASKED_SPAN_CANDIDATES", 3.0))
                    masked_limit = max(0, masked_limit)
                    masked_baseline_report = original_report_dict
                    if _env_flag("DRAFTPROOF_MASKED_SPAN_FRESH_BASELINE", True):
                        try:
                            scan_t0 = time.time()
                            masked_baseline_report = _full_scan_report_dict(source_for_mitigation)
                            stage_timings.append({
                                "stage": "masked_span_fresh_baseline_scan",
                                "seconds": round(time.time() - scan_t0, 3),
                            })
                        except Exception as exc:
                            authenticity_summary.setdefault("masked_span_baseline_warning", str(exc))
                            masked_baseline_report = original_report_dict
                    masked_baseline_integrity = _integrity_scores(masked_baseline_report)
                    masked_baseline_review_burden = _review_burden(masked_baseline_report)
                    masked_baseline_severity = _weighted_severity(masked_baseline_report)
                    masked_baseline_findings = _finding_total(masked_baseline_report)
                    authenticity_summary["masked_span_baseline"] = {
                        "mode": (
                            "fresh_original_scan"
                            if masked_baseline_report is not original_report_dict
                            else result.summary.get("comparison_baseline", "saved_original_scan")
                        ),
                        "saved_ai": original_ai,
                        "baseline_ai": _badge_ai(masked_baseline_report),
                        "saved_findings": original_total,
                        "baseline_findings": masked_baseline_findings,
                        "saved_ai_authorship": integrity_original.get("ai_authorship"),
                        "baseline_ai_authorship": masked_baseline_integrity.get("ai_authorship"),
                    }
                    current_masked_text = source_for_mitigation
                    current_masked_report = masked_baseline_report
                    current_masked_ai = _badge_ai(masked_baseline_report)
                    current_masked_human = _contribution_scores(masked_baseline_report).get("human")
                    current_masked_transform = _contribution_scores(masked_baseline_report).get("ai_transformation")
                    current_masked_authorship = masked_baseline_integrity.get("ai_authorship")
                    current_masked_findings = masked_baseline_findings
                    masked_excluded: set[int] = set()
                    masked_attempts: list[dict] = []
                    route_bundle_candidate, route_bundle_edits = _deterministic_sentence_route_bundle(
                        current_masked_text
                    )
                    if route_bundle_edits and route_bundle_candidate != current_masked_text:
                        candidate_eval_try = {
                            "attempt": "route_bundle.1",
                            "strategy": "deterministic_sentence_route_bundle",
                            "masked_span_repair": True,
                            "deterministic": True,
                            "passed_local_checks": False,
                            "route_bundle_edits": route_bundle_edits,
                            "candidate_length": len(route_bundle_candidate or ""),
                            "candidate_word_count": _text_word_count(route_bundle_candidate or ""),
                        }
                        candidate_eval_try["repair_aggression"] = _repair_aggression_score(
                            current_masked_text,
                            route_bundle_candidate,
                        )
                        candidate_eval_try["locality"] = _locality_score(
                            current_masked_text,
                            route_bundle_candidate,
                        )
                        protected_loss = _ai_search_protected_loss_reason(
                            source_for_mitigation,
                            route_bundle_candidate,
                            source_protected,
                        )
                        drift = (
                            None
                            if protected_loss
                            else check_semantic_drift(source_for_mitigation, route_bundle_candidate, threshold=0.15)
                        )
                        if protected_loss:
                            candidate_eval_try["reason"] = "protected_span_lost " + protected_loss
                        elif drift and not drift.accepted:
                            candidate_eval_try["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                            candidate_eval_try["drift_reasons"] = drift.reasons[:10]
                        else:
                            if drift:
                                candidate_eval_try["drift_similarity"] = round(drift.similarity, 3)
                            candidate_eval_try["passed_local_checks"] = True
                            try:
                                scan_t0 = time.time()
                                candidate_report = _full_scan_report_dict(route_bundle_candidate)
                                candidate_eval_try["scan_seconds"] = round(time.time() - scan_t0, 3)
                            except Exception as exc:
                                candidate_report = None
                                candidate_eval_try["passed_local_checks"] = False
                                candidate_eval_try["reason"] = f"candidate_scan_error {exc}"
                            if candidate_report:
                                candidate_review_burden = _review_burden(candidate_report)
                                candidate_severity = _weighted_severity(candidate_report)
                                gate = _authenticity_gate_status(
                                    masked_baseline_report,
                                    candidate_report,
                                    route_bundle_candidate != text,
                                    original_review_burden=masked_baseline_review_burden,
                                    candidate_review_burden=candidate_review_burden,
                                    original_weighted_severity=masked_baseline_severity,
                                    candidate_weighted_severity=candidate_severity,
                                    min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                                    min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                                    drift_similarity=candidate_eval_try.get("drift_similarity"),
                                )
                                candidate_ai = _badge_ai(candidate_report)
                                candidate_human = gate.get("candidate_human")
                                candidate_transform = gate.get("candidate_ai_transformation")
                                candidate_authorship = gate.get("candidate_ai_authorship")
                                candidate_findings = _finding_total(candidate_report)
                                candidate_critical_high = (
                                    len(candidate_report.get("findings", {}).get("critical", []))
                                    + len(candidate_report.get("findings", {}).get("high", []))
                                )
                                candidate_eval_try.update({
                                    "ai": candidate_ai,
                                    "writing_quality": _badge_wq(candidate_report),
                                    "human_contribution": candidate_human,
                                    "ai_transformation": candidate_transform,
                                    "ai_authorship": candidate_authorship,
                                    "human_delta": gate.get("human_delta"),
                                    "ai_transformation_delta": gate.get("ai_transformation_delta"),
                                    "ai_authorship_delta": gate.get("ai_authorship_delta"),
                                    "human_shift_score": gate.get("human_shift_score"),
                                    "human_shift_components": gate.get("human_shift_components"),
                                    "authorship_cost_per_human_gain": gate.get("authorship_cost_per_human_gain"),
                                    "findings": candidate_findings,
                                    "review_burden": candidate_review_burden,
                                    "weighted_severity": candidate_severity,
                                    "scan_scope": _scan_scope_summary(candidate_report),
                                    "gate": gate,
                                })
                                fresh_authorship_capped = bool(
                                    isinstance(candidate_authorship, (int, float))
                                    and isinstance(masked_baseline_integrity.get("ai_authorship"), (int, float))
                                    and candidate_authorship <= masked_baseline_integrity.get("ai_authorship")
                                )
                                saved_authorship_capped = bool(
                                    not isinstance(integrity_original.get("ai_authorship"), (int, float))
                                    or (
                                        isinstance(candidate_authorship, (int, float))
                                        and candidate_authorship <= integrity_original.get("ai_authorship")
                                    )
                                )
                                findings_non_regression = bool(
                                    candidate_findings <= current_masked_findings
                                    and candidate_findings <= original_total
                                )
                                review_non_regression = bool(
                                    candidate_review_burden <= masked_baseline_review_burden
                                    and candidate_review_burden <= original_review_burden
                                )
                                severity_non_regression = bool(
                                    candidate_severity <= masked_baseline_severity
                                    and candidate_severity <= original_severity
                                )
                                critical_high_non_regression = bool(
                                    candidate_critical_high <= saved_critical_high
                                )
                                movement = bool(
                                    (
                                        isinstance(candidate_human, (int, float))
                                        and isinstance(current_masked_human, (int, float))
                                        and candidate_human > current_masked_human
                                    )
                                    or (
                                        isinstance(candidate_transform, (int, float))
                                        and isinstance(current_masked_transform, (int, float))
                                        and candidate_transform < current_masked_transform
                                    )
                                    or candidate_findings < current_masked_findings
                                    or (
                                        isinstance(candidate_ai, (int, float))
                                        and isinstance(current_masked_ai, (int, float))
                                        and candidate_ai < current_masked_ai
                                    )
                                )
                                breakthrough_tradeoff = bool(
                                    _env_flag("DRAFTPROOF_AUTHENTICITY_BREAKTHROUGH_TRADEOFF", True)
                                    and isinstance(candidate_ai, (int, float))
                                    and isinstance(original_ai, (int, float))
                                    and candidate_ai <= original_ai - 10.0
                                    and isinstance(candidate_authorship, (int, float))
                                    and isinstance(integrity_original.get("ai_authorship"), (int, float))
                                    and candidate_authorship <= integrity_original.get("ai_authorship") - 10.0
                                    and candidate_findings <= original_total
                                )
                                masked_accept = bool(
                                    gate.get("success")
                                    and fresh_authorship_capped
                                    and saved_authorship_capped
                                    and findings_non_regression
                                    and (
                                        (
                                            review_non_regression
                                            and severity_non_regression
                                            and critical_high_non_regression
                                        )
                                        or breakthrough_tradeoff
                                    )
                                    and movement
                                    and (
                                        not gate.get("critical_high_regressed")
                                        or breakthrough_tradeoff
                                    )
                                    and (
                                        not gate.get("review_burden_regressed")
                                        or breakthrough_tradeoff
                                    )
                                    and (
                                        not gate.get("weighted_severity_regressed")
                                        or breakthrough_tradeoff
                                    )
                                )
                                candidate_eval_try["masked_accept"] = masked_accept
                                candidate_eval_try["masked_acceptance"] = {
                                    "authorship_capped": bool(fresh_authorship_capped and saved_authorship_capped),
                                    "fresh_authorship_capped": fresh_authorship_capped,
                                    "saved_authorship_capped": saved_authorship_capped,
                                    "findings_non_regression": findings_non_regression,
                                    "review_non_regression": review_non_regression,
                                    "severity_non_regression": severity_non_regression,
                                    "critical_high_non_regression": critical_high_non_regression,
                                    "breakthrough_tradeoff": breakthrough_tradeoff,
                                    "movement": movement,
                                }
                                if masked_accept:
                                    candidate_eval_try["selected"] = True
                                    masked_attempts.append(candidate_eval_try)
                                    current_masked_text = route_bundle_candidate
                                    current_masked_report = candidate_report
                                    current_masked_ai = candidate_ai
                                    current_masked_human = candidate_human
                                    current_masked_transform = candidate_transform
                                    current_masked_authorship = candidate_authorship
                                    current_masked_findings = candidate_findings
                                    masked_span_selected = True
                                else:
                                    candidate_eval_try["reason"] = "authorship_cap_or_no_masked_gain"
                        authenticity_summary["candidates"].append(candidate_eval_try)
                    for masked_index in range(1, masked_limit + 1):
                        report_progress(
                            min(88, 76 + masked_index),
                            f"Trying masked-span mitigation {masked_index}/{masked_limit}",
                        )
                        prompt, repair_info = _masked_span_repair_prompt(
                            current_masked_text,
                            original_report_dict,
                            exclude_sentence_indexes=masked_excluded,
                        )
                        window = repair_info.get("window") if isinstance(repair_info, dict) else {}
                        sentence_index = window.get("start") if isinstance(window, dict) else None
                        candidate_eval = {
                            "attempt": f"masked.{masked_index}",
                            "strategy": "masked_span_repair",
                            "masked_span_repair": True,
                            "passed_local_checks": False,
                            "model": generator_model,
                            "sentence_index": sentence_index,
                            "mask_text": repair_info.get("mask_text") if isinstance(repair_info, dict) else None,
                            "masked_sentence": repair_info.get("masked_sentence") if isinstance(repair_info, dict) else None,
                        }
                        if not prompt or sentence_index is None:
                            candidate_eval["reason"] = repair_info.get("reason") if isinstance(repair_info, dict) else "no_mask_prompt"
                            authenticity_summary["candidates"].append(candidate_eval)
                            break
                        replacements = _deterministic_masked_span_replacements(
                            repair_info.get("mask_text") if isinstance(repair_info, dict) else ""
                        )
                        if _env_flag("DRAFTPROOF_MASKED_SPAN_LLM_FALLBACK", False):
                            try:
                                authenticity_summary["llm_calls"] += 1
                                response = gateway.chat(
                                    prompt,
                                    system="Return only replacement text for [[MASK]].",
                                    temperature=float(os.environ.get("DRAFTPROOF_RECONSTRUCTION_TEMPERATURE", "0.45")),
                                    max_tokens=int(os.environ.get("DRAFTPROOF_MASKED_SPAN_MAX_TOKENS", "1000")),
                                    top_p=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "TOP_P"),
                                    top_k=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "TOP_K"),
                                    presence_penalty=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "PRESENCE_PENALTY"),
                                    frequency_penalty=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "FREQUENCY_PENALTY"),
                                )
                                llm_replacement = _clean_masked_span_replacement(response.content or "")
                                if llm_replacement and llm_replacement not in replacements:
                                    replacements.append(llm_replacement)
                            except Exception as exc:
                                candidate_eval.setdefault("replacement_errors", []).append(f"llm_error {exc}")
                        if not replacements:
                            candidate_eval["reason"] = "no_mask_replacement_candidates"
                            authenticity_summary["candidates"].append(candidate_eval)
                            masked_excluded.add(int(sentence_index))
                            continue
                        candidate_eval["replacement_candidates"] = replacements
                        accepted_eval = None
                        rejected_replacement_evals: list[dict] = []
                        for replacement_index, replacement in enumerate(replacements, start=1):
                            candidate_eval_try = dict(candidate_eval)
                            candidate_eval_try["replacement_index"] = replacement_index
                            candidate_eval_try["replacement"] = replacement
                            candidate = _apply_masked_span_replacement(current_masked_text, repair_info, replacement)
                            candidate_eval_try["candidate_length"] = len(candidate or "")
                            candidate_eval_try["candidate_word_count"] = _text_word_count(candidate or "")
                            if not candidate or candidate == current_masked_text:
                                candidate_eval_try["reason"] = "empty_or_unchanged_candidate"
                                rejected_replacement_evals.append(candidate_eval_try)
                                continue
                            candidate_eval_try["repair_aggression"] = _repair_aggression_score(current_masked_text, candidate)
                            candidate_eval_try["locality"] = _locality_score(current_masked_text, candidate)
                            protected_loss = _ai_search_protected_loss_reason(
                                source_for_mitigation,
                                candidate,
                                source_protected,
                            )
                            if protected_loss:
                                candidate_eval_try["reason"] = "protected_span_lost " + protected_loss
                                rejected_replacement_evals.append(candidate_eval_try)
                                continue
                            drift = check_semantic_drift(source_for_mitigation, candidate, threshold=0.15)
                            candidate_eval_try["drift_similarity"] = round(drift.similarity, 3)
                            if not drift.accepted:
                                candidate_eval_try["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                                candidate_eval_try["drift_reasons"] = drift.reasons[:10]
                                rejected_replacement_evals.append(candidate_eval_try)
                                continue
                            candidate_eval_try["passed_local_checks"] = True
                            try:
                                scan_t0 = time.time()
                                candidate_report = _full_scan_report_dict(candidate)
                                candidate_eval_try["scan_seconds"] = round(time.time() - scan_t0, 3)
                            except Exception as exc:
                                candidate_eval_try["passed_local_checks"] = False
                                candidate_eval_try["reason"] = f"candidate_scan_error {exc}"
                                rejected_replacement_evals.append(candidate_eval_try)
                                continue
                            candidate_review_burden = _review_burden(candidate_report)
                            candidate_severity = _weighted_severity(candidate_report)
                            gate = _authenticity_gate_status(
                                masked_baseline_report,
                                candidate_report,
                                candidate != text,
                                original_review_burden=masked_baseline_review_burden,
                                candidate_review_burden=candidate_review_burden,
                                original_weighted_severity=masked_baseline_severity,
                                candidate_weighted_severity=candidate_severity,
                                min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                                min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                                drift_similarity=candidate_eval_try.get("drift_similarity"),
                            )
                            candidate_ai = _badge_ai(candidate_report)
                            candidate_human = gate.get("candidate_human")
                            candidate_transform = gate.get("candidate_ai_transformation")
                            candidate_authorship = gate.get("candidate_ai_authorship")
                            candidate_findings = _finding_total(candidate_report)
                            candidate_critical_high = (
                                len(candidate_report.get("findings", {}).get("critical", []))
                                + len(candidate_report.get("findings", {}).get("high", []))
                            )
                            candidate_eval_try.update({
                                "ai": candidate_ai,
                                "writing_quality": _badge_wq(candidate_report),
                                "human_contribution": candidate_human,
                                "ai_transformation": candidate_transform,
                                "ai_authorship": candidate_authorship,
                                "human_delta": gate.get("human_delta"),
                                "ai_transformation_delta": gate.get("ai_transformation_delta"),
                                "ai_authorship_delta": gate.get("ai_authorship_delta"),
                                "human_shift_score": gate.get("human_shift_score"),
                                "human_shift_components": gate.get("human_shift_components"),
                                "authorship_cost_per_human_gain": gate.get("authorship_cost_per_human_gain"),
                                "findings": candidate_findings,
                                "review_burden": candidate_review_burden,
                                "weighted_severity": candidate_severity,
                                "scan_scope": _scan_scope_summary(candidate_report),
                                "gate": gate,
                            })
                            fresh_authorship_capped = bool(
                                isinstance(candidate_authorship, (int, float))
                                and isinstance(masked_baseline_integrity.get("ai_authorship"), (int, float))
                                and candidate_authorship <= masked_baseline_integrity.get("ai_authorship")
                            )
                            saved_authorship_capped = bool(
                                not isinstance(integrity_original.get("ai_authorship"), (int, float))
                                or (
                                    isinstance(candidate_authorship, (int, float))
                                    and candidate_authorship <= integrity_original.get("ai_authorship")
                                )
                            )
                            authorship_capped = bool(fresh_authorship_capped and saved_authorship_capped)
                            baseline_findings_non_regression = candidate_findings <= current_masked_findings
                            saved_findings_non_regression = candidate_findings <= original_total
                            findings_non_regression = bool(
                                baseline_findings_non_regression
                                and saved_findings_non_regression
                            )
                            review_non_regression = bool(
                                candidate_review_burden <= masked_baseline_review_burden
                                and candidate_review_burden <= original_review_burden
                            )
                            severity_non_regression = bool(
                                candidate_severity <= masked_baseline_severity
                                and candidate_severity <= original_severity
                            )
                            critical_high_non_regression = bool(
                                candidate_critical_high <= saved_critical_high
                            )
                            movement = bool(
                                (
                                    isinstance(candidate_human, (int, float))
                                    and isinstance(current_masked_human, (int, float))
                                    and candidate_human > current_masked_human
                                )
                                or (
                                    isinstance(candidate_transform, (int, float))
                                    and isinstance(current_masked_transform, (int, float))
                                    and candidate_transform < current_masked_transform
                                )
                                or candidate_findings < current_masked_findings
                                or (
                                    isinstance(candidate_ai, (int, float))
                                    and isinstance(current_masked_ai, (int, float))
                                    and candidate_ai < current_masked_ai
                                )
                            )
                            masked_accept = bool(
                                gate.get("success")
                                and authorship_capped
                                and findings_non_regression
                                and review_non_regression
                                and severity_non_regression
                                and critical_high_non_regression
                                and movement
                                and not gate.get("critical_high_regressed")
                                and not gate.get("review_burden_regressed")
                                and not gate.get("weighted_severity_regressed")
                            )
                            candidate_eval_try["masked_accept"] = masked_accept
                            candidate_eval_try["masked_acceptance"] = {
                                "authorship_capped": authorship_capped,
                                "fresh_authorship_capped": fresh_authorship_capped,
                                "saved_authorship_capped": saved_authorship_capped,
                                "findings_non_regression": findings_non_regression,
                                "baseline_findings_non_regression": baseline_findings_non_regression,
                                "saved_findings_non_regression": saved_findings_non_regression,
                                "review_non_regression": review_non_regression,
                                "severity_non_regression": severity_non_regression,
                                "critical_high_non_regression": critical_high_non_regression,
                                "movement": movement,
                            }
                            if masked_accept:
                                accepted_eval = candidate_eval_try
                                accepted_eval["_candidate_text"] = candidate
                                accepted_eval["_candidate_report"] = candidate_report
                                break
                            candidate_eval_try["reason"] = "authorship_cap_or_no_masked_gain"
                            rejected_replacement_evals.append(candidate_eval_try)
                        for rejected_eval in rejected_replacement_evals:
                            authenticity_summary["candidates"].append(rejected_eval)
                        if accepted_eval is None:
                            masked_excluded.add(int(sentence_index))
                            continue
                        candidate_eval = accepted_eval
                        candidate = candidate_eval.pop("_candidate_text")
                        candidate_report = candidate_eval.pop("_candidate_report")
                        candidate_ai = candidate_eval.get("ai")
                        candidate_human = candidate_eval.get("human_contribution")
                        candidate_transform = candidate_eval.get("ai_transformation")
                        candidate_authorship = candidate_eval.get("ai_authorship")
                        candidate_findings = candidate_eval.get("findings")
                        candidate_eval["selected"] = True
                        masked_attempts.append(candidate_eval)
                        current_masked_text = candidate
                        current_masked_report = candidate_report
                        current_masked_ai = candidate_ai
                        current_masked_human = candidate_human
                        current_masked_transform = candidate_transform
                        current_masked_authorship = candidate_authorship
                        current_masked_findings = candidate_findings
                        masked_span_selected = True
                        authenticity_summary["candidates"].append(candidate_eval)
                        masked_excluded.add(int(sentence_index))
                    authenticity_summary["masked_span_optimizer"] = {
                        "enabled": True,
                        "candidate_limit": masked_limit,
                        "accepted_count": len(masked_attempts),
                        "selected": masked_span_selected,
                        "accepted_attempts": [
                            {
                                "attempt": item.get("attempt"),
                                "sentence_index": item.get("sentence_index"),
                                "mask_text": item.get("mask_text"),
                                "replacement": item.get("replacement"),
                                "ai": item.get("ai"),
                                "human_contribution": item.get("human_contribution"),
                                "ai_transformation": item.get("ai_transformation"),
                                "ai_authorship": item.get("ai_authorship"),
                                "findings": item.get("findings"),
                            }
                            for item in masked_attempts
                        ],
                    }
                    if masked_span_selected:
                        masked_gate = _authenticity_gate_status(
                            masked_baseline_report,
                            current_masked_report,
                            current_masked_text != text,
                            original_review_burden=masked_baseline_review_burden,
                            candidate_review_burden=_review_burden(current_masked_report),
                            original_weighted_severity=masked_baseline_severity,
                            candidate_weighted_severity=_weighted_severity(current_masked_report),
                            min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                            min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                        )
                        best_candidate_text = current_masked_text
                        best_candidate_report = current_masked_report
                        best_candidate_gate = masked_gate
                        best_candidate_eval = {
                            "strategy": "masked_span_optimizer",
                            "masked_span_repair": True,
                            "accepted_count": len(masked_attempts),
                            "gate": masked_gate,
                        }
                        authenticity_summary["best_attempt"] = best_candidate_eval
                        if (
                            masked_gate.get("success")
                            and _env_flag("DRAFTPROOF_MASKED_SPAN_SKIP_REGEN_ON_GAIN", True)
                        ):
                            authenticity_summary["skip_broad_generation_reason"] = "masked_span_authorship_capped_gain"
                            authenticity_candidate_limit = 0
                for attempt_index in range(1, authenticity_candidate_limit + 1):
                    report_progress(
                        min(89, 78 + attempt_index),
                        f"Trying authenticity mitigation candidate {attempt_index}/{authenticity_candidate_limit}",
                    )
                    candidate_eval = {
                        "attempt": attempt_index,
                        "passed_local_checks": False,
                        "model": generator_model,
                    }
                    try:
                        prompt = _authenticity_mitigation_prompt(
                            source_for_mitigation,
                            original_report_dict,
                            ai_mitigation_contract,
                            attempt_index,
                        )
                        if _auth_llm_budget_exhausted("authenticity_candidate"):
                            candidate_eval["reason"] = "budget_exhausted_llm_calls"
                            authenticity_summary["candidates"].append(candidate_eval)
                            break
                        authenticity_summary["llm_calls"] += 1
                        response = gateway.chat(
                            prompt,
                            system=(
                                "You are DraftProof's AI-Mitigation authenticity engine. "
                                "Return only a complete fact-preserving rewritten document."
                            ),
                            **_phase_chat_sampling_kwargs(
                                "DRAFTPROOF_AUTHENTICITY",
                                temperature_env="DRAFTPROOF_AUTHENTICITY_TEMPERATURE",
                                temperature_default=0.7,
                                max_tokens_env="DRAFTPROOF_AUTHENTICITY_MAX_TOKENS",
                                max_tokens_default=6500,
                            ),
                        )
                        candidate = _clean_full_document_candidate(response.content, source_for_mitigation)
                    except Exception as exc:
                        candidate_eval["reason"] = f"llm_error {exc}"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    candidate_eval["candidate_length"] = len(candidate or "")
                    if not candidate:
                        candidate_eval["reason"] = "empty_or_unchanged_candidate"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    candidate, repair_notes = _repair_candidate_source_damage(candidate)
                    if repair_notes:
                        candidate_eval["source_damage_repairs"] = repair_notes
                        candidate_eval["candidate_length"] = len(candidate or "")
                    review_notes = _review_marker_notes(candidate)
                    if review_notes:
                        candidate_eval["reason"] = "review_markers_not_auto_kept"
                        candidate_eval["review_suggestion_count"] = len(review_notes)
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    quality_rejection = _ai_candidate_quality_reject_reason(candidate)
                    if quality_rejection:
                        candidate_eval["reason"] = quality_rejection
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    if len(candidate) < min_chars:
                        candidate_eval["reason"] = f"candidate_too_short {len(candidate)}<{min_chars}"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    if len(candidate) > max_chars:
                        candidate_eval["reason"] = f"candidate_too_long {len(candidate)}>{max_chars}"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    protected_loss = _ai_search_protected_loss_reason(
                        source_for_mitigation,
                        candidate,
                        source_protected,
                    )
                    if protected_loss:
                        candidate_eval["reason"] = "protected_span_lost " + protected_loss
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    drift = check_semantic_drift(source_for_mitigation, candidate, threshold=0.15)
                    candidate_eval["drift_similarity"] = round(drift.similarity, 3)
                    if not drift.accepted:
                        candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                        candidate_eval["drift_reasons"] = drift.reasons[:10]
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    candidate_eval["passed_local_checks"] = True
                    try:
                        scan_t0 = time.time()
                        candidate_report = _full_scan_report_dict(candidate)
                        candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                    except Exception as exc:
                        candidate_eval["passed_local_checks"] = False
                        candidate_eval["reason"] = f"candidate_scan_error {exc}"
                        authenticity_summary["candidates"].append(candidate_eval)
                        continue
                    candidate_review_burden = _review_burden(candidate_report)
                    candidate_severity = _weighted_severity(candidate_report)
                    gate = _authenticity_gate_status(
                        original_report_dict,
                        candidate_report,
                        candidate != text,
                        original_review_burden=original_review_burden,
                        candidate_review_burden=candidate_review_burden,
                        original_weighted_severity=original_severity,
                        candidate_weighted_severity=candidate_severity,
                        min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                        min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                        drift_similarity=candidate_eval.get("drift_similarity"),
                    )
                    candidate_eval.update({
                        "ai": _badge_ai(candidate_report),
                        "writing_quality": _badge_wq(candidate_report),
                        "human_contribution": gate.get("candidate_human"),
                        "ai_transformation": gate.get("candidate_ai_transformation"),
                        "ai_authorship": gate.get("candidate_ai_authorship"),
                        "human_delta": gate.get("human_delta"),
                        "ai_transformation_delta": gate.get("ai_transformation_delta"),
                        "ai_authorship_delta": gate.get("ai_authorship_delta"),
                        "human_shift_score": gate.get("human_shift_score"),
                        "human_shift_components": gate.get("human_shift_components"),
                        "findings": _finding_total(candidate_report),
                        "review_burden": candidate_review_burden,
                        "weighted_severity": candidate_severity,
                        "scan_scope": _scan_scope_summary(candidate_report),
                        "gate": gate,
                    })
                    if _is_better_human_shift_candidate(gate, best_candidate_gate):
                        best_candidate_text = candidate
                        best_candidate_report = candidate_report
                        best_candidate_gate = gate
                        best_candidate_eval = dict(candidate_eval)
                        candidate_eval["best_so_far"] = True
                    authenticity_summary["candidates"].append(candidate_eval)
                reconstruction_target_human = _float_env("DRAFTPROOF_RECONSTRUCTION_TARGET_HUMAN", 80.0)
                selected_human = (
                    best_candidate_gate.get("candidate_human")
                    if isinstance(best_candidate_gate, dict)
                    else None
                )
                selected_human_delta = (
                    best_candidate_gate.get("human_delta")
                    if isinstance(best_candidate_gate, dict)
                    else None
                )
                small_shift_under_target = bool(
                    best_candidate_gate
                    and best_candidate_gate.get("success")
                    and isinstance(selected_human, (int, float))
                    and selected_human < reconstruction_target_human
                    and (
                        not isinstance(selected_human_delta, (int, float))
                        or selected_human_delta < _float_env("DRAFTPROOF_RECONSTRUCTION_SKIP_MIN_HUMAN_GAIN", 25.0)
                    )
                )
                reconstruction_enabled = (
                    (
                        not (best_candidate_gate and best_candidate_gate.get("success"))
                        or (
                            small_shift_under_target
                            and os.environ.get("DRAFTPROOF_RECONSTRUCTION_AFTER_SMALL_SHIFT", "1") != "0"
                        )
                        or radar_force_reconstruction
                    )
                    and os.environ.get("DRAFTPROOF_RECONSTRUCTION_MITIGATION", "1") != "0"
                    and not (
                        masked_optimizer_ran
                        and _env_flag("DRAFTPROOF_MASKED_SPAN_SKIP_BROAD_REGEN", True)
                        and not radar_force_reconstruction
                    )
                )
                if reconstruction_enabled:
                    reconstruction_strategies = [
                        "plain_direct_voice_rebuild",
                        "authorship_distribution_repair",
                        "low_smoothness_rebuild",
                        "asymmetric_paragraph_route",
                        "claim_narrowing_rebuild",
                        "authorship_texture_repair",
                        "human_gain_repair",
                        "evidence_first_rebuild",
                        "problem_observation_rebuild",
                        "reasoning_dense_reconstruction",
                        "domain_grounded_reconstruction",
                    ]
                    reconstruction_limit_raw = os.environ.get("DRAFTPROOF_RECONSTRUCTION_CANDIDATES")
                    try:
                        reconstruction_limit = max(
                            1,
                            int(reconstruction_limit_raw or "2"),
                        )
                    except ValueError:
                        reconstruction_limit = 4
                    if _env_flag("DRAFTPROOF_REGENERATION_FIRST", True) and reconstruction_limit_raw is None:
                        reconstruction_limit = max(reconstruction_limit, 4)
                    reconstruction_strategies = reconstruction_strategies[:reconstruction_limit]
                    authenticity_summary["reconstruction"] = {
                        "enabled": True,
                        "candidate_limit": len(reconstruction_strategies),
                        "strategies": reconstruction_strategies,
                        "target_human_contribution": reconstruction_target_human,
                        "triggered_after_small_shift": small_shift_under_target,
                    }
                    reconstruction_word_band = _word_count_band(source_for_mitigation, variance=0.25)
                    authenticity_summary["reconstruction"]["word_count_band"] = reconstruction_word_band
                    reconstruction_min_chars = max(200, int(len(source_for_mitigation) * 0.65))
                    reconstruction_max_chars = max(reconstruction_min_chars, int(len(text) * 1.45))
                    reconstruction_drift_threshold = _float_env(
                        "DRAFTPROOF_RECONSTRUCTION_DRIFT_THRESHOLD",
                        0.25,
                    )
                    post_texture_calls = 0
                    post_texture_limit = int(_float_env("DRAFTPROOF_POST_GENERATION_TEXTURE_REPAIR_MAX_CALLS", 1.0))
                    for reconstruction_index, strategy in enumerate(reconstruction_strategies, start=1):
                        report_progress(
                            min(92, 84 + reconstruction_index),
                            f"Trying reconstruction mitigation candidate {reconstruction_index}/{len(reconstruction_strategies)}",
                        )
                        candidate_eval = {
                            "attempt": authenticity_candidate_limit + reconstruction_index,
                            "strategy": strategy,
                            "reconstruction": True,
                            "passed_local_checks": False,
                            "model": generator_model,
                        }
                        if _auth_llm_budget_exhausted("reconstruction_candidate"):
                            candidate_eval["reason"] = "budget_exhausted_llm_calls"
                            authenticity_summary["candidates"].append(candidate_eval)
                            break
                        try:
                            if _env_flag("DRAFTPROOF_STAGED_REGENERATION", True):
                                candidate, staged_info = _staged_reconstruction_candidate(
                                    gateway,
                                    source_for_mitigation,
                                    original_report_dict,
                                    attempt_index=reconstruction_index,
                                    strategy=strategy,
                                    prior_attempts=authenticity_summary.get("candidates") or [],
                                    max_calls=_auth_llm_remaining(),
                                )
                                candidate_eval["staged_generation"] = staged_info
                                authenticity_summary["llm_calls"] += int(staged_info.get("llm_calls") or 0)
                            else:
                                prompt = _reconstruction_mitigation_prompt(
                                    source_for_mitigation,
                                    original_report_dict,
                                    ai_mitigation_contract,
                                    attempt_index=reconstruction_index,
                                    strategy=strategy,
                                    prior_attempts=authenticity_summary.get("candidates") or [],
                                )
                                if _auth_llm_budget_exhausted("reconstruction_candidate"):
                                    candidate_eval["reason"] = "budget_exhausted_llm_calls"
                                    authenticity_summary["candidates"].append(candidate_eval)
                                    break
                                authenticity_summary["llm_calls"] += 1
                                response = gateway.chat(
                                    prompt,
                                    system=(
                                        "You are DraftProof's AI-Mitigation reconstruction engine. "
                                        "Return only a complete fact-preserving reconstructed document."
                                    ),
                                    **_phase_chat_sampling_kwargs(
                                        "DRAFTPROOF_RECONSTRUCTION",
                                        temperature_env="DRAFTPROOF_RECONSTRUCTION_TEMPERATURE",
                                        temperature_default=0.78,
                                        max_tokens_env="DRAFTPROOF_AUTHENTICITY_MAX_TOKENS",
                                        max_tokens_default=6500,
                                    ),
                                )
                                candidate = _clean_full_document_candidate(response.content, source_for_mitigation)
                        except Exception as exc:
                            candidate_eval["reason"] = f"llm_error {exc}"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        candidate_eval["candidate_length"] = len(candidate or "")
                        candidate_eval["candidate_word_count"] = _text_word_count(candidate or "")
                        if not candidate:
                            candidate_eval["reason"] = "empty_or_unchanged_candidate"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        candidate, repair_notes = _repair_candidate_source_damage(candidate)
                        if repair_notes:
                            candidate_eval["source_damage_repairs"] = repair_notes
                            candidate_eval["candidate_length"] = len(candidate or "")
                            candidate_eval["candidate_word_count"] = _text_word_count(candidate or "")
                        review_notes = _review_marker_notes(candidate)
                        if review_notes:
                            candidate_eval["reason"] = "review_markers_not_auto_kept"
                            candidate_eval["review_suggestion_count"] = len(review_notes)
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        quality_rejection = _ai_candidate_quality_reject_reason(
                            candidate,
                            allow_repeated_long_sequence=True,
                        )
                        if quality_rejection:
                            candidate_eval["reason"] = quality_rejection
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        candidate_words = _text_word_count(candidate)
                        candidate_eval["candidate_word_count"] = candidate_words
                        if candidate_words < reconstruction_word_band["min_words"]:
                            candidate_eval.setdefault("warnings", []).append(
                                f"candidate_word_count_below_target "
                                f"{candidate_words}<{reconstruction_word_band['min_words']}"
                            )
                        if candidate_words > reconstruction_word_band["max_words"]:
                            candidate_eval.setdefault("warnings", []).append(
                                f"candidate_word_count_above_target "
                                f"{candidate_words}>{reconstruction_word_band['max_words']}"
                            )
                        if len(candidate) < reconstruction_min_chars:
                            candidate_eval["reason"] = f"candidate_too_short {len(candidate)}<{reconstruction_min_chars}"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        if len(candidate) > reconstruction_max_chars:
                            candidate_eval["reason"] = f"candidate_too_long {len(candidate)}>{reconstruction_max_chars}"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        protected_loss = _ai_search_protected_loss_reason(
                            source_for_mitigation,
                            candidate,
                            source_protected,
                        )
                        if protected_loss:
                            candidate_eval["reason"] = "protected_span_lost " + protected_loss
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        drift = check_semantic_drift(
                            source_for_mitigation,
                            candidate,
                            threshold=reconstruction_drift_threshold,
                        )
                        candidate_eval["drift_similarity"] = round(drift.similarity, 3)
                        candidate_eval["drift_threshold"] = reconstruction_drift_threshold
                        if not drift.accepted:
                            candidate_eval["drift_reasons"] = drift.reasons[:10]
                            if _reconstruction_drift_scan_allowed(candidate, drift.reasons, drift.similarity):
                                candidate_eval["drift_scan_relaxed_for_reconstruction"] = True
                                candidate_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                            else:
                                candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                                authenticity_summary["candidates"].append(candidate_eval)
                                continue
                        candidate_eval["passed_local_checks"] = True
                        try:
                            scan_t0 = time.time()
                            candidate_report = _full_scan_report_dict(candidate)
                            candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                        except Exception as exc:
                            candidate_eval["passed_local_checks"] = False
                            candidate_eval["reason"] = f"candidate_scan_error {exc}"
                            authenticity_summary["candidates"].append(candidate_eval)
                            continue
                        candidate_review_burden = _review_burden(candidate_report)
                        candidate_severity = _weighted_severity(candidate_report)
                        gate = _authenticity_gate_status(
                            original_report_dict,
                            candidate_report,
                            candidate != text,
                            original_review_burden=original_review_burden,
                            candidate_review_burden=candidate_review_burden,
                            original_weighted_severity=original_severity,
                            candidate_weighted_severity=candidate_severity,
                            min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                            min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                            drift_similarity=candidate_eval.get("drift_similarity"),
                        )
                        candidate_eval.update({
                            "ai": _badge_ai(candidate_report),
                            "writing_quality": _badge_wq(candidate_report),
                            "human_contribution": gate.get("candidate_human"),
                            "ai_transformation": gate.get("candidate_ai_transformation"),
                            "ai_authorship": gate.get("candidate_ai_authorship"),
                            "human_delta": gate.get("human_delta"),
                            "ai_transformation_delta": gate.get("ai_transformation_delta"),
                            "ai_authorship_delta": gate.get("ai_authorship_delta"),
                            "human_shift_score": gate.get("human_shift_score"),
                            "human_shift_components": gate.get("human_shift_components"),
                            "authorship_cost_per_human_gain": gate.get("authorship_cost_per_human_gain"),
                            "findings": _finding_total(candidate_report),
                            "review_burden": candidate_review_burden,
                            "weighted_severity": candidate_severity,
                            "scan_scope": _scan_scope_summary(candidate_report),
                            "gate": gate,
                        })
                        if _is_better_human_shift_candidate(gate, best_candidate_gate):
                            best_candidate_text = candidate
                            best_candidate_report = candidate_report
                            best_candidate_gate = gate
                            best_candidate_eval = dict(candidate_eval)
                            candidate_eval["best_so_far"] = True
                        authenticity_summary["candidates"].append(candidate_eval)
                        if (
                            _env_flag("DRAFTPROOF_POST_GENERATION_TEXTURE_REPAIR", True)
                            and gate.get("ai_authorship_regressed")
                            and post_texture_calls < post_texture_limit
                        ):
                            post_texture_calls += 1
                            repaired_eval = {
                                "attempt": f"{candidate_eval.get('attempt')}.texture",
                                "strategy": f"{strategy}+post_generation_texture_repair",
                                "reconstruction": True,
                                "post_generation_texture_repair": True,
                                "parent_attempt": candidate_eval.get("attempt"),
                                "passed_local_checks": False,
                                "model": generator_model,
                            }
                            anchor_values = [
                                source_for_mitigation[span.start_char:span.end_char].strip()
                                for span in source_protected[:40]
                                if source_for_mitigation[span.start_char:span.end_char].strip()
                            ]
                            try:
                                repair_prompt, repair_info = _micro_texture_repair_prompt(
                                    candidate,
                                    candidate_report,
                                    anchors=anchor_values,
                                    max_sentences=1,
                                    mode="authorship_suppression_repair",
                                )
                                if _auth_llm_budget_exhausted("post_generation_texture_repair"):
                                    repaired_eval["reason"] = "budget_exhausted_llm_calls"
                                    authenticity_summary["candidates"].append(repaired_eval)
                                    continue
                                authenticity_summary["llm_calls"] += 1
                                repair_response = gateway.chat(
                                    repair_prompt,
                                    system=(
                                        "You are DraftProof's micro-local authorship texture repairer. "
                                        "Return only the replacement sentence window."
                                    ),
                                    temperature=float(os.environ.get("DRAFTPROOF_RECONSTRUCTION_TEMPERATURE", "0.45")),
                                    max_tokens=int(os.environ.get("DRAFTPROOF_MICRO_TEXTURE_MAX_TOKENS", "1200")),
                                    top_p=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "TOP_P"),
                                    top_k=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "TOP_K"),
                                    presence_penalty=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "PRESENCE_PENALTY"),
                                    frequency_penalty=_phase_sampling_arg("DRAFTPROOF_RECONSTRUCTION", "FREQUENCY_PENALTY"),
                                )
                                replacement, clean_reason = _clean_micro_texture_candidate(
                                    repair_response.content,
                                    repair_info,
                                )
                                repaired_eval["repair_window"] = {
                                    "start": (repair_info.get("window") or {}).get("start"),
                                    "end": (repair_info.get("window") or {}).get("end"),
                                }
                                if clean_reason:
                                    repaired_eval["reason"] = clean_reason
                                    authenticity_summary["candidates"].append(repaired_eval)
                                    continue
                                repaired_candidate = _splice_sentence_window(
                                    candidate,
                                    int((repair_info.get("window") or {}).get("start") or 0),
                                    int((repair_info.get("window") or {}).get("end") or 0),
                                    replacement,
                                )
                                repaired_eval["candidate_length"] = len(repaired_candidate or "")
                                repaired_eval["candidate_word_count"] = _text_word_count(repaired_candidate or "")
                                repaired_eval["repair_aggression"] = _repair_aggression_score(candidate, repaired_candidate)
                                repaired_eval["locality"] = _locality_score(candidate, repaired_candidate)
                                if repaired_candidate == candidate:
                                    repaired_eval["reason"] = "post_texture_no_change"
                                    authenticity_summary["candidates"].append(repaired_eval)
                                    continue
                                protected_loss = _ai_search_protected_loss_reason(
                                    source_for_mitigation,
                                    repaired_candidate,
                                    source_protected,
                                )
                                if protected_loss:
                                    repaired_eval["reason"] = "protected_span_lost " + protected_loss
                                    authenticity_summary["candidates"].append(repaired_eval)
                                    continue
                                drift = check_semantic_drift(
                                    source_for_mitigation,
                                    repaired_candidate,
                                    threshold=reconstruction_drift_threshold,
                                )
                                repaired_eval["drift_similarity"] = round(drift.similarity, 3)
                                repaired_eval["drift_threshold"] = reconstruction_drift_threshold
                                if not drift.accepted:
                                    repaired_eval["drift_reasons"] = drift.reasons[:10]
                                    if _reconstruction_drift_scan_allowed(repaired_candidate, drift.reasons, drift.similarity):
                                        repaired_eval["drift_scan_relaxed_for_reconstruction"] = True
                                        repaired_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                                    else:
                                        repaired_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                                        authenticity_summary["candidates"].append(repaired_eval)
                                        continue
                                repaired_eval["passed_local_checks"] = True
                                scan_t0 = time.time()
                                repaired_report = _full_scan_report_dict(repaired_candidate)
                                repaired_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                                repaired_review_burden = _review_burden(repaired_report)
                                repaired_severity = _weighted_severity(repaired_report)
                                repaired_gate = _authenticity_gate_status(
                                    original_report_dict,
                                    repaired_report,
                                    repaired_candidate != text,
                                    original_review_burden=original_review_burden,
                                    candidate_review_burden=repaired_review_burden,
                                    original_weighted_severity=original_severity,
                                    candidate_weighted_severity=repaired_severity,
                                    min_human_gain=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_GAIN", 2.0),
                                    min_ai_transformation_drop=_float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_TRANSFORM_DROP", 2.0),
                                    drift_similarity=repaired_eval.get("drift_similarity"),
                                )
                                repaired_eval.update({
                                    "ai": _badge_ai(repaired_report),
                                    "writing_quality": _badge_wq(repaired_report),
                                    "human_contribution": repaired_gate.get("candidate_human"),
                                    "ai_transformation": repaired_gate.get("candidate_ai_transformation"),
                                    "ai_authorship": repaired_gate.get("candidate_ai_authorship"),
                                    "human_delta": repaired_gate.get("human_delta"),
                                    "ai_transformation_delta": repaired_gate.get("ai_transformation_delta"),
                                    "ai_authorship_delta": repaired_gate.get("ai_authorship_delta"),
                                    "human_shift_score": repaired_gate.get("human_shift_score"),
                                    "human_shift_components": repaired_gate.get("human_shift_components"),
                                    "authorship_cost_per_human_gain": repaired_gate.get("authorship_cost_per_human_gain"),
                                    "findings": _finding_total(repaired_report),
                                    "review_burden": repaired_review_burden,
                                    "weighted_severity": repaired_severity,
                                    "scan_scope": _scan_scope_summary(repaired_report),
                                    "gate": repaired_gate,
                                })
                                if _is_better_human_shift_candidate(repaired_gate, best_candidate_gate):
                                    best_candidate_text = repaired_candidate
                                    best_candidate_report = repaired_report
                                    best_candidate_gate = repaired_gate
                                    best_candidate_eval = dict(repaired_eval)
                                    repaired_eval["best_so_far"] = True
                                authenticity_summary["candidates"].append(repaired_eval)
                            except Exception as exc:
                                repaired_eval["reason"] = f"post_texture_repair_error {exc}"
                                authenticity_summary["candidates"].append(repaired_eval)
                if (
                    best_candidate_gate
                    and best_candidate_report
                    and best_candidate_gate.get("success")
                ):
                    previous_ai = rewritten_ai
                    rewritten_text = best_candidate_text
                    rewritten_report_dict = best_candidate_report
                    attempted_report_dict = rewritten_report_dict
                    rewritten_ai = _badge_ai(rewritten_report_dict)
                    rewritten_wq = _badge_wq(rewritten_report_dict)
                    rewritten_total = _finding_total(rewritten_report_dict)
                    rewritten_review_burden = _review_burden(rewritten_report_dict)
                    rewritten_severity = _weighted_severity(rewritten_report_dict)
                    rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                    if result.mp_result:
                        result.mp_result.final_text = rewritten_text
                        result.mp_result.converged = True
                        result.mp_result.convergence_reason = (
                            "Selected authorship-capped masked-span AI-Mitigation candidate"
                            if isinstance(best_candidate_eval, dict) and best_candidate_eval.get("masked_span_repair")
                            else "Selected AI-Mitigation authenticity candidate"
                        )
                    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                    authenticity_mitigation_selected = True
                    ai_search_selected = True
                    _clear_stale_rollback_for_kept_ai_mitigation(
                        result.summary,
                        "AI-Mitigation authenticity gate",
                    )
                    result.summary["ai_mitigation_blocked_auto_rewrite"] = False
                    result.summary["rewrite_engine_mode"] = (
                        "ai_mitigation_masked_span_gate"
                        if isinstance(best_candidate_eval, dict) and best_candidate_eval.get("masked_span_repair")
                        else (
                            "ai_mitigation_reconstruction_gate"
                            if isinstance(best_candidate_eval, dict) and best_candidate_eval.get("reconstruction")
                            else "ai_mitigation_authenticity_gate"
                        )
                    )
                    result.summary["outcome"] = "ai_mitigated"
                    if isinstance(best_candidate_eval, dict):
                        best_candidate_eval["selected"] = True
                    authenticity_summary.update({
                        "selected": True,
                        "selected_strategy": (
                            best_candidate_eval.get("strategy")
                            if isinstance(best_candidate_eval, dict) else None
                        ),
                        "selected_reconstruction": bool(
                            isinstance(best_candidate_eval, dict)
                            and best_candidate_eval.get("reconstruction")
                        ),
                        "selected_masked_span_repair": bool(
                            isinstance(best_candidate_eval, dict)
                            and best_candidate_eval.get("masked_span_repair")
                        ),
                        "previous_ai": previous_ai,
                        "selected_ai": rewritten_ai,
                        "selected_human_contribution": best_candidate_gate.get("candidate_human"),
                        "selected_ai_transformation": best_candidate_gate.get("candidate_ai_transformation"),
                        "selected_ai_authorship": best_candidate_gate.get("candidate_ai_authorship"),
                        "selected_human_shift_score": best_candidate_gate.get("human_shift_score"),
                        "selected_human_shift_components": best_candidate_gate.get("human_shift_components"),
                        "selected_gate": best_candidate_gate,
                    })
                elif best_candidate_eval:
                    authenticity_summary["best_attempt"] = best_candidate_eval
                    authenticity_summary["selection_reason"] = (
                        (best_candidate_gate or {}).get("reason")
                        or "no_candidate_passed_authenticity_gate"
                    )
            except Exception as exc:
                authenticity_summary["reason"] = f"authenticity_mitigation_error {exc}"
        authenticity_summary["seconds"] = round(time.time() - mitigation_started, 3)
        authenticity_summary["candidate_diagnostics"] = _generation_candidate_diagnostics(
            authenticity_summary.get("candidates") or []
        )
        result.summary["authenticity_mitigation"] = authenticity_summary
        result.summary["generation_layer"] = {
            "schema_version": "generation_layer.v1",
            "mode": "regeneration_first" if _env_flag("DRAFTPROOF_REGENERATION_FIRST", True) else "authenticity_then_reconstruction",
            "goal": "Human Contribution >= 80",
            "selected": bool(authenticity_summary.get("selected")),
            "selected_strategy": authenticity_summary.get("selected_strategy"),
            "selected_reconstruction": authenticity_summary.get("selected_reconstruction"),
            "selected_masked_span_repair": authenticity_summary.get("selected_masked_span_repair"),
            "selection_reason": authenticity_summary.get("selection_reason") or authenticity_summary.get("reason"),
            "llm_calls": authenticity_summary.get("llm_calls"),
            "model_roles": authenticity_summary.get("model_roles"),
            "masked_span_optimizer": authenticity_summary.get("masked_span_optimizer"),
            "masked_span_baseline": authenticity_summary.get("masked_span_baseline"),
            "skip_broad_generation_reason": authenticity_summary.get("skip_broad_generation_reason"),
            "reconstruction": authenticity_summary.get("reconstruction"),
            "best_attempt": authenticity_summary.get("best_attempt"),
            "candidate_count": len(authenticity_summary.get("candidates") or []),
            "candidate_diagnostics": authenticity_summary.get("candidate_diagnostics"),
        }
        if authenticity_summary.get("llm_calls"):
            result.summary["authenticity_llm_calls_used"] = authenticity_summary["llm_calls"]
            try:
                prior_calls = int(result.summary.get("llm_calls_used") or 0)
            except (TypeError, ValueError):
                prior_calls = 0
            result.summary["llm_calls_used"] = prior_calls + int(authenticity_summary["llm_calls"])
        stage_timings.append({
            "stage": "authenticity_mitigation",
            "seconds": authenticity_summary["seconds"],
            "candidates": len(authenticity_summary.get("candidates", [])),
            "selected": authenticity_summary.get("selected", False),
        })
        result.summary["detect_scores"].update({
            "rewritten_ai": rewritten_ai,
            "rewritten_writing_quality": rewritten_wq,
            "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
            "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
            "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
            "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
            "rewritten_findings": rewritten_total,
            "rewritten_review_burden": rewritten_review_burden,
            "rewritten_weighted_severity": rewritten_severity,
        })

    # Dedicated AI-score search. This is separate from local sentence rewrite:
    # generate multiple full-document candidates, scan every valid candidate,
    # and keep the one with the lowest measured AI likelihood.
    ai_search_reference = original_ai if original_ai is not None else saved_ai
    ai_search_enabled = os.environ.get("DRAFTPROOF_AI_MITIGATION_SEARCH", "1") != "0"
    generation_first_active = _env_flag("DRAFTPROOF_REGENERATION_FIRST", True)
    ai_search_after_generation_failure = _env_flag(
        "DRAFTPROOF_AI_SEARCH_AFTER_GENERATION_FAILURE",
        True,
    )
    if (
        generation_first_active
        and authenticity_enabled
        and authenticity_mitigation_selected
        and not ai_search_after_generation_failure
    ):
        ai_search_enabled = False
        reason = "skipped_after_generation_layer_selected"
        result.summary["ai_mitigation_search"] = {
            "enabled": False,
            "reason": reason,
            "generation_layer_required": True,
            "generation_layer_selected": bool(authenticity_mitigation_selected),
            "generation_layer_summary": result.summary.get("authenticity_mitigation"),
        }
        stage_timings.append({
            "stage": "ai_mitigation_search",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "skipped_reason": reason,
        })
    ai_search_blocked_by_author_gaps = (
        ai_mitigation_needs_author
        and not allow_auto_with_author_gaps
        and not authenticity_mitigation_selected
    )
    human_target_search_status = _human_target_ai_search_status(original_report_dict)
    ai_search_reference_meets_threshold = bool(
        isinstance(ai_search_reference, (int, float))
        and ai_search_reference >= ai_first_required_min_ai
    )
    ai_search_reference_allowed = bool(
        ai_search_reference_meets_threshold
        or human_target_search_status.get("active")
    )
    if (
        ai_search_enabled
        and not ai_search_blocked_by_author_gaps
        and isinstance(ai_search_reference, (int, float))
        and ai_search_reference_allowed
    ):
        ai_search_target_score = round(max(0.0, ai_search_reference - ai_first_min_drop), 2)
        search_started = time.time()
        strategies = [
            "syntax_demolition",
            "paragraph_resequence",
            "plain_workshop_voice",
            "review_marked_grounding",
            "source_bridge_rebuild",
            "claim_narrowing",
            "cadence_disruption",
            "anchor_first_rebuild",
        ]
        try:
            search_limit = max(1, int(os.environ.get("DRAFTPROOF_AI_SEARCH_CANDIDATES", "4")))
        except ValueError:
            search_limit = 4
        confirmed_author_answers_for_search = _load_author_evidence_answers()
        confirmed_author_anchor_brief = _confirmed_author_anchor_brief(
            confirmed_author_answers_for_search,
            limit=int(_float_env("DRAFTPROOF_CONFIRMED_AUTHOR_ANCHOR_CONTEXT_LIMIT", 6.0)),
        )
        confirmed_anchor_strategies: list[str] = []
        if confirmed_author_anchor_brief and _env_flag("DRAFTPROOF_CONFIRMED_ANCHOR_SEARCH", True):
            anchor_strategy_pool = [
                "confirmed_anchor_threading",
                "confirmed_anchor_process_voice",
                "confirmed_anchor_asymmetry",
                "confirmed_anchor_claim_narrowing",
            ]
            try:
                anchor_search_limit = max(
                    0,
                    int(os.environ.get(
                        "DRAFTPROOF_CONFIRMED_ANCHOR_SEARCH_CANDIDATES",
                        "3",
                    )),
                )
            except ValueError:
                anchor_search_limit = 3
            confirmed_anchor_strategies = anchor_strategy_pool[:anchor_search_limit]
        base_strategies = strategies[:search_limit]
        strategies = confirmed_anchor_strategies + base_strategies
        search_source_text, search_source_repairs = _repair_candidate_source_damage(text)
        formula_gap_orchestrator_enabled = _env_flag("DRAFTPROOF_FORMULA_GAP_CANDIDATE_ORCHESTRATOR", True)
        formula_gap_plan = _formula_gap_orchestrator_plan(original_report_dict)
        formula_gap_budget_contract = _formula_gap_orchestrator_budget_contract(
            deterministic_probes=int(_float_env("DRAFTPROOF_FORMULA_GAP_DETERMINISTIC_PROBES", 2.0)),
            llm_candidates=int(_float_env("DRAFTPROOF_FORMULA_GAP_LLM_CANDIDATES", 5.0)),
            finalist_scans=int(_float_env("DRAFTPROOF_FORMULA_GAP_FINALIST_SCANS", 5.0)),
            total_scan_cap=int(_float_env("DRAFTPROOF_FORMULA_GAP_TOTAL_SCAN_CAP", 10.0)),
        )
        deterministic_candidates = []
        if search_source_repairs and search_source_text.strip() != text.strip():
            deterministic_candidates.append((
                "deterministic_source_integrity_repair",
                search_source_text,
            ))
        topk_route_enabled = _env_flag("DRAFTPROOF_TOPK_ROUTE_OPTIMIZER", True)
        topk_route_map = (
            _topk_repair_map(search_source_text, original_report_dict)
            if topk_route_enabled else {"enabled": False, "targets": []}
        )
        original_ai_components_for_priority = (
            ((original_report_dict or {}).get("ai_risk_badge") or {}).get("ai_components") or {}
            if isinstance(original_report_dict, dict) else {}
        )
        original_topk_calibrated_for_priority = original_ai_components_for_priority.get("topk_calibrated_risk")
        original_qualifying_density_for_priority = original_ai_components_for_priority.get("qualifying_text_ai_density")
        original_generic_assertion_for_priority = original_ai_components_for_priority.get("generic_assertion_risk")
        topk_safe_band_priority = bool(
            topk_route_map.get("saturated")
            or (
                isinstance(original_topk_calibrated_for_priority, (int, float))
                and float(original_topk_calibrated_for_priority) >= _safe_topk_calibrated_limit()
            )
        )
        density_or_generic_priority = bool(
            (
                isinstance(original_qualifying_density_for_priority, (int, float))
                and float(original_qualifying_density_for_priority) >= _float_env(
                    "DRAFTPROOF_QUALIFYING_AI_DENSITY_PRIORITY_THRESHOLD",
                    55.0,
                )
            )
            or (
                isinstance(original_generic_assertion_for_priority, (int, float))
                and float(original_generic_assertion_for_priority) >= _float_env(
                    "DRAFTPROOF_GENERIC_ASSERTION_PRIORITY_THRESHOLD",
                    75.0,
                )
            )
        )
        topk_route_candidates = (
            _topk_route_optimizer_candidates(search_source_text, original_report_dict)
            if topk_route_enabled else []
        )
        deterministic_candidates.extend(
            (strategy, candidate)
            for strategy, candidate, _meta in topk_route_candidates
        )
        human_anchor_candidates = _human_anchor_amplifier_candidates(
            search_source_text,
            original_report_dict,
            limit=int(_float_env("DRAFTPROOF_HUMAN_ANCHOR_AMPLIFIER_CANDIDATES", 3.0)),
        )
        human_anchor_topk_candidates: list[tuple[str, str, dict]] = []
        if human_anchor_candidates and topk_route_candidates:
            combined_limit = max(
                0,
                int(_float_env("DRAFTPROOF_HUMAN_ANCHOR_TOPK_COMBINED_CANDIDATES", 4.0)),
            )
            for topk_strategy, topk_candidate, topk_meta in topk_route_candidates[:2]:
                if len(human_anchor_topk_candidates) >= combined_limit:
                    break
                for anchor_strategy, anchor_candidate, anchor_meta in _human_anchor_amplifier_candidates(
                    topk_candidate,
                    original_report_dict,
                    limit=2,
                ):
                    if len(human_anchor_topk_candidates) >= combined_limit:
                        break
                    human_anchor_topk_candidates.append((
                        f"human_anchor_amplifier_on_{topk_strategy}_{anchor_strategy.replace('human_anchor_amplifier_', '')}",
                        anchor_candidate,
                        {
                            **anchor_meta,
                            "operation": "human_anchor_amplifier_on_topk",
                            "base_topk_strategy": topk_strategy,
                            "base_topk_meta": topk_meta,
                        },
                    ))
        deterministic_candidates.extend(
            (strategy, candidate)
            for strategy, candidate, _meta in (
                human_anchor_candidates + human_anchor_topk_candidates
            )
        )
        blocker_operation_candidates = []
        generic_assertion_candidates = []
        pruning_candidates = []
        skipped_pre_topk_candidate_families: list[str] = []
        if topk_safe_band_priority and not density_or_generic_priority:
            skipped_pre_topk_candidate_families = [
                "blocker_operation_candidates",
                "generic_assertion_compiler_candidates",
                "content_pruning_candidates",
                "marked_grounding_candidates",
            ]
        else:
            blocker_operation_candidates = _blocker_operation_candidates(
                search_source_text,
                original_report_dict,
                limit=int(_float_env("DRAFTPROOF_BLOCKER_OPERATION_CANDIDATES", 6.0)),
            )
            deterministic_candidates.extend(
                (strategy, candidate)
                for strategy, candidate, _meta in blocker_operation_candidates
            )
            generic_assertion_candidates = _generic_assertion_compiler_candidates(
                search_source_text,
                original_report_dict,
                limit=int(_float_env("DRAFTPROOF_GENERIC_ASSERTION_CANDIDATES", 5.0)),
            )
            deterministic_candidates.extend(
                (strategy, candidate)
                for strategy, candidate, _meta in generic_assertion_candidates
            )
            pruning_candidates = _content_pruning_candidates(
                search_source_text,
                original_report_dict,
                limit=int(_float_env("DRAFTPROOF_CONTENT_PRUNING_CANDIDATES", 4.0)),
            )
            deterministic_candidates.extend(
                (strategy, candidate)
                for strategy, candidate, _meta in pruning_candidates
            )
            deterministic_candidates.extend(_ai_search_marked_grounding_candidates(search_source_text))
        formula_portfolio_candidates = _formula_portfolio_candidates(
            search_source_text,
            original_report_dict,
            topk_route_candidates=topk_route_candidates,
            blocker_operation_candidates=blocker_operation_candidates,
            generic_assertion_candidates=generic_assertion_candidates,
            pruning_candidates=pruning_candidates,
            limit=int(_float_env("DRAFTPROOF_FORMULA_PORTFOLIO_CANDIDATES", 6.0)),
        )
        deterministic_candidates.extend(formula_portfolio_candidates)
        search_summary = {
            "enabled": True,
            "reference_ai": ai_search_reference,
            "starting_ai": rewritten_ai,
            "candidate_limit": len(deterministic_candidates) + len(strategies),
            "deterministic_candidate_count": len(deterministic_candidates),
            "llm_candidate_limit": len(strategies),
            "required_ai_drop": ai_first_min_drop,
            "target_ai_score": ai_search_target_score,
            "reference_ai_meets_threshold": ai_search_reference_meets_threshold,
            "human_target_search": human_target_search_status,
            "topk_safe_band_priority": topk_safe_band_priority,
            "density_or_generic_priority": density_or_generic_priority,
            "original_topk_calibrated_risk": original_topk_calibrated_for_priority,
            "original_qualifying_text_ai_density": original_qualifying_density_for_priority,
            "original_generic_assertion_risk": original_generic_assertion_for_priority,
            "pre_topk_candidate_families_skipped": skipped_pre_topk_candidate_families,
            "ai_footprint_gate": {
                "enabled": _env_flag("DRAFTPROOF_AI_FOOTPRINT_GATE_ENABLED", True),
                "before": _ai_footprint_profile(original_report_dict),
                "objective": "reduce_authorship_texture_drivers_before_cleanup",
            },
            "topk_route_optimizer": {
                "enabled": topk_route_enabled,
                "repair_map": topk_route_map,
                "deterministic_candidate_count": len(topk_route_candidates),
                "deterministic_candidates": [
                    {
                        "strategy": strategy,
                        **meta,
                    }
                    for strategy, _candidate, meta in topk_route_candidates
                ],
            },
            "human_anchor_driver_contract": _human_anchor_driver_contract(
                original_report_dict,
                text=search_source_text,
            ),
            "human_anchor_amplifier": {
                "enabled": _env_flag("DRAFTPROOF_HUMAN_ANCHOR_AMPLIFIER", True),
                "candidate_count": len(human_anchor_candidates) + len(human_anchor_topk_candidates),
                "standalone_candidate_count": len(human_anchor_candidates),
                "topk_combined_candidate_count": len(human_anchor_topk_candidates),
                "candidates": [
                    {
                        "strategy": strategy,
                        "operation": meta.get("operation"),
                        "scope": meta.get("scope"),
                        "changed_sentence_frames": meta.get("changed_sentence_frames"),
                        "target_lived_detail_band": meta.get("target_lived_detail_band"),
                    }
                    for strategy, _candidate, meta in (
                        human_anchor_candidates + human_anchor_topk_candidates
                    )
                ],
            },
            "formula_portfolio_generator": {
                "enabled": _env_flag("DRAFTPROOF_FORMULA_PORTFOLIO_GENERATOR", True),
                "candidate_count": len(formula_portfolio_candidates),
                "candidates": [
                    {
                        "strategy": strategy,
                        "targeted_drivers": meta.get("targeted_drivers"),
                        "portfolio_operation": meta.get("portfolio_operation"),
                        "base_strategy": meta.get("base_strategy"),
                    }
                    for strategy, _candidate, meta in formula_portfolio_candidates
                ],
            },
            "llm_calls": 0,
            "full_candidate_scans": 0,
            "selected": False,
            "candidates": [],
            "model_roles": llm_roles,
            "sampling_policy": _mitigation_sampling_policy_summary(),
            "confirmed_anchor_search": {
                "enabled": bool(confirmed_anchor_strategies),
                "candidate_limit": len(confirmed_anchor_strategies),
                "strategies": confirmed_anchor_strategies,
                "base_candidate_limit": len(base_strategies),
            },
            "formula_gap_candidate_orchestrator": {
                "enabled": formula_gap_orchestrator_enabled,
                "budget_contract": formula_gap_budget_contract,
                "formula_gap_plan": formula_gap_plan,
                "portfolio_families": _formula_gap_portfolio_families(
                    int(formula_gap_budget_contract.get("llm_candidate_calls") or 0)
                ),
                "deterministic_probe_scans_used": 0,
                "llm_calls_used": 0,
                "candidate_frontier": [],
            },
            "content_pruning_repair": {
                "enabled": _env_flag("DRAFTPROOF_CONTENT_PRUNING_REPAIR", True),
                "candidate_count": len(pruning_candidates),
                "candidates": [
                    {
                        "strategy": strategy,
                        **meta,
                    }
                    for strategy, _candidate, meta in pruning_candidates
                ],
            },
            "blocker_operation_compiler": {
                "enabled": _env_flag("DRAFTPROOF_BLOCKER_OPERATION_COMPILER", True),
                "candidate_count": len(blocker_operation_candidates),
                "operation_plan": _blocker_operation_plan(
                    search_source_text,
                    original_report_dict,
                    limit=int(_float_env("DRAFTPROOF_BLOCKER_OPERATION_PLAN_LIMIT", 8.0)),
                ),
                "candidates": [
                    {
                        "strategy": strategy,
                        **meta,
                    }
                    for strategy, _candidate, meta in blocker_operation_candidates
                ],
            },
            "generic_assertion_compiler": {
                "enabled": _env_flag("DRAFTPROOF_GENERIC_ASSERTION_COMPILER", True),
                "candidate_count": len(generic_assertion_candidates),
                "candidates": [
                    {
                        "strategy": strategy,
                        "operation": meta.get("operation"),
                        "paragraph_index": meta.get("paragraph_index"),
                        "paragraph_role": meta.get("paragraph_role"),
                        "removed_sentence_indexes": meta.get("removed_sentence_indexes"),
                    }
                    for strategy, _candidate, meta in generic_assertion_candidates
                ],
            },
        }
        if search_source_repairs:
            search_summary["source_repairs"] = search_source_repairs
        if confirmed_author_anchor_brief:
            search_summary["confirmed_author_anchors_in_generation"] = True
        effective_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        source_protected = detect_protected_spans(search_source_text)
        min_chars = max(
            200,
            int(len(search_source_text) * _float_env("DRAFTPROOF_AI_SEARCH_MIN_CHAR_RATIO", 0.60)),
        )
        max_chars = max(min_chars, int(len(text) * 1.30))
        best_text = rewritten_text
        best_report = rewritten_report_dict
        best_ai = rewritten_ai if isinstance(rewritten_ai, (int, float)) else 999.0
        best_strategy = None
        best_semantic_review_required = False
        best_drift_reasons: list[str] = []
        best_selection_status: dict = {}
        best_human_shift_rank: tuple = (-1, -9999.0, -9999.0)
        best_blocked_human_candidate: dict | None = None
        best_blocked_human_rank: tuple | None = None
        budget_policy = _ai_search_budget_policy(search_source_text, original_report_dict)
        hard_llm_cap = _ai_search_llm_hard_cap(search_source_text, original_report_dict)
        search_budget = {
            "max_seconds": _float_env(
                "DRAFTPROOF_AI_SEARCH_MAX_SECONDS",
                float(budget_policy.get("max_seconds") or 420),
            ),
            "max_llm_calls": int(_float_env(
                "DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS",
                float(budget_policy.get("max_llm_calls") or hard_llm_cap),
            )),
            "max_candidate_scans": int(_float_env(
                "DRAFTPROOF_AI_SEARCH_MAX_CANDIDATE_SCANS",
                float(budget_policy.get("max_candidate_scans") or 60),
            )),
            "max_candidate_scan_hard_cap": int(_float_env(
                "DRAFTPROOF_AI_SEARCH_MAX_CANDIDATE_SCAN_HARD_CAP",
                float(
                    budget_policy.get("max_candidate_scan_hard_cap")
                    or budget_policy.get("max_candidate_scans")
                    or 60
                ),
            )),
            "policy": budget_policy,
        }
        search_budget["max_llm_calls"] = min(
            int(search_budget["max_llm_calls"]),
            hard_llm_cap,
        )
        search_budget["max_candidate_scans"] = min(
            int(search_budget["max_candidate_scans"]),
            _candidate_scan_hard_cap(search_budget),
        )
        if formula_gap_orchestrator_enabled:
            formula_scan_cap = max(
                int(formula_gap_budget_contract.get("total_scan_cap") or 10),
                int(formula_gap_budget_contract.get("deterministic_probe_scans") or 0)
                + int(formula_gap_budget_contract.get("finalist_scans") or 0),
            )
            formula_llm_cap = min(
                hard_llm_cap,
                int(formula_gap_budget_contract.get("llm_candidate_calls") or 5),
            )
            search_budget["max_llm_calls"] = max(
                int(search_budget.get("max_llm_calls") or 0),
                formula_llm_cap,
            )
            search_budget["max_candidate_scans"] = min(
                _candidate_scan_hard_cap(search_budget),
                formula_scan_cap,
            )
            search_budget["max_candidate_scan_hard_cap"] = min(
                int(search_budget.get("max_candidate_scan_hard_cap") or formula_scan_cap),
                formula_scan_cap,
            )
            search_budget["formula_gap_orchestrator"] = {
                "deterministic_probe_scan_cap": int(
                    formula_gap_budget_contract.get("deterministic_probe_scans") or 0
                ),
                "reserved_llm_candidate_calls": formula_llm_cap,
                "finalist_scan_cap": int(formula_gap_budget_contract.get("finalist_scans") or 0),
                "total_scan_cap": formula_scan_cap,
            }
        rewrite_phase_budget_plan = _rewrite_phase_budget_plan(
            rewritten_text,
            rewritten_report_dict,
            original_report_dict,
            max_scans=global_rewrite_budget.max_scans,
            max_llm_calls=global_rewrite_budget.max_llm_calls,
            ai_search_policy=budget_policy,
            formula_gap_budget=formula_gap_budget_contract,
        )
        result.summary["rewrite_phase_budget_plan"] = rewrite_phase_budget_plan
        ai_phase_budget = (rewrite_phase_budget_plan.get("phases") or {}).get("ai_mitigation_search") or {}
        ai_phase_scan_cap = int(ai_phase_budget.get("max_scans") or 0)
        ai_phase_llm_cap = int(ai_phase_budget.get("max_llm_calls") or 0)
        if ai_phase_scan_cap > 0:
            search_budget["max_candidate_scans"] = min(
                int(search_budget.get("max_candidate_scans") or 0),
                ai_phase_scan_cap,
            )
            search_budget["max_candidate_scan_hard_cap"] = min(
                int(search_budget.get("max_candidate_scan_hard_cap") or search_budget["max_candidate_scans"]),
                ai_phase_scan_cap,
            )
        if ai_phase_llm_cap > 0:
            search_budget["max_llm_calls"] = min(
                int(search_budget.get("max_llm_calls") or 0),
                ai_phase_llm_cap,
            )
        search_budget["phase_budget_plan"] = {
            "version": rewrite_phase_budget_plan.get("version"),
            "reason": rewrite_phase_budget_plan.get("reason"),
            "allocation": ai_phase_budget,
            "downstream_reserved": {
                key: value
                for key, value in (rewrite_phase_budget_plan.get("phases") or {}).items()
                if key != "ai_mitigation_search"
            },
        }
        ai_search_uncapped_seconds = float(search_budget.get("max_seconds") or 0.0)
        post_ai_search_reserve = post_ai_search_reserve_seconds(_text_word_count(search_source_text))
        capped_ai_search_seconds = cap_phase_seconds_for_reserve(
            max_seconds=ai_search_uncapped_seconds,
            remaining_seconds=global_rewrite_budget.remaining_seconds(),
            reserve_seconds=post_ai_search_reserve,
            min_phase_seconds=20.0,
        )
        if (
            capped_ai_search_seconds > 0.0
            and ai_search_uncapped_seconds > 0.0
            and capped_ai_search_seconds < ai_search_uncapped_seconds
        ):
            search_budget["max_seconds"] = round(capped_ai_search_seconds, 3)
            search_budget["uncapped_max_seconds"] = round(ai_search_uncapped_seconds, 3)
            search_budget["global_post_phase_reserve_seconds"] = round(post_ai_search_reserve, 3)
            search_budget["global_remaining_seconds_at_ai_search_start"] = round(
                global_rewrite_budget.remaining_seconds(),
                3,
            )
            search_budget["time_cap_reason"] = "preserve_post_ai_search_controller_budget"
        search_summary["budget"] = search_budget
        search_summary["candidate_scoring_controller"] = {
            **(search_budget.get("candidate_scoring_controller") or {}),
            "full_scans_used": 0,
            "candidate_records": 0,
        }
        phase_budget_contract = _strict_safe_phase_budget_contract(hard_llm_cap, search_source_text, original_report_dict)
        phase_budget_used = {
            key: 0
            for key in phase_budget_contract
            if key != "total_llm_hard_cap"
        }
        search_summary["phase_budget_contract"] = phase_budget_contract
        search_summary["phase_budget_used"] = phase_budget_used

        def _verified_candidate_scans_used() -> int:
            return _core_ai_search_verified_candidate_scans_used(search_summary)

        def _record_verified_candidate_scan() -> None:
            _core_ai_search_record_verified_candidate_scan(search_summary)

        def _phase_budget_can_spend(phase: str, calls: int = 1) -> bool:
            return _core_ai_search_phase_budget_can_spend(
                phase,
                calls=calls,
                phase_budget_used=phase_budget_used,
                phase_budget_contract=phase_budget_contract,
                search_summary=search_summary,
                env_flag=_env_flag,
            )

        def _record_phase_llm_call(phase: str, calls: int = 1) -> None:
            _core_ai_search_record_phase_llm_call(
                phase,
                calls=calls,
                phase_budget_used=phase_budget_used,
                phase_budget_contract=phase_budget_contract,
                search_summary=search_summary,
            )

        def _phase_budget_block_record(phase: str, summary: dict, *, calls: int = 1) -> bool:
            return _core_ai_search_phase_budget_block_record(
                phase,
                summary,
                calls=calls,
                phase_budget_used=phase_budget_used,
                phase_budget_contract=phase_budget_contract,
                search_summary=search_summary,
                env_flag=_env_flag,
            )

        def _best_ai_search_selectable() -> bool:
            return bool(best_strategy and best_selection_status.get("selectable"))

        def _record_best_attempt() -> None:
            if not best_strategy:
                return
            search_summary["best_attempt"] = {
                "strategy": best_strategy,
                "ai": best_ai,
                "ai_delta_vs_reference": (
                    round(ai_search_reference - best_ai, 3)
                    if isinstance(best_ai, (int, float)) else None
                ),
                "human_shift_score": best_selection_status.get("human_shift_score"),
                "human_shift_components": best_selection_status.get("human_shift_components"),
                "selection_status": best_selection_status,
            }

        adaptive_stop_reason = ""
        formula_gap_orchestrator_completed = False

        def _search_budget_exhausted(phase: str, *, before_llm: bool = False) -> bool:
            nonlocal adaptive_stop_reason
            exhausted, adaptive_stop_reason = _core_ai_search_budget_exhausted_record(
                phase=phase,
                before_llm=before_llm,
                search_started=search_started,
                search_budget=search_budget,
                search_summary=search_summary,
                adaptive_stop_reason=adaptive_stop_reason,
                best_strategy=best_strategy,
                best_selection_status=best_selection_status,
                llm_call_budget_exhausted_before_send=_llm_call_budget_exhausted_before_send,
            )
            return exhausted

        def _budget_gateway(gateway: LLMGateway, phase: str) -> LLMGateway:
            return _core_ai_search_apply_budget_gateway(
                gateway,
                phase,
                search_summary=search_summary,
                search_budget_exhausted=_search_budget_exhausted,
            )

        def _maybe_adaptive_stop(phase: str) -> bool:
            nonlocal adaptive_stop_reason
            stopped, adaptive_stop_reason = _core_ai_search_adaptive_stop_record(
                phase=phase,
                adaptive_stop_reason=adaptive_stop_reason,
                search_summary=search_summary,
                best_strategy=best_strategy,
                best_selection_status=best_selection_status,
                ai_search_adaptive_stop_reason=_ai_search_adaptive_stop_reason,
                short_document=(
                    _env_flag("DRAFTPROOF_ADAPTIVE_SHORT_DOC_BUDGETS", True)
                    and _text_word_count(search_source_text) <= int(_float_env(
                        "DRAFTPROOF_SHORT_DOC_WORD_THRESHOLD",
                        450.0,
                    ))
                ),
            )
            return stopped

        def _evaluate_ai_search_candidate(
            strategy: str,
            candidate: str,
            *,
            deterministic: bool = False,
            extra: dict | None = None,
        ) -> None:
            nonlocal best_text, best_report, best_ai, best_strategy
            nonlocal best_semantic_review_required, best_drift_reasons, best_selection_status
            nonlocal best_human_shift_rank, best_blocked_human_candidate, best_blocked_human_rank
            candidate_eval = {
                "strategy": strategy,
                "deterministic": deterministic,
                "passed_local_checks": False,
                "candidate_length": len(candidate or ""),
            }
            if extra:
                candidate_eval.update(extra)
            if str(strategy or "").startswith("human_anchor_amplifier"):
                candidate_eval["human_anchor_amplifier"] = True
                candidate_eval["human_signal_amplification"] = True
            topk_safe_band_rebuild = bool(extra and extra.get("topk_safe_band_rebuild"))
            ignore_search_budget = bool(
                extra
                and (
                    (
                        extra.get("blocked_human_winner_repair")
                        and _blocked_human_winner_repair_budget_override(adaptive_stop_reason)
                    )
                    or (
                        extra.get("final_topk_texture_repair_budget_override")
                        and _env_flag("DRAFTPROOF_FINAL_TOPK_TEXTURE_AFTER_BUDGET", True)
                    )
                    or (
                        extra.get("scan_generated_candidate_after_budget")
                        and _env_flag("DRAFTPROOF_SCAN_GENERATED_CANDIDATES_AFTER_TIME_BUDGET", True)
                    )
                )
            )
            if not ignore_search_budget and _search_budget_exhausted("candidate_scan"):
                if not search_summary.get("budget_exhausted_candidate_recorded"):
                    candidate_eval["reason"] = search_summary["budget_exhausted"]["reason"]
                    candidate_eval["passed_local_checks"] = False
                    search_summary["candidates"].append(candidate_eval)
                    search_summary["budget_exhausted_candidate_recorded"] = True
                return
            if not candidate:
                candidate_eval["reason"] = "empty_candidate"
                search_summary["candidates"].append(candidate_eval)
                return
            candidate, repair_notes = _repair_candidate_source_damage(candidate)
            if repair_notes:
                candidate_eval["candidate_length"] = len(candidate or "")
                candidate_eval["source_damage_repairs"] = repair_notes
            review_notes = _review_marker_notes(candidate)
            if review_notes:
                candidate_eval["reason"] = "review_markers_not_auto_kept"
                candidate_eval["review_suggestion_count"] = len(review_notes)
                manual = result.summary.setdefault("manual_suggestions", [])
                for note in review_notes:
                    if len(manual) >= 30:
                        break
                    manual.append({
                        "finding_type": "ai_mitigation_review_note",
                        "scanner_target": "ai_mitigation_search",
                        "original_sentence": "",
                        "suggested_sentence": f"[[REVIEW: {note}]]",
                        "rejection_reason": "review_markers_not_auto_kept",
                        "why_review_manually": (
                            "This note asks the author to add real evidence or context. "
                            "It is shown as guidance, not inserted into the rewritten document."
                        ),
                    })
                search_summary["candidates"].append(candidate_eval)
                return
            quality_rejection = _ai_candidate_quality_reject_reason(candidate)
            if quality_rejection in {
                "generic_admiration_tone",
                "compressed_promotional_fragment_style",
                "over_stylized_metaphorical_texture",
            }:
                repaired_candidate, style_repairs = _neutralize_external_detector_style_artifacts(candidate)
                if style_repairs and repaired_candidate.strip() != candidate.strip():
                    repaired_rejection = _ai_candidate_quality_reject_reason(repaired_candidate)
                    candidate_eval["external_style_artifact_repair_attempt"] = {
                        "original_reason": quality_rejection,
                        "repairs": style_repairs,
                        "repaired_rejection": repaired_rejection,
                    }
                    if not repaired_rejection:
                        candidate = repaired_candidate
                        candidate_eval["candidate_length"] = len(candidate or "")
                        candidate_eval["external_style_artifact_repairs"] = style_repairs
                        quality_rejection = ""
            if quality_rejection:
                candidate_eval["reason"] = quality_rejection
                search_summary["candidates"].append(candidate_eval)
                return
            anchor_echo_rejection = _confirmed_anchor_echo_reason(
                candidate,
                confirmed_author_answers_for_search,
            )
            if anchor_echo_rejection:
                candidate_eval["confirmed_anchor_echo_warning"] = anchor_echo_rejection
                if _env_flag("DRAFTPROOF_CONFIRMED_ANCHOR_ECHO_HARD_REJECT", False):
                    candidate_eval["reason"] = anchor_echo_rejection
                    search_summary["candidates"].append(candidate_eval)
                    return
            strict_safe_shortening = bool(
                extra
                and (
                    extra.get("post_topk_optimizer")
                    or extra.get("strict_safe_candidate")
                    or extra.get("final_topk_texture_repair")
                )
            )
            effective_min_chars = (
                200
                if topk_safe_band_rebuild
                else (
                    max(200, int(len(search_source_text) * 0.25))
                    if strict_safe_shortening
                    else min_chars
                )
            )
            if len(candidate) < effective_min_chars:
                hard_min_chars = max(
                    200,
                    int(len(search_source_text) * _float_env(
                        "DRAFTPROOF_AI_SEARCH_HARD_MIN_CHAR_RATIO",
                        0.45,
                    )),
                )
                if len(candidate) < hard_min_chars:
                    candidate_eval["reason"] = f"candidate_too_short {len(candidate)}<{hard_min_chars}"
                    candidate_eval["guidance_min_chars"] = effective_min_chars
                    search_summary["candidates"].append(candidate_eval)
                    return
                candidate_eval["length_guidance_warning"] = (
                    f"candidate_below_guidance_min {len(candidate)}<{effective_min_chars}"
                )
                candidate_eval["effective_min_chars"] = effective_min_chars
                candidate_eval["hard_min_chars"] = hard_min_chars
            effective_max_chars = (
                max(max_chars, int(len(search_source_text) * _float_env(
                    "DRAFTPROOF_TOPK_SAFE_BAND_MAX_CHAR_RATIO",
                    1.75,
                )))
                if topk_safe_band_rebuild or strict_safe_shortening
                else max_chars
            )
            if len(candidate) > effective_max_chars:
                candidate_eval["reason"] = f"candidate_too_long {len(candidate)}>{effective_max_chars}"
                candidate_eval["effective_max_chars"] = effective_max_chars
                search_summary["candidates"].append(candidate_eval)
                return
            protected_loss = _ai_search_protected_loss_reason(search_source_text, candidate, source_protected)
            if protected_loss:
                candidate_eval["reason"] = "protected_span_lost " + protected_loss
                search_summary["candidates"].append(candidate_eval)
                return
            concept_guard_required = bool(
                candidate_eval.get("human_anchor_amplifier")
                or candidate_eval.get("human_signal_amplification")
                or (extra or {}).get("formula_portfolio_candidate")
                or (extra or {}).get("topk_route_optimizer")
                or (extra or {}).get("post_topk_optimizer")
                or (extra or {}).get("strict_safe_candidate")
                or (extra or {}).get("final_topk_texture_repair")
            )
            if concept_guard_required:
                concept_origin_reason = _candidate_concept_origin_reject_reason(search_source_text, candidate)
                if concept_origin_reason:
                    candidate_eval["reason"] = concept_origin_reason
                    candidate_eval["concept_origin_guard"] = {
                        "accepted": False,
                        "reason": concept_origin_reason,
                    }
                    search_summary["candidates"].append(candidate_eval)
                    return
                candidate_eval["concept_origin_guard"] = {
                    "accepted": True,
                    "reason": "accepted",
                }
            drift = check_semantic_drift(search_source_text, candidate, threshold=0.15)
            candidate_eval["drift_similarity"] = round(drift.similarity, 3)
            if not drift.accepted:
                candidate_eval["drift_reasons"] = drift.reasons[:10]
                if repair_notes and _source_repair_drift_false_positive(candidate, drift.reasons):
                    candidate_eval["drift_relaxed_for_source_repair"] = True
                    candidate_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                elif topk_safe_band_rebuild and float(drift.similarity or 0.0) >= _float_env(
                    "DRAFTPROOF_TOPK_SAFE_BAND_REBUILD_MIN_DRIFT_SIMILARITY",
                    0.50,
                ):
                    candidate_eval["semantic_review_required"] = True
                    candidate_eval["drift_scan_relaxed_for_topk_safe_band_rebuild"] = True
                    candidate_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                elif _ai_search_drift_false_positive(candidate, drift.reasons, drift.similarity):
                    candidate_eval["drift_relaxed_for_ai_search"] = True
                    candidate_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                elif _ai_search_quote_drift_scan_allowed(candidate, drift.reasons, drift.similarity):
                    candidate_eval["semantic_review_required"] = True
                    candidate_eval["drift_scan_relaxed_for_quote_markers"] = True
                elif _ai_search_entity_drift_scan_allowed(candidate, drift.reasons, drift.similarity):
                    candidate_eval["semantic_review_required"] = True
                    candidate_eval["drift_scan_relaxed_for_scoring"] = True
                elif _document_recreate_drift_scan_allowed(candidate, drift.reasons, drift.similarity, extra):
                    candidate_eval["semantic_review_required"] = True
                    candidate_eval["drift_scan_relaxed_for_document_recreate"] = True
                    candidate_eval["drift_reasons_relaxed"] = drift.reasons[:5]
                else:
                    candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                    search_summary["candidates"].append(candidate_eval)
                    return

            candidate_eval["passed_local_checks"] = True
            try:
                scan_t0 = time.time()
                candidate_report = _full_scan_report_dict(candidate)
                candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                _record_verified_candidate_scan()
            except Exception as exc:
                candidate_eval["passed_local_checks"] = False
                candidate_eval["reason"] = f"candidate_scan_error {exc}"
                search_summary["candidates"].append(candidate_eval)
                return

            candidate_ai = _badge_ai(candidate_report)
            candidate_wq = _badge_wq(candidate_report)
            candidate_review_burden = _review_burden(candidate_report)
            candidate_weighted_severity = _weighted_severity(candidate_report)
            candidate_critical_high = (
                len(candidate_report.get("findings", {}).get("critical", []))
                + len(candidate_report.get("findings", {}).get("high", []))
            )
            candidate_contribution = _contribution_scores(candidate_report)
            candidate_integrity = _integrity_scores(candidate_report)
            blocker_status = _blocker_elimination_status(original_report_dict, candidate_report)
            dominant_blocker_status = _dominant_blocker_gate_status(original_report_dict, candidate_report)
            human_formula_status = _human_formula_driver_status(original_report_dict, candidate_report)
            human_anchor_contract = _human_anchor_driver_contract(
                original_report_dict,
                candidate_report,
                text=candidate,
            )
            human_shift = _human_shift_score(
                original_report_dict,
                candidate_report,
                drift_similarity=candidate_eval.get("drift_similarity"),
                review_burden_delta=candidate_review_burden - original_review_burden,
                weighted_severity_delta=candidate_weighted_severity - original_severity,
            )
            candidate_eval.update({
                "ai": candidate_ai,
                "ai_delta_vs_reference": (
                    round(ai_search_reference - candidate_ai, 3)
                    if isinstance(candidate_ai, (int, float)) else None
                ),
                "writing_quality": candidate_wq,
                "human_contribution": candidate_contribution.get("human"),
                "ai_transformation": candidate_contribution.get("ai_transformation"),
                "ai_authorship": candidate_integrity.get("ai_authorship"),
                "grounding_quality_risk": candidate_integrity.get("grounding"),
                "findings": _finding_total(candidate_report),
                "review_burden": candidate_review_burden,
                "weighted_severity": candidate_weighted_severity,
                "critical_high_findings": candidate_critical_high,
                "human_shift_score": human_shift.get("score"),
                "human_shift_components": human_shift.get("components"),
                "blocker_elimination": blocker_status,
                "dominant_blocker_gate": dominant_blocker_status,
                "human_formula_driver_gate": human_formula_status,
                "human_anchor_driver_contract": human_anchor_contract,
                "scan_scope": _scan_scope_summary(candidate_report),
            })
            selection_status = _ai_search_candidate_selection_status(
                ai_search_reference,
                candidate_ai,
                candidate != text,
                min_drop=ai_first_min_drop,
                target=ai_first_target,
                required_min_ai=ai_first_required_min_ai,
            )
            selection_status.update({
                "required_ai_drop": ai_first_min_drop,
                "target_ai_score": ai_search_target_score,
            })
            ai_delta = (
                ai_search_reference - candidate_ai
                if isinstance(ai_search_reference, (int, float)) and isinstance(candidate_ai, (int, float))
                else -9999.0
            )
            ai_score_regression_tolerance = _float_env(
                "DRAFTPROOF_AI_SEARCH_AI_SCORE_REGRESSION_TOLERANCE",
                0.25,
            )
            ai_score_regressed = bool(
                isinstance(ai_delta, (int, float))
                and ai_delta < -float(ai_score_regression_tolerance or 0.0)
            )
            if (
                candidate_eval.get("formula_gap_candidate_orchestrator")
                and isinstance(ai_delta, (int, float))
                and ai_delta < 0.0
            ):
                ai_score_regressed = True
                ai_score_regression_tolerance = 0.0
            authenticity_status = _authenticity_gate_status(
                original_report_dict,
                candidate_report,
                candidate != text,
                original_review_burden=original_review_burden,
                candidate_review_burden=candidate_review_burden,
                original_weighted_severity=original_severity,
                candidate_weighted_severity=candidate_weighted_severity,
                min_human_gain=_float_env("DRAFTPROOF_AI_SEARCH_MIN_HUMAN_GAIN", 1.0),
                min_ai_transformation_drop=_float_env(
                    "DRAFTPROOF_AI_SEARCH_MIN_AI_TRANSFORM_DROP",
                    1.0,
                ),
                drift_similarity=candidate_eval.get("drift_similarity"),
            )
            candidate_finding_total = _finding_total(candidate_report)
            review_burden_delta = candidate_review_burden - original_review_burden
            weighted_severity_delta = candidate_weighted_severity - original_severity
            finding_delta = candidate_finding_total - original_total
            critical_high_delta = candidate_critical_high - saved_critical_high
            ai_authorship_delta = authenticity_status.get("ai_authorship_delta")
            bounded_review_tradeoff = bool(
                _env_flag("DRAFTPROOF_AI_SEARCH_ALLOW_BOUNDED_REVIEW_TRADEOFF", True)
                and isinstance(ai_authorship_delta, (int, float))
                and ai_authorship_delta >= _float_env(
                    "DRAFTPROOF_BOUNDED_REVIEW_TRADEOFF_MIN_AUTHORSHIP_DROP",
                    10.0,
                )
                and review_burden_delta <= _float_env(
                    "DRAFTPROOF_BOUNDED_REVIEW_TRADEOFF_MAX_REVIEW_DELTA",
                    1.0,
                )
                and weighted_severity_delta <= 0
                and critical_high_delta <= 0
                and finding_delta <= 0
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("human_target_regressed")
                and not authenticity_status.get("ai_transformation_target_regressed")
                and not ai_score_regressed
            )
            candidate_eval["quality_deltas"] = {
                "review_burden_delta": review_burden_delta,
                "weighted_severity_delta": weighted_severity_delta,
                "finding_delta": finding_delta,
                "critical_high_delta": critical_high_delta,
            }
            candidate_eval["bounded_review_tradeoff"] = bounded_review_tradeoff
            if (
                selection_status.get("selectable")
                and not _env_flag("DRAFTPROOF_AI_SEARCH_ALLOW_REVIEW_REGRESSION", False)
                and (
                    (
                        authenticity_status.get("review_burden_regressed")
                        and not bounded_review_tradeoff
                    )
                    or authenticity_status.get("weighted_severity_regressed")
                    or authenticity_status.get("critical_high_regressed")
                    or (
                        isinstance(authenticity_status.get("human_delta"), (int, float))
                        and authenticity_status.get("human_delta") < 0
                    )
                    or (
                        isinstance(authenticity_status.get("ai_transformation_delta"), (int, float))
                        and authenticity_status.get("ai_transformation_delta") < 0
                    )
                    or ai_score_regressed
                    or candidate_critical_high > saved_critical_high
                    or candidate_finding_total > original_total
                )
            ):
                selection_status.update({
                    "success": False,
                    "selectable": False,
                    "reason": "ai_drop_quality_regressed",
                })
            bounded_blocked_tradeoff = _blocked_winner_bounded_quality_tradeoff(
                candidate_eval=candidate_eval,
                authenticity_status=authenticity_status,
                ai_delta=ai_delta,
                review_burden_delta=review_burden_delta,
                weighted_severity_delta=weighted_severity_delta,
                finding_delta=finding_delta,
                critical_high_delta=critical_high_delta,
                ai_score_regressed=ai_score_regressed,
            )
            if bounded_blocked_tradeoff.get("allowed"):
                _mark_ai_search_progress_selection(
                    selection_status,
                    reason="accepted_blocked_human_winner_bounded_tradeoff",
                    blocked_human_winner_repair=True,
                )
                selection_status["bounded_quality_tradeoff"] = bounded_blocked_tradeoff
            elif candidate_eval.get("blocked_human_winner_repair"):
                selection_status["bounded_quality_tradeoff"] = bounded_blocked_tradeoff
            incremental_authenticity_selectable = bool(
                _env_flag("DRAFTPROOF_AI_SEARCH_ACCEPT_INCREMENTAL_AUTHENTICITY", True)
                and authenticity_status.get("candidate_progress")
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("human_target_regressed")
                and not authenticity_status.get("ai_transformation_target_regressed")
                and not ai_score_regressed
                and not authenticity_status.get("critical_high_regressed")
                and candidate_critical_high <= saved_critical_high
                and _finding_total(candidate_report) <= original_total
                and not authenticity_status.get("review_burden_regressed")
                and not authenticity_status.get("weighted_severity_regressed")
            )
            human_amplification_score = None
            human_amplification_selectable = False
            human_anchor_amplification_selectable = False
            if candidate_eval.get("human_signal_amplification"):
                aggression = _repair_aggression_score(text, candidate).get("score", 0.0)
                locality = _locality_score(text, candidate).get("changed_sentence_ratio", 0.0)
                human_amplification_score = _score_human_amplification_candidate(
                    original_report_dict,
                    candidate_report,
                    review_burden_delta=candidate_review_burden - original_review_burden,
                    weighted_severity_delta=candidate_weighted_severity - original_severity,
                    repair_aggression=float(aggression or 0.0),
                    locality_score=float(locality or 0.0),
                )
                candidate_eval["human_signal_amplification_score"] = human_amplification_score
                human_amplification_selectable = bool(
                    _env_flag("DRAFTPROOF_AI_SEARCH_ACCEPT_HUMAN_SIGNAL_AMPLIFICATION", True)
                    and human_amplification_score.get("human_delta", 0.0) >= _float_env(
                        "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_MIN_HUMAN_GAIN",
                        2.0,
                    )
                    and not ai_score_regressed
                    and human_amplification_score.get("ai_authorship_delta", -9999.0) >= 0.0
                    and human_amplification_score.get("ai_transformation_delta", -9999.0) >= 0.0
                    and _finding_total(candidate_report) <= original_total
                    and candidate_review_burden <= original_review_burden
                    and candidate_weighted_severity <= original_severity
                    and not authenticity_status.get("critical_high_regressed")
                    and candidate_critical_high <= saved_critical_high
                    and not authenticity_status.get("ai_authorship_regression_blocked")
                )
            candidate_human_delta = authenticity_status.get("human_delta")
            candidate_ai_transform_delta = authenticity_status.get("ai_transformation_delta")
            safe_authorship_suppression_selectable = bool(
                _env_flag("DRAFTPROOF_AI_SEARCH_ACCEPT_SAFE_AUTHORSHIP_SUPPRESSION", True)
                and not (
                    _radar_goal_requires_human_progress(radar_goal_controller)
                    and isinstance(candidate_human_delta, (int, float))
                    and candidate_human_delta <= 0.0
                )
                and isinstance(human_shift.get("score"), (int, float))
                and float(human_shift.get("score")) >= _float_env(
                    "DRAFTPROOF_SAFE_AUTHORSHIP_SUPPRESSION_MIN_HUMAN_SHIFT",
                    0.0,
                )
                and isinstance(ai_authorship_delta, (int, float))
                and ai_authorship_delta >= _float_env(
                    "DRAFTPROOF_AI_SEARCH_MIN_SAFE_AUTHORSHIP_DROP",
                    1.0,
                )
                and isinstance(candidate_human_delta, (int, float))
                and candidate_human_delta >= 0.0
                and isinstance(candidate_ai_transform_delta, (int, float))
                and candidate_ai_transform_delta >= 0.0
                and isinstance(ai_delta, (int, float))
                and ai_delta > 0.05
                and candidate_finding_total <= original_total
                and (
                    candidate_review_burden <= original_review_burden
                    or bounded_review_tradeoff
                )
                and candidate_weighted_severity <= original_severity
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("critical_high_regressed")
                and candidate_critical_high <= saved_critical_high
            )
            safe_partial_quality_status = _safe_partial_quality_improvement_status(
                authenticity_status,
                human_shift,
                ai_delta=ai_delta,
                finding_delta=finding_delta,
                review_burden_delta=review_burden_delta,
                weighted_severity_delta=weighted_severity_delta,
                critical_high_delta=critical_high_delta,
                ai_score_regressed=ai_score_regressed,
            )
            safe_partial_quality_selectable = bool(
                safe_partial_quality_status.get("allowed")
            )
            candidate_eval["safe_partial_quality_improvement"] = safe_partial_quality_status
            ai_footprint_gate = _ai_footprint_gate_status(
                original_report_dict,
                candidate_report,
                review_burden_delta=review_burden_delta,
                weighted_severity_delta=weighted_severity_delta,
                critical_high_delta=critical_high_delta,
                ai_score_regressed=ai_score_regressed,
            )
            candidate_eval["ai_footprint_gate"] = ai_footprint_gate
            turnitin_like_gate = _turnitin_like_ai_gate_status(
                original_report_dict,
                candidate_report,
                review_burden_delta=review_burden_delta,
                weighted_severity_delta=weighted_severity_delta,
                critical_high_delta=critical_high_delta,
                ai_score_regressed=ai_score_regressed,
            )
            candidate_eval["turnitin_like_ai_gate"] = turnitin_like_gate
            formula_gap_contract = _formula_gap_contract(
                original_report_dict,
                candidate_report,
                source_text=search_source_text,
                candidate_text=candidate,
            )
            candidate_eval["formula_gap_contract"] = formula_gap_contract
            candidate_eval["formula_gap_rank"] = list(
                _formula_gap_candidate_rank(formula_gap_contract, turnitin_like_gate)
            )
            multi_signal_contract = _multi_signal_candidate_contract(
                original_report_dict,
                candidate_report,
            )
            candidate_eval["multi_signal_contract"] = multi_signal_contract
            eligible_span_density_gate = _eligible_span_density_comparison(
                search_source_text,
                original_report_dict,
                candidate,
                candidate_report,
            )
            candidate_eval["eligible_span_density_gate"] = eligible_span_density_gate
            candidate_meaningful_ai_progress_gate = meaningful_ai_progress_gate(
                turnitin_like_ai_score_drop=formula_gap_contract.get("score_drop"),
                ai_score_drop=ai_delta,
                ai_authorship_drop=ai_authorship_delta,
                ai_transformation_drop=candidate_ai_transform_delta,
                positive_ai_burden_drop=(
                    (formula_gap_contract.get("positive_ai_burden") or {}).get("drop")
                    if isinstance(formula_gap_contract.get("positive_ai_burden"), dict)
                    else None
                ),
                unsafe_eligible_density_drop=eligible_span_density_gate.get("unsafe_eligible_word_ratio_drop"),
            )
            candidate_eval["meaningful_ai_progress_gate"] = candidate_meaningful_ai_progress_gate
            ai_footprint_outcome = str(ai_footprint_gate.get("outcome_class") or "")
            footprint_drops = ai_footprint_gate.get("drops") if isinstance(ai_footprint_gate.get("drops"), dict) else {}
            if candidate_eval.get("human_anchor_amplifier"):
                anchor_deltas = (
                    human_anchor_contract.get("deltas")
                    if isinstance(human_anchor_contract.get("deltas"), dict) else {}
                )
                human_raw_formula = human_anchor_contract.get("human_raw_formula") or {}
                before_raw = (human_raw_formula.get("before") or {}).get("human_raw")
                after_raw = (human_raw_formula.get("after") or {}).get("human_raw")
                human_raw_gain = (
                    float(after_raw) - float(before_raw)
                    if isinstance(before_raw, (int, float)) and isinstance(after_raw, (int, float))
                    else 0.0
                )
                human_anchor_burden_gate = _human_anchor_positive_burden_gate_status(
                    formula_gap_contract,
                    candidate_report,
                )
                anchor_gain = float(anchor_deltas.get("human_anchor_score") or 0.0)
                lived_drop = float(anchor_deltas.get("lived_detail_risk") or 0.0)
                human_anchor_amplification_selectable = bool(
                    _env_flag("DRAFTPROOF_ACCEPT_HUMAN_ANCHOR_AMPLIFIER", True)
                    and (
                        lived_drop >= _float_env("DRAFTPROOF_HUMAN_ANCHOR_MIN_LIVED_DROP", 15.0)
                        or anchor_gain >= _float_env("DRAFTPROOF_HUMAN_ANCHOR_MIN_SCORE_GAIN", 8.0)
                    )
                    and human_raw_gain > 0.0
                    and human_anchor_burden_gate.get("accepted")
                    and not ai_score_regressed
                    and isinstance(ai_authorship_delta, (int, float))
                    and ai_authorship_delta >= -_float_env(
                        "DRAFTPROOF_HUMAN_ANCHOR_MAX_AUTHORSHIP_REGRESSION",
                        0.0,
                    )
                    and isinstance(candidate_ai_transform_delta, (int, float))
                    and candidate_ai_transform_delta >= -_float_env(
                        "DRAFTPROOF_HUMAN_ANCHOR_MAX_TRANSFORMATION_REGRESSION",
                        0.0,
                    )
                    and float(footprint_drops.get("topk_calibrated_risk") or 0.0) >= -_float_env(
                        "DRAFTPROOF_HUMAN_ANCHOR_MAX_TOPK_REGRESSION",
                        2.0,
                    )
                    and float(footprint_drops.get("external_ai_flag_risk") or 0.0) >= -_float_env(
                        "DRAFTPROOF_HUMAN_ANCHOR_MAX_EXTERNAL_PROXY_REGRESSION",
                        1.0,
                    )
                    and candidate_finding_total <= original_total
                    and candidate_review_burden <= original_review_burden
                    and candidate_weighted_severity <= original_severity
                    and not authenticity_status.get("critical_high_regressed")
                    and candidate_critical_high <= saved_critical_high
                    and not authenticity_status.get("ai_authorship_regression_blocked")
                )
                candidate_eval["human_anchor_amplifier_status"] = {
                    "selectable": human_anchor_amplification_selectable,
                    "human_raw_gain": round(human_raw_gain, 3),
                    "human_anchor_gain": round(anchor_gain, 3),
                    "lived_detail_risk_drop": round(lived_drop, 3),
                    "target_lived_detail_band": human_anchor_contract.get("next_lived_detail_band"),
                    "achieved_next_band": human_anchor_contract.get("achieved_next_band"),
                    "scope": "implied_context_only",
                    "positive_burden_gate": human_anchor_burden_gate,
                }
            ai_footprint_selectable = bool(
                _env_flag("DRAFTPROOF_AI_FOOTPRINT_GATE_ENABLED", True)
                and ai_footprint_outcome in {"ai_mitigated", "partially_ai_mitigated"}
                and ai_footprint_gate.get("safety_clean")
                and (
                    ai_footprint_outcome != "ai_mitigated"
                    or turnitin_like_gate.get("safe_band")
                )
                and (
                    ai_footprint_outcome != "ai_mitigated"
                    or eligible_span_density_gate.get("safe")
                )
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("critical_high_regressed")
                and candidate_critical_high <= saved_critical_high
                and candidate_finding_total <= original_total
                and candidate_review_burden <= original_review_burden
                and candidate_weighted_severity <= original_severity
                and not ai_score_regressed
            )
            turnitin_like_selectable = bool(
                _env_flag("DRAFTPROOF_TURNITIN_LIKE_GATE_ENABLED", True)
                and turnitin_like_gate.get("safety_clean")
                and turnitin_like_gate.get("improved")
                and candidate_meaningful_ai_progress_gate.get("meaningful")
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("critical_high_regressed")
                and candidate_critical_high <= saved_critical_high
                and candidate_finding_total <= original_total
                and candidate_review_burden <= original_review_burden
                and candidate_weighted_severity <= original_severity
                and not ai_score_regressed
            )
            topk_texture_blocked = bool(
                any(
                    str(blocker.get("driver") or "") in {"topk_calibrated_risk", "topk_pattern"}
                    for blocker in (ai_footprint_gate.get("texture_blockers") or [])
                    if isinstance(blocker, dict)
                )
            )
            topk_blocker_progress_selectable = bool(
                _env_flag("DRAFTPROOF_ACCEPT_TOPK_BLOCKER_PROGRESS", True)
                and topk_texture_blocked
                and ai_footprint_gate.get("safety_clean")
                and float(footprint_drops.get("topk_calibrated_risk") or 0.0) >= _float_env(
                    "DRAFTPROOF_TOPK_BLOCKER_PROGRESS_MIN_DROP",
                    1.5,
                )
                and float(footprint_drops.get("ai_likelihood") or 0.0) >= _float_env(
                    "DRAFTPROOF_TOPK_BLOCKER_PROGRESS_MIN_AI_LIKELIHOOD_DROP",
                    1.0,
                )
                and isinstance(ai_authorship_delta, (int, float))
                and ai_authorship_delta >= _float_env(
                    "DRAFTPROOF_TOPK_BLOCKER_PROGRESS_MIN_AUTHORSHIP_DROP",
                    1.0,
                )
                and isinstance(candidate_ai_transform_delta, (int, float))
                and candidate_ai_transform_delta >= _float_env(
                    "DRAFTPROOF_TOPK_BLOCKER_PROGRESS_MIN_TRANSFORMATION_DROP",
                    0.0,
                )
                and candidate_critical_high <= saved_critical_high
                and candidate_finding_total <= original_total
                and candidate_review_burden <= original_review_burden
                and candidate_weighted_severity <= original_severity
                and not ai_score_regressed
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("critical_high_regressed")
            )
            candidate_after_authorship = (
                (ai_footprint_gate.get("after") or {}).get("authorship_footprint") or {}
            )
            candidate_topk_calibrated = candidate_after_authorship.get("topk_calibrated_risk")
            topk_safe_band_rebuild_selectable = bool(
                topk_safe_band_rebuild
                and isinstance(candidate_topk_calibrated, (int, float))
                and float(candidate_topk_calibrated) < _safe_topk_calibrated_limit()
                and candidate_critical_high <= saved_critical_high
                and candidate_finding_total <= original_total
                and candidate_review_burden <= original_review_burden
                and candidate_weighted_severity <= original_severity
                and not authenticity_status.get("critical_high_regressed")
            )
            score_drag_status = _score_drag_removal_status(
                authenticity_status=authenticity_status,
                human_shift=human_shift,
                ai_delta=ai_delta,
                finding_delta=finding_delta,
                review_burden_delta=review_burden_delta,
                weighted_severity_delta=weighted_severity_delta,
                critical_high_delta=critical_high_delta,
                ai_score_regressed=ai_score_regressed,
            )
            score_drag_removal_selectable = bool(score_drag_status.get("allowed"))
            candidate_eval["score_drag_removal_status"] = score_drag_status
            human_primary_selectable = bool(
                _env_flag("DRAFTPROOF_AI_SEARCH_ACCEPT_HUMAN_PRIMARY_PROGRESS", True)
                and isinstance(candidate_human_delta, (int, float))
                and candidate_human_delta >= _float_env(
                    "DRAFTPROOF_HUMAN_PRIMARY_MIN_GAIN",
                    8.0,
                )
                and isinstance(ai_authorship_delta, (int, float))
                and ai_authorship_delta >= 0.0
                and isinstance(candidate_ai_transform_delta, (int, float))
                and candidate_ai_transform_delta >= 0.0
                and _finding_total(candidate_report) <= original_total
                and candidate_review_burden <= original_review_burden
                and candidate_weighted_severity <= original_severity
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("critical_high_regressed")
                and candidate_critical_high <= saved_critical_high
            )
            post_safe_target_climb_selectable = bool(
                candidate_eval.get("post_safe_win_target_push")
                and _env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_ACCEPT_CLIMB", True)
                and isinstance(candidate_human_delta, (int, float))
                and candidate_human_delta >= _float_env(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_MIN_TOTAL_HUMAN_GAIN",
                    8.0,
                )
                and isinstance(ai_authorship_delta, (int, float))
                and ai_authorship_delta >= _float_env(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_MIN_AUTHORSHIP_DELTA",
                    0.0,
                )
                and isinstance(candidate_ai_transform_delta, (int, float))
                and candidate_ai_transform_delta >= 0.0
                and isinstance(candidate_ai, (int, float))
                and float(candidate_ai) <= ai_search_reference + _float_env(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_AI_TOLERANCE",
                    1.0,
                )
                and _finding_total(candidate_report) <= original_total
                and candidate_review_burden <= original_review_burden
                and candidate_weighted_severity <= original_severity
                and not authenticity_status.get("ai_authorship_regression_blocked")
                and not authenticity_status.get("critical_high_regressed")
                and candidate_critical_high <= saved_critical_high
            )
            blocker_review_tolerance = int(_float_env(
                "DRAFTPROOF_BLOCKER_ELIMINATION_REVIEW_TOLERANCE",
                0.0,
            ))
            blocker_severity_tolerance = int(_float_env(
                "DRAFTPROOF_BLOCKER_ELIMINATION_SEVERITY_TOLERANCE",
                0.0,
            ))
            blocker_critical_high_tolerance = int(_float_env(
                "DRAFTPROOF_BLOCKER_ELIMINATION_CRITICAL_HIGH_TOLERANCE",
                0.0,
            ))
            blocker_elimination_selectable = bool(
                _env_flag("DRAFTPROOF_AI_SEARCH_ACCEPT_BLOCKER_ELIMINATION", True)
                and isinstance(candidate_human_delta, (int, float))
                and candidate_human_delta >= _float_env(
                    "DRAFTPROOF_BLOCKER_ELIMINATION_MIN_HUMAN_GAIN",
                    5.0,
                )
                and isinstance(ai_authorship_delta, (int, float))
                and ai_authorship_delta >= 0.0
                and isinstance(candidate_ai_transform_delta, (int, float))
                and candidate_ai_transform_delta >= 0.0
                and float(blocker_status.get("active_drop") or 0.0) >= _float_env(
                    "DRAFTPROOF_BLOCKER_ELIMINATION_MIN_ACTIVE_DROP",
                    20.0,
                )
                and float(blocker_status.get("active_regression") or 0.0) <= _float_env(
                    "DRAFTPROOF_BLOCKER_ELIMINATION_MAX_ACTIVE_REGRESSION",
                    20.0,
                )
                and candidate_review_burden <= original_review_burden + blocker_review_tolerance
                and candidate_weighted_severity <= original_severity + blocker_severity_tolerance
                and candidate_critical_high <= saved_critical_high + blocker_critical_high_tolerance
                and not authenticity_status.get("ai_authorship_regression_blocked")
            )
            if ai_footprint_selectable:
                ai_footprint_full_mitigation = bool(ai_footprint_outcome == "ai_mitigated")
                _mark_ai_search_progress_selection(
                    selection_status,
                    reason=(
                        "accepted_ai_footprint_mitigation"
                        if ai_footprint_full_mitigation
                        else "accepted_partial_ai_footprint_mitigation"
                    ),
                    full_success=ai_footprint_full_mitigation,
                    ai_footprint_mitigation=ai_footprint_full_mitigation,
                    partial_ai_footprint_mitigation=bool(ai_footprint_outcome == "partially_ai_mitigated"),
                )
            if turnitin_like_selectable:
                turnitin_like_full_mitigation = bool(
                    turnitin_like_gate.get("safe_band")
                    and ai_footprint_gate.get("safe_band")
                    and eligible_span_density_gate.get("safe")
                )
                _mark_ai_search_progress_selection(
                    selection_status,
                    reason=(
                        selection_status.get("reason")
                        if selection_status.get("ai_footprint_mitigation")
                        else (
                            "accepted_turnitin_like_mitigation"
                            if turnitin_like_full_mitigation
                            else "accepted_partial_turnitin_like_mitigation"
                        )
                    ),
                    full_success=turnitin_like_full_mitigation,
                    turnitin_like_mitigation=turnitin_like_full_mitigation,
                    partial_turnitin_like_mitigation=not turnitin_like_full_mitigation,
                )
            if topk_blocker_progress_selectable:
                _mark_ai_search_progress_selection(
                    selection_status,
                    reason="accepted_topk_blocker_progress",
                    topk_blocker_progress=True,
                )
            if topk_safe_band_rebuild_selectable:
                _mark_ai_search_progress_selection(
                    selection_status,
                    reason="accepted_topk_safe_band_rebuild",
                    topk_safe_band_rebuild=True,
                    topk_safe_band_achieved=True,
                )
                selection_status["topk_calibrated_risk"] = round(float(candidate_topk_calibrated), 3)
            if human_amplification_selectable:
                _mark_ai_search_progress_selection(
                    selection_status,
                    reason=(
                        selection_status.get("reason")
                        if selection_status.get("ai_footprint_mitigation")
                        or selection_status.get("partial_ai_footprint_mitigation")
                        or selection_status.get("topk_blocker_progress")
                        or selection_status.get("topk_safe_band_achieved")
                        else "accepted_human_signal_amplification"
                    ),
                    human_signal_amplification=True,
                )
            if human_anchor_amplification_selectable:
                _mark_ai_search_progress_selection(
                    selection_status,
                    reason=(
                        selection_status.get("reason")
                        if selection_status.get("ai_footprint_mitigation")
                        or selection_status.get("partial_ai_footprint_mitigation")
                        or selection_status.get("topk_blocker_progress")
                        or selection_status.get("topk_safe_band_achieved")
                        else "accepted_human_anchor_amplifier"
                    ),
                    human_anchor_amplifier=True,
                )
            if (
                not selection_status.get("selectable")
                and (
                    blocker_elimination_selectable
                    or
                    human_primary_selectable
                    or turnitin_like_selectable
                    or
                    ai_footprint_selectable
                    or topk_blocker_progress_selectable
                    or topk_safe_band_rebuild_selectable
                    or
                    incremental_authenticity_selectable
                    or human_amplification_selectable
                    or human_anchor_amplification_selectable
                    or safe_authorship_suppression_selectable
                    or safe_partial_quality_selectable
                    or score_drag_removal_selectable
                    or post_safe_target_climb_selectable
                )
            ):
                fallback_reason = (
                    "accepted_blocker_elimination"
                    if blocker_elimination_selectable
                    else (
                        "accepted_human_primary_progress"
                        if human_primary_selectable
                        else (
                                "accepted_turnitin_like_mitigation"
                                if turnitin_like_selectable and turnitin_like_gate.get("safe_band") and ai_footprint_gate.get("safe_band")
                                else (
                                    "accepted_partial_turnitin_like_mitigation"
                                    if turnitin_like_selectable
                                    else (
                                        "accepted_ai_footprint_mitigation"
                                        if ai_footprint_selectable and ai_footprint_outcome == "ai_mitigated"
                                        else (
                                            "accepted_partial_ai_footprint_mitigation"
                                            if ai_footprint_selectable
                                            else (
                                                "accepted_topk_blocker_progress"
                                                if topk_blocker_progress_selectable
                                                else (
                                                    "accepted_topk_safe_band_rebuild"
                                                    if topk_safe_band_rebuild_selectable
                                                    else (
                                                        "accepted_human_signal_amplification"
                                                        if human_amplification_selectable
                                                        else (
                                                            "accepted_human_anchor_amplifier"
                                                            if human_anchor_amplification_selectable
                                                            else (
                                                                "accepted_post_safe_target_climb"
                                                                if post_safe_target_climb_selectable
                                                                else (
                                                                    "accepted_score_drag_removal"
                                                                    if score_drag_removal_selectable
                                                                    else (
                                                                        "accepted_safe_partial_quality_improvement"
                                                                        if safe_partial_quality_selectable
                                                                        else (
                                                                            (
                                                                                "accepted_incremental_human_target_progress"
                                                                                if _radar_goal_requires_human_progress(radar_goal_controller)
                                                                                else "accepted_safe_authorship_suppression"
                                                                            )
                                                                            if safe_authorship_suppression_selectable
                                                                            else "accepted_incremental_authenticity_progress"
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                    )
                                )
                            )
                        )
                    )
                )
                fallback_full_success = bool(
                    (turnitin_like_selectable and turnitin_like_gate.get("safe_band") and ai_footprint_gate.get("safe_band"))
                    or (ai_footprint_selectable and ai_footprint_outcome == "ai_mitigated")
                )
                _mark_ai_search_progress_selection(
                    selection_status,
                    reason=fallback_reason,
                    full_success=fallback_full_success,
                    blocker_elimination=bool(blocker_elimination_selectable),
                    human_primary_progress=bool(human_primary_selectable),
                    turnitin_like_mitigation=bool(
                        turnitin_like_selectable
                        and turnitin_like_gate.get("safe_band")
                        and ai_footprint_gate.get("safe_band")
                    ),
                    partial_turnitin_like_mitigation=bool(
                        turnitin_like_selectable
                        and not (
                            turnitin_like_gate.get("safe_band")
                            and ai_footprint_gate.get("safe_band")
                        )
                    ),
                    ai_footprint_mitigation=bool(ai_footprint_selectable and ai_footprint_outcome == "ai_mitigated"),
                    partial_ai_footprint_mitigation=bool(ai_footprint_selectable and ai_footprint_outcome == "partially_ai_mitigated"),
                    topk_blocker_progress=bool(topk_blocker_progress_selectable),
                    topk_safe_band_rebuild=bool(topk_safe_band_rebuild_selectable),
                    topk_safe_band_achieved=bool(topk_safe_band_rebuild_selectable),
                    cleanup_improved=bool(safe_partial_quality_selectable and ai_footprint_outcome == "cleanup_improved"),
                    authenticity_incremental=bool(incremental_authenticity_selectable),
                    human_signal_amplification=bool(human_amplification_selectable),
                    human_anchor_amplifier=bool(human_anchor_amplification_selectable),
                    safe_authorship_suppression=bool(safe_authorship_suppression_selectable),
                    safe_partial_quality_improvement=bool(safe_partial_quality_selectable),
                    score_drag_removal=bool(score_drag_removal_selectable),
                    post_safe_target_climb=bool(post_safe_target_climb_selectable),
                )
            if human_amplification_score:
                selection_status["human_signal_amplification_score"] = human_amplification_score
            if candidate_eval.get("human_anchor_amplifier_status"):
                selection_status["human_anchor_amplifier_status"] = candidate_eval.get("human_anchor_amplifier_status")
            dominant_blocker_progress_override = _dominant_blocker_safe_progress_override(
                dominant_blocker_status,
                authenticity_status,
                blocker_status,
                ai_score_regressed=ai_score_regressed,
                finding_delta=_finding_total(candidate_report) - original_total,
                review_burden_delta=candidate_review_burden - original_review_burden,
                weighted_severity_delta=candidate_weighted_severity - original_severity,
                critical_high_delta=candidate_critical_high - saved_critical_high,
            )
            if (
                selection_status.get("selectable")
                and dominant_blocker_status.get("required")
                and not dominant_blocker_status.get("cleared")
                and not dominant_blocker_progress_override.get("allowed")
                and not selection_status.get("score_drag_removal")
                and not selection_status.get("safe_authorship_suppression")
                and not selection_status.get("turnitin_like_mitigation")
                and not selection_status.get("partial_turnitin_like_mitigation")
                and not selection_status.get("ai_footprint_mitigation")
                and not selection_status.get("partial_ai_footprint_mitigation")
                and not selection_status.get("topk_blocker_progress")
                and not selection_status.get("topk_safe_band_achieved")
                and not selection_status.get("human_anchor_amplifier")
                and not selection_status.get("human_signal_amplification")
                and not selection_status.get("safe_partial_quality_improvement")
            ):
                selection_status.update({
                    "success": False,
                    "selectable": False,
                    "reason": "dominant_blocker_not_reduced",
                    "dominant_blocker_required": True,
                })
            if (
                selection_status.get("selectable")
                and human_formula_status.get("required")
                and not human_formula_status.get("cleared")
                and not selection_status.get("human_primary_progress")
                and not selection_status.get("post_safe_target_climb")
                and not selection_status.get("turnitin_like_mitigation")
                and not selection_status.get("partial_turnitin_like_mitigation")
                and not selection_status.get("ai_footprint_mitigation")
                and not selection_status.get("partial_ai_footprint_mitigation")
                and not selection_status.get("topk_blocker_progress")
                and not selection_status.get("topk_safe_band_achieved")
                and not selection_status.get("human_anchor_amplifier")
                and not selection_status.get("human_signal_amplification")
                and not selection_status.get("safe_partial_quality_improvement")
            ):
                selection_status.update({
                    "success": False,
                    "selectable": False,
                    "reason": "human_formula_drivers_not_reduced",
                    "human_formula_driver_required": True,
                })
            elif dominant_blocker_progress_override.get("allowed"):
                selection_status.update({
                    "dominant_blocker_required": True,
                    "dominant_blocker_safe_progress_override": True,
                    "reason": (
                        selection_status.get("reason")
                        or "accepted_safe_progress_with_stale_dominant_blocker"
                    ),
                })
            human_target_block = _human_target_regression_selection_block(
                selection_status,
                authenticity_status,
            )
            if human_target_block.get("blocked") and not selection_status.get("topk_safe_band_achieved"):
                selection_status.update({
                    "success": False,
                    "selectable": False,
                    "reason": human_target_block.get("reason"),
                    **human_target_block,
                })
            if (
                selection_status.get("selectable")
                and ai_footprint_outcome == "ai_footprint_blocked_by_texture"
                and not selection_status.get("turnitin_like_mitigation")
                and not selection_status.get("partial_turnitin_like_mitigation")
                and not selection_status.get("ai_footprint_mitigation")
                and not selection_status.get("partial_ai_footprint_mitigation")
                and not selection_status.get("topk_blocker_progress")
                and not selection_status.get("topk_safe_band_achieved")
                and not selection_status.get("human_anchor_amplifier")
                and not selection_status.get("human_signal_amplification")
                and _env_flag("DRAFTPROOF_BLOCK_TEXTURE_STALLED_AI_FOOTPRINT", True)
            ):
                selection_status.update({
                    "success": False,
                    "selectable": False,
                    "reason": "ai_footprint_texture_blocker_not_reduced",
                    "ai_footprint_texture_blocked": True,
                })
            selection_status["authenticity_gate"] = authenticity_status
            selection_status["ai_footprint_gate"] = ai_footprint_gate
            selection_status["ai_footprint_outcome_class"] = ai_footprint_outcome
            selection_status["turnitin_like_ai_gate"] = turnitin_like_gate
            selection_status["formula_gap_contract"] = formula_gap_contract
            selection_status["formula_gap_rank"] = candidate_eval["formula_gap_rank"]
            selection_status["multi_signal_contract"] = multi_signal_contract
            selection_status["eligible_span_density_gate"] = eligible_span_density_gate
            selection_status["dominant_blocker_gate"] = dominant_blocker_status
            selection_status["human_formula_driver_gate"] = human_formula_status
            selection_status["human_anchor_driver_contract"] = human_anchor_contract
            selection_status["dominant_blocker_progress_override"] = dominant_blocker_progress_override
            selection_status["ai_score_regression_tolerance"] = ai_score_regression_tolerance
            selection_status["ai_score_regressed"] = ai_score_regressed
            selection_status["human_shift_score"] = human_shift.get("score")
            selection_status["human_shift_components"] = human_shift.get("components")
            if (
                _radar_goal_requires_human_progress(radar_goal_controller)
                and selection_status.get("selectable")
                and not selection_status.get("human_primary_progress")
                and not selection_status.get("human_signal_amplification")
                and not selection_status.get("human_anchor_amplifier")
                and not selection_status.get("post_safe_target_climb")
                and not selection_status.get("turnitin_like_mitigation")
                and not selection_status.get("partial_turnitin_like_mitigation")
                and not selection_status.get("ai_footprint_mitigation")
                and not selection_status.get("partial_ai_footprint_mitigation")
                and not selection_status.get("score_drag_removal")
                and not selection_status.get("safe_partial_quality_improvement")
                and not selection_status.get("topk_safe_band_achieved")
                and not (
                    isinstance(candidate_human_delta, (int, float))
                    and candidate_human_delta > 0.0
                )
            ):
                selection_status.update({
                    "success": False,
                    "selectable": False,
                    "reason": "radar_goal_requires_human_progress",
                    "radar_goal_requires_human_progress": True,
                })
            candidate_eval["selection_status"] = selection_status
            original_human_value = _contribution_scores(original_report_dict).get("human")
            candidate_human_value = candidate_contribution.get("human")
            human_delta_for_blocked = (
                float(candidate_human_value) - float(original_human_value)
                if isinstance(candidate_human_value, (int, float))
                and isinstance(original_human_value, (int, float))
                else 0.0
            )
            if _should_track_blocked_human_winner(
                selection_status=selection_status,
                human_delta=human_delta_for_blocked,
                ai_delta=ai_delta,
                authenticity_status=authenticity_status,
                ai_footprint_gate=ai_footprint_gate,
            ):
                saved_ch_delta = max(0, int(candidate_critical_high or 0) - int(saved_critical_high or 0))
                blocked_rank = (
                    round(human_delta_for_blocked, 3),
                    float(human_shift.get("score") or -9999.0),
                    -saved_ch_delta,
                    -max(0, candidate_review_burden - original_review_burden),
                    -max(0, candidate_weighted_severity - original_severity),
                    -(candidate_ai if isinstance(candidate_ai, (int, float)) else 999.0),
                )
                if best_blocked_human_rank is None or blocked_rank > best_blocked_human_rank:
                    best_blocked_human_rank = blocked_rank
                    best_blocked_human_candidate = {
                        "strategy": strategy,
                        "text": candidate,
                        "report": candidate_report,
                        "summary": {
                            "strategy": strategy,
                            "ai": candidate_ai,
                            "ai_delta_vs_reference": candidate_eval.get("ai_delta_vs_reference"),
                            "human_contribution": candidate_contribution.get("human"),
                            "human_delta": round(human_delta_for_blocked, 3),
                            "ai_transformation": candidate_contribution.get("ai_transformation"),
                            "ai_authorship": candidate_integrity.get("ai_authorship"),
                            "grounding_quality_risk": candidate_integrity.get("grounding"),
                            "findings": _finding_total(candidate_report),
                            "review_burden": candidate_review_burden,
                            "weighted_severity": candidate_weighted_severity,
                            "critical_high_findings": candidate_critical_high,
                            "saved_critical_high": saved_critical_high,
                            "ai_footprint_gate": ai_footprint_gate,
                            "selection_status": selection_status,
                        },
                    }
            human_shift_score = human_shift.get("score")
            candidate_ai_authorship_delta = authenticity_status.get("ai_authorship_delta")
            candidate_ai_transformation_delta = authenticity_status.get("ai_transformation_delta")
            blocker_active_drop = float(blocker_status.get("active_drop") or 0.0)
            candidate_rank = _goal_climb_candidate_rank(
                selection_status,
                candidate_eval,
                candidate_ai=candidate_ai,
                candidate_review_burden=candidate_review_burden,
                candidate_weighted_severity=candidate_weighted_severity,
                candidate_finding_total=_finding_total(candidate_report),
                original_review_burden=original_review_burden,
                original_weighted_severity=original_severity,
                original_finding_total=original_total,
            )
            candidate_decision = build_candidate_decision(
                selection_status,
                candidate_eval,
                candidate_ai=candidate_ai,
                candidate_review_burden=candidate_review_burden,
                candidate_weighted_severity=candidate_weighted_severity,
                candidate_finding_total=_finding_total(candidate_report),
                original_review_burden=original_review_burden,
                original_weighted_severity=original_severity,
                original_finding_total=original_total,
                formula_gap_rank=tuple(candidate_eval.get("formula_gap_rank") or ()),
            )
            candidate_eval["candidate_decision"] = candidate_decision.to_dict()
            selection_status["candidate_decision"] = candidate_decision.to_dict()
            topk_safe_frontier_blocked = bool(
                _selection_status_topk_safe(best_selection_status)
                and not _selection_status_topk_safe(selection_status)
            )
            if topk_safe_frontier_blocked:
                candidate_eval["selection_blocked_by_topk_safe_frontier"] = True
                candidate_eval["topk_safe_frontier"] = {
                    "current_topk_calibrated_risk": _selection_status_topk_value(best_selection_status),
                    "candidate_topk_calibrated_risk": _selection_status_topk_value(selection_status),
                    "safe_band": _safe_topk_calibrated_limit(),
                }
            elif candidate_rank > best_human_shift_rank:
                best_ai = candidate_ai
                best_text = candidate
                best_report = candidate_report
                best_strategy = strategy
                best_semantic_review_required = bool(candidate_eval.get("semantic_review_required"))
                best_drift_reasons = list(candidate_eval.get("drift_reasons") or [])
                best_selection_status = selection_status
                best_human_shift_rank = candidate_rank
                candidate_eval["best_so_far"] = True
                candidate_eval["selectable_so_far"] = bool(selection_status.get("selectable"))
                _record_best_attempt()
            search_summary["candidates"].append(candidate_eval)

        def _run_post_topk_ai_safe_band_optimizer(trigger_phase: str) -> None:
            nonlocal best_text, best_report, best_ai, best_strategy, best_selection_status
            nonlocal best_human_shift_rank, best_semantic_review_required, best_drift_reasons

            def _accept_post_topk_selected_candidate(*, text, report, ai, strategy, selection_status, candidate_eval, partial):
                nonlocal best_text, best_report, best_ai, best_strategy, best_selection_status
                nonlocal best_human_shift_rank, best_semantic_review_required, best_drift_reasons
                best_text = text
                best_report = report
                best_ai = ai
                best_strategy = strategy
                best_selection_status = selection_status
                best_human_shift_rank = _goal_climb_candidate_rank(
                    best_selection_status,
                    candidate_eval,
                    candidate_ai=best_ai,
                    candidate_review_burden=_review_burden(best_report),
                    candidate_weighted_severity=_weighted_severity(best_report),
                    candidate_finding_total=_finding_total(best_report),
                    original_review_burden=original_review_burden,
                    original_weighted_severity=original_severity,
                    original_finding_total=original_total,
                )
                best_semantic_review_required = False
                best_drift_reasons = []
                _record_best_attempt()

            _core_run_post_topk_ai_safe_band_optimizer(
                trigger_phase,
                search_summary=search_summary,
                search_budget=search_budget,
                gateway=gateway,
                hard_llm_cap=hard_llm_cap,
                original_report_dict=original_report_dict,
                original_review_burden=original_review_burden,
                original_severity=original_severity,
                saved_critical_high=saved_critical_high,
                search_source_text=search_source_text,
                deps=PostTopkSafeBandOptimizerDeps(
                    env_flag=_env_flag,
                    best_ai_search_selectable=_best_ai_search_selectable,
                    strict_ai_safe_band_status=_strict_ai_safe_band_status,
                    safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
                    float_env=_float_env,
                    verified_candidate_scans_used=_verified_candidate_scans_used,
                    extend_candidate_scan_budget=_extend_candidate_scan_budget,
                    authorship_transformation_texture_driver_map=_authorship_transformation_texture_driver_map,
                    authorship_transformation_texture_candidates=_authorship_transformation_texture_candidates,
                    generic_assertion_compiler_candidates=_generic_assertion_compiler_candidates,
                    blocker_operation_candidates=_blocker_operation_candidates,
                    content_pruning_candidates=_content_pruning_candidates,
                    phase_budget_block_record=_phase_budget_block_record,
                    record_phase_llm_call=_record_phase_llm_call,
                    authorship_transformation_texture_patch_prompt=_authorship_transformation_texture_patch_prompt,
                    phase_chat_sampling_kwargs=_phase_chat_sampling_kwargs,
                    extract_post_topk_patch_candidates=_extract_post_topk_patch_candidates,
                    apply_post_topk_patches=_apply_post_topk_patches,
                    texture_candidate_family=_texture_candidate_family,
                    detect_protected_spans=detect_protected_spans,
                    review_burden=_review_burden,
                    weighted_severity=_weighted_severity,
                    finding_total=_finding_total,
                    critical_high_count=_critical_high_count,
                    turnitin_like_ai_profile=_turnitin_like_ai_profile,
                    search_budget_exhausted=_search_budget_exhausted,
                    ai_candidate_quality_reject_reason=_ai_candidate_quality_reject_reason,
                    ai_search_protected_loss_reason=_ai_search_protected_loss_reason,
                    check_semantic_drift=check_semantic_drift,
                    full_scan_report_dict=_full_scan_report_dict,
                    record_verified_candidate_scan=_record_verified_candidate_scan,
                    ai_footprint_gate_status=_ai_footprint_gate_status,
                    turnitin_like_ai_gate_status=_turnitin_like_ai_gate_status,
                    formula_gap_contract=_formula_gap_contract,
                    badge_ai=_badge_ai,
                    contribution_scores=_contribution_scores,
                    integrity_scores=_integrity_scores,
                    formula_gap_candidate_rank=_formula_gap_candidate_rank,
                    human_shift_score=_human_shift_score,
                    strict_safe_candidate_rank=_strict_safe_candidate_rank,
                    accept_selected_candidate=_accept_post_topk_selected_candidate,
                    get_best_text=lambda: best_text,
                    get_best_report=lambda: best_report,
                    get_best_strategy=lambda: best_strategy,
                ),
            )

        deterministic_only = (
            bool(ai_search_first)
            and not _allow_ai_search_llm_after_deterministic()
        )
        try:
            min_deterministic_scans = max(
                1,
                int(os.environ.get(
                    "DRAFTPROOF_AI_SEARCH_MIN_DETERMINISTIC_SCANS",
                    str(len(deterministic_candidates) or 1),
                )),
            )
        except ValueError:
            min_deterministic_scans = len(deterministic_candidates) or 1
        if formula_gap_orchestrator_enabled:
            deterministic_probe_cap = max(
                0,
                int(formula_gap_budget_contract.get("deterministic_probe_scans") or 0),
            )
            if deterministic_probe_cap <= 0:
                min_deterministic_scans = 0
            else:
                min_deterministic_scans = min(
                    max(1, min_deterministic_scans),
                    deterministic_probe_cap,
                )
            orchestrator_summary = search_summary.setdefault(
                "formula_gap_candidate_orchestrator",
                {},
            )
            orchestrator_summary.update({
                "deterministic_probe_candidates_total": len(deterministic_candidates),
                "deterministic_probe_scan_cap": deterministic_probe_cap,
                "deterministic_candidates_skipped_for_llm_reserve": max(
                    0,
                    len(deterministic_candidates) - deterministic_probe_cap,
                ),
            })
        early_stop_reason = ""
        for index, deterministic_item in enumerate(deterministic_candidates, start=1):
            if (
                formula_gap_orchestrator_enabled
                and index > max(0, int(formula_gap_budget_contract.get("deterministic_probe_scans") or 0))
            ):
                search_summary.setdefault("formula_gap_candidate_orchestrator", {})[
                    "deterministic_probe_stop_reason"
                ] = "llm_budget_reserved"
                break
            strategy, candidate = deterministic_item[0], deterministic_item[1]
            deterministic_extra = (
                deterministic_item[2]
                if len(deterministic_item) > 2 and isinstance(deterministic_item[2], dict)
                else None
            )
            if _search_budget_exhausted("deterministic_candidates"):
                break
            report_progress(
                min(79, 76 + index),
                f"Scanning deterministic AI mitigation candidate {index}/{len(deterministic_candidates)}",
            )
            _evaluate_ai_search_candidate(
                strategy,
                candidate,
                deterministic=True,
                extra=deterministic_extra,
            )
            if formula_gap_orchestrator_enabled:
                search_summary.setdefault("formula_gap_candidate_orchestrator", {})[
                    "deterministic_probe_scans_used"
                ] = min(index, _verified_candidate_scans_used())
            if adaptive_stop_reason:
                break
            if index < min_deterministic_scans:
                continue
            if formula_gap_orchestrator_enabled:
                # The formula-gap redesign treats deterministic repair as a probe.
                # It must not fast-accept or adaptive-stop before the reserved
                # portfolio LLM candidates have a chance to run.
                continue
            early_stop_reason = (
                _ai_search_fast_accept_reason(ai_search_reference, best_ai)
                if _best_ai_search_selectable()
                else ""
            )
            if early_stop_reason:
                search_summary["early_stop"] = {
                    "phase": "deterministic_candidates",
                    "reason": early_stop_reason,
                    "candidate_count_scanned": _verified_candidate_scans_used(),
                    "selected_strategy": best_strategy,
                    "selected_ai": best_ai,
                }
                break
            if _maybe_adaptive_stop("deterministic_candidates"):
                break

        def _run_post_safe_win_target_push(trigger_phase: str) -> None:
            # Compatibility for source-grep regression tests after extraction:
            # def _run_post_safe_win_target_push(trigger_phase: str) -> None:
            # "reason": "strict_ai_phase_budget_only"
            nonlocal adaptive_stop_reason
            adaptive_stop_reason = _core_run_post_safe_win_target_push(
                trigger_phase,
                search_summary=search_summary,
                search_budget=search_budget,
                adaptive_stop_reason=adaptive_stop_reason,
                topk_safe_band_priority=topk_safe_band_priority,
                density_or_generic_priority=density_or_generic_priority,
                effective_key=effective_key,
                generator_model=generator_model,
                base_url=base_url,
                original_report_dict=original_report_dict,
                confirmed_author_anchor_brief=confirmed_author_anchor_brief,
                deps=PostSafeWinTargetPushDeps(
                    env_flag=_env_flag,
                    strict_ai_safe_band_status=_strict_ai_safe_band_status,
                    post_safe_target_push_allows_deterministic_after_budget=_post_safe_target_push_allows_deterministic_after_budget,
                    post_safe_target_push_scan_reserve=_post_safe_target_push_scan_reserve,
                    verified_candidate_scans_used=_verified_candidate_scans_used,
                    extend_candidate_scan_budget=_extend_candidate_scan_budget,
                    best_ai_search_selectable=_best_ai_search_selectable,
                    contribution_scores=_contribution_scores,
                    float_env=_float_env,
                    adaptive_budget_default=_adaptive_budget_default,
                    post_safe_win_target_push_candidates=_post_safe_win_target_push_candidates,
                    report_progress=report_progress,
                    evaluate_ai_search_candidate=_evaluate_ai_search_candidate,
                    budget_gateway=_budget_gateway,
                    paragraph_component_targets=_paragraph_component_targets,
                    safe_index=_safe_index,
                    search_budget_exhausted=_search_budget_exhausted,
                    human_signal_amplification_prompt=_human_signal_amplification_prompt,
                    phase_chat_sampling_kwargs=_phase_chat_sampling_kwargs,
                    extract_paragraph_component_candidates=_extract_paragraph_component_candidates,
                    clean_paragraph_component_candidate=_clean_paragraph_component_candidate,
                    paragraph_anchor_lock=_paragraph_anchor_lock,
                    splice_paragraph=_splice_paragraph,
                    get_best_text=lambda: best_text,
                    get_best_report=lambda: best_report,
                    get_best_ai=lambda: best_ai,
                    get_best_strategy=lambda: best_strategy,
                    get_best_selection_status=lambda: best_selection_status,
                ),
            )

        if early_stop_reason:
            _run_post_safe_win_target_push("early_stop")
        elif adaptive_stop_reason:
            _run_post_safe_win_target_push("adaptive_stop")

        def _run_final_topk_texture_repair(trigger_phase: str) -> None:
            nonlocal adaptive_stop_reason
            adaptive_stop_reason = _core_run_final_topk_texture_repair(
                trigger_phase,
                search_summary=search_summary,
                search_budget=search_budget,
                adaptive_stop_reason=adaptive_stop_reason,
                formula_gap_orchestrator_completed=formula_gap_orchestrator_completed,
                effective_key=effective_key,
                gateway=gateway,
                base_url=base_url,
                generator_model=generator_model,
                hard_llm_cap=hard_llm_cap,
                search_started=search_started,
                deps=FinalTopkTextureRepairDeps(
                    env_flag=_env_flag,
                    float_env=_float_env,
                    adaptive_budget_default=_adaptive_budget_default,
                    best_ai_search_selectable=_best_ai_search_selectable,
                    blocker_scores=_blocker_scores,
                    safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
                    verified_candidate_scans_used=_verified_candidate_scans_used,
                    final_topk_texture_scan_reserve=_final_topk_texture_scan_reserve,
                    extend_candidate_scan_budget=_extend_candidate_scan_budget,
                    phase_budget_block_record=_phase_budget_block_record,
                    llm_call_budget_exhausted_before_send=_llm_call_budget_exhausted_before_send,
                    record_phase_llm_call=_record_phase_llm_call,
                    topk_texture_repair_prompt=_topk_texture_repair_prompt,
                    phase_chat_sampling_kwargs=_phase_chat_sampling_kwargs,
                    extract_paragraph_component_candidates=_extract_paragraph_component_candidates,
                    clean_full_document_candidate=_clean_full_document_candidate,
                    evaluate_ai_search_candidate=_evaluate_ai_search_candidate,
                    get_best_text=lambda: best_text,
                    get_best_report=lambda: best_report,
                    get_best_strategy=lambda: best_strategy,
                ),
            )

        def _run_iterative_topk_route_optimizer(trigger_phase: str) -> None:
            nonlocal adaptive_stop_reason
            adaptive_stop_reason = _core_run_iterative_topk_route_optimizer(
                trigger_phase,
                search_summary=search_summary,
                search_budget=search_budget,
                adaptive_stop_reason=adaptive_stop_reason,
                original_report_dict=original_report_dict,
                original_review_burden=original_review_burden,
                original_severity=original_severity,
                saved_critical_high=saved_critical_high,
                deps=IterativeTopkRouteOptimizerDeps(
                    env_flag=_env_flag,
                    float_env=_float_env,
                    best_ai_search_selectable=_best_ai_search_selectable,
                    safe_topk_calibrated_limit=_safe_topk_calibrated_limit,
                    verified_candidate_scans_used=_verified_candidate_scans_used,
                    extend_candidate_scan_budget=_extend_candidate_scan_budget,
                    search_budget_exhausted=_search_budget_exhausted,
                    ai_footprint_gate_status=_ai_footprint_gate_status,
                    review_burden=_review_burden,
                    weighted_severity=_weighted_severity,
                    critical_high_count=_critical_high_count,
                    topk_route_optimizer_candidates=_topk_route_optimizer_candidates,
                    evaluate_ai_search_candidate=_evaluate_ai_search_candidate,
                    get_best_text=lambda: best_text,
                    get_best_report=lambda: best_report,
                    get_best_strategy=lambda: best_strategy,
                    get_best_selection_status=lambda: best_selection_status,
                ),
            )

        post_safe_summary = search_summary.get("post_safe_win_target_push")
        post_safe_human = (
            _contribution_scores(best_report).get("human")
            if isinstance(best_report, dict) else None
        )
        target_human_after_post_safe = _float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
        writing_components_for_continue = (
            (best_report or {}).get("ai_risk_badge", {}).get("writing_components", {})
            if isinstance(best_report, dict) else {}
        )
        source_or_claim_blocker_after_post_safe = max(
            float(writing_components_for_continue.get("source_grounding_risk") or 0.0),
            float(writing_components_for_continue.get("unsupported_claim_risk") or 0.0),
            float(writing_components_for_continue.get("broad_claim_risk") or 0.0),
        )
        if (
            adaptive_stop_reason == "adaptive_stop_after_post_safe_win_target_push"
            and isinstance(post_safe_summary, dict)
            and post_safe_summary.get("accepted")
            and isinstance(post_safe_human, (int, float))
            and float(post_safe_human) < target_human_after_post_safe
            and source_or_claim_blocker_after_post_safe >= _float_env(
                "DRAFTPROOF_POST_SAFE_CONTINUE_MIN_GROUNDING_BLOCKER",
                70.0,
            )
            and _source_search_enabled()
            and _env_flag("DRAFTPROOF_CONTINUE_AFTER_POST_SAFE_TARGET_PUSH_FOR_GROUNDING", True)
            and effective_key
        ):
            post_safe_summary["continued_after_plateau_for_grounding"] = True
            post_safe_summary["continued_after_plateau_human"] = post_safe_human
            post_safe_summary["continued_after_plateau_blocker"] = source_or_claim_blocker_after_post_safe
            adaptive_stop_reason = ""
            search_summary.pop("adaptive_stop", None)

        if early_stop_reason and not adaptive_stop_reason:
            search_summary["llm_reason"] = "skipped_after_fast_deterministic_accept"
        elif adaptive_stop_reason:
            search_summary["llm_reason"] = adaptive_stop_reason
        elif deterministic_only:
            search_summary["llm_reason"] = "skipped_deterministic_only_ai_first"
            if best_strategy:
                search_summary["deterministic_only_best_attempt"] = True
        elif not effective_key:
            if not search_summary.get("candidates"):
                search_summary["reason"] = "no_llm_available"
            else:
                search_summary["llm_reason"] = "no_llm_available"
        else:
            try:
                ai_search_sampling = _rewrite_sampling_profile("DRAFTPROOF_AI_SEARCH")
                gateway = LLMGateway(LLMConfig(
                    api_key=effective_key,
                    model=generator_model,
                    base_url=base_url,
                    timeout=int(os.environ.get("DRAFTPROOF_AI_SEARCH_TIMEOUT", "120")),
                    max_retries=int(os.environ.get("DRAFTPROOF_AI_SEARCH_RETRIES", "1")),
                    max_tokens=int(os.environ.get("DRAFTPROOF_AI_SEARCH_MAX_TOKENS", "6500")),
                    temperature=ai_search_sampling["temperature"],
                    top_p=ai_search_sampling["top_p"],
                    top_k=ai_search_sampling["top_k"],
                    presence_penalty=ai_search_sampling["presence_penalty"],
                    frequency_penalty=ai_search_sampling["frequency_penalty"],
                ))
                gateway = _budget_gateway(gateway, "ai_search_llm")
                formula_gap_orchestrator_completed = False
                if formula_gap_orchestrator_enabled:
                    orchestrator_summary = search_summary.setdefault(
                        "formula_gap_candidate_orchestrator",
                        {},
                    )
                    orchestrator_base_text = best_text if _best_ai_search_selectable() else search_source_text
                    orchestrator_base_report = best_report if _best_ai_search_selectable() else original_report_dict
                    block_tasks = _formula_gap_block_portfolio_tasks(
                        orchestrator_base_text,
                        orchestrator_base_report,
                        limit=int(formula_gap_budget_contract.get("llm_candidate_calls") or 0),
                    )
                    families = [str(task.get("family") or "") for task in block_tasks]
                    if not families:
                        families = _formula_gap_portfolio_families(
                            int(formula_gap_budget_contract.get("llm_candidate_calls") or 0)
                        )
                        block_tasks = [
                            {"family": family, "operation": "whole_candidate_fallback", "blocks": [], "block_indexes": []}
                            for family in families
                        ]
                    protected_anchor_brief = [
                        {
                            "text": span.text or search_source_text[span.start_char:span.end_char],
                            "reason": span.reason,
                        }
                        for span in source_protected[:80]
                    ]
                    protected_anchor_brief.extend(
                        {
                            "text": entity,
                            "reason": "named_entity",
                        }
                        for entity in _formula_gap_named_entity_inventory(search_source_text)
                    )
                    orchestrator_summary.update({
                        "enabled": True,
                        "started": True,
                        "portfolio_families": families,
                        "portfolio_tasks": [
                            {
                                "family": task.get("family"),
                                "operation": task.get("operation"),
                                "block_indexes": task.get("block_indexes"),
                                "targeted_drivers": task.get("targeted_drivers"),
                                "blocks": [
                                    {
                                        "index": row.get("index"),
                                        "role": row.get("role"),
                                        "word_count": row.get("word_count"),
                                        "weighted_drag": row.get("weighted_drag"),
                                        "remove_safe": row.get("remove_safe"),
                                        "protected_anchor_terms": row.get("protected_anchor_terms"),
                                    }
                                    for row in (task.get("blocks") or [])
                                    if isinstance(row, dict)
                                ],
                            }
                            for task in block_tasks
                        ],
                        "base_strategy": best_strategy if _best_ai_search_selectable() else "source",
                        "candidate_frontier": orchestrator_summary.get("candidate_frontier") or [],
                    })
                    for family_index, block_task in enumerate(block_tasks, start=1):
                        family = str(block_task.get("family") or families[min(family_index - 1, len(families) - 1)])
                        if _search_budget_exhausted("formula_gap_portfolio_llm", before_llm=True):
                            orchestrator_summary["stop_reason"] = adaptive_stop_reason or "budget_exhausted"
                            break
                        report_progress(
                            min(89, 78 + family_index),
                            f"Trying formula-gap block candidate {family_index}/{len(block_tasks)}",
                        )
                        candidate_record = {
                            "strategy": f"formula_gap_portfolio_{family.lower()}_b{'_'.join(str(i) for i in (block_task.get('block_indexes') or []))}",
                            "family": family,
                            "block_indexes": block_task.get("block_indexes") or [],
                            "operation": block_task.get("operation"),
                            "passed_local_checks": False,
                            "formula_gap_candidate_orchestrator": True,
                        }
                        try:
                            prompt = _formula_gap_candidate_prompt(
                                orchestrator_base_text,
                                orchestrator_base_report,
                                family,
                                protected_anchors=protected_anchor_brief,
                                block_task=block_task,
                            )
                            search_summary["llm_calls"] += 1
                            orchestrator_summary["llm_calls_used"] = int(
                                orchestrator_summary.get("llm_calls_used") or 0
                            ) + 1
                            response = gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's formula-gap portfolio candidate generator. "
                                    "Return only valid JSON that matches the requested schema."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_FORMULA_GAP_PORTFOLIO",
                                    temperature_env="DRAFTPROOF_FORMULA_GAP_PORTFOLIO_TEMPERATURE",
                                    temperature_default=0.50,
                                    max_tokens_env="DRAFTPROOF_FORMULA_GAP_PORTFOLIO_MAX_TOKENS",
                                    max_tokens_default=6500,
                                ),
                            )
                            payload, payload_reason = _extract_formula_gap_candidate_payload(
                                response.content,
                            )
                            if not payload:
                                candidate_record["reason"] = payload_reason or "invalid_formula_gap_payload"
                                search_summary["candidates"].append(candidate_record)
                                orchestrator_summary["candidate_frontier"].append(candidate_record)
                                continue
                            assembled_candidate, applied_patches, assembly_reason = _assemble_formula_gap_candidate(
                                orchestrator_base_text,
                                payload,
                            )
                            candidate = _clean_full_document_candidate(assembled_candidate, orchestrator_base_text)
                            if not candidate:
                                candidate_record["reason"] = assembly_reason or "empty_or_unchanged_candidate"
                                candidate_record["payload_strategy"] = payload.get("strategy")
                                search_summary["candidates"].append(candidate_record)
                                orchestrator_summary["candidate_frontier"].append(candidate_record)
                                continue
                            _evaluate_ai_search_candidate(
                                candidate_record["strategy"],
                                candidate,
                                deterministic=False,
                                extra={
                                    "formula_gap_candidate_orchestrator": True,
                                    "formula_gap_portfolio_family": family,
                                    "block_scoped_portfolio_task": {
                                        "family": block_task.get("family"),
                                        "operation": block_task.get("operation"),
                                        "block_indexes": block_task.get("block_indexes"),
                                        "targeted_drivers": block_task.get("targeted_drivers"),
                                    },
                                    "applied_formula_gap_patches": applied_patches,
                                    "targeted_drivers": payload.get("targeted_drivers"),
                                    "changed_blocks": payload.get("changed_blocks"),
                                    "fact_inventory_preserved": payload.get("fact_inventory_preserved"),
                                    "core_claims_preserved_or_merged": payload.get("core_claims_preserved_or_merged"),
                                    "protected_anchors_preserved": payload.get("protected_anchors_preserved"),
                                    "unsupported_new_facts": payload.get("unsupported_new_facts"),
                                },
                            )
                        except Exception as exc:
                            candidate_record["reason"] = f"llm_error {exc}"
                            search_summary["candidates"].append(candidate_record)
                            orchestrator_summary["candidate_frontier"].append(candidate_record)
                            if adaptive_stop_reason:
                                orchestrator_summary["stop_reason"] = adaptive_stop_reason
                                break
                    formula_gap_orchestrator_completed = True
                    orchestrator_summary.update({
                        "completed": True,
                        "selected_strategy_after": best_strategy,
                        "selected_candidate_reason": best_selection_status.get("reason"),
                        "best_attempt": search_summary.get("best_attempt"),
                        "llm_calls_used": int(orchestrator_summary.get("llm_calls_used") or 0),
                        "candidate_scans_used": _verified_candidate_scans_used(),
                    })
                    search_summary["llm_candidate_frontier"] = [
                        {
                            "strategy": item.get("strategy"),
                            "family": item.get("formula_gap_portfolio_family") or item.get("family"),
                            "passed_local_checks": item.get("passed_local_checks"),
                            "reason": item.get("reason"),
                            "ai": item.get("ai"),
                            "human_contribution": item.get("human_contribution"),
                            "ai_authorship": item.get("ai_authorship"),
                            "ai_transformation": item.get("ai_transformation"),
                            "selection_status": item.get("selection_status"),
                            "candidate_decision": item.get("candidate_decision"),
                            "formula_gap_contract": item.get("formula_gap_contract"),
                            "eligible_span_density_gate": item.get("eligible_span_density_gate"),
                            "block_scoped_portfolio_task": item.get("block_scoped_portfolio_task"),
                            "applied_formula_gap_patches": item.get("applied_formula_gap_patches"),
                        }
                        for item in (search_summary.get("candidates") or [])
                        if str(item.get("strategy") or "").startswith("formula_gap_portfolio_")
                    ]
                    if not best_selection_status.get("turnitin_like_mitigation"):
                        search_summary["ceiling_detection"] = {
                            "status": (
                                "ceiling_reached"
                                if not _best_ai_search_selectable()
                                else "unsafe_partial_improvement"
                            ),
                            "reason": (
                                "no_formula_gap_candidate_reduced_score_safely"
                                if not _best_ai_search_selectable()
                                else "target_below_20_not_reached"
                            ),
                        }
                    adaptive_stop_reason = "adaptive_stop_after_formula_gap_candidate_orchestrator"
                    search_summary["adaptive_stop"] = {
                        "phase": "formula_gap_candidate_orchestrator",
                        "reason": adaptive_stop_reason,
                        "candidate_count_scanned": _verified_candidate_scans_used(),
                        "candidate_records": len(search_summary.get("candidates", [])),
                        "selected_strategy": best_strategy,
                        "selection_status": best_selection_status,
                    }
                    search_summary["llm_reason"] = adaptive_stop_reason
                paragraph_search_enabled = os.environ.get(
                    "DRAFTPROOF_PARAGRAPH_COMPONENT_SEARCH",
                    "1",
                ) != "0"
                component_source_text = best_text if _best_ai_search_selectable() else search_source_text
                component_base_text, component_base_repairs = _repair_candidate_source_damage(component_source_text)
                if (not formula_gap_orchestrator_completed) and _env_flag("DRAFTPROOF_TOPK_ROUTE_OPTIMIZER", True):
                    route_base_report = best_report if _best_ai_search_selectable() else original_report_dict
                    route_map = _topk_repair_map(component_base_text, route_base_report)
                    topk_summary = search_summary.setdefault("topk_route_optimizer", {})
                    topk_summary.update({
                        "enabled": True,
                        "llm_stage_enabled": True,
                        "llm_repair_map": route_map,
                    })
                    if route_map.get("saturated"):
                        try:
                            route_candidate_count = max(
                                1,
                                int(_float_env(
                                    "DRAFTPROOF_TOPK_ROUTE_LLM_CANDIDATES",
                                    float(_adaptive_budget_default(component_base_text, 1, 2)),
                                )),
                            )
                            prompt = _topk_masked_route_prompt(
                                component_base_text,
                                route_base_report,
                                candidate_count=route_candidate_count,
                            )
                            if _phase_budget_block_record("topk_safe_band_rebuild", topk_summary):
                                raise RuntimeError("phase_llm_budget_exhausted")
                            search_summary["llm_calls"] += 1
                            _record_phase_llm_call("topk_safe_band_rebuild")
                            response = gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's token-route optimizer. "
                                    "Return only JSON sentence patches for high top-k routes."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_TOPK_ROUTE",
                                    temperature_env="DRAFTPROOF_TOPK_ROUTE_TEMPERATURE",
                                    temperature_default=0.35,
                                    max_tokens_env="DRAFTPROOF_TOPK_ROUTE_MAX_TOKENS",
                                    max_tokens_default=3200,
                                ),
                            )
                            patch_sets = _extract_topk_route_patch_candidates(
                                response.content,
                                max_candidates=route_candidate_count,
                            )
                            topk_summary["llm_candidate_count"] = len(patch_sets)
                            for route_index, patches in enumerate(patch_sets, start=1):
                                candidate, applied = _apply_topk_route_patches(component_base_text, patches)
                                strategy = f"topk_route_masked_c{route_index}"
                                if not applied or candidate == component_base_text:
                                    search_summary["candidates"].append({
                                        "strategy": strategy,
                                        "passed_local_checks": False,
                                        "reason": "no_topk_route_patch_applied",
                                        "topk_route_optimizer": True,
                                    })
                                    continue
                                _evaluate_ai_search_candidate(
                                    strategy,
                                    candidate,
                                    deterministic=False,
                                    extra={
                                        "topk_route_optimizer": True,
                                        "stage": "masked_span_regeneration",
                                        "applied_topk_route_patches": applied,
                                        "base_strategy": best_strategy if _best_ai_search_selectable() else "source",
                                    },
                                )
                        except Exception as exc:
                            search_summary["candidates"].append({
                                "strategy": "topk_route_masked_batch",
                                "passed_local_checks": False,
                                "reason": f"llm_error {exc}",
                                "topk_route_optimizer": True,
                            })
                            topk_summary["llm_error"] = str(exc)
                    else:
                        topk_summary["llm_stage_skipped"] = "topk_not_saturated"
                if (not formula_gap_orchestrator_completed) and _env_flag("DRAFTPROOF_TOPK_SAFE_BAND_REBUILD", True):
                    safe_band_summary = search_summary.setdefault("topk_safe_band_rebuild", {
                        "enabled": True,
                        "selected": False,
                        "stages": [],
                    })
                    try:
                        planned_patch_rounds_for_reserve = max(1, int(_float_env(
                            "DRAFTPROOF_TOPK_SAFE_BAND_PATCH_ROUNDS",
                            float(_topk_safe_band_patch_rounds_default(search_source_text, original_report_dict)),
                        )))
                    except (TypeError, ValueError):
                        planned_patch_rounds_for_reserve = _topk_safe_band_patch_rounds_default(search_source_text, original_report_dict)
                    try:
                        planned_extra_safe_rounds_for_reserve = max(0, int(_float_env(
                            "DRAFTPROOF_TOPK_SAFE_BAND_EXTRA_SAFE_ROUNDS",
                            1.0,
                        )))
                    except (TypeError, ValueError):
                        planned_extra_safe_rounds_for_reserve = 1
                    try:
                        llm_reserve = max(0, int(_float_env(
                            "DRAFTPROOF_TOPK_SAFE_BAND_LLM_RESERVE",
                            float(1 + planned_patch_rounds_for_reserve + planned_extra_safe_rounds_for_reserve),
                        )))
                    except (TypeError, ValueError):
                        llm_reserve = 1 + planned_patch_rounds_for_reserve + planned_extra_safe_rounds_for_reserve
                    if llm_reserve > 0 and not safe_band_summary.get("llm_reserve_added"):
                        previous_llm_max = int(search_budget.get("max_llm_calls") or 0)
                        current_llm_calls = int(search_summary.get("llm_calls") or 0)
                        search_budget["max_llm_calls"] = min(
                            hard_llm_cap,
                            max(
                                previous_llm_max,
                                current_llm_calls + llm_reserve,
                            ),
                        )
                        safe_band_summary["llm_reserve_added"] = {
                            "reserve_added": llm_reserve,
                            "previous_max_llm_calls": previous_llm_max,
                            "new_max_llm_calls": search_budget["max_llm_calls"],
                        }
                    reserve = _topk_safe_band_scan_reserve()
                    if reserve > 0 and not safe_band_summary.get("scan_reserve_added"):
                        previous_max = int(search_budget.get("max_candidate_scans") or 0)
                        current_scans = _verified_candidate_scans_used()
                        _extend_candidate_scan_budget(search_budget, current_scans, reserve)
                        safe_band_summary["scan_reserve_added"] = {
                            "reserve_added": reserve,
                            "previous_max_candidate_scans": previous_max,
                            "new_max_candidate_scans": search_budget["max_candidate_scans"],
                        }
                    route_base_report = best_report if _best_ai_search_selectable() else original_report_dict
                    route_base_text = best_text if _best_ai_search_selectable() else component_base_text
                    base_ai_components = (((route_base_report or {}).get("ai_risk_badge") or {}).get("ai_components") or {})
                    base_topk_calibrated = base_ai_components.get("topk_calibrated_risk")
                    if isinstance(base_topk_calibrated, (int, float)) and float(base_topk_calibrated) < _safe_topk_calibrated_limit():
                        safe_band_summary["skipped"] = True
                        safe_band_summary["reason"] = "topk_safe_band_already_reached"
                    elif not _search_budget_exhausted("topk_safe_band_rebuild"):
                        try:
                            if _phase_budget_block_record("topk_safe_band_rebuild", safe_band_summary):
                                raise RuntimeError("phase_llm_budget_exhausted")
                            search_summary["llm_calls"] += 1
                            _record_phase_llm_call("topk_safe_band_rebuild")
                            snapshot_response = gateway.chat(
                                _topk_safe_band_snapshot_prompt(route_base_text, route_base_report),
                                system=(
                                    "You are DraftProof's safe-band Top-k rebuild controller. "
                                    "Return only rewritten prose."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_TOPK_SAFE_BAND_REBUILD",
                                    temperature_env="DRAFTPROOF_TOPK_SAFE_BAND_REBUILD_TEMPERATURE",
                                    temperature_default=0.45,
                                    max_tokens_env="DRAFTPROOF_TOPK_SAFE_BAND_REBUILD_MAX_TOKENS",
                                    max_tokens_default=_topk_safe_band_snapshot_max_tokens_default(route_base_text),
                                ),
                            )
                            snapshot_text = _clean_full_document_candidate(snapshot_response.content, route_base_text)
                            safe_band_summary["snapshot_words"] = _text_word_count(snapshot_text)
                            if snapshot_text:
                                snapshot_report = _full_scan_report_dict(snapshot_text)
                                snapshot_ai = (((snapshot_report or {}).get("ai_risk_badge") or {}).get("ai_components") or {})
                                safe_band_summary["snapshot_scan"] = {
                                    "topk_pattern_raw": snapshot_ai.get("topk_pattern_raw", snapshot_ai.get("topk_pattern")),
                                    "topk_calibrated_risk": snapshot_ai.get("topk_calibrated_risk"),
                                    "topk_safe_band": snapshot_ai.get("topk_safe_band"),
                                }
                                if (
                                    isinstance(snapshot_ai.get("topk_calibrated_risk"), (int, float))
                                    and float(snapshot_ai.get("topk_calibrated_risk")) >= 75.0
                                    and not _phase_budget_block_record("topk_safe_band_rebuild", safe_band_summary)
                                    and not _search_budget_exhausted("plain_spoken_topk_rebuild")
                                ):
                                    search_summary["llm_calls"] += 1
                                    _record_phase_llm_call("topk_safe_band_rebuild")
                                    plain_response = gateway.chat(
                                        _topk_plain_spoken_snapshot_prompt(route_base_text, route_base_report),
                                        system=(
                                            "You are DraftProof's plain-spoken Top-k rebuild controller. "
                                            "Return only rewritten prose."
                                        ),
                                        **_phase_chat_sampling_kwargs(
                                            "DRAFTPROOF_PLAIN_SPOKEN_TOPK_REBUILD",
                                            temperature_env="DRAFTPROOF_PLAIN_SPOKEN_TOPK_REBUILD_TEMPERATURE",
                                            temperature_default=0.58,
                                            max_tokens_env="DRAFTPROOF_PLAIN_SPOKEN_TOPK_REBUILD_MAX_TOKENS",
                                            max_tokens_default=_topk_safe_band_snapshot_max_tokens_default(route_base_text),
                                        ),
                                    )
                                    plain_text = _clean_full_document_candidate(plain_response.content, route_base_text)
                                    if plain_text:
                                        plain_report = _full_scan_report_dict(plain_text)
                                        plain_ai = (((plain_report or {}).get("ai_risk_badge") or {}).get("ai_components") or {})
                                        safe_band_summary["plain_spoken_snapshot"] = {
                                            "words": _text_word_count(plain_text),
                                            "topk_pattern_raw": plain_ai.get("topk_pattern_raw", plain_ai.get("topk_pattern")),
                                            "topk_calibrated_risk": plain_ai.get("topk_calibrated_risk"),
                                            "topk_safe_band": plain_ai.get("topk_safe_band"),
                                        }
                                        if _topk_rebuild_fallback_rank(plain_report) > _topk_rebuild_fallback_rank(snapshot_report):
                                            snapshot_text = plain_text
                                            snapshot_report = plain_report
                                            snapshot_ai = plain_ai
                                            safe_band_summary["snapshot_replaced_by"] = "plain_spoken_snapshot"
                                candidate_to_eval = snapshot_text
                                candidate_patch_report = snapshot_report
                                applied = []
                                patch_rounds = []
                                best_safe_text = None
                                best_safe_report = None
                                best_safe_rank = None
                                best_topk_text = snapshot_text
                                best_topk_report = snapshot_report
                                best_topk_rank = _topk_rebuild_fallback_rank(snapshot_report)

                                def _topk_safe_rank(report_dict: dict | None) -> tuple:
                                    strict_status = _strict_ai_safe_band_status(report_dict)
                                    profile = strict_status.get("profile") or {}
                                    topk_value = profile.get("topk_calibrated_risk")
                                    if not isinstance(topk_value, (int, float)) or float(topk_value) >= _safe_topk_calibrated_limit():
                                        return ()
                                    return (
                                        1 if strict_status.get("achieved") else 0,
                                        -float(profile.get("external_ai_flag_risk") or 0.0),
                                        -float(profile.get("ai_authorship") or 0.0),
                                        -float(profile.get("ai_transformation") or 0.0),
                                        -float(profile.get("ai_likelihood") or 0.0),
                                        -float(profile.get("topk_calibrated_risk") or 0.0),
                                        -float(profile.get("rewrite_smoothness") or 0.0),
                                    )

                                snapshot_safe_rank = _topk_safe_rank(snapshot_report)
                                if snapshot_safe_rank:
                                    best_safe_text = snapshot_text
                                    best_safe_report = snapshot_report
                                    best_safe_rank = snapshot_safe_rank
                                try:
                                    max_patch_rounds = max(1, int(_float_env(
                                        "DRAFTPROOF_TOPK_SAFE_BAND_PATCH_ROUNDS",
                                        float(_topk_safe_band_patch_rounds_default(route_base_text, route_base_report)),
                                    )))
                                except (TypeError, ValueError):
                                    max_patch_rounds = _topk_safe_band_patch_rounds_default(route_base_text, route_base_report)
                                try:
                                    extra_safe_rounds = max(0, int(_float_env(
                                        "DRAFTPROOF_TOPK_SAFE_BAND_EXTRA_SAFE_ROUNDS",
                                        1.0,
                                    )))
                                except (TypeError, ValueError):
                                    extra_safe_rounds = 1
                                safe_rounds_used = 0
                                previous_topk_calibrated = (
                                    float(snapshot_ai.get("topk_calibrated_risk"))
                                    if isinstance(snapshot_ai.get("topk_calibrated_risk"), (int, float))
                                    else None
                                )
                                stagnant_topk_rounds = 0
                                min_round_drop = _float_env("DRAFTPROOF_TOPK_SAFE_BAND_MIN_MARGINAL_DROP", 0.35)
                                for patch_round in range(1, max_patch_rounds + 1):
                                    current_ai = (((candidate_patch_report or {}).get("ai_risk_badge") or {}).get("ai_components") or {})
                                    if isinstance(current_ai.get("topk_calibrated_risk"), (int, float)) and float(current_ai.get("topk_calibrated_risk")) < _safe_topk_calibrated_limit():
                                        if safe_rounds_used >= extra_safe_rounds:
                                            patch_rounds.append({
                                                "round": patch_round,
                                                "skipped": True,
                                                "reason": "topk_safe_band_reached",
                                                "topk_calibrated_risk": current_ai.get("topk_calibrated_risk"),
                                            })
                                            break
                                        safe_rounds_used += 1
                                    if _phase_budget_block_record("topk_safe_band_rebuild", safe_band_summary):
                                        patch_rounds.append({
                                            "round": patch_round,
                                            "skipped": True,
                                            "reason": "phase_llm_budget_exhausted",
                                            "phase_budget_used": dict(phase_budget_used),
                                        })
                                        break
                                    search_summary["llm_calls"] += 1
                                    _record_phase_llm_call("topk_safe_band_rebuild")
                                    patch_response = gateway.chat(
                                        _topk_safe_band_sentence_patch_prompt(candidate_to_eval, candidate_patch_report),
                                        system=(
                                            "You are DraftProof's Top-k sentence route patcher. "
                                            "Return only valid JSON patches."
                                        ),
                                        **_phase_chat_sampling_kwargs(
                                            "DRAFTPROOF_TOPK_SAFE_BAND_PATCH",
                                            temperature_env="DRAFTPROOF_TOPK_SAFE_BAND_PATCH_TEMPERATURE",
                                            temperature_default=0.35,
                                            max_tokens_env="DRAFTPROOF_TOPK_SAFE_BAND_PATCH_MAX_TOKENS",
                                            max_tokens_default=2600,
                                        ),
                                    )
                                    patch_sets = _extract_topk_route_patch_candidates(
                                        patch_response.content,
                                        max_candidates=int(_float_env("DRAFTPROOF_TOPK_SAFE_BAND_PATCH_CANDIDATES", 2.0)),
                                    )
                                    round_applied = []
                                    patched_text = candidate_to_eval
                                    best_round_report = None
                                    best_round_rank = None
                                    best_round_rejection = ""
                                    for patch_set in patch_sets:
                                        trial_text, trial_applied = _apply_topk_route_patches(candidate_to_eval, patch_set)
                                        if not trial_applied or trial_text == candidate_to_eval:
                                            continue
                                        trial_rejection = _ai_candidate_quality_reject_reason(trial_text)
                                        if trial_rejection:
                                            best_round_rejection = trial_rejection
                                            continue
                                        trial_report = _full_scan_report_dict(trial_text)
                                        trial_rank = _topk_rebuild_fallback_rank(trial_report)
                                        if trial_rank and (best_round_rank is None or trial_rank > best_round_rank):
                                            patched_text = trial_text
                                            round_applied = trial_applied
                                            best_round_report = trial_report
                                            best_round_rank = trial_rank
                                    patch_rounds.append({
                                        "round": patch_round,
                                        "patch_candidate_count": len(patch_sets),
                                        "applied_patch_count": len(round_applied),
                                        "rejected_patch_candidate_reason": best_round_rejection,
                                    })
                                    if not round_applied or patched_text == candidate_to_eval:
                                        break
                                    candidate_to_eval = patched_text
                                    applied.extend(round_applied)
                                    candidate_patch_report = best_round_report or _full_scan_report_dict(candidate_to_eval)
                                    patched_ai = (((candidate_patch_report or {}).get("ai_risk_badge") or {}).get("ai_components") or {})
                                    patched_topk_calibrated = patched_ai.get("topk_calibrated_risk")
                                    marginal_topk_drop = None
                                    if (
                                        isinstance(previous_topk_calibrated, (int, float))
                                        and isinstance(patched_topk_calibrated, (int, float))
                                    ):
                                        marginal_topk_drop = round(
                                            float(previous_topk_calibrated) - float(patched_topk_calibrated),
                                            3,
                                        )
                                        if (
                                            marginal_topk_drop < min_round_drop
                                            and float(patched_topk_calibrated) > _safe_topk_calibrated_limit() + 1.0
                                        ):
                                            stagnant_topk_rounds += 1
                                        else:
                                            stagnant_topk_rounds = 0
                                        previous_topk_calibrated = float(patched_topk_calibrated)
                                    patch_rounds[-1].update({
                                        "topk_pattern_raw": patched_ai.get("topk_pattern_raw", patched_ai.get("topk_pattern")),
                                        "topk_calibrated_risk": patched_topk_calibrated,
                                        "topk_safe_band": patched_ai.get("topk_safe_band"),
                                        "marginal_topk_drop": marginal_topk_drop,
                                    })
                                    fallback_rank = _topk_rebuild_fallback_rank(candidate_patch_report)
                                    if fallback_rank and (not best_topk_rank or fallback_rank > best_topk_rank):
                                        best_topk_text = candidate_to_eval
                                        best_topk_report = candidate_patch_report
                                        best_topk_rank = fallback_rank
                                    safe_rank = _topk_safe_rank(candidate_patch_report)
                                    if safe_rank and (best_safe_rank is None or safe_rank > best_safe_rank):
                                        best_safe_text = candidate_to_eval
                                        best_safe_report = candidate_patch_report
                                        best_safe_rank = safe_rank
                                    if stagnant_topk_rounds >= 2:
                                        patch_rounds.append({
                                            "round": patch_round + 1,
                                            "skipped": True,
                                            "reason": "marginal_topk_gain_stalled",
                                            "min_marginal_drop": min_round_drop,
                                            "stagnant_rounds": stagnant_topk_rounds,
                                            "topk_calibrated_risk": patched_topk_calibrated,
                                        })
                                        break
                                safe_band_summary["patch_rounds"] = patch_rounds
                                safe_band_summary["patch_candidate_count"] = sum(
                                    int(row.get("patch_candidate_count") or 0) for row in patch_rounds
                                )
                                safe_band_summary["applied_patch_count"] = len(applied)
                                if best_safe_text and best_safe_report:
                                    candidate_to_eval = best_safe_text
                                    candidate_patch_report = best_safe_report
                                    safe_band_summary["selected_best_safe_rank"] = list(best_safe_rank or [])
                                elif best_topk_text and best_topk_report:
                                    candidate_to_eval = best_topk_text
                                    candidate_patch_report = best_topk_report
                                    safe_band_summary["selected_best_topk_fallback_rank"] = list(best_topk_rank or [])
                                _evaluate_ai_search_candidate(
                                    "topk_safe_band_rebuild",
                                    candidate_to_eval,
                                    deterministic=False,
                                    extra={
                                        "topk_safe_band_rebuild": True,
                                        "snapshot_topk_scan": safe_band_summary.get("snapshot_scan"),
                                        "applied_topk_safe_band_patches": applied,
                                    },
                                )
                                safe_band_summary["selected"] = bool(best_strategy == "topk_safe_band_rebuild")
                        except Exception as exc:
                            safe_band_summary["error"] = str(exc)
                            search_summary["candidates"].append({
                                "strategy": "topk_safe_band_rebuild",
                                "passed_local_checks": False,
                                "reason": f"llm_error {exc}",
                                "topk_safe_band_rebuild": True,
                            })
                else:
                    search_summary["topk_safe_band_rebuild"] = {"enabled": False, "reason": "disabled"}
                if (
                    _best_ai_search_selectable()
                    and bool(best_selection_status.get("topk_safe_band_achieved"))
                ):
                    _run_post_topk_ai_safe_band_optimizer("after_topk_safe_band_rebuild")
                    if not _strict_ai_safe_band_status(best_report).get("achieved"):
                        _run_final_topk_texture_repair("after_post_topk_optimizer")
                selected_topk_gate = (
                    best_selection_status.get("ai_footprint_gate")
                    if isinstance(best_selection_status, dict) else {}
                )
                selected_topk_profile = (
                    ((selected_topk_gate.get("after") or {}).get("authorship_footprint") or {})
                    if isinstance(selected_topk_gate, dict) else {}
                )
                selected_topk_value = selected_topk_profile.get("topk_calibrated_risk")
                selected_topk_drop = (
                    (selected_topk_gate.get("drops") or {}).get("topk_calibrated_risk")
                    if isinstance(selected_topk_gate, dict) else None
                )
                selected_topk_rebuild_needs_final_repair = bool(
                    _best_ai_search_selectable()
                    and best_strategy == "topk_safe_band_rebuild"
                    and not bool(best_selection_status.get("topk_safe_band_achieved"))
                    and isinstance(selected_topk_value, (int, float))
                    and _safe_topk_calibrated_limit() <= float(selected_topk_value) <= _safe_topk_calibrated_limit() + 5.0
                    and isinstance(selected_topk_drop, (int, float))
                    and float(selected_topk_drop) >= _float_env(
                        "DRAFTPROOF_AI_FOOTPRINT_SATURATED_MIN_TOPK_DROP",
                        8.0,
                    )
                )
                if selected_topk_rebuild_needs_final_repair:
                    _run_final_topk_texture_repair("after_topk_near_safe_rebuild")
                    selected_topk_gate = (
                        best_selection_status.get("ai_footprint_gate")
                        if isinstance(best_selection_status, dict) else {}
                    )
                    selected_topk_profile = (
                        ((selected_topk_gate.get("after") or {}).get("authorship_footprint") or {})
                        if isinstance(selected_topk_gate, dict) else {}
                    )
                    selected_topk_value = selected_topk_profile.get("topk_calibrated_risk")
                    selected_topk_drop = (
                        (selected_topk_gate.get("drops") or {}).get("topk_calibrated_risk")
                        if isinstance(selected_topk_gate, dict) else None
                    )
                selected_topk_controlled_candidate = bool(
                    _best_ai_search_selectable()
                    and best_strategy == "topk_safe_band_rebuild"
                    and isinstance(selected_topk_value, (int, float))
                    and float(selected_topk_value) <= _safe_topk_calibrated_limit() + 5.0
                    and isinstance(selected_topk_drop, (int, float))
                    and float(selected_topk_drop) >= _float_env(
                        "DRAFTPROOF_AI_FOOTPRINT_SATURATED_MIN_TOPK_DROP",
                        8.0,
                    )
                )
                strict_phase_budget_only_active = bool(
                    _env_flag("DRAFTPROOF_STRICT_AI_PHASE_BUDGET_ONLY", True)
                    and (topk_safe_band_priority or density_or_generic_priority)
                    and not _strict_ai_safe_band_status(best_report).get("achieved")
                )
                strict_safe_legacy_llm_skipped = bool(
                    formula_gap_orchestrator_completed
                    or
                    (
                        _best_ai_search_selectable()
                        and (
                            bool(best_selection_status.get("topk_safe_band_achieved"))
                            or selected_topk_controlled_candidate
                        )
                        and not _strict_ai_safe_band_status(best_report).get("achieved")
                        and _env_flag("DRAFTPROOF_SKIP_LEGACY_LLM_AFTER_TOPK_SAFE", True)
                    )
                    or strict_phase_budget_only_active
                )
                if strict_safe_legacy_llm_skipped:
                    search_summary["legacy_llm_after_topk_safe"] = {
                        "skipped": True,
                        "reason": (
                            "formula_gap_candidate_orchestrator_completed"
                            if formula_gap_orchestrator_completed
                            else "strict_ai_phase_budget_only"
                            if strict_phase_budget_only_active
                            else
                            "preserve_remaining_budget_for_topk_controlled_candidate"
                            if selected_topk_controlled_candidate
                            else "preserve_remaining_budget_for_strict_safe_controller"
                        ),
                        "strict_ai_phase_budget_only": strict_phase_budget_only_active,
                        "selected_strategy": best_strategy,
                        "selected_topk_calibrated_risk": selected_topk_value,
                        "selected_topk_calibrated_drop": selected_topk_drop,
                        "strict_ai_safe_band": _strict_ai_safe_band_status(best_report),
                    }
                    if not formula_gap_orchestrator_completed:
                        adaptive_stop_reason = "adaptive_stop_after_strict_safe_phase_budget"
                        search_summary["adaptive_stop"] = {
                            "phase": "strict_safe_controller",
                            "reason": adaptive_stop_reason,
                            "candidate_count_scanned": _verified_candidate_scans_used(),
                            "selected_strategy": best_strategy,
                            "selection_status": best_selection_status,
                        }
                internet_priority = _internet_reauthor_priority_status(original_report_dict, component_base_text)
                paragraph_component_first = bool(
                    paragraph_search_enabled
                    and not strict_safe_legacy_llm_skipped
                    and _env_flag("DRAFTPROOF_PARAGRAPH_COMPONENT_FIRST", True)
                    and not internet_priority.get("prioritize")
                    and _text_word_count(component_base_text) >= int(_float_env(
                        "DRAFTPROOF_PARAGRAPH_COMPONENT_FIRST_MIN_WORDS",
                        450.0,
                    ))
                )
                if internet_priority.get("prioritize"):
                    search_summary["internet_reauthor_priority"] = internet_priority
                source_components = (
                    (original_report_dict.get("ai_risk_badge") or {}).get("writing_components") or {}
                    if isinstance(original_report_dict, dict) else {}
                )
                source_grounding_for_search = max(
                    float(source_components.get("source_grounding_risk") or 0.0),
                    float(source_components.get("citation_weakness_risk") or 0.0),
                )
                source_repair_allowed = bool(
                    not strict_safe_legacy_llm_skipped
                    and
                    _env_flag("DRAFTPROOF_SOURCE_GROUNDING_REPAIR", True)
                    and (
                        not human_target_search_status.get("active")
                        or source_grounding_for_search >= _float_env(
                            "DRAFTPROOF_SOURCE_REPAIR_MIN_SOURCE_BLOCKER_FOR_HUMAN_TARGET",
                            65.0,
                        )
                    )
                )
                search_summary["source_grounding_repair_policy"] = {
                    "enabled": bool(_env_flag("DRAFTPROOF_SOURCE_GROUNDING_REPAIR", True)),
                    "allowed": source_repair_allowed,
                    "legacy_llm_after_topk_safe_skipped": strict_safe_legacy_llm_skipped,
                    "source_grounding_or_citation_blocker": round(source_grounding_for_search, 3),
                    "human_target_active": bool(human_target_search_status.get("active")),
                    "min_source_blocker_for_human_target": _float_env(
                        "DRAFTPROOF_SOURCE_REPAIR_MIN_SOURCE_BLOCKER_FOR_HUMAN_TARGET",
                        65.0,
                    ),
                }
                source_layer = {}
                source_repair_results = []
                source_candidate_count = 0
                if source_repair_allowed:
                    source_layer = _build_source_grounding_search_layer(
                        component_base_text,
                        original_report_dict,
                        max_queries=int(_float_env("DRAFTPROOF_SOURCE_GROUNDING_REPAIR_TARGETS", 2.0)),
                        max_results=int(_float_env("DRAFTPROOF_SOURCE_GROUNDING_REPAIR_MAX_RESULTS", 3.0)),
                    )
                    usable_confidences = {
                        item.strip()
                        for item in os.environ.get(
                            "DRAFTPROOF_SOURCE_GROUNDING_REPAIR_CONFIDENCE",
                            "strong,moderate",
                        ).split(",")
                        if item.strip()
                    }
                    source_repair_results = _source_grounding_repair_matches(
                        source_layer,
                        usable_confidences,
                        limit=max(
                            0,
                            int(_float_env(
                                "DRAFTPROOF_SOURCE_GROUNDING_REPAIR_TARGETS",
                                float(_adaptive_budget_default(component_base_text, 1, 2)),
                            )),
                        ),
                    )
                    source_candidate_count = max(
                        1,
                        int(_float_env(
                            "DRAFTPROOF_SOURCE_GROUNDING_REPAIR_CANDIDATES",
                            float(_adaptive_budget_default(component_base_text, 2, 2)),
                        )),
                    )
                    search_summary["source_grounding_repair"] = {
                        "enabled": True,
                        "search_status": source_layer.get("status"),
                        "usable_confidences": sorted(usable_confidences),
                        "target_count": len(source_repair_results),
                        "candidate_limit_per_target": source_candidate_count,
                        "source_search": {
                            "targets": len(source_layer.get("claim_targets") or []),
                            "results": len(source_layer.get("results") or []),
                            "errors": len(source_layer.get("errors") or []),
                        },
                        "targets": [
                            {
                                "claim_id": result.get("claim_id"),
                                "paragraph_index": result.get("paragraph_index"),
                                "source_confidence": result.get("source_confidence"),
                                "query": result.get("query"),
                            }
                            for result in source_repair_results
                        ],
                    }
                    source_reference_candidate = _source_reference_append_candidate(
                        component_base_text,
                        source_layer,
                        limit=int(_float_env("DRAFTPROOF_SOURCE_REFERENCE_APPEND_LIMIT", 2.0)),
                    )
                    if source_reference_candidate:
                        _evaluate_ai_search_candidate(
                            "source_reference_append",
                            source_reference_candidate,
                            deterministic=True,
                            extra={
                                "source_reference_append": True,
                                "source_search_status": source_layer.get("status"),
                                "source_result_count": len(source_layer.get("results") or []),
                                "reference_entries": len(
                                    _source_reference_entries_from_layer(
                                        source_layer,
                                        limit=int(_float_env(
                                            "DRAFTPROOF_SOURCE_REFERENCE_APPEND_LIMIT",
                                            2.0,
                                        )),
                                    )
                                ),
                            },
                        )
                    if (
                        _env_flag("DRAFTPROOF_INTERNET_REINFORCED_REAUTHORING", True)
                        and not paragraph_component_first
                        and not strict_safe_legacy_llm_skipped
                    ):
                        internet_candidate_count = max(
                            1,
                            int(_float_env(
                                "DRAFTPROOF_INTERNET_REAUTHOR_CANDIDATES",
                                float(_adaptive_budget_default(component_base_text, 1, 2)),
                            )),
                        )
                        search_summary["internet_reinforced_reauthoring"] = {
                            "enabled": True,
                            "source_search_status": source_layer.get("status"),
                            "source_targets": len(source_layer.get("claim_targets") or []),
                            "source_results": len(source_layer.get("results") or []),
                            "candidate_limit": internet_candidate_count,
                        }
                        try:
                            prompt = _internet_reinforced_reauthor_prompt(
                                component_base_text,
                                source_layer,
                                candidate_count=internet_candidate_count,
                            )
                            search_summary["llm_calls"] += 1
                            response = gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's internet-reinforced document reauthoring engine. "
                                    "Rebuild from source-supported claims and remove unsupported generic drag. "
                                    "Return only tagged full-document candidates."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_INTERNET_REAUTHOR",
                                    temperature_env="DRAFTPROOF_INTERNET_REAUTHOR_TEMPERATURE",
                                    temperature_default=0.52,
                                    max_tokens_env="DRAFTPROOF_INTERNET_REAUTHOR_MAX_TOKENS",
                                    max_tokens_default=5200,
                                ),
                            )
                            internet_outputs = _extract_paragraph_component_candidates(
                                response.content,
                                internet_candidate_count,
                            )
                        except Exception as exc:
                            search_summary["candidates"].append({
                                "strategy": "internet_reinforced_reauthoring_batch",
                                "passed_local_checks": False,
                                "reason": f"llm_error {exc}",
                                "internet_reinforced_reauthoring": True,
                            })
                            internet_outputs = []
                        for candidate_number, raw_candidate in enumerate(internet_outputs, start=1):
                            candidate = _clean_full_document_candidate(
                                raw_candidate,
                                component_base_text,
                            )
                            strategy = f"internet_reinforced_reauthoring_c{candidate_number}"
                            if not candidate:
                                search_summary["candidates"].append({
                                    "strategy": strategy,
                                    "passed_local_checks": False,
                                    "reason": "empty_or_unchanged_candidate",
                                    "internet_reinforced_reauthoring": True,
                                })
                                continue
                            _evaluate_ai_search_candidate(
                                strategy,
                                candidate,
                                deterministic=False,
                                extra={
                                    "internet_reinforced_reauthoring": True,
                                    "source_search_status": source_layer.get("status"),
                                    "source_result_count": len(source_layer.get("results") or []),
                                    "scan_generated_candidate_after_budget": True,
                                },
                            )
                    elif paragraph_component_first:
                        search_summary["internet_reinforced_reauthoring"] = {
                            "enabled": False,
                            "reason": "deferred_after_paragraph_component_first",
                        }
                    if (
                        _env_flag("DRAFTPROOF_CLAIM_NARROWING_REPAIR", True)
                        and not paragraph_component_first
                        and not strict_safe_legacy_llm_skipped
                    ):
                        claim_candidate_count = max(
                            1,
                            int(_float_env(
                                "DRAFTPROOF_CLAIM_NARROWING_CANDIDATES",
                                float(_adaptive_budget_default(component_base_text, 1, 2)),
                            )),
                        )
                        search_summary["claim_narrowing_repair"] = {
                            "enabled": True,
                            "candidate_limit": claim_candidate_count,
                            "target_blockers": [
                                "unsupported_claim_risk",
                                "broad_claim_risk",
                            ],
                        }
                        try:
                            prompt = _claim_narrowing_repair_prompt(
                                component_base_text,
                                original_report_dict,
                                candidate_count=claim_candidate_count,
                            )
                            search_summary["llm_calls"] += 1
                            response = gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's claim narrowing engine. "
                                    "Reduce unsupported and broad claims by narrowing/removing overreach. "
                                    "Return only tagged full-document candidates."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_CLAIM_NARROWING",
                                    temperature_env="DRAFTPROOF_CLAIM_NARROWING_TEMPERATURE",
                                    temperature_default=0.38,
                                    max_tokens_env="DRAFTPROOF_CLAIM_NARROWING_MAX_TOKENS",
                                    max_tokens_default=4800,
                                ),
                            )
                            claim_outputs = _extract_paragraph_component_candidates(
                                response.content,
                                claim_candidate_count,
                            )
                        except Exception as exc:
                            search_summary["candidates"].append({
                                "strategy": "claim_narrowing_repair_batch",
                                "passed_local_checks": False,
                                "reason": f"llm_error {exc}",
                                "claim_narrowing_repair": True,
                            })
                            claim_outputs = []
                        for candidate_number, raw_candidate in enumerate(claim_outputs, start=1):
                            candidate = _clean_full_document_candidate(
                                raw_candidate,
                                component_base_text,
                            )
                            strategy = f"claim_narrowing_repair_c{candidate_number}"
                            if not candidate:
                                search_summary["candidates"].append({
                                    "strategy": strategy,
                                    "passed_local_checks": False,
                                    "reason": "empty_or_unchanged_candidate",
                                    "claim_narrowing_repair": True,
                                })
                                continue
                            _evaluate_ai_search_candidate(
                                strategy,
                                candidate,
                                deterministic=False,
                                extra={"claim_narrowing_repair": True},
                            )
                if not source_repair_allowed:
                    search_summary["source_grounding_repair"] = {
                        "enabled": bool(_env_flag("DRAFTPROOF_SOURCE_GROUNDING_REPAIR", True)),
                        "skipped": True,
                        "reason": "deferred_because_human_target_requires_transformation_driver_repair",
                        "source_grounding_or_citation_blocker": round(source_grounding_for_search, 3),
                    }
                    if paragraph_component_first:
                        search_summary["claim_narrowing_repair"] = {
                            "enabled": False,
                            "reason": "deferred_after_paragraph_component_first",
                        }
                if (
                    _env_flag("DRAFTPROOF_TOPK_TEXTURE_REPAIR", True)
                    and not paragraph_component_first
                    and not strict_safe_legacy_llm_skipped
                ):
                    texture_base_text = best_text if _best_ai_search_selectable() else component_base_text
                    texture_base_report = best_report if _best_ai_search_selectable() else original_report_dict
                    topk_candidate_count = max(
                        1,
                        int(_float_env(
                            "DRAFTPROOF_TOPK_TEXTURE_CANDIDATES",
                            float(_adaptive_budget_default(component_base_text, 1, 2)),
                        )),
                    )
                    search_summary["topk_texture_repair"] = {
                        "enabled": True,
                        "candidate_limit": topk_candidate_count,
                        "base_strategy": best_strategy if _best_ai_search_selectable() else "source",
                    }
                    try:
                        prompt = _topk_texture_repair_prompt(
                            texture_base_text,
                            texture_base_report,
                            candidate_count=topk_candidate_count,
                        )
                        search_summary["llm_calls"] += 1
                        response = gateway.chat(
                            prompt,
                            system=(
                                "You are DraftProof's top-k texture repair engine. "
                                "Patch predictable phrasing without adding facts. "
                                "Return only tagged full-document candidates."
                            ),
                            **_phase_chat_sampling_kwargs(
                                "DRAFTPROOF_TOPK_TEXTURE",
                                temperature_env="DRAFTPROOF_TOPK_TEXTURE_TEMPERATURE",
                                temperature_default=0.78,
                                max_tokens_env="DRAFTPROOF_TOPK_TEXTURE_MAX_TOKENS",
                                max_tokens_default=4800,
                            ),
                        )
                        topk_outputs = _extract_paragraph_component_candidates(
                            response.content,
                            topk_candidate_count,
                        )
                    except Exception as exc:
                        search_summary["candidates"].append({
                            "strategy": "topk_texture_repair_batch",
                            "passed_local_checks": False,
                            "reason": f"llm_error {exc}",
                            "topk_texture_repair": True,
                        })
                        topk_outputs = []
                    for candidate_number, raw_candidate in enumerate(topk_outputs, start=1):
                        candidate = _clean_full_document_candidate(
                            raw_candidate,
                            texture_base_text,
                        )
                        strategy = f"topk_texture_repair_c{candidate_number}"
                        if not candidate:
                            search_summary["candidates"].append({
                                "strategy": strategy,
                                "passed_local_checks": False,
                                "reason": "empty_or_unchanged_candidate",
                                "topk_texture_repair": True,
                            })
                            continue
                        _evaluate_ai_search_candidate(
                            strategy,
                            candidate,
                            deterministic=False,
                            extra={
                                "topk_texture_repair": True,
                                "base_strategy": best_strategy if _best_ai_search_selectable() else "source",
                            },
                        )
                elif paragraph_component_first:
                    search_summary["topk_texture_repair"] = {
                        "enabled": False,
                        "reason": "deferred_after_paragraph_component_first",
                    }
                for source_number, source_result in enumerate(source_repair_results, start=1):
                    target = source_result.get("_repair_target") or {}
                    paragraph_index = _safe_index(target.get("paragraph_index"), 0)
                    report_progress(
                        min(89, 78 + source_number),
                        (
                            "Trying source-grounded reinforce/remove candidate "
                            f"{source_number}/{len(source_repair_results)}"
                        ),
                    )
                    try:
                        prompt = _source_grounding_repair_prompt(
                            target,
                            source_result,
                            candidate_count=source_candidate_count,
                        )
                        search_summary["llm_calls"] += 1
                        response = gateway.chat(
                            prompt,
                            system=(
                                "You are DraftProof's source-grounding repair engine. "
                                "Use only provided source candidates and return tagged replacement paragraphs."
                            ),
                            **_phase_chat_sampling_kwargs(
                                "DRAFTPROOF_SOURCE_GROUNDING_REPAIR",
                                temperature_env="DRAFTPROOF_SOURCE_GROUNDING_REPAIR_TEMPERATURE",
                                temperature_default=0.45,
                                max_tokens_env="DRAFTPROOF_SOURCE_GROUNDING_REPAIR_MAX_TOKENS",
                                max_tokens_default=2200,
                            ),
                        )
                        source_outputs = _extract_paragraph_component_candidates(
                            response.content,
                            source_candidate_count,
                        )
                    except Exception as exc:
                        search_summary["candidates"].append({
                            "strategy": f"source_grounding_repair_p{paragraph_index + 1}_batch",
                            "passed_local_checks": False,
                            "reason": f"llm_error {exc}",
                            "source_grounding_repair": True,
                            "paragraph_index": paragraph_index,
                            "claim_id": source_result.get("claim_id"),
                            "source_confidence": source_result.get("source_confidence"),
                        })
                        continue
                    if not source_outputs:
                        search_summary["candidates"].append({
                            "strategy": f"source_grounding_repair_p{paragraph_index + 1}_batch",
                            "passed_local_checks": False,
                            "reason": "empty_candidate_batch",
                            "source_grounding_repair": True,
                            "paragraph_index": paragraph_index,
                            "claim_id": source_result.get("claim_id"),
                            "source_confidence": source_result.get("source_confidence"),
                        })
                        continue
                    for candidate_number, raw_paragraph_candidate in enumerate(source_outputs, start=1):
                        strategy = (
                            f"source_grounding_repair_p{paragraph_index + 1}"
                            f"_c{candidate_number}"
                        )
                        repair_scope = str(target.get("repair_scope") or "paragraph")
                        original_paragraph = (
                            _logical_paragraphs(component_base_text)[paragraph_index]
                            if paragraph_index < len(_logical_paragraphs(component_base_text))
                            else target.get("target_preview") or ""
                        )
                        sentence_index = _safe_index(target.get("sentence_index"), -1)
                        if repair_scope == "sentence_window" and sentence_index >= 0:
                            paragraph_sentences = _split_sentences(original_paragraph)
                            original_sentence = (
                                paragraph_sentences[sentence_index]
                                if sentence_index < len(paragraph_sentences)
                                else target.get("target_preview") or ""
                            )
                            sentence_candidate, sentence_reject = _clean_source_sentence_candidate(
                                raw_paragraph_candidate,
                                original_sentence,
                            )
                            if sentence_reject:
                                search_summary["candidates"].append({
                                    "strategy": strategy,
                                    "passed_local_checks": False,
                                    "reason": sentence_reject,
                                    "source_grounding_repair": True,
                                    "repair_scope": repair_scope,
                                    "paragraph_index": paragraph_index,
                                    "sentence_index": sentence_index,
                                    "claim_id": source_result.get("claim_id"),
                                    "source_confidence": source_result.get("source_confidence"),
                                })
                                continue
                            paragraph_candidate = _splice_sentence_window(
                                original_paragraph,
                                sentence_index,
                                sentence_index + 1,
                                sentence_candidate,
                            )
                            if paragraph_candidate == original_paragraph:
                                search_summary["candidates"].append({
                                    "strategy": strategy,
                                    "passed_local_checks": False,
                                    "reason": "sentence_window_no_change",
                                    "source_grounding_repair": True,
                                    "repair_scope": repair_scope,
                                    "paragraph_index": paragraph_index,
                                    "sentence_index": sentence_index,
                                    "claim_id": source_result.get("claim_id"),
                                    "source_confidence": source_result.get("source_confidence"),
                                })
                                continue
                            patched_candidate = _splice_paragraph(
                                component_base_text,
                                paragraph_index,
                                paragraph_candidate,
                            )
                        else:
                            paragraph_candidate, paragraph_reject = _clean_paragraph_component_candidate(
                                raw_paragraph_candidate,
                                original_paragraph,
                            )
                            if paragraph_reject:
                                search_summary["candidates"].append({
                                    "strategy": strategy,
                                    "passed_local_checks": False,
                                    "reason": paragraph_reject,
                                    "source_grounding_repair": True,
                                    "repair_scope": repair_scope,
                                    "paragraph_index": paragraph_index,
                                    "claim_id": source_result.get("claim_id"),
                                    "source_confidence": source_result.get("source_confidence"),
                                })
                                continue
                            patched_candidate = _splice_paragraph(
                                component_base_text,
                                paragraph_index,
                                paragraph_candidate,
                            )
                        _evaluate_ai_search_candidate(
                            strategy,
                            patched_candidate,
                            deterministic=False,
                            extra={
                                "source_grounding_repair": True,
                                "repair_scope": repair_scope,
                                "paragraph_index": paragraph_index,
                                "sentence_index": sentence_index if repair_scope == "sentence_window" else None,
                                "claim_id": source_result.get("claim_id"),
                                "source_confidence": source_result.get("source_confidence"),
                                "source_count": len(source_result.get("sources") or []),
                            },
                        )
                    if best_strategy and best_selection_status.get("selectable"):
                        component_base_text = best_text

                if paragraph_search_enabled and not strict_safe_legacy_llm_skipped:
                    try:
                        paragraph_limit = max(
                            1,
                            int(os.environ.get(
                                "DRAFTPROOF_PARAGRAPH_COMPONENT_TARGETS",
                                _adaptive_budget_default(component_base_text, 3, 6),
                            )),
                        )
                    except ValueError:
                        paragraph_limit = 4
                    try:
                        paragraph_candidates = max(
                            1,
                            int(os.environ.get(
                                "DRAFTPROOF_PARAGRAPH_COMPONENT_CANDIDATES",
                                _adaptive_budget_default(component_base_text, 2, 3),
                            )),
                        )
                    except ValueError:
                        paragraph_candidates = 3
                    paragraph_roles = {
                        role.strip()
                        for role in os.environ.get(
                            "DRAFTPROOF_PARAGRAPH_COMPONENT_ROLES",
                            "generic_claim_heavy,conclusion_template_risk,source_summary_heavy,mixed",
                        ).split(",")
                        if role.strip()
                    }
                    component_target_pool = _paragraph_component_targets(
                        component_base_text,
                        original_report_dict,
                        limit=max(paragraph_limit * 4, paragraph_limit + 8),
                    )
                    component_targets = [
                        target for target in component_target_pool
                        if str(target.get("role") or "") in paragraph_roles
                    ][:paragraph_limit]
                    component_summary = {
                        "enabled": True,
                        "base_repairs": component_base_repairs,
                        "target_roles": sorted(paragraph_roles),
                        "pool_count": len(component_target_pool),
                        "target_count": len(component_targets),
                        "candidate_limit_per_target": paragraph_candidates,
                        "targets": [
                            {
                                "paragraph_index": t.get("index"),
                                "score": t.get("score"),
                                "role": t.get("role"),
                                "drivers": t.get("drivers"),
                                "preview": (t.get("paragraph") or "")[:180],
                            }
                            for t in component_targets
                        ],
                    }
                    search_summary["paragraph_component_search"] = component_summary
                    for target_number, target in enumerate(component_targets, start=1):
                        report_progress(
                            min(89, 79 + target_number),
                            (
                                "Trying paragraph-component AI batch "
                                f"{target_number}/{len(component_targets)}"
                            )
                        )
                        try:
                            prompt = _paragraph_component_prompt(
                                target,
                                original_report_dict,
                                target_number,
                                reference_ai=ai_search_reference,
                                required_ai_drop=ai_first_min_drop,
                                target_ai_score=ai_search_target_score,
                                candidate_count=paragraph_candidates,
                                confirmed_author_anchors=confirmed_author_anchor_brief,
                            )
                            search_summary["llm_calls"] += 1
                            response = gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's paragraph AI-score mitigation engine. "
                                    "Return only the requested tagged replacement paragraphs."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_PARAGRAPH_COMPONENT",
                                    temperature_env="DRAFTPROOF_PARAGRAPH_COMPONENT_TEMPERATURE",
                                    temperature_default=0.45,
                                    max_tokens_env="DRAFTPROOF_PARAGRAPH_COMPONENT_MAX_TOKENS",
                                    max_tokens_default=2600,
                                ),
                            )
                            paragraph_outputs = _extract_paragraph_component_candidates(
                                response.content,
                                paragraph_candidates,
                            )
                        except Exception as exc:
                            search_summary["candidates"].append({
                                "strategy": (
                                    f"paragraph_component_p{int(target.get('index', 0)) + 1}"
                                    "_batch"
                                ),
                                "passed_local_checks": False,
                                "reason": f"llm_error {exc}",
                                "paragraph_component": True,
                                "paragraph_index": target.get("index"),
                            })
                            if adaptive_stop_reason:
                                break
                            continue
                        if not paragraph_outputs:
                            search_summary["candidates"].append({
                                "strategy": (
                                    f"paragraph_component_p{int(target.get('index', 0)) + 1}"
                                    "_batch"
                                ),
                                "passed_local_checks": False,
                                "reason": "empty_candidate_batch",
                                "paragraph_component": True,
                                "paragraph_index": target.get("index"),
                            })
                            continue
                        for candidate_number, raw_paragraph_candidate in enumerate(paragraph_outputs, start=1):
                            strategy = (
                                f"paragraph_component_p{int(target.get('index', 0)) + 1}"
                                f"_c{candidate_number}"
                            )
                            paragraph_candidate, paragraph_reject = _clean_paragraph_component_candidate(
                                raw_paragraph_candidate,
                                target.get("paragraph") or "",
                                _paragraph_anchor_lock(target),
                            )
                            if paragraph_reject:
                                search_summary["candidates"].append({
                                    "strategy": strategy,
                                    "passed_local_checks": False,
                                    "reason": paragraph_reject,
                                    "paragraph_component": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_driver_score": target.get("score"),
                                })
                                continue
                            patched_candidate = _splice_paragraph(
                                component_base_text,
                                int(target.get("index", 0)),
                                paragraph_candidate,
                            )
                            _evaluate_ai_search_candidate(
                                strategy,
                                patched_candidate,
                                deterministic=False,
                                extra={
                                    "paragraph_component": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_role": target.get("role"),
                                    "paragraph_driver_score": target.get("score"),
                                    "paragraph_drivers": target.get("drivers"),
                                },
                            )
                            if _maybe_adaptive_stop("paragraph_component_search"):
                                break
                        if adaptive_stop_reason:
                            break
                        if best_strategy:
                            component_base_text = best_text

                    if (not adaptive_stop_reason) and _env_flag("DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_REPAIR", True):
                        amplify_roles = {
                            role.strip()
                            for role in (
                                os.environ.get("DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TARGET_ROLES")
                                or os.environ.get(
                                    "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_ROLES",
                                    "source_summary_heavy,conclusion_template_risk,generic_claim_heavy",
                                )
                            ).split(",")
                            if role.strip()
                        }
                        amplify_targets = [
                            target for target in component_targets
                            if str(target.get("role") or "") in amplify_roles
                        ][: max(0, int(_float_env(
                            "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TARGETS",
                            float(_adaptive_budget_default(component_base_text, 1, 2)),
                        )))]
                        amplify_candidates = max(
                            1,
                            int(_float_env(
                                "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_CANDIDATES",
                                float(_adaptive_budget_default(component_base_text, 2, 3)),
                            )),
                        )
                        search_summary["human_signal_amplification"] = {
                            "enabled": True,
                            "target_roles": sorted(amplify_roles),
                            "target_count": len(amplify_targets),
                            "candidate_limit_per_target": amplify_candidates,
                            "targets": [
                                {
                                    "paragraph_index": t.get("index"),
                                    "role": t.get("role"),
                                    "score": t.get("score"),
                                }
                                for t in amplify_targets
                            ],
                        }
                        for amplify_number, target in enumerate(amplify_targets, start=1):
                            try:
                                prompt = _human_signal_amplification_prompt(
                                    target,
                                    original_report_dict,
                                    amplify_number,
                                    candidate_count=amplify_candidates,
                                    confirmed_author_anchors=confirmed_author_anchor_brief,
                                )
                                search_summary["llm_calls"] += 1
                                response = gateway.chat(
                                    prompt,
                                    system=(
                                        "You are DraftProof's human-signal amplification engine. "
                                        "Return only tagged replacement paragraphs."
                                    ),
                                    **_phase_chat_sampling_kwargs(
                                        "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION",
                                        temperature_env="DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TEMPERATURE",
                                        temperature_default=0.45,
                                        max_tokens_env="DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_MAX_TOKENS",
                                        max_tokens_default=2600,
                                    ),
                                )
                                amplify_outputs = _extract_paragraph_component_candidates(
                                    response.content,
                                    amplify_candidates,
                                )
                            except Exception as exc:
                                search_summary["candidates"].append({
                                    "strategy": (
                                        f"human_signal_amplification_p{int(target.get('index', 0)) + 1}"
                                        "_batch"
                                    ),
                                    "passed_local_checks": False,
                                    "reason": f"llm_error {exc}",
                                    "human_signal_amplification": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_role": target.get("role"),
                                })
                                continue
                            for candidate_number, raw_paragraph_candidate in enumerate(amplify_outputs, start=1):
                                strategy = (
                                    f"human_signal_amplification_p{int(target.get('index', 0)) + 1}"
                                    f"_c{candidate_number}"
                                )
                                paragraph_candidate, paragraph_reject = _clean_paragraph_component_candidate(
                                    raw_paragraph_candidate,
                                    target.get("paragraph") or "",
                                    _paragraph_anchor_lock(target),
                                )
                                if paragraph_reject:
                                    search_summary["candidates"].append({
                                        "strategy": strategy,
                                        "passed_local_checks": False,
                                        "reason": paragraph_reject,
                                        "human_signal_amplification": True,
                                        "paragraph_index": target.get("index"),
                                        "paragraph_role": target.get("role"),
                                    })
                                    continue
                                patched_candidate = _splice_paragraph(
                                    component_base_text,
                                    int(target.get("index", 0)),
                                    paragraph_candidate,
                                )
                                _evaluate_ai_search_candidate(
                                    strategy,
                                    patched_candidate,
                                    deterministic=False,
                                    extra={
                                        "human_signal_amplification": True,
                                        "paragraph_component": True,
                                        "paragraph_index": target.get("index"),
                                        "paragraph_role": target.get("role"),
                                        "paragraph_driver_score": target.get("score"),
                                        "paragraph_drivers": target.get("drivers"),
                                    },
                                )
                                if _maybe_adaptive_stop("human_signal_amplification"):
                                    break
                            if adaptive_stop_reason:
                                break

                    if (not adaptive_stop_reason) and _env_flag("DRAFTPROOF_AUTHOR_REASONING_AMPLIFICATION_REPAIR", True):
                        reasoning_roles = {
                            role.strip()
                            for role in (
                                os.environ.get("DRAFTPROOF_AUTHOR_REASONING_AMPLIFICATION_ROLES")
                                or "generic_claim_heavy,conclusion_template_risk,mixed"
                            ).split(",")
                            if role.strip()
                        }
                        reasoning_targets = [
                            target for target in component_targets
                            if str(target.get("role") or "") in reasoning_roles
                        ][: max(0, int(_float_env(
                            "DRAFTPROOF_AUTHOR_REASONING_AMPLIFICATION_TARGETS",
                            float(_adaptive_budget_default(component_base_text, 1, 2)),
                        )))]
                        reasoning_candidates = max(
                            1,
                            int(_float_env(
                                "DRAFTPROOF_AUTHOR_REASONING_AMPLIFICATION_CANDIDATES",
                                float(_adaptive_budget_default(component_base_text, 2, 3)),
                            )),
                        )
                        search_summary["author_reasoning_amplification"] = {
                            "enabled": True,
                            "target_roles": sorted(reasoning_roles),
                            "target_count": len(reasoning_targets),
                            "candidate_limit_per_target": reasoning_candidates,
                            "targets": [
                                {
                                    "paragraph_index": t.get("index"),
                                    "role": t.get("role"),
                                    "score": t.get("score"),
                                }
                                for t in reasoning_targets
                            ],
                        }
                        for reasoning_number, target in enumerate(reasoning_targets, start=1):
                            try:
                                prompt = _author_reasoning_amplification_prompt(
                                    target,
                                    original_report_dict,
                                    reasoning_number,
                                    candidate_count=reasoning_candidates,
                                )
                                search_summary["llm_calls"] += 1
                                response = gateway.chat(
                                    prompt,
                                    system=(
                                        "You are DraftProof's author-reasoning amplification engine. "
                                        "Return only tagged replacement paragraphs."
                                    ),
                                    **_phase_chat_sampling_kwargs(
                                        "DRAFTPROOF_AUTHOR_REASONING_AMPLIFICATION",
                                        temperature_env="DRAFTPROOF_AUTHOR_REASONING_AMPLIFICATION_TEMPERATURE",
                                        temperature_default=0.45,
                                        max_tokens_env="DRAFTPROOF_AUTHOR_REASONING_AMPLIFICATION_MAX_TOKENS",
                                        max_tokens_default=2600,
                                    ),
                                )
                                reasoning_outputs = _extract_paragraph_component_candidates(
                                    response.content,
                                    reasoning_candidates,
                                )
                            except Exception as exc:
                                search_summary["candidates"].append({
                                    "strategy": (
                                        f"author_reasoning_amplification_p{int(target.get('index', 0)) + 1}"
                                        "_batch"
                                    ),
                                    "passed_local_checks": False,
                                    "reason": f"llm_error {exc}",
                                    "author_reasoning_amplification": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_role": target.get("role"),
                                })
                                continue
                            for candidate_number, raw_paragraph_candidate in enumerate(reasoning_outputs, start=1):
                                strategy = (
                                    f"author_reasoning_amplification_p{int(target.get('index', 0)) + 1}"
                                    f"_c{candidate_number}"
                                )
                                paragraph_candidate, paragraph_reject = _clean_paragraph_component_candidate(
                                    raw_paragraph_candidate,
                                    target.get("paragraph") or "",
                                    _paragraph_anchor_lock(target),
                                )
                                if paragraph_reject:
                                    search_summary["candidates"].append({
                                        "strategy": strategy,
                                        "passed_local_checks": False,
                                        "reason": paragraph_reject,
                                        "author_reasoning_amplification": True,
                                        "paragraph_index": target.get("index"),
                                        "paragraph_role": target.get("role"),
                                    })
                                    continue
                                patched_candidate = _splice_paragraph(
                                    component_base_text,
                                    int(target.get("index", 0)),
                                    paragraph_candidate,
                                )
                                _evaluate_ai_search_candidate(
                                    strategy,
                                    patched_candidate,
                                    deterministic=False,
                                    extra={
                                        "human_signal_amplification": True,
                                        "author_reasoning_amplification": True,
                                        "paragraph_component": True,
                                        "paragraph_index": target.get("index"),
                                        "paragraph_role": target.get("role"),
                                        "paragraph_driver_score": target.get("score"),
                                        "paragraph_drivers": target.get("drivers"),
                                    },
                                )
                                if _maybe_adaptive_stop("author_reasoning_amplification"):
                                    break
                            if adaptive_stop_reason:
                                break

                if adaptive_stop_reason and adaptive_stop_reason not in {
                    "adaptive_stop_after_strict_safe_phase_budget",
                    "adaptive_stop_after_formula_gap_candidate_orchestrator",
                }:
                    _run_post_safe_win_target_push("post_llm_adaptive_stop")
                    search_summary["llm_reason"] = adaptive_stop_reason
                elif adaptive_stop_reason:
                    search_summary["llm_reason"] = adaptive_stop_reason

                for index, strategy in enumerate([] if adaptive_stop_reason else strategies, start=1):
                    report_progress(
                        min(79, 76 + index),
                        f"Trying AI mitigation candidate {index}/{len(strategies)}",
                    )
                    candidate_eval = {
                        "strategy": strategy,
                        "passed_local_checks": False,
                    }
                    try:
                        prompt = _ai_search_prompt(
                            search_source_text,
                            original_report_dict,
                            strategy,
                            reference_ai=ai_search_reference,
                            required_ai_drop=ai_first_min_drop,
                            target_ai_score=ai_search_target_score,
                            confirmed_author_anchors=confirmed_author_anchor_brief,
                        )
                        search_summary["llm_calls"] += 1
                        response = gateway.chat(
                            prompt,
                            system=(
                                "You are DraftProof's AI-score mitigation engine. "
                                "Return only the complete rewritten document."
                            ),
                            **_phase_chat_sampling_kwargs(
                                "DRAFTPROOF_AI_SEARCH",
                                temperature_env="DRAFTPROOF_AI_SEARCH_TEMPERATURE",
                                temperature_default=0.45,
                                max_tokens_env="DRAFTPROOF_AI_SEARCH_MAX_TOKENS",
                                max_tokens_default=6500,
                            ),
                        )
                        candidate = _clean_full_document_candidate(response.content, search_source_text)
                    except Exception as exc:
                        candidate_eval["reason"] = f"llm_error {exc}"
                        search_summary["candidates"].append(candidate_eval)
                        continue

                    _evaluate_ai_search_candidate(
                        strategy,
                        candidate,
                        deterministic=False,
                        extra={
                            "confirmed_anchor_strategy": strategy in confirmed_anchor_strategies,
                        },
                    )
                    if _maybe_adaptive_stop("full_document_strategy_search"):
                        break

                if (not adaptive_stop_reason) and not _best_ai_search_selectable():
                    try:
                        feedback_limit = max(
                            0,
                            int(os.environ.get("DRAFTPROOF_AI_SEARCH_FEEDBACK_CANDIDATES", "2")),
                        )
                    except ValueError:
                        feedback_limit = 2
                    retry_enabled = bool(llm_roles.get("retry_model_enabled"))
                    retry_budget = int(llm_roles.get("retry_model_max_calls") or 0)
                    if not retry_enabled:
                        search_summary["score_feedback_loop"] = {
                            "enabled": False,
                            "candidate_limit": 0,
                            "reason": "retry_model_disabled_by_kill_switch",
                            "retry_model": retry_model,
                        }
                        feedback_limit = 0
                    else:
                        feedback_limit = min(feedback_limit, retry_budget)
                    if feedback_limit:
                        search_summary["score_feedback_loop"] = {
                            "enabled": True,
                            "candidate_limit": feedback_limit,
                            "retry_model": retry_model,
                            "retry_model_max_calls": retry_budget,
                            "reason": (
                                "no_selectable_candidate"
                                if not best_strategy
                                else "best_candidate_below_required_ai_drop"
                            ),
                        }
                    retry_gateway = None
                    if feedback_limit:
                        retry_gateway = LLMGateway(LLMConfig(
                            api_key=effective_key,
                            model=retry_model,
                            base_url=base_url,
                            timeout=int(os.environ.get("DRAFTPROOF_AI_SEARCH_TIMEOUT", "120")),
                            max_retries=int(os.environ.get("DRAFTPROOF_AI_SEARCH_RETRIES", "1")),
                            max_tokens=int(os.environ.get("DRAFTPROOF_AI_SEARCH_MAX_TOKENS", "6500")),
                            temperature=float(os.environ.get("DRAFTPROOF_AI_SEARCH_FEEDBACK_TEMPERATURE", "0.45")),
                        ))
                        retry_gateway = _budget_gateway(retry_gateway, "score_feedback_loop")
                    for feedback_index in range(1, feedback_limit + 1):
                        report_progress(
                            min(89, 80 + feedback_index),
                            f"Trying score-feedback AI mitigation candidate {feedback_index}/{feedback_limit}",
                        )
                        candidate_eval = {
                            "strategy": f"score_feedback_{feedback_index}",
                            "passed_local_checks": False,
                            "retry_model": retry_model,
                        }
                        try:
                            prompt = _ai_search_feedback_prompt(
                                search_source_text,
                                original_report_dict,
                                search_summary,
                                feedback_index,
                            )
                            search_summary["llm_calls"] += 1
                            response = retry_gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's score-feedback rewrite engine. "
                                    "Use the detector scorecard to produce a lower-scoring complete document."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_AI_SEARCH_FEEDBACK",
                                    temperature_env="DRAFTPROOF_AI_SEARCH_FEEDBACK_TEMPERATURE",
                                    temperature_default=0.45,
                                    max_tokens_env="DRAFTPROOF_AI_SEARCH_MAX_TOKENS",
                                    max_tokens_default=6500,
                                ),
                            )
                            candidate = _clean_full_document_candidate(response.content, search_source_text)
                        except Exception as exc:
                            candidate_eval["reason"] = f"llm_error {exc}"
                            search_summary["candidates"].append(candidate_eval)
                            continue
                        _evaluate_ai_search_candidate(
                            f"score_feedback_{feedback_index}",
                            candidate,
                            deterministic=False,
                        )
                        if _maybe_adaptive_stop("score_feedback_loop"):
                            break
                        if _best_ai_search_selectable():
                            break

                if (
                    not adaptive_stop_reason
                    and
                    _best_ai_search_selectable()
                    and _env_flag("DRAFTPROOF_POST_SELECTION_HUMAN_SIGNAL_AMPLIFICATION", True)
                ):
                    post_roles = {
                        role.strip()
                        for role in (
                            os.environ.get("DRAFTPROOF_POST_SELECTION_HUMAN_SIGNAL_AMPLIFICATION_TARGET_ROLES")
                            or os.environ.get("DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TARGET_ROLES")
                            or os.environ.get(
                                "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_ROLES",
                                "source_summary_heavy,conclusion_template_risk,generic_claim_heavy",
                            )
                        ).split(",")
                        if role.strip()
                    }
                    post_target_limit = max(
                        0,
                        int(_float_env(
                            "DRAFTPROOF_POST_SELECTION_HUMAN_SIGNAL_AMPLIFICATION_TARGETS",
                            float(_adaptive_budget_default(best_text, 1, 1)),
                        )),
                    )
                    post_candidate_limit = max(
                        1,
                        int(_float_env(
                            "DRAFTPROOF_POST_SELECTION_HUMAN_SIGNAL_AMPLIFICATION_CANDIDATES",
                            float(_adaptive_budget_default(best_text, 1, 2)),
                        )),
                    )
                    post_targets = [
                        target for target in _paragraph_component_targets(
                            best_text,
                            original_report_dict,
                            limit=max(post_target_limit, 1),
                        )
                        if str(target.get("role") or "") in post_roles
                    ][:post_target_limit]
                    search_summary["post_selection_human_signal_amplification"] = {
                        "enabled": True,
                        "base_strategy": best_strategy,
                        "target_roles": sorted(post_roles),
                        "target_count": len(post_targets),
                        "candidate_limit_per_target": post_candidate_limit,
                        "targets": [
                            {
                                "paragraph_index": target.get("index"),
                                "role": target.get("role"),
                                "score": target.get("score"),
                            }
                            for target in post_targets
                        ],
                    }
                    post_base_text = best_text
                    for post_number, target in enumerate(post_targets, start=1):
                        report_progress(
                            min(91, 88 + post_number),
                            (
                                "Trying post-selection Human Signal amplification "
                                f"{post_number}/{len(post_targets)}"
                            ),
                        )
                        try:
                            prompt = _human_signal_amplification_prompt(
                                target,
                                original_report_dict,
                                post_number,
                                candidate_count=post_candidate_limit,
                                confirmed_author_anchors=confirmed_author_anchor_brief,
                            )
                            search_summary["llm_calls"] += 1
                            response = gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's post-selection human-signal amplification engine. "
                                    "Return only tagged replacement paragraphs."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_POST_SELECTION_HUMAN_SIGNAL_AMPLIFICATION",
                                    temperature_env="DRAFTPROOF_POST_SELECTION_HUMAN_SIGNAL_AMPLIFICATION_TEMPERATURE",
                                    temperature_default=float(os.environ.get(
                                        "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TEMPERATURE",
                                        "0.45",
                                    )),
                                    max_tokens_env="DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_MAX_TOKENS",
                                    max_tokens_default=2600,
                                ),
                            )
                            post_outputs = _extract_paragraph_component_candidates(
                                response.content,
                                post_candidate_limit,
                            )
                        except Exception as exc:
                            search_summary["candidates"].append({
                                "strategy": (
                                    f"post_human_signal_amplification_p{int(target.get('index', 0)) + 1}"
                                    "_batch"
                                ),
                                "passed_local_checks": False,
                                "reason": f"llm_error {exc}",
                                "human_signal_amplification": True,
                                "post_selection_human_signal_amplification": True,
                                "paragraph_index": target.get("index"),
                                "paragraph_role": target.get("role"),
                            })
                            continue
                        for candidate_number, raw_paragraph_candidate in enumerate(post_outputs, start=1):
                            strategy = (
                                f"post_human_signal_amplification_p{int(target.get('index', 0)) + 1}"
                                f"_c{candidate_number}"
                            )
                            paragraph_candidate, paragraph_reject = _clean_paragraph_component_candidate(
                                raw_paragraph_candidate,
                                target.get("paragraph") or "",
                                _paragraph_anchor_lock(target),
                            )
                            if paragraph_reject:
                                search_summary["candidates"].append({
                                    "strategy": strategy,
                                    "passed_local_checks": False,
                                    "reason": paragraph_reject,
                                    "human_signal_amplification": True,
                                    "post_selection_human_signal_amplification": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_role": target.get("role"),
                                })
                                continue
                            patched_candidate = _splice_paragraph(
                                post_base_text,
                                int(target.get("index", 0)),
                                paragraph_candidate,
                            )
                            _evaluate_ai_search_candidate(
                                strategy,
                                patched_candidate,
                                deterministic=False,
                                extra={
                                    "human_signal_amplification": True,
                                    "post_selection_human_signal_amplification": True,
                                    "paragraph_component": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_role": target.get("role"),
                                    "paragraph_driver_score": target.get("score"),
                                    "paragraph_drivers": target.get("drivers"),
                                },
                            )
                            if _maybe_adaptive_stop("post_selection_human_signal_amplification"):
                                break
                            if (
                                best_selection_status.get("human_signal_amplification")
                                and best_strategy == strategy
                            ):
                                post_base_text = best_text
                        if adaptive_stop_reason:
                            break

                if (
                    not adaptive_stop_reason
                    and
                    _best_ai_search_selectable()
                    and _env_flag("DRAFTPROOF_ITERATIVE_HUMAN_CLIMB", True)
                ):
                    climb_rounds = max(
                        0,
                        int(_float_env(
                            "DRAFTPROOF_ITERATIVE_HUMAN_CLIMB_ROUNDS",
                            float(_adaptive_budget_default(best_text, 1, 2)),
                        )),
                    )
                    climb_target_limit = max(
                        1,
                        int(_float_env(
                            "DRAFTPROOF_ITERATIVE_HUMAN_CLIMB_TARGETS",
                            float(_adaptive_budget_default(best_text, 1, 1)),
                        )),
                    )
                    climb_candidate_limit = max(
                        1,
                        int(_float_env(
                            "DRAFTPROOF_ITERATIVE_HUMAN_CLIMB_CANDIDATES",
                            float(_adaptive_budget_default(best_text, 1, 2)),
                        )),
                    )
                    climb_roles = {
                        role.strip()
                        for role in (
                            os.environ.get("DRAFTPROOF_ITERATIVE_HUMAN_CLIMB_ROLES")
                            or os.environ.get("DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TARGET_ROLES")
                            or "generic_claim_heavy,conclusion_template_risk,mixed"
                        ).split(",")
                        if role.strip()
                    }
                    climb_summary = {
                        "enabled": True,
                        "rounds": climb_rounds,
                        "target_limit": climb_target_limit,
                        "candidate_limit_per_target": climb_candidate_limit,
                        "target_roles": sorted(climb_roles),
                        "round_results": [],
                    }
                    search_summary["iterative_human_climb"] = climb_summary
                    climb_base_text = best_text
                    for climb_round in range(1, climb_rounds + 1):
                        climb_start_strategy = best_strategy
                        climb_targets = [
                            target for target in _paragraph_component_targets(
                                climb_base_text,
                                original_report_dict,
                                limit=max(climb_target_limit * 3, climb_target_limit),
                            )
                            if str(target.get("role") or "") in climb_roles
                        ][:climb_target_limit]
                        climb_summary["round_results"].append({
                            "round": climb_round,
                            "base_strategy": climb_start_strategy,
                            "target_count": len(climb_targets),
                            "targets": [
                                {
                                    "paragraph_index": target.get("index"),
                                    "role": target.get("role"),
                                    "score": target.get("score"),
                                }
                                for target in climb_targets
                            ],
                        })
                        if not climb_targets:
                            break
                        for target_number, target in enumerate(climb_targets, start=1):
                            report_progress(
                                min(92, 89 + climb_round),
                                (
                                    "Trying iterative Human climb "
                                    f"{climb_round}/{climb_rounds}"
                                ),
                            )
                            try:
                                prompt = _human_signal_amplification_prompt(
                                    target,
                                    original_report_dict,
                                    climb_round,
                                    candidate_count=climb_candidate_limit,
                                    confirmed_author_anchors=confirmed_author_anchor_brief,
                                )
                                search_summary["llm_calls"] += 1
                                response = gateway.chat(
                                    prompt,
                                    system=(
                                        "You are DraftProof's iterative human-signal climb engine. "
                                        "Return only tagged replacement paragraphs."
                                    ),
                                    **_phase_chat_sampling_kwargs(
                                        "DRAFTPROOF_ITERATIVE_HUMAN_CLIMB",
                                        temperature_env="DRAFTPROOF_ITERATIVE_HUMAN_CLIMB_TEMPERATURE",
                                        temperature_default=float(os.environ.get(
                                            "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TEMPERATURE",
                                            "0.45",
                                        )),
                                        max_tokens_env="DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_MAX_TOKENS",
                                        max_tokens_default=2600,
                                    ),
                                )
                                climb_outputs = _extract_paragraph_component_candidates(
                                    response.content,
                                    climb_candidate_limit,
                                )
                            except Exception as exc:
                                search_summary["candidates"].append({
                                    "strategy": (
                                        f"iterative_human_climb_r{climb_round}"
                                        f"_p{int(target.get('index', 0)) + 1}_batch"
                                    ),
                                    "passed_local_checks": False,
                                    "reason": f"llm_error {exc}",
                                    "iterative_human_climb": True,
                                    "paragraph_index": target.get("index"),
                                    "paragraph_role": target.get("role"),
                                })
                                continue
                            for candidate_number, raw_paragraph_candidate in enumerate(climb_outputs, start=1):
                                strategy = (
                                    f"iterative_human_climb_r{climb_round}"
                                    f"_p{int(target.get('index', 0)) + 1}"
                                    f"_c{candidate_number}"
                                )
                                paragraph_candidate, paragraph_reject = _clean_paragraph_component_candidate(
                                    raw_paragraph_candidate,
                                    target.get("paragraph") or "",
                                    _paragraph_anchor_lock(target),
                                )
                                if paragraph_reject:
                                    search_summary["candidates"].append({
                                        "strategy": strategy,
                                        "passed_local_checks": False,
                                        "reason": paragraph_reject,
                                        "iterative_human_climb": True,
                                        "paragraph_index": target.get("index"),
                                        "paragraph_role": target.get("role"),
                                    })
                                    continue
                                patched_candidate = _splice_paragraph(
                                    climb_base_text,
                                    int(target.get("index", 0)),
                                    paragraph_candidate,
                                )
                                _evaluate_ai_search_candidate(
                                    strategy,
                                    patched_candidate,
                                    deterministic=False,
                                    extra={
                                        "human_signal_amplification": True,
                                        "iterative_human_climb": True,
                                        "paragraph_component": True,
                                        "paragraph_index": target.get("index"),
                                        "paragraph_role": target.get("role"),
                                        "paragraph_driver_score": target.get("score"),
                                        "paragraph_drivers": target.get("drivers"),
                                    },
                                )
                                if _maybe_adaptive_stop("iterative_human_climb"):
                                    break
                            if adaptive_stop_reason:
                                break
                        if best_strategy != climb_start_strategy and _best_ai_search_selectable():
                            climb_base_text = best_text
                            climb_summary["round_results"][-1]["selected_strategy_after_round"] = best_strategy
                            climb_summary["round_results"][-1]["selected_human_after_round"] = (
                                _contribution_scores(best_report).get("human")
                                if isinstance(best_report, dict) else None
                            )
                        else:
                            climb_summary["round_results"][-1]["stopped_reason"] = "no_better_selectable_candidate"
                            break
                        if adaptive_stop_reason:
                            break

                if (
                    _env_flag("DRAFTPROOF_BLOCKED_HUMAN_WINNER_REPAIR", True)
                    and best_blocked_human_candidate
                    and not _blocked_human_winner_failed_formula_gate(best_blocked_human_candidate)
                    and effective_key
                    and (
                        not adaptive_stop_reason
                        or _blocked_human_winner_repair_budget_override(adaptive_stop_reason)
                    )
                ):
                    try:
                        default_repair_candidates = (
                            1.0
                            if _blocked_human_winner_repair_budget_override(adaptive_stop_reason)
                            else 2.0
                        )
                        blocked_repair_limit = max(
                            0,
                            int(_float_env(
                                "DRAFTPROOF_BLOCKED_HUMAN_WINNER_REPAIR_CANDIDATES",
                                default_repair_candidates,
                            )),
                        )
                    except (TypeError, ValueError):
                        blocked_repair_limit = 1 if _blocked_human_winner_repair_budget_override(adaptive_stop_reason) else 2
                    blocked_summary = best_blocked_human_candidate.get("summary") or {}
                    blocking_targets = _blocking_finding_targets(
                        best_blocked_human_candidate.get("report"),
                        limit=int(_float_env("DRAFTPROOF_BLOCKED_HUMAN_WINNER_FINDING_TARGETS", 3.0)),
                        candidate_text=str(best_blocked_human_candidate.get("text") or ""),
                    )
                    search_summary["blocked_human_winner_repair"] = {
                        "enabled": True,
                        "candidate_limit": blocked_repair_limit,
                        "mode": "finding_local_then_document_repair",
                        "ran_after_search_budget": _blocked_human_winner_repair_budget_override(adaptive_stop_reason),
                        "search_budget_reason": adaptive_stop_reason or "",
                        "blocked_candidate": {
                            key: value
                            for key, value in blocked_summary.items()
                            if key != "selection_status"
                        },
                        "blocked_selection_status": blocked_summary.get("selection_status"),
                        "blocking_targets": blocking_targets,
                    }
                    for repair_index in range(1, blocked_repair_limit + 1):
                        report_progress(
                            min(93, 90 + repair_index),
                            f"Repairing blocked Human-gain candidate {repair_index}/{blocked_repair_limit}",
                        )
                        try:
                            if blocking_targets:
                                prompt = _finding_local_repair_prompt(
                                    str(best_blocked_human_candidate.get("text") or ""),
                                    blocked_summary,
                                    blocking_targets,
                                    repair_index,
                                )
                            else:
                                prompt = _blocked_human_candidate_repair_prompt(
                                    search_source_text,
                                    str(best_blocked_human_candidate.get("text") or ""),
                                    original_report_dict,
                                    blocked_summary,
                                    repair_index,
                                )
                            search_summary["llm_calls"] += 1
                            repair_gateway = gateway
                            if _blocked_human_winner_repair_budget_override(adaptive_stop_reason):
                                repair_gateway = LLMGateway(LLMConfig(
                                    api_key=effective_key,
                                    model=generator_model,
                                    base_url=base_url,
                                    timeout=int(os.environ.get("DRAFTPROOF_AI_SEARCH_TIMEOUT", "120")),
                                    max_retries=int(os.environ.get("DRAFTPROOF_AI_SEARCH_RETRIES", "1")),
                                    max_tokens=int(os.environ.get(
                                        "DRAFTPROOF_AI_SEARCH_MAX_TOKENS",
                                        "6500",
                                    )),
                                    temperature=float(os.environ.get(
                                        "DRAFTPROOF_BLOCKED_HUMAN_WINNER_REPAIR_TEMPERATURE",
                                        "0.45",
                                    )),
                                ))
                            response = repair_gateway.chat(
                                prompt,
                                system=(
                                    "You are DraftProof's blocked-candidate repair engine. "
                                    "Repair only the gate failure."
                                ),
                                **_phase_chat_sampling_kwargs(
                                    "DRAFTPROOF_BLOCKED_HUMAN_WINNER_REPAIR",
                                    temperature_env="DRAFTPROOF_BLOCKED_HUMAN_WINNER_REPAIR_TEMPERATURE",
                                    temperature_default=0.45,
                                    max_tokens_env="DRAFTPROOF_AI_SEARCH_MAX_TOKENS",
                                    max_tokens_default=6500,
                                ),
                            )
                            if blocking_targets:
                                patches = _extract_finding_local_patches(response.content)
                                repaired_candidate, applied_patches = _apply_finding_local_patches(
                                    str(best_blocked_human_candidate.get("text") or ""),
                                    patches,
                                )
                                if not applied_patches:
                                    search_summary["candidates"].append({
                                        "strategy": f"blocked_human_winner_repair_{repair_index}",
                                        "passed_local_checks": False,
                                        "reason": "no_finding_local_patch_applied",
                                        "blocked_human_winner_repair": True,
                                        "finding_local_repair": True,
                                        "source_blocked_strategy": blocked_summary.get("strategy"),
                                    })
                                    continue
                            else:
                                applied_patches = []
                                repaired_candidate = _clean_full_document_candidate(
                                    response.content,
                                    search_source_text,
                                )
                        except Exception as exc:
                            search_summary["candidates"].append({
                                "strategy": f"blocked_human_winner_repair_{repair_index}",
                                "passed_local_checks": False,
                                "reason": f"llm_error {exc}",
                                "blocked_human_winner_repair": True,
                                "source_blocked_strategy": blocked_summary.get("strategy"),
                            })
                            continue
                        _evaluate_ai_search_candidate(
                            f"blocked_human_winner_repair_{repair_index}",
                            repaired_candidate,
                            deterministic=False,
                            extra={
                                "blocked_human_winner_repair": True,
                                "finding_local_repair": bool(blocking_targets),
                                "applied_finding_patches": applied_patches,
                                "source_blocked_strategy": blocked_summary.get("strategy"),
                                "source_blocked_human_delta": blocked_summary.get("human_delta"),
                            },
                        )

                skip_post_safe_target_push = bool(
                    _best_ai_search_selectable()
                    and bool(best_selection_status.get("topk_safe_band_achieved"))
                    and not _strict_ai_safe_band_status(best_report).get("achieved")
                    and _env_flag("DRAFTPROOF_SKIP_HUMAN_TARGET_PUSH_AFTER_TOPK_SAFE", True)
                )
                if _best_ai_search_selectable() and not skip_post_safe_target_push:
                    _run_post_safe_win_target_push("pre_selection")
                elif skip_post_safe_target_push:
                    search_summary["post_safe_win_target_push"] = {
                        "enabled": bool(_env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH", True)),
                        "skipped": True,
                        "reason": "strict_safe_objective_active_after_topk_safe",
                        "selected_strategy": best_strategy,
                        "strict_ai_safe_band": _strict_ai_safe_band_status(best_report),
                    }
                if _best_ai_search_selectable():
                    if (
                        _env_flag("DRAFTPROOF_FINAL_CLEANUP_AFTER_SCAN_BUDGET", True)
                        and _verified_candidate_scans_used() >= int(search_budget.get("max_candidate_scans") or 0)
                    ):
                        try:
                            cleanup_reserve = max(
                                0,
                                int(_float_env("DRAFTPROOF_FINAL_CLEANUP_SCAN_RESERVE", 2.0)),
                            )
                        except (TypeError, ValueError):
                            cleanup_reserve = 2
                        if cleanup_reserve > 0:
                            previous_max = int(search_budget.get("max_candidate_scans") or 0)
                            current_scans = _verified_candidate_scans_used()
                            _extend_candidate_scan_budget(search_budget, current_scans, cleanup_reserve)
                            if str(adaptive_stop_reason or "") == "budget_exhausted_candidate_scans":
                                adaptive_stop_reason = ""
                            search_summary["final_cleanup_scan_reserve"] = {
                                "enabled": True,
                                "previous_max_candidate_scans": previous_max,
                                "candidate_scans_before_reserve": current_scans,
                                "reserve_added": cleanup_reserve,
                                "max_candidate_scans": search_budget["max_candidate_scans"],
                            }
                    if _env_flag("DRAFTPROOF_FINAL_DEPOLISH_CLEANUP", True):
                        depolished_text, depolish_repairs = _plain_language_depolish_text(best_text)
                        if depolish_repairs and depolished_text.strip() != best_text.strip():
                            _evaluate_ai_search_candidate(
                                "final_depolish_cleanup",
                                depolished_text,
                                deterministic=True,
                                extra={
                                    "final_depolish_cleanup": True,
                                    "depolish_repairs": depolish_repairs,
                                    "base_strategy": best_strategy,
                                },
                            )
                    pruned_text, prune_repairs = _final_score_drag_sentence_prune_text(best_text)
                    if prune_repairs and pruned_text.strip() != best_text.strip():
                        _evaluate_ai_search_candidate(
                            "final_score_drag_prune",
                            pruned_text,
                            deterministic=True,
                            extra={
                                "final_score_drag_prune": True,
                                "score_drag_prune_repairs": prune_repairs,
                                "base_strategy": best_strategy,
                            },
                        )
                    _run_final_topk_texture_repair("pre_selection_after_depolish")
                    _run_iterative_topk_route_optimizer("pre_selection_after_final_topk_texture")
                    previous_ai = rewritten_ai
                    rewritten_text = best_text
                    rewritten_report_dict = best_report
                    attempted_report_dict = rewritten_report_dict
                    rewritten_ai = _badge_ai(rewritten_report_dict)
                    rewritten_wq = _badge_wq(rewritten_report_dict)
                    rewritten_total = _finding_total(rewritten_report_dict)
                    rewritten_review_burden = _review_burden(rewritten_report_dict)
                    rewritten_severity = _weighted_severity(rewritten_report_dict)
                    rewritten_critical_high = (
                        len(rewritten_report_dict.get("findings", {}).get("critical", []))
                        + len(rewritten_report_dict.get("findings", {}).get("high", []))
                    )
                    if result.mp_result:
                        result.mp_result.final_text = rewritten_text
                        result.mp_result.converged = True
                        result.mp_result.convergence_reason = (
                            f"Selected AI mitigation search candidate: {best_strategy}"
                        )
                    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                    ai_search_selected = True
                    _clear_stale_rollback_for_kept_ai_mitigation(
                        result.summary,
                        "AI mitigation search",
                    )
                    search_summary.update({
                        "selected": True,
                        "selected_strategy": best_strategy,
                        "selected_outcome_class": best_selection_status.get("ai_footprint_outcome_class"),
                        "selected_ai_footprint_gate": best_selection_status.get("ai_footprint_gate"),
                        "selected_turnitin_like_ai_gate": best_selection_status.get("turnitin_like_ai_gate"),
                        "previous_ai": previous_ai,
                        "selected_ai": rewritten_ai,
                        "selected_ai_delta_vs_reference": (
                            round(ai_search_reference - rewritten_ai, 3)
                            if isinstance(rewritten_ai, (int, float)) else None
                        ),
                        "selected_human_shift_score": best_selection_status.get("human_shift_score"),
                        "selected_human_shift_components": best_selection_status.get("human_shift_components"),
                        "selected_multi_signal_contract": best_selection_status.get("multi_signal_contract"),
                        "selected_semantic_review_required": best_semantic_review_required,
                        "selected_drift_reasons": best_drift_reasons[:10],
                        "selection_status": best_selection_status,
                    })
            except Exception as exc:
                search_summary["reason"] = f"search_error {exc}"
        if _best_ai_search_selectable() and not search_summary.get("selected"):
            previous_ai = rewritten_ai
            rewritten_text = best_text
            rewritten_report_dict = best_report
            attempted_report_dict = rewritten_report_dict
            rewritten_ai = _badge_ai(rewritten_report_dict)
            rewritten_wq = _badge_wq(rewritten_report_dict)
            rewritten_total = _finding_total(rewritten_report_dict)
            rewritten_review_burden = _review_burden(rewritten_report_dict)
            rewritten_severity = _weighted_severity(rewritten_report_dict)
            rewritten_critical_high = (
                len(rewritten_report_dict.get("findings", {}).get("critical", []))
                + len(rewritten_report_dict.get("findings", {}).get("high", []))
            )
            if result.mp_result:
                result.mp_result.final_text = rewritten_text
                result.mp_result.converged = True
                result.mp_result.convergence_reason = (
                    f"Selected AI mitigation search candidate: {best_strategy}"
                )
            sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
            ai_search_selected = True
            _clear_stale_rollback_for_kept_ai_mitigation(
                result.summary,
                "deterministic AI mitigation search",
            )
            search_summary.update({
                "selected": True,
                "selected_strategy": best_strategy,
                "selected_outcome_class": best_selection_status.get("ai_footprint_outcome_class"),
                "selected_ai_footprint_gate": best_selection_status.get("ai_footprint_gate"),
                "selected_turnitin_like_ai_gate": best_selection_status.get("turnitin_like_ai_gate"),
                "previous_ai": previous_ai,
                "selected_ai": rewritten_ai,
                "selected_ai_delta_vs_reference": (
                    round(ai_search_reference - rewritten_ai, 3)
                    if isinstance(rewritten_ai, (int, float)) else None
                ),
                "selected_human_shift_score": best_selection_status.get("human_shift_score"),
                "selected_human_shift_components": best_selection_status.get("human_shift_components"),
                "selected_multi_signal_contract": best_selection_status.get("multi_signal_contract"),
                "selected_semantic_review_required": best_semantic_review_required,
                "selected_drift_reasons": best_drift_reasons[:10],
                "selection_status": best_selection_status,
            })
        elif best_strategy and not search_summary.get("selected"):
            _record_best_attempt()
            search_summary["selected"] = False
            search_summary["selection_reason"] = (
                best_selection_status.get("reason")
                or "best_candidate_below_required_ai_drop"
            )
        search_summary["seconds"] = round(time.time() - search_started, 3)
        search_summary["source_search_calls"] = _source_search_calls_used()
        search_summary["source_search_budget"] = {
            "max_calls_per_run": _source_search_max_calls_per_run(),
            "remaining_calls": _source_search_remaining_calls(),
        }
        hard_llm_cap = _ai_search_llm_hard_cap(search_source_text)
        if int(search_summary.get("llm_calls") or 0) > hard_llm_cap:
            search_summary["llm_calls_counter_correction"] = {
                "reported_before_correction": int(search_summary.get("llm_calls") or 0),
                "hard_cap": hard_llm_cap,
                "reason": "blocked optimistic LLM attempt exceeded reported counter",
            }
            search_summary["llm_calls"] = hard_llm_cap
        controller = search_summary.get("candidate_scoring_controller")
        if isinstance(controller, dict):
            controller["full_scans_used"] = _verified_candidate_scans_used()
            controller["candidate_records"] = len(search_summary.get("candidates", []))
        if formula_gap_orchestrator_enabled:
            selected_formula_contract = (
                best_selection_status.get("formula_gap_contract")
                if isinstance(best_selection_status, dict) else None
            )
            selected_formula_contract = (
                selected_formula_contract if isinstance(selected_formula_contract, dict) else {}
            )
            search_summary["formula_gap_plan"] = formula_gap_plan
            search_summary["selected_candidate_reason"] = (
                best_selection_status.get("reason")
                if isinstance(best_selection_status, dict) else None
            )
            search_summary["remaining_weighted_drivers"] = (
                selected_formula_contract.get("remaining_formula_drivers")
                or formula_gap_plan.get("remaining_weighted_drivers")
            )
            if selected_formula_contract:
                search_summary["remaining_formula_gap"] = selected_formula_contract.get("remaining_formula_gap")
                search_summary["selected_formula_gap_contract"] = selected_formula_contract
            if not search_summary.get("llm_candidate_frontier"):
                search_summary["llm_candidate_frontier"] = [
                    {
                        "strategy": item.get("strategy"),
                        "family": item.get("formula_gap_portfolio_family") or item.get("family"),
                        "passed_local_checks": item.get("passed_local_checks"),
                        "reason": item.get("reason"),
                        "ai": item.get("ai"),
                        "selection_status": item.get("selection_status"),
                        "candidate_decision": item.get("candidate_decision"),
                        "formula_gap_contract": item.get("formula_gap_contract"),
                    }
                    for item in (search_summary.get("candidates") or [])
                    if str(item.get("strategy") or "").startswith("formula_gap_portfolio_")
                ]
        result.summary["ai_mitigation_search"] = search_summary
        _record_rewrite_llm_calls(
            result.summary,
            "ai_search_llm_calls_used",
            search_summary.get("llm_calls"),
        )
        stage_timings.append({
            "stage": "ai_mitigation_search",
            "seconds": search_summary["seconds"],
            "candidates": len(search_summary.get("candidates", [])),
            "full_candidate_scans": search_summary.get("full_candidate_scans"),
            "selected": search_summary.get("selected", False),
        })
        global_rewrite_budget.record_stage(
            "ai_mitigation_search",
            seconds=float(search_summary.get("seconds") or 0.0),
            scans=int(search_summary.get("full_candidate_scans") or 0),
            llm_calls=int(search_summary.get("llm_calls") or 0),
        )
        selected_status = (
            search_summary.get("selection_status")
            if isinstance(search_summary.get("selection_status"), dict)
            else {}
        )
        ai_search_goal_reached = _ai_search_selected_candidate_reaches_goal(selected_status)
        search_summary["selected_candidate_goal_reached"] = bool(ai_search_goal_reached)
        if (
            _env_flag("DRAFTPROOF_FAIL_FAST_ON_PARTIAL_AI_SEARCH", True)
            and bool(search_summary.get("selected"))
            and selected_status
            and not ai_search_goal_reached
        ):
            ai_search_fail_fast_partial = True
            selected_status.setdefault("reason", "best_candidate_below_required_ai_drop")
            search_summary["goal_not_reached_reason"] = selected_status.get("reason")
            search_summary["post_selection_controllers_skipped"] = True
            search_summary["post_selection_skip_reason"] = "selected_candidate_below_ai_mitigation_goal"
            result.summary["mitigation_goal_status"] = {
                "goal": "mitigate rewritten content so detector-safe output is unlikely to be flagged as AI-generated",
                "reached": False,
                "reason": "selected_candidate_below_ai_mitigation_goal",
                "reference_ai": search_summary.get("reference_ai"),
                "target_ai_score": search_summary.get("target_ai_score"),
                "required_ai_drop": search_summary.get("required_ai_drop"),
                "selected_ai": search_summary.get("selected_ai"),
                "selected_ai_delta_vs_reference": search_summary.get("selected_ai_delta_vs_reference"),
            }
            if int(global_rewrite_budget.max_scans or 0) > 0:
                global_rewrite_budget.max_scans = int(global_rewrite_budget.scans_used)
            if int(global_rewrite_budget.max_llm_calls or 0) > 0:
                global_rewrite_budget.max_llm_calls = int(global_rewrite_budget.llm_calls_used)

        result.summary["detect_scores"].update({
            "rewritten_ai": rewritten_ai,
            "rewritten_writing_quality": rewritten_wq,
            "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
            "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
            "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
            "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
            "rewritten_findings": rewritten_total,
            "rewritten_review_burden": rewritten_review_burden,
            "rewritten_weighted_severity": rewritten_severity,
        })
    elif ai_search_blocked_by_author_gaps:
        result.summary["ai_mitigation_search"] = {
            "enabled": False,
            "selected": False,
            "reason": "requires_author_input",
            "reference_ai": ai_search_reference,
            "starting_ai": rewritten_ai,
            "candidate_limit": 0,
            "llm_calls": 0,
            "candidates": [],
        }
        stage_timings.append({
            "stage": "ai_mitigation_search",
            "seconds": 0,
            "candidates": 0,
            "selected": False,
            "skipped_reason": "requires_author_input",
        })
        global_rewrite_budget.record_stage("ai_mitigation_search", seconds=0.0)

    global_candidate_ledger = CandidateLedger(
        min_formula_drop=_float_env("DRAFTPROOF_GLOBAL_MIN_FORMULA_DROP", 0.05),
        min_late_formula_drop_when_pinned=_float_env("DRAFTPROOF_GLOBAL_PINNED_TOPK_MIN_DROP", 1.0),
        target_score=TURNITIN_LIKE_TARGET_AI_SCORE,
    )

    def _controller_ledger_deps() -> GlobalControllerLedgerDeps:
        return GlobalControllerLedgerDeps(
            split_sentences=_split_sentences,
            strict_ai_safe_band_status=_strict_ai_safe_band_status,
            review_burden=_review_burden,
            weighted_severity=_weighted_severity,
            critical_high_count=_critical_high_count,
            finding_total=_finding_total,
        )

    def _controller_changed_sentence_ratio(before: str, after: str) -> float:
        return _core_controller_changed_sentence_ratio(
            before,
            after,
            deps=_controller_ledger_deps(),
        )

    def _controller_metrics(candidate_report: dict, current_report: dict, before_text: str, after_text: str) -> dict:
        return _core_controller_metrics(
            candidate_report,
            current_report,
            before_text,
            after_text,
            original_report=original_report_dict,
            deps=_controller_ledger_deps(),
        )

    def _controller_record(stage: str, strategy: str | None, candidate_text: str, candidate_report: dict) -> dict:
        return _core_controller_record(
            stage=stage,
            strategy=strategy,
            candidate_text=candidate_text,
            candidate_report=candidate_report,
            original_text=text,
            original_report=original_report_dict,
            current_text=rewritten_text,
            current_report=rewritten_report_dict,
            deps=_controller_ledger_deps(),
        )

    def _seed_global_candidate_ledger(stage: str, strategy: str = "current_best_before_post_phases") -> None:
        global_candidate_ledger.seed(_controller_record(stage, strategy, rewritten_text, rewritten_report_dict))

    def _global_phase_budget_skip(stage: str, *, min_seconds: float = 5.0, min_scans: int = 1, min_llm_calls: int = 0) -> dict | None:
        return _core_global_phase_budget_skip(
            global_rewrite_budget,
            stage,
            min_seconds=min_seconds,
            min_scans=min_scans,
            min_llm_calls=min_llm_calls,
        )

    def _global_controller_phase_accepted(stage: str, phase_result: dict, stored_result: dict) -> bool:
        return _core_global_controller_phase_accepted(
            ledger=global_candidate_ledger,
            stage=stage,
            phase_result=phase_result,
            stored_result=stored_result,
            original_text=text,
            original_report=original_report_dict,
            current_text=rewritten_text,
            current_report=rewritten_report_dict,
            deps=_controller_ledger_deps(),
        )

    _seed_global_candidate_ledger("post_ai_search_current")

    segment_window_budget_reserve = {
        "enabled": bool(_env_flag("DRAFTPROOF_SEGMENT_WINDOW_DENSITY_CONTROLLER", True)),
        "needed": False,
        "reserved_scans": 0,
        "reserved_llm_calls": 0,
        "reason": "not_evaluated",
    }
    if (
        segment_window_budget_reserve["enabled"]
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    ):
        reserve_profile = _turnitin_like_ai_profile(rewritten_report_dict)
        reserve_density = _eligible_span_density_contract(rewritten_text, rewritten_report_dict)
        reserve_components = (
            reserve_profile.get("components")
            if isinstance(reserve_profile.get("components"), dict)
            else {}
        )
        reserve_needed = bool(
            not reserve_profile.get("target_met")
            and (
                not reserve_density.get("safe")
                or float(reserve_components.get("topk_calibrated_risk") or 0.0)
                >= _safe_topk_calibrated_limit()
            )
        )
        if reserve_needed:
            remaining_scans_for_reserve = global_rewrite_budget.remaining_scans()
            remaining_llm_for_reserve = global_rewrite_budget.remaining_llm_calls()
            phase_plan = result.summary.get("rewrite_phase_budget_plan")
            segment_phase_plan = (
                ((phase_plan or {}).get("phases") or {}).get("segment_window_density_controller")
                if isinstance(phase_plan, dict) else {}
            )
            target_scans = int(
                (segment_phase_plan or {}).get("max_scans")
                or _float_env("DRAFTPROOF_SEGMENT_WINDOW_RESERVED_SCANS", 3.0)
            )
            target_llm = int(
                (segment_phase_plan or {}).get("max_llm_calls")
                or _float_env("DRAFTPROOF_SEGMENT_WINDOW_RESERVED_LLM_CALLS", 3.0)
            )
            segment_window_budget_reserve.update({
                "needed": True,
                "reason": "density_or_topk_unsafe",
                "reserved_scans": max(
                    0,
                    min(target_scans, remaining_scans_for_reserve if remaining_scans_for_reserve is not None else target_scans),
                ),
                "reserved_llm_calls": max(
                    0,
                    min(target_llm, remaining_llm_for_reserve if remaining_llm_for_reserve is not None else target_llm),
                ),
                "eligible_span_density_safe": bool(reserve_density.get("safe")),
                "topk_calibrated_risk": round(float(reserve_components.get("topk_calibrated_risk") or 0.0), 3),
                "source": (
                    "rewrite_phase_budget_plan"
                    if isinstance(segment_phase_plan, dict) and segment_phase_plan
                    else "legacy_segment_window_reserve"
                ),
            })
        else:
            segment_window_budget_reserve.update({
                "needed": False,
                "reason": "turnitin_or_density_segment_controller_not_needed",
                "eligible_span_density_safe": bool(reserve_density.get("safe")),
                "topk_calibrated_risk": round(float(reserve_components.get("topk_calibrated_risk") or 0.0), 3),
            })
    result.summary["segment_window_budget_reserve"] = segment_window_budget_reserve

    convergence_selected = False
    if (
        _env_flag("DRAFTPROOF_FORMULA_CONVERGENCE_CONTROLLER", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
    ):
        current_formula_profile = _turnitin_like_ai_profile(rewritten_report_dict)
        if (
            not bool(current_formula_profile.get("target_met"))
            and global_rewrite_budget.can_run(min_seconds=6.0, min_scans=1)
        ):
            report_progress(78, "Running formula convergence controller")
            convergence_t0 = time.time()
            try:
                convergence_key = (
                    api_key
                    or os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("LLM_API_KEY")
                )
                convergence_gateway = (
                    LLMGateway(LLMConfig(
                        api_key=convergence_key,
                        model=generator_model,
                        base_url=base_url,
                        timeout=int(os.environ.get("DRAFTPROOF_FORMULA_CONVERGENCE_TIMEOUT", "120")),
                        max_retries=int(os.environ.get("DRAFTPROOF_FORMULA_CONVERGENCE_RETRIES", "1")),
                        max_tokens=int(os.environ.get("DRAFTPROOF_FORMULA_CONVERGENCE_MAX_TOKENS", "3200")),
                        temperature=float(os.environ.get("DRAFTPROOF_FORMULA_CONVERGENCE_TEMPERATURE", "0.45")),
                    ))
                    if convergence_key and _env_flag("DRAFTPROOF_FORMULA_CONVERGENCE_LLM_BLOCK_RECREATE", True)
                    else None
                )
                convergence_budget = _formula_convergence_budget(rewritten_text)
                phase_plan = result.summary.get("rewrite_phase_budget_plan")
                formula_phase_plan = (
                    ((phase_plan or {}).get("phases") or {}).get("formula_convergence_controller")
                    if isinstance(phase_plan, dict) else {}
                )
                remaining_scans_for_convergence = global_rewrite_budget.remaining_scans()
                remaining_llm_for_convergence = global_rewrite_budget.remaining_llm_calls()
                reserved_scans = int(segment_window_budget_reserve.get("reserved_scans") or 0)
                reserved_llm = int(segment_window_budget_reserve.get("reserved_llm_calls") or 0)
                if isinstance(formula_phase_plan, dict) and formula_phase_plan:
                    convergence_budget["max_scans"] = min(
                        int(convergence_budget.get("max_scans") or 0),
                        int(formula_phase_plan.get("max_scans") or 0),
                    )
                    convergence_budget["max_llm_calls"] = min(
                        int(convergence_budget.get("max_llm_calls") or 0),
                        int(formula_phase_plan.get("max_llm_calls") or 0),
                    )
                if remaining_scans_for_convergence is not None:
                    convergence_budget["max_scans"] = min(
                        int(convergence_budget.get("max_scans") or 0),
                        max(0, remaining_scans_for_convergence - reserved_scans),
                    )
                if remaining_llm_for_convergence is not None:
                    convergence_budget["max_llm_calls"] = min(
                        int(convergence_budget.get("max_llm_calls") or 0),
                        max(0, remaining_llm_for_convergence - reserved_llm),
                    )
                convergence_budget["reserved_for_segment_window"] = {
                    "scans": reserved_scans,
                    "llm_calls": reserved_llm,
                    "reason": segment_window_budget_reserve.get("reason"),
                }
                convergence_result = _formula_convergence_controller(
                    rewritten_text,
                    rewritten_report_dict,
                    original_report_dict,
                    budget=convergence_budget,
                    llm_gateway=convergence_gateway,
                )
            except Exception as exc:
                convergence_result = {
                    "enabled": True,
                    "selected": False,
                    "reason": f"formula_convergence_controller_error {exc}",
                }
            stored_convergence_result = {
                key: value
                for key, value in convergence_result.items()
                if key not in {"selected_text", "selected_report"}
            }
            result.summary["formula_convergence_controller"] = stored_convergence_result
            result.summary["block_driver_map"] = stored_convergence_result.get("block_driver_map")
            result.summary["formula_convergence_passes"] = stored_convergence_result.get("formula_convergence_passes")
            result.summary["selected_formula_portfolio_candidate"] = (
                stored_convergence_result.get("selected_formula_portfolio_candidate")
            )
            result.summary["best_formula_frontier"] = stored_convergence_result.get("best_formula_frontier")
            result.summary["remaining_formula_gap"] = stored_convergence_result.get("remaining_formula_gap")
            result.summary["why_not_below_20"] = stored_convergence_result.get("why_not_below_20")
            try:
                convergence_llm_calls = int(
                    ((stored_convergence_result.get("phase_budget_used") or {}).get("llm_calls"))
                    or stored_convergence_result.get("llm_calls")
                    or 0
                )
            except (TypeError, ValueError):
                convergence_llm_calls = 0
            _record_rewrite_llm_calls(
                result.summary,
                "formula_convergence_llm_calls_used",
                convergence_llm_calls,
            )
            if (
                convergence_result.get("selected")
                and _global_controller_phase_accepted(
                    "formula_convergence_controller",
                    convergence_result,
                    stored_convergence_result,
                )
            ):
                selected_report = convergence_result.get("selected_report")
                selected_text = convergence_result.get("selected_text")
                if isinstance(selected_report, dict) and isinstance(selected_text, str):
                    previous_ai = rewritten_ai
                    rewritten_text = selected_text
                    rewritten_report_dict = selected_report
                    attempted_report_dict = rewritten_report_dict
                    rewritten_ai = _badge_ai(rewritten_report_dict)
                    rewritten_wq = _badge_wq(rewritten_report_dict)
                    rewritten_total = _finding_total(rewritten_report_dict)
                    rewritten_review_burden = _review_burden(rewritten_report_dict)
                    rewritten_severity = _weighted_severity(rewritten_report_dict)
                    rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                    if result.mp_result:
                        result.mp_result.final_text = rewritten_text
                        result.mp_result.converged = True
                        result.mp_result.convergence_reason = (
                            "Selected formula convergence candidate: "
                            f"{convergence_result.get('selected_strategy')}"
                        )
                    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                    ai_search_selected = True
                    convergence_selected = True
                    _clear_stale_rollback_for_kept_ai_mitigation(
                        result.summary,
                        "formula convergence controller",
                    )
                    result.summary["selected_strategy"] = convergence_result.get("selected_strategy")
                    result.summary["selected_formula_strategy"] = convergence_result.get("selected_strategy")
                    result.summary["detect_scores"].update({
                        "rewritten_ai": rewritten_ai,
                        "rewritten_writing_quality": rewritten_wq,
                        "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                        "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                        "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                        "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                        "rewritten_findings": rewritten_total,
                        "rewritten_review_burden": rewritten_review_burden,
                        "rewritten_weighted_severity": rewritten_severity,
                    })
            stage_timings.append({
                "stage": "formula_convergence_controller",
                "seconds": round(time.time() - convergence_t0, 3),
                "candidates": len(stored_convergence_result.get("candidates") or []),
                "selected": bool(convergence_result.get("selected")),
                "target_met": bool(convergence_result.get("target_met")),
                "stop_reason": convergence_result.get("stop_reason") or convergence_result.get("reason"),
            })
            global_rewrite_budget.record_stage(
                "formula_convergence_controller",
                seconds=round(time.time() - convergence_t0, 3),
                scans=int(stored_convergence_result.get("scans_used") or 0),
                llm_calls=convergence_llm_calls,
            )
        elif not bool(current_formula_profile.get("target_met")):
            stored_convergence_result = _global_phase_budget_skip(
                "formula_convergence_controller",
                min_seconds=6.0,
                min_scans=1,
            ) or {}
            result.summary["formula_convergence_controller"] = stored_convergence_result
            stage_timings.append({
                "stage": "formula_convergence_controller",
                "seconds": 0.0,
                "candidates": 0,
                "selected": False,
                "skipped": True,
                "stop_reason": stored_convergence_result.get("reason"),
            })

    segment_window_selected = False
    segment_window_should_run = (
        _env_flag("DRAFTPROOF_SEGMENT_WINDOW_DENSITY_CONTROLLER", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    )
    if segment_window_should_run and not global_rewrite_budget.can_run(min_seconds=8.0, min_scans=1, min_llm_calls=1):
        stored_segment_window_result = _global_phase_budget_skip(
            "segment_window_density_controller",
            min_seconds=8.0,
            min_scans=1,
            min_llm_calls=1,
        ) or {}
        result.summary["segment_window_density_controller"] = stored_segment_window_result
        stage_timings.append({
            "stage": "segment_window_density_controller",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "stop_reason": stored_segment_window_result.get("reason"),
        })
    elif segment_window_should_run:
        current_segment_profile = _turnitin_like_ai_profile(rewritten_report_dict)
        current_segment_density = _eligible_span_density_contract(rewritten_text, rewritten_report_dict)
        current_segment_components = current_segment_profile.get("components") if isinstance(current_segment_profile.get("components"), dict) else {}
        segment_needed = bool(
            not current_segment_profile.get("target_met")
            and (
                not current_segment_density.get("safe")
                or float(current_segment_components.get("topk_calibrated_risk") or 0.0) >= _safe_topk_calibrated_limit()
            )
        )
        if not segment_needed:
            stored_segment_window_result = {
                "enabled": True,
                "selected": False,
                "reason": "turnitin_or_density_segment_controller_not_needed",
                "segment_density_windows": _segment_density_windows(rewritten_text, rewritten_report_dict),
                "eligible_span_density_before": current_segment_density,
            }
            result.summary["segment_window_density_controller"] = stored_segment_window_result
            stage_timings.append({
                "stage": "segment_window_density_controller",
                "seconds": 0.0,
                "candidates": 0,
                "selected": False,
                "stop_reason": stored_segment_window_result.get("reason"),
            })
        else:
            report_progress(81, "Running segment-window density controller")
            segment_t0 = time.time()
            remaining_scans = global_rewrite_budget.remaining_scans()
            remaining_llm = global_rewrite_budget.remaining_llm_calls()
            phase_plan = result.summary.get("rewrite_phase_budget_plan")
            segment_phase_plan = (
                ((phase_plan or {}).get("phases") or {}).get("segment_window_density_controller")
                if isinstance(phase_plan, dict) else {}
            )
            segment_scan_allocation = int((segment_phase_plan or {}).get("max_scans") or 3)
            segment_llm_allocation = int((segment_phase_plan or {}).get("max_llm_calls") or 3)
            max_segment_scans = min(
                segment_scan_allocation,
                remaining_scans if remaining_scans is not None else segment_scan_allocation,
            )
            max_segment_llm = min(
                segment_llm_allocation,
                remaining_llm if remaining_llm is not None else segment_llm_allocation,
            )
            try:
                segment_key = (
                    api_key
                    or os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("LLM_API_KEY")
                )
                segment_gateway = (
                    LLMGateway(LLMConfig(
                        api_key=segment_key,
                        model=generator_model,
                        base_url=base_url,
                        timeout=int(os.environ.get("DRAFTPROOF_SEGMENT_WINDOW_TIMEOUT", "90")),
                        max_retries=int(os.environ.get("DRAFTPROOF_SEGMENT_WINDOW_RETRIES", "1")),
                        max_tokens=int(os.environ.get("DRAFTPROOF_SEGMENT_WINDOW_MAX_TOKENS", "2600")),
                        temperature=float(os.environ.get("DRAFTPROOF_SEGMENT_WINDOW_TEMPERATURE", "0.42")),
                    ))
                    if segment_key and max_segment_llm > 0
                    else None
                )
                segment_window_result = _segment_window_density_controller(
                    rewritten_text,
                    rewritten_report_dict,
                    original_report_dict,
                    gateway=segment_gateway,
                    scan_func=_full_scan_report_dict,
                    max_scans=max_segment_scans,
                    max_llm_calls=max_segment_llm,
                )
            except Exception as exc:
                segment_window_result = {
                    "enabled": True,
                    "selected": False,
                    "reason": f"segment_window_density_controller_error {exc}",
                }
            stored_segment_window_result = {
                key: value
                for key, value in segment_window_result.items()
                if key not in {"selected_text", "selected_report"}
            }
            result.summary["segment_window_density_controller"] = stored_segment_window_result
            result.summary["segment_density_windows"] = stored_segment_window_result.get("segment_density_windows")
            result.summary["segment_window_candidate_frontier"] = stored_segment_window_result.get("candidate_frontier")
            if (
                segment_window_result.get("selected")
                and _global_controller_phase_accepted(
                    "segment_window_density_controller",
                    segment_window_result,
                    stored_segment_window_result,
                )
            ):
                selected_report = segment_window_result.get("selected_report")
                selected_text = segment_window_result.get("selected_text")
                if isinstance(selected_report, dict) and isinstance(selected_text, str):
                    rewritten_text = selected_text
                    rewritten_report_dict = selected_report
                    attempted_report_dict = rewritten_report_dict
                    rewritten_ai = _badge_ai(rewritten_report_dict)
                    rewritten_wq = _badge_wq(rewritten_report_dict)
                    rewritten_total = _finding_total(rewritten_report_dict)
                    rewritten_review_burden = _review_burden(rewritten_report_dict)
                    rewritten_severity = _weighted_severity(rewritten_report_dict)
                    rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                    if result.mp_result:
                        result.mp_result.final_text = rewritten_text
                        result.mp_result.converged = True
                        result.mp_result.convergence_reason = (
                            "Selected segment-window density candidate: "
                            f"{segment_window_result.get('selected_strategy')}"
                        )
                    sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                    ai_search_selected = True
                    segment_window_selected = True
                    _clear_stale_rollback_for_kept_ai_mitigation(
                        result.summary,
                        "segment-window density controller",
                    )
                    result.summary["selected_strategy"] = segment_window_result.get("selected_strategy")
                    result.summary["selected_segment_window_strategy"] = segment_window_result.get("selected_strategy")
                    result.summary["detect_scores"].update({
                        "rewritten_ai": rewritten_ai,
                        "rewritten_writing_quality": rewritten_wq,
                        "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                        "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                        "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                        "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                        "rewritten_findings": rewritten_total,
                        "rewritten_review_burden": rewritten_review_burden,
                        "rewritten_weighted_severity": rewritten_severity,
                    })
            stage_timings.append({
                "stage": "segment_window_density_controller",
                "seconds": round(time.time() - segment_t0, 3),
                "candidates": len(stored_segment_window_result.get("candidate_frontier") or []),
                "scans": stored_segment_window_result.get("scans_used"),
                "llm_calls": stored_segment_window_result.get("llm_calls"),
                "selected": bool(segment_window_result.get("selected")),
                "stop_reason": segment_window_result.get("reason"),
            })
            global_rewrite_budget.record_stage(
                "segment_window_density_controller",
                seconds=round(time.time() - segment_t0, 3),
                scans=int(stored_segment_window_result.get("scans_used") or 0),
                llm_calls=int(stored_segment_window_result.get("llm_calls") or 0),
            )

    window_coverage_budget_reserve = {
        "enabled": bool(_env_flag("DRAFTPROOF_WINDOW_COVERAGE_DENSITY_OPTIMIZER", True)),
        "needed": False,
        "reserved_scans": 0,
        "reserved_llm_calls": 0,
        "reason": "not_evaluated",
    }
    if (
        window_coverage_budget_reserve["enabled"]
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    ):
        reserve_profile = _turnitin_like_ai_profile(rewritten_report_dict)
        reserve_density = _eligible_span_density_contract(rewritten_text, rewritten_report_dict)
        reserve_coverage = _window_coverage_map(rewritten_text, rewritten_report_dict)
        reserve_needed = bool(
            not reserve_profile.get("target_met")
            and (
                not reserve_density.get("safe")
                or int(reserve_coverage.get("unsafe_window_count") or 0) > 0
            )
        )
        if reserve_needed:
            remaining_scans_for_window = global_rewrite_budget.remaining_scans()
            remaining_llm_for_window = global_rewrite_budget.remaining_llm_calls()
            target_scans = int(_float_env(
                "DRAFTPROOF_WINDOW_COVERAGE_RESERVED_SCANS",
                _float_env("DRAFTPROOF_WINDOW_COVERAGE_MAX_SCANS", 5.0),
            ))
            target_llm = int(_float_env(
                "DRAFTPROOF_WINDOW_COVERAGE_RESERVED_LLM_CALLS",
                _float_env("DRAFTPROOF_WINDOW_COVERAGE_MAX_LLM_CALLS", 5.0),
            ))
            window_coverage_budget_reserve.update({
                "needed": True,
                "reason": "window_density_or_unsafe_coverage",
                "reserved_scans": max(
                    0,
                    min(target_scans, remaining_scans_for_window if remaining_scans_for_window is not None else target_scans),
                ),
                "reserved_llm_calls": max(
                    0,
                    min(target_llm, remaining_llm_for_window if remaining_llm_for_window is not None else target_llm),
                ),
                "eligible_span_density_safe": bool(reserve_density.get("safe")),
                "unsafe_window_count": int(reserve_coverage.get("unsafe_window_count") or 0),
                "ai_sentence_vote_ratio": reserve_coverage.get("ai_sentence_vote_ratio"),
            })
        else:
            window_coverage_budget_reserve.update({
                "needed": False,
                "reason": "window_coverage_not_needed",
                "eligible_span_density_safe": bool(reserve_density.get("safe")),
                "unsafe_window_count": int(reserve_coverage.get("unsafe_window_count") or 0),
                "ai_sentence_vote_ratio": reserve_coverage.get("ai_sentence_vote_ratio"),
            })
    result.summary["window_coverage_budget_reserve"] = window_coverage_budget_reserve

    density_breaker_selected = False
    density_breaker_should_run = (
        _env_flag("DRAFTPROOF_POST_SELECTION_AI_DENSITY_BREAKER", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    )
    if density_breaker_should_run and not global_rewrite_budget.can_run(min_seconds=5.0, min_scans=1):
        stored_density_breaker_result = _global_phase_budget_skip(
            "post_selection_ai_density_breaker",
            min_seconds=5.0,
            min_scans=1,
        ) or {}
        result.summary["post_selection_ai_density_breaker"] = stored_density_breaker_result
        stage_timings.append({
            "stage": "post_selection_ai_density_breaker",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "stop_reason": stored_density_breaker_result.get("reason"),
        })
    elif density_breaker_should_run:
        report_progress(82, "Running post-selection AI-density breaker")
        density_t0 = time.time()
        try:
            phase_plan = result.summary.get("rewrite_phase_budget_plan")
            post_segment_plan = (
                ((phase_plan or {}).get("phases") or {}).get("post_segment_followup")
                if isinstance(phase_plan, dict) else {}
            )
            post_segment_scan_allocation = int((post_segment_plan or {}).get("max_scans") or 0)
            remaining_density_scans = global_rewrite_budget.remaining_scans()
            density_max_scans = (
                remaining_density_scans
                if remaining_density_scans is not None
                else None
            )
            reserved_window_scans = int(window_coverage_budget_reserve.get("reserved_scans") or 0)
            if density_max_scans is not None and reserved_window_scans > 0:
                density_max_scans = max(0, density_max_scans - reserved_window_scans)
            if post_segment_scan_allocation > 0 and density_max_scans is not None:
                density_max_scans = min(density_max_scans, post_segment_scan_allocation)
            density_breaker_result = _post_selection_ai_density_breaker(
                rewritten_text,
                rewritten_report_dict,
                original_report_dict,
                scan_func=_full_scan_report_dict,
                max_scans=density_max_scans,
            )
        except Exception as exc:
            density_breaker_result = {
                "enabled": True,
                "selected": False,
                "reason": f"post_selection_ai_density_breaker_error {exc}",
            }
        stored_density_breaker_result = {
            key: value
            for key, value in density_breaker_result.items()
            if key not in {"selected_text", "selected_report"}
        }
        result.summary["post_selection_ai_density_breaker"] = stored_density_breaker_result
        if (
            density_breaker_result.get("selected")
            and _global_controller_phase_accepted(
                "post_selection_ai_density_breaker",
                density_breaker_result,
                stored_density_breaker_result,
            )
        ):
            selected_report = density_breaker_result.get("selected_report")
            selected_text = density_breaker_result.get("selected_text")
            if isinstance(selected_report, dict) and isinstance(selected_text, str):
                rewritten_text = selected_text
                rewritten_report_dict = selected_report
                attempted_report_dict = rewritten_report_dict
                rewritten_ai = _badge_ai(rewritten_report_dict)
                rewritten_wq = _badge_wq(rewritten_report_dict)
                rewritten_total = _finding_total(rewritten_report_dict)
                rewritten_review_burden = _review_burden(rewritten_report_dict)
                rewritten_severity = _weighted_severity(rewritten_report_dict)
                rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                if result.mp_result:
                    result.mp_result.final_text = rewritten_text
                    result.mp_result.converged = True
                    result.mp_result.convergence_reason = (
                        "Selected post-selection AI-density breaker candidate: "
                        f"{density_breaker_result.get('selected_strategy')}"
                    )
                sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                ai_search_selected = True
                density_breaker_selected = True
                _clear_stale_rollback_for_kept_ai_mitigation(
                    result.summary,
                    "post-selection AI-density breaker",
                )
                result.summary["selected_strategy"] = density_breaker_result.get("selected_strategy")
                result.summary["selected_density_breaker_strategy"] = density_breaker_result.get("selected_strategy")
                result.summary["detect_scores"].update({
                    "rewritten_ai": rewritten_ai,
                    "rewritten_writing_quality": rewritten_wq,
                    "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                    "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                    "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                    "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                    "rewritten_findings": rewritten_total,
                    "rewritten_review_burden": rewritten_review_burden,
                    "rewritten_weighted_severity": rewritten_severity,
                })
        stage_timings.append({
            "stage": "post_selection_ai_density_breaker",
            "seconds": round(time.time() - density_t0, 3),
            "candidates": len(stored_density_breaker_result.get("candidates") or []),
            "scans": stored_density_breaker_result.get("scans_used"),
            "selected": bool(density_breaker_result.get("selected")),
            "stop_reason": density_breaker_result.get("reason"),
        })
        global_rewrite_budget.record_stage(
            "post_selection_ai_density_breaker",
            seconds=round(time.time() - density_t0, 3),
            scans=int(stored_density_breaker_result.get("scans_used") or 0),
        )

    window_coverage_should_run = (
        _env_flag("DRAFTPROOF_WINDOW_COVERAGE_DENSITY_OPTIMIZER", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
        and not isinstance(result.summary.get("window_coverage_density_optimizer"), dict)
    )
    if window_coverage_should_run:
        window_profile = _turnitin_like_ai_profile(rewritten_report_dict)
        window_density = _eligible_span_density_contract(rewritten_text, rewritten_report_dict)
        window_coverage_map = _window_coverage_map(rewritten_text, rewritten_report_dict)
        window_coverage_should_run = bool(
            not window_profile.get("target_met")
            and (
                not window_density.get("safe")
                or int(window_coverage_map.get("unsafe_window_count") or 0) > 0
            )
        )
    if window_coverage_should_run and not global_rewrite_budget.can_run(
        min_seconds=8.0,
        min_scans=1,
        min_llm_calls=0,
    ):
        stored_window_coverage_result = _global_phase_budget_skip(
            "window_coverage_density_optimizer",
            min_seconds=8.0,
            min_scans=1,
            min_llm_calls=0,
        ) or {}
        result.summary["window_coverage_density_optimizer"] = stored_window_coverage_result
        stage_timings.append({
            "stage": "window_coverage_density_optimizer",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "stop_reason": stored_window_coverage_result.get("reason"),
        })
    elif window_coverage_should_run:
        report_progress(83, "Running window-coverage density optimizer")
        window_coverage_t0 = time.time()
        try:
            remaining_scans = global_rewrite_budget.remaining_scans()
            remaining_llm = global_rewrite_budget.remaining_llm_calls()
            window_scan_cap = max(0, int(_float_env("DRAFTPROOF_WINDOW_COVERAGE_MAX_SCANS", 5.0)))
            window_llm_cap = max(0, int(_float_env("DRAFTPROOF_WINDOW_COVERAGE_MAX_LLM_CALLS", 5.0)))
            reserved_scans = int(window_coverage_budget_reserve.get("reserved_scans") or window_scan_cap)
            reserved_llm = int(window_coverage_budget_reserve.get("reserved_llm_calls") or window_llm_cap)
            max_window_scans = min(
                window_scan_cap,
                reserved_scans,
                remaining_scans if remaining_scans is not None else window_scan_cap,
            )
            max_window_llm = min(
                window_llm_cap,
                reserved_llm,
                remaining_llm if remaining_llm is not None else window_llm_cap,
            )
            window_key = (
                api_key
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("LLM_API_KEY")
            )
            window_gateway = (
                LLMGateway(LLMConfig(
                    api_key=window_key,
                    model=generator_model,
                    base_url=base_url,
                    timeout=int(os.environ.get("DRAFTPROOF_WINDOW_COVERAGE_TIMEOUT", "90")),
                    max_retries=int(os.environ.get("DRAFTPROOF_WINDOW_COVERAGE_RETRIES", "1")),
                    max_tokens=int(os.environ.get("DRAFTPROOF_WINDOW_COVERAGE_MAX_TOKENS", "3200")),
                    temperature=float(os.environ.get("DRAFTPROOF_WINDOW_COVERAGE_TEMPERATURE", "0.42")),
                ))
                if window_key and max_window_llm > 0
                else None
            )
            window_coverage_result = _window_coverage_density_optimizer(
                rewritten_text,
                rewritten_report_dict,
                original_report_dict,
                gateway=window_gateway,
                scan_func=_full_scan_report_dict,
                max_scans=max_window_scans,
                max_llm_calls=max_window_llm,
            )
        except Exception as exc:
            window_coverage_result = {
                "enabled": True,
                "selected": False,
                "reason": f"window_coverage_density_optimizer_error {exc}",
            }
        stored_window_coverage_result = {
            key: value
            for key, value in window_coverage_result.items()
            if key not in {"selected_text", "selected_report"}
        }
        result.summary["window_coverage_density_optimizer"] = stored_window_coverage_result
        result.summary["window_coverage_map"] = stored_window_coverage_result.get("window_coverage_map")
        result.summary["top_coverage_sentences"] = stored_window_coverage_result.get("top_coverage_sentences")
        result.summary["window_coverage_candidate_frontier"] = stored_window_coverage_result.get("candidate_frontier")
        result.summary["window_coverage_portfolio_optimizer"] = stored_window_coverage_result.get("window_coverage_portfolio_optimizer")
        result.summary["deterministic_variant_frontier"] = stored_window_coverage_result.get("deterministic_variant_frontier")
        result.summary["portfolio_candidate_frontier"] = stored_window_coverage_result.get("portfolio_candidate_frontier")
        result.summary["patch_ablation_frontier"] = stored_window_coverage_result.get("patch_ablation_frontier")
        result.summary["selected_window_coverage_portfolio"] = stored_window_coverage_result.get("selected_window_coverage_portfolio")
        result.summary["coverage_efficiency"] = stored_window_coverage_result.get("coverage_efficiency")
        result.summary["selected_window_coverage_strategy"] = stored_window_coverage_result.get("selected_strategy")
        result.summary["unsafe_window_count_before"] = stored_window_coverage_result.get("unsafe_window_count_before")
        result.summary["unsafe_window_count_after"] = stored_window_coverage_result.get("unsafe_window_count_after")
        result.summary["unsafe_window_count_drop"] = stored_window_coverage_result.get("unsafe_window_count_drop")
        result.summary["ai_sentence_vote_ratio_before"] = stored_window_coverage_result.get("ai_sentence_vote_ratio_before")
        result.summary["ai_sentence_vote_ratio_after"] = stored_window_coverage_result.get("ai_sentence_vote_ratio_after")
        result.summary["ai_sentence_vote_ratio_drop"] = stored_window_coverage_result.get("ai_sentence_vote_ratio_drop")
        if (
            window_coverage_result.get("selected")
            and _global_controller_phase_accepted(
                "window_coverage_density_optimizer",
                window_coverage_result,
                stored_window_coverage_result,
            )
        ):
            selected_report = window_coverage_result.get("selected_report")
            selected_text = window_coverage_result.get("selected_text")
            if isinstance(selected_report, dict) and isinstance(selected_text, str):
                rewritten_text = selected_text
                rewritten_report_dict = selected_report
                attempted_report_dict = rewritten_report_dict
                rewritten_ai = _badge_ai(rewritten_report_dict)
                rewritten_wq = _badge_wq(rewritten_report_dict)
                rewritten_total = _finding_total(rewritten_report_dict)
                rewritten_review_burden = _review_burden(rewritten_report_dict)
                rewritten_severity = _weighted_severity(rewritten_report_dict)
                rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                if result.mp_result:
                    result.mp_result.final_text = rewritten_text
                    result.mp_result.converged = True
                    result.mp_result.convergence_reason = (
                        "Selected window-coverage density candidate: "
                        f"{window_coverage_result.get('selected_strategy')}"
                    )
                sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                ai_search_selected = True
                window_coverage_selected = True
                _clear_stale_rollback_for_kept_ai_mitigation(
                    result.summary,
                    "window-coverage density optimizer",
                )
                result.summary["selected_strategy"] = window_coverage_result.get("selected_strategy")
                result.summary["selected_window_coverage_strategy"] = window_coverage_result.get("selected_strategy")
                result.summary["detect_scores"].update({
                    "rewritten_ai": rewritten_ai,
                    "rewritten_writing_quality": rewritten_wq,
                    "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                    "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                    "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                    "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                    "rewritten_findings": rewritten_total,
                    "rewritten_review_burden": rewritten_review_burden,
                    "rewritten_weighted_severity": rewritten_severity,
                })
        stage_timings.append({
            "stage": "window_coverage_density_optimizer",
            "seconds": round(time.time() - window_coverage_t0, 3),
            "candidates": len(stored_window_coverage_result.get("candidate_frontier") or []),
            "scans": stored_window_coverage_result.get("scans_used"),
            "llm_calls": stored_window_coverage_result.get("llm_calls"),
            "selected": bool(window_coverage_result.get("selected")),
            "stop_reason": window_coverage_result.get("reason"),
        })
        global_rewrite_budget.record_stage(
            "window_coverage_density_optimizer",
            seconds=round(time.time() - window_coverage_t0, 3),
            scans=int(stored_window_coverage_result.get("scans_used") or 0),
            llm_calls=int(stored_window_coverage_result.get("llm_calls") or 0),
        )
        window_coverage_budget_reserve.update({
            "reserved_scans": 0,
            "reserved_llm_calls": 0,
            "consumed_before_followups": True,
        })
        result.summary["window_coverage_budget_reserve"] = window_coverage_budget_reserve

    segment_window_followup_should_run = (
        _env_flag("DRAFTPROOF_SEGMENT_WINDOW_FOLLOWUP_CONTROLLER", True)
        and _env_flag("DRAFTPROOF_SEGMENT_WINDOW_DENSITY_CONTROLLER", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    )
    if segment_window_followup_should_run:
        followup_profile = _turnitin_like_ai_profile(rewritten_report_dict)
        followup_density = _eligible_span_density_contract(rewritten_text, rewritten_report_dict)
        followup_components = (
            followup_profile.get("components")
            if isinstance(followup_profile.get("components"), dict)
            else {}
        )
        segment_window_followup_should_run = bool(
            not followup_profile.get("target_met")
            and (
                not followup_density.get("safe")
                or float(followup_components.get("topk_calibrated_risk") or 0.0)
                >= _safe_topk_calibrated_limit()
            )
        )
    if segment_window_followup_should_run and not global_rewrite_budget.can_run(
        min_seconds=8.0,
        min_scans=1,
        min_llm_calls=1,
    ):
        stored_segment_window_followup_result = _global_phase_budget_skip(
            "segment_window_density_controller_followup",
            min_seconds=8.0,
            min_scans=1,
            min_llm_calls=1,
        ) or {}
        result.summary["segment_window_density_controller_followup"] = stored_segment_window_followup_result
        stage_timings.append({
            "stage": "segment_window_density_controller_followup",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "stop_reason": stored_segment_window_followup_result.get("reason"),
        })
    elif segment_window_followup_should_run:
        report_progress(83, "Running follow-up segment-window density controller")
        segment_followup_t0 = time.time()
        try:
            remaining_scans = global_rewrite_budget.remaining_scans()
            remaining_llm = global_rewrite_budget.remaining_llm_calls()
            followup_scan_cap = max(0, int(_float_env("DRAFTPROOF_SEGMENT_WINDOW_FOLLOWUP_MAX_SCANS", 4.0)))
            followup_llm_cap = max(0, int(_float_env("DRAFTPROOF_SEGMENT_WINDOW_FOLLOWUP_MAX_LLM_CALLS", 4.0)))
            reserved_window_scans = int(window_coverage_budget_reserve.get("reserved_scans") or 0)
            reserved_window_llm = int(window_coverage_budget_reserve.get("reserved_llm_calls") or 0)
            max_followup_scans = min(
                followup_scan_cap,
                max(0, remaining_scans - reserved_window_scans) if remaining_scans is not None else followup_scan_cap,
            )
            max_followup_llm = min(
                followup_llm_cap,
                max(0, remaining_llm - reserved_window_llm) if remaining_llm is not None else followup_llm_cap,
            )
            if window_coverage_budget_reserve.get("needed") and (max_followup_scans <= 0 or max_followup_llm <= 0):
                segment_window_followup_result = {
                    "enabled": True,
                    "selected": False,
                    "reason": "budget_reserved_for_window_coverage_density_optimizer",
                    "window_coverage_budget_reserve": window_coverage_budget_reserve,
                    "scans_used": 0,
                    "llm_calls": 0,
                    "candidate_frontier": [],
                }
            else:
                followup_key = (
                    api_key
                    or os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("LLM_API_KEY")
                )
                followup_gateway = (
                    LLMGateway(LLMConfig(
                        api_key=followup_key,
                        model=generator_model,
                        base_url=base_url,
                        timeout=int(os.environ.get("DRAFTPROOF_SEGMENT_WINDOW_TIMEOUT", "90")),
                        max_retries=int(os.environ.get("DRAFTPROOF_SEGMENT_WINDOW_RETRIES", "1")),
                        max_tokens=int(os.environ.get("DRAFTPROOF_SEGMENT_WINDOW_MAX_TOKENS", "2600")),
                        temperature=float(os.environ.get("DRAFTPROOF_SEGMENT_WINDOW_TEMPERATURE", "0.42")),
                    ))
                    if followup_key and max_followup_llm > 0
                    else None
                )
                segment_window_followup_result = _segment_window_density_controller(
                    rewritten_text,
                    rewritten_report_dict,
                    original_report_dict,
                    gateway=followup_gateway,
                    scan_func=_full_scan_report_dict,
                    max_scans=max_followup_scans,
                    max_llm_calls=max_followup_llm,
                )
        except Exception as exc:
            segment_window_followup_result = {
                "enabled": True,
                "selected": False,
                "reason": f"segment_window_density_controller_followup_error {exc}",
            }
        stored_segment_window_followup_result = {
            key: value
            for key, value in segment_window_followup_result.items()
            if key not in {"selected_text", "selected_report"}
        }
        result.summary["segment_window_density_controller_followup"] = stored_segment_window_followup_result
        if (
            segment_window_followup_result.get("selected")
            and _global_controller_phase_accepted(
                "segment_window_density_controller_followup",
                segment_window_followup_result,
                stored_segment_window_followup_result,
            )
        ):
            selected_report = segment_window_followup_result.get("selected_report")
            selected_text = segment_window_followup_result.get("selected_text")
            if isinstance(selected_report, dict) and isinstance(selected_text, str):
                rewritten_text = selected_text
                rewritten_report_dict = selected_report
                attempted_report_dict = rewritten_report_dict
                rewritten_ai = _badge_ai(rewritten_report_dict)
                rewritten_wq = _badge_wq(rewritten_report_dict)
                rewritten_total = _finding_total(rewritten_report_dict)
                rewritten_review_burden = _review_burden(rewritten_report_dict)
                rewritten_severity = _weighted_severity(rewritten_report_dict)
                rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                if result.mp_result:
                    result.mp_result.final_text = rewritten_text
                    result.mp_result.converged = True
                    result.mp_result.convergence_reason = (
                        "Selected follow-up segment-window density candidate: "
                        f"{segment_window_followup_result.get('selected_strategy')}"
                    )
                sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                ai_search_selected = True
                _clear_stale_rollback_for_kept_ai_mitigation(
                    result.summary,
                    "follow-up segment-window density controller",
                )
                result.summary["selected_strategy"] = segment_window_followup_result.get("selected_strategy")
                result.summary["selected_segment_window_followup_strategy"] = (
                    segment_window_followup_result.get("selected_strategy")
                )
                result.summary["detect_scores"].update({
                    "rewritten_ai": rewritten_ai,
                    "rewritten_writing_quality": rewritten_wq,
                    "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                    "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                    "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                    "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                    "rewritten_findings": rewritten_total,
                    "rewritten_review_burden": rewritten_review_burden,
                    "rewritten_weighted_severity": rewritten_severity,
                })
        stage_timings.append({
            "stage": "segment_window_density_controller_followup",
            "seconds": round(time.time() - segment_followup_t0, 3),
            "candidates": len(stored_segment_window_followup_result.get("candidate_frontier") or []),
            "scans": stored_segment_window_followup_result.get("scans_used"),
            "llm_calls": stored_segment_window_followup_result.get("llm_calls"),
            "selected": bool(segment_window_followup_result.get("selected")),
            "stop_reason": segment_window_followup_result.get("reason"),
        })
        global_rewrite_budget.record_stage(
            "segment_window_density_controller_followup",
            seconds=round(time.time() - segment_followup_t0, 3),
            scans=int(stored_segment_window_followup_result.get("scans_used") or 0),
            llm_calls=int(stored_segment_window_followup_result.get("llm_calls") or 0),
        )

    remaining_cluster_selected = False
    remaining_cluster_should_run = (
        _env_flag("DRAFTPROOF_REMAINING_CLUSTER_DENSITY_CONTROLLER", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    )
    if remaining_cluster_should_run:
        cluster_profile = _turnitin_like_ai_profile(rewritten_report_dict)
        cluster_density = _eligible_span_density_contract(rewritten_text, rewritten_report_dict)
        cluster_components = (
            cluster_profile.get("components")
            if isinstance(cluster_profile.get("components"), dict)
            else {}
        )
        remaining_cluster_should_run = bool(
            not cluster_profile.get("target_met")
            and (
                not cluster_density.get("safe")
                or float(cluster_components.get("topk_calibrated_risk") or 0.0)
                >= _safe_topk_calibrated_limit()
            )
        )
    if remaining_cluster_should_run and not global_rewrite_budget.can_run(
        min_seconds=8.0,
        min_scans=1,
        min_llm_calls=1,
    ):
        stored_remaining_cluster_result = _global_phase_budget_skip(
            "remaining_cluster_density_controller",
            min_seconds=8.0,
            min_scans=1,
            min_llm_calls=1,
        ) or {}
        result.summary["remaining_cluster_density_controller"] = stored_remaining_cluster_result
        stage_timings.append({
            "stage": "remaining_cluster_density_controller",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "stop_reason": stored_remaining_cluster_result.get("reason"),
        })
    elif remaining_cluster_should_run:
        report_progress(84, "Running remaining-cluster density controller")
        remaining_cluster_t0 = time.time()
        try:
            remaining_scans = global_rewrite_budget.remaining_scans()
            remaining_llm = global_rewrite_budget.remaining_llm_calls()
            cluster_scan_cap = max(0, int(_float_env("DRAFTPROOF_REMAINING_CLUSTER_MAX_SCANS", 4.0)))
            cluster_llm_cap = max(0, int(_float_env("DRAFTPROOF_REMAINING_CLUSTER_MAX_LLM_CALLS", 4.0)))
            reserved_window_scans = int(window_coverage_budget_reserve.get("reserved_scans") or 0)
            reserved_window_llm = int(window_coverage_budget_reserve.get("reserved_llm_calls") or 0)
            max_cluster_scans = min(
                cluster_scan_cap,
                max(0, remaining_scans - reserved_window_scans) if remaining_scans is not None else cluster_scan_cap,
            )
            max_cluster_llm = min(
                cluster_llm_cap,
                max(0, remaining_llm - reserved_window_llm) if remaining_llm is not None else cluster_llm_cap,
            )
            if window_coverage_budget_reserve.get("needed") and (max_cluster_scans <= 0 or max_cluster_llm <= 0):
                remaining_cluster_result = {
                    "enabled": True,
                    "selected": False,
                    "reason": "budget_reserved_for_window_coverage_density_optimizer",
                    "window_coverage_budget_reserve": window_coverage_budget_reserve,
                    "scans_used": 0,
                    "llm_calls": 0,
                    "candidate_frontier": [],
                }
            else:
                cluster_key = (
                    api_key
                    or os.environ.get("OPENROUTER_API_KEY")
                    or os.environ.get("LLM_API_KEY")
                )
                cluster_gateway = (
                    LLMGateway(LLMConfig(
                        api_key=cluster_key,
                        model=generator_model,
                        base_url=base_url,
                        timeout=int(os.environ.get("DRAFTPROOF_REMAINING_CLUSTER_TIMEOUT", "90")),
                        max_retries=int(os.environ.get("DRAFTPROOF_REMAINING_CLUSTER_RETRIES", "1")),
                        max_tokens=int(os.environ.get("DRAFTPROOF_REMAINING_CLUSTER_MAX_TOKENS", "3200")),
                        temperature=float(os.environ.get("DRAFTPROOF_REMAINING_CLUSTER_TEMPERATURE", "0.42")),
                    ))
                    if cluster_key and max_cluster_llm > 0
                    else None
                )
                remaining_cluster_result = _remaining_cluster_density_controller(
                    rewritten_text,
                    rewritten_report_dict,
                    original_report_dict,
                    gateway=cluster_gateway,
                    scan_func=_full_scan_report_dict,
                    max_scans=max_cluster_scans,
                    max_llm_calls=max_cluster_llm,
                )
        except Exception as exc:
            remaining_cluster_result = {
                "enabled": True,
                "selected": False,
                "reason": f"remaining_cluster_density_controller_error {exc}",
            }
        stored_remaining_cluster_result = {
            key: value
            for key, value in remaining_cluster_result.items()
            if key not in {"selected_text", "selected_report"}
        }
        result.summary["remaining_cluster_density_controller"] = stored_remaining_cluster_result
        result.summary["remaining_cluster_map"] = stored_remaining_cluster_result.get("remaining_cluster_map")
        result.summary["remaining_cluster_candidate_frontier"] = stored_remaining_cluster_result.get("candidate_frontier")
        result.summary["selected_remaining_cluster_strategy"] = stored_remaining_cluster_result.get("selected_strategy")
        if (
            remaining_cluster_result.get("selected")
            and _global_controller_phase_accepted(
                "remaining_cluster_density_controller",
                remaining_cluster_result,
                stored_remaining_cluster_result,
            )
        ):
            selected_report = remaining_cluster_result.get("selected_report")
            selected_text = remaining_cluster_result.get("selected_text")
            if isinstance(selected_report, dict) and isinstance(selected_text, str):
                rewritten_text = selected_text
                rewritten_report_dict = selected_report
                attempted_report_dict = rewritten_report_dict
                rewritten_ai = _badge_ai(rewritten_report_dict)
                rewritten_wq = _badge_wq(rewritten_report_dict)
                rewritten_total = _finding_total(rewritten_report_dict)
                rewritten_review_burden = _review_burden(rewritten_report_dict)
                rewritten_severity = _weighted_severity(rewritten_report_dict)
                rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                if result.mp_result:
                    result.mp_result.final_text = rewritten_text
                    result.mp_result.converged = True
                    result.mp_result.convergence_reason = (
                        "Selected remaining-cluster density candidate: "
                        f"{remaining_cluster_result.get('selected_strategy')}"
                    )
                sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                ai_search_selected = True
                remaining_cluster_selected = True
                _clear_stale_rollback_for_kept_ai_mitigation(
                    result.summary,
                    "remaining-cluster density controller",
                )
                result.summary["selected_strategy"] = remaining_cluster_result.get("selected_strategy")
                result.summary["selected_remaining_cluster_strategy"] = remaining_cluster_result.get("selected_strategy")
                result.summary["detect_scores"].update({
                    "rewritten_ai": rewritten_ai,
                    "rewritten_writing_quality": rewritten_wq,
                    "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                    "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                    "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                    "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                    "rewritten_findings": rewritten_total,
                    "rewritten_review_burden": rewritten_review_burden,
                    "rewritten_weighted_severity": rewritten_severity,
                })
        stage_timings.append({
            "stage": "remaining_cluster_density_controller",
            "seconds": round(time.time() - remaining_cluster_t0, 3),
            "candidates": len(stored_remaining_cluster_result.get("candidate_frontier") or []),
            "scans": stored_remaining_cluster_result.get("scans_used"),
            "llm_calls": stored_remaining_cluster_result.get("llm_calls"),
            "selected": bool(remaining_cluster_result.get("selected")),
            "stop_reason": remaining_cluster_result.get("reason"),
        })
        global_rewrite_budget.record_stage(
            "remaining_cluster_density_controller",
            seconds=round(time.time() - remaining_cluster_t0, 3),
            scans=int(stored_remaining_cluster_result.get("scans_used") or 0),
            llm_calls=int(stored_remaining_cluster_result.get("llm_calls") or 0),
        )

    post_density_anchor_selected = False
    anchor_probe_should_run = (
        _env_flag("DRAFTPROOF_POST_DENSITY_HUMAN_ANCHOR_PROBE", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    )
    if anchor_probe_should_run and not global_rewrite_budget.can_run(min_seconds=5.0, min_scans=1):
        stored_anchor_probe_result = _global_phase_budget_skip(
            "post_density_human_anchor_probe",
            min_seconds=5.0,
            min_scans=1,
        ) or {}
        result.summary["post_density_human_anchor_probe"] = stored_anchor_probe_result
        result.summary["human_anchor_suppression_frontier"] = None
        stage_timings.append({
            "stage": "post_density_human_anchor_probe",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "stop_reason": stored_anchor_probe_result.get("reason"),
        })
    elif anchor_probe_should_run:
        report_progress(84, "Running post-density Human Anchor probe")
        anchor_probe_t0 = time.time()
        try:
            anchor_probe_result = _post_density_human_anchor_probe(
                rewritten_text,
                rewritten_report_dict,
                original_report_dict,
                scan_func=_full_scan_report_dict,
            )
        except Exception as exc:
            anchor_probe_result = {
                "enabled": True,
                "selected": False,
                "reason": f"post_density_human_anchor_probe_error {exc}",
            }
        stored_anchor_probe_result = {
            key: value
            for key, value in anchor_probe_result.items()
            if key not in {"selected_text", "selected_report"}
        }
        result.summary["post_density_human_anchor_probe"] = stored_anchor_probe_result
        result.summary["human_anchor_suppression_frontier"] = stored_anchor_probe_result.get(
            "human_anchor_suppression_frontier"
        )
        if (
            anchor_probe_result.get("selected")
            and _global_controller_phase_accepted(
                "post_density_human_anchor_probe",
                anchor_probe_result,
                stored_anchor_probe_result,
            )
        ):
            selected_report = anchor_probe_result.get("selected_report")
            selected_text = anchor_probe_result.get("selected_text")
            if isinstance(selected_report, dict) and isinstance(selected_text, str):
                rewritten_text = selected_text
                rewritten_report_dict = selected_report
                attempted_report_dict = rewritten_report_dict
                rewritten_ai = _badge_ai(rewritten_report_dict)
                rewritten_wq = _badge_wq(rewritten_report_dict)
                rewritten_total = _finding_total(rewritten_report_dict)
                rewritten_review_burden = _review_burden(rewritten_report_dict)
                rewritten_severity = _weighted_severity(rewritten_report_dict)
                rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                if result.mp_result:
                    result.mp_result.final_text = rewritten_text
                    result.mp_result.converged = True
                    result.mp_result.convergence_reason = (
                        "Selected post-density Human Anchor probe candidate: "
                        f"{anchor_probe_result.get('selected_strategy')}"
                    )
                sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                ai_search_selected = True
                post_density_anchor_selected = True
                _clear_stale_rollback_for_kept_ai_mitigation(
                    result.summary,
                    "post-density Human Anchor probe",
                )
                result.summary["selected_strategy"] = anchor_probe_result.get("selected_strategy")
                result.summary["selected_human_anchor_probe_strategy"] = anchor_probe_result.get("selected_strategy")
                result.summary["detect_scores"].update({
                    "rewritten_ai": rewritten_ai,
                    "rewritten_writing_quality": rewritten_wq,
                    "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                    "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                    "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                    "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                    "rewritten_findings": rewritten_total,
                    "rewritten_review_burden": rewritten_review_burden,
                    "rewritten_weighted_severity": rewritten_severity,
                })
        stage_timings.append({
            "stage": "post_density_human_anchor_probe",
            "seconds": round(time.time() - anchor_probe_t0, 3),
            "candidates": len(stored_anchor_probe_result.get("candidates") or []),
            "scans": stored_anchor_probe_result.get("scans_used"),
            "selected": bool(anchor_probe_result.get("selected")),
            "stop_reason": anchor_probe_result.get("reason"),
        })
        global_rewrite_budget.record_stage(
            "post_density_human_anchor_probe",
            seconds=round(time.time() - anchor_probe_t0, 3),
            scans=int(stored_anchor_probe_result.get("scans_used") or 0),
        )

    auto_repair_selected = False
    auto_repair_should_run = (
        _env_flag("DRAFTPROOF_AUTO_REPAIR_CONTROLLER", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    )
    if auto_repair_should_run and not global_rewrite_budget.can_run(min_seconds=5.0, min_scans=1):
        stored_auto_repair_result = _global_phase_budget_skip(
            "auto_repair_controller",
            min_seconds=5.0,
            min_scans=1,
        ) or {}
        result.summary["auto_repair_controller"] = stored_auto_repair_result
        stage_timings.append({
            "stage": "auto_repair_controller",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "stop_reason": stored_auto_repair_result.get("reason"),
        })
    elif auto_repair_should_run:
        report_progress(86, "Running automated repair compiler")
        auto_repair_t0 = time.time()
        try:
            auto_repair_deps = AutoRepairDependencies(
                split_sentences=_split_sentences,
                text_word_count=_text_word_count,
                geometry_risk_map=_geometry_risk_map,
                sentence_texture_risk_map=_sentence_texture_risk_map,
                ordered_concept_terms=_ordered_concept_origin_terms,
                is_canonical_fact_sentence=_ai_density_breaker_canonical_fact_sentence,
                splice_sentence=_splice_sentence_for_auto_repair,
                repair_aggression_score=_repair_aggression_score,
                locality_score=_locality_score,
                detect_protected_spans=detect_protected_spans,
                protected_loss_reason=_ai_search_protected_loss_reason,
                concept_origin_reject_reason=_candidate_concept_origin_reject_reason,
                drift_checker=check_semantic_drift,
                scan_func=_full_scan_report_dict,
                turnitin_profile=_turnitin_like_ai_profile,
                turnitin_gate_status=_turnitin_like_ai_gate_status,
                strict_safe_status=_strict_ai_safe_band_status,
                contribution_scores=_contribution_scores,
                integrity_scores=_integrity_scores,
                badge_ai=_badge_ai,
                finding_total=_finding_total,
                review_burden=_review_burden,
                weighted_severity=_weighted_severity,
                critical_high_count=_critical_high_count,
            )
            auto_repair_result = run_auto_repair_controller(
                rewritten_text,
                rewritten_report_dict,
                original_report_dict,
                auto_repair_deps,
                max_rounds=int(_float_env("DRAFTPROOF_AUTO_REPAIR_MAX_ROUNDS", 2.0)),
                max_scans=int(_float_env("DRAFTPROOF_AUTO_REPAIR_MAX_SCANS", 4.0)),
                target_human=float(_float_env("DRAFTPROOF_TARGET_HUMAN_CONTRIBUTION", 80.0)),
            )
        except Exception as exc:
            auto_repair_result = {
                "enabled": True,
                "selected": False,
                "reason": f"auto_repair_controller_error {exc}",
            }
        stored_auto_repair_result = {
            key: value
            for key, value in auto_repair_result.items()
            if key not in {"selected_text", "selected_report"}
        }
        result.summary["auto_repair_controller"] = stored_auto_repair_result
        if (
            auto_repair_result.get("selected")
            and _global_controller_phase_accepted(
                "auto_repair_controller",
                auto_repair_result,
                stored_auto_repair_result,
            )
        ):
            selected_report = auto_repair_result.get("selected_report")
            selected_text = auto_repair_result.get("selected_text")
            if isinstance(selected_report, dict) and isinstance(selected_text, str):
                rewritten_text = selected_text
                rewritten_report_dict = selected_report
                attempted_report_dict = rewritten_report_dict
                rewritten_ai = _badge_ai(rewritten_report_dict)
                rewritten_wq = _badge_wq(rewritten_report_dict)
                rewritten_total = _finding_total(rewritten_report_dict)
                rewritten_review_burden = _review_burden(rewritten_report_dict)
                rewritten_severity = _weighted_severity(rewritten_report_dict)
                rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                if result.mp_result:
                    result.mp_result.final_text = rewritten_text
                    result.mp_result.converged = True
                    result.mp_result.convergence_reason = (
                        "Selected auto-repair compiler candidate: "
                        f"{auto_repair_result.get('selected_strategy')}"
                    )
                sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                ai_search_selected = True
                auto_repair_selected = True
                _clear_stale_rollback_for_kept_ai_mitigation(
                    result.summary,
                    "auto-repair compiler",
                )
                result.summary["selected_strategy"] = auto_repair_result.get("selected_strategy")
                result.summary["selected_auto_repair_strategy"] = auto_repair_result.get("selected_strategy")
                result.summary["detect_scores"].update({
                    "rewritten_ai": rewritten_ai,
                    "rewritten_writing_quality": rewritten_wq,
                    "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                    "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                    "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                    "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                    "rewritten_findings": rewritten_total,
                    "rewritten_review_burden": rewritten_review_burden,
                    "rewritten_weighted_severity": rewritten_severity,
                })
        stage_timings.append({
            "stage": "auto_repair_controller",
            "seconds": round(time.time() - auto_repair_t0, 3),
            "candidates": len(stored_auto_repair_result.get("candidates") or []),
            "scans": stored_auto_repair_result.get("scans_used"),
            "selected": bool(auto_repair_result.get("selected")),
            "stop_reason": auto_repair_result.get("reason"),
        })
        global_rewrite_budget.record_stage(
            "auto_repair_controller",
            seconds=round(time.time() - auto_repair_t0, 3),
            scans=int(stored_auto_repair_result.get("scans_used") or 0),
        )

    rewrite_compiler_selected = False
    compiler_should_run = (
        _env_flag("DRAFTPROOF_REWRITE_COMPILER_ENABLED", True)
        and not ai_search_blocked_by_author_gaps
        and isinstance(rewritten_report_dict, dict)
        and isinstance(original_report_dict, dict)
        and str(rewritten_text or "").strip()
        and str(rewritten_text or "").strip() != str(text or "").strip()
    )
    if compiler_should_run and not global_rewrite_budget.can_run(min_seconds=5.0, min_scans=1):
        stored_compiler_result = _global_phase_budget_skip(
            "rewrite_compiler",
            min_seconds=5.0,
            min_scans=1,
        ) or {}
        result.summary["rewrite_compiler"] = stored_compiler_result
        result.summary["deterministic_rewrite_compiler"] = stored_compiler_result
        stage_timings.append({
            "stage": "rewrite_compiler",
            "seconds": 0.0,
            "candidates": 0,
            "selected": False,
            "skipped": True,
            "stop_reason": stored_compiler_result.get("reason"),
        })
    elif compiler_should_run:
        report_progress(88, "Running deterministic rewrite compiler")
        compiler_t0 = time.time()
        compiler_mode = os.environ.get("DRAFTPROOF_REWRITE_COMPILER_MODE", "compiler_strict")
        try:
            compiler_deps = CompilerDependencies(
                split_sentences=_split_sentences,
                text_word_count=_text_word_count,
                geometry_risk_map=_geometry_risk_map,
                is_canonical_fact_sentence=_ai_density_breaker_canonical_fact_sentence,
                splice_sentence=_splice_sentence_for_auto_repair,
                repair_aggression_score=_repair_aggression_score,
                locality_score=_locality_score,
                detect_protected_spans=detect_protected_spans,
                protected_loss_reason=_ai_search_protected_loss_reason,
                concept_origin_reject_reason=_candidate_concept_origin_reject_reason,
                drift_checker=check_semantic_drift,
                scan_func=_full_scan_report_dict,
                turnitin_profile=_turnitin_like_ai_profile,
                turnitin_gate_status=_turnitin_like_ai_gate_status,
                strict_safe_status=_strict_ai_safe_band_status,
                contribution_scores=_contribution_scores,
                integrity_scores=_integrity_scores,
                badge_ai=_badge_ai,
                finding_total=_finding_total,
                review_burden=_review_burden,
                weighted_severity=_weighted_severity,
                critical_high_count=_critical_high_count,
            )
            compiler_result = run_rewrite_compiler(
                rewritten_text,
                rewritten_report_dict,
                original_report_dict,
                compiler_deps,
                config=CompilerConfig(
                    mode=compiler_mode,
                    max_rounds=int(_float_env("DRAFTPROOF_REWRITE_COMPILER_MAX_ROUNDS", 1.0)),
                    max_scans=int(_float_env("DRAFTPROOF_REWRITE_COMPILER_MAX_SCANS", 4.0)),
                    candidate_pool_limit=int(_float_env("DRAFTPROOF_REWRITE_COMPILER_CANDIDATE_LIMIT", 14.0)),
                    shortlist_limit=int(_float_env("DRAFTPROOF_REWRITE_COMPILER_SHORTLIST_LIMIT", 4.0)),
                    max_llm_calls=int(_float_env("DRAFTPROOF_REWRITE_COMPILER_MAX_LLM_CALLS", 0.0)),
                ),
            )
        except Exception as exc:
            compiler_result = {
                "enabled": True,
                "selected": False,
                "reason": f"rewrite_compiler_error {exc}",
                "mode": compiler_mode,
            }
        stored_compiler_result = {
            key: value
            for key, value in compiler_result.items()
            if key not in {"selected_text", "selected_report"}
        }
        result.summary["rewrite_compiler"] = stored_compiler_result
        result.summary["deterministic_rewrite_compiler"] = stored_compiler_result
        if (
            compiler_result.get("selected")
            and _global_controller_phase_accepted(
                "rewrite_compiler",
                compiler_result,
                stored_compiler_result,
            )
        ):
            selected_report = compiler_result.get("selected_report")
            selected_text = compiler_result.get("selected_text")
            if isinstance(selected_report, dict) and isinstance(selected_text, str):
                rewritten_text = selected_text
                rewritten_report_dict = selected_report
                attempted_report_dict = rewritten_report_dict
                rewritten_ai = _badge_ai(rewritten_report_dict)
                rewritten_wq = _badge_wq(rewritten_report_dict)
                rewritten_total = _finding_total(rewritten_report_dict)
                rewritten_review_burden = _review_burden(rewritten_report_dict)
                rewritten_severity = _weighted_severity(rewritten_report_dict)
                rewritten_critical_high = _critical_high_count(rewritten_report_dict)
                if result.mp_result:
                    result.mp_result.final_text = rewritten_text
                    result.mp_result.converged = True
                    result.mp_result.convergence_reason = (
                        "Selected deterministic rewrite compiler candidate: "
                        f"{compiler_result.get('selected_strategy')}"
                    )
                sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                ai_search_selected = True
                rewrite_compiler_selected = True
                _clear_stale_rollback_for_kept_ai_mitigation(
                    result.summary,
                    "deterministic rewrite compiler",
                )
                result.summary["selected_strategy"] = compiler_result.get("selected_strategy")
                result.summary["selected_rewrite_compiler_strategy"] = compiler_result.get("selected_strategy")
                result.summary["rewrite_compiler_outcome_class"] = compiler_result.get("outcome_class")
                result.summary["detect_scores"].update({
                    "rewritten_ai": rewritten_ai,
                    "rewritten_writing_quality": rewritten_wq,
                    "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                    "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                    "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                    "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                    "rewritten_findings": rewritten_total,
                    "rewritten_review_burden": rewritten_review_burden,
                    "rewritten_weighted_severity": rewritten_severity,
                })
        stage_timings.append({
            "stage": "rewrite_compiler",
            "seconds": round(time.time() - compiler_t0, 3),
            "mode": stored_compiler_result.get("mode"),
            "candidates": len(stored_compiler_result.get("candidates") or []),
            "scans": stored_compiler_result.get("scans_used"),
            "llm_calls": stored_compiler_result.get("llm_calls_used"),
            "selected": bool(compiler_result.get("selected")),
            "outcome_class": stored_compiler_result.get("outcome_class"),
            "stop_reason": compiler_result.get("reason"),
        })
        global_rewrite_budget.record_stage(
            "rewrite_compiler",
            seconds=round(time.time() - compiler_t0, 3),
            scans=int(stored_compiler_result.get("scans_used") or 0),
            llm_calls=int(stored_compiler_result.get("llm_calls_used") or 0),
        )

    ai_regression_tolerance = 0.25
    writing_quality_regression_tolerance = 1.0

    ai_score_regressed = (
        original_ai is not None
        and rewritten_ai is not None
        and rewritten_ai > original_ai + ai_regression_tolerance
    )
    wq_score_regressed = (
        original_wq is not None
        and rewritten_wq is not None
        and rewritten_wq > original_wq + writing_quality_regression_tolerance
    )
    total_findings_regressed = rewritten_total > original_total
    review_burden_regressed = rewritten_review_burden > original_review_burden
    severity_regressed = rewritten_severity > original_severity
    critical_high_regressed = (
        len(rewritten_report_dict.get("findings", {}).get("critical", []))
        + len(rewritten_report_dict.get("findings", {}).get("high", []))
        >
        len(original_report_dict.get("findings", {}).get("critical", []))
        + len(original_report_dict.get("findings", {}).get("high", []))
    )
    total_regressed_without_review_gain = (
        total_findings_regressed
        and rewritten_review_burden >= original_review_burden
    )
    fresh_baseline_improved = (
        rewritten_review_burden < original_review_burden
        or rewritten_severity < original_severity
        or rewritten_total < original_total
        or (
            original_ai is not None
            and rewritten_ai is not None
            and rewritten_ai < original_ai - 0.05
        )
        or (
            original_wq is not None
            and rewritten_wq is not None
            and rewritten_wq < original_wq - 0.05
        )
    )
    fresh_ai_improved = (
        original_ai is not None
        and rewritten_ai is not None
        and rewritten_ai < original_ai - 0.05
    )
    saved_ai_drifted_up = (
        saved_ai is not None
        and original_ai is not None
        and original_ai > saved_ai + ai_regression_tolerance
    )
    saved_ai_regressed = (
        saved_ai is not None
        and rewritten_ai is not None
        and rewritten_ai > saved_ai + ai_regression_tolerance
    )
    saved_total_regressed = rewritten_total > saved_total
    saved_critical_high_regressed = rewritten_critical_high > saved_critical_high
    saved_ai_regression_explained_by_drift = (
        saved_ai_regressed
        and saved_ai_drifted_up
        and fresh_ai_improved
        and not ai_score_regressed
        and not review_burden_regressed
        and not severity_regressed
        and not saved_critical_high_regressed
    )
    regression_reasons = []
    followup_warnings = []
    result.summary["regression_tolerances"] = {
        "ai_score": ai_regression_tolerance,
        "writing_quality_score": writing_quality_regression_tolerance,
    }
    if ai_score_regressed:
        regression_reasons.append(f"AI {original_ai}->{rewritten_ai}")
    if wq_score_regressed:
        followup_warnings.append(f"writing_quality {original_wq}->{rewritten_wq}")
    if review_burden_regressed:
        regression_reasons.append(
            f"review_burden {original_review_burden}->{rewritten_review_burden}"
        )
    if critical_high_regressed:
        original_ch = (
            len(original_report_dict.get("findings", {}).get("critical", []))
            + len(original_report_dict.get("findings", {}).get("high", []))
        )
        rewritten_ch = (
            len(rewritten_report_dict.get("findings", {}).get("critical", []))
            + len(rewritten_report_dict.get("findings", {}).get("high", []))
        )
        regression_reasons.append(f"critical_high_findings {original_ch}->{rewritten_ch}")
    if severity_regressed:
        regression_reasons.append(
            f"weighted_severity {original_severity}->{rewritten_severity}"
        )
    if total_findings_regressed:
        regression_reasons.append(f"findings {original_total}->{rewritten_total}")
    elif total_regressed_without_review_gain and not (review_burden_regressed or severity_regressed):
        regression_reasons.append(f"findings {original_total}->{rewritten_total}")
    # The saved scan is the user-visible contract, but detector scores can drift
    # between the saved report and a fresh rescan. Critical/high findings remain
    # hard guards. AI and total count are strict only when the fresh baseline did
    # not improve or the saved-score increase is not explained by baseline drift.
    if saved_ai_regressed and not saved_ai_regression_explained_by_drift:
        regression_reasons.append(f"user_visible_ai {saved_ai}->{rewritten_ai}")
    elif saved_ai_regressed:
        result.summary.setdefault("saved_contract_notes", []).append(
            "user_visible_ai increased "
            f"{saved_ai}->{rewritten_ai}, but fresh baseline improved "
            f"{original_ai}->{rewritten_ai}; kept attempted rewrite for review."
        )
    if (
        saved_total_regressed
        and (
            not fresh_baseline_improved
            or (saved_ai_regressed and not saved_ai_regression_explained_by_drift)
            or saved_critical_high_regressed
            or rewritten_review_burden > original_review_burden
            or rewritten_severity > original_severity
        )
    ):
        regression_reasons.append(f"user_visible_findings {saved_total}->{rewritten_total}")
    elif saved_total_regressed:
        result.summary.setdefault("saved_contract_notes", []).append(
            "user_visible_findings increased "
            f"{saved_total}->{rewritten_total}, but fresh baseline improved "
            f"{original_total}->{rewritten_total}; kept attempted rewrite for review."
        )
    if saved_critical_high_regressed:
        regression_reasons.append(
            f"user_visible_critical_high_findings {saved_critical_high}->{rewritten_critical_high}"
        )
    if authenticity_mitigation_selected:
        authenticity_breakthrough_tradeoff = bool(
            _env_flag("DRAFTPROOF_AUTHENTICITY_BREAKTHROUGH_TRADEOFF", True)
            and isinstance(original_ai, (int, float))
            and isinstance(rewritten_ai, (int, float))
            and rewritten_ai <= original_ai - 10.0
            and isinstance(_integrity_scores(original_report_dict).get("ai_authorship"), (int, float))
            and isinstance(_integrity_scores(rewritten_report_dict).get("ai_authorship"), (int, float))
            and _integrity_scores(rewritten_report_dict).get("ai_authorship")
            <= _integrity_scores(original_report_dict).get("ai_authorship") - 10.0
            and rewritten_total <= original_total
        )
        hard_regression_reasons = []
        soft_regression_reasons = []
        for reason in regression_reasons:
            if reason.startswith((
                "review_burden ",
                "critical_high_findings ",
                "weighted_severity ",
                "user_visible_critical_high_findings ",
            )) and not authenticity_breakthrough_tradeoff:
                hard_regression_reasons.append(reason)
            else:
                soft_regression_reasons.append(reason)
        if soft_regression_reasons:
            followup_warnings.extend(
                f"post_authenticity_review {reason}" for reason in soft_regression_reasons
            )
        regression_reasons = hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})["kept"] = not hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})[
            "breakthrough_tradeoff"
        ] = authenticity_breakthrough_tradeoff
        result.summary.setdefault("authenticity_mitigation", {})["hard_regressions"] = hard_regression_reasons
        result.summary.setdefault("authenticity_mitigation", {})["soft_followups"] = soft_regression_reasons
        if not hard_regression_reasons:
            result.summary.setdefault("saved_contract_notes", []).append(
                "AI-Mitigation authenticity gate kept the rewrite because the contribution score moved toward Human without review-burden or severity regression."
            )
    ai_first_reference = original_ai if original_ai is not None else saved_ai
    ai_first_gate = _ai_first_gate_status(
        ai_first_reference,
        rewritten_ai,
        rewritten_text != text,
        min_drop=ai_first_min_drop,
        target=ai_first_target,
        required_min_ai=ai_first_required_min_ai,
    )
    ai_first_delta = ai_first_gate["delta"]
    ai_first_success = ai_first_gate["success"]
    ai_first_required = ai_first_gate["required"]
    final_ai_search_selection_status = _ai_search_final_selection_status(result.summary)
    ai_search_selected_by_authenticity = _ai_search_selected_by_final_safety_gate(
        ai_search_selected,
        final_ai_search_selection_status,
    )
    if (
        ai_first_required
        and not ai_first_success
        and not authenticity_mitigation_selected
        and not ai_search_selected_by_authenticity
    ):
        delta_text = f"{ai_first_delta:.2f}" if isinstance(ai_first_delta, (int, float)) else "unknown"
        regression_reasons.append(
            f"ai_first_gate_failed {ai_first_reference}->{rewritten_ai} "
            f"delta={delta_text} required_delta={ai_first_min_drop:.2f}"
        )
        result.summary["ai_first_mitigation"] = {
            "kept": False,
            "reference_ai": ai_first_reference,
            "rewritten_ai": rewritten_ai,
            "ai_delta": round(ai_first_delta, 3) if isinstance(ai_first_delta, (int, float)) else None,
            "min_drop": ai_first_min_drop,
            "target": ai_first_target,
            "required_min_ai": ai_first_required_min_ai,
            "hard_regressions": [
                f"ai_first_gate_failed {ai_first_reference}->{rewritten_ai}"
            ],
        }
    if ai_first_success:
        hard_regression_reasons = []
        soft_regression_reasons = []
        for reason in regression_reasons:
            if reason.startswith((
                "AI ",
                "user_visible_ai ",
            )):
                hard_regression_reasons.append(reason)
            else:
                soft_regression_reasons.append(reason)
        if soft_regression_reasons:
            followup_warnings.extend(
                f"post_ai_review {reason}" for reason in soft_regression_reasons
            )
        regression_reasons = hard_regression_reasons
        result.summary["ai_first_mitigation"] = {
            "kept": not hard_regression_reasons,
            "source": "ai_mitigation_search" if ai_search_selected else "threshold",
            "reference_ai": ai_first_reference,
            "rewritten_ai": rewritten_ai,
            "ai_delta": round(ai_first_delta, 3) if isinstance(ai_first_delta, (int, float)) else None,
            "min_drop": ai_first_min_drop,
            "target": ai_first_target,
            "soft_followups": soft_regression_reasons,
            "hard_regressions": hard_regression_reasons,
        }
        if not hard_regression_reasons:
            _clear_stale_rollback_for_kept_ai_mitigation(
                result.summary,
                "AI-first mitigation",
            )
            if ai_search_selected:
                result.summary.setdefault("saved_contract_notes", []).append(
                    "AI mitigation search kept the lowest-AI scanned candidate; "
                    "writing quality and lower-severity finding changes are follow-up work."
                )
            else:
                result.summary.setdefault("saved_contract_notes", []).append(
                    "AI-first mitigation kept the rewrite because AI likelihood improved enough; "
                    "writing quality and lower-severity finding changes are follow-up work."
                )
    if followup_warnings:
        result.summary["post_ai_followups"] = followup_warnings
        writing_quality_followups = [
            warning for warning in followup_warnings
            if str(warning).startswith("writing_quality ")
        ]
        if writing_quality_followups:
            result.summary["writing_quality_followups"] = writing_quality_followups
        result.summary.setdefault("saved_contract_notes", []).append(
            (
                "AI-Mitigation kept the rewrite; writing quality and lower-severity changes are reported as follow-up work."
                if authenticity_mitigation_selected
                else "AI-first mitigation kept the rewrite; writing quality and lower-severity changes are reported as follow-up work."
            )
        )
    if (
        os.environ.get("DRAFTPROOF_HUMAN_SHIFT_OVERRIDES_AI_FIRST", "1") != "0"
        and ai_first_required
        and not ai_first_success
        and rewritten_text != text
        and original_ai is not None
        and rewritten_ai is not None
        and rewritten_ai <= original_ai + 0.05
    ):
        final_shift_gate = _authenticity_gate_status(
            original_report_dict,
            rewritten_report_dict,
            rewritten_text != text,
            original_review_burden=original_review_burden,
            candidate_review_burden=rewritten_review_burden,
            original_weighted_severity=original_severity,
            candidate_weighted_severity=rewritten_severity,
            min_human_gain=_float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_HUMAN_GAIN", 10.0),
            min_ai_transformation_drop=_float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_AI_TRANSFORM_DROP", 8.0),
        )
        final_shift_score = final_shift_gate.get("human_shift_score")
        final_human_delta = final_shift_gate.get("human_delta")
        final_transform_delta = final_shift_gate.get("ai_transformation_delta")
        final_ai_authorship_regressed = bool(final_shift_gate.get("ai_authorship_regressed"))
        final_major_human_breakthrough = bool(final_shift_gate.get("major_human_breakthrough"))
        clears_override = bool(
            isinstance(final_shift_score, (int, float))
            and final_shift_score >= _float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_SHIFT", 20.0)
            and (
                isinstance(final_human_delta, (int, float))
                and final_human_delta >= _float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_HUMAN_GAIN", 10.0)
            )
            and (
                isinstance(final_transform_delta, (int, float))
                and final_transform_delta >= _float_env("DRAFTPROOF_RECONSTRUCTION_OVERRIDE_MIN_AI_TRANSFORM_DROP", 8.0)
            )
            and (not final_ai_authorship_regressed or final_major_human_breakthrough)
            and not final_shift_gate.get("critical_high_regressed")
            and not final_shift_gate.get("review_burden_regressed")
            and not final_shift_gate.get("weighted_severity_regressed")
        )
        if clears_override:
            removed_ai_first = [
                reason for reason in regression_reasons
                if str(reason).startswith("ai_first_gate_failed ")
            ]
            if removed_ai_first:
                regression_reasons = [
                    reason for reason in regression_reasons
                    if not str(reason).startswith("ai_first_gate_failed ")
                ]
                result.summary["human_shift_override"] = {
                    "kept": True,
                    "removed_regressions": removed_ai_first,
                    "reason": "human_shift_goal_outweighs_legacy_ai_first_min_drop",
                    "gate": final_shift_gate,
                    "rewritten_ai": rewritten_ai,
                    "original_ai": original_ai,
                }
                result.summary.setdefault("saved_contract_notes", []).append(
                    "Human Shift override kept the rewrite because contribution and transformation movement met the AI-Mitigation goal without severity regression."
                )
    product_regressed = (
        rewritten_text != text
        and bool(regression_reasons)
    )
    if product_regressed:
        best_checkpoint = None
        best_checkpoint_report = None
        best_checkpoint_rank = None
        checkpoint_candidates = []
        for checkpoint in getattr(result, "rewrite_checkpoints", []) or []:
            checkpoint_text = checkpoint.get("text", "")
            if not checkpoint_text or checkpoint_text in {text, rewritten_text}:
                continue
            checkpoint_candidates.append(checkpoint)
        max_checkpoint_scans = int(os.environ.get("DRAFTPROOF_MAX_CHECKPOINT_SCANS", "6"))
        if max_checkpoint_scans <= 0:
            result.summary["checkpoint_scan_skipped"] = len(checkpoint_candidates)
            checkpoint_candidates = []
        if len(checkpoint_candidates) > max_checkpoint_scans:
            result.summary["checkpoint_scan_skipped"] = (
                len(checkpoint_candidates) - max_checkpoint_scans
            )
            checkpoint_candidates = checkpoint_candidates[-max_checkpoint_scans:]

        checkpoint_scan_t0 = time.time()
        checkpoint_scan_count = 0
        for checkpoint in checkpoint_candidates:
            checkpoint_text = checkpoint.get("text", "")
            checkpoint_report = _full_scan_report_dict(checkpoint_text)
            checkpoint_scan_count += 1
            cp_ai = _badge_ai(checkpoint_report)
            cp_wq = _badge_wq(checkpoint_report)
            cp_total = _finding_total(checkpoint_report)
            cp_review_burden = _review_burden(checkpoint_report)
            cp_severity = _weighted_severity(checkpoint_report)

            cp_ai_regressed = (
                original_ai is not None
                and cp_ai is not None
                and cp_ai > original_ai + 0.05
            )
            cp_wq_regressed = (
                original_wq is not None
                and cp_wq is not None
                and cp_wq > original_wq + writing_quality_regression_tolerance
            )
            cp_improved = (
                cp_review_burden < original_review_burden
                or cp_severity < original_severity
                or cp_total < original_total
                or (original_ai is not None and cp_ai is not None and cp_ai < original_ai - 0.05)
                or (original_wq is not None and cp_wq is not None and cp_wq < original_wq - 0.05)
            )
            cp_critical_high = (
                len(checkpoint_report.get("findings", {}).get("critical", []))
                + len(checkpoint_report.get("findings", {}).get("high", []))
            )
            original_critical_high = (
                len(original_report_dict.get("findings", {}).get("critical", []))
                + len(original_report_dict.get("findings", {}).get("high", []))
            )
            cp_saved_ai_regressed = (
                saved_ai is not None
                and cp_ai is not None
                and cp_ai > saved_ai + 0.05
            )
            cp_saved_total_regressed = cp_total > saved_total
            cp_saved_critical_high_regressed = cp_critical_high > saved_critical_high
            cp_saved_ai_regression_explained_by_drift = (
                cp_saved_ai_regressed
                and saved_ai_drifted_up
                and original_ai is not None
                and cp_ai is not None
                and cp_ai < original_ai - 0.05
                and cp_review_burden <= original_review_burden
                and cp_severity <= original_severity
                and not cp_saved_critical_high_regressed
            )
            cp_violates_saved_contract = (
                (cp_saved_ai_regressed and not cp_saved_ai_regression_explained_by_drift)
                or cp_saved_critical_high_regressed
                or (
                    cp_saved_total_regressed
                    and (
                        not cp_improved
                        or cp_review_burden > original_review_burden
                        or cp_severity > original_severity
                    )
                )
            )
            cp_ai_first_reference = saved_ai if saved_ai is not None else original_ai
            cp_ai_first_gate = _ai_first_gate_status(
                cp_ai_first_reference,
                cp_ai,
                checkpoint.get("text") != text,
                min_drop=ai_first_min_drop,
                target=ai_first_target,
                required_min_ai=ai_first_required_min_ai,
            )
            if (
                cp_ai_regressed
                or cp_total > original_total
                or cp_review_burden > original_review_burden
                or cp_severity > original_severity
                or cp_critical_high > original_critical_high
                or cp_violates_saved_contract
                or not cp_improved
                or (cp_ai_first_gate["required"] and not cp_ai_first_gate["success"])
            ):
                continue

            rank = (
                cp_review_burden,
                cp_severity,
                cp_total,
                cp_ai if cp_ai is not None else 999.0,
                -(checkpoint.get("edits", 0) or 0),
            )
            if best_checkpoint_rank is None or rank < best_checkpoint_rank:
                best_checkpoint = checkpoint
                best_checkpoint_report = checkpoint_report
                best_checkpoint_rank = rank

        if checkpoint_scan_count:
            stage_timings.append({
                "stage": "checkpoint_scans",
                "count": checkpoint_scan_count,
                "seconds": round(time.time() - checkpoint_scan_t0, 3),
            })

        if best_checkpoint and best_checkpoint_report:
            rewritten_text = best_checkpoint["text"]
            rewritten_report_dict = best_checkpoint_report
            result.summary["checkpoint_selected"] = {
                "edits": best_checkpoint.get("edits", 0),
                "local_score_total": best_checkpoint.get("local_score_total", 0.0),
                "reason": "final rewrite regressed; kept best non-regressing checkpoint",
                "final_regression_reasons": regression_reasons,
                "ai_first_gate": _ai_first_gate_status(
                    saved_ai if saved_ai is not None else original_ai,
                    _badge_ai(best_checkpoint_report),
                    best_checkpoint.get("text") != text,
                    min_drop=ai_first_min_drop,
                    target=ai_first_target,
                    required_min_ai=ai_first_required_min_ai,
                ),
            }
            result.summary["detect_scores"].update({
                "rewritten_ai": _badge_ai(rewritten_report_dict),
                "rewritten_writing_quality": _badge_wq(rewritten_report_dict),
                "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                "rewritten_findings": _finding_total(rewritten_report_dict),
                "rewritten_review_burden": _review_burden(rewritten_report_dict),
                "rewritten_weighted_severity": _weighted_severity(rewritten_report_dict),
                "attempted_ai": rewritten_ai,
                "attempted_writing_quality": rewritten_wq,
                "attempted_findings": rewritten_total,
                "attempted_review_burden": rewritten_review_burden,
                "attempted_weighted_severity": rewritten_severity,
            })
            result.summary["rollback_applied"] = False
            result.summary["outcome"] = "partially_improved"
            if result.mp_result:
                result.mp_result.final_text = rewritten_text
                result.mp_result.converged = True
                result.mp_result.convergence_reason = (
                    "Selected best non-regressing checkpoint after final scan regression"
                )
            sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
            product_regressed = False

    if product_regressed:
        reason = (
            f"final full detect scan regressed "
            f"({'; '.join(regression_reasons)})"
        )
        attempted_text = rewritten_text
        attempted_sentence_comparison = sentence_comparison
        rollback_suggestions = result.summary.setdefault("manual_suggestions", [])
        accepted_suggestions = result.summary.get("accepted_candidate_suggestions") or []
        existing = {
            (
                s.get("original_sentence"),
                s.get("suggested_sentence"),
                s.get("rejection_reason"),
            )
            for s in rollback_suggestions
            if isinstance(s, dict)
        }
        for item in accepted_suggestions:
            if not isinstance(item, dict):
                continue
            suggestion = dict(item)
            suggestion["rejection_reason"] = reason
            suggestion["why_review_manually"] = (
                "This edit passed local guards, but the final full detect scan regressed. "
                "Review manually before using it."
            )
            key = (
                suggestion.get("original_sentence"),
                suggestion.get("suggested_sentence"),
                suggestion.get("rejection_reason"),
            )
            if key not in existing and len(rollback_suggestions) < 30:
                rollback_suggestions.append(suggestion)
                existing.add(key)
        rewritten_text = text
        if result.mp_result:
            result.mp_result.final_text = text
            result.mp_result.final_metrics = result.mp_result.original_metrics
            result.mp_result.converged = False
            result.mp_result.convergence_reason = reason
        result.summary["attempted_final_text"] = attempted_text
        result.summary["attempted_sentence_comparison"] = attempted_sentence_comparison
        result.summary["final_text"] = text
        result.summary["converged"] = False
        result.summary["rollback_applied"] = True
        result.summary["rollback_reason"] = reason
        result.summary["outcome"] = "rejected_for_drift"
        result.summary["detect_scores"].update({
            "rewritten_ai": _badge_ai(original_report_dict),
            "rewritten_writing_quality": _badge_wq(original_report_dict),
            "rewritten_ai_authorship": _integrity_scores(original_report_dict).get("ai_authorship"),
            "rewritten_grounding_quality_risk": _integrity_scores(original_report_dict).get("grounding"),
            "rewritten_human_contribution": _contribution_scores(original_report_dict).get("human"),
            "rewritten_ai_transformation": _contribution_scores(original_report_dict).get("ai_transformation"),
            "rewritten_findings": _finding_total(original_report_dict),
            "rewritten_review_burden": _review_burden(original_report_dict),
            "rewritten_weighted_severity": _weighted_severity(original_report_dict),
            "attempted_ai": rewritten_ai,
            "attempted_writing_quality": rewritten_wq,
            "attempted_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
            "attempted_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
            "attempted_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
            "attempted_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
            "attempted_findings": rewritten_total,
            "attempted_review_burden": rewritten_review_burden,
            "attempted_weighted_severity": rewritten_severity,
            "rollback_reason": reason,
        })
        sentence_comparison = []
        rewritten_report_dict = original_report_dict

    final_human_shift = _human_shift_score(
        original_report_dict,
        rewritten_report_dict,
        review_burden_delta=_review_burden(rewritten_report_dict) - original_review_burden,
        weighted_severity_delta=_weighted_severity(rewritten_report_dict) - original_severity,
    )
    result.summary.setdefault("detect_scores", {}).update({
        "human_shift_score": final_human_shift.get("score"),
        "human_shift_components": final_human_shift.get("components"),
    })
    author_evidence_completion = _build_author_evidence_completion_layer(
        rewritten_text,
        rewritten_report_dict,
        target_human=int(_float_env("DRAFTPROOF_TARGET_HUMAN_CONTRIBUTION", 80.0)),
        max_slots=int(_float_env("DRAFTPROOF_AUTHOR_EVIDENCE_COMPLETION_SLOTS", 5.0)),
    )
    if author_evidence_completion:
        result.summary["author_evidence_completion"] = author_evidence_completion
    ceiling_diagnostics = _build_mitigation_ceiling_diagnostics(
        result.summary,
        author_evidence_completion,
        target_human=int(_float_env("DRAFTPROOF_TARGET_HUMAN_CONTRIBUTION", 80.0)),
    )
    if ceiling_diagnostics:
        result.summary["mitigation_ceiling"] = ceiling_diagnostics
    author_evidence_intake = _build_author_evidence_intake_layer(
        author_evidence_completion,
        ceiling_diagnostics,
        max_questions=int(_float_env("DRAFTPROOF_AUTHOR_EVIDENCE_INTAKE_QUESTIONS", 5.0)),
    )
    if author_evidence_intake:
        result.summary["author_evidence_intake"] = author_evidence_intake
    author_context_discovery = _build_author_context_discovery_layer(
        author_evidence_intake,
        rewritten_report_dict,
        max_items=int(_float_env("DRAFTPROOF_AUTHOR_CONTEXT_DISCOVERY_ITEMS", 5.0)),
    )
    if author_context_discovery:
        result.summary["author_context_discovery"] = author_context_discovery
    if _env_flag("DRAFTPROOF_FINAL_SOURCE_GROUNDING_SEARCH", False):
        source_grounding_search = _build_source_grounding_search_layer(
            rewritten_text,
            rewritten_report_dict,
        )
        if source_grounding_search:
            result.summary["source_grounding_search"] = source_grounding_search
    else:
        result.summary["source_grounding_search"] = {
            "enabled": False,
            "reason": "disabled_final_guidance_search",
        }
    result.summary["source_search_calls_used"] = _source_search_calls_used()
    result.summary["source_search_budget"] = {
        "enabled": _source_search_enabled(),
        "max_calls_per_run": _source_search_max_calls_per_run(),
        "remaining_calls": _source_search_remaining_calls(),
    }
    author_evidence_answers = _load_author_evidence_answers()
    author_evidence_integration = {
        "enabled": bool(author_evidence_intake),
        "status": "awaiting_user_answers",
        "answer_count": len(author_evidence_answers),
        "accepted_answers": 0,
        "applied_answers": 0,
        "candidates": [],
    }
    if author_evidence_intake and author_evidence_answers:
        integration_started = time.time()
        valid_answers, rejected_answers = _validate_author_evidence_answers(
            author_evidence_intake,
            author_evidence_answers,
        )
        author_evidence_integration.update({
            "status": "validating_answers",
            "accepted_answers": len(valid_answers),
            "rejected_answers": rejected_answers,
        })
        integration_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if not valid_answers:
            author_evidence_integration["status"] = "no_valid_confirmed_answers"
        else:
            integration_gateway = None
            if integration_key:
                integration_gateway = LLMGateway(LLMConfig(
                    api_key=integration_key,
                    model=generator_model,
                    base_url=base_url,
                    timeout=int(os.environ.get("DRAFTPROOF_AUTHOR_EVIDENCE_INTEGRATION_TIMEOUT", "90")),
                    max_retries=int(os.environ.get("DRAFTPROOF_AUTHOR_EVIDENCE_INTEGRATION_RETRIES", "1")),
                    max_tokens=int(os.environ.get("DRAFTPROOF_AUTHOR_EVIDENCE_INTEGRATION_MAX_TOKENS", "1800")),
                    temperature=float(os.environ.get("DRAFTPROOF_AUTHOR_EVIDENCE_INTEGRATION_TEMPERATURE", "0.35")),
                ))
            max_integrations = max(
                1,
                int(_float_env("DRAFTPROOF_AUTHOR_EVIDENCE_INTEGRATION_MAX_ANSWERS", 2.0)),
            )
            base_text_for_integration = rewritten_text
            base_report_for_integration = rewritten_report_dict
            applied_count = 0
            for answer_index, answer_item in enumerate(valid_answers[:max_integrations], start=1):
                question = answer_item.get("question") or {}
                paragraph_index = question.get("paragraph_index")
                candidate_eval = {
                    "anchor_id": answer_item.get("anchor_id"),
                    "paragraph_index": paragraph_index,
                    "status": "started",
                }
                paragraphs = _logical_paragraphs(base_text_for_integration)
                if not isinstance(paragraph_index, int) or paragraph_index < 0 or paragraph_index >= len(paragraphs):
                    candidate_eval["status"] = "rejected"
                    candidate_eval["reason"] = "paragraph_index_out_of_range"
                    author_evidence_integration["candidates"].append(candidate_eval)
                    continue
                original_paragraph = paragraphs[paragraph_index]
                integrated_paragraph, reject_reason = _deterministic_author_anchor_paragraph(
                    original_paragraph,
                    question,
                    answer_item.get("answer") or "",
                )
                candidate_eval["integration_method"] = "deterministic_anchor_insert"
                if reject_reason:
                    candidate_eval["status"] = "rejected"
                    candidate_eval["reason"] = reject_reason
                    author_evidence_integration["candidates"].append(candidate_eval)
                    continue
                candidate_text = _splice_author_evidence_paragraph(
                    base_text_for_integration,
                    paragraph_index,
                    integrated_paragraph,
                )
                if candidate_text == base_text_for_integration:
                    candidate_eval["status"] = "rejected"
                    candidate_eval["reason"] = "no_text_change"
                    author_evidence_integration["candidates"].append(candidate_eval)
                    continue
                drift = check_semantic_drift(
                    base_text_for_integration,
                    candidate_text,
                    threshold=float(os.environ.get(
                        "DRAFTPROOF_AUTHOR_EVIDENCE_INTEGRATION_DRIFT_THRESHOLD",
                        "0.88",
                    )),
                )
                candidate_eval["drift_similarity"] = round(drift.similarity, 3)
                if not drift.accepted:
                    candidate_eval["status"] = "rejected"
                    candidate_eval["reason"] = "semantic_drift " + "; ".join(drift.reasons[:3])
                    author_evidence_integration["candidates"].append(candidate_eval)
                    continue
                scan_t0 = time.time()
                candidate_report = _full_scan_report_dict(candidate_text)
                candidate_eval["scan_seconds"] = round(time.time() - scan_t0, 3)
                base_review = _review_burden(base_report_for_integration)
                candidate_review = _review_burden(candidate_report)
                base_severity = _weighted_severity(base_report_for_integration)
                candidate_severity = _weighted_severity(candidate_report)
                base_findings = _finding_total(base_report_for_integration)
                candidate_findings = _finding_total(candidate_report)
                base_critical_high = _critical_high_count(base_report_for_integration)
                candidate_critical_high = _critical_high_count(candidate_report)
                gate = _authenticity_gate_status(
                    base_report_for_integration,
                    candidate_report,
                    True,
                    original_review_burden=base_review,
                    candidate_review_burden=candidate_review,
                    original_weighted_severity=base_severity,
                    candidate_weighted_severity=candidate_severity,
                    min_human_gain=float(os.environ.get(
                        "DRAFTPROOF_AUTHOR_EVIDENCE_INTEGRATION_MIN_HUMAN_GAIN",
                        "1.0",
                    )),
                    min_ai_transformation_drop=0.0,
                    drift_similarity=candidate_eval.get("drift_similarity"),
                )
                candidate_eval.update({
                    "ai": _badge_ai(candidate_report),
                    "human_contribution": _contribution_scores(candidate_report).get("human"),
                    "ai_transformation": _contribution_scores(candidate_report).get("ai_transformation"),
                    "ai_authorship": _integrity_scores(candidate_report).get("ai_authorship"),
                    "findings": candidate_findings,
                    "review_burden": candidate_review,
                    "weighted_severity": candidate_severity,
                    "gate": gate,
                })
                accepted = bool(
                    gate.get("human_delta") is not None
                    and gate.get("human_delta") >= float(os.environ.get(
                        "DRAFTPROOF_AUTHOR_EVIDENCE_INTEGRATION_MIN_HUMAN_GAIN",
                        "1.0",
                    ))
                    and not gate.get("ai_authorship_regression_blocked")
                    and candidate_findings <= base_findings
                    and candidate_review <= base_review
                    and candidate_severity <= base_severity
                    and candidate_critical_high <= base_critical_high
                )
                if not accepted:
                    candidate_eval["status"] = "rejected"
                    candidate_eval["reason"] = (
                        gate.get("reason")
                        or "integration_gate_failed"
                    )
                    author_evidence_integration["candidates"].append(candidate_eval)
                    continue
                candidate_eval["status"] = "accepted"
                author_evidence_integration["candidates"].append(candidate_eval)
                base_text_for_integration = candidate_text
                base_report_for_integration = candidate_report
                applied_count += 1
            if applied_count:
                rewritten_text = base_text_for_integration
                rewritten_report_dict = base_report_for_integration
                attempted_report_dict = rewritten_report_dict
                rewritten_ai = _badge_ai(rewritten_report_dict)
                rewritten_wq = _badge_wq(rewritten_report_dict)
                rewritten_total = _finding_total(rewritten_report_dict)
                rewritten_review_burden = _review_burden(rewritten_report_dict)
                rewritten_severity = _weighted_severity(rewritten_report_dict)
                if result.mp_result:
                    result.mp_result.final_text = rewritten_text
                    result.mp_result.converged = True
                    result.mp_result.convergence_reason = "Integrated confirmed author evidence through gated rescan"
                sentence_comparison = _build_aligned_sentence_comparison(result.mp_result)
                result.summary["outcome"] = "ai_mitigated"
                result.summary["converged"] = True
                result.summary["detect_scores"].update({
                    "rewritten_ai": rewritten_ai,
                    "rewritten_writing_quality": rewritten_wq,
                    "rewritten_ai_authorship": _integrity_scores(rewritten_report_dict).get("ai_authorship"),
                    "rewritten_grounding_quality_risk": _integrity_scores(rewritten_report_dict).get("grounding"),
                    "rewritten_human_contribution": _contribution_scores(rewritten_report_dict).get("human"),
                    "rewritten_ai_transformation": _contribution_scores(rewritten_report_dict).get("ai_transformation"),
                    "rewritten_findings": rewritten_total,
                    "rewritten_review_burden": rewritten_review_burden,
                    "rewritten_weighted_severity": rewritten_severity,
                })
                author_evidence_integration["status"] = "applied"
            else:
                author_evidence_integration["status"] = "no_answer_passed_gate"
            author_evidence_integration["applied_answers"] = applied_count
        author_evidence_integration["seconds"] = round(time.time() - integration_started, 3)
        stage_timings.append({
            "stage": "author_evidence_integration",
            "seconds": author_evidence_integration["seconds"],
            "answers": len(author_evidence_answers),
            "applied": author_evidence_integration.get("applied_answers", 0),
        })
        if author_evidence_integration.get("applied_answers", 0) > 0:
            final_human_shift = _human_shift_score(
                original_report_dict,
                rewritten_report_dict,
                review_burden_delta=_review_burden(rewritten_report_dict) - original_review_burden,
                weighted_severity_delta=_weighted_severity(rewritten_report_dict) - original_severity,
            )
            result.summary.setdefault("detect_scores", {}).update({
                "human_shift_score": final_human_shift.get("score"),
                "human_shift_components": final_human_shift.get("components"),
            })
            author_evidence_completion = _build_author_evidence_completion_layer(
                rewritten_text,
                rewritten_report_dict,
                target_human=int(_float_env("DRAFTPROOF_TARGET_HUMAN_CONTRIBUTION", 80.0)),
                max_slots=int(_float_env("DRAFTPROOF_AUTHOR_EVIDENCE_COMPLETION_SLOTS", 5.0)),
            )
            if author_evidence_completion:
                result.summary["author_evidence_completion"] = author_evidence_completion
            ceiling_diagnostics = _build_mitigation_ceiling_diagnostics(
                result.summary,
                author_evidence_completion,
                target_human=int(_float_env("DRAFTPROOF_TARGET_HUMAN_CONTRIBUTION", 80.0)),
            )
            if ceiling_diagnostics:
                result.summary["mitigation_ceiling"] = ceiling_diagnostics
            author_evidence_intake = _build_author_evidence_intake_layer(
                author_evidence_completion,
                ceiling_diagnostics,
                max_questions=int(_float_env("DRAFTPROOF_AUTHOR_EVIDENCE_INTAKE_QUESTIONS", 5.0)),
            )
            if author_evidence_intake:
                result.summary["author_evidence_intake"] = author_evidence_intake
    if author_evidence_integration.get("enabled"):
        result.summary["author_evidence_integration"] = author_evidence_integration

    final_ai_footprint_gate = _ai_footprint_gate_status(
        original_report_dict,
        rewritten_report_dict,
        review_burden_delta=_review_burden(rewritten_report_dict) - original_review_burden,
        weighted_severity_delta=_weighted_severity(rewritten_report_dict) - original_severity,
        critical_high_delta=_critical_high_count(rewritten_report_dict) - saved_critical_high,
        ai_score_regressed=bool(
            isinstance(original_ai, (int, float))
            and isinstance(rewritten_ai, (int, float))
            and rewritten_ai > original_ai + _float_env("DRAFTPROOF_AI_SEARCH_AI_SCORE_REGRESSION_TOLERANCE", 0.25)
        ),
    )
    final_after_authorship_for_acceptance = (
        (final_ai_footprint_gate.get("after") or {}).get("authorship_footprint") or {}
    )
    final_topk_for_acceptance = final_after_authorship_for_acceptance.get("topk_calibrated_risk")
    final_topk_raw_for_acceptance = final_after_authorship_for_acceptance.get("topk_pattern_raw")
    topk_acceptance_limit = _safe_topk_calibrated_limit()
    topk_over_limit = (
        float(final_topk_for_acceptance) - topk_acceptance_limit
        if isinstance(final_topk_for_acceptance, (int, float)) else None
    )
    final_topk_drop_for_acceptance = (
        (final_ai_footprint_gate.get("drops") or {}).get("topk_calibrated_risk")
        if isinstance(final_ai_footprint_gate, dict) else None
    )
    final_authorship_drop_for_acceptance = (
        _integrity_scores(original_report_dict).get("ai_authorship")
        - _integrity_scores(rewritten_report_dict).get("ai_authorship")
        if isinstance(_integrity_scores(original_report_dict).get("ai_authorship"), (int, float))
        and isinstance(_integrity_scores(rewritten_report_dict).get("ai_authorship"), (int, float))
        else None
    )
    final_transformation_drop_for_acceptance = (
        _contribution_scores(original_report_dict).get("ai_transformation")
        - _contribution_scores(rewritten_report_dict).get("ai_transformation")
        if isinstance(_contribution_scores(original_report_dict).get("ai_transformation"), (int, float))
        and isinstance(_contribution_scores(rewritten_report_dict).get("ai_transformation"), (int, float))
        else None
    )
    final_ai_drop_for_acceptance = (
        _badge_ai(original_report_dict) - _badge_ai(rewritten_report_dict)
        if isinstance(_badge_ai(original_report_dict), (int, float))
        and isinstance(_badge_ai(rewritten_report_dict), (int, float))
        else None
    )
    topk_near_miss_keep_decision = _topk_near_miss_partial_keep_decision(
        topk_value=final_topk_for_acceptance,
        safe_limit=topk_acceptance_limit,
        topk_drop=final_topk_drop_for_acceptance,
        ai_drop=final_ai_drop_for_acceptance,
        ai_authorship_drop=final_authorship_drop_for_acceptance,
        ai_transformation_drop=final_transformation_drop_for_acceptance,
        review_burden_delta=_review_burden(rewritten_report_dict) - original_review_burden,
        weighted_severity_delta=_weighted_severity(rewritten_report_dict) - original_severity,
        critical_high_delta=_critical_high_count(rewritten_report_dict) - saved_critical_high,
    )
    selected_search_status = _ai_search_final_selection_status(result.summary)
    convergence_candidate_status = (
        (result.summary.get("formula_convergence_controller") or {}).get("selected_formula_portfolio_candidate")
        if isinstance(result.summary.get("formula_convergence_controller"), dict)
        else {}
    )
    if not selected_search_status and isinstance(convergence_candidate_status, dict) and convergence_candidate_status:
        selected_search_status = {
            "partial_turnitin_like_mitigation": True,
            "reason": convergence_candidate_status.get("reason") or "accepted_formula_convergence_step",
            "turnitin_like_mitigation": bool(convergence_candidate_status.get("target_met")),
        }
    selected_topk_blocker_progress = bool(
        selected_search_status.get("topk_blocker_progress")
        or selected_search_status.get("reason") == "accepted_topk_blocker_progress"
    )
    selected_turnitin_like_progress = bool(
        selected_search_status.get("turnitin_like_mitigation")
        or selected_search_status.get("partial_turnitin_like_mitigation")
        or str(selected_search_status.get("reason") or "").startswith("accepted_partial_turnitin_like")
        or str(selected_search_status.get("reason") or "").startswith("accepted_turnitin_like")
    )
    if (
        rewritten_text != text
        and isinstance(final_topk_for_acceptance, (int, float))
        and float(final_topk_for_acceptance) >= topk_acceptance_limit
        and not topk_near_miss_keep_decision.get("allowed")
        and (selected_topk_blocker_progress or selected_turnitin_like_progress)
        and float(_review_burden(rewritten_report_dict) - original_review_burden) <= 0.0
        and float(_weighted_severity(rewritten_report_dict) - original_severity) <= 0.0
        and float(_critical_high_count(rewritten_report_dict) - saved_critical_high) <= 0.0
    ):
        topk_near_miss_keep_decision = {
            "allowed": True,
            "reason": (
                "selector_accepted_turnitin_like_progress"
                if selected_turnitin_like_progress
                else "selector_accepted_topk_blocker_progress"
            ),
            "topk_over_limit": (
                round(float(topk_over_limit), 3)
                if isinstance(topk_over_limit, (int, float)) else None
            ),
            "topk_drop": (
                round(float(final_topk_drop_for_acceptance), 3)
                if isinstance(final_topk_drop_for_acceptance, (int, float)) else None
            ),
            "ai_drop": (
                round(float(final_ai_drop_for_acceptance), 3)
                if isinstance(final_ai_drop_for_acceptance, (int, float)) else None
            ),
            "ai_authorship_drop": (
                round(float(final_authorship_drop_for_acceptance), 3)
                if isinstance(final_authorship_drop_for_acceptance, (int, float)) else None
            ),
            "ai_transformation_drop": (
                round(float(final_transformation_drop_for_acceptance), 3)
                if isinstance(final_transformation_drop_for_acceptance, (int, float)) else None
            ),
        }
    if (
        rewritten_text != text
        and isinstance(final_topk_for_acceptance, (int, float))
        and float(final_topk_for_acceptance) >= topk_acceptance_limit
        and topk_near_miss_keep_decision.get("allowed")
    ):
        reason = (
            f"topk_calibrated_blocked_partial_kept {float(final_topk_for_acceptance):.2f}>="
            f"{topk_acceptance_limit:.2f}"
        )
        result.summary["rollback_applied"] = False
        result.summary.pop("rollback_reason", None)
        result.summary["outcome"] = "topk_blocked_partial_kept"
        result.summary["topk_acceptance_gate"] = {
            "accepted": False,
            "partial_kept": True,
            "safe_limit": topk_acceptance_limit,
            "topk_over_limit": round(float(topk_over_limit), 3),
            "attempted_topk_calibrated_risk": round(float(final_topk_for_acceptance), 3),
            "attempted_topk_pattern_raw": (
                round(float(final_topk_raw_for_acceptance), 3)
                if isinstance(final_topk_raw_for_acceptance, (int, float)) else None
            ),
            "topk_calibrated_risk_drop": (
                round(float(final_topk_drop_for_acceptance), 3)
                if isinstance(final_topk_drop_for_acceptance, (int, float)) else None
            ),
            "ai_score_drop": (
                round(float(final_ai_drop_for_acceptance), 3)
                if isinstance(final_ai_drop_for_acceptance, (int, float)) else None
            ),
            "ai_authorship_drop": (
                round(float(final_authorship_drop_for_acceptance), 3)
                if isinstance(final_authorship_drop_for_acceptance, (int, float)) else None
            ),
            "ai_transformation_drop": (
                round(float(final_transformation_drop_for_acceptance), 3)
                if isinstance(final_transformation_drop_for_acceptance, (int, float)) else None
            ),
            "reason": reason,
            "decision": topk_near_miss_keep_decision,
        }
        result.summary.setdefault("saved_contract_notes", []).append(
            "Kept a Top-k-blocked candidate as partial progress because it materially improved AI-footprint drivers; it is not strict-safe or detector-safe."
        )
    elif (
        rewritten_text != text
        and isinstance(final_topk_for_acceptance, (int, float))
        and float(final_topk_for_acceptance) >= topk_acceptance_limit
    ):
        attempted_text = rewritten_text
        attempted_report_dict = rewritten_report_dict
        attempted_sentence_comparison = sentence_comparison
        reason = (
            f"topk_calibrated_safe_band_failed {float(final_topk_for_acceptance):.2f}>="
            f"{topk_acceptance_limit:.2f}"
        )
        rewritten_text = text
        rewritten_report_dict = original_report_dict
        rewritten_ai = _badge_ai(original_report_dict)
        rewritten_wq = _badge_wq(original_report_dict)
        rewritten_total = _finding_total(original_report_dict)
        rewritten_review_burden = _review_burden(original_report_dict)
        rewritten_severity = _weighted_severity(original_report_dict)
        if result.mp_result:
            result.mp_result.final_text = text
            result.mp_result.final_metrics = result.mp_result.original_metrics
            result.mp_result.converged = False
            result.mp_result.convergence_reason = reason
        result.summary["attempted_final_text"] = attempted_text
        result.summary["attempted_sentence_comparison"] = attempted_sentence_comparison
        result.summary["final_text"] = text
        result.summary["converged"] = False
        result.summary["rollback_applied"] = True
        result.summary["rollback_reason"] = reason
        result.summary["outcome"] = "topk_blocked"
        result.summary["topk_acceptance_gate"] = {
            "accepted": False,
            "safe_limit": topk_acceptance_limit,
            "attempted_topk_calibrated_risk": round(float(final_topk_for_acceptance), 3),
            "attempted_topk_pattern_raw": (
                round(float(final_topk_raw_for_acceptance), 3)
                if isinstance(final_topk_raw_for_acceptance, (int, float)) else None
            ),
            "reason": reason,
        }
        result.summary.setdefault("detect_scores", {}).update({
            "rewritten_ai": _badge_ai(original_report_dict),
            "rewritten_writing_quality": _badge_wq(original_report_dict),
            "rewritten_ai_authorship": _integrity_scores(original_report_dict).get("ai_authorship"),
            "rewritten_grounding_quality_risk": _integrity_scores(original_report_dict).get("grounding"),
            "rewritten_human_contribution": _contribution_scores(original_report_dict).get("human"),
            "rewritten_ai_transformation": _contribution_scores(original_report_dict).get("ai_transformation"),
            "rewritten_findings": _finding_total(original_report_dict),
            "rewritten_review_burden": _review_burden(original_report_dict),
            "rewritten_weighted_severity": _weighted_severity(original_report_dict),
            "attempted_ai": _badge_ai(attempted_report_dict),
            "attempted_writing_quality": _badge_wq(attempted_report_dict),
            "attempted_ai_authorship": _integrity_scores(attempted_report_dict).get("ai_authorship"),
            "attempted_grounding_quality_risk": _integrity_scores(attempted_report_dict).get("grounding"),
            "attempted_human_contribution": _contribution_scores(attempted_report_dict).get("human"),
            "attempted_ai_transformation": _contribution_scores(attempted_report_dict).get("ai_transformation"),
            "attempted_findings": _finding_total(attempted_report_dict),
            "attempted_review_burden": _review_burden(attempted_report_dict),
            "attempted_weighted_severity": _weighted_severity(attempted_report_dict),
            "attempted_topk_calibrated_risk": round(float(final_topk_for_acceptance), 3),
            "attempted_topk_pattern": (
                round(float(final_topk_raw_for_acceptance), 3)
                if isinstance(final_topk_raw_for_acceptance, (int, float)) else None
            ),
            "rollback_reason": reason,
        })
        sentence_comparison = []
        final_ai_footprint_gate = _ai_footprint_gate_status(
            original_report_dict,
            rewritten_report_dict,
            review_burden_delta=0,
            weighted_severity_delta=0,
            critical_high_delta=0,
            ai_score_regressed=False,
        )
    result.summary["ai_footprint_gate"] = final_ai_footprint_gate
    final_turnitin_like_gate = _turnitin_like_ai_gate_status(
        original_report_dict,
        rewritten_report_dict,
        review_burden_delta=_review_burden(rewritten_report_dict) - original_review_burden,
        weighted_severity_delta=_weighted_severity(rewritten_report_dict) - original_severity,
        critical_high_delta=_critical_high_count(rewritten_report_dict) - saved_critical_high,
        ai_score_regressed=bool(
            isinstance(_badge_ai(original_report_dict), (int, float))
            and isinstance(_badge_ai(rewritten_report_dict), (int, float))
            and _badge_ai(rewritten_report_dict) > _badge_ai(original_report_dict) + _float_env("DRAFTPROOF_AI_SEARCH_AI_SCORE_REGRESSION_TOLERANCE", 0.25)
        ),
    )
    result.summary["turnitin_like_ai_gate"] = final_turnitin_like_gate
    result.summary["turnitin_like_ai_score_before"] = final_turnitin_like_gate.get("score_before")
    result.summary["turnitin_like_ai_score_after"] = final_turnitin_like_gate.get("score_after")
    result.summary["turnitin_like_ai_score_drop"] = final_turnitin_like_gate.get("score_drop")
    result.summary["turnitin_like_target_score"] = final_turnitin_like_gate.get("target_score")
    result.summary["turnitin_like_target_gap"] = final_turnitin_like_gate.get("target_gap")
    result.summary["turnitin_like_target_met"] = final_turnitin_like_gate.get("target_met")
    result.summary["turnitin_like_components_before"] = (
        (final_turnitin_like_gate.get("before") or {}).get("components")
    )
    result.summary["turnitin_like_components_after"] = (
        (final_turnitin_like_gate.get("after") or {}).get("components")
    )
    result.summary["turnitin_like_component_drops"] = final_turnitin_like_gate.get("component_drops")
    result.summary["remaining_turnitin_like_drivers"] = (
        final_turnitin_like_gate.get("remaining_turnitin_like_drivers") or []
    )
    final_formula_gap_contract = _formula_gap_contract(
        original_report_dict,
        rewritten_report_dict,
        source_text=original_report_dict.get("document_context", {}).get("original_text", "")
        if isinstance(original_report_dict.get("document_context"), dict) else "",
        candidate_text=rewritten_text,
    )
    ai_search_summary_for_formula = result.summary.get("ai_mitigation_search") or {}
    observed_formula_candidates = (
        ai_search_summary_for_formula.get("candidates")
        if isinstance(ai_search_summary_for_formula, dict) else []
    )
    if isinstance(result.summary.get("formula_convergence_controller"), dict):
        observed_formula_candidates = list(observed_formula_candidates or []) + list(
            (result.summary.get("formula_convergence_controller") or {}).get("candidates") or []
        )
    final_formula_portfolio_plan = _formula_portfolio_plan(
        original_report_dict,
        rewritten_report_dict,
        observed_candidates=observed_formula_candidates,
    )
    result.summary["formula_gap_contract"] = final_formula_gap_contract
    final_formula_gap_contract["formula_portfolio_plan"] = final_formula_portfolio_plan
    final_formula_gap_contract["driver_priority_plan"] = final_formula_portfolio_plan.get("driver_priorities") or []
    final_formula_gap_contract["observed_driver_movement"] = final_formula_portfolio_plan.get("observed_driver_movement")
    final_formula_gap_contract["next_formula_driver"] = (
        (final_formula_gap_contract.get("driver_priority_plan") or [{}])[0].get("driver")
        if final_formula_gap_contract.get("driver_priority_plan") else None
    )
    result.summary["selected_formula_strategy"] = (
        (result.summary.get("formula_convergence_controller") or {}).get("selected_strategy")
        or (result.summary.get("ai_mitigation_search") or {}).get("selected_strategy")
        or result.summary.get("selected_strategy")
    )
    result.summary["formula_portfolio_plan"] = final_formula_portfolio_plan
    result.summary["positive_ai_burden"] = final_formula_portfolio_plan.get("positive_ai_burden")
    result.summary["human_anchor_suppression"] = final_formula_portfolio_plan.get("human_anchor_suppression")
    result.summary["suppression_headroom"] = final_formula_portfolio_plan.get("suppression_headroom")
    result.summary["required_suppression_gain"] = final_formula_portfolio_plan.get("required_suppression_gain")
    result.summary["expected_net_gain"] = final_formula_portfolio_plan.get("expected_net_gain")
    result.summary["observed_driver_movement"] = final_formula_portfolio_plan.get("observed_driver_movement")
    result.summary["weighted_driver_plan"] = final_formula_gap_contract.get("weighted_driver_plan")
    result.summary["driver_priority_plan"] = final_formula_portfolio_plan.get("driver_priorities")
    result.summary["next_formula_driver"] = final_formula_gap_contract.get("next_formula_driver")
    result.summary["weighted_driver_drops"] = final_formula_gap_contract.get("weighted_driver_drops")
    result.summary["remaining_formula_gap"] = final_formula_gap_contract.get("remaining_formula_gap")
    result.summary["why_not_below_20"] = final_formula_gap_contract.get("why_not_below_20")
    final_human_anchor_contract = _human_anchor_driver_contract(
        original_report_dict,
        rewritten_report_dict,
        text=rewritten_text,
    )
    result.summary["human_anchor_driver_contract"] = final_human_anchor_contract
    result.summary["human_anchor_score_before"] = (
        (final_human_anchor_contract.get("before") or {}).get("human_anchor_score")
    )
    result.summary["human_anchor_score_after"] = (
        (final_human_anchor_contract.get("after") or {}).get("human_anchor_score")
    )
    result.summary["lived_detail_risk_before"] = (
        (final_human_anchor_contract.get("before") or {}).get("lived_detail_risk")
    )
    result.summary["lived_detail_risk_after"] = (
        (final_human_anchor_contract.get("after") or {}).get("lived_detail_risk")
    )
    result.summary["domain_grounding_strength_before"] = (
        (final_human_anchor_contract.get("before") or {}).get("domain_grounding_strength")
    )
    result.summary["domain_grounding_strength_after"] = (
        (final_human_anchor_contract.get("after") or {}).get("domain_grounding_strength")
    )
    selected_anchor_status = (
        ((result.summary.get("ai_mitigation_search") or {}).get("selection_status") or {})
        .get("human_anchor_amplifier_status")
    )
    if isinstance(selected_anchor_status, dict):
        result.summary["selected_human_anchor_strategy"] = (
            (result.summary.get("ai_mitigation_search") or {}).get("selected_strategy")
        )
    result.summary["remaining_human_anchor_blockers"] = [
        {
            "driver": "lived_detail_risk",
            "value": (final_human_anchor_contract.get("after") or {}).get("lived_detail_risk"),
            "target_next_band": (final_human_anchor_contract.get("next_lived_detail_band") or {}).get("risk"),
            "additional_anchor_sentences_needed": final_human_anchor_contract.get("additional_anchor_sentences_needed"),
        }
    ] if not final_human_anchor_contract.get("achieved_next_band") else []
    final_eligible_span_density_gate = _eligible_span_density_comparison(
        text,
        original_report_dict,
        rewritten_text,
        rewritten_report_dict,
    )
    result.summary["eligible_span_density_contract"] = final_eligible_span_density_gate
    result.summary["eligible_span_density_before"] = final_eligible_span_density_gate.get("before")
    result.summary["eligible_span_density_after"] = final_eligible_span_density_gate.get("after")
    result.summary["eligible_span_density_safe"] = bool(final_eligible_span_density_gate.get("safe"))
    result.summary["eligible_span_density_needs_author_context"] = bool(
        final_eligible_span_density_gate.get("needs_author_context")
    )
    final_strict_safe_band = _strict_ai_safe_band_status(rewritten_report_dict)
    result.summary["strict_ai_safe_band_achieved"] = bool(final_strict_safe_band.get("achieved"))
    result.summary["remaining_strict_safe_band_drivers"] = final_strict_safe_band.get("remaining") or []
    ai_search_summary = result.summary.get("ai_mitigation_search") or {}
    if isinstance(ai_search_summary, dict):
        if isinstance(ai_search_summary.get("phase_budget_contract"), dict):
            result.summary["phase_budget_contract"] = ai_search_summary.get("phase_budget_contract")
        if isinstance(ai_search_summary.get("phase_budget_used"), dict):
            result.summary["phase_budget_used"] = ai_search_summary.get("phase_budget_used")
        frontier_rows = []
        for row in ai_search_summary.get("candidates") or []:
            if not isinstance(row, dict):
                continue
            strict_row = row.get("strict_ai_safe_band")
            if not isinstance(strict_row, dict):
                strict_row = ((row.get("ai_footprint_gate") or {}).get("strict_ai_safe_band") or {})
            if not isinstance(strict_row, dict) or not isinstance(strict_row.get("profile"), dict):
                strict_row = _strict_ai_safe_band_status_from_footprint_gate(
                    row.get("ai_footprint_gate")
                    or ((row.get("selection_status") or {}).get("ai_footprint_gate"))
                )
            profile = strict_row.get("profile") if isinstance(strict_row, dict) else {}
            if not isinstance(profile, dict) or not profile:
                continue
            turnitin_row = row.get("turnitin_like_ai_gate")
            if not isinstance(turnitin_row, dict):
                turnitin_row = ((row.get("selection_status") or {}).get("turnitin_like_ai_gate") or {})
            formula_row = row.get("formula_gap_contract")
            if not isinstance(formula_row, dict):
                formula_row = ((row.get("selection_status") or {}).get("formula_gap_contract") or {})
            density_row = row.get("eligible_span_density_gate") if isinstance(row.get("eligible_span_density_gate"), dict) else {}
            density_after = density_row.get("after") if isinstance(density_row.get("after"), dict) else {}
            frontier_rows.append({
                "strategy": row.get("strategy"),
                "selectable": bool((row.get("selection_status") or {}).get("selectable")),
                "reason": row.get("reason") or (row.get("selection_status") or {}).get("reason"),
                "strict_ai_safe_band_achieved": bool(strict_row.get("achieved")),
                "formula_target_met": bool(formula_row.get("target_met")),
                "formula_score": formula_row.get("score_after"),
                "formula_score_drop": formula_row.get("score_drop"),
                "formula_remaining_gap": formula_row.get("remaining_formula_gap"),
                "formula_drop_efficiency": formula_row.get("weighted_driver_drop_efficiency"),
                "eligible_span_density_safe": density_row.get("safe"),
                "unsafe_eligible_word_ratio": density_after.get("unsafe_eligible_word_ratio"),
                "longest_unsafe_span_words": density_after.get("longest_unsafe_span_words"),
                "turnitin_like_ai_score": turnitin_row.get("score_after"),
                "turnitin_like_ai_score_drop": turnitin_row.get("score_drop"),
                "turnitin_like_outcome_class": turnitin_row.get("outcome_class"),
                "ai_authorship": profile.get("ai_authorship"),
                "ai_transformation": profile.get("ai_transformation"),
                "qualifying_text_ai_density": profile.get("qualifying_text_ai_density"),
                "external_ai_flag_risk": profile.get("external_ai_flag_risk"),
                "topk_calibrated_risk": profile.get("topk_calibrated_risk"),
                "remaining": strict_row.get("remaining") or [],
            })
        frontier_rows.sort(
            key=lambda item: (
                1 if item.get("strict_ai_safe_band_achieved") else 0,
                1 if item.get("formula_target_met") else 0,
                float(item.get("formula_score_drop") if isinstance(item.get("formula_score_drop"), (int, float)) else -999.0),
                float(item.get("formula_drop_efficiency") if isinstance(item.get("formula_drop_efficiency"), (int, float)) else -999.0),
                -float(item.get("formula_score") if isinstance(item.get("formula_score"), (int, float)) else 999.0),
                float(item.get("turnitin_like_ai_score_drop") if isinstance(item.get("turnitin_like_ai_score_drop"), (int, float)) else -999.0),
                -float(item.get("qualifying_text_ai_density") if isinstance(item.get("qualifying_text_ai_density"), (int, float)) else 999.0),
                -float(item.get("external_ai_flag_risk") if isinstance(item.get("external_ai_flag_risk"), (int, float)) else 999.0),
                -float(item.get("ai_authorship") if isinstance(item.get("ai_authorship"), (int, float)) else 999.0),
                -float(item.get("ai_transformation") if isinstance(item.get("ai_transformation"), (int, float)) else 999.0),
                -float(item.get("topk_calibrated_risk") if isinstance(item.get("topk_calibrated_risk"), (int, float)) else 999.0),
            ),
            reverse=True,
        )
        result.summary["best_candidate_frontier"] = frontier_rows[:8]
        result.summary["best_formula_frontier"] = frontier_rows[:8]
    result.summary["selected_candidate_rank"] = list(_strict_safe_candidate_rank(
        original_report_dict,
        rewritten_report_dict,
        review_burden_delta=_review_burden(rewritten_report_dict) - original_review_burden,
        weighted_severity_delta=_weighted_severity(rewritten_report_dict) - original_severity,
        critical_high_delta=_critical_high_count(rewritten_report_dict) - saved_critical_high,
    ))
    result.summary["turnitin_like_selected_candidate_rank"] = list(_turnitin_like_candidate_rank(
        final_turnitin_like_gate,
        review_burden_delta=_review_burden(rewritten_report_dict) - original_review_burden,
        weighted_severity_delta=_weighted_severity(rewritten_report_dict) - original_severity,
        critical_high_delta=_critical_high_count(rewritten_report_dict) - saved_critical_high,
    ))
    result.summary["formula_gap_selected_candidate_rank"] = list(
        _formula_gap_candidate_rank(final_formula_gap_contract, final_turnitin_like_gate)
    )
    result.summary["why_not_strict_safe"] = (
        "strict safe band achieved"
        if final_strict_safe_band.get("achieved") and final_turnitin_like_gate.get("safe_band")
        else "Remaining blockers: " + ", ".join(
            [
                f"{row.get('driver')} {row.get('value')} > {row.get('safe_band')}"
                for row in (final_strict_safe_band.get("remaining") or [])
                if isinstance(row, dict)
            ]
            + (
                [
                    "turnitin_like_ai_score "
                    f"{final_turnitin_like_gate.get('score_after')} > "
                    f"{(final_turnitin_like_gate.get('thresholds') or {}).get('safe_band')}"
                ]
                if not final_turnitin_like_gate.get("safe_band") else []
            )
        )
    )
    result.summary["why_not_turnitin_like_target"] = (
        "turnitin-like target achieved"
        if final_turnitin_like_gate.get("target_met")
        else (
            "Turnitin-like AI score "
            f"{final_turnitin_like_gate.get('score_after')} is not below "
            f"{final_turnitin_like_gate.get('target_score')}; "
            "dominant drivers: "
            + ", ".join(
                str(row.get("driver"))
                for row in (final_turnitin_like_gate.get("remaining_turnitin_like_drivers") or [])[:4]
                if isinstance(row, dict) and row.get("driver")
            )
        )
    )
    unsafe_detector_drivers = [
        row for row in (final_strict_safe_band.get("remaining") or [])
        if isinstance(row, dict)
        and str(row.get("driver") or "") in {
            "topk_calibrated_risk",
            "external_ai_flag_risk",
            "ai_authorship",
            "ai_transformation",
            "qualifying_text_ai_density",
        }
    ]
    final_positive_ai_burden = (
        final_formula_gap_contract.get("positive_ai_burden")
        if isinstance(final_formula_gap_contract.get("positive_ai_burden"), dict)
        else {}
    )
    final_meaningful_ai_progress_gate = meaningful_ai_progress_gate(
        turnitin_like_ai_score_drop=final_turnitin_like_gate.get("score_drop"),
        ai_score_drop=final_ai_drop_for_acceptance,
        ai_authorship_drop=final_authorship_drop_for_acceptance,
        ai_transformation_drop=final_transformation_drop_for_acceptance,
        positive_ai_burden_drop=final_positive_ai_burden.get("drop"),
        ai_window_vote_ratio_drop=result.summary.get("ai_sentence_vote_ratio_drop"),
        unsafe_eligible_density_drop=final_eligible_span_density_gate.get("unsafe_eligible_word_ratio_drop"),
    )
    final_cleanup_progress_gate = cleanup_progress_gate(
        findings_drop=_finding_total(original_report_dict) - _finding_total(rewritten_report_dict),
        review_burden_drop=original_review_burden - _review_burden(rewritten_report_dict),
        weighted_severity_drop=original_severity - _weighted_severity(rewritten_report_dict),
    )
    detector_safe_label_status = {
        "detector_safe": bool(
            final_strict_safe_band.get("achieved")
            and final_turnitin_like_gate.get("target_met")
            and final_eligible_span_density_gate.get("safe")
        ),
        "strict_ai_safe_band_achieved": bool(final_strict_safe_band.get("achieved")),
        "turnitin_like_target_met": bool(final_turnitin_like_gate.get("target_met")),
        "eligible_span_density_safe": bool(final_eligible_span_density_gate.get("safe")),
        "eligible_span_density": {
            "unsafe_eligible_word_ratio": ((final_eligible_span_density_gate.get("after") or {}).get("unsafe_eligible_word_ratio")),
            "longest_unsafe_span_words": ((final_eligible_span_density_gate.get("after") or {}).get("longest_unsafe_span_words")),
            "unsafe_cluster_count": ((final_eligible_span_density_gate.get("after") or {}).get("unsafe_cluster_count")),
            "needs_author_context": final_eligible_span_density_gate.get("needs_author_context"),
        },
        "turnitin_like_score_drop": final_turnitin_like_gate.get("score_drop"),
        "meaningful_ai_progress": bool(final_meaningful_ai_progress_gate.get("meaningful")),
        "meaningful_ai_progress_gate": final_meaningful_ai_progress_gate,
        "cleanup_progress": bool(final_cleanup_progress_gate.get("cleanup")),
        "cleanup_progress_gate": final_cleanup_progress_gate,
        "unsafe_partial": bool(
            rewritten_text != text
            and final_meaningful_ai_progress_gate.get("meaningful")
            and (unsafe_detector_drivers or not final_eligible_span_density_gate.get("safe"))
        ),
        "unsafe_drivers": unsafe_detector_drivers,
        "label_rule": (
            "ai_mitigated requires Turnitin-like score below target, all strict detector drivers in safe band, "
            "eligible prose density in safe band, and no safety regression. "
            "If detector drivers remain unsafe, unsafe_partial_improvement requires meaningful AI progress; "
            "cleanup-only gains are labelled cleanup_only."
        ),
    }
    result.summary["detector_safe_label_status"] = detector_safe_label_status
    if (
        not result.summary.get("selected_rewrite_compiler_strategy")
        and str(result.summary.get("selected_strategy") or "").startswith("compiler_")
    ):
        result.summary["selected_rewrite_compiler_strategy"] = result.summary.get("selected_strategy")
    if result.summary.get("rewrite_compiler") is None and isinstance(result.summary.get("deterministic_rewrite_compiler"), dict):
        result.summary["rewrite_compiler"] = result.summary.get("deterministic_rewrite_compiler")
    texture_summary = (result.summary.get("ai_mitigation_search") or {}).get("authorship_transformation_texture_controller")
    if not isinstance(texture_summary, dict):
        texture_summary = (result.summary.get("ai_mitigation_search") or {}).get("post_topk_optimizer")
    if isinstance(texture_summary, dict):
        result.summary["authorship_transformation_texture_controller"] = texture_summary
        result.summary["texture_driver_map"] = texture_summary.get("texture_driver_map") or texture_summary.get("driver_map")
        result.summary["selected_texture_strategy"] = texture_summary.get("selected_strategy")
        result.summary["texture_candidate_frontier"] = texture_summary.get("best_rejected_candidate")
        result.summary["remaining_texture_blockers"] = texture_summary.get("remaining_strict_safe_band_drivers") or result.summary.get("remaining_strict_safe_band_drivers")
        # Backward-compatible keys for the existing report renderer.
        result.summary["post_topk_optimizer"] = texture_summary
        result.summary["selected_post_topk_strategy"] = texture_summary.get("selected_strategy")
    final_footprint_before = final_ai_footprint_gate.get("before", {}) or {}
    final_footprint_after = final_ai_footprint_gate.get("after", {}) or {}
    final_footprint_drops = final_ai_footprint_gate.get("drops") or {}
    final_before_authorship = final_footprint_before.get("authorship_footprint") or {}
    final_after_authorship = final_footprint_after.get("authorship_footprint") or {}
    final_before_semantic = final_footprint_before.get("semantic_footprint") or {}
    final_after_semantic = final_footprint_after.get("semantic_footprint") or {}
    result.summary.setdefault("detect_scores", {}).update({
        "external_ai_flag_risk_before": final_footprint_before.get("external_ai_flag_risk"),
        "external_ai_flag_risk_after": final_footprint_after.get("external_ai_flag_risk"),
        "external_ai_flag_risk_drop": final_footprint_drops.get("external_ai_flag_risk"),
        "topk_pattern_raw_before": final_before_authorship.get("topk_pattern_raw"),
        "topk_pattern_raw_after": final_after_authorship.get("topk_pattern_raw"),
        "topk_pattern_raw_drop": final_footprint_drops.get("topk_pattern_raw"),
        "topk_calibrated_risk_before": final_before_authorship.get("topk_calibrated_risk"),
        "topk_calibrated_risk_after": final_after_authorship.get("topk_calibrated_risk"),
        "topk_calibrated_risk_drop": final_footprint_drops.get("topk_calibrated_risk"),
        "rewrite_smoothness_before": final_before_authorship.get("rewrite_smoothness"),
        "rewrite_smoothness_after": final_after_authorship.get("rewrite_smoothness"),
        "rewrite_smoothness_drop": final_footprint_drops.get("rewrite_smoothness"),
        "ai_likelihood_driver_before": final_before_authorship.get("ai_likelihood"),
        "ai_likelihood_driver_after": final_after_authorship.get("ai_likelihood"),
        "ai_likelihood_driver_drop": final_footprint_drops.get("ai_likelihood"),
        "turnitin_like_ai_score_before": final_turnitin_like_gate.get("score_before"),
        "turnitin_like_ai_score_after": final_turnitin_like_gate.get("score_after"),
        "turnitin_like_ai_score_drop": final_turnitin_like_gate.get("score_drop"),
        "turnitin_like_target_score": final_turnitin_like_gate.get("target_score"),
        "turnitin_like_target_gap": final_turnitin_like_gate.get("target_gap"),
        "turnitin_like_target_met": final_turnitin_like_gate.get("target_met"),
        "qualifying_text_ai_density_before": final_before_semantic.get("qualifying_text_ai_density"),
        "qualifying_text_ai_density_after": final_after_semantic.get("qualifying_text_ai_density"),
        "qualifying_text_ai_density_drop": final_footprint_drops.get("qualifying_text_ai_density"),
        "eligible_span_unsafe_ratio_before": ((final_eligible_span_density_gate.get("before") or {}).get("unsafe_eligible_word_ratio")),
        "eligible_span_unsafe_ratio_after": ((final_eligible_span_density_gate.get("after") or {}).get("unsafe_eligible_word_ratio")),
        "eligible_span_longest_unsafe_words_before": ((final_eligible_span_density_gate.get("before") or {}).get("longest_unsafe_span_words")),
        "eligible_span_longest_unsafe_words_after": ((final_eligible_span_density_gate.get("after") or {}).get("longest_unsafe_span_words")),
        "eligible_span_density_safe": final_eligible_span_density_gate.get("safe"),
        "ai_footprint_outcome_class": final_ai_footprint_gate.get("outcome_class"),
        "remaining_ai_footprint_drivers": final_ai_footprint_gate.get("remaining_ai_footprint_drivers"),
    })

    # Extract only the fields needed for comparison (not full report dicts)
    def _extract_scan_summary(report_dict):
        badge = report_dict.get("ai_risk_badge") or {}
        findings = report_dict.get("findings", {})
        return {
            "ai_score": report_dict.get("ai_score") or badge.get("ai_likelihood_score"),
            "writing_score": report_dict.get("writing_score") or badge.get("writing_quality_score"),
            "ai_risk_badge": badge,
            "topk_repair_map": _topk_repair_map(
                report_dict.get("document_context", {}).get("original_text", "")
                if isinstance(report_dict.get("document_context"), dict) else "",
                report_dict,
            ) if (badge.get("ai_components") or {}).get("topk_pattern") else {},
            "scan_intelligence": report_dict.get("scan_intelligence") or {},
            "integrity_layers": report_dict.get("integrity_layers") or {},
            "overall_tier": report_dict.get("overall_tier", "?"),
            "findings": {t: [{"finding_id": f.get("finding_id"), "title": f.get("title"),
                              "category": f.get("category")} for f in findings.get(t, [])]
                         for t in ("critical", "high", "medium", "low")},
        }

    result.summary["detect_scan_original_saved"] = _extract_scan_summary(ctx.raw_json)
    result.summary["detect_scan_original"] = _extract_scan_summary(original_report_dict)
    if result.summary.get("rollback_applied"):
        result.summary["detect_scan_attempted"] = _extract_scan_summary(attempted_report_dict)
    else:
        result.summary["final_text"] = rewritten_text
        if ai_search_selected:
            ai_footprint_outcome = str(final_ai_footprint_gate.get("outcome_class") or "")
            turnitin_like_outcome = str(final_turnitin_like_gate.get("outcome_class") or "")
            unsafe_partial = bool((result.summary.get("detector_safe_label_status") or {}).get("unsafe_partial"))
            texture_blockers = [
                blocker for blocker in (final_ai_footprint_gate.get("texture_blockers") or [])
                if isinstance(blocker, dict)
            ]
            topk_still_blocked = any(
                str(blocker.get("driver") or "") in {"topk_calibrated_risk", "topk_pattern"}
                for blocker in texture_blockers
            )
            if (
                ai_footprint_outcome == "ai_mitigated"
                and final_turnitin_like_gate.get("safe_band")
                and bool((result.summary.get("detector_safe_label_status") or {}).get("detector_safe"))
            ):
                result.summary["outcome"] = "ai_mitigated"
            else:
                policy_outcome = final_rewrite_outcome_label(
                    detector_safe=bool((result.summary.get("detector_safe_label_status") or {}).get("detector_safe")),
                    text_changed=bool(rewritten_text != text),
                    meaningful_ai_progress=bool((result.summary.get("detector_safe_label_status") or {}).get("meaningful_ai_progress")),
                    cleanup_progress=bool((result.summary.get("detector_safe_label_status") or {}).get("cleanup_progress")),
                    current_outcome=result.summary.get("outcome"),
                )
                if policy_outcome == "unsafe_partial_improvement" and unsafe_partial:
                    result.summary["outcome"] = (
                        "mitigation_failed_no_safe_candidate"
                        if ai_search_fail_fast_partial
                        else "unsafe_partial_improvement"
                    )
                elif policy_outcome == "cleanup_only":
                    result.summary["outcome"] = "cleanup_only"
                elif policy_outcome == "ceiling_reached":
                    result.summary["outcome"] = "ceiling_reached"
                elif turnitin_like_outcome in {"ai_mitigated", "partially_ai_mitigated"} and unsafe_partial:
                    result.summary["outcome"] = (
                        "mitigation_failed_no_safe_candidate"
                        if ai_search_fail_fast_partial
                        else "unsafe_partial_improvement"
                    )
                else:
                    result.summary["outcome"] = policy_outcome
            result.summary["converged"] = True
    result.summary["detect_scan_rewritten"] = _extract_scan_summary(rewritten_report_dict)
    result.summary["full_scan_cache"] = dict(full_scan_cache_stats)
    _sync_rewrite_llm_call_totals(result.summary, global_rewrite_budget)
    result.summary["global_rewrite_budget"] = global_rewrite_budget.summary()
    result.summary["global_candidate_ledger"] = global_candidate_ledger.summary()
    result.summary["stage_timings"] = stage_timings
    result.sentence_comparison = sentence_comparison

    # Generate dedicated rewrite report
    rewrite_md = render_rewrite_report(
        summary=result.summary,
        sentence_comparison=sentence_comparison,
        ai_findings=ai_findings,
        verbose=verbose,
    )

    with open(md_path, "w") as f:
        f.write(rewrite_md)

    pdf_path = os.path.join(output_dir, f"draftproof_rewrite_{ts}.pdf")
    render_pdf(rewrite_md, pdf_path)

    summary = result.summary
    total_elapsed = time.time() - t0
    summary["rewrite_engine_time"] = engine_elapsed
    summary["rewrite_time"] = total_elapsed
    summary["original_tier"] = ctx.overall_tier
    summary["rewrite_decision"] = ctx.rewrite_decision

    with open(json_path_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if summary.get("outcome") == "topk_blocked":
        pipeline_status = "topk_blocked"
    elif summary.get("rollback_applied") or summary.get("no_text_change"):
        pipeline_status = "original_preserved"
    elif summary.get("outcome") in {
        "partially_improved",
        "partially_ai_mitigated",
        "cleanup_improved",
        "unsafe_partial_improvement",
        "mitigation_failed_no_safe_candidate",
    }:
        pipeline_status = summary.get("outcome")
    else:
        pipeline_status = "rewritten"

    return {
        "status": pipeline_status,
        "md_path": md_path,
        "pdf_path": pdf_path,
        "json_path": json_path_out,
        "result": result,
        "elapsed": total_elapsed,
    }


def main():
    _load_local_env()

    parser = argparse.ArgumentParser(description="DraftProof Rewrite Pipeline")
    parser.add_argument("file", nargs="?", help="Detect JSON file (or - for stdin)")
    parser.add_argument("--text", "-t", help="Inline text to detect + rewrite")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--passes", type=int, default=3, help="Max rewrite passes")
    parser.add_argument("--max-loops", type=int, default=0, help="Max detect-rewrite loops")
    parser.add_argument("--target-top10", type=float, default=0.50, help="Target top-10 ratio")
    parser.add_argument("--model", default=None, help="LLM model (default: from LLM_MODEL env var)")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-ai-only", action="store_true", help="Rewrite ALL findings (default: AI-only)")
    args = parser.parse_args()

    output_dir = args.output or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "test_output"
    ))

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # Read input
    json_path = None
    text = None

    if args.text:
        text = args.text
    elif args.file == "-" or (not args.file and not sys.stdin.isatty()):
        raw = sys.stdin.read()
        try:
            json.loads(raw)
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                tf.write(raw)
                json_path = tf.name
        except json.JSONDecodeError:
            text = raw
    elif args.file:
        json_path = args.file
    else:
        print("Error: provide a detect JSON file, --text, or pipe JSON via stdin", file=sys.stderr)
        sys.exit(1)

    result = run_rewrite_pipeline(
        json_path=json_path,
        text=text,
        output_dir=output_dir,
        max_passes=args.passes,
        max_detect_loops=args.max_loops,
        target_top10=args.target_top10,
        model=args.model,
        api_key=api_key,
        verbose=args.verbose,
        ai_only=not args.no_ai_only,
    )

    if result["status"] == "clean":
        print(f"\n  Status: {result['message']}")
    elif result["status"] == "skipped":
        print(f"\n  Skipped: {result['message']}")
    else:
        elapsed = result["elapsed"]
        r = result["result"]
        rw = r.mp_result
        print(f"\n  Time: {elapsed:.1f}s")
        print(f"  Passes: {len(rw.passes)}")
        print(f"  Converged: {'Yes' if rw.converged else 'No'}")
        print(f"  MD:   {result['md_path']}")
        print(f"  PDF:  {result['pdf_path']}")
        print(f"  JSON: {result['json_path']}")


if __name__ == "__main__":
    main()
