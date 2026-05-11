"""Test the re-architected rewrite module.

Tests fixability routing, voice guard, transactional apply,
floor detection, outcome classification, and report integration.

Run:  cd poc && python test_rewrite_v2.py
"""

import sys
import os
import re
import json
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from detect.base import Finding, DetectResult
from rewrite import (
    RewriteConfig, RewriteBudget, RewriteOutcome, RewriteSurface, FloorReason,
    RewritePlanner, RewritePlan, RewriteAction, FixabilityDecision,
    VoiceGuard, VoiceProfile, analyze_voice,
    weighted_finding_score, weighted_rewritable_risk, CandidateScore,
    score_candidate, best_candidate,
    transactional_apply, DriftCheck, RegressionMemory,
    detect_protected_spans, check_semantic_drift,
    compute_rewrite_surface, classify_floor,
    protected_spans_preserved, affected_region,
    FINDING_ROUTING, EDIT_RADIUS,
    FIXABILITY_AUTO, FIXABILITY_PARTIAL, FIXABILITY_MANUAL, FIXABILITY_PROTECTED,
)
from rewrite.voice import VoiceCheck
from rewrite.parse_detect import DetectJSONParser
from rewrite.parse_detect import DetectJSONContext
from rewrite.rewrite import (
    _candidate_task_instruction,
    _candidate_style_reject_reason,
    _plan_rewrite_operation,
    _paragraph_coherence_reject_reason,
    _rewrite_operation_lines,
    _target_predictability_acceptance,
    _select_density_paragraph,
    _density_paragraph_prompt,
    _density_paragraph_reject_reason,
    _density_entity_only_drift,
    _density_transformation_too_small,
    _density_local_signal_acceptance,
    _select_best_density_candidate,
    _clean_density_rescue_output,
    _density_rescue_prompt,
    _density_rescue_retry_prompt,
    _density_repair_prompt,
    _splice_density_candidate,
    run_rewrite,
)
from rewrite.mitigation import build_mitigation_plan
from rewrite.auto_repair_controller import (
    AutoRepairDependencies,
    compile_plan as auto_repair_compile_plan,
    candidate_pool as auto_repair_candidate_pool,
    run_auto_repair_controller,
)
from rewrite_compiler import CompilerConfig, CompilerDependencies, run_rewrite_compiler
from rewrite_compiler.evaluator import evaluate_quality as compiler_evaluate_quality
from rewrite_compiler.planner import build_plan as compiler_build_plan
from rewrite_compiler.selector import evaluate_scanned_candidate as compiler_evaluate_scanned_candidate
from rewrite_compiler.validator import validate_candidate as compiler_validate_candidate
from rewrite_controller import (
    CandidateLedger,
    build_candidate_decision,
    RewriteRunBudget,
    build_candidate_record,
    cap_phase_seconds_for_reserve,
    classify_ai_search_candidate,
    post_ai_search_reserve_seconds,
    resolve_global_rewrite_seconds,
    assemble_formula_gap_candidate,
    build_eligible_span_density_contract,
    build_segment_density_windows,
    compare_eligible_span_density,
    evaluate_text_quality_regression,
    extract_formula_gap_candidate_payload,
    extract_segment_window_payload,
    formula_gap_block_portfolio_tasks,
    formula_gap_budget_contract,
    formula_gap_candidate_prompt,
    formula_gap_named_entity_inventory,
    formula_gap_plan,
    formula_gap_portfolio_families,
    assemble_segment_window_candidate,
    segment_patchwork_budget,
    segment_window_candidate_prompt,
    segment_window_is_canonical_fact_sentence,
    segment_window_tasks,
)
from detect.topk_calibration import calibrate_topk_risk
from detect.turnitin_like import TURNITIN_LIKE_TARGET_AI_SCORE, turnitin_like_ai_profile
from rewrite_pipeline import (
    run_rewrite_pipeline,
    _build_aligned_sentence_comparison,
    _ai_first_gate_status,
    _ai_search_marked_grounding_candidates,
    _clear_stale_rollback_for_kept_ai_mitigation,
    _ai_search_fast_accept_reason,
    _ai_search_candidate_selection_status,
    _review_marker_notes,
    _ai_candidate_quality_reject_reason,
    _source_repair_brief,
    _ai_search_prompt,
    _ai_search_feedback_prompt,
    _safe_partial_quality_improvement_status,
    _ai_footprint_profile,
    _turnitin_like_ai_profile,
    _turnitin_like_ai_gate_status,
    _turnitin_like_candidate_rank,
    _formula_gap_contract,
    _formula_gap_candidate_rank,
    _formula_gap_changed_word_count,
    _candidate_concept_origin_reject_reason,
    _formula_convergence_primary_burden_gate_status,
    _formula_portfolio_plan,
    _multi_signal_candidate_contract,
    _ai_footprint_gate_status,
    _strict_ai_safe_band_status,
    _strict_ai_safe_band_status_from_footprint_gate,
    _topk_rebuild_fallback_rank,
    _topk_near_miss_partial_keep_decision,
    _selection_status_topk_safe,
    _strict_safe_phase_budget_contract,
    _strict_safe_candidate_rank,
    _safe_topk_limit,
    _ai_search_selected_by_final_safety_gate,
    _ai_search_final_selection_status,
    _detector_progress_rank,
    _allow_ai_search_llm_after_deterministic,
    _load_local_env,
    _repair_candidate_source_damage,
    _source_repair_drift_false_positive,
    _ai_search_protected_loss_reason,
    _ai_search_drift_false_positive,
    _ai_search_quote_drift_scan_allowed,
    _ai_search_entity_drift_scan_allowed,
    _document_recreate_drift_scan_allowed,
    _reconstruction_drift_scan_allowed,
    _scan_scope_summary,
    _human_shift_score,
    _goal_climb_candidate_rank,
    _authenticity_gate_status,
    _human_target_regression_selection_block,
    _human_formula_driver_status,
    _optimization_candidate_status,
    _select_best_optimization_candidate,
    _metric_repair_diagnosis,
    _human_gain_stage_target,
    _is_better_human_shift_candidate,
    _anchor_lock_mapping,
    _freeze_anchor_text,
    _restore_anchor_placeholders,
    _freeze_anchor_payload,
    _repair_aggression_score,
    _split_sentences,
    _text_word_count,
    _sentence_texture_risk_map,
    _ordered_concept_origin_terms,
    _micro_texture_window,
    _splice_sentence_window,
    _splice_sentence_for_auto_repair,
    _locality_score,
    _micro_texture_repair_prompt,
    _clean_micro_texture_candidate,
    _masked_span_repair_prompt,
    _clean_masked_span_replacement,
    _apply_masked_span_replacement,
    _micro_repair_gain_efficiency,
    _micro_texture_iteration_status,
    _iterative_micro_texture_repair,
    _generation_candidate_diagnostics,
    _build_reconstruction_meaning_brief,
    _build_regeneration_blueprint,
    _generation_context_ledger,
    _reconstruction_mitigation_prompt,
    _reference_entries_from_text,
    _staged_generation_section_plan,
    _staged_reconstruction_section_prompt,
    _llm_role_config,
    _retry_model_enabled,
    _paragraph_component_targets,
    _paragraph_component_prompt,
    _logical_paragraphs,
    _ai_search_prompt,
    _paragraph_role,
    _neutralize_external_detector_style_artifacts,
    _synthetic_meta_anchor_artifact_reason,
    _human_signal_amplification_prompt,
    _author_reasoning_amplification_prompt,
    _score_human_amplification_candidate,
    _content_pruning_candidates,
    _generic_assertion_compiler_candidates,
    _radar_blocker_option_matrix,
    _radar_goal_controller_status,
    _radar_goal_requires_human_progress,
    _blocker_operation_plan,
    _blocker_operation_candidates,
    _post_safe_win_target_push_candidates,
    _human_signal_construction_candidates,
    _human_anchor_amplifier_candidates,
    _human_anchor_suppression_frontier,
    _human_anchor_suppression_frontier_candidates,
    _formula_feasibility_estimator,
    _geometry_risk_map,
    _coordinated_micro_perturbation_candidates,
    _anti_smoothing_guard_status,
    _formula_portfolio_candidates,
    _formula_block_driver_map,
    _formula_convergence_controller,
    _formula_convergence_llm_patch_candidates,
    _ai_density_breaker_canonical_fact_sentence,
    _ai_density_breaker_sentence_route,
    _post_selection_ai_density_breaker_candidates,
    _post_selection_ai_density_breaker_acceptance,
    _post_density_human_anchor_probe_candidates,
    _post_density_human_anchor_probe_acceptance,
    _contribution_scores,
    _integrity_scores,
    _record_rewrite_llm_calls,
    _human_anchor_driver_contract,
    _human_anchor_positive_burden_gate_status,
    _author_stance_thread_candidates,
    _human_target_ai_search_status,
    _blocker_scores,
    _dominant_blocker_gate_status,
    _dominant_blocker_safe_progress_override,
    _ai_search_adaptive_stop_reason,
    _should_track_blocked_human_winner,
    _blocked_human_winner_repair_budget_override,
    _post_safe_target_push_allows_deterministic_after_budget,
    _post_safe_target_push_scan_reserve,
    _final_topk_texture_scan_reserve,
    _blocked_winner_bounded_quality_tradeoff,
    _score_drag_removal_status,
    _block_level_decisions,
    _adaptive_budget_default,
    _internet_reauthor_priority_status,
    _build_author_evidence_completion_layer,
    _build_mitigation_ceiling_diagnostics,
    _build_author_evidence_intake_layer,
    _build_author_context_discovery_layer,
    _source_grounding_claim_targets,
    _source_grounding_targets_from_block_decisions,
    _citation_reference_search_targets,
    _source_search_depth_status,
    _source_grounding_repair_matches,
    _source_reference_entries_from_layer,
    _source_reference_append_candidate,
    _source_grounding_query,
    _normalize_tavily_results,
    _source_result_confidence,
    _source_grounding_repair_prompt,
    _internet_reinforced_reauthor_prompt,
    _claim_narrowing_repair_prompt,
    _topk_texture_repair_prompt,
    _topk_safe_band_snapshot_prompt,
    _topk_safe_band_sentence_patch_prompt,
    _topk_plain_spoken_snapshot_prompt,
    _topk_safe_band_patch_rounds_default,
    _topk_safe_band_snapshot_max_tokens_default,
    _topk_repair_map,
    _topk_route_optimizer_candidates,
    _topk_masked_route_prompt,
    _extract_topk_route_patch_candidates,
    _apply_topk_route_patches,
    _post_topk_driver_map,
    _authorship_transformation_texture_driver_map,
    _authorship_transformation_texture_candidates,
    _post_topk_convergence_candidates,
    _extract_post_topk_patch_candidates,
    _apply_post_topk_patches,
    _phase_sampling_arg,
    _source_search_enabled,
    _plain_language_depolish_text,
    _final_score_drag_sentence_prune_text,
    _protected_anchor_brief_for_prompt,
    _build_source_grounding_search_layer,
    _confirmed_author_anchor_brief,
    _blocker_elimination_status,
    _confirmed_anchor_echo_reason,
    _validate_author_evidence_answers,
    _author_answer_relevance,
    _author_evidence_integration_prompt,
    _deterministic_author_anchor_paragraph,
    _clean_author_evidence_integrated_paragraph,
    _splice_author_evidence_paragraph,
    _blocked_human_candidate_repair_prompt,
    _blocking_finding_targets,
    _finding_local_repair_prompt,
    _extract_finding_local_patches,
    _apply_finding_local_patches,
    _extract_paragraph_component_candidates,
    _clean_paragraph_component_candidate,
    _paragraph_anchor_lock,
    _clean_source_sentence_candidate,
    _splice_paragraph,
    _rewrite_sampling_profile,
    _phase_sampling_arg,
    _phase_chat_sampling_kwargs,
    _mitigation_sampling_policy_summary,
    _llm_call_budget_exhausted_before_send,
    _ai_search_budget_policy,
    _ai_search_llm_hard_cap,
    _verified_candidate_scan_budget,
    _extend_candidate_scan_budget,
    _resolve_stage_llm_budget,
)
import llm.gateway as llm_gateway_module
from llm.gateway import LLMConfig, LLMGateway, _model_capabilities
from report import ReportBuilder, report_to_dict
from report.render_rewrite import render_rewrite_report


# ── Test data ──────────────────────────────────────────────────────────

SAMPLE_TEXT = """I believe the results show that AI detection tools have a 45.2% accuracy rate on academic writing. According to Smith (2023), "the methodology was flawed." Furthermore, the study found that students often struggle with paraphrasing effectively. In conclusion, more research is needed to understand the impact of these tools on education."""

PERSONAL_TEXT = """In my experience, I think the results are somewhat surprising. Perhaps the methodology could be improved. We found that the participants responded well to the intervention. I believe this has important implications for future research."""

BLAND_TEXT = """The results are surprising. The methodology should be improved. The participants responded well to the intervention. This has important implications for future research."""

FINDINGS = [
    Finding(finding_type="high_predictability", risk_level="high", evidence_strength="strong", detail="formulaic structure", evidence="top10=0.82, generic patterns", recommendation="rewrite", suggested_action_type="auto"),
    Finding(finding_type="generic_phrase", risk_level="medium", evidence_strength="moderate", detail="generic phrase", evidence='"Furthermore"', recommendation="replace with dash or nothing", suggested_action_type="auto"),
    Finding(finding_type="generic_phrase", risk_level="medium", evidence_strength="moderate", detail="generic phrase", evidence='"In conclusion"', recommendation="replace with specific statement", suggested_action_type="auto"),
    Finding(finding_type="style_shift", risk_level="low", evidence_strength="weak", detail="tone shift", evidence="formal→informal transition", recommendation="smooth voice", suggested_action_type="auto"),
    Finding(finding_type="uncited_claim", risk_level="high", evidence_strength="strong", detail="no citation", evidence="AI detection tools accuracy claim has no source", recommendation="add citation", suggested_action_type="manual"),
    Finding(finding_type="exact_copy", risk_level="high", evidence_strength="strong", detail="verbatim match", evidence='"the methodology was flawed."', recommendation="quote properly with citation", suggested_action_type="manual"),
    Finding(finding_type="close_paraphrase", risk_level="medium", evidence_strength="moderate", detail="close to source", evidence="75% overlap with Smith 2023", recommendation="rephrase with attribution", suggested_action_type="auto"),
]


def make_detect_result(findings=None):
    if findings is None:
        findings = FINDINGS
    return DetectResult(
        scanner="test_scanner",
        overall_risk=0.75,
        confidence="high",
        confidence_reason="multiple strong signals",
        findings=findings,
    )


passed = 0
failed = 0


def assert_test(condition, name):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. FIXABILITY ROUTING")
print("=" * 70)

for finding_type, route in FINDING_ROUTING.items():
    assert_test(
        route["fixability"] in {"auto", "partial", "manual", "protected"},
        f"{finding_type} → fixability={route['fixability']}",
    )

# Auto findings: predictability, generic_phrase, style_shift
auto_types = [ft for ft, r in FINDING_ROUTING.items() if r["fixability"] == "auto"]
assert_test(len(auto_types) >= 5, f"at least 5 auto types (got {len(auto_types)})")

# Manual findings: uncited_claim, missing_from_bib
manual_types = [ft for ft, r in FINDING_ROUTING.items() if r["fixability"] == "manual"]
assert_test(len(manual_types) >= 2, f"at least 2 manual types (got {len(manual_types)})")

# Protected findings: exact_copy, direct_quote_mismatch
protected_types = [ft for ft, r in FINDING_ROUTING.items() if r["fixability"] == "protected"]
assert_test(len(protected_types) >= 2, f"at least 2 protected types (got {len(protected_types)})")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("2. REWRITE PLANNER")
print("=" * 70)

dr = make_detect_result()
planner = RewritePlanner()
plan = planner.plan([dr])

assert_test(len(plan.actions) == 7, f"7 total actions (got {len(plan.actions)})")
assert_test(len(plan.auto_fixable) == 2, f"2 medium auto-fixable (got {len(plan.auto_fixable)})")
assert_test(len(plan.manual_required) == 4, f"4 manual/review-required (got {len(plan.manual_required)})")
assert_test(len(plan.protected) == 1, f"1 protected (got {len(plan.protected)})")
assert_test(plan.rewritable_risk > 0, f"rewritable_risk > 0 (got {plan.rewritable_risk})")
assert_test(plan.rewritable_risk < plan.total_weighted_risk, f"rewritable < total ({plan.rewritable_risk} < {plan.total_weighted_risk})")

# Auto-fixable should be sorted by weight desc
weights = [a.weight for a in plan.auto_fixable]
assert_test(weights == sorted(weights, reverse=True), "auto-fixable sorted by weight desc")

# Each action has fixability and reason
for a in plan.actions:
    assert_test(
        a.fixability in {"auto", "partial", "manual", "protected"},
        f"{a.finding.finding_type} has valid fixability={a.fixability}",
    )
    assert_test(len(a.reason) > 0, f"{a.finding.finding_type} has reason")

print(planner.plan_summary(plan))


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("3. VOICE PROFILE & GUARD")
print("=" * 70)

# Personal text should have high first-person ratio
vp_personal = analyze_voice(PERSONAL_TEXT)
assert_test(vp_personal.first_person_ratio > 0.05, f"personal text has first_person > 0.05 (got {vp_personal.first_person_ratio})")
assert_test(vp_personal.hedge_ratio > 0.01, f"personal text has hedges (got {vp_personal.hedge_ratio})")
assert_test(vp_personal.lexical_diversity > 0.5, f"personal text has diversity > 0.5 (got {vp_personal.lexical_diversity})")

# VoiceGuard should reject erosion
guard = VoiceGuard()
check = guard.check(PERSONAL_TEXT, BLAND_TEXT)
assert_test(not check.accepted, f"voice guard rejects bland rewrite")
assert_test("first_person" in check.reject_reason, f"reject reason mentions first_person (got '{check.reject_reason}')")

# VoiceGuard should accept minor changes
minor_rewrite = PERSONAL_TEXT.replace("I think the results", "I believe the results")
check2 = guard.check(PERSONAL_TEXT, minor_rewrite)
assert_test(check2.accepted, f"voice guard accepts minor change")

# Same text should always pass
check3 = guard.check(PERSONAL_TEXT, PERSONAL_TEXT)
assert_test(check3.accepted, f"voice guard accepts identical text")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("4. PROTECTED SPANS & REWRITE SURFACE")
print("=" * 70)

protected = detect_protected_spans(SAMPLE_TEXT)
reasons = [p.reason for p in protected]
assert_test("numeric" in reasons, f"found numeric spans (reasons: {reasons})")
assert_test("citation" in reasons, f"found citation spans (reasons: {reasons})")
assert_test("direct_quote" in reasons, f"found direct_quote spans (reasons: {reasons})")

surface = compute_rewrite_surface(SAMPLE_TEXT, protected)
assert_test(surface.rewrite_surface_ratio > 0.5, f"surface ratio > 0.5 (got {surface.rewrite_surface_ratio:.2f})")
assert_test(not surface.is_mostly_protected, f"sample text is not mostly protected")

# Preserved check
good_rewrite = SAMPLE_TEXT.replace("Furthermore", "—").replace("In conclusion", "Overall")
assert_test(
    protected_spans_preserved(SAMPLE_TEXT, good_rewrite, protected),
    "good rewrite preserves protected spans",
)

bad_rewrite = SAMPLE_TEXT.replace("45.2%", "a certain").replace("Smith (2023)", "a researcher")
assert_test(
    not protected_spans_preserved(SAMPLE_TEXT, bad_rewrite, protected),
    "bad rewrite loses protected spans",
)


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("5. SEMANTIC DRIFT GUARD")
print("=" * 70)

# Accept minor rewrite
drift_ok = check_semantic_drift(SAMPLE_TEXT, good_rewrite, threshold=0.85)
assert_test(drift_ok.accepted, f"minor rewrite passes drift check")

# Reject meaning change (needs dramatic keyword shift to trigger)
meaning_change = "The cat sat quietly on the mat while the dog ran through the park. Nothing about academic writing or detection tools was mentioned."
drift_bad = check_semantic_drift(SAMPLE_TEXT, meaning_change, threshold=0.85)
assert_test(not drift_bad.accepted, f"meaning change fails drift check")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("6. FLOOR DETECTION")
print("=" * 70)

factual = classify_floor("The accuracy was 45.2% on January 15, 2023.", 0.1)
assert_test(factual == "factual", f"numbers+dates → factual (got {factual})")

protected_floor = classify_floor("According to Smith (2023), the results were significant.", 0.5)
assert_test(protected_floor == "protected", f"high protected coverage → protected (got {protected_floor})")

normal = classify_floor("The results demonstrate a clear pattern of behavior.", 0.1)
assert_test(normal == "standard_phrase", f"normal text → standard_phrase (got {normal})")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("7. FIXABILITY-AWARE SCORING")
print("=" * 70)

fixability_map = {ftype: route["fixability"] for ftype, route in FINDING_ROUTING.items()}

total = weighted_finding_score(FINDINGS)
rewritable = weighted_rewritable_risk(FINDINGS, fixability_map)
assert_test(rewritable < total, f"rewritable ({rewritable}) < total ({total})")

# Manual findings should contribute 0 to rewritable score
manual_findings = [f for f in FINDINGS if fixability_map.get(f.finding_type) == "manual"]
manual_rewritable = weighted_rewritable_risk(manual_findings, fixability_map)
assert_test(manual_rewritable == 0, f"manual findings contribute 0 to rewritable (got {manual_rewritable})")

# Candidate scoring
drift = DriftCheck(accepted=True, similarity=0.92, reasons=[])
score = score_candidate(
    original_findings=FINDINGS,
    candidate_findings=FINDINGS[:4],
    original_text=SAMPLE_TEXT,
    candidate_text=good_rewrite,
    drift_check=drift,
    fixability_map=fixability_map,
)
assert_test(score.accepted, f"candidate accepted")
assert_test(score.total > 0, f"score > 0 (got {score.total})")
assert_test(score.voice_preservation > 0, f"voice_preservation > 0 (got {score.voice_preservation})")
assert_test(score.specificity_gain >= 0, f"specificity_gain >= 0 (got {score.specificity_gain})")

# Drift-rejected candidate should be hard-rejected
bad_drift = DriftCheck(accepted=False, similarity=0.3, reasons=["entity_lost"])
bad_score = score_candidate(
    original_findings=FINDINGS,
    candidate_findings=FINDINGS[:4],
    original_text=SAMPLE_TEXT,
    candidate_text=meaning_change,
    drift_check=bad_drift,
    fixability_map=fixability_map,
)
assert_test(not bad_score.accepted, f"drift-rejected candidate is not accepted")
assert_test(bad_score.total == 0, f"drift-rejected score is 0 (got {bad_score.total})")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("8. TRANSACTIONAL APPLY")
print("=" * 70)

config = RewriteConfig()

# Good rewrite: preserves protected spans, passes drift
tx_good = transactional_apply(SAMPLE_TEXT, good_rewrite, protected, config)
assert_test(tx_good.accepted, f"good rewrite accepted")
assert_test(tx_good.text == good_rewrite, f"accepted text is the candidate")

# Bad rewrite: loses protected spans
tx_bad1 = transactional_apply(SAMPLE_TEXT, bad_rewrite, protected, config)
assert_test(not tx_bad1.accepted, f"protected-span-losing rewrite rejected")
assert_test("protected_span_lost" in tx_bad1.reason, f"reason mentions protected_span_lost")
assert_test(tx_bad1.text == SAMPLE_TEXT, f"rejected text is the snapshot")

# Bad rewrite: loses meaning (dramatic text change)
dramatic_change = "The cat sat quietly on the mat while the dog ran through the park."
tx_bad2 = transactional_apply(SAMPLE_TEXT, dramatic_change, protected, config)
assert_test(not tx_bad2.accepted, f"meaning-changing rewrite rejected")

# Bad rewrite: voice erosion
tx_voice = transactional_apply(PERSONAL_TEXT, BLAND_TEXT, [], config, voice_guard=VoiceGuard())
assert_test(not tx_voice.accepted, f"voice-eroding rewrite rejected")
assert_test("voice_eroded" in tx_voice.reason, f"reason mentions voice_eroded")

# Empty protected spans: should work
tx_no_protected = transactional_apply("Hello world.", "Hello earth.", [], config)
assert_test(tx_no_protected.accepted, f"no protected spans, minor change accepted")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("9. EDIT RADIUS")
print("=" * 70)

assert_test("span" in EDIT_RADIUS, "span scope defined")
assert_test("sentence" in EDIT_RADIUS, "sentence scope defined")
assert_test("paragraph" in EDIT_RADIUS, "paragraph scope defined")

for scope, limits in EDIT_RADIUS.items():
    assert_test(limits["max_char_delta"] > 0, f"{scope} has max_char_delta > 0")
    assert_test(limits["max_semantic_drift"] > 0, f"{scope} has max_semantic_drift > 0")
    assert_test(limits["max_char_delta"] <= 0.5, f"{scope} char_delta <= 0.5")
    assert_test(limits["max_semantic_drift"] <= 0.15, f"{scope} drift <= 0.15")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("10. REWRITE CONFIG")
print("=" * 70)

cfg = RewriteConfig()
assert_test(cfg.max_passes == 2, f"default max_passes=2 (got {cfg.max_passes})")
assert_test(cfg.budget.max_changed_sentence_ratio == 0.20, f"default sentence ratio=0.20 (got {cfg.budget.max_changed_sentence_ratio})")
assert_test(cfg.budget.max_changed_char_ratio == 0.15, f"default char ratio=0.15 (got {cfg.budget.max_changed_char_ratio})")
assert_test(cfg.budget.max_total_changed_sentence_ratio == 0.30, f"total sentence cap=0.30")
assert_test(cfg.budget.max_total_changed_char_ratio == 0.25, f"total char cap=0.25")
assert_test(not cfg.suggestion_only, f"suggestion_only defaults to False")
assert_test(cfg.max_llm_calls == 8, f"default max_llm_calls=8 (got {cfg.max_llm_calls})")
assert_test(cfg.max_auto_targets == 8, f"default max_auto_targets=8 (got {cfg.max_auto_targets})")
assert_test(cfg.max_density_passes == 1, f"default max_density_passes=1 (got {cfg.max_density_passes})")
assert_test(cfg.max_rewrite_seconds == 120, f"default max_rewrite_seconds=120 (got {cfg.max_rewrite_seconds})")
assert_test(cfg.max_detect_loops == 0, f"default max_detect_loops=0 (got {cfg.max_detect_loops})")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("11. REGRESSION MEMORY")
print("=" * 70)

mem = RegressionMemory()
mem.record("span1", "drift", "candidate1", 0.3)
mem.record("span2", "voice", "candidate2", 0.5)
assert_test(mem.count == 2, f"2 rejections recorded (got {mem.count})")
assert_test(mem.was_rejected("span1", "drift"), f"span1/drift was rejected")
assert_test(not mem.was_rejected("span1", "voice"), f"span1/voice was not rejected")
summary = mem.summary()
assert_test(len(summary) == 2, f"summary has 2 entries")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("12. BEST CANDIDATE SELECTION")
print("=" * 70)

candidates = [
    CandidateScore(total=0.3, finding_reduction=0.2, semantic_preservation=0.9, voice_preservation=0.85, style_match=0.8, source_grounding=1.0, length_stability=1.0, specificity_gain=0.1, accepted=True, reject_reasons=[]),
    CandidateScore(total=0.6, finding_reduction=0.5, semantic_preservation=0.88, voice_preservation=0.9, style_match=0.85, source_grounding=1.0, length_stability=0.9, specificity_gain=0.2, accepted=True, reject_reasons=[]),
    CandidateScore(total=0, finding_reduction=0, semantic_preservation=0, voice_preservation=0, style_match=0, source_grounding=0, length_stability=0, specificity_gain=0, accepted=False, reject_reasons=["drift"]),
]

best = best_candidate(candidates)
assert_test(best is not None, f"best candidate found")
assert_test(best.total == 0.6, f"best score is 0.6 (got {best.total})")

# All rejected
all_rejected = [
    CandidateScore(total=0, finding_reduction=0, semantic_preservation=0, voice_preservation=0, style_match=0, source_grounding=0, length_stability=0, specificity_gain=0, accepted=False, reject_reasons=["drift"]),
]
assert_test(best_candidate(all_rejected) is None, f"no best from all-rejected")

# Below min_delta
low_scores = [
    CandidateScore(total=0.03, finding_reduction=0.01, semantic_preservation=0.9, voice_preservation=1.0, style_match=1.0, source_grounding=1.0, length_stability=1.0, specificity_gain=0, accepted=True, reject_reasons=[]),
]
assert_test(best_candidate(low_scores) is None, f"below min_delta rejected")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("13. AFFECTED REGION")
print("=" * 70)

sentences = [
    "First sentence.",
    "Second sentence.",
    "Third sentence.",
    "Fourth sentence.",
    "Fifth sentence.",
]

region = affected_region(2, sentences)
assert_test(2 in region, f"target sentence in region")
assert_test(1 in region, f"previous neighbor in region")
assert_test(3 in region, f"next neighbor in region")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("14. REWRITE OUTCOME ENUM")
print("=" * 70)

outcomes = [o.value for o in RewriteOutcome]
expected = ["improved", "partially_improved", "floor_reached", "manual_required", "rejected_for_drift", "suggestion_only"]
assert_test(outcomes == expected, f"outcomes match: {outcomes}")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("15. CONTEXT-AWARE REWRITE CONTRACT")
print("=" * 70)

json_ctx = DetectJSONParser.parse_dict({
    "input_text": "First sentence. Second sentence.",
    "findings": {
        "medium": [{
            "finding_id": "f001",
            "title": "medium_predictability",
            "category": "predictability",
            "adjusted_risk": "medium",
            "actionability": "auto_fixable",
            "sentence_id": "s002",
            "sentence_index": 2,
            "evidence": "Second sentence.",
            "detail": "predictable",
            "recommendation": "rewrite",
        }]
    }
})
parsed_finding = json_ctx.detect_results[0].findings[0]
assert_test(parsed_finding.location["sentence_index"] == 1, "s002 parses to zero-based sentence_index=1")

strict_findings = [
    Finding(finding_type="low_specificity", risk_level="medium", evidence_strength="moderate", detail="", evidence="Claim.", recommendation="", suggested_action_type="", actionability="auto_fixable"),
    Finding(finding_type="similarity_overlap", risk_level="medium", evidence_strength="moderate", detail="", evidence="Overlap.", recommendation="", suggested_action_type="", actionability="auto_fixable"),
    Finding(finding_type="repetitive_sentence_structure", risk_level="medium", evidence_strength="moderate", detail="", evidence="Repeat.", recommendation="", suggested_action_type="", actionability="auto_fixable"),
]
strict_plan = planner.plan([make_detect_result(strict_findings)])
assert_test(len(strict_plan.auto_fixable) == 0, "low_specificity/similarity/structure are not auto-rewritten")
assert_test(len(strict_plan.manual_required) == 3, "strict findings route to manual review")

prompt = _candidate_task_instruction(
    Finding(finding_type="medium_predictability", risk_level="medium", evidence_strength="moderate", detail="", evidence="", recommendation="", suggested_action_type=""),
    180,
)
assert_test("Return exactly 3 candidates" in prompt, "prompt asks for exactly 3 candidates")
assert_test("<TARGET>" in prompt, "prompt references marked target sentence")
assert_test("technical accuracy" in prompt and "digital landscape" in prompt, "prompt includes anti-polish examples")
assert_test("students' with 'learners" in prompt or "students' with 'learners" in prompt.replace("’", "'"), "prompt preserves student voice level")
assert_test("risk mitigation, not writing improvement" in prompt, "prompt frames task as mitigation not polishing")
assert_test("methods, source relationships, motivations" in prompt, "prompt forbids unsupported method/source/motivation additions")
assert_test("near-original candidate is better than an unsupported rewrite" in prompt, "prompt prefers conservative candidates")

operation = _plan_rewrite_operation(
    finding=Finding(
        finding_type="medium_predictability",
        risk_level="medium",
        evidence_strength="moderate",
        detail="",
        evidence="",
        recommendation="",
        suggested_action_type="",
    ),
    edit_brief={
        "signals": {
            "score": 0.51,
            "top10_ratio": 0.7,
            "problem_tokens": [{"token": "only"}, {"token": "knowledge"}],
            "predictable_token_spans": ["only place to acquire knowledge"],
        }
    },
    metadata_context={},
    original_sentence="In the contemporary education environment, school is no longer the only place to acquire knowledge.",
    previous_sentence="",
    next_sentence="Online information filled almost everyone’s life.",
    domain_anchors=["education", "school", "knowledge", "information"],
)
assert_test(operation["operation"] == "rebuild_predictable_span", "signals choose predictable-span rewrite operation")
operation_lines = "\n".join(_rewrite_operation_lines(operation))
assert_test("Operation problem tokens" in operation_lines, "operation prompt exposes problem tokens")
assert_test("Forbidden operation moves" in operation_lines, "operation prompt exposes forbidden moves")
assert_test("Sentence Total Reconstruction" in operation_lines, "operation prompt names total reconstruction mode")
operation_prompt = _candidate_task_instruction(
    Finding(finding_type="medium_predictability", risk_level="medium", evidence_strength="moderate", detail="", evidence="", recommendation="", suggested_action_type=""),
    180,
    operation,
)
assert_test("minimal edit around the predictable span" in operation_prompt, "candidate shapes follow rewrite operation")
assert_test("Sentence Total Reconstruction" in operation_prompt, "candidate prompt names total reconstruction mode")
assert_test("wipe the original sentence syntax" in operation_prompt, "candidate prompt asks to discard original syntax")
assert_test("retain only core technical keywords" in operation_prompt, "candidate prompt preserves core anchors")

long_operation = _plan_rewrite_operation(
    finding=Finding(
        finding_type="medium_predictability",
        risk_level="medium",
        evidence_strength="moderate",
        detail="",
        evidence="",
        recommendation="",
        suggested_action_type="",
    ),
    edit_brief={"signals": {"problem_tokens": [{"token": "procedure"}]}},
    metadata_context={},
    original_sentence=(
        "This enables the student to understand the precision required for each procedure as well as the specific techniques involved, "
        "and move the student from simplicity to precision, plus, which provides significant encouragement and stimulates the desire to learn for them."
    ),
    previous_sentence="",
    next_sentence="",
    domain_anchors=["student", "precision", "procedure", "techniques"],
)
assert_test(long_operation["operation"] == "shorten_and_reorder", "long predictable sentence chooses shorten/reorder operation")

for bad in (
    "The chart improves technical accuracy for each learner.",
    "Students face operational obstacles during practice.",
    "This visible learning framework preserves technical rigor in the digital landscape.",
    "Inclusive learning design is embedded within technical teaching and is especially evident when learners execute a controlled haircut.",
    "This breakdown helps students master the exact grip and tension, giving them the boost needed to perform.",
    "The Graduated haircut serves as a specific case here.",
    "Teaching it in a salon classroom presents a constant hurdle.",
    "Constructing triangle shapes requires lifting hair at projection angles from 1 to 90 degrees, yet learners frequently fail to see how a chosen degree creates that specific stacked silhouette.",
    "Taking the Graduated structure as a case, we see how these procedures guide the cut.",
):
    reason = _candidate_style_reject_reason("Students use the chart during practice.", bad)
    assert_test(bool(reason), f"anti-polish guard rejects: {bad}")

coherence_reason = _paragraph_coherence_reject_reason(
    "Students use the chart during practice.",
    "The chart supports a visible learning framework.",
    "I use the chart before cutting starts.",
    "They compare the guide after each section.",
    ["chart", "practice", "cutting", "guide"],
)
assert_test(bool(coherence_reason), "paragraph coherence guard rejects unsupported abstraction/anchor loss")
colloquial_reason = _paragraph_coherence_reject_reason(
    "In the contemporary education environment, school is no longer the only place to acquire knowledge.",
    "With online info filling daily life, school no longer stands as the only way to gain knowledge.",
    "",
    "Online information filled almost everyone’s life, and everyone can obtain all kinds of knowledge and information from the internet now.",
    ["education", "school", "knowledge", "information"],
)
assert_test(
    "unsupported_new_phrase" in colloquial_reason,
    "paragraph coherence guard rejects unsupported colloquial rewrite phrasing",
)
latest_colloquial_reason = _paragraph_coherence_reject_reason(
    "In the contemporary education environment, school is no longer the only place to acquire knowledge.",
    "Since online info fills daily life, school no longer stands as the sole place to gain knowledge.",
    "",
    "Online information filled almost everyone’s life, and everyone can obtain all kinds of knowledge and information from the internet now.",
    ["education", "school", "knowledge", "information"],
)
assert_test(
    "unsupported_new_phrase" in latest_colloquial_reason,
    "paragraph coherence guard rejects latest online-info candidate",
)
relationship_practice_reason = _paragraph_coherence_reject_reason(
    "Smaller class sizes make this relationship more apparent.",
    "Smaller classes help reveal how this relationship works in practice.",
    "When there are fewer learners, I can watch each section more closely.",
    "This makes it easier to see where the haircut guide is being lost.",
    ["class", "sizes", "relationship", "learners", "section", "haircut"],
)
assert_test(
    "unsupported_new_phrase" in relationship_practice_reason,
    "paragraph coherence guard rejects generic relationship-works candidate",
)
relationship_observe_reason = _paragraph_coherence_reject_reason(
    "Smaller class sizes make this relationship more apparent.",
    "With fewer learners in class, I can better observe how this relationship works.",
    "When there are fewer learners, I can watch each section more closely.",
    "This makes it easier to see where the haircut guide is being lost.",
    ["class", "sizes", "relationship", "learners", "section", "haircut"],
)
assert_test(
    "unsupported_new_phrase" in relationship_observe_reason,
    "paragraph coherence guard rejects observe-relationship-works candidate",
)
embedded_evident_reason = _paragraph_coherence_reject_reason(
    "It sits inside the way the skill is taught.",
    "Inclusive learning design is embedded within how technical skills are taught, which is especially evident when learners execute a controlled haircut.",
    "In adult VET hairdressing training, inclusive learning design should not sit outside technical skill teaching.",
    "Learners move from watching a demonstration to producing a controlled haircut themselves.",
    ["inclusive learning", "technical skills", "haircut", "learners"],
)
assert_test(
    bool(embedded_evident_reason),
    "paragraph coherence guard rejects formal density-rebuild phrasing",
)
especially_clear_reason = _paragraph_coherence_reject_reason(
    "It sits inside the way the skill is taught.",
    "Inclusive learning design is part of how technical skills are taught, not separate from them, and this is especially clear from my experience teaching Certificate III Hairdressing.",
    "In adult VET hairdressing training, inclusive learning design should not sit outside technical skill teaching.",
    "Learners move from watching a demonstration to producing a controlled haircut themselves.",
    ["inclusive learning", "technical skills", "haircut", "learners"],
)
assert_test(
    bool(especially_clear_reason),
    "paragraph coherence guard rejects especially-clear experience phrasing",
)
current_learning_reason = _paragraph_coherence_reject_reason(
    "In the contemporary education environment, school is no longer the only place to acquire knowledge.",
    "School remains a part of education, but many now get knowledge from other places in the current learning environment.",
    "",
    "Online information filled almost everyone’s life, and everyone can obtain all kinds of knowledge and information from the internet now.",
    ["education", "school", "knowledge", "information"],
)
assert_test(
    "unsupported_new_phrase" in current_learning_reason,
    "paragraph coherence guard rejects broad current-learning candidate",
)
knowledge_happens_reason = _paragraph_coherence_reject_reason(
    "In the contemporary education environment, school is no longer the only place to acquire knowledge.",
    "In today's education environment, acquiring knowledge happens in more places than just school.",
    "",
    "Online information filled almost everyone’s life, and everyone can obtain all kinds of knowledge and information from the internet now.",
    ["education", "school", "knowledge", "information"],
)
assert_test(
    "unsupported_new_phrase" in knowledge_happens_reason,
    "paragraph coherence guard rejects broad knowledge-happens candidate",
)
method_reason = _paragraph_coherence_reject_reason(
    "This enables the student to understand the precision required for each procedure and stimulates the desire to learn for them.",
    "Students learn the exact precision and steps for each procedure through guided practice, moving from simple attempts to careful execution, which encourages them and sparks their interest in continuing to improve.",
    "",
    "",
    ["student", "precision", "procedure", "learn"],
)
assert_test(
    "unsupported_new_phrase" in method_reason,
    "paragraph coherence guard rejects unsupported method/motivation additions",
)
latest_method_reason = _paragraph_coherence_reject_reason(
    "This enables the student to understand the precision required for each procedure as well as the specific techniques involved, and move the student from simplicity to precision, plus, which provides significant encouragement and stimulates the desire to learn for them.",
    "By breaking down each step with scaffolding, students grasp the exact precision and techniques needed, shifting from basic attempts to more accurate cuts, which encourages them and sparks their interest in practicing further.",
    "",
    "",
    ["student", "precision", "procedure", "techniques"],
)
assert_test(
    "unsupported_new_phrase" in latest_method_reason,
    "paragraph coherence guard rejects latest scaffolding candidate",
)
method_label_reason = _paragraph_coherence_reject_reason(
    "This enables the student to understand the precision required for each procedure as well as the specific techniques involved, and move the student from simplicity to precision, plus, which provides significant encouragement and stimulates the desire to learn for them.",
    "By focusing on each procedure's precision and techniques, this method moves students from simple to precise work and encourages their desire to keep learning.",
    "",
    "",
    ["student", "precision", "procedure", "techniques"],
)
assert_test(
    "unsupported_new_phrase" in method_label_reason,
    "paragraph coherence guard rejects method-label candidate",
)
new_terms_reason = _paragraph_coherence_reject_reason(
    "Students use the chart during practice.",
    "Students use the chart during structured guidance materials and repeated practice activities.",
    "",
    "",
    ["students", "chart", "practice"],
)
assert_test(
    "unsupported_new_terms" in new_terms_reason,
    "paragraph coherence guard rejects clusters of unsupported new content words",
)
grounded_paragraph_terms_reason = _paragraph_coherence_reject_reason(
    "Learners can then see how a small change in angle shifts the weight distribution.",
    "A small angle adjustment changes where the weight falls in the haircut structure.",
    "The graduated haircut structure is shown on the head chart first.",
    "The learner checks the weight line after each section.",
    ["learners", "angle", "weight", "haircut", "structure"],
    "The graduated haircut structure is shown on the head chart first. Learners can then see how a small change in angle shifts the weight distribution. The learner checks the weight line after each section.",
)
assert_test(
    not grounded_paragraph_terms_reason,
    "paragraph coherence guard allows terms grounded in paragraph context",
)

pred_raw = {
    "sentences": [
        {
            "sentence_id": "s001",
            "sentence": "In my class, learners first map the section on a chart.",
            "risk_label": "low",
            "score": 0.2,
            "top10_ratio": 0.3,
            "top50_ratio": 0.5,
            "avg_surprisal": 4.0,
            "paragraph_id": "p001",
            "top_predicted_tokens": [],
            "predictable_token_spans": [],
        },
        {
            "sentence_id": "s002",
            "sentence": "This helps students understand the process.",
            "risk_label": "medium",
            "score": 0.52,
            "top10_ratio": 0.72,
            "top50_ratio": 0.9,
            "avg_surprisal": 2.5,
            "paragraph_id": "p001",
            "token_results": [{"token": "helps", "raw_token": " helps", "rank": 1, "probability": 0.2, "top10": True}],
            "top_predicted_tokens": [{"token": "helps", "rank": 1, "probability": 0.2, "top10": True}],
            "predictable_token_spans": ["helps students"],
        },
    ]
}
brief_finding = Finding(
    finding_type="medium_predictability",
    risk_level="medium",
    evidence_strength="moderate",
    detail="predictable",
    evidence="This helps students understand the process.",
    recommendation="rewrite",
    suggested_action_type="",
    location={"sentence_index": 1},
    metadata={"finding_id": "f002"},
    actionability="auto_fixable",
)
brief_builder = ReportBuilder()
brief_builder.set_meta(original_text="In my class, learners first map the section on a chart. This helps students understand the process.")
brief_builder.add_detection(DetectResult(
    scanner="predictability",
    overall_risk=0.5,
    confidence="medium",
    confidence_reason="test",
    findings=[brief_finding],
    raw=pred_raw,
))
brief_json = report_to_dict(brief_builder.build())
briefs = brief_json.get("rewrite_edit_briefs") or []
assert_test(len(briefs) == 1, "detect JSON includes rewrite_edit_briefs")
assert_test(briefs[0]["sentence_index"] == 1, "rewrite brief sentence_index is zero-based")
assert_test(briefs[0]["previous_sentence"].startswith("In my class"), "rewrite brief includes previous sentence")
assert_test(briefs[0]["signals"]["problem_tokens"], "rewrite brief includes problem tokens")

rollback_report = render_rewrite_report(
    summary={
        "rollback_applied": True,
        "converged": False,
        "detect_scan_original_saved": {
            "ai_risk_badge": {"ai_likelihood_score": 40.0, "writing_quality_score": 50.0},
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
        },
        "detect_scan_rewritten": {
            "ai_risk_badge": {"ai_likelihood_score": 40.0, "writing_quality_score": 50.0},
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
        },
        "detect_scan_attempted": {
            "ai_risk_badge": {"ai_likelihood_score": 45.0, "writing_quality_score": 50.0},
            "findings": {"critical": [], "high": [{"finding_id": "new"}], "medium": [], "low": []},
        },
        "attempted_sentence_comparison": [{
            "orig_sentence": "Students use the chart.",
            "new_sentence": "The chart supports a visible learning framework.",
        }],
        "manual_suggestions": [{
            "finding_type": "medium_predictability",
            "rejection_reason": "final full detect scan regressed",
            "original_sentence": "Students use the chart.",
            "suggested_sentence": "Students check the chart before the next section.",
        }],
    },
    sentence_comparison=[],
    ai_findings=[],
)
assert_test("## Attempted Rewrite" in rollback_report, "rollback report shows attempted rewrite")
assert_test("## Manual Suggestions" in rollback_report, "rollback report keeps manual suggestions")

weak_ai_gate = _ai_first_gate_status(59.73, 59.42, True)
assert_test(
    weak_ai_gate["required"] and not weak_ai_gate["success"],
    "AI-first gate rejects tiny mitigation below 5-point drop",
)
cross_ai_gate = _ai_first_gate_status(62.0, 59.8, True)
assert_test(
    cross_ai_gate["required"] and cross_ai_gate["success"],
    "AI-first gate accepts crossing below 60 percent",
)
tiny_search_status = _ai_search_candidate_selection_status(57.78, 57.72, True)
assert_test(
    tiny_search_status["improved"]
    and not tiny_search_status["selectable"]
    and tiny_search_status["reason"] == "best_candidate_below_required_ai_drop",
    "AI search tracks tiny score drops without selecting them as mitigation success",
)
safe_partial_status = _safe_partial_quality_improvement_status(
    {
        "human_delta": 0.0,
        "ai_authorship_delta": 1.0,
        "ai_transformation_delta": 0.0,
        "ai_authorship_regression_blocked": False,
        "critical_high_regressed": False,
        "review_burden_regressed": False,
        "weighted_severity_regressed": False,
    },
    {"score": 1.119},
    ai_delta=0.27,
    finding_delta=-1,
    review_burden_delta=0,
    weighted_severity_delta=-1,
    critical_high_delta=0,
    ai_score_regressed=False,
)
assert_test(
    safe_partial_status["allowed"],
    "safe partial quality gate keeps small rescanned improvements instead of returning unchanged text",
)
safe_quality_only_status = _safe_partial_quality_improvement_status(
    {
        "human_delta": 0.0,
        "ai_authorship_delta": 0.0,
        "ai_transformation_delta": 0.0,
        "ai_authorship_regression_blocked": False,
        "critical_high_regressed": False,
        "review_burden_regressed": False,
        "weighted_severity_regressed": False,
    },
    {"score": 0.019},
    ai_delta=0.10,
    finding_delta=-2,
    review_burden_delta=-3,
    weighted_severity_delta=-5,
    critical_high_delta=0,
    ai_score_regressed=False,
)
assert_test(
    safe_quality_only_status["allowed"]
    and safe_quality_only_status["quality_only_improved"],
    "safe partial quality gate keeps finding/review/severity reductions even when AI drop is tiny",
)
assert_test(
    _ai_search_selected_by_final_safety_gate(
        True,
        {"safe_partial_quality_improvement": True},
    ),
    "final rollback gate preserves selected safe partial quality improvements",
)
assert_test(
    _ai_search_selected_by_final_safety_gate(
        True,
        {
            "selectable": True,
            "reason": "accepted_partial_turnitin_like_mitigation",
            "turnitin_like_ai_gate": {"score_drop": 2.562},
            "formula_gap_contract": {"score_drop": 2.562, "weighted_formula_score_drop": 2.562},
        },
    ),
    "final rollback gate preserves selected partial Turnitin-like formula improvements",
)
assert_test(
    _ai_search_selected_by_final_safety_gate(
        True,
        {
            "selectable": True,
            "reason": "accepted_formula_convergence_step",
            "formula_gap_contract": {"score_drop": 0.0, "weighted_formula_score_drop": 0.0},
        },
    ),
    "final rollback gate preserves selected non-worsening formula convergence candidates",
)
assert_test(
    not _ai_search_selected_by_final_safety_gate(
        True,
        {
            "selectable": True,
            "turnitin_like_ai_gate": {"score_drop": 0.1},
            "formula_gap_contract": {"score_drop": -0.1, "weighted_formula_score_drop": -0.1},
        },
    ),
    "final rollback gate rejects selected candidates when measured formula score worsens",
)
assert_test(
    _ai_search_final_selection_status({
        "ai_mitigation_search": {
            "best_attempt": {
                "selection_status": {
                    "selectable": True,
                    "partial_turnitin_like_mitigation": True,
                }
            }
        }
    }).get("partial_turnitin_like_mitigation"),
    "final rollback gate reads nested best-attempt selection status",
)
cleanup_formula_only_status = {
    "selectable": True,
    "reference_ai": 54.62,
    "partial_turnitin_like_mitigation": True,
    "ai_footprint_outcome_class": "cleanup_improved",
    "authenticity_gate": {
        "human_delta": 2.0,
        "candidate_human": 55.0,
        "ai_authorship_delta": 0.0,
        "ai_transformation_delta": 2.0,
    },
    "ai_footprint_gate": {
        "outcome_class": "cleanup_improved",
        "drops": {
            "topk_calibrated_risk": 0.0,
            "ai_likelihood": -0.01,
            "ai_authorship": 0.0,
            "external_ai_flag_risk": 1.007,
            "qualifying_text_ai_density": 0.01,
        },
    },
    "turnitin_like_ai_gate": {
        "safety_clean": True,
        "improved": True,
        "score_drop": 2.562,
        "component_drops": {
            "ai_likelihood": -0.005,
            "topk_calibrated_risk": 0.0,
            "semantic_uniformity": 0.961,
            "rewrite_smoothness": 0.006,
            "patchwork_expansion": 1.6,
        },
    },
    "formula_gap_contract": {
        "target_met": False,
        "score_drop": 2.562,
        "weighted_formula_score_drop": 2.562,
        "weighted_driver_drop_efficiency": 0.077,
        "weighted_driver_drops": {
            "ai_likelihood": {"drop": -0.005},
            "topk_calibrated_risk": {"drop": 0.0},
            "semantic_uniformity": {"drop": 0.961},
            "rewrite_smoothness": {"drop": 0.006},
            "patchwork_expansion": {"drop": 1.6},
        },
    },
    "human_shift_score": 8.024,
    "human_shift_components": {
        "semantic_uniformity_reduction": 8.01,
        "rewrite_smoothness_reduction": 0.06,
    },
}
topk_driver_progress_status = {
    "selectable": True,
    "reference_ai": 54.62,
    "partial_turnitin_like_mitigation": True,
    "topk_blocker_progress": True,
    "ai_footprint_outcome_class": "ai_footprint_blocked_by_texture",
    "authenticity_gate": {
        "human_delta": 0.0,
        "candidate_human": 53.0,
        "ai_authorship_delta": 2.0,
        "ai_transformation_delta": 2.0,
    },
    "ai_footprint_gate": {
        "outcome_class": "ai_footprint_blocked_by_texture",
        "drops": {
            "topk_calibrated_risk": 6.384,
            "ai_likelihood": 1.79,
            "ai_authorship": 2.0,
            "external_ai_flag_risk": 2.124,
            "qualifying_text_ai_density": 0.5,
        },
    },
    "turnitin_like_ai_gate": {
        "safety_clean": True,
        "improved": True,
        "score_drop": 2.321,
        "component_drops": {
            "ai_likelihood": 0.806,
            "topk_calibrated_risk": 1.277,
            "semantic_uniformity": 0.0,
            "rewrite_smoothness": 0.201,
            "patchwork_expansion": 0.0,
        },
    },
    "formula_gap_contract": {
        "target_met": False,
        "score_drop": 2.321,
        "weighted_formula_score_drop": 2.321,
        "weighted_driver_drop_efficiency": 0.05,
        "weighted_driver_drops": {
            "ai_likelihood": {"drop": 0.806},
            "topk_calibrated_risk": {"drop": 1.277},
            "semantic_uniformity": {"drop": 0.0},
            "rewrite_smoothness": {"drop": 0.201},
            "patchwork_expansion": {"drop": 0.0},
        },
    },
    "human_shift_score": 5.0,
    "human_shift_components": {
        "semantic_uniformity_reduction": 0.0,
        "rewrite_smoothness_reduction": 0.201,
    },
}
assert_test(
    classify_ai_search_candidate(cleanup_formula_only_status)["class"] == "formula_progress"
    and classify_ai_search_candidate(topk_driver_progress_status)["class"] == "detector_progress",
    "AI-search selection policy classifies formula-only progress separately from detector-driver progress",
)
assert_test(
    _detector_progress_rank(topk_driver_progress_status)
    > _detector_progress_rank(cleanup_formula_only_status),
    "detector progress rank prefers Top-k/AI-likelihood/authorship movement over cleanup-only formula gain",
)
assert_test(
    _goal_climb_candidate_rank(
        topk_driver_progress_status,
        {},
        candidate_ai=52.83,
        candidate_review_burden=51,
        candidate_weighted_severity=127,
        candidate_finding_total=64,
        original_review_burden=54,
        original_weighted_severity=133,
        original_finding_total=67,
    )
    > _goal_climb_candidate_rank(
        cleanup_formula_only_status,
        {},
        candidate_ai=54.63,
        candidate_review_burden=51,
        candidate_weighted_severity=127,
        candidate_finding_total=64,
        original_review_burden=54,
        original_weighted_severity=133,
        original_finding_total=67,
    ),
    "AI-search selector rejects headline AI regression before choosing formula-only progress",
)
cleanup_formula_no_ai_regression_status = dict(cleanup_formula_only_status)
cleanup_formula_no_ai_regression_status["reference_ai"] = 54.62
assert_test(
    _goal_climb_candidate_rank(
        cleanup_formula_no_ai_regression_status,
        {},
        candidate_ai=54.62,
        candidate_review_burden=51,
        candidate_weighted_severity=127,
        candidate_finding_total=64,
        original_review_burden=54,
        original_weighted_severity=133,
        original_finding_total=67,
    )
    > _goal_climb_candidate_rank(
        topk_driver_progress_status,
        {},
        candidate_ai=52.83,
        candidate_review_burden=51,
        candidate_weighted_severity=127,
        candidate_finding_total=64,
        original_review_burden=54,
        original_weighted_severity=133,
        original_finding_total=67,
    ),
    "AI-search selector chooses the larger measured formula drop when headline AI does not regress",
)
low_weight_raw_movement_status = {
    "selectable": True,
    "reference_ai": 50.0,
    "partial_turnitin_like_mitigation": True,
    "authenticity_gate": {
        "human_delta": 0.0,
        "candidate_human": 50.0,
        "ai_authorship_delta": 0.0,
        "ai_transformation_delta": 0.0,
    },
    "turnitin_like_ai_gate": {
        "safety_clean": True,
        "improved": True,
        "score_drop": 1.6,
        "component_drops": {"patchwork_expansion": 1.6},
    },
    "formula_gap_contract": {
        "target_met": False,
        "score_drop": 1.6,
        "weighted_formula_score_drop": 1.6,
        "weighted_driver_drop_efficiency": 0.1,
    },
}
high_weight_smaller_raw_status = {
    "selectable": True,
    "reference_ai": 50.0,
    "partial_turnitin_like_mitigation": True,
    "authenticity_gate": {
        "human_delta": 0.0,
        "candidate_human": 50.0,
        "ai_authorship_delta": 1.0,
        "ai_transformation_delta": 0.0,
    },
    "ai_footprint_gate": {
        "drops": {"ai_likelihood": 5.0},
    },
    "turnitin_like_ai_gate": {
        "safety_clean": True,
        "improved": True,
        "score_drop": 2.25,
        "component_drops": {"ai_likelihood": 2.25},
    },
    "formula_gap_contract": {
        "target_met": False,
        "score_drop": 2.25,
        "weighted_formula_score_drop": 2.25,
        "weighted_driver_drop_efficiency": 0.1,
    },
}
assert_test(
    _goal_climb_candidate_rank(
        high_weight_smaller_raw_status,
        {},
        candidate_ai=49.0,
        original_review_burden=10,
        original_weighted_severity=10,
        original_finding_total=10,
    )
    > _goal_climb_candidate_rank(
        low_weight_raw_movement_status,
        {},
        candidate_ai=49.0,
        original_review_burden=10,
        original_weighted_severity=10,
        original_finding_total=10,
    ),
    "AI-search selector follows weighted formula impact, not raw signal movement size",
)
def make_footprint_report(
    *,
    ai_authorship,
    human,
    ai_transformation,
    grounding,
    human_anchor,
    smoothness,
    semantic_uniformity,
    ai_likelihood,
    topk_pattern,
    topk_calibrated_risk=None,
    generic_assertion_risk,
    qualifying_text_ai_density=0,
    unsupported_claim_risk,
    broad_claim_risk,
    discourse,
    expansion=0,
    section_style=0,
    signal_agreement=0,
):
    return {
        "integrity_layers": {
            "layers": {
                "ai_authorship_risk": {"score": ai_authorship},
                "human_contribution_signal": {"score": human},
                "ai_transformation_risk": {"score": ai_transformation},
                "grounding_quality_risk": {"score": grounding},
            }
        },
        "ai_risk_badge": {
            "ai_likelihood_score": ai_likelihood,
            "ai_components": {
                "topk_pattern": topk_pattern,
                "topk_pattern_raw": topk_pattern,
                "topk_calibrated_risk": (
                    topk_calibrated_risk
                    if topk_calibrated_risk is not None
                    else calibrate_topk_risk(topk_pattern, eligible_sentence_count=3)["topk_calibrated_risk"]
                ),
                "generic_assertion_risk": generic_assertion_risk,
                "qualifying_text_ai_density": qualifying_text_ai_density,
            },
            "writing_components": {
                "unsupported_claim_risk": unsupported_claim_risk,
                "broad_claim_risk": broad_claim_risk,
            },
            "transformation_classification": {
                "features": {
                    "human_anchor_score": human_anchor / 100,
                    "rewrite_smoothness": smoothness / 100,
                    "ai_likelihood": ai_likelihood / 100,
                    "semantic_uniformity_risk": semantic_uniformity / 100,
                    "discourse_regularity_risk": discourse / 100,
                    "outline_to_text_expansion": expansion / 100,
                    "section_style_variance": section_style / 100,
                    "signal_agreement_score": signal_agreement / 100,
                }
            },
        },
        "findings": {"critical": [], "high": [], "medium": [], "low": []},
    }

footprint_original = make_footprint_report(
    ai_authorship=61,
    human=39,
    ai_transformation=61,
    grounding=55,
    human_anchor=22,
    smoothness=74,
    semantic_uniformity=66,
    ai_likelihood=70,
    topk_pattern=100,
    generic_assertion_risk=65,
    unsupported_claim_risk=90,
    broad_claim_risk=85,
    discourse=40,
)
footprint_cleanup = make_footprint_report(
    ai_authorship=61,
    human=39,
    ai_transformation=61,
    grounding=52,
    human_anchor=22,
    smoothness=74,
    semantic_uniformity=66,
    ai_likelihood=70,
    topk_pattern=100,
    generic_assertion_risk=65,
    unsupported_claim_risk=88,
    broad_claim_risk=85,
    discourse=40,
)
footprint_partial = make_footprint_report(
    ai_authorship=58,
    human=42,
    ai_transformation=57,
    grounding=52,
    human_anchor=24,
    smoothness=68,
    semantic_uniformity=61,
    ai_likelihood=64,
    topk_pattern=20,
    topk_calibrated_risk=18,
    generic_assertion_risk=64,
    unsupported_claim_risk=88,
    broad_claim_risk=84,
    discourse=36,
)
footprint_topk_still_unsafe = make_footprint_report(
    ai_authorship=57,
    human=42,
    ai_transformation=57,
    grounding=52,
    human_anchor=24,
    smoothness=68,
    semantic_uniformity=61,
    ai_likelihood=64,
    topk_pattern=90,
    topk_calibrated_risk=62,
    generic_assertion_risk=64,
    unsupported_claim_risk=88,
    broad_claim_risk=84,
    discourse=36,
)
cleanup_gate = _ai_footprint_gate_status(
    footprint_original,
    footprint_cleanup,
    review_burden_delta=-2,
    weighted_severity_delta=-4,
    critical_high_delta=0,
    ai_score_regressed=False,
)
partial_gate = _ai_footprint_gate_status(
    footprint_original,
    footprint_partial,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
    ai_score_regressed=False,
)
unsafe_topk_gate = _ai_footprint_gate_status(
    footprint_original,
    footprint_topk_still_unsafe,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
    ai_score_regressed=False,
)
footprint_stalled_topk = make_footprint_report(
    ai_authorship=57,
    human=39,
    ai_transformation=61,
    grounding=34,
    human_anchor=22,
    smoothness=51,
    semantic_uniformity=39,
    ai_likelihood=57,
    topk_pattern=100,
    topk_calibrated_risk=100,
    generic_assertion_risk=65,
    unsupported_claim_risk=20,
    broad_claim_risk=75,
    discourse=14,
)
stalled_topk_gate = _ai_footprint_gate_status(
    footprint_original,
    footprint_stalled_topk,
    review_burden_delta=-7,
    weighted_severity_delta=-17,
    critical_high_delta=0,
    ai_score_regressed=False,
)
assert_test(
    cleanup_gate["outcome_class"] == "cleanup_improved"
    and not cleanup_gate["material_driver_moved"],
    "AI-footprint gate separates review cleanup from mitigation movement",
)
assert_test(
    partial_gate["outcome_class"] == "partially_ai_mitigated"
    and partial_gate["material_driver_moved"]
    and partial_gate["drops"]["external_ai_flag_risk"] > cleanup_gate["drops"]["external_ai_flag_risk"],
    "AI-footprint gate recognizes material authorship/texture movement",
)
assert_test(
    unsafe_topk_gate["outcome_class"] == "ai_footprint_blocked_by_texture"
    and any(
        blocker.get("reason") == "topk_calibrated_above_safe_level"
        for blocker in unsafe_topk_gate["texture_blockers"]
    ),
    "AI-footprint gate blocks mitigation while calibrated Top-k remains above the safe level",
)
assert_test(
    stalled_topk_gate["outcome_class"] == "ai_footprint_blocked_by_texture"
    and not stalled_topk_gate["material_driver_moved"]
    and stalled_topk_gate["texture_blockers"],
    "AI-footprint gate blocks mitigation claims when top-k remains pinned and smoothness regresses",
)
turnitin_formula_original = make_footprint_report(
    ai_authorship=50,
    human=42,
    ai_transformation=50,
    grounding=45,
    human_anchor=20,
    smoothness=60,
    semantic_uniformity=50,
    ai_likelihood=70,
    topk_pattern=80,
    topk_calibrated_risk=40,
    generic_assertion_risk=40,
    qualifying_text_ai_density=40,
    unsupported_claim_risk=30,
    broad_claim_risk=30,
    discourse=30,
    expansion=20,
    section_style=55,
    signal_agreement=60,
)
turnitin_formula_candidate = make_footprint_report(
    ai_authorship=45,
    human=56,
    ai_transformation=44,
    grounding=45,
    human_anchor=62,
    smoothness=50,
    semantic_uniformity=45,
    ai_likelihood=58,
    topk_pattern=50,
    topk_calibrated_risk=30,
    generic_assertion_risk=40,
    qualifying_text_ai_density=38,
    unsupported_claim_risk=30,
    broad_claim_risk=30,
    discourse=30,
    expansion=20,
    section_style=20,
    signal_agreement=45,
)
turnitin_original_profile = _turnitin_like_ai_profile(turnitin_formula_original)
turnitin_candidate_profile = _turnitin_like_ai_profile(turnitin_formula_candidate)
turnitin_gate = _turnitin_like_ai_gate_status(
    turnitin_formula_original,
    turnitin_formula_candidate,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
    ai_score_regressed=False,
)
assert_test(
    turnitin_original_profile["components"]["patchwork_expansion"] == 55.0
    and turnitin_original_profile["score"] == 49.9
    and turnitin_candidate_profile["human_anchor_suppression"] == 27.9
    and turnitin_candidate_profile["target_score"] == TURNITIN_LIKE_TARGET_AI_SCORE,
    "Turnitin-like formula computes weighted score, patchwork max, human-anchor suppression, and target",
)
assert_test(
    turnitin_gate["safe_band"]
    and turnitin_gate["target_met"]
    and turnitin_gate["outcome_class"] == "ai_mitigated"
    and turnitin_gate["score_drop"] > 25,
    "Turnitin-like gate accepts only below-target formula reduction under existing safety constraints",
)
turnitin_micro_before = make_footprint_report(
    ai_authorship=50,
    human=50,
    ai_transformation=50,
    grounding=45,
    human_anchor=25,
    smoothness=45,
    semantic_uniformity=45,
    ai_likelihood=50,
    topk_pattern=85,
    topk_calibrated_risk=70,
    generic_assertion_risk=45,
    qualifying_text_ai_density=45,
    unsupported_claim_risk=35,
    broad_claim_risk=35,
    discourse=35,
)
turnitin_micro_after = make_footprint_report(
    ai_authorship=50,
    human=50,
    ai_transformation=50,
    grounding=45,
    human_anchor=25,
    smoothness=45,
    semantic_uniformity=45,
    ai_likelihood=49.99,
    topk_pattern=85,
    topk_calibrated_risk=70,
    generic_assertion_risk=45,
    qualifying_text_ai_density=45,
    unsupported_claim_risk=35,
    broad_claim_risk=35,
    discourse=35,
)
turnitin_micro_gate = _turnitin_like_ai_gate_status(
    turnitin_micro_before,
    turnitin_micro_after,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
    ai_score_regressed=False,
)
assert_test(
    turnitin_micro_gate["outcome_class"] == "partially_ai_mitigated"
    and 0 < turnitin_micro_gate["score_drop"] < 1.0,
    "Turnitin-like gate preserves safety-clean micro formula drops instead of applying a minimum-drop cliff",
)
turnitin_at_target_profile = turnitin_like_ai_profile(
    features={"ai_likelihood": 44.4444444444},
    ai_components={},
)
turnitin_below_target_profile = turnitin_like_ai_profile(
    features={"ai_likelihood": 44.4422222222},
    ai_components={},
)
assert_test(
    turnitin_at_target_profile["score"] == 20.0
    and not turnitin_at_target_profile["target_met"]
    and turnitin_below_target_profile["score"] == 19.999
    and turnitin_below_target_profile["target_met"],
    "Turnitin-like target is strict: 19.999 passes and 20.0 fails",
)
turnitin_authorship_backfire = _turnitin_like_ai_gate_status(
    turnitin_formula_original,
    make_footprint_report(
        ai_authorship=55,
        human=58,
        ai_transformation=44,
        grounding=45,
        human_anchor=70,
        smoothness=35,
        semantic_uniformity=30,
        ai_likelihood=35,
        topk_pattern=40,
        topk_calibrated_risk=20,
        generic_assertion_risk=20,
        qualifying_text_ai_density=20,
        unsupported_claim_risk=20,
        broad_claim_risk=20,
        discourse=20,
        expansion=10,
        section_style=10,
        signal_agreement=20,
    ),
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
    ai_score_regressed=False,
)
assert_test(
    turnitin_authorship_backfire["improved"]
    and not turnitin_authorship_backfire["safety_clean"]
    and turnitin_authorship_backfire["outcome_class"] == "no_turnitin_like_improvement",
    "Turnitin-like gate does not let human-anchor suppression hide authorship regression",
)
turnitin_cleanup_rank = _turnitin_like_candidate_rank(
    _turnitin_like_ai_gate_status(
        turnitin_formula_original,
        turnitin_formula_original,
        review_burden_delta=-2,
        weighted_severity_delta=-2,
        critical_high_delta=0,
        ai_score_regressed=False,
    ),
    review_burden_delta=-2,
    weighted_severity_delta=-2,
    critical_high_delta=0,
)
turnitin_mitigation_rank = _turnitin_like_candidate_rank(
    turnitin_gate,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
)
assert_test(
    turnitin_mitigation_rank > turnitin_cleanup_rank,
    "Turnitin-like rank prioritizes formula-driver reduction over cleanup-only candidates",
)
formula_gap = _formula_gap_contract(
    turnitin_formula_original,
    turnitin_formula_candidate,
    source_text="The education system is changing quickly. Students need judgement.",
    candidate_text="The education system is changing. Students need judgement, but the classroom still matters.",
)
assert_test(
    formula_gap["score_before"] == turnitin_original_profile["score"]
    and formula_gap["score_after"] == turnitin_candidate_profile["score"]
    and formula_gap["score_drop"] == turnitin_gate["score_drop"]
    and formula_gap["target_met"] is True
    and formula_gap["remaining_formula_gap"] == 0.0
    and formula_gap["changed_word_count"] > 0
    and "ai_likelihood" in (formula_gap["weighted_driver_drops"] or {})
    and formula_gap["driver_priority_plan"]
    and formula_gap["priority_basis"],
    "Formula-gap contract reports exact shared score movement, weighted driver drops, target state, and change budget",
)
formula_priority_gap = _formula_gap_contract(turnitin_formula_original, turnitin_formula_original)
priority_plan = formula_priority_gap["driver_priority_plan"]
assert_test(
    priority_plan[0]["driver"] in {"ai_likelihood", "topk_calibrated_risk"}
    and all("actionability" in row and "backfire_risk" in row for row in priority_plan)
    and all("expected_net_gain" in row for row in priority_plan)
    and formula_priority_gap["next_formula_driver"] == priority_plan[0]["driver"],
    "Formula-gap priority plan combines weighted impact with actionability, headroom, and backfire risk",
)
suppression_only_candidate = make_footprint_report(
    ai_authorship=50,
    human=42,
    ai_transformation=50,
    grounding=45,
    human_anchor=40,
    smoothness=60,
    semantic_uniformity=50,
    ai_likelihood=70,
    topk_pattern=80,
    topk_calibrated_risk=40,
    generic_assertion_risk=40,
    qualifying_text_ai_density=40,
    unsupported_claim_risk=30,
    broad_claim_risk=30,
    discourse=30,
    expansion=20,
    section_style=55,
    signal_agreement=60,
)
suppression_only_gap = _formula_gap_contract(turnitin_formula_original, suppression_only_candidate)
assert_test(
    suppression_only_gap["weighted_driver_drops"]["human_anchor_suppression"]["gain"] == 9.0
    and suppression_only_gap["score_drop"] == 9.0
    and suppression_only_gap["positive_ai_burden"]["before"] == suppression_only_gap["positive_ai_burden"]["after"],
    "Human Anchor suppression reduces the Turnitin-like score one-for-one when positive drivers are unchanged",
)
portfolio_plan = _formula_portfolio_plan(
    turnitin_formula_original,
    turnitin_formula_candidate,
    observed_candidates=[
        {
            "strategy": "safe_human_anchor_candidate",
            "formula_gap_contract": suppression_only_gap,
            "turnitin_like_ai_gate": {"safety_clean": True},
            "selection_status": {"selectable": True, "reason": "accepted_formula_portfolio_partial"},
        },
        {
            "strategy": "blocked_likelihood_candidate",
            "formula_gap_contract": formula_gap,
            "turnitin_like_ai_gate": {"safety_clean": False},
            "selection_status": {"selectable": False, "reason": "review_burden_regressed"},
        },
    ],
)
assert_test(
    portfolio_plan["positive_ai_burden"]["before"] > portfolio_plan["human_anchor_suppression"]["before"]
    and portfolio_plan["suppression_headroom"] >= 0
    and portfolio_plan["observed_driver_movement"]["human_anchor_suppression"]["best_safe_drop"] == 9.0
    and portfolio_plan["driver_priorities"]
    and portfolio_plan["selected_driver_portfolio"],
    "Formula portfolio plan exposes positive AI burden, Human Anchor suppression, observed movement, and selected driver portfolio",
)
formula_one_signal_backfire = _formula_gap_contract(
    turnitin_formula_original,
    make_footprint_report(
        ai_authorship=50,
        human=41,
        ai_transformation=52,
        grounding=45,
        human_anchor=0,
        smoothness=100,
        semantic_uniformity=100,
        ai_likelihood=100,
        topk_pattern=30,
        topk_calibrated_risk=10,
        generic_assertion_risk=40,
        qualifying_text_ai_density=40,
        unsupported_claim_risk=30,
        broad_claim_risk=30,
        discourse=30,
        expansion=100,
        section_style=100,
        signal_agreement=100,
    ),
    source_text="A generic paragraph.",
    candidate_text="A generic paragraph with a lower token route but worse model cadence.",
)
assert_test(
    formula_one_signal_backfire["component_drops"]["topk_calibrated_risk"] > 0
    and formula_one_signal_backfire["score_drop"] < 0
    and not formula_one_signal_backfire["target_met"],
    "Formula-gap contract rejects one-signal wins when total weighted score gets worse",
)
assert_test(
    _formula_gap_candidate_rank(formula_gap, turnitin_gate)
    > _formula_gap_candidate_rank(formula_one_signal_backfire, {"safety_clean": True}),
    "Formula-gap rank selects total formula closure over isolated signal movement",
)
concept_guard_source = (
    "The United States has a strong cultural influence. "
    "American movies, music, fashion, and social media trends are consumed globally."
)
concept_guard_bad = (
    "The useful check is whether the student can explain the steps, not only show the final answer. "
    "The United States has a strong cultural influence. "
    "American movies, music, fashion, and social media trends are consumed globally."
)
concept_guard_good = (
    "The United States has a strong cultural influence. "
    "American movies, music, fashion, and social media trends are consumed globally. "
    "This point should stay tied to united and states, not treated as a wider claim."
)
assert_test(
    "unsupported_concept_origin" in _candidate_concept_origin_reject_reason(
        concept_guard_source,
        concept_guard_bad,
    )
    and not _candidate_concept_origin_reject_reason(
        concept_guard_source,
        concept_guard_good,
    ),
    "concept-origin guard rejects unsupported imported concepts without topic-specific hardcoding",
)
primary_pinned_current = make_footprint_report(
    ai_authorship=53,
    human=56,
    ai_transformation=44,
    grounding=45,
    human_anchor=36,
    smoothness=45.63,
    semantic_uniformity=55.1,
    ai_likelihood=52.52,
    topk_pattern=100,
    topk_calibrated_risk=92.634,
    generic_assertion_risk=90,
    qualifying_text_ai_density=60.27,
    unsupported_claim_risk=25,
    broad_claim_risk=25,
    discourse=45,
    expansion=65,
    section_style=65,
    signal_agreement=46.67,
)
anchor_heavy_candidate = make_footprint_report(
    ai_authorship=51,
    human=64,
    ai_transformation=36,
    grounding=45,
    human_anchor=48,
    smoothness=45.29,
    semantic_uniformity=46.3,
    ai_likelihood=51.08,
    topk_pattern=96,
    topk_calibrated_risk=91.489,
    generic_assertion_risk=90,
    qualifying_text_ai_density=60.27,
    unsupported_claim_risk=25,
    broad_claim_risk=25,
    discourse=45,
    expansion=45,
    section_style=45,
    signal_agreement=46.67,
)
primary_driver_candidate = make_footprint_report(
    ai_authorship=50,
    human=60,
    ai_transformation=40,
    grounding=45,
    human_anchor=42,
    smoothness=44,
    semantic_uniformity=50,
    ai_likelihood=48,
    topk_pattern=90,
    topk_calibrated_risk=86,
    generic_assertion_risk=85,
    qualifying_text_ai_density=56,
    unsupported_claim_risk=25,
    broad_claim_risk=25,
    discourse=42,
    expansion=45,
    section_style=45,
    signal_agreement=46.67,
)
anchor_heavy_gate = _formula_convergence_primary_burden_gate_status(
    primary_pinned_current,
    anchor_heavy_candidate,
    _formula_gap_contract(primary_pinned_current, anchor_heavy_candidate),
)
primary_driver_gate = _formula_convergence_primary_burden_gate_status(
    primary_pinned_current,
    primary_driver_candidate,
    _formula_gap_contract(primary_pinned_current, primary_driver_candidate),
)
assert_test(
    anchor_heavy_gate["accepted"]
    and anchor_heavy_gate["reason"] == "accepted"
    and primary_driver_gate["accepted"],
    "formula convergence records pinned-driver diagnostics without discarding safe formula drops",
)
assert_test(
    _formula_gap_changed_word_count("one two three", "one two four five") == 2,
    "Formula-gap changed-word budget counts replacements and insertions for efficiency ranking",
)
rewrite_source_for_turnitin_target = open(__file__.replace("test_rewrite_v2.py", "rewrite_pipeline.py")).read()
assert_test(
    "DRAFTPROOF_TURNITIN_LIKE_SAFE_BAND" not in rewrite_source_for_turnitin_target
    and "DRAFTPROOF_TURNITIN_LIKE_MIN_DROP" not in rewrite_source_for_turnitin_target
    and "TURNITIN_LIKE_TARGET_AI_SCORE" in rewrite_source_for_turnitin_target,
    "Turnitin-like target and partial-progress gate are shared code, not environment-tuned cliffs",
)
assert_test(
    _safe_topk_limit() == 25.0,
    "Calibrated Top-k safe mark is fixed at 25 and is not an environment tuning knob",
)
calibrated_mid = calibrate_topk_risk(67.37, eligible_sentence_count=30)
calibrated_high = calibrate_topk_risk(90.0, eligible_sentence_count=30)
calibrated_non_prose = calibrate_topk_risk(10.0, eligible_sentence_count=0)
assert_test(
    calibrated_mid["topk_pattern_raw"] == 67.37
    and calibrated_mid["topk_calibrated_risk"] < 25
    and calibrated_mid["topk_safe_band"] is True
    and calibrated_high["topk_calibrated_risk"] >= 25
    and calibrated_high["topk_safe_band"] is False
    and calibrated_non_prose["topk_safe_band"] is False,
    "Top-k calibration gates safe band on calibrated risk, not raw GPT-2 Top-k",
)
strict_safe_report = make_footprint_report(
    ai_authorship=34,
    human=66,
    ai_transformation=34,
    grounding=40,
    human_anchor=40,
    smoothness=30,
    semantic_uniformity=30,
    ai_likelihood=34,
    topk_pattern=60,
    topk_calibrated_risk=18,
    generic_assertion_risk=20,
    unsupported_claim_risk=20,
    broad_claim_risk=20,
    discourse=20,
)
strict_blocked_report = make_footprint_report(
    ai_authorship=51,
    human=44,
    ai_transformation=56,
    grounding=58,
    human_anchor=30,
    smoothness=38,
    semantic_uniformity=40,
    ai_likelihood=51,
    topk_pattern=60,
    topk_calibrated_risk=18,
    generic_assertion_risk=90,
    unsupported_claim_risk=25,
    broad_claim_risk=15,
    discourse=30,
)
strict_safe_status = _strict_ai_safe_band_status(strict_safe_report)
strict_blocked_status = _strict_ai_safe_band_status(strict_blocked_report)
strict_from_gate_status = _strict_ai_safe_band_status_from_footprint_gate(
    _ai_footprint_gate_status(strict_safe_report, strict_blocked_report)
)
strict_missing_gate_status = _strict_ai_safe_band_status_from_footprint_gate({})
assert_test(
    strict_safe_status["achieved"]
    and not strict_safe_status["remaining"]
    and not strict_blocked_status["achieved"]
    and not strict_from_gate_status["achieved"]
    and not strict_missing_gate_status["achieved"]
    and strict_missing_gate_status.get("unscored")
    and strict_from_gate_status["profile"]["topk_calibrated_risk"] == 18.0
    and [item["driver"] for item in strict_blocked_status["remaining"]] == [
        "ai_authorship",
        "ai_transformation",
        "external_ai_flag_risk",
    ],
    "post-Top-k strict safe band is recoverable from the canonical candidate footprint gate",
)
topk_improved_generic_backfire_report = make_footprint_report(
    ai_authorship=43,
    human=48,
    ai_transformation=52,
    grounding=55,
    human_anchor=31,
    smoothness=38,
    semantic_uniformity=41,
    ai_likelihood=45,
    topk_pattern=73,
    topk_calibrated_risk=30.328,
    generic_assertion_risk=90,
    unsupported_claim_risk=25,
    broad_claim_risk=15,
    discourse=30,
)
multi_signal_contract = _multi_signal_candidate_contract(
    footprint_original,
    topk_improved_generic_backfire_report,
)
assert_test(
    multi_signal_contract["needs_balance_repair"]
    and any(
        item["driver"] == "generic_assertion_risk"
        and item["increase"] >= 15
        for item in multi_signal_contract["severe_backfires"]
    )
    and any(
        item["driver"] == "topk_calibrated_risk"
        and item["drop"] > 60
        for item in multi_signal_contract["improvements"]
    ),
    "multi-signal contract catches Top-k wins that backfire generic assertion risk",
)
density_blocked_report = make_footprint_report(
    ai_authorship=34,
    human=66,
    ai_transformation=34,
    grounding=40,
    human_anchor=40,
    smoothness=30,
    semantic_uniformity=30,
    ai_likelihood=34,
    topk_pattern=60,
    topk_calibrated_risk=18,
    generic_assertion_risk=20,
    qualifying_text_ai_density=72,
    unsupported_claim_risk=20,
    broad_claim_risk=20,
    discourse=20,
)
density_gate = _ai_footprint_gate_status(strict_safe_report, density_blocked_report)
density_strict = _strict_ai_safe_band_status(density_blocked_report)
assert_test(
    not density_gate["safe_band"]
    and any(
        row.get("driver") == "qualifying_text_ai_density"
        for row in density_gate["remaining_ai_footprint_drivers"]
    )
    and not density_strict["achieved"],
    "qualifying AI-density is a first-class strict-safe blocker, not a side signal",
)
weak_topk_balanced_rank = _goal_climb_candidate_rank(
    {
        "selectable": True,
        "topk_blocker_progress": True,
        "ai_footprint_outcome_class": "ai_footprint_blocked_by_texture",
        "authenticity_gate": {
            "human_delta": 3,
            "candidate_human": 40,
            "ai_authorship_delta": 5,
            "ai_transformation_delta": 3,
        },
        "ai_footprint_gate": {
            "drops": {
                "topk_calibrated_risk": 4.638,
                "external_ai_flag_risk": 6.624,
                "ai_likelihood": 4.6,
            },
        },
        "multi_signal_contract": {
            "balance_score": 97.221,
            "severe_backfires": [],
        },
    },
    {},
)
strong_topk_backfire_rank = _goal_climb_candidate_rank(
    {
        "selectable": True,
        "topk_blocker_progress": True,
        "ai_footprint_outcome_class": "ai_footprint_blocked_by_texture",
        "authenticity_gate": {
            "human_delta": 11,
            "candidate_human": 48,
            "ai_authorship_delta": 18,
            "ai_transformation_delta": 11,
        },
        "ai_footprint_gate": {
            "drops": {
                "topk_calibrated_risk": 69.672,
                "external_ai_flag_risk": 21.126,
                "ai_likelihood": 17.5,
            },
        },
        "multi_signal_contract": {
            "balance_score": 58.0,
            "severe_backfires": [{"driver": "generic_assertion_risk"}],
        },
    },
    {},
)
assert_test(
    strong_topk_backfire_rank > weak_topk_balanced_rank,
    "goal selector does not let balanced cleanup beat major Top-k/authorship movement",
)
topk_only_partial_rank = _goal_climb_candidate_rank(
    {
        "selectable": True,
        "topk_blocker_progress": True,
        "ai_footprint_outcome_class": "ai_footprint_blocked_by_texture",
        "authenticity_gate": {
            "human_delta": 1,
            "candidate_human": 38,
            "ai_authorship_delta": 5,
            "ai_transformation_delta": 4,
        },
        "ai_footprint_gate": {
            "drops": {
                "topk_calibrated_risk": 7.0,
                "external_ai_flag_risk": 5.0,
                "ai_likelihood": 5.0,
                "qualifying_text_ai_density": 5.0,
            },
        },
    },
    {},
)
human_anchor_partial_rank = _goal_climb_candidate_rank(
    {
        "selectable": True,
        "human_anchor_amplifier": True,
        "topk_blocker_progress": True,
        "ai_footprint_outcome_class": "ai_footprint_blocked_by_texture",
        "authenticity_gate": {
            "human_delta": 8,
            "candidate_human": 45,
            "ai_authorship_delta": 6,
            "ai_transformation_delta": 8,
        },
        "human_anchor_driver_contract": {
            "deltas": {
                "human_anchor_score": 24,
                "lived_detail_risk": 30,
            },
        },
        "ai_footprint_gate": {
            "drops": {
                "topk_calibrated_risk": 6.0,
                "external_ai_flag_risk": 4.5,
                "ai_likelihood": 5.0,
                "qualifying_text_ai_density": 6.0,
            },
        },
    },
    {},
)
assert_test(
    human_anchor_partial_rank > topk_only_partial_rank,
    "goal selector prioritizes measured Human Anchor movement over weaker Top-k-only partials",
)
topk_near_miss_a = make_footprint_report(
    ai_authorship=50,
    human=45,
    ai_transformation=55,
    grounding=55,
    human_anchor=30,
    smoothness=35,
    semantic_uniformity=48,
    ai_likelihood=50,
    topk_pattern=69,
    topk_calibrated_risk=26.3,
    generic_assertion_risk=90,
    unsupported_claim_risk=25,
    broad_claim_risk=15,
    discourse=30,
)
topk_near_miss_b = make_footprint_report(
    ai_authorship=50,
    human=45,
    ai_transformation=55,
    grounding=55,
    human_anchor=30,
    smoothness=35,
    semantic_uniformity=48,
    ai_likelihood=50,
    topk_pattern=70,
    topk_calibrated_risk=27.4,
    generic_assertion_risk=90,
    unsupported_claim_risk=25,
    broad_claim_risk=15,
    discourse=30,
)
assert_test(
    _topk_rebuild_fallback_rank(topk_near_miss_a) > _topk_rebuild_fallback_rank(topk_near_miss_b),
    "Top-k rebuild fallback keeps the better near-miss instead of the later worse round",
)
near_miss_keep = _topk_near_miss_partial_keep_decision(
    topk_value=25.654,
    safe_limit=25.0,
    topk_drop=74.346,
    ai_drop=15.05,
    ai_authorship_drop=15.0,
    ai_transformation_drop=9.0,
    review_burden_delta=-48,
    weighted_severity_delta=-102,
    critical_high_delta=0,
)
topk_blocked_keep = _topk_near_miss_partial_keep_decision(
    topk_value=27.4,
    safe_limit=25.0,
    topk_drop=72.6,
    ai_drop=15.0,
    ai_authorship_drop=15.0,
    ai_transformation_drop=9.0,
    review_burden_delta=-48,
    weighted_severity_delta=-102,
    critical_high_delta=0,
)
topk_blocked_reject = _topk_near_miss_partial_keep_decision(
    topk_value=90.0,
    safe_limit=25.0,
    topk_drop=3.0,
    ai_drop=15.0,
    ai_authorship_drop=15.0,
    ai_transformation_drop=9.0,
    review_burden_delta=-48,
    weighted_severity_delta=-102,
    critical_high_delta=0,
)
assert_test(
    near_miss_keep["allowed"]
    and topk_blocked_keep["allowed"]
    and not topk_blocked_reject["allowed"]
    and topk_blocked_reject["reason"] == "topk_drop_too_small",
    "Top-k final gate preserves material partial progress instead of rolling back on arbitrary miss distance",
)
assert_test(
    _selection_status_topk_safe({
        "topk_safe_band_achieved": True,
        "ai_footprint_gate": {
            "after": {"authorship_footprint": {"topk_calibrated_risk": 18.92}}
        },
    })
    and not _selection_status_topk_safe({
        "topk_blocker_progress": True,
        "ai_footprint_gate": {
            "after": {"authorship_footprint": {"topk_calibrated_risk": 72.938}}
        },
    }),
    "selector distinguishes Top-k-safe frontier from merely improved Top-k-blocked progress",
)
saturated_topk_report = {"ai_risk_badge": {"ai_components": {"topk_calibrated_risk": 100.0}}}
phase_contract = _strict_safe_phase_budget_contract(17, "word " * 900, saturated_topk_report)
phase_contract_cap10 = _strict_safe_phase_budget_contract(10, "word " * 900, saturated_topk_report)
phase_contract_lower = _strict_safe_phase_budget_contract(7)
assert_test(
    phase_contract["total_llm_hard_cap"] == 15
    and phase_contract["topk_safe_band_rebuild"] == 12
    and phase_contract["authorship_transformation_texture_controller"] == 2
    and phase_contract["final_texture_proxy_repair"] == 1
    and phase_contract["emergency_diagnostic_reserve"] == 0
    and phase_contract["post_topk_strict_safe_optimizer"] == 0
    and phase_contract_cap10["topk_safe_band_rebuild"] == 7
    and phase_contract_cap10["authorship_transformation_texture_controller"] == 2
    and phase_contract_cap10["final_texture_proxy_repair"] == 1
    and sum(value for key, value in phase_contract_lower.items() if key != "total_llm_hard_cap") <= 7,
    "strict-safe phase budget contract reserves downstream texture calls under the 10-call cap",
)
strict_rank_base = make_footprint_report(
    ai_authorship=46,
    human=47,
    ai_transformation=53,
    grounding=20,
    human_anchor=30,
    smoothness=40,
    semantic_uniformity=45,
    ai_likelihood=45,
    topk_pattern=70,
    topk_calibrated_risk=21,
    generic_assertion_risk=90,
    unsupported_claim_risk=10,
    broad_claim_risk=10,
    discourse=20,
)
strict_rank_candidate_good = make_footprint_report(
    ai_authorship=44,
    human=47,
    ai_transformation=49,
    grounding=20,
    human_anchor=30,
    smoothness=38,
    semantic_uniformity=43,
    ai_likelihood=43,
    topk_pattern=69,
    topk_calibrated_risk=22,
    generic_assertion_risk=82,
    unsupported_claim_risk=10,
    broad_claim_risk=10,
    discourse=18,
)
strict_rank_candidate_worse = make_footprint_report(
    ai_authorship=45,
    human=48,
    ai_transformation=51,
    grounding=20,
    human_anchor=30,
    smoothness=38,
    semantic_uniformity=43,
    ai_likelihood=43,
    topk_pattern=74,
    topk_calibrated_risk=26,
    generic_assertion_risk=80,
    unsupported_claim_risk=10,
    broad_claim_risk=10,
    discourse=18,
)
assert_test(
    _strict_safe_candidate_rank(strict_rank_base, strict_rank_candidate_good)
    > _strict_safe_candidate_rank(strict_rank_base, strict_rank_candidate_worse),
    "strict-safe rank preserves Top-k-safe candidates and prefers external/authorship/transformation movement",
)
post_topk_patch_payload = json.dumps({
    "candidates": [
        {
            "reason": "compress broad claim",
            "patches": [
                {"paragraph_index": 1, "replacement": "This narrower paragraph keeps the point but removes broad reusable claims."}
            ],
        }
    ]
})
post_topk_patches = _extract_post_topk_patch_candidates(post_topk_patch_payload)
post_topk_text, post_topk_applied = _apply_post_topk_patches(
    "Heading\n\nThis is the first paragraph.\n\nThis paragraph contains a broad reusable claim that should be narrowed.",
    post_topk_patches[0]["patches"],
)
assert_test(
    len(post_topk_patches) == 1
    and len(post_topk_applied) == 1
    and "narrower paragraph" in post_topk_text,
    "post-Top-k JSON patch candidates parse and apply paragraph-local replacements",
)
post_topk_generic_source = (
    "Education is changing rapidly. This highlights the importance of helping students develop important skills. "
    "Teachers should support learning in many different ways. This affects classroom practice over time.\n\n"
    "Students use YouTube, TikTok, AI tools, search engines, and online courses before they ask a teacher. "
    "That creates a problem of what to trust.\n\n"
    "In conclusion, education should prepare students for a changing world. "
    "This shows that teachers play a crucial role in student success."
)
post_topk_generic_report = make_footprint_report(
    ai_authorship=50,
    human=45,
    ai_transformation=55,
    grounding=45,
    human_anchor=30,
    smoothness=35,
    semantic_uniformity=49,
    ai_likelihood=50,
    topk_pattern=64,
    topk_calibrated_risk=21,
    generic_assertion_risk=90,
    unsupported_claim_risk=25,
    broad_claim_risk=15,
    discourse=27,
)
post_topk_driver_map = _post_topk_driver_map(post_topk_generic_source, post_topk_generic_report)
texture_driver_map = _authorship_transformation_texture_driver_map(post_topk_generic_source, post_topk_generic_report)
post_topk_convergence_candidates = _post_topk_convergence_candidates(
    post_topk_generic_source,
    post_topk_generic_report,
    limit=10,
)
texture_candidates = _authorship_transformation_texture_candidates(
    post_topk_generic_source,
    post_topk_generic_report,
    limit=10,
)
assert_test(
    post_topk_driver_map["generic_sentence_ratio"] > 0
    and post_topk_convergence_candidates
    and any(meta.get("post_topk_convergence") for _s, _c, meta in post_topk_convergence_candidates)
    and any(meta.get("operation") == "authorship_suppression_candidate" for _s, _c, meta in post_topk_convergence_candidates)
    and any(meta.get("operation") == "transformation_reduction_candidate" for _s, _c, meta in post_topk_convergence_candidates)
    and any("This highlights the importance" not in candidate for _s, candidate, _m in post_topk_convergence_candidates),
    "post-Top-k convergence optimizer builds strict-safe authorship, transformation, and generic-collapse candidates",
)
assert_test(
    texture_driver_map["kind"] == "authorship_transformation_texture_driver_map"
    and texture_driver_map["ranked_blocks"]
    and texture_driver_map["authorship_drivers"]["sentence_length_uniformity"] >= 0
    and texture_candidates
    and all(meta.get("authorship_transformation_texture_controller") for _s, _c, meta in texture_candidates)
    and {
        meta.get("texture_candidate_family")
        for _s, _c, meta in texture_candidates
    }.intersection({"AUTHORSHIP_SUPPRESSION", "TRANSFORMATION_DETEMPLATE", "HYBRID_TEXTURE_COLLAPSE", "LOW_VALUE_REMOVE"}),
    "authorship/transformation texture controller maps and labels targeted candidate families",
)
safe_partial_stop_reason = _ai_search_adaptive_stop_reason(
    {
        "selectable": True,
        "safe_partial_quality_improvement": True,
        "ai_footprint_gate": cleanup_gate,
        "dominant_blocker_gate": {"required": True, "cleared": False},
        "authenticity_gate": {
            "human_delta": 0.0,
            "ai_authorship_delta": 1.0,
            "ai_transformation_delta": 0.0,
        },
    },
    phase="claim_narrowing",
    short_document=False,
)
assert_test(
    safe_partial_stop_reason == "",
    "adaptive stop does not stop on cleanup-only wins while AI-footprint drivers are unchanged",
)
partial_footprint_stop_reason = _ai_search_adaptive_stop_reason(
    {
        "selectable": True,
        "partial_ai_footprint_mitigation": True,
        "ai_footprint_gate": partial_gate,
    },
    phase="claim_narrowing",
    short_document=False,
)
assert_test(
    partial_footprint_stop_reason == "adaptive_stop_after_ai_footprint_claim_narrowing",
    "adaptive stop can stop after real AI-footprint movement",
)
meaningful_search_status = _ai_search_candidate_selection_status(57.78, 52.50, True)
assert_test(
    meaningful_search_status["selectable"],
    "AI search selects candidates only after the required AI drop is met",
)
negative_shift_rank = _goal_climb_candidate_rank(
    {
        "selectable": True,
        "human_shift_score": -0.5,
        "authenticity_gate": {
            "candidate_human": 65,
            "human_delta": 1,
            "ai_authorship_delta": 1,
            "ai_transformation_delta": 1,
        },
        "human_shift_components": {"semantic_uniformity_reduction": -3.0},
    },
    {"human_contribution": 65},
    candidate_ai=46.2,
    candidate_review_burden=4,
    candidate_weighted_severity=52,
    candidate_finding_total=45,
    original_review_burden=5,
    original_weighted_severity=57,
    original_finding_total=49,
)
positive_shift_rank = _goal_climb_candidate_rank(
    {
        "selectable": True,
        "human_shift_score": 4.1,
        "authenticity_gate": {
            "candidate_human": 65,
            "human_delta": 1,
            "ai_authorship_delta": 1,
            "ai_transformation_delta": 1,
        },
        "human_shift_components": {"rewrite_smoothness_reduction": 1.4},
    },
    {"human_contribution": 65},
    candidate_ai=46.8,
    candidate_review_burden=2,
    candidate_weighted_severity=54,
    candidate_finding_total=49,
    original_review_burden=5,
    original_weighted_severity=57,
    original_finding_total=49,
)
assert_test(
    positive_shift_rank > negative_shift_rank,
    "goal-climb selector ranks positive Human Shift above lower-AI negative-shift candidates",
)

def make_shift_report(
    *,
    ai_authorship,
    human,
    ai_transformation,
    grounding,
    human_anchor,
    smoothness,
    semantic_uniformity,
    ai_likelihood=0,
    expansion=0,
    discourse=0,
    section_style=0,
    source_similarity=0,
    topk_pattern=0,
    predictability=0,
    generic_assertion_risk=0,
    unsupported_claim_risk=0,
    broad_claim_risk=0,
    source_grounding_risk=0,
    high_count=0,
):
    return {
        "integrity_layers": {
            "layers": {
                "ai_authorship_risk": {"score": ai_authorship},
                "human_contribution_signal": {"score": human},
                "ai_transformation_risk": {"score": ai_transformation},
                "grounding_quality_risk": {"score": grounding},
            }
        },
        "ai_risk_badge": {
            "transformation_classification": {
                "features": {
                    "human_anchor_score": human_anchor / 100,
                    "rewrite_smoothness": smoothness / 100,
                    "ai_likelihood": ai_likelihood / 100,
                    "outline_to_text_expansion": expansion / 100,
                    "semantic_uniformity_risk": semantic_uniformity / 100,
                    "discourse_regularity_risk": discourse / 100,
                    "section_style_variance": section_style / 100,
                    "source_similarity": source_similarity / 100,
                }
            },
            "ai_likelihood_score": ai_likelihood,
            "ai_components": {
                "topk_pattern": topk_pattern,
                "predictability": predictability,
                "generic_assertion_risk": generic_assertion_risk,
            },
            "writing_components": {
                "unsupported_claim_risk": unsupported_claim_risk,
                "broad_claim_risk": broad_claim_risk,
                "source_grounding_risk": source_grounding_risk,
            },
        },
        "findings": {
            "critical": [],
            "high": [{"finding_id": f"h{i}"} for i in range(high_count)],
            "medium": [],
            "low": [],
        },
    }

shift_original = make_shift_report(
    ai_authorship=59,
    human=67,
    ai_transformation=33,
    grounding=25,
    human_anchor=61,
    smoothness=46,
    semantic_uniformity=46,
)
shift_candidate = make_shift_report(
    ai_authorship=48,
    human=69,
    ai_transformation=31,
    grounding=23,
    human_anchor=64,
    smoothness=42,
    semantic_uniformity=40,
)
shift_regression = make_shift_report(
    ai_authorship=47,
    human=65,
    ai_transformation=35,
    grounding=45,
    human_anchor=58,
    smoothness=60,
    semantic_uniformity=60,
)
positive_shift = _human_shift_score(shift_original, shift_candidate, drift_similarity=0.97)
negative_shift = _human_shift_score(shift_original, shift_regression, drift_similarity=0.97)
assert_test(
    positive_shift["score"] > 15,
    "Human Shift Score rewards AI-authorship reduction plus human-side signal movement",
)
assert_test(
    negative_shift["score"] < positive_shift["score"],
    "Human Shift Score penalizes grounding, smoothness, and semantic-uniformity regression",
)
authenticity_gate = _authenticity_gate_status(
    shift_original,
    shift_candidate,
    True,
    original_review_burden=2,
    candidate_review_burden=2,
    original_weighted_severity=4,
    candidate_weighted_severity=4,
)
assert_test(
    not authenticity_gate["success"]
    and authenticity_gate["candidate_progress"]
    and authenticity_gate["reason"] == "candidate_progress_below_target"
    and authenticity_gate["human_shift_score"] == positive_shift["score"],
    "target-aware authenticity gate records positive movement as candidate progress below target",
)
authenticity_regression_gate = _authenticity_gate_status(
    shift_original,
    make_shift_report(
        ai_authorship=48,
        human=69,
        ai_transformation=31,
        grounding=23,
        human_anchor=64,
        smoothness=42,
        semantic_uniformity=40,
        high_count=1,
    ),
    True,
    original_review_burden=2,
    candidate_review_burden=2,
    original_weighted_severity=4,
    candidate_weighted_severity=4,
)
assert_test(
    not authenticity_regression_gate["success"]
    and authenticity_regression_gate["reason"] == "critical_high_regressed",
    "authenticity gate still rejects candidates that add critical/high findings",
)
negative_shift_gate = _authenticity_gate_status(
    shift_original,
    shift_regression,
    True,
    original_review_burden=2,
    candidate_review_burden=2,
    original_weighted_severity=4,
    candidate_weighted_severity=4,
)
assert_test(
    not negative_shift_gate["success"]
    and negative_shift_gate["reason"] == "human_target_regressed",
    "authenticity gate rejects AI drops that are outweighed by human-side regressions",
)
target_regression_gate = _authenticity_gate_status(
    make_shift_report(
        ai_authorship=48,
        human=72,
        ai_transformation=28,
        grounding=50,
        human_anchor=58,
        smoothness=46,
        semantic_uniformity=46,
    ),
    make_shift_report(
        ai_authorship=46,
        human=70,
        ai_transformation=30,
        grounding=44,
        human_anchor=58,
        smoothness=45,
        semantic_uniformity=30,
    ),
    True,
    original_review_burden=9,
    candidate_review_burden=9,
    original_weighted_severity=33,
    candidate_weighted_severity=33,
)
assert_test(
    not target_regression_gate["success"]
    and not target_regression_gate["candidate_progress"]
    and target_regression_gate["reason"] == "human_target_regressed"
    and target_regression_gate["human_target_regressed"]
    and target_regression_gate["ai_transformation_target_regressed"],
    "authenticity gate blocks authorship-only wins that move away from the Human 80 target",
)
target_regression_selection_block = _human_target_regression_selection_block(
    {
        "selectable": True,
        "reason": "accepted_incremental_authenticity_progress",
    },
    {
        **target_regression_gate,
        "candidate_human": 70,
        "human_delta": -2,
        "ai_transformation_delta": -2,
    },
    target_human=80,
)
assert_test(
    target_regression_selection_block["blocked"]
    and target_regression_selection_block["reason"] == "human_target_regressed"
    and target_regression_selection_block["human_target_guard_required"],
    "selector-level Human target guard blocks fallback acceptance when Human and AI Transformation regress",
)
no_target_progress_gate = _authenticity_gate_status(
    make_shift_report(
        ai_authorship=48,
        human=72,
        ai_transformation=28,
        grounding=50,
        human_anchor=58,
        smoothness=46,
        semantic_uniformity=46,
    ),
    make_shift_report(
        ai_authorship=46,
        human=72,
        ai_transformation=28,
        grounding=44,
        human_anchor=58,
        smoothness=45,
        semantic_uniformity=30,
    ),
    True,
    original_review_burden=9,
    candidate_review_burden=9,
    original_weighted_severity=33,
    candidate_weighted_severity=33,
)
assert_test(
    not no_target_progress_gate["success"]
    and not no_target_progress_gate["candidate_progress"]
    and no_target_progress_gate["reason"] == "no_human_target_progress",
    "authenticity gate blocks below-target candidates that do not increase Human or reduce AI Transformation",
)
formula_safe_partial_gate = _human_formula_driver_status(
    make_shift_report(
        ai_authorship=41,
        human=55,
        ai_transformation=45,
        grounding=51,
        human_anchor=60,
        smoothness=60,
        semantic_uniformity=60,
        ai_likelihood=60,
        expansion=40,
        discourse=50,
    ),
    make_shift_report(
        ai_authorship=37,
        human=57,
        ai_transformation=43,
        grounding=39,
        human_anchor=60,
        smoothness=58,
        semantic_uniformity=57,
        ai_likelihood=56,
        expansion=38,
        discourse=49,
    ),
)
assert_test(
    formula_safe_partial_gate["cleared"]
    and formula_safe_partial_gate["safe_partial_progress"]
    and formula_safe_partial_gate["human_delta"] == 2,
    "Human formula gate accepts safe partial progress when formula drivers improve below the 80 target",
)
major_gate_env = {
    "DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_THRESHOLD": os.environ.get("DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_THRESHOLD"),
    "DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_GAIN": os.environ.get("DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_GAIN"),
}
os.environ["DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_THRESHOLD"] = "80"
os.environ["DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_GAIN"] = "10"
ai_authorship_regressing_candidate = make_shift_report(
    ai_authorship=62,
    human=72,
    ai_transformation=35,
    grounding=20,
    human_anchor=85,
    smoothness=20,
    semantic_uniformity=20,
)
ai_authorship_regression_gate = _authenticity_gate_status(
    shift_original,
    ai_authorship_regressing_candidate,
    True,
    original_review_burden=2,
    candidate_review_burden=2,
    original_weighted_severity=4,
    candidate_weighted_severity=4,
)
assert_test(
    not ai_authorship_regression_gate["success"]
    and ai_authorship_regression_gate["reason"] == "ai_authorship_regressed"
    and ai_authorship_regression_gate["ai_authorship_regression_blocked"]
    and ai_authorship_regression_gate["human_gain_with_authorship_regression"]
    and ai_authorship_regression_gate["false_positive_improvement"],
    "authenticity gate labels Human gains that worsen AI Authorship as false-positive improvement",
)
major_breakthrough_gate = _authenticity_gate_status(
    shift_original,
    make_shift_report(
        ai_authorship=61,
        human=82,
        ai_transformation=18,
        grounding=25,
        human_anchor=85,
        smoothness=30,
        semantic_uniformity=25,
    ),
    True,
    original_review_burden=2,
    candidate_review_burden=2,
    original_weighted_severity=4,
    candidate_weighted_severity=4,
)
assert_test(
    not major_breakthrough_gate["success"]
    and major_breakthrough_gate["major_human_breakthrough"]
    and major_breakthrough_gate["ai_authorship_regression_blocked"],
    "authenticity gate hard-rejects AI Authorship regression even after major Human breakthrough",
)
section_baseline_scores = {
    "ai_score": 64.0,
    "human": 33,
    "ai_transformation": 67,
    "ai_authorship": 64,
    "grounding": 72,
    "findings": 11,
}
drifty_candidate = {
    "strategy": "anchor_dense_compressed",
    "mechanical": {"passed": True},
    "scan_scores": {
        "ai_score": 51.79,
        "human": 52,
        "ai_transformation": 48,
        "ai_authorship": 52,
        "grounding": 56,
        "findings": 8,
        "semantic_drift": True,
        "generic_phrase_count": 1,
    },
}
stable_candidate = {
    "strategy": "recommended_two_pass",
    "mechanical": {"passed": True},
    "scan_scores": {
        "ai_score": 57.78,
        "human": 53,
        "ai_transformation": 47,
        "ai_authorship": 58,
        "grounding": 56,
        "findings": 9,
        "semantic_drift": False,
        "generic_phrase_count": 1,
    },
}
invalid_candidate = {
    "strategy": "missing_anchor",
    "mechanical": {"passed": False, "missing": ["SHBHCUT006"]},
    "scan_scores": {
        "ai_score": 40,
        "human": 70,
        "ai_transformation": 30,
        "ai_authorship": 40,
        "grounding": 40,
        "findings": 3,
        "semantic_drift": False,
        "generic_phrase_count": 0,
    },
}
drifty_status = _optimization_candidate_status(
    drifty_candidate,
    baseline=section_baseline_scores,
)
assert_test(
    not drifty_status["accepted"]
    and "semantic_drift" in drifty_status["reject_reasons"],
    "optimization selector rejects strong AI reduction when semantic drift appears",
)
selection = _select_best_optimization_candidate(
    [invalid_candidate, drifty_candidate, stable_candidate],
    baseline=section_baseline_scores,
)
assert_test(
    selection["selected"]
    and selection["selected"]["strategy"] == "recommended_two_pass"
    and selection["accepted_count"] == 1,
    "optimization selector chooses best Pareto-valid candidate after hard rejects",
)
assert_test(
    selection["selected"]["optimization_status"]["components"]["human_gain"] == 20,
    "optimization selector records candidate score components",
)
high_ai_drop_candidate = {
    "strategy": "ai_score_drop_but_low_human",
    "mechanical": {"passed": True},
    "scan_scores": {
        "ai_score": 45,
        "human": 55,
        "ai_transformation": 45,
        "ai_authorship": 45,
        "grounding": 56,
        "findings": 8,
        "semantic_drift": False,
        "generic_phrase_count": 0,
    },
}
high_human_candidate = {
    "strategy": "human_gain_repair",
    "mechanical": {"passed": True},
    "scan_scores": {
        "ai_score": 57,
        "human": 63,
        "ai_transformation": 37,
        "ai_authorship": 55,
        "grounding": 58,
        "findings": 8,
        "semantic_drift": False,
        "generic_phrase_count": 0,
    },
}
authorship_regression_candidate = {
    "strategy": "semantic_human_gain_but_authorship_regressed",
    "mechanical": {"passed": True},
    "scan_scores": {
        "ai_score": 62,
        "human": 70,
        "ai_transformation": 30,
        "ai_authorship": 70,
        "grounding": 58,
        "findings": 8,
        "semantic_drift": False,
        "generic_phrase_count": 0,
    },
}
human_first_selection = _select_best_optimization_candidate(
    [authorship_regression_candidate, high_ai_drop_candidate, high_human_candidate],
    baseline=section_baseline_scores,
)
assert_test(
    human_first_selection["selected"]
    and human_first_selection["selected"]["strategy"] == "human_gain_repair",
    "optimization selector prioritizes human-stage gain over lower AI score when both are drift-safe",
)
authorship_regression_status = _optimization_candidate_status(
    authorship_regression_candidate,
    baseline=section_baseline_scores,
)
assert_test(
    not authorship_regression_status["accepted"]
    and "ai_authorship_increase" in authorship_regression_status["reject_reasons"],
    "optimization selector hard-rejects Human gain when AI Authorship increases",
)
anchor_mapping = _anchor_lock_mapping(["SHBHCUT002", "SHBHCUT003", "III", "SHBHCUT002"])
frozen_anchor_text = _freeze_anchor_text(
    "Keep SHBHCUT002 and SHBHCUT003 unchanged. Certificate III remains readable.",
    anchor_mapping,
)
assert_test(
    "[[DP_ANCHOR_001]]" in frozen_anchor_text
    and "SHBHCUT002" not in frozen_anchor_text
    and "Certificate III" in frozen_anchor_text,
    "anchor lock freezes long protected anchors without freezing short ambiguous anchors",
)
assert_test(
    _restore_anchor_placeholders(frozen_anchor_text, anchor_mapping)
    == "Keep SHBHCUT002 and SHBHCUT003 unchanged. Certificate III remains readable.",
    "anchor lock restores placeholders exactly after texture repair",
)
numeric_anchor_mapping = _anchor_lock_mapping(["0", "1", "7", "2017"])
numeric_frozen = _freeze_anchor_text(
    "0 to 180 degrees, 1 to 90 degrees, 7 procedures, CESE 2017",
    numeric_anchor_mapping,
)
assert_test(
    _restore_anchor_placeholders(numeric_frozen, numeric_anchor_mapping)
    == "0 to 180 degrees, 1 to 90 degrees, 7 procedures, CESE 2017"
    and "201[[DP_ANCHOR" not in numeric_frozen
    and numeric_frozen.count("[[DP_ANCHOR_") >= 4,
    "anchor lock freezes standalone numeric anchors without corrupting years or codes",
)
frozen_payload = _freeze_anchor_payload(
    {"anchors": ["SHBHCUT002"], "body": "SHBHCUT003 appears in context."},
    anchor_mapping,
)
assert_test(
    json.dumps(frozen_payload).count("SHBHCUT") == 0,
    "anchor lock freezes nested section context payloads",
)
assert_test(
    _model_capabilities("openai/gpt-4.1-mini").get("top_k") is False
    and _model_capabilities("qwen/qwen3-32b").get("top_k") is True,
    "model capability normalization disables top_k for OpenAI models only",
)
sampling_env_keys = [
    "DRAFTPROOF_AI_SEARCH_TOP_P",
    "DRAFTPROOF_AI_SEARCH_PRESENCE_PENALTY",
    "DRAFTPROOF_AI_SEARCH_FREQUENCY_PENALTY",
    "DRAFTPROOF_DRIVER_SUPPRESSION_TOP_P",
]
previous_sampling_env = {key: os.environ.get(key) for key in sampling_env_keys}
for key in sampling_env_keys:
    os.environ.pop(key, None)
default_sampling = _rewrite_sampling_profile("DRAFTPROOF_AI_SEARCH")
assert_test(
    default_sampling["top_p"] == 0.82
    and default_sampling["presence_penalty"] == 0.15
    and default_sampling["frequency_penalty"] == 0.25,
    "rewrite sampling profile does not collapse to temperature-only defaults",
)
assert_test(
    _phase_sampling_arg("DRAFTPROOF_DRIVER_SUPPRESSION", "TOP_P") == 0.82,
    "phase sampling inherits mitigation top_p when phase env is unset",
)
os.environ["DRAFTPROOF_DRIVER_SUPPRESSION_TOP_P"] = "0.91"
assert_test(
    _phase_sampling_arg("DRAFTPROOF_DRIVER_SUPPRESSION", "TOP_P") == 0.91,
    "phase sampling env overrides inherited mitigation top_p",
)
phase_kwargs = _phase_chat_sampling_kwargs(
    "DRAFTPROOF_AI_SEARCH",
    temperature_env="DRAFTPROOF_AI_SEARCH_TEMPERATURE",
    temperature_default=0.45,
    max_tokens_env="DRAFTPROOF_AI_SEARCH_MAX_TOKENS",
    max_tokens_default=6500,
)
assert_test(
    phase_kwargs["top_p"] == 0.82
    and phase_kwargs["presence_penalty"] == 0.15
    and phase_kwargs["frequency_penalty"] == 0.25,
    "chat generation kwargs include non-temperature sampling controls by default",
)
captured_llm_payload = {}
original_requests_post = llm_gateway_module.requests.post


class _FakeLLMHTTPResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "model": "openai/gpt-4.1-mini",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 1},
        }


def _fake_requests_post(url, headers=None, json=None, timeout=None):
    captured_llm_payload.update(json or {})
    return _FakeLLMHTTPResponse()


try:
    llm_gateway_module.requests.post = _fake_requests_post
    LLMGateway(LLMConfig(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4.1-mini",
        max_retries=1,
    )).chat("test", **phase_kwargs)
finally:
    llm_gateway_module.requests.post = original_requests_post
assert_test(
    captured_llm_payload.get("top_p") == 0.82
    and captured_llm_payload.get("presence_penalty") == 0.15
    and captured_llm_payload.get("frequency_penalty") == 0.25
    and "top_k" not in captured_llm_payload,
    "OpenAI gateway payload sends normalized sampling controls and strips unsupported top_k",
)
policy_summary = _mitigation_sampling_policy_summary()
assert_test(
    policy_summary["ai_search_top_p"] == 0.82
    and policy_summary["ai_search_presence_penalty"] == 0.15
    and policy_summary["ai_search_frequency_penalty"] == 0.25,
    "mitigation sampling policy summary exposes effective non-temperature controls",
)
assert_test(
    not _llm_call_budget_exhausted_before_send(2, 2)
    and _llm_call_budget_exhausted_before_send(3, 2),
    "LLM budget guard allows the final permitted call after optimistic pre-send increment",
)
old_hard_cap = os.environ.get("DRAFTPROOF_AI_SEARCH_HARD_MAX_LLM_CALLS")
try:
    os.environ.pop("DRAFTPROOF_AI_SEARCH_HARD_MAX_LLM_CALLS", None)
    short_policy = _ai_search_budget_policy("word " * 300)
    medium_policy = _ai_search_budget_policy("word " * 900)
    saturated_medium_policy = _ai_search_budget_policy("word " * 900, saturated_topk_report)
    long_policy = _ai_search_budget_policy("word " * 2200, saturated_topk_report)
    default_hard_cap = _ai_search_llm_hard_cap("word " * 900, saturated_topk_report)
    os.environ["DRAFTPROOF_AI_SEARCH_HARD_MAX_LLM_CALLS"] = "10"
    explicit_hard_cap = _ai_search_llm_hard_cap("word " * 900)
finally:
    if old_hard_cap is None:
        os.environ.pop("DRAFTPROOF_AI_SEARCH_HARD_MAX_LLM_CALLS", None)
    else:
        os.environ["DRAFTPROOF_AI_SEARCH_HARD_MAX_LLM_CALLS"] = old_hard_cap
assert_test(
    short_policy["max_llm_calls"] == 6
    and short_policy["max_candidate_scans"] == 4
    and short_policy["candidate_scoring_controller"]["policy"] == "verified_finalist_full_scans"
    and medium_policy["max_llm_calls"] == 10
    and medium_policy["max_candidate_scans"] == 6
    and saturated_medium_policy["max_llm_calls"] == 15
    and saturated_medium_policy["max_candidate_scans"] == 8
    and saturated_medium_policy["max_candidate_scan_hard_cap"] == 11
    and long_policy["max_llm_calls"] == 17
    and long_policy["max_candidate_scans"] == 12
    and default_hard_cap == 10
    and explicit_hard_cap == 10,
    "AI search budget policy verifies bounded finalists instead of full-scanning raw candidate volume",
)
verified_budget_probe = _verified_candidate_scan_budget("word " * 900, saturated_topk_report)
assert_test(
    verified_budget_probe["max_candidate_scans"] == 8
    and verified_budget_probe["max_candidate_scan_hard_cap"] == 11
    and verified_budget_probe["pressure_bonus"] == 2,
    "verified finalist scan budget scales sublinearly with document size and active driver pressure",
)
scan_budget_probe = {"max_candidate_scans": 8, "max_candidate_scan_hard_cap": 11}
_extend_candidate_scan_budget(scan_budget_probe, 8, 12)
_extend_candidate_scan_budget(scan_budget_probe, 11, 12)
assert_test(
    scan_budget_probe["max_candidate_scans"] == 11,
    "AI search scan reserves cannot grow beyond the size-policy hard cap",
)
medium_topk_prompt = _topk_safe_band_snapshot_prompt(
    "The United States has many strengths and challenges. " * 90,
    {"ai_risk_badge": {"ai_components": {"topk_calibrated_risk": 100.0}}},
)
plain_topk_prompt = _topk_plain_spoken_snapshot_prompt(
    "The United States has many strengths and challenges. " * 90,
    {"ai_risk_badge": {"ai_components": {"topk_calibrated_risk": 100.0}}},
)
assert_test(
    "280 to 560 words" not in medium_topk_prompt
    and "rough annotated prose" not in medium_topk_prompt
    and "short country profile" in medium_topk_prompt
    and "no metaphors" in plain_topk_prompt
    and "plain everyday nouns and verbs" in plain_topk_prompt
    and "patch all listed sentences" in _topk_safe_band_sentence_patch_prompt(
        "The United States has influence. The country also has problems.",
        {"ai_risk_badge": {"ai_components": {"topk_calibrated_risk": 100.0}}},
    )
    and _topk_safe_band_patch_rounds_default("word " * 900, saturated_topk_report) == 10
    and _topk_safe_band_snapshot_max_tokens_default("word " * 900) == 3600,
    "Top-k safe-band rebuild uses document-sized prose, not compressed snapshot fragments",
)
with open(os.path.join(os.path.dirname(__file__), "rewrite_pipeline.py"), "r", encoding="utf-8") as fp:
    rewrite_pipeline_source = fp.read()
assert_test(
    "DRAFTPROOF_TOPK_SAFE_BAND_MAX_CHAR_RATIO" in rewrite_pipeline_source
    and "if topk_safe_band_rebuild or strict_safe_shortening" in rewrite_pipeline_source
    and "effective_max_chars" in rewrite_pipeline_source,
    "Top-k/strict-safe candidates use a relaxed effective length gate before rejection",
)
assert_test(
    "DRAFTPROOF_AI_SEARCH_HARD_MIN_CHAR_RATIO" in rewrite_pipeline_source
    and "length_guidance_warning" in rewrite_pipeline_source
    and "candidate_below_guidance_min" in rewrite_pipeline_source,
    "AI-search length is guidance for substantial candidates, not a near-miss hard rollback",
)
assert_test(
    "DRAFTPROOF_SCAN_GENERATED_CANDIDATES_AFTER_TIME_BUDGET" in rewrite_pipeline_source
    and "scan_generated_candidate_after_budget" in rewrite_pipeline_source,
    "Already-generated high-value candidates get a scanner pass instead of being wasted after time budget trips",
)
assert_test(
    'DRAFTPROOF_TOPK_CAN_BORROW_UNUSED_PHASE_BUDGET", False' in rewrite_pipeline_source
    and "density_or_generic_priority" in rewrite_pipeline_source
    and "original_qualifying_text_ai_density" in rewrite_pipeline_source
    and "qualifying_text_ai_density" in rewrite_pipeline_source
    and "DRAFTPROOF_STRICT_AI_PHASE_BUDGET_ONLY" in rewrite_pipeline_source
    and "strict_ai_phase_budget_only" in rewrite_pipeline_source,
    "Controller budget and selection prioritize density without default Top-k borrowing or legacy LLM spillover",
)
assert_test(
    'def _run_post_safe_win_target_push(trigger_phase: str) -> None:' in rewrite_pipeline_source
    and 'reason": "strict_ai_phase_budget_only"' in rewrite_pipeline_source,
    "Post-safe Human target push cannot bypass strict AI phase-budget-only mode",
)
for key, value in previous_sampling_env.items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
low_aggression = _repair_aggression_score(
    "Learners name the seven cutting steps. The guide can still disappear in their hands.",
    "Learners name the seven cutting steps. Still, the guide can disappear in their hands.",
)
high_aggression = _repair_aggression_score(
    "Learners name the seven cutting steps. The guide can still disappear in their hands.",
    "Inclusive education improves confidence through structured support and reflective practice.",
)
assert_test(
    low_aggression["score"] < high_aggression["score"],
    "repair aggression score separates micro-local texture edits from broad rewrites",
)
texture_map_text = (
    "Learners name the seven cutting steps. "
    "Furthermore, this important approach supports effective skill development in a significant way. "
    "The guide can still disappear in their hands."
)
texture_map = _sentence_texture_risk_map(
    texture_map_text,
    {"rewrite_edit_briefs": [{"sentence_index": 1, "signals": {"finding_type": "medium_predictability"}}]},
)
assert_test(
    texture_map[0]["sentence_index"] == 1
    and texture_map[0]["risk"] > texture_map[-1]["risk"],
    "sentence texture risk map prioritizes scanner-pointed smooth/generic sentence",
)
texture_window = _micro_texture_window(
    texture_map_text,
    {"rewrite_edit_briefs": [{"sentence_index": 1, "signals": {"finding_type": "medium_predictability"}}]},
    max_sentences=1,
)
assert_test(
    texture_window["start"] == 1
    and texture_window["end"] <= 3
    and "Furthermore" in texture_window["text"],
    "micro texture window limits repair to top-risk one or two sentences",
)
heading_window = _micro_texture_window(
    "Title\n\nIntroduction\n\nSmooth heading sentence should not be touched. Furthermore, this important sentence is too polished. Local action remains.",
    {"rewrite_edit_briefs": [{"sentence_index": 0, "signals": {"finding_type": "medium_predictability"}}]},
    max_sentences=1,
)
assert_test(
    heading_window["start"] != 0
    and "Furthermore" in heading_window["text"],
    "micro texture window skips title/heading-contaminated sentence chunks",
)
spliced_texture = _splice_sentence_window(
    texture_map_text,
    texture_window["start"],
    texture_window["end"],
    "That part matters less on paper than it does at the mannequin.",
)
assert_test(
    spliced_texture.startswith("Learners name the seven cutting steps.")
    and spliced_texture.endswith("The guide can still disappear in their hands.")
    and "Furthermore" not in spliced_texture,
    "micro texture splice changes only the selected sentence window",
)
locality = _locality_score(texture_map_text, spliced_texture)
assert_test(
    locality["changed_sentence_ratio"] <= 0.25 or locality["changed_sentences"] == 1,
    "locality score measures changed sentence footprint for micro repairs",
)
micro_prompt, micro_info = _micro_texture_repair_prompt(
    texture_map_text,
    {"rewrite_edit_briefs": [{"sentence_index": 1, "signals": {"finding_type": "medium_predictability"}}]},
    ["SHBHCUT002"],
    max_sentences=1,
)
assert_test(
    "Patch only the target sentence window" in micro_prompt
    and "Return only the replacement sentence window" in micro_prompt
    and "reorder_paragraph" in micro_prompt
    and "target_window" in micro_prompt,
    "micro texture repair prompt is operation-level, not section-level",
)
clean_micro, clean_micro_reason = _clean_micro_texture_candidate(
    "That part matters less on paper than it does at the mannequin.",
    micro_info,
)
assert_test(
    clean_micro
    and not clean_micro_reason,
    "micro texture candidate cleaner accepts bounded replacement window",
)
masked_prompt, masked_info = _masked_span_repair_prompt(
    "Furthermore, this important approach supports effective skill development. Local action remains.",
    {
        "rewrite_edit_briefs": [{
            "sentence_index": 0,
            "problem_tokens": ["Furthermore"],
            "signals": {"generic": "generic connector"},
        }]
    },
)
masked_replacement = _clean_masked_span_replacement("But")
masked_text = _apply_masked_span_replacement(
    "Furthermore, this important approach supports effective skill development. Local action remains.",
    masked_info,
    masked_replacement,
)
masked_paragraph_text = _apply_masked_span_replacement(
    "Furthermore, this important approach supports effective skill development.\n\nLocal action remains.",
    masked_info,
    masked_replacement,
)
assert_test(
    "[[MASK]]" in masked_prompt
    and masked_info["mask_text"].lower() == "furthermore"
    and masked_text.startswith("But, this important approach")
    and "Local action remains." in masked_text,
    "masked span repair masks only the high-risk span and preserves surrounding text",
)
assert_test(
    masked_paragraph_text.startswith("But, this important approach")
    and "\n\nLocal action remains." in masked_paragraph_text,
    "masked span repair preserves paragraph breaks when patching an exact sentence",
)
aggressive_status = _optimization_candidate_status(
    {
        "strategy": "authorship_texture_repair",
        "mechanical": {"passed": True},
        "original_text": "Learners name the seven cutting steps. The guide can still disappear in their hands.",
        "candidate_text": "Inclusive education improves confidence through structured support and reflective practice.",
        "scan_scores": {
            "ai_score": 52,
            "human": 50,
            "ai_transformation": 50,
            "ai_authorship": 50,
            "grounding": 58,
            "findings": 8,
            "semantic_drift": False,
            "generic_phrase_count": 0,
        },
    },
    baseline=section_baseline_scores,
)
assert_test(
    not aggressive_status["accepted"]
    and "repair_aggression_high" in aggressive_status["reject_reasons"]
    and "repair_locality_high" in aggressive_status["reject_reasons"],
    "optimization selector hard-rejects over-aggressive and non-local texture repair",
)
micro_policy_env_names = [
    "DRAFTPROOF_MICRO_TEXTURE_MAX_TOTAL_AGGRESSION",
    "DRAFTPROOF_MICRO_TEXTURE_MIN_HUMAN_GAIN",
    "DRAFTPROOF_MICRO_TEXTURE_MIN_GAIN_EFFICIENCY",
    "DRAFTPROOF_MICRO_TEXTURE_MAX_ITERATIONS",
    "DRAFTPROOF_TEXTURE_REPAIR_MAX_LOCALITY",
]
micro_policy_env = {name: os.environ.get(name) for name in micro_policy_env_names}
os.environ["DRAFTPROOF_MICRO_TEXTURE_MAX_TOTAL_AGGRESSION"] = "0.18"
os.environ["DRAFTPROOF_MICRO_TEXTURE_MIN_HUMAN_GAIN"] = "1"
os.environ["DRAFTPROOF_MICRO_TEXTURE_MIN_GAIN_EFFICIENCY"] = "10"
os.environ["DRAFTPROOF_MICRO_TEXTURE_MAX_ITERATIONS"] = "5"
os.environ["DRAFTPROOF_TEXTURE_REPAIR_MAX_LOCALITY"] = "0.25"
micro_baseline_scan = {"human": 40, "ai_transformation": 60, "ai_authorship": 50, "findings": 6}
micro_attempts = [
    {
        "repair_aggression": {"score": 0.034},
        "locality": {"changed_sentence_ratio": 0.091},
        "scan_scores": {"human": 46, "ai_transformation": 54, "ai_authorship": 50, "findings": 6},
    }
]
micro_status = _micro_texture_iteration_status(
    micro_attempts,
    baseline_scan=micro_baseline_scan,
)
assert_test(
    micro_status["continue"]
    and micro_status["metrics"]["cumulative_aggression"] == 0.034
    and micro_status["metrics"]["gain_efficiency"] > 170,
    "micro texture iteration policy accepts high-efficiency low-aggression human gain",
)
budget_status = _micro_texture_iteration_status(
    micro_attempts + [
        {
            "repair_aggression": {"score": 0.151},
            "locality": {"changed_sentence_ratio": 0.18},
            "scan_scores": {"human": 49, "ai_transformation": 51, "ai_authorship": 50, "findings": 6},
        }
    ],
    baseline_scan=micro_baseline_scan,
)
assert_test(
    not budget_status["continue"]
    and "cumulative_aggression_budget_exhausted" in budget_status["stop_reasons"],
    "micro texture iteration policy stops when cumulative aggression budget is exhausted",
)
authorship_stop = _micro_texture_iteration_status(
    micro_attempts + [
        {
            "repair_aggression": {"score": 0.02},
            "locality": {"changed_sentence_ratio": 0.12},
            "scan_scores": {"human": 48, "ai_transformation": 52, "ai_authorship": 51, "findings": 6},
        }
    ],
    baseline_scan=micro_baseline_scan,
)
assert_test(
    not authorship_stop["continue"]
    and "ai_authorship_regression" in authorship_stop["stop_reasons"],
    "micro texture iteration policy stops on AI Authorship regression",
)
findings_stop = _micro_texture_iteration_status(
    micro_attempts + [
        {
            "repair_aggression": {"score": 0.02},
            "locality": {"changed_sentence_ratio": 0.12},
            "scan_scores": {"human": 48, "ai_transformation": 52, "ai_authorship": 50, "findings": 7},
        }
    ],
    baseline_scan=micro_baseline_scan,
)
assert_test(
    not findings_stop["continue"]
    and "findings_regression" in findings_stop["stop_reasons"],
    "micro texture iteration policy stops when scanner findings increase",
)
diminishing_stop = _micro_texture_iteration_status(
    micro_attempts + [
        {
            "repair_aggression": {"score": 0.05},
            "locality": {"changed_sentence_ratio": 0.12},
            "scan_scores": {"human": 46.2, "ai_transformation": 53.8, "ai_authorship": 50, "findings": 6},
        }
    ],
    baseline_scan=micro_baseline_scan,
)
assert_test(
    not diminishing_stop["continue"]
    and "diminishing_human_gain" in diminishing_stop["stop_reasons"]
    and _micro_repair_gain_efficiency(6, 0.034) > _micro_repair_gain_efficiency(0.2, 0.05),
    "micro texture iteration policy detects diminishing marginal gain efficiency",
)
os.environ["DRAFTPROOF_MICRO_TEXTURE_MAX_TOTAL_AGGRESSION"] = "1.0"
iterative_source = (
    "Learners name the seven cutting steps. "
    "Furthermore, this important approach supports effective skill development in a significant way. "
    "Additionally, this important process supports effective technical progress in a significant way. "
    "The guide can still disappear in their hands."
)
iterative_raw = {
    "rewrite_edit_briefs": [
        {"sentence_index": 1, "signals": {"finding_type": "medium_predictability"}},
        {"sentence_index": 2, "signals": {"finding_type": "medium_predictability"}},
    ]
}
iterative_scans = [
    {"human": 46, "ai_transformation": 54, "ai_authorship": 50, "findings": 6},
    {"human": 49, "ai_transformation": 51, "ai_authorship": 50, "findings": 6},
]

def _test_micro_generate(_prompt, repair_info, _attempt_index):
    target = (repair_info.get("window") or {}).get("text") or ""
    if target.startswith("Furthermore"):
        return "That part matters less on paper than it does at the mannequin."
    return "It still gets messy when the hands have to follow the guide."

def _test_micro_scan(_candidate_text):
    return iterative_scans.pop(0)

iterative_loop = _iterative_micro_texture_repair(
    iterative_source,
    iterative_raw,
    baseline_scan=micro_baseline_scan,
    generate_replacement=_test_micro_generate,
    scan_candidate=_test_micro_scan,
    max_attempts=2,
)
assert_test(
    iterative_loop["accepted_count"] == 2
    and iterative_loop["scan_scores"]["human"] == 49
    and iterative_loop["repaired_sentence_indexes"] == [1, 2]
    and "Furthermore" not in iterative_loop["text"]
    and "Additionally" not in iterative_loop["text"],
    "iterative micro texture loop repairs separate high-risk windows under policy control",
)
for name, value in micro_policy_env.items():
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
assert_test(
    _human_gain_stage_target(53) == 60
    and _human_gain_stage_target(63) == 70
    and _human_gain_stage_target(73) == 80,
    "human gain repair uses 60/70/80 threshold ladder",
)
assert_test(
    not _is_better_human_shift_candidate(
        {"success": False, "candidate_human": 63, "human_delta": 30, "human_shift_score": 50, "ai_authorship_delta": -1, "ai_transformation_delta": 12},
        {"success": False, "candidate_human": 55, "human_delta": 22, "human_shift_score": 70, "ai_authorship_delta": 15, "ai_transformation_delta": 22},
    ),
    "live pipeline best-attempt ranking does not prefer human gain when AI Authorship regresses",
)
drift_diagnosis = _metric_repair_diagnosis({
    "human": 52,
    "ai_transformation": 48,
    "ai_authorship": 52,
    "semantic_drift": True,
})
assert_test(
    drift_diagnosis["repair_type"] == "semantic_drift_rollback",
    "metric repair diagnosis prioritizes drift rollback over score gains",
)
authorship_diagnosis = _metric_repair_diagnosis({
    "human": 53,
    "ai_transformation": 47,
    "ai_authorship": 58,
    "ai_score": 57.78,
    "grounding": 56,
    "findings": 9,
    "semantic_drift": False,
    "generic_phrase_count": 1,
})
assert_test(
    authorship_diagnosis["repair_type"] == "authorship_texture_repair"
    and any("statistical smoothness" in item or "sentence rhythm" in item for item in authorship_diagnosis["instructions"]),
    "metric repair diagnosis targets AUTHORSHIP_TEXTURE_REPAIR before semantic human gain",
)
candidate_diagnostics = _generation_candidate_diagnostics([
    {
        "attempt": 1,
        "strategy": "authorship_texture_repair",
        "reconstruction": True,
        "passed_local_checks": True,
        "reason": "candidate_word_count_too_low 420<500",
        "candidate_word_count": 420,
        "staged_generation": {
            "enabled": True,
            "llm_calls": 3,
            "assembled_word_count": 420,
            "source_draft_included": False,
            "sections": [{"heading": "Body", "actual_words": 420}],
        },
    },
    {
        "attempt": 2,
        "strategy": "human_gain_repair",
        "reconstruction": True,
        "passed_local_checks": True,
        "gate": {
            "success": False,
            "reason": "ai_authorship_regression",
            "ai_authorship_regression_blocked": True,
        },
        "human_contribution": 50,
        "ai_authorship": 72,
        "human_delta": 10,
        "ai_authorship_delta": -4,
    },
])
assert_test(
    candidate_diagnostics["candidate_count"] == 2
    and candidate_diagnostics["reason_counts"]["candidate_word_count_too_low"] == 1
    and candidate_diagnostics["reason_counts"]["ai_authorship_regression"] == 1
    and candidate_diagnostics["candidates"][0]["staged_generation"]["source_draft_included"] is False
    and candidate_diagnostics["candidates"][1]["gate_ai_authorship_regression_blocked"] is True,
    "generation candidate diagnostics expose failed reconstruction and gate reasons without source draft text",
)
for name, value in major_gate_env.items():
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value

reconstruction_source = (
    "Inclusive learning design in Certificate III Hairdressing needs to protect technical standards.\n\n"
    "Learners can repeat haircut terminology but still struggle when they move from mannequin practice to live client sectioning. "
    "According to Smith (2023), \"what students know,\" must be checked through practice. "
    "This shows that support is important for all learners."
)
reconstruction_raw = {
    "integrity_layers": {
        "layers": {
            "ai_authorship_risk": {"score": 68},
            "human_contribution_signal": {"score": 27},
            "ai_transformation_risk": {"score": 73},
            "grounding_quality_risk": {"score": 72},
        }
    },
    "findings": {
        "high": [{
            "title": "semantic_uniformity",
            "evidence": "This shows that support is important for all learners.",
            "recommendation": "Rebuild the paragraph around a concrete reasoning move.",
        }],
        "medium": [],
        "low": [],
        "critical": [],
    },
    "ai_mitigation": {
        "target_segments": [{
            "segment_id": "s002",
            "paragraph_id": "p001",
            "text": "This shows that support is important for all learners.",
            "primary_signal": {"key": "generic_assertion_risk", "score": 90},
            "lever": "specificity",
            "bucket": "needs_author_context",
            "action": "Change sentence route using existing paragraph context, not synonym swaps.",
            "auto_apply": True,
        }],
        "component_actions": [{
            "component": "semantic_uniformity_risk",
            "score": 72,
            "action": "Add sharper paragraph roles and section-specific reasoning.",
        }]
    },
    "scan_intelligence": {
        "semantic_shape": {
            "semantic_uniformity_risk": 0.62,
            "paragraph_role_repetition": 0.58,
        },
        "human_contribution_contract": {
            "schema_version": "human_contribution_contract.v1",
            "weak_subsignals": ["causal_reasoning", "source_claim_ownership"],
            "generation_readiness": {
                "target_human_contribution": 80,
                "estimated_auto_reachable_human_contribution": 58,
                "requires_author_input_for_80": True,
            },
            "paragraph_levers": [{
                "paragraph_id": "p001",
                "recommended_role": "source_to_claim_reasoning",
                "rewrite_lever": "Connect source idea to the learner sectioning problem.",
            }],
        },
        "integrity_layers": {
            "layers": {
                "ai_authorship_risk": {
                    "signals": [
                        {"key": "generic_assertion_risk", "score": 90},
                        {"key": "topk_pattern", "score": 87},
                    ]
                }
            }
        },
        "industry_baseline": {
            "schema_version": "industry_baseline.v1",
            "policy": {
                "grounding_is_not_ai_authorship": True,
                "human_noise_is_not_typo_injection": True,
            },
            "layers": {
                "ai_authorship_risk": {
                    "positive_components": [
                        {"key": "token_predictability", "score": 88},
                        {"key": "semantic_uniformity", "score": 62},
                    ],
                    "suppressors": [
                        {"key": "authorship_friction", "score": 21},
                        {"key": "domain_cognition", "score": 48},
                    ],
                    "excludes": ["source_grounding_risk"],
                },
                "human_contribution_signal": {
                    "components": [
                        {"key": "causal_reasoning", "score": 20},
                        {"key": "local_constraint_awareness", "score": 32},
                    ]
                },
            },
        },
    },
    "rewrite_constraints": {
        "allowed_additions": ["step-by-step process descriptions"],
        "preserve_terms": ['"what students know,"'],
        "do_not_add": ["new citation or reference"],
        "rewrite_rule": "Use implied process detail only.",
    },
}
meaning_brief = _build_reconstruction_meaning_brief(reconstruction_source, reconstruction_raw)
assert_test(
    meaning_brief["claims"]
    and any("live client sectioning" in claim for claim in meaning_brief["claims"]),
    "reconstruction meaning brief extracts submitted claims",
)
assert_test(
    '"what students know,"' in meaning_brief["protected_facts"],
    "reconstruction meaning brief preserves protected quoted facts",
)
assert_test(
    meaning_brief["weak_grounding_zones"][0]["signal"] == "semantic_uniformity",
    "reconstruction meaning brief carries scan weak zones",
)
assert_test(
    meaning_brief["word_count_band"]["min_words"] <= meaning_brief["word_count_band"]["source_word_count"] <= meaning_brief["word_count_band"]["max_words"],
    "reconstruction meaning brief includes document word-count band",
)
assert_test(
    any(row.get("key") == "generic_assertion_risk" for row in meaning_brief["integrity_targets"])
    and meaning_brief["target_segments"][0]["segment_id"] == "s002",
    "reconstruction meaning brief carries integrity drivers and target segments",
)
assert_test(
    "step-by-step process descriptions" in meaning_brief["allowed_existing_additions"],
    "reconstruction meaning brief carries scanner-allowed additions",
)
assert_test(
    meaning_brief["human_contribution_contract"]["schema_version"] == "human_contribution_contract.v1",
    "reconstruction meaning brief carries scanner human contribution contract",
)
assert_test(
    meaning_brief["industry_baseline"]["schema_version"] == "industry_baseline.v1"
    and meaning_brief["industry_baseline"]["policy"]["grounding_is_not_ai_authorship"],
    "reconstruction meaning brief carries industry baseline contract",
)
regeneration_blueprint = _build_regeneration_blueprint(
    reconstruction_source,
    reconstruction_raw,
    "claim_narrowing_rebuild",
)
assert_test(
    regeneration_blueprint["schema_version"] == "regeneration_blueprint.v1"
    and regeneration_blueprint["strategy_family"] == "claim_narrowing_rebuild",
    "regeneration blueprint records strategy family",
)
assert_test(
    regeneration_blueprint["paragraph_plans"]
    and regeneration_blueprint["paragraph_plans"][0]["role"] == "narrow_claim",
    "regeneration blueprint creates paragraph-level roles",
)
assert_test(
    "generic_assertion_risk" in regeneration_blueprint["global_driver_targets"],
    "regeneration blueprint targets scanner drivers",
)
assert_test(
    "token_predictability" in regeneration_blueprint["industry_baseline_focus"]["ai_authorship_positive_components"]
    and "authorship_friction" in regeneration_blueprint["industry_baseline_focus"]["authorship_suppressors"],
    "regeneration blueprint carries industry-baseline authorship drivers and suppressors",
)
reconstruction_prompt = _reconstruction_mitigation_prompt(
    reconstruction_source,
    reconstruction_raw,
    reconstruction_raw["ai_mitigation"],
    attempt_index=1,
    strategy="reasoning_dense_reconstruction",
    prior_attempts=[{
        "strategy": "old",
        "human_shift_score": -10.2,
        "reason": "human_shift_score_too_low",
        "candidate_human": 53,
        "human_delta": -16,
        "ai_authorship_delta": -3,
        "ai_transformation_delta": -16,
        "human_shift_components": {
            "human_contribution_gain": -16,
            "ai_authorship_reduction": -3,
            "grounding_risk_reduction": -35,
        },
    }],
)
assert_test(
    "Generate a new document from the scanner context ledger" in reconstruction_prompt
    and "not modification of the submitted prose" in reconstruction_prompt,
    "reconstruction prompt frames task as regeneration rather than modifying submitted content",
)
assert_test(
    "Human Contribution >= 80" in reconstruction_prompt
    and "Return " in reconstruction_prompt
    and " words only" in reconstruction_prompt,
    "reconstruction prompt targets 80 human contribution and enforces word-count band",
)
assert_test(
    "industry-baseline AI Authorship drivers" in reconstruction_prompt
    and "Do not use typo/noise tricks" in reconstruction_prompt
    and "Treat grounding quality as separate from AI authorship" in reconstruction_prompt,
    "reconstruction prompt follows industry-baseline mitigation policy",
)
assert_test(
    "Do not follow the submitted sentence order as a scaffold" in reconstruction_prompt
    and "generation_context_ledger.v1" in reconstruction_prompt
    and "SOURCE DRAFT" not in reconstruction_prompt,
    "reconstruction prompt uses scanner context ledger instead of source draft scaffold",
)
assert_test(
    "Scanner-derived regeneration blueprint" in reconstruction_prompt
    and "candidate must use the 'reasoning_dense_reconstruction' family" in reconstruction_prompt,
    "reconstruction prompt includes staged blueprint and candidate family",
)
assert_test(
    "Do not invent personal observations" in reconstruction_prompt
    and "Narrow unsupported claims" in reconstruction_prompt,
    "reconstruction prompt forbids fabricated grounding and requires narrowing unsupported claims",
)
assert_test(
    "raises Human Contribution but raises AI Authorship will be rejected" in reconstruction_prompt,
    "reconstruction prompt states Human Shift acceptance failure mode",
)
assert_test(
    "human_shift=-10.2" in reconstruction_prompt,
    "reconstruction prompt includes failed Human Shift feedback",
)
assert_test(
    "scanner_gate_feedback.v1" in reconstruction_prompt
    and "Preserve every source-to-claim relation" in reconstruction_prompt
    and "AUTHORSHIP_TEXTURE_REPAIR" in reconstruction_prompt
    and "Do not replace author-owned reasoning" in reconstruction_prompt,
    "reconstruction prompt converts scanner/gate failures into next-generation controls",
)
assert_test(
    "HUMAN_GAIN_REPAIR" in reconstruction_prompt
    and "human_contribution_ladder" in reconstruction_prompt,
    "reconstruction prompt carries human gain repair controls and threshold ladder",
)
assert_test(
    "reduce statistical smoothness through rhythm variance" in reconstruction_prompt,
    "reconstruction prompt carries authorship texture repair controls",
)
reference_source = reconstruction_source + "\n\nReferences\n\nSmith, A. (2023). Inclusive assessment in vocational education. https://example.test/smith\n\nTAFE Victoria. (2024). Reasonable adjustment guide."
reference_entries = _reference_entries_from_text(reference_source)
assert_test(
    len(reference_entries) == 2
    and "Smith, A. (2023)" in reference_entries[0]
    and "TAFE Victoria. (2024)" in reference_entries[1],
    "reference extraction preserves bibliography entries as context ledger data",
)
staged_brief = _build_reconstruction_meaning_brief(reference_source, reconstruction_raw)
staged_blueprint = _build_regeneration_blueprint(reference_source, reconstruction_raw, "reasoning_dense_reconstruction")
staged_ledger = _generation_context_ledger(staged_brief, staged_blueprint)
section_plans = _staged_generation_section_plan(staged_ledger, max_sections=3)
assert_test(
    section_plans
    and "source_preview" not in json.dumps(section_plans)
    and "References" not in [row.get("heading") for row in section_plans],
    "staged section plan excludes source previews and reference heading from LLM-owned sections",
)
section_anchor_ledger = {
    "generation_handoff": {
        "document_profile": {"title": "Test"},
        "section_generation_units": [
            {
                "section_id": "s1",
                "heading": "Practice",
                "target_words": {"ideal": 120},
                "must_preserve_anchors": ["SHBHCUT003"],
                "meaning_inventory": [
                    {"anchors": ["0", "1", "7", "90 degrees"]},
                ],
            }
        ],
    }
}
section_anchor_plan = _staged_generation_section_plan(section_anchor_ledger, max_sections=1)[0]
assert_test(
    all(anchor in section_anchor_plan.get("must_preserve_anchors", []) for anchor in ["SHBHCUT003", "0", "1", "7", "90 degrees"]),
    "staged section plan carries meaning-inventory anchors into anchor lock",
)
section_prompt = _staged_reconstruction_section_prompt(
    staged_ledger,
    {"schema_version": "scanner_gate_feedback.v1", "next_candidate_controls": ["raise Human Contribution"]},
    section_plans[0],
    strategy="reasoning_dense_reconstruction",
    attempt_index=1,
)
assert_test(
    "section_generation_context.v1" in section_prompt
    and "original submitted prose is unavailable by design" in section_prompt
    and "Do not output the section heading" in section_prompt
    and "SOURCE DRAFT" not in section_prompt,
    "staged section prompt generates from bounded context without source draft exposure",
)
human_gain_section_prompt = _staged_reconstruction_section_prompt(
    staged_ledger,
    {"schema_version": "scanner_gate_feedback.v1", "next_candidate_controls": ["HUMAN_GAIN_REPAIR"]},
    section_plans[0],
    strategy="human_gain_repair",
    attempt_index=1,
)
assert_test(
    "HUMAN_GAIN_REPAIR is active" in human_gain_section_prompt
    and "Do not invent new concrete details" in human_gain_section_prompt,
    "staged section prompt activates bounded human gain repair controls",
)
authorship_texture_section_prompt = _staged_reconstruction_section_prompt(
    staged_ledger,
    {"schema_version": "scanner_gate_feedback.v1", "next_candidate_controls": ["AUTHORSHIP_TEXTURE_REPAIR"]},
    {
        **section_plans[0],
        "must_preserve_anchors": ["SHBHCUT002", "SHBHCUT003", "III"],
        "claim_inventory_slice": ["SHBHCUT002 and SHBHCUT003 must stay exact."],
    },
    strategy="authorship_texture_repair",
    attempt_index=1,
)
assert_test(
    "AUTHORSHIP_TEXTURE_REPAIR is active" in authorship_texture_section_prompt
    and "Do not add more semantic human anchors" in authorship_texture_section_prompt
    and "fake randomness" in authorship_texture_section_prompt,
    "staged section prompt activates authorship texture repair controls",
)
assert_test(
    "[[DP_ANCHOR_" in authorship_texture_section_prompt
    and "SHBHCUT002" not in authorship_texture_section_prompt
    and "Anchor lock is active" in authorship_texture_section_prompt,
    "staged authorship texture prompt freezes anchors into immutable placeholders",
)
authorship_distribution_section_prompt = _staged_reconstruction_section_prompt(
    staged_ledger,
    {"schema_version": "scanner_gate_feedback.v1", "next_candidate_controls": []},
    {
        **section_plans[0],
        "must_preserve_anchors": ["SHBHCUT002"],
        "claim_inventory_slice": ["Keep the same meaning while changing distributional texture."],
    },
    strategy="authorship_distribution_repair",
    attempt_index=1,
)
assert_test(
    "AUTHORSHIP_DISTRIBUTION_REPAIR is active" in authorship_distribution_section_prompt
    and "lower AI Authorship" in authorship_distribution_section_prompt
    and "Do not add random errors" in authorship_distribution_section_prompt,
    "staged section prompt activates authorship distribution repair controls",
)
assert_test(
    _reconstruction_drift_scan_allowed(
        "The reconstruction keeps the same learner problem but changes the transition.",
        ["lost_named_entity: 'However'"],
        0.842,
    ),
    "reconstruction drift allows discourse-marker noise to proceed to scan",
)
assert_test(
    _reconstruction_drift_scan_allowed(
        "Education should focus on what students know, and also on how students think.",
        ["quote_lost: count 2"],
        0.72,
    ),
    "reconstruction drift allows quote-marker noise after protected content passes at moderate similarity",
)
assert_test(
    not _reconstruction_drift_scan_allowed(
        "The reconstruction drops the unit name.",
        ["lost_named_entity: 'Certificate III'"],
        0.90,
    ),
    "reconstruction drift still blocks critical entity loss",
)
however_drift = check_semantic_drift(
    "However, students need support.",
    "Students still need support.",
    threshold=0.50,
)
assert_test(
    all("However" not in reason for reason in however_drift.reasons),
    "semantic drift does not treat However as a protected named entity",
)
education_heading_drift = check_semantic_drift(
    "Education today should pay more attention to the learning process.",
    "The learning process should receive more attention today.",
    threshold=0.50,
)
assert_test(
    all("Education" not in reason for reason in education_heading_drift.reasons),
    "semantic drift does not treat sentence-start Education as a protected named entity",
)
rhetorical_quote_drift = check_semantic_drift(
    'Schools should ask "What answer did the student give?" and then look at the process.',
    "Schools should ask what answer the student gave and then look at the process.",
    threshold=0.50,
)
assert_test(
    all(not reason.startswith("quote_lost") for reason in rhetorical_quote_drift.reasons),
    "semantic drift does not protect rhetorical question quote markers as source quotes",
)

ai_search_candidates = _ai_search_marked_grounding_candidates(
    "The system needs a practical method for training. "
    "Students learn better when the work is clear. "
    "This creates a stronger result for the class. "
    "The process should support learners during practice. "
    "It can help students understand the method. "
    "The teacher has to manage the lesson carefully. "
    "This is important because the skill requires control. "
    "The final result depends on the student applying the steps. "
    "Assessment should include the learning process. "
    "The goal is to improve confidence and understanding."
)
candidate_labels = [label for label, _ in ai_search_candidates]
candidate_text = "\n".join(text for _, text in ai_search_candidates)
assert_test(
    "deterministic_process_anchor_generic" in candidate_labels,
    "AI search includes deterministic process-anchor candidate",
)
assert_test(
    "[[REVIEW:" in candidate_text,
    "AI search keeps marked review-grounding candidates",
)
assert_test(
    "In practice" in candidate_text or "For this task" in candidate_text,
    "AI search process-anchor candidate adds concrete author/process anchors",
)
assert_test(
    "In my chair: " not in candidate_text and "During consultation: " not in candidate_text,
    "AI search process anchors preserve original sentence casing after prefix",
)
assert_test(
    _review_marker_notes("Sentence. [[REVIEW: Add exact source.]]"),
    "review markers are extracted for manual suggestions",
)
bad_anchor_candidate = " ".join(
    f"In my chair: Sentence {i} has enough words to look like a real rewritten sentence."
    for i in range(10)
)
assert_test(
    _ai_candidate_quality_reject_reason(bad_anchor_candidate).startswith("synthetic_anchor_overuse"),
    "AI search rejects synthetic anchor overuse in final candidates",
)
synthetic_meta_anchor_candidate = (
    "The United States has a large economy and many global companies. "
    "When this is applied in practice, major technology companies such as Apple and Microsoft are often named. "
    "During review, cities such as New York City and Los Angeles are used as examples of diversity. "
    "I would narrow the point this way: wealth and inequality can exist at the same time."
)
assert_test(
    _ai_candidate_quality_reject_reason(synthetic_meta_anchor_candidate).startswith(
        "synthetic_meta_anchor_artifact"
    ),
    "AI search rejects synthetic meta/process anchor filler in final candidates",
)
assert_test(
    _ai_candidate_quality_reject_reason("Introduction Inclusive learning design starts here."),
    "AI search rejects merged heading text",
)
assert_test(
    _ai_candidate_quality_reject_reason(
        "Research shows that social media and AI tools are transformative for students and affect academic performance."
    ) == "unsupported_source_attribution",
    "AI search rejects uncited source-attribution claims from source grounding repair",
)
assert_test(
    _ai_candidate_quality_reject_reason(
        "A study indicates that students increasingly engage with AI tools in their learning processes."
    ) == "unsupported_source_attribution",
    "AI search rejects uncited study-attribution claims from source grounding repair",
)
assert_test(
    _ai_candidate_quality_reject_reason(
        "I encourage open discussion so learners can With only six learners in my current HBB26 intake, smaller classes help. "
        "The rest of the document continues normally with enough words to form a candidate."
    ) == "dangling_sentence_fragment_join",
    "AI search rejects dangling sentence-fragment joins",
)
generic_admiration_candidate = (
    "1776: American colonies broke from Britain. The Revolutionary War began. Freedom, democracy, individual rights became core ideals. "
    "The Constitution balanced power across branches. Expansion swept westward. Industrial surge: 1800s and 1900s burst with swift growth. "
    "Immigrants arrived seeking work and safety. Millions shaped a diverse society. "
    "Silicon Valley giants: Apple, Microsoft, Google, Tesla carve innovation's edge. Entrepreneurship thrives in the culture. "
    "Global companies wield economic influence worldwide. Universities attract international students. "
    "Hollywood ships culture worldwide. Music, film, sports stars become global fame magnets. "
    "Basketball and football symbolize national identity. Younger generations echo individuality and self-expression. "
    "Economic power, cultural sway, diversity: forces molding world history."
)
assert_test(
    _ai_candidate_quality_reject_reason(generic_admiration_candidate)
    in {"generic_admiration_tone", "compressed_promotional_fragment_style"},
    "AI search rejects external-detector generic admiration/praise-list style",
)
neutralized_admiration_candidate, neutralized_repairs = _neutralize_external_detector_style_artifacts(
    generic_admiration_candidate
)
assert_test(
    neutralized_repairs
    and not _ai_candidate_quality_reject_reason(neutralized_admiration_candidate),
    "AI search can repair external-detector praise-list artifacts before scoring",
)
stylized_detector_candidate = (
    "Recently, the U.S. carved a sharp route: pulling strings in politics, money, tech cogs, culture strands. "
    "Though young beside ancient civilizations, its sprint to power stunned many. "
    "Famed for democracy and economic muscle, the nation mirrors a sprawling mix of folks and viewpoints. "
    "Under wins, stubborn blocks -- unequal ground, political cracks, health troubles, social tension. "
    "1776 cuts a sharp corner in America's story. That year, thirteen colonies snapped from Britain, "
    "sparking revolt and laying U.S. foundations. At this young nation's heart: freedom, self-rule, personal stakes. "
    "Across centuries, the nation pushed west, powered by factories and surges of newcomers. "
    "Newcomers chased safer harbors, fresh chances, weaving the nation's social tapestry. "
    "Economic muscle still defines the U.S. today. Ground zero for massive economies, towering corporations."
)
assert_test(
    _ai_candidate_quality_reject_reason(stylized_detector_candidate)
    == "over_stylized_metaphorical_texture",
    "AI search rejects over-stylized metaphor/quirky texture flagged by external detectors",
)
neutralized_stylized_candidate, stylized_repairs = _neutralize_external_detector_style_artifacts(
    stylized_detector_candidate
)
assert_test(
    stylized_repairs
    and not _ai_candidate_quality_reject_reason(neutralized_stylized_candidate),
    "AI search can neutralize over-stylized metaphor texture before scoring",
)
assert_test(
    _ai_candidate_quality_reject_reason(
        "Britain.Revolutionary War came next, carving out a fresh nation. "
        "shape.Freedom and democracy became core ideals. starts.Millions arrived. "
        "The paragraph has enough remaining words to be treated as a generated candidate for the guard."
    ).startswith("missing_sentence_spacing_artifact"),
    "AI search rejects missing sentence-spacing artifacts that external detectors highlight",
)
assert_test(
    _ai_candidate_quality_reject_reason(
        "With only six learners in my current HBB26 intake, smaller class sizes let me observe technique closely. "
        "The lesson continues with another sentence about practice and assessment. "
        "With only six learners in my current HBB26 intake, smaller class sizes let me observe technique closely."
    ).startswith("repeated_long_sequence"),
    "AI search rejects repeated long text sequences inside candidates",
)
assert_test(
    _ai_candidate_quality_reject_reason(
        "Its influence appears in politics and culture. "
        "Its influence appears in technology and business. "
        "Its influence appears in sport and media. "
        "The country has a varied history. "
        "Some changes were positive for particular communities. "
        "Some problems remained visible across public life. "
        "The paragraph continues with ordinary detail. "
        "Another sentence gives enough length for candidate quality checks. "
        "A final sentence keeps the sample above the opening threshold."
    ).startswith("repeated_sentence_opening"),
    "AI search rejects repeated sentence-opening route artifacts",
)
assert_test(
    not _ai_candidate_quality_reject_reason(
        "With only six learners in my current HBB26 intake, smaller class sizes let me observe technique closely. "
        "The lesson returns to six learners in my current HBB26 intake when assessment pressure changes the support pattern.",
        allow_repeated_long_sequence=True,
    ),
    "reconstruction candidates can proceed to scan with repeated preserved source sequences",
)
assert_test(
    not _ai_candidate_quality_reject_reason(
        "The body paragraph explains the learner's cutting task without repeating itself.\n\n"
        "References\n\n"
        "Australian Government Department of Employment and Workplace Relations. (2023). "
        "Training package materials. https://example.edu/source-one\n"
        "Australian Government Department of Employment and Workplace Relations. (2024). "
        "Training package update. https://example.edu/source-two"
    ),
    "AI search allows repeated publisher names inside reference lists",
)
damaged_source = (
    "Maintaining standards while improving access The process design does not lower the standard. "
    "ith only six participants, I can watch the method more closely. "
    "This does not simplify the work, but it clarifies the process steps. "
    "This does not simplify the work, but it clarifies the process steps."
)
repair_brief = _source_repair_brief(damaged_source)
assert_test(
    "broken word fragment" in repair_brief and "repeated sentence" in repair_brief,
    "AI search prompt identifies source damage for LLM repair",
)
repair_prompt = _ai_search_prompt(damaged_source, {}, "syntax_demolition")
assert_test(
    "Source repair requirements" in repair_prompt and "remove accidental duplicate fragments" in repair_prompt,
    "AI search prompt tells LLM to repair damaged source text",
)
repaired_candidate, repair_notes = _repair_candidate_source_damage(
    "Introduction Process design starts here. "
    "ith only six participants, I can observe closely. "
    "This does not simplify the work, but it clarifies the process steps. "
    "This does not simplify the work, but it clarifies the process steps. "
    "Conclusion This review ends here."
)
assert_test(
    not re.search(r"\bith only\b", repaired_candidate, re.I)
    and "Introduction Process" not in repaired_candidate
    and "Conclusion This" not in repaired_candidate,
    "AI search repairs inherited source damage before candidate gates",
)
assert_test(
    "fixed_broken_with_fragment" in repair_notes
    and any(note.startswith("split_") for note in repair_notes),
    "AI search records source damage repairs on candidates",
)
assert_test(
    "removed_duplicate_sentences:1" in repair_notes
    and repaired_candidate.count("This does not simplify the work") == 1,
    "AI search removes repeated exact sentences before candidate gates",
)
overlap_damaged, overlap_notes = _repair_candidate_source_damage(
    "With only six participants in the current review, smaller groups let me observe the method. "
    "I encourage open discussion so participants can With only six participants in the current review, smaller groups let me observe the method. "
    "I encourage open discussion so participants can describe how they perceive the process shape. "
    "describe how they perceive the process shape. "
    "People gain confidence when their ideas are acknowledged. "
    "A competent participant can explain the steps, identify the guide, check balance, adjust People gain confidence when their ideas are acknowledged. "
    "A competent participant can explain the steps, identify the guide, check balance, adjust the plan, and apply the method to real cases. "
    "the plan, and apply the method to real cases."
)
assert_test(
    "participants can With only" not in overlap_damaged
    and "adjust People gain" not in overlap_damaged
    and not re.search(r"(?<!can )describe how they perceive", overlap_damaged)
    and overlap_damaged.count("the plan, and apply") == 1,
    "AI search repairs overlapping fragment damage before quality gates",
)
assert_test(
    any(note.startswith("removed_dangling_prefix") for note in overlap_notes)
    and any(note.startswith("removed_duplicate_fragments") for note in overlap_notes),
    "AI search records overlapping fragment repairs",
)
latest_selected_damage, latest_selected_notes = _repair_candidate_source_damage(
    "Project Process Review Introduction\n\n"
    "The process design starts here. The standard should not depend on chance. "
    "Background\n\n"
    "The challenge begins here. They must practise, receive corrections, and repeat the skill. "
    "Method\n\n"
    "A demonstration reveals the facilitator's actions. "
    "The sources address different aspects of the practical challenge. "
    "Group A and Group B focus on overload. "
    "Group C and Group D. Group C and Group D. "
    "Group E and Group F describe multiple pathways. "
    "The policy defines the boundary for adjustment and maintaining assessment integrity. "
    "multiple pathways. The standard should not depend on chance in practice."
)
assert_test(
    "Review Introduction" not in latest_selected_damage
    and "chance.\n\nBackground" in latest_selected_damage
    and "skill.\n\nMethod" in latest_selected_damage,
    "AI search repairs selected-candidate heading placement before final output",
)
assert_test(
    "Group C and Group D. Group C and Group D." not in latest_selected_damage
    and "integrity. multiple pathways." not in latest_selected_damage,
    "AI search repairs selected-candidate conclusion source fragments",
)
assert_test(
    "split_title_from_heading" in latest_selected_notes
    and any(
        note.startswith("split_sentence_before_heading")
        or note.startswith("split_orphaned_heading")
        for note in latest_selected_notes
    )
    and "removed_duplicate_sentences:1" in latest_selected_notes,
    "AI search records selected-candidate source artifact repairs",
)
assert_test(
    _source_repair_drift_false_positive(
        "inclusive learning design in certificate iii hairdressing",
        ["lost_named_entity: 'Inclusive Learning Design'"],
    ),
    "AI search relaxes capitalization-only named entity drift after source repair",
)
assert_test(
    _source_repair_drift_false_positive(
        "Inclusive learning design in Certificate III Hairdressing.",
        ["lost_named_entity: 'Hairdressing Introduction Inclusive'"],
    ),
    "AI search relaxes merged-heading named entity drift after source repair",
)
assert_test(
    not _source_repair_drift_false_positive(
        "Inclusive learning design.",
        ["citation_lost: '(CESE, 2017)'"],
    ),
    "AI search does not relax citation drift after source repair",
)
protected_original = (
    "CESE (2017) explains the issue. "
    "He discusses practice-based knowledge (pp. 149-150). "
    "A learner may say “I don’t know”."
)
protected_candidate = (
    "CESE (2017) explains the issue differently. "
    "He discusses practice-based knowledge (pp. 149-150). "
    "A learner may say \"I don't know\"."
)
assert_test(
    not _ai_search_protected_loss_reason(
        protected_original,
        protected_candidate,
        detect_protected_spans(protected_original),
    ),
    "AI search protected check allows quote and citation punctuation normalization",
)
quote_comma_original = 'Education should focus on “what students know,” and “how students think.”'
quote_comma_candidate = 'Education should focus on "what students know" and "how students think".'
assert_test(
    not _ai_search_protected_loss_reason(
        quote_comma_original,
        quote_comma_candidate,
        detect_protected_spans(quote_comma_original),
    ),
    "AI search protected check preserves quote content without rejecting comma placement changes",
)
rhetorical_quote_original = 'Schools should ask, "What answer did the student give?" before marking the work.'
rhetorical_quote_candidate = "Schools should ask what answer the student gave before marking the work."
assert_test(
    not _ai_search_protected_loss_reason(
        rhetorical_quote_original,
        rhetorical_quote_candidate,
        detect_protected_spans(rhetorical_quote_original),
    ),
    "AI search protected check treats short question quotes as rhetorical prompts when content words remain",
)
assert_test(
    _ai_search_protected_loss_reason(
        protected_original,
        protected_candidate.replace("2017", "2018", 1),
        detect_protected_spans(protected_original),
    ).startswith("number_lost"),
    "AI search protected check still rejects lost numeric protected spans",
)
assert_test(
    _ai_search_drift_false_positive(
        "Inclusive Certificate III Hairdressing content with learners and the centre citation retained.",
        [
            "lost_named_entity: 'Hairdressing Introduction Inclusive'",
            "lost_named_entity: 'With'",
            "lost_named_entity: 'Learners'",
            "lost_named_entity: 'The Centre'",
        ],
        0.95,
    ),
    "AI search relaxes high-similarity entity noise from merged headings and sentence starts",
)
assert_test(
    not _ai_search_drift_false_positive(
        "Certificate III Hairdressing content.",
        ["lost_named_entity: 'Box Hill Institute'"],
        0.95,
    ),
    "AI search does not relax missing real named entities",
)
assert_test(
    not _ai_search_drift_false_positive(
        "Certificate III Hairdressing content with learners.",
        ["lost_named_entity: 'Learners'"],
        0.70,
    ),
    "AI search does not relax low-similarity drift",
)
assert_test(
    _ai_search_entity_drift_scan_allowed(
        "The learner content keeps DEWR (2026), CESE (2017), and the haircutting claims.",
        [
            "lost_named_entity: 'With'",
            "lost_named_entity: 'The Centre'",
            "lost_named_entity: 'Competency'",
        ],
        0.95,
    ),
    "AI search can score high-similarity candidates with non-critical entity-only drift",
)
assert_test(
    _ai_search_quote_drift_scan_allowed(
        'Education should focus on "what students know" and "how students think".',
        ["quote_lost: count 2"],
        0.82,
    )
    and not _ai_search_quote_drift_scan_allowed(
        'Education should focus on changed content.',
        ["quote_lost: count 2", "lost_named_entity: 'Teacher'"],
        0.82,
    ),
    "AI search can score quote-marker drift after protected quote content has passed",
)
assert_test(
    not _ai_search_entity_drift_scan_allowed(
        "Certificate III Hairdressing content.",
        ["lost_named_entity: 'Box Hill Institute'"],
        0.95,
    ),
    "AI search still blocks scoring candidates that lose critical entities",
)
assert_test(
    _document_recreate_drift_scan_allowed(
        "The United States discussion keeps 1776, the Constitution, civil rights, technology, and global politics.",
        [
            "lost_named_entity: 'Hollywood'",
            "lost_named_entity: 'The National Basketball Association'",
        ],
        0.62,
        {"internet_reinforced_reauthoring": True},
    )
    and not _document_recreate_drift_scan_allowed(
        "Hairdressing content with learners.",
        ["lost_named_entity: 'Box Hill Institute'"],
        0.80,
        {"internet_reinforced_reauthoring": True},
    ),
    "document-level recreate can be scanned after losing non-critical examples but not protected course anchors",
)
feedback_prompt = _ai_search_feedback_prompt(
    "Original source text.",
    {"ai_risk_badge": {"ai_components": {"generic_assertion_risk": 90.0}}},
    {
        "reference_ai": 57.78,
        "candidates": [
            {
                "strategy": "deterministic_process_anchor_generic",
                "ai": 57.83,
                "ai_delta_vs_reference": -0.05,
                "writing_quality": 47.18,
                "findings": 54,
            },
            {
                "strategy": "syntax_demolition",
                "reason": "semantic_drift lost_named_entity: 'With'",
                "drift_reasons": ["lost_named_entity: 'With'"],
            },
        ],
    },
    1,
)
assert_test(
    "Reference AI score: 57.78" in feedback_prompt
    and "Target AI score" in feedback_prompt
    and "AI=57.83" in feedback_prompt
    and "generic_assertion_risk=90.00%" in feedback_prompt
    and "semantic_drift" in feedback_prompt,
    "AI search feedback prompt gives LLM actual scores and rejection reasons",
)
paragraph_search_text = (
    "Short title.\n\n"
    "Learners should understand the process because this is important for competency. "
    "This can support learning and helps students improve. "
    "When learners pause, they may need a short moment to process what they have just practised. "
    "The process should also help learners become more confident and capable in assessment. "
    "This means the educator needs to provide guidance that supports learning and improves outcomes. "
    "These claims sound broad unless the paragraph connects them to the learner's actual cutting task.\n\n"
    "CESE (2017) explains working memory limits in practical learning."
)
paragraph_search_json = {
    "ai_risk_badge": {
        "ai_components": {
            "generic_assertion_risk": 90.0,
            "qualifying_text_ai_density": 70.0,
        }
    },
    "rewrite_edit_briefs": [
        {
            "target_sentence": (
                "When learners pause, they may need a short moment to process what they have just practised."
            ),
            "signals": {
                "score": 0.50,
                "predictable_token_spans": ["they may", "to process"],
            },
            "domain_anchors": ["learners", "pause", "practised"],
        }
    ],
}
paragraph_targets = _paragraph_component_targets(paragraph_search_text, paragraph_search_json, limit=2)
assert_test(
    paragraph_targets
    and paragraph_targets[0]["index"] == 1
    and paragraph_targets[0]["drivers"]["rewrite_brief_count"] == 1,
    "paragraph component search ranks paragraph-level AI drivers",
)
conclusion_target_text = paragraph_search_text + "\n\nThat, perhaps, is what this lesson should prepare learners for."
conclusion_targets = _paragraph_component_targets(conclusion_target_text, paragraph_search_json, limit=5)
assert_test(
    any(target.get("role") == "conclusion_template_risk" for target in conclusion_targets),
    "paragraph component search keeps short conclusion-template targets for amplification",
)
confirmed_anchor_brief = _confirmed_author_anchor_brief([
    {
        "anchor_id": "anchor_1",
        "answer": "During a classroom practice task, I saw students make repeated mistakes first, then improve after feedback and another attempt.",
        "confidence": "confirmed",
        "permission_to_use": True,
    }
])
paragraph_prompt = _paragraph_component_prompt(
    paragraph_targets[0],
    paragraph_search_json,
    1,
    reference_ai=57.78,
    required_ai_drop=5.0,
    target_ai_score=52.78,
    candidate_count=3,
    confirmed_author_anchors=confirmed_anchor_brief,
)
assert_test(
    "TARGET PARAGRAPH" in paragraph_prompt
    and "generic_assertion_risk=90.00%" in paragraph_prompt
    and "target AI<=52.78" in paragraph_prompt
    and "Confirmed author anchors available before generation" in paragraph_prompt
    and "Each anchor may be used at most once" in paragraph_prompt
    and "[[DP_ANCHOR_" in paragraph_prompt
    and "Anchor placeholders required in replacement" in paragraph_prompt
    and "Target replacement length" in paragraph_prompt
    and "guidance, not a hard gate" in paragraph_prompt
    and "Return exactly 3 alternative replacement paragraphs" in paragraph_prompt,
    "paragraph component prompt passes score drivers, confirmed anchors, and scoped rewrite instruction",
)
anchor_search_prompt = _ai_search_prompt(
    paragraph_search_text,
    paragraph_search_json,
    "confirmed_anchor_threading",
    reference_ai=57.78,
    required_ai_drop=5.0,
    target_ai_score=52.78,
    confirmed_author_anchors=confirmed_anchor_brief,
)
assert_test(
    "Strategy: confirmed-anchor threading" in anchor_search_prompt
    and "Use only the confirmed author anchors provided below" in anchor_search_prompt
    and "Confirmed author anchors available before generation" in anchor_search_prompt
    and "Do not repeat, paraphrase, or echo the same anchor" in anchor_search_prompt
    and "Do not invent a wider story" in anchor_search_prompt,
    "AI mitigation search has dedicated confirmed-anchor strategies",
)
anchor_echo_reason = _confirmed_anchor_echo_reason(
    (
        "I saw students make repeated mistakes first, then improve after feedback. "
        "Later I watched the same repeated mistakes improve only after feedback and another attempt."
    ),
    [
        {
            "anchor_id": "anchor_1",
            "answer": "During a classroom practice task, I saw students make repeated mistakes first, then improve after feedback and another attempt.",
            "confidence": "confirmed",
            "permission_to_use": True,
        }
    ],
)
assert_test(
    anchor_echo_reason.startswith("confirmed_anchor_repeated"),
    "confirmed anchor echo guard blocks repeated use of the same author anchor",
)
extracted_paragraph_candidates = _extract_paragraph_component_candidates(
    "<CANDIDATE_1>\nFirst replacement paragraph.\n</CANDIDATE_1>\n"
    "<CANDIDATE_2>\nSecond replacement paragraph.\n</CANDIDATE_2>\n"
    "<CANDIDATE_3>\nThird replacement paragraph.\n</CANDIDATE_3>",
    3,
)
assert_test(
    extracted_paragraph_candidates == [
        "First replacement paragraph.",
        "Second replacement paragraph.",
        "Third replacement paragraph.",
    ],
    "paragraph component parser extracts batched tagged candidates",
)
clean_paragraph, clean_reason = _clean_paragraph_component_candidate(
    "In practice, learners pause after the cut and compare the guide before the next section. "
    "The educator can ask them to point to the line they followed, name the section they just cut, "
    "and decide whether the next subsection should be smaller before they continue. "
    "That check ties the learning claim to a visible cutting decision instead of leaving it as a broad statement about improvement.",
    paragraph_targets[0]["paragraph"],
)
assert_test(
    clean_paragraph and not clean_reason,
    "paragraph component cleaner accepts a replacement paragraph",
)
anchor_locked_replacement, anchor_locked_reason = _clean_paragraph_component_candidate(
    (
        "Learners still need to connect [[DP_ANCHOR_001]] to the way they handle subsection control. "
        "The point is practical: the code matters only if the learner can show where the guide moved."
    ),
    (
        "Learners complete SHBHCUT004 while checking subsection control and guide movement across the "
        "practical cutting task, then explain what changed before moving into the next section."
    ),
    _paragraph_anchor_lock({
        "paragraph": (
            "Learners complete SHBHCUT004 while checking subsection control and guide movement across the "
            "practical cutting task, then explain what changed before moving into the next section."
        ),
    }),
)
missing_anchor_replacement, missing_anchor_reason = _clean_paragraph_component_candidate(
    "Learners still need to connect the unit code to subsection control and explain what changed before moving into the next section.",
    (
        "Learners complete SHBHCUT004 while checking subsection control and guide movement across the "
        "practical cutting task, then explain what changed before moving into the next section."
    ),
    _paragraph_anchor_lock({
        "paragraph": (
            "Learners complete SHBHCUT004 while checking subsection control and guide movement across the "
            "practical cutting task, then explain what changed before moving into the next section."
        ),
    }),
)
assert_test(
    "SHBHCUT004" in anchor_locked_replacement
    and not anchor_locked_reason
    and missing_anchor_reason.startswith("anchor_placeholder_lost"),
    "paragraph generation anchor lock restores required anchors and rejects missing placeholders",
)
spliced = _splice_paragraph(paragraph_search_text, paragraph_targets[0]["index"], clean_paragraph)
assert_test(
    clean_paragraph in spliced
    and "Short title." in spliced
    and "CESE (2017)" in spliced,
    "paragraph component splice patches only the target paragraph into full text",
)
source_role = _paragraph_role(
    "CESE (2017) explains that working memory can be overloaded during complex practical learning. "
    "Chandler and Sweller (1991) also describe how instruction should reduce unnecessary load when learners are still developing basic schemas. "
    "These sources are useful because they explain why practical tasks need to be sequenced carefully.",
    {"word_count": 51, "generic_assertion_hits": 3, "source_gap": False},
)
assert_test(
    source_role == "source_summary_heavy",
    "paragraph role detects source-summary-heavy paragraphs for amplification",
)
conclusion_role = _paragraph_role(
    "Conclusion\nThis review has discussed the main issues that affect practical skills teaching and assessment.",
    {"word_count": 14, "generic_assertion_hits": 2, "source_gap": True},
    is_last=True,
)
assert_test(
    conclusion_role == "conclusion_template_risk",
    "paragraph role detects conclusion template risk for amplification",
)
anchor_rich_role = _paragraph_role(
    "In my practice I noticed that my participants handled the task, process, method, "
    "procedure, tool choice, workflow, condition, constraint, measurement, testing, "
    "feedback, and case review more confidently after I slowed the demonstration.",
    {"word_count": 35, "generic_assertion_hits": 2, "source_gap": True},
)
assert_test(
    anchor_rich_role == "human_anchor_rich",
    "paragraph role preserves human-anchor-rich paragraphs",
)
amplification_prompt = _human_signal_amplification_prompt(
    {
        "role": "source_summary_heavy",
        "paragraph": (
            "CESE (2017) explains working memory limits in practical learning. "
            "Chandler and Sweller (1991) describe cognitive load."
        ),
        "previous_paragraph": "Learners were preparing for sectioning practice.",
        "next_paragraph": "The next paragraph discusses feedback timing.",
        "drivers": {"generic_assertion_hits": 5, "word_count": 42},
        "domain_anchors": ["CESE (2017)", "Chandler and Sweller (1991)", "sectioning"],
    },
    paragraph_search_json,
    1,
    candidate_count=3,
    confirmed_author_anchors=confirmed_anchor_brief,
)
assert_test(
    "HUMAN_SIGNAL_AMPLIFICATION_REPAIR" in amplification_prompt
    and "Controlled operation: add a source-to-practice bridge" in amplification_prompt
    and "Confirmed author anchors available before generation" in amplification_prompt
    and "Human Contribution must increase by at least 2" in amplification_prompt
    and "AI Authorship must not increase" in amplification_prompt
    and "[[DP_ANCHOR_" in amplification_prompt
    and "Anchor placeholders required in replacement" in amplification_prompt
    and "Target replacement length" in amplification_prompt
    and "guidance, not a hard gate" in amplification_prompt
    and "invent new evidence" in amplification_prompt
    and "generic connectors" in amplification_prompt,
    "human signal amplification prompt enforces operation-level gate",
)
generic_amplification_prompt = _human_signal_amplification_prompt(
    {
        "role": "generic_claim_heavy",
        "paragraph": "Technology can help students learn, but it can also become a shortcut.",
        "drivers": {"generic_assertion_hits": 4, "word_count": 12},
    },
    paragraph_search_json,
    1,
    candidate_count=1,
)
assert_test(
    "Controlled operation: narrow the claim with one author-reasoning trace" in generic_amplification_prompt
    and "Include exactly one" in generic_amplification_prompt
    and "does not introduce a new fact" in generic_amplification_prompt
    and "add one author judgement or reasoning trace if it changes no factual claim" in generic_amplification_prompt,
    "generic-claim amplification targets bounded author reasoning instead of evidence invention",
)
reasoning_amplification_prompt = _author_reasoning_amplification_prompt(
    {
        "role": "generic_claim_heavy",
        "paragraph": (
            "Learners complete SHBHCUT004 while checking subsection control and guide movement, "
            "but the claim can become too broad if it is not tied to the practical task."
        ),
        "drivers": {"generic_assertion_hits": 4, "word_count": 12},
        "previous_paragraph": "Students use AI tools and online platforms.",
        "next_paragraph": "Teachers need to check whether students understand the answer.",
    },
    paragraph_search_json,
    1,
    candidate_count=2,
)
assert_test(
    "AUTHOR_REASONING_AMPLIFICATION_REPAIR" in reasoning_amplification_prompt
    and "This is not evidence insertion" in reasoning_amplification_prompt
    and "do not write 'in my class'" in reasoning_amplification_prompt
    and "narrow one broad claim into a defensible condition" in reasoning_amplification_prompt
    and "[[DP_ANCHOR_" in reasoning_amplification_prompt
    and "Anchor placeholders required in replacement" in reasoning_amplification_prompt
    and "Target replacement length" in reasoning_amplification_prompt
    and "guidance, not a hard gate" in reasoning_amplification_prompt
    and "return exactly 2 alternatives" in reasoning_amplification_prompt,
    "author reasoning amplification prompt targets implied reasoning without fake context",
)
human_amp_original = {
    "integrity_layers": {
        "layers": {
            "human_contribution_signal": {"score": 51},
            "ai_transformation_risk": {"score": 49},
            "ai_authorship_risk": {"score": 58},
            "grounding_quality_risk": {"score": 58},
        }
    }
}
human_amp_candidate = {
    "integrity_layers": {
        "layers": {
            "human_contribution_signal": {"score": 54},
            "ai_transformation_risk": {"score": 48},
            "ai_authorship_risk": {"score": 57},
            "grounding_quality_risk": {"score": 58},
        }
    }
}
human_amp_score = _score_human_amplification_candidate(
    human_amp_original,
    human_amp_candidate,
    review_burden_delta=-1,
    weighted_severity_delta=-8,
    repair_aggression=0.04,
    locality_score=0.09,
)
assert_test(
    human_amp_score["human_delta"] == 3
    and human_amp_score["ai_authorship_delta"] == 1
    and human_amp_score["ai_transformation_delta"] == 1
    and human_amp_score["score"] > 25,
    "human signal amplification scorer rewards human gain under authorship cap",
)
author_completion = _build_author_evidence_completion_layer(
    paragraph_search_text,
    {
        "ai_risk_badge": {
            "ai_components": {
                "generic_assertion_risk": 90.0,
                "qualifying_text_ai_density": 80.0,
            },
            "writing_components": {
                "lived_detail_risk": 80.0,
                "source_grounding_risk": 70.0,
                "unsupported_claim_risk": 80.0,
            },
        },
        "integrity_layers": {
            "layers": {
                "human_contribution_signal": {"score": 34},
                "ai_transformation_risk": {"score": 66},
                "ai_authorship_risk": {"score": 80},
                "grounding_quality_risk": {"score": 69},
            }
        },
    },
    max_slots=2,
)
assert_test(
    author_completion.get("draft_text")
    and "[[ADD REAL AUTHOR ANCHOR" in author_completion.get("draft_text")
    and author_completion.get("auto_apply") is False
    and len(author_completion.get("slots") or []) <= 2
    and author_completion.get("estimated_human_after_completion", {}).get("high", 0) > 34,
    "author evidence completion creates explicit user-fill anchor slots with projected Human lift",
)
blocked_repair_prompt = _blocked_human_candidate_repair_prompt(
    paragraph_search_text,
    paragraph_search_text.replace("Teachers are important", "I would check whether teachers are important"),
    paragraph_search_json,
    {
        "strategy": "human_signal_amplification_p2_c1",
        "ai": 86.0,
        "human_contribution": 39,
        "human_delta": 5,
        "ai_authorship": 86,
        "critical_high_findings": 1,
        "saved_critical_high": 0,
        "selection_status": {
            "reason": "candidate_not_below_reference",
            "authenticity_gate": {
                "critical_high_regressed": True,
                "ai_authorship_regressed": True,
            },
        },
    },
    1,
)
assert_test(
    "BLOCKED_HUMAN_WINNER_REPAIR" in blocked_repair_prompt
    and "Preserve the Human Contribution gain" in blocked_repair_prompt
    and "Remove whatever created the new critical/high finding" in blocked_repair_prompt
    and "Reduce AI score further" in blocked_repair_prompt
    and "<BLOCKED_CANDIDATE>" in blocked_repair_prompt,
    "blocked Human winner repair prompt targets failed gate without restarting rewrite",
)
finding_targets = _blocking_finding_targets({
    "findings": {
        "high": [{
            "finding_id": "f001",
            "title": "unsupported_claim",
            "category": "writing_quality",
            "detail": "Claim is too strong for the evidence supplied.",
            "recommendation": "Narrow the claim.",
            "rewrite_context": {
                "target_sentence": "AI tools always improve student learning when used in schools.",
                "paragraph_excerpt": "AI tools always improve student learning when used in schools.",
            },
        }],
        "medium": [],
        "low": [],
    }
})
document_level_targets = _blocking_finding_targets(
    {
        "findings": {
            "critical": [],
            "high": [{
                "finding_id": "f999",
                "title": "draft_evolution",
                "category": "ai_generation",
                "detail": "document-level jump",
                "recommendation": "Patch the template-like conclusion sentence.",
                "rewrite_context": {
                    "paragraph_excerpt": (
                        "In this review, inclusive learning design within Certificate III Hairdressing has been examined. "
                        "Furthermore, CAST and Jwad et al. present multiple pathways for learning."
                    ),
                },
            }],
            "medium": [],
        },
    },
    candidate_text=(
        "In this review, inclusive learning design within Certificate III Hairdressing has been examined. "
        "Furthermore, CAST and Jwad et al. present multiple pathways for learning."
    ),
)
fragment_candidate = (
    "In vocational courses like hairdressing, inclusive learning design can contribute to "
    "expanding learners' access to varied educational formats. The traditional reliance on "
    "a single teaching style has diminished as educators adapt to evolving needs."
)
fragment_targets = _blocking_finding_targets(
    {
        "findings": {
            "critical": [],
            "high": [{
                "finding_id": "f100",
                "title": "draft_evolution",
                "category": "ai_generation",
                "detail": "document-level jump",
                "rewrite_context": {
                    "target_sentence": (
                        "In vocational courses like hairdressing, inclusive learning design can contribute to "
                        "expanding learners' access to varied educational formats. The traditional reliance on a single teaching style"
                    ),
                },
            }],
            "medium": [],
        },
    },
    candidate_text=fragment_candidate,
)
finding_local_prompt = _finding_local_repair_prompt(
    "AI tools always improve student learning when used in schools.",
    {"strategy": "paragraph_resequence", "human_delta": 9, "weighted_severity": 58},
    finding_targets,
    1,
)
finding_local_patches = _extract_finding_local_patches(
    '{"patches":[{"target":"AI tools always improve student learning when used in schools.",'
    '"replacement":"AI tools can support student learning when students still have to explain their choices."}]}'
)
polished_finding_local_patches = _extract_finding_local_patches(
    '{"patches":[{"target":"In practical haircutting classes, some learners do not ask for help directly.",'
    '"replacement":"In practical haircutting classes, there are instances where learners may not directly seek assistance."}]}'
)
patched_text, applied_patches = _apply_finding_local_patches(
    "AI tools always improve student learning when used in schools.",
    finding_local_patches,
)
assert_test(
    finding_targets
    and "FINDING_LOCAL_BLOCKED_WINNER_REPAIR" in finding_local_prompt
    and finding_local_patches
    and "can support" in patched_text
    and applied_patches,
    "finding-local blocked winner repair extracts targets, parses JSON patches, and splices exact text",
)
assert_test(
    document_level_targets
    and document_level_targets[0]["target_sentence"]
    and document_level_targets[0]["target_sentence"] != ""
    and document_level_targets[0]["target_sentence"] in (
        "In this review, inclusive learning design within Certificate III Hairdressing has been examined. "
        "Furthermore, CAST and Jwad et al. present multiple pathways for learning."
    )
    and document_level_targets[0]["target_source"].startswith("paragraph_excerpt"),
    "blocking target extraction derives exact candidate sentence for document-level findings",
)
assert_test(
    fragment_targets
    and fragment_targets[0]["target_sentence"] == fragment_candidate
    and fragment_targets[0]["target_source"] == "explicit_target_expanded_to_sentence",
    "blocking target extraction expands scanner fragments to full candidate sentences",
)
assert_test(
    polished_finding_local_patches == [],
    "finding-local blocked winner repair rejects polished generic patch replacements",
)
assert_test(
    _should_track_blocked_human_winner(
        selection_status={"selectable": False, "reason": "ai_drop_quality_regressed"},
        human_delta=1.0,
        ai_delta=7.4,
        authenticity_status={
            "ai_authorship_delta": 8.0,
            "ai_transformation_delta": 1.0,
            "review_burden_regressed": True,
            "weighted_severity_regressed": True,
        },
    ),
    "blocked Human winner tracker keeps small-Human but large-authorship candidates for repair",
)
assert_test(
    _blocked_human_winner_repair_budget_override("budget_exhausted_llm_calls")
    and _blocked_human_winner_repair_budget_override("budget_exhausted_candidate_scans")
    and not _blocked_human_winner_repair_budget_override("budget_exhausted_time"),
    "blocked Human winner repair has a bounded post-budget reserve for call/scan exhaustion",
)
assert_test(
    _post_safe_target_push_allows_deterministic_after_budget("budget_exhausted_llm_calls")
    and _post_safe_target_push_allows_deterministic_after_budget("budget_exhausted_candidate_scans")
    and not _post_safe_target_push_allows_deterministic_after_budget("budget_exhausted_time"),
    "post-safe target push has a bounded deterministic reserve after call/scan exhaustion",
)
assert_test(
    _post_safe_target_push_scan_reserve("budget_exhausted_candidate_scans") == 3
    and _post_safe_target_push_scan_reserve("budget_exhausted_llm_calls") == 0
    and _final_topk_texture_scan_reserve("budget_exhausted_candidate_scans") == 1,
    "target-push/final-texture reserves are tiny and budget-specific",
)
assert_test(
    _blocked_winner_bounded_quality_tradeoff(
        candidate_eval={"blocked_human_winner_repair": True},
        authenticity_status={
            "human_delta": 1.0,
            "ai_authorship_delta": 7.0,
            "ai_transformation_delta": 1.0,
            "human_shift_score": 3.688,
        },
        ai_delta=7.15,
        review_burden_delta=0,
        weighted_severity_delta=2,
        finding_delta=2,
        critical_high_delta=0,
        ai_score_regressed=False,
    ).get("allowed"),
    "blocked Human winner bounded tradeoff accepts large attribution gain with small severity cost",
)
ceiling_diagnostics = _build_mitigation_ceiling_diagnostics(
    {
        "detect_scores": {
            "original_ai": 80.0,
            "rewritten_ai": 78.0,
            "original_human_contribution": 34.0,
            "rewritten_human_contribution": 38.0,
            "original_ai_authorship": 80.0,
            "rewritten_ai_authorship": 78.0,
            "original_ai_transformation": 66.0,
            "rewritten_ai_transformation": 62.0,
            "rewritten_review_burden": 16,
            "rewritten_weighted_severity": 50,
        },
        "ai_mitigation_search": {
            "candidates": [
                {
                    "ai": 78.0,
                    "human_contribution": 38.0,
                    "selection_status": {"selectable": True},
                },
                {
                    "ai": 77.0,
                    "human_contribution": 43.0,
                    "selection_status": {
                        "selectable": False,
                        "authenticity_gate": {"reason": "review_burden_regressed"},
                    },
                },
            ]
        },
    },
    {
        "enabled": True,
        "slots": [{"slot": 1}, {"slot": 2}],
        "estimated_human_after_completion": {"low": 46, "high": 63},
    },
)
assert_test(
    ceiling_diagnostics.get("primary_blocker") == "missing_author_owned_evidence_and_context"
    and ceiling_diagnostics["candidate_frontier"]["best_seen_human"] == 43.0
    and ceiling_diagnostics["candidate_frontier"]["best_safe_human"] == 38.0
    and ceiling_diagnostics["author_evidence_gap"]["slot_count"] == 2,
    "mitigation ceiling diagnostics expose safe frontier and missing author-evidence blocker",
)
intake_layer = _build_author_evidence_intake_layer(
    {
        "enabled": True,
        "target_human_contribution": 80,
        "current_human_contribution": 38,
        "estimated_human_after_completion": {"low": 46, "high": 63},
        "slots": [
            {
                "slot": 1,
                "paragraph_index": 2,
                "paragraph_role": "generic_claim_heavy",
                "target_paragraph_preview": "Students are surrounded by too much information and must decide which source to trust.",
            }
        ],
    },
    ceiling_diagnostics,
)
assert_test(
    intake_layer.get("enabled")
    and intake_layer["questions"][0]["answer_type"] == "real_example_or_observation"
    and "Do not invent" in intake_layer["llm_supervisor_prompt"]
    and any("must not create" in item for item in intake_layer["close_gap_policy"]),
    "author evidence intake lets LLM close gaps only through confirmed anchors",
)
context_discovery = _build_author_context_discovery_layer(
    intake_layer,
    {
        "ai_risk_badge": {
            "ai_components": {
                "generic_assertion_risk": 90.0,
                "qualifying_text_ai_density": 80.0,
            },
            "writing_components": {
                "lived_detail_risk": 80.0,
                "source_grounding_risk": 70.0,
                "unsupported_claim_risk": 80.0,
            },
        }
    },
    max_items=2,
)
assert_test(
    context_discovery.get("enabled")
    and context_discovery["context_cards"][0]["safe_answer_shape"]
    and "must not answer on the user's behalf" in context_discovery["llm_task_prompt"]
    and context_discovery["handoff_env"]["json"] == "DRAFTPROOF_AUTHOR_EVIDENCE_ANSWERS_JSON"
    and "permission_to_use" in context_discovery["answer_payload_schema"]["answers"][0],
    "author context discovery lets LLM ask and shape answers without fabricating author context",
)
source_targets = _source_grounding_claim_targets(
    (
        "Students are surrounded by information from teachers, search engines, social media, "
        "online courses, and AI tools. The real challenge is knowing what is accurate, useful, "
        "ethical, and worth trusting.\n\n"
        "Teachers should guide students to question sources, compare viewpoints, and apply "
        "knowledge in real situations because information access is no longer the main barrier."
    ),
    {
        "ai_risk_badge": {
            "writing_components": {
                "source_grounding_risk": 90.0,
                "unsupported_claim_risk": 85.0,
            },
            "ai_components": {"generic_assertion_risk": 72.0},
        }
    },
    limit=2,
)
assert_test(
    source_targets
    and source_targets[0]["query"]
    and "evidence" in source_targets[0]["query"]
    and "author-owned lived evidence" in source_targets[0]["why_needed"],
    "source grounding search targets public evidence gaps without treating them as author context",
)
assert_test(
    "education" in _source_grounding_query("Teachers guide students to judge online information in education."),
    "source grounding query preserves claim keywords",
)
social_learning_query = _source_grounding_query(
    "They also learn from YouTube, social media, online courses, websites, AI tools, and people they follow online."
)
framework_query = _source_grounding_query(
    "Integrating broader aims into traditional educational frameworks remains an unresolved tension."
)
assert_test(
    "social media" in social_learning_query
    and "AI tools" in social_learning_query
    and len(social_learning_query) < 260,
    "source grounding query converts broad claims into concise research search terms",
)
assert_test(
    "educational" in framework_query
    and "evidence research study report" in framework_query,
    "source grounding query maps broad claims to evidence-friendly terms without topic-specific expansions",
)
depolished_text, depolish_repairs = _plain_language_depolish_text(
    "Therefore, it is crucial for education to emphasize the learning journey, "
    "incorporating elements like drafts, feedback, discussions, reflections, and continuous improvement."
)
assert_test(
    depolish_repairs
    and "Therefore" not in depolished_text
    and "learning journey" not in depolished_text,
    "final depolish cleanup removes late-stage polished connector artifacts",
)
score_drag_text = (
    "The classroom pattern remains. "
    "During the week, students attend classes, hear explanations, finish homework, and sit exams. "
    "I think that structure still matters because it gives students routine and direction. "
    "But the world outside school has changed much faster than the classroom.\n\n"
    "In the classroom and outside it, information is everywhere. "
    "I think the harder challenge now is knowing what to trust.\n\n"
    "It should prepare them for life in a world full of information and distractions. "
    "Students need knowledge, but they also need judgment."
)
score_drag_pruned, score_drag_repairs = _final_score_drag_sentence_prune_text(score_drag_text)
assert_test(
    score_drag_repairs
    and "world outside school has changed much faster" not in score_drag_pruned
    and "information is everywhere" not in score_drag_pruned
    and "Students need knowledge" in score_drag_pruned,
    "final score-drag pruning removes broad unsupported sentences without deleting anchored follow-up",
)
normalized_tavily = _normalize_tavily_results(
    {
        "results": [
            {
                "title": "Digital literacy and evaluating online information",
                "url": "https://example.edu/digital-literacy",
                "content": "Students need support evaluating online information and comparing sources.",
                "score": 0.7,
            },
            {
                "title": "Unrelated result",
                "url": "https://example.com/other",
                "content": "Other topic.",
                "score": 0.2,
            },
        ]
    },
    "Students evaluate online information and compare sources.",
    limit=2,
)
assert_test(
    normalized_tavily[0]["url"] == "https://example.edu/digital-literacy"
    and normalized_tavily[0]["claim_keyword_overlap"],
    "tavily results normalize with relevance metadata",
)
assert_test(
    _source_result_confidence(normalized_tavily) in {"strong", "moderate"}
    and _source_result_confidence([]) == "none",
    "source search assigns confidence before generation can use results",
)
irrelevant_high_provider_tavily = _normalize_tavily_results(
    {
        "results": [
            {
                "title": "Opera performance calendar",
                "url": "https://example.com/events",
                "content": "Application deadline, auditions, venue details, and ticket information.",
                "score": 0.99,
            },
        ]
    },
    "Students evaluate online information and compare sources.",
    limit=1,
)
assert_test(
    irrelevant_high_provider_tavily[0]["relevance_score"] < 0.10
    and _source_result_confidence(irrelevant_high_provider_tavily) == "very_weak",
    "source search does not treat high provider score with zero claim overlap as usable evidence",
)
citation_like_tavily = [
    {
        "source_quality": "medium",
        "relevance_score": 0.37,
        "claim_keyword_overlap": ["2024", "cast", "learning", "students"],
        "substantive_claim_keyword_overlap": ["2024", "cast", "learning"],
    }
]
assert_test(
    _source_result_confidence(citation_like_tavily) == "moderate",
    "citation-reference search treats medium sources with strong exact overlap as usable",
)
pruning_source = (
    "Students need structure because education can be confusing. This point is useful for the essay.\n\n"
    "Technology is changing education rapidly. It is important to consider that students can learn from many different sources. "
    "This creates a significant challenge because information can be useful, confusing, accurate, inaccurate, ethical, or not ethical. "
    "This shows that education should support students in many ways and teachers should help them develop important skills.\n\n"
    "Teachers can ask students to compare sources before they use them in a draft. That keeps the focus on judgement, not only access."
)
source_reinforce_prompt = _source_grounding_repair_prompt(
    {
        "paragraph_index": 2,
        "paragraph_role": "generic_claim_heavy",
        "claim": "Teachers should help students compare online sources before trusting them.",
        "target_preview": "Teachers should help students compare online sources before trusting them.",
    },
    {
        "source_confidence": "moderate",
        "sources": normalized_tavily[:1],
    },
    candidate_count=2,
)
assert_test(
    "First try to reinforce" in source_reinforce_prompt
    and "remove or narrow" in source_reinforce_prompt
    and "do not create lived experience" in source_reinforce_prompt
    and "same sentence also names the source" in source_reinforce_prompt
    and "<CANDIDATE_1>" in source_reinforce_prompt,
    "source-grounding block path reinforces first and narrows/removes unsupported content if source support is weak",
)
internet_reauthor_prompt = _internet_reinforced_reauthor_prompt(
    pruning_source,
    {
        "claim_targets": [
            {
                "id": "source_claim_1",
                "paragraph_index": 1,
                "claim": "Teachers should help students compare online sources.",
                "paragraph_role": "generic_claim_heavy",
            }
        ],
        "results": [
            {
                "claim_id": "source_claim_1",
                "source_confidence": "moderate",
                "sources": normalized_tavily[:1],
            }
        ],
    },
    candidate_count=2,
)
assert_test(
    "Rebuild the document around supported claims" in internet_reauthor_prompt
    and "remove unsupported generic drag" in internet_reauthor_prompt
    and "Return exactly 2 complete document candidates" in internet_reauthor_prompt,
    "internet-reinforced reauthoring rebuilds full document instead of sentence repair",
)
quote_anchor_source = (
    "Education should not only focus on “what students know,” but also on “how students think.”"
)
quote_anchor_brief = _protected_anchor_brief_for_prompt(quote_anchor_source)
quote_anchor_prompt = _internet_reinforced_reauthor_prompt(
    quote_anchor_source,
    {"claim_targets": [], "results": []},
    candidate_count=1,
)
assert_test(
    any("what students know" in item.get("text", "") for item in quote_anchor_brief)
    and any("how students think" in item.get("text", "") for item in quote_anchor_brief)
    and "Protected anchors" in quote_anchor_prompt
    and "what students know" in quote_anchor_prompt
    and "how students think" in quote_anchor_prompt
    and "keep the quote exactly" in quote_anchor_prompt,
    "full-document repair prompts carry exact quote anchors before generation",
)
quote_anchor_ai_search_prompt = _ai_search_prompt(
    quote_anchor_source,
    {"rewrite_edit_briefs": []},
    "syntax_demolition",
)
quote_anchor_paragraph_prompt = _paragraph_component_prompt(
    {
        "paragraph": quote_anchor_source,
        "previous_paragraph": "",
        "next_paragraph": "",
        "drivers": {},
        "target_sentences": [quote_anchor_source],
        "problem_spans": [],
        "domain_anchors": [],
    },
    {"rewrite_edit_briefs": []},
    1,
    candidate_count=1,
)
assert_test(
    "Protected anchors" in quote_anchor_ai_search_prompt
    and "what students know" in quote_anchor_ai_search_prompt
    and "how students think" in quote_anchor_ai_search_prompt
    and "Protected anchors" in quote_anchor_paragraph_prompt
    and "what students know" in quote_anchor_paragraph_prompt
    and "how students think" in quote_anchor_paragraph_prompt,
    "AI-search and paragraph-component prompts carry exact quote anchors before generation",
)
blocker_status = _blocker_elimination_status(
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 90.0,
                "broad_claim_risk": 85.0,
                "source_grounding_risk": 100.0,
                "lived_detail_risk": 65.0,
            },
            "ai_components": {
                "generic_assertion_risk": 65.0,
                "topk_pattern": 85.0,
                "predictability": 47.0,
            },
        }
    },
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 70.0,
                "broad_claim_risk": 65.0,
                "source_grounding_risk": 60.0,
                "lived_detail_risk": 55.0,
            },
            "ai_components": {
                "generic_assertion_risk": 55.0,
                "topk_pattern": 80.0,
                "predictability": 45.0,
            },
        }
    },
)
assert_test(
    blocker_status["active_drop"] >= 100
    and blocker_status["drops"]["unsupported_claim_risk"] == 20.0
    and blocker_status["top_remaining"][0]["key"] == "unsupported_claim_risk",
    "blocker elimination status measures active blocker reduction directly",
)
raw_high_calibrated_blockers = _blocker_scores({
    "ai_risk_badge": {
        "ai_components": {
            "topk_pattern": 81.56,
            "topk_pattern_raw": 81.56,
            "topk_calibrated_risk": 18.0,
        }
    }
})
assert_test(
    raw_high_calibrated_blockers["topk_pattern"] == 18.0
    and raw_high_calibrated_blockers["topk_calibrated_risk"] == 18.0
    and raw_high_calibrated_blockers["topk_pattern_raw"] == 81.56,
    "rewrite blocker scoring uses calibrated Top-k while preserving raw Top-k as diagnostic",
)
dominant_blocker_status = _dominant_blocker_gate_status(
    {
        "ai_risk_badge": {
            "writing_components": {"unsupported_claim_risk": 90.0},
            "ai_components": {"generic_assertion_risk": 90.0},
        }
    },
    {
        "ai_risk_badge": {
            "writing_components": {"unsupported_claim_risk": 90.0},
            "ai_components": {"generic_assertion_risk": 90.0},
        }
    },
)
dominant_blocker_clear = _dominant_blocker_gate_status(
    {
        "ai_risk_badge": {
            "writing_components": {"unsupported_claim_risk": 90.0},
            "ai_components": {"generic_assertion_risk": 90.0},
        }
    },
    {
        "ai_risk_badge": {
            "writing_components": {"unsupported_claim_risk": 82.0},
            "ai_components": {"generic_assertion_risk": 90.0},
        }
    },
)
assert_test(
    dominant_blocker_status["required"]
    and not dominant_blocker_status["cleared"]
    and dominant_blocker_status["reason"] == "dominant_blocker_not_reduced"
    and dominant_blocker_clear["cleared"],
    "dominant blocker gate blocks weak wins while unsupported/generic blockers stay unchanged",
)
dominant_source_clear = _dominant_blocker_gate_status(
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 90.0,
                "source_grounding_risk": 100.0,
                "broad_claim_risk": 85.0,
            },
            "ai_components": {"generic_assertion_risk": 65.0, "topk_pattern": 87.0},
        }
    },
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 90.0,
                "source_grounding_risk": 70.0,
                "broad_claim_risk": 75.0,
            },
            "ai_components": {"generic_assertion_risk": 65.0, "topk_pattern": 87.0},
        }
    },
)
assert_test(
    dominant_source_clear["cleared"]
    and dominant_source_clear["drops"]["source_grounding_risk"] == 30.0,
    "dominant blocker gate counts source/broad blocker movement, not only unsupported claims",
)
dominant_target_gap_status = _dominant_blocker_gate_status(
    {
        "integrity_layers": {
            "layers": {
                "human_contribution_signal": {"score": 72},
                "ai_transformation_risk": {"score": 28},
            }
        },
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 70.0,
                "broad_claim_risk": 65.0,
                "source_grounding_risk": 40.0,
            },
            "ai_components": {"topk_pattern": 81.56, "generic_assertion_risk": 25.0},
        },
    },
    {
        "integrity_layers": {
            "layers": {
                "human_contribution_signal": {"score": 70},
                "ai_transformation_risk": {"score": 30},
            }
        },
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 40.0,
                "broad_claim_risk": 65.0,
                "source_grounding_risk": 40.0,
            },
            "ai_components": {"topk_pattern": 80.95, "generic_assertion_risk": 25.0},
        },
    },
)
assert_test(
    dominant_target_gap_status["required"]
    and dominant_target_gap_status["active_threshold"] == 65.0
    and set(dominant_target_gap_status["active_keys"]) == {
        "unsupported_claim_risk",
        "broad_claim_risk",
    },
    "dominant blocker gate lowers its active threshold without treating raw Top-k as a direct blocker",
)
human_target_search_status = _human_target_ai_search_status(
    {
        "scan_intelligence": {
            "transformation": {
                "contribution": {
                    "human_contribution_ratio": 73,
                    "ai_transformation_ratio": 27,
                }
            }
        },
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 80.0,
                "source_grounding_risk": 70.0,
                "broad_claim_risk": 65.0,
            },
            "ai_components": {"topk_pattern": 77.0, "generic_assertion_risk": 65.0},
        },
    }
)
human_target_reached_status = _human_target_ai_search_status(
    {
        "scan_intelligence": {
            "transformation": {
                "contribution": {
                    "human_contribution_ratio": 82,
                    "ai_transformation_ratio": 18,
                }
            }
        },
        "ai_risk_badge": {
            "writing_components": {"unsupported_claim_risk": 80.0},
            "ai_components": {"topk_pattern": 77.0},
        },
    }
)
assert_test(
    human_target_search_status["active"]
    and human_target_search_status["reason"] == "human_below_target_with_active_blockers"
    and not human_target_reached_status["active"],
    "AI mitigation search stays enabled below AI threshold when Human target is still unmet",
)
assert_test(
    _ai_search_adaptive_stop_reason(
        {
            "selectable": True,
            "dominant_blocker_gate": {"required": True, "cleared": True},
            "authenticity_gate": {
                "human_delta": 10.0,
                "ai_authorship_delta": 4.0,
                "ai_transformation_delta": 10.0,
                "critical_high_regressed": False,
                "review_burden_regressed": False,
                "weighted_severity_regressed": False,
            },
        },
        phase="paragraph_component_search",
    ) == "adaptive_stop_after_paragraph_component_search"
    and not _ai_search_adaptive_stop_reason(
        {
            "selectable": True,
            "dominant_blocker_gate": {"required": True, "cleared": False},
            "authenticity_gate": {
                "human_delta": 10.0,
                "ai_authorship_delta": 4.0,
                "ai_transformation_delta": 10.0,
            },
        },
        phase="paragraph_component_search",
    ),
    "AI search adaptive stop ends candidate search only after dominant blockers and safe movement clear",
)
assert_test(
    _ai_search_adaptive_stop_reason(
        {
            "selectable": True,
            "dominant_blocker_gate": {
                "required": True,
                "cleared": False,
                "active_keys": ["unsupported_claim_risk"],
            },
            "dominant_blocker_safe_progress_override": True,
            "authenticity_gate": {
                "human_delta": 5.0,
                "ai_authorship_delta": 1.0,
                "ai_transformation_delta": 4.0,
                "critical_high_regressed": False,
                "review_burden_regressed": False,
                "weighted_severity_regressed": False,
            },
        },
        phase="deterministic_candidates",
        short_document=True,
    ) == "adaptive_stop_after_deterministic_candidates",
    "short-document adaptive stop accepts safe deterministic progress with stale blocker override",
)
radar_controller_status = _radar_goal_controller_status(
    {
        "integrity_layers": {
            "layers": {
                "human_contribution_signal": {"score": 54},
            }
        },
        "scan_intelligence": {
            "document": {"word_count": 625},
            "blocker_radar": {
                "blockers": [
                    {
                        "key": "topk_pattern",
                        "label": "Top-k predictability",
                        "layer": "ai_authorship_risk",
                        "score": 81,
                        "severity": "high",
                        "scope": "localized",
                        "sentence_ids": ["s022"],
                        "paragraph_ids": ["p003"],
                        "diagnostic_flags": {
                            "texture_pressure": True,
                            "evidence_gap": False,
                            "source_dependency": False,
                            "author_context_gap": False,
                        },
                    },
                    {
                        "key": "unsupported_claim_risk",
                        "label": "Unsupported claim risk",
                        "layer": "grounding_quality_risk",
                        "score": 70,
                        "severity": "high",
                        "scope": "document_wide",
                        "sentence_ids": [],
                        "paragraph_ids": ["p001", "p002"],
                        "diagnostic_flags": {
                            "texture_pressure": False,
                            "evidence_gap": True,
                            "source_dependency": False,
                            "author_context_gap": True,
                        },
                    },
                    {
                        "key": "citation_weakness_risk",
                        "label": "Citation weakness",
                        "layer": "grounding_quality_risk",
                        "score": 50,
                        "severity": "medium",
                        "scope": "document_wide",
                        "sentence_ids": [],
                        "paragraph_ids": ["p002"],
                        "diagnostic_flags": {
                            "texture_pressure": False,
                            "evidence_gap": True,
                            "source_dependency": True,
                            "author_context_gap": False,
                        },
                    },
                ]
            }
        }
    }
)
radar_option_matrix = radar_controller_status["option_matrix"]
radar_rows = {row["blocker_key"]: row for row in radar_option_matrix.get("options_by_blocker", [])}
assert_test(
    radar_option_matrix.get("policy", {}).get("owner") == "rewrite_controller"
    and radar_option_matrix.get("policy", {}).get("primary_goal") == "human_contribution_above_80"
    and radar_option_matrix.get("policy", {}).get("default_sequence") == ["repair_no_llm", "recreate_llm_tavily", "remove"]
    and radar_rows["topk_pattern"]["options"][0]["operation"] == "deterministic_topk_texture_repair"
    and radar_rows["topk_pattern"]["options"][0]["goal"]["role"] == "enable_human_gain_by_capping_ai_authorship_and_transformation"
    and "remove" not in radar_rows["topk_pattern"]["controller_sequence"]
    and radar_rows["unsupported_claim_risk"]["controller_sequence"] == ["repair_no_llm", "recreate_llm_tavily", "remove"]
    and radar_rows["unsupported_claim_risk"]["options"][0]["goal"]["role"] == "direct_human_contribution_gain"
    and "candidate_human_contribution_increases_or_reaches_80" in radar_rows["unsupported_claim_risk"]["options"][0]["goal"]["acceptance_gate"]
    and radar_rows["citation_weakness_risk"]["options"][0]["operation"] == "deterministic_existing_source_bridge_or_narrow"
    and "no_llm" in radar_rows["citation_weakness_risk"]["options"][0]["requires"]
    and "tavily_max_5_searches" in radar_rows["citation_weakness_risk"]["options"][1]["requires"],
    "radar option matrix lays goal-serving repair/recreate/remove choices by blocker without moving strategy into scanner",
)
assert_test(
    radar_controller_status.get("execute_before_local_rewrite") is True
    and radar_controller_status.get("force_broad_reconstruction") is True
    and radar_controller_status.get("document_size_class") == "short"
    and radar_controller_status.get("max_recreate_blocks") == 2,
    "radar goal controller drives execution before local repair and adapts recreate scope to content size",
)
assert_test(
    _radar_goal_requires_human_progress(radar_controller_status) is True
    and _radar_goal_requires_human_progress({
        **radar_controller_status,
        "current_human_contribution": 82,
    }) is False,
    "radar goal controller blocks zero-Human side wins while Human target is unmet",
)
stale_wrapper_scan = {
    "document_context": {"word_count": 625},
    "ai_risk_badge": {
        "writing_components": {
            "unsupported_claim_risk": 70.0,
            "broad_claim_risk": 65.0,
            "source_grounding_risk": 40.0,
        },
        "ai_components": {
            "topk_pattern": 81.0,
            "predictability": 45.0,
        },
    },
}
fresh_baseline_scan = {
    **stale_wrapper_scan,
    "integrity_layers": {
        "layers": {
            "human_contribution_signal": {"score": 54.0},
            "ai_transformation_risk": {"score": 46.0},
            "ai_authorship_risk": {"score": 48.0},
            "grounding_quality_risk": {"score": 50.0},
        }
    },
}
stale_controller = _radar_goal_controller_status(stale_wrapper_scan)
fresh_controller = _radar_goal_controller_status(fresh_baseline_scan)
assert_test(
    stale_controller.get("active") is False
    and stale_controller.get("current_human_contribution") is None
    and fresh_controller.get("active") is True
    and fresh_controller.get("current_human_contribution") == 54.0
    and _radar_goal_requires_human_progress(fresh_controller) is True,
    "radar controller must be derived from the fresh baseline scan before selector gates run",
)
blocker_plan = _blocker_operation_plan(
    pruning_source,
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 90.0,
                "broad_claim_risk": 85.0,
                "source_grounding_risk": 70.0,
                "lived_detail_risk": 80.0,
            },
            "ai_components": {
                "generic_assertion_risk": 90.0,
                "topk_pattern": 85.0,
                "predictability": 47.0,
            },
        },
        "rewrite_edit_briefs": [
            {
                "target_sentence": "Technology is changing education rapidly.",
                "signals": {"score": 0.9},
            }
        ],
    },
    limit=4,
)
blocker_candidates = _blocker_operation_candidates(
    pruning_source,
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 90.0,
                "broad_claim_risk": 85.0,
                "source_grounding_risk": 70.0,
                "lived_detail_risk": 80.0,
            },
            "ai_components": {
                "generic_assertion_risk": 90.0,
                "topk_pattern": 85.0,
                "predictability": 47.0,
            },
        },
        "rewrite_edit_briefs": [
            {
                "target_sentence": "Technology is changing education rapidly.",
                "signals": {"score": 0.9},
            }
        ],
    },
    limit=4,
)
assert_test(
    blocker_plan.get("operations")
    and blocker_plan.get("block_decisions")
    and blocker_plan["operations"][0]["operation"] in {"delete_or_compress", "claim_narrow", "compress_or_delete"}
    and blocker_candidates
    and any(meta.get("operation") in {"delete_paragraph", "compress_or_narrow_paragraph", "claim_narrow"} for _s, _c, meta in blocker_candidates)
    and all("operation_plan" in meta and "block_decision" in meta for _s, _c, meta in blocker_candidates),
    "blocker compiler converts scanner blockers into hard reinforce/remove/narrow decisions",
)
manual_decisions = _block_level_decisions(
    [
        {
            "paragraph_index": 0,
            "role": "generic_claim_heavy",
            "blockers": ["unsupported_claim_risk", "source_grounding_risk"],
            "word_count": 60,
            "generic_density": 2.5,
            "drivers": {
                "source_gap": True,
                "generic_assertion_hits": 3,
                "concrete_anchor_hits": 1,
            },
        },
        {
            "paragraph_index": 1,
            "role": "conclusion_template_risk",
            "blockers": ["generic_assertion_risk"],
            "word_count": 45,
            "generic_density": 4.0,
            "drivers": {
                "source_gap": True,
                "generic_assertion_hits": 5,
                "concrete_anchor_hits": 0,
            },
        },
    ],
    pruning_source,
)
assert_test(
    manual_decisions[0]["decision"] == "reinforce_with_public_source"
    and manual_decisions[0]["fallback_if_failed"] == "remove_or_compress"
    and manual_decisions[1]["decision"] == "remove_or_compress",
    "block decisions reinforce salvageable claims first and remove unsalvageable drag as last resort",
)
decision_source_targets = _source_grounding_targets_from_block_decisions(
    pruning_source,
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 90.0,
                "source_grounding_risk": 90.0,
            },
            "ai_components": {"generic_assertion_risk": 90.0},
        }
    },
    manual_decisions,
    limit=5,
)
assert_test(
    len(decision_source_targets) == 1
    and decision_source_targets[0]["block_decision"]["decision"] == "reinforce_with_public_source"
    and decision_source_targets[0]["block_decision"]["fallback_if_failed"] == "remove_or_compress",
    "source search targets only reinforce-designated salvageable blocks",
)
source_repair_match_layer = {
    "claim_targets": [
        {
            "id": "claim_target_1",
            "paragraph_index": 0,
            "claim": "Teachers help students judge online information.",
            "query": "teacher guidance source evaluation students education research",
        }
    ],
    "results": [
        {
            "claim_id": "block_decision_0",
            "paragraph_index": 0,
            "source_confidence": "moderate",
            "sources": [{"title": "Source evaluation in education"}],
        },
        {
            "claim_id": "unmatched",
            "paragraph_index": 8,
            "source_confidence": "moderate",
            "sources": [{"title": "Unmatched"}],
        },
    ],
}
citation_reference_source_text = (
    "Introductory sentence without citations.\n\n"
    "Inclusive learning design supports diverse learners (CAST, 2024; Jwad et al., 2022). "
    "Billett (2013) discusses practice-based learning."
)
citation_reference_targets = _citation_reference_search_targets(
    citation_reference_source_text,
    {
        "ai_risk_badge": {
            "writing_components": {
                "citation_weakness_risk": 50.0,
                "source_grounding_risk": 40.0,
            }
        }
    },
    limit=3,
)
source_repair_matches = _source_grounding_repair_matches(
    source_repair_match_layer,
    {"moderate", "strong"},
    limit=2,
)
source_reference_layer = {
    "results": [
        {
            "source_confidence": "strong",
            "sources": [
                {
                    "title": "[PDF] Teaching the 21st Century Learning Skills with the Critical Thinking Technique",
                    "url": "https://files.eric.ed.gov/fulltext/EJ1385999.pdf",
                },
                {
                    "title": "The integration of 21st century skills in the curriculum of education",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11336407/",
                },
            ],
        }
    ]
}
source_reference_entries = _source_reference_entries_from_layer(source_reference_layer, limit=2)
source_reference_candidate = _source_reference_append_candidate(
    "Students need help judging information online.",
    source_reference_layer,
    limit=2,
)
assert_test(
    len(citation_reference_targets) == 3
    and citation_reference_targets[0]["paragraph_role"] == "citation_reference"
    and citation_reference_targets[0]["repair_scope"] == "sentence_window"
    and citation_reference_targets[0]["paragraph_index"] == 1
    and citation_reference_targets[0]["sentence_index"] == 0
    and "CAST" in citation_reference_targets[0]["query"]
    and "2024" in citation_reference_targets[0]["query"]
    and any("Billett" in target["query"] for target in citation_reference_targets),
    "citation-reference search targets existing citation markers with real paragraph/sentence indexes",
)
source_sentence_candidate, source_sentence_reject = _clean_source_sentence_candidate(
    "CAST's Universal Design for Learning guidance links learner variability to flexible learning goals and materials.",
    "Inclusive learning design supports diverse learners (CAST, 2024; Jwad et al., 2022).",
)
assert_test(
    source_sentence_candidate
    and not source_sentence_reject
    and _clean_paragraph_component_candidate(
        source_sentence_candidate,
        (
            "Inclusive learning design supports diverse learners (CAST, 2024; Jwad et al., 2022). "
            "Billett (2013) discusses practice-based learning. The paragraph continues with more teaching context. "
            "It also connects demonstration, guided repetition, and learner variability across several classroom decisions."
        ),
    )[0],
    "paragraph component cleaner treats length as scanner-scored guidance rather than a hard rejection",
)
assert_test(
    len(source_repair_matches) == 1
    and source_repair_matches[0]["_repair_target"]["id"] == "claim_target_1",
    "source grounding repair maps search results back by paragraph when claim ids differ",
)
assert_test(
    len(source_reference_entries) == 2
    and "[PDF]" not in source_reference_entries[0]
    and "References" in source_reference_candidate
    and "https://files.eric.ed.gov/fulltext/EJ1385999.pdf" in source_reference_candidate,
    "source search can create a verifiable reference-append candidate without inventing evidence",
)
previous_search_enabled = os.environ.get("DRAFTPROOF_SOURCE_SEARCH_ENABLED")
try:
    os.environ["DRAFTPROOF_SOURCE_SEARCH_ENABLED"] = "1"
    severe_priority = _internet_reauthor_priority_status(
        {
            "ai_risk_badge": {
                "writing_components": {
                    "unsupported_claim_risk": 90.0,
                    "broad_claim_risk": 70.0,
                    "source_grounding_risk": 85.0,
                },
                "ai_components": {
                    "generic_assertion_risk": 88.0,
                    "topk_pattern": 87.0,
                    "qualifying_text_ai_density": 82.0,
                },
            }
        },
        pruning_source,
    )
    mild_priority = _internet_reauthor_priority_status(
        {
            "ai_risk_badge": {
                "writing_components": {
                    "unsupported_claim_risk": 60.0,
                    "broad_claim_risk": 55.0,
                    "source_grounding_risk": 65.0,
                },
                "ai_components": {
                    "generic_assertion_risk": 62.0,
                    "topk_pattern": 64.0,
                    "qualifying_text_ai_density": 62.0,
                },
            }
        },
        pruning_source,
    )
finally:
    if previous_search_enabled is None:
        os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_ENABLED", None)
    else:
        os.environ["DRAFTPROOF_SOURCE_SEARCH_ENABLED"] = previous_search_enabled
assert_test(
    severe_priority.get("prioritize") is True
    and "unsupported_claim_risk" in severe_priority.get("severe_keys", [])
    and mild_priority.get("prioritize") is False,
    "severe document blockers force internet reauthoring before paragraph-component search can spend the LLM budget",
)
newline_structured_source = (
    "Inclusive Learning Design in Certificate III Hairdressing\n"
    "Introduction\n"
    "Inclusive learning design in adult VET hairdressing training is part of teaching technical skills, not separate from it.\n"
    "When learners start to get lost\n"
    "Learners can name the seven cutting procedures, but naming them does not always turn into correct action on mannequins."
)
assert_test(
    len(_logical_paragraphs(newline_structured_source)) == 5,
    "single-newline headed submissions are split into logical blocks for paragraph targeting",
)
generic_compiler_source = (
    "Overview\n"
    "Technology is changing education rapidly. It is important to consider that students can learn from many different sources. "
    "This shows that education should support students in many ways and teachers should help them develop important skills.\n"
    "HBB26 learners practise SHBHCUT006 with six learners at Box Hill. I compare the guide line with each learner before they continue.\n"
    "Conclusion\n"
    "The goal should be to help students achieve success in a changing system."
)
generic_compiler_candidates = _generic_assertion_compiler_candidates(
    generic_compiler_source,
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 85.0,
                "broad_claim_risk": 82.0,
            },
            "ai_components": {
                "generic_assertion_risk": 90.0,
                "topk_pattern": 70.0,
            },
        }
    },
    limit=4,
)
assert_test(
    generic_compiler_candidates
    and all("HBB26" in candidate and "SHBHCUT006" in candidate for _s, candidate, _m in generic_compiler_candidates)
    and any("important to consider" not in candidate for _s, candidate, _m in generic_compiler_candidates)
    and any(meta.get("generic_assertion_compiler") for _s, _c, meta in generic_compiler_candidates),
    "generic assertion compiler narrows or prunes broad claims while preserving code anchors",
)
claim_narrowing_prompt = _claim_narrowing_repair_prompt(
    pruning_source,
    {
        "ai_risk_badge": {
            "writing_components": {
                "unsupported_claim_risk": 90.0,
                "broad_claim_risk": 85.0,
            },
            "ai_components": {
                "generic_assertion_risk": 65.0,
                "topk_pattern": 80.0,
            },
        }
    },
    candidate_count=2,
)
topk_texture_prompt = _topk_texture_repair_prompt(
    pruning_source,
    {
        "ai_risk_badge": {
            "ai_components": {"topk_pattern": 80.0, "predictability": 47.0},
            "writing_components": {"unsupported_claim_risk": 70.0, "broad_claim_risk": 65.0},
        }
    },
    candidate_count=2,
)
assert_test(
    "weaken absolute claims" in claim_narrowing_prompt
    and "unsupported_claim_risk must drop" in claim_narrowing_prompt
    and "calibrated Top-k risk must move below" in topk_texture_prompt
    and "Do not add new facts" in topk_texture_prompt,
    "targeted claim narrowing and top-k texture prompts attack remaining blockers directly",
)
topk_route_report = {
    "ai_risk_badge": {
        "ai_components": {"topk_pattern": 100.0, "predictability": 50.0},
        "writing_components": {},
    },
    "predictability": {
        "all_sentences": [
            {
                "sentence_id": "s001",
                "sentence": "The United States is often described as one of the most influential countries in modern history.",
                "top10_ratio": 0.75,
                "top50_ratio": 0.875,
                "predictability_risk": 0.56,
                "predictable_token_spans": ["States is", "described as one of the most"],
                "top_predicted_tokens": [{"token": "of", "rank": 1, "probability": 0.9, "top10": True}],
            },
            {
                "sentence_id": "s002",
                "sentence": "It has shaped global politics, economics, technology, entertainment, and education for many decades.",
                "top10_ratio": 0.65,
                "top50_ratio": 0.94,
                "predictability_risk": 0.54,
                "predictable_token_spans": ["for many decades"],
            },
        ]
    },
}
topk_route_map = _topk_repair_map(
    "The United States is often described as one of the most influential countries in modern history. "
    "It has shaped global politics, economics, technology, entertainment, and education for many decades.",
    topk_route_report,
    limit=2,
)
topk_route_candidates = _topk_route_optimizer_candidates(
    "The United States is often described as one of the most influential countries in modern history. "
    "It has shaped global politics, economics, technology, entertainment, and education for many decades.",
    topk_route_report,
    limit=2,
)
topk_mask_prompt = _topk_masked_route_prompt("The United States is often described as one country.", topk_route_report, candidate_count=1)
topk_patch_sets = _extract_topk_route_patch_candidates(
    '{"candidates":[{"patches":[{"sentence_id":"s001","original_sentence":"The United States is often described as one country.","replacement_sentence":"One common description of the United States is that it is one country."}]}]}',
    max_candidates=1,
)
topk_patched_text, topk_applied = _apply_topk_route_patches(
    "The United States is often described as one country.",
    topk_patch_sets[0],
)
assert_test(
    topk_route_map["saturated"]
    and topk_route_map["targets"][0]["predictable_token_spans"]
    and topk_route_candidates
    and "often described as" not in topk_route_candidates[0][1]
    and "Return valid JSON only" in topk_mask_prompt
    and "claim -> explanation -> implication" in topk_mask_prompt
    and _phase_sampling_arg("DRAFTPROOF_TOPK_ROUTE", "TOP_P") == 0.72
    and _phase_sampling_arg("DRAFTPROOF_TOPK_ROUTE", "FREQUENCY_PENALTY") == 0.35
    and topk_applied
    and topk_patched_text.startswith("One common description"),
    "top-k route optimizer builds repair map, deterministic candidates, JSON patch candidates, and phase sampling controls",
)
previous_search_enabled = os.environ.get("DRAFTPROOF_SOURCE_SEARCH_ENABLED")
previous_tavily_key = os.environ.get("TAVILY_API_KEY")
try:
    os.environ["DRAFTPROOF_SOURCE_SEARCH_ENABLED"] = "1"
    os.environ.pop("TAVILY_API_KEY", None)
    source_layer = _build_source_grounding_search_layer(
        "Teachers should guide students to compare online sources before trusting them.",
        {
            "ai_risk_badge": {
                "writing_components": {
                    "source_grounding_risk": 90.0,
                    "unsupported_claim_risk": 85.0,
                },
                "ai_components": {"generic_assertion_risk": 72.0},
            }
        },
        max_queries=1,
        max_results=1,
    )
finally:
    if previous_search_enabled is None:
        os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_ENABLED", None)
    else:
        os.environ["DRAFTPROOF_SOURCE_SEARCH_ENABLED"] = previous_search_enabled
    if previous_tavily_key is None:
        os.environ.pop("TAVILY_API_KEY", None)
    else:
        os.environ["TAVILY_API_KEY"] = previous_tavily_key
assert_test(
    source_layer.get("status") == "missing_api_key"
    and source_layer.get("auto_apply") is False
    and any("must not be converted into author-owned" in item for item in source_layer.get("policy", [])),
    "source grounding search is optional and cannot fabricate author-owned context",
)
previous_search_enabled = os.environ.get("DRAFTPROOF_SOURCE_SEARCH_ENABLED")
previous_search_auto = os.environ.get("DRAFTPROOF_SOURCE_SEARCH_AUTO_ENABLE_WITH_KEY")
previous_tavily_key = os.environ.get("TAVILY_API_KEY")
try:
    os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_ENABLED", None)
    os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_AUTO_ENABLE_WITH_KEY", None)
    os.environ["TAVILY_API_KEY"] = "dummy"
    key_only_enabled = _source_search_enabled()
    os.environ["DRAFTPROOF_SOURCE_SEARCH_ENABLED"] = "1"
    explicit_enabled = _source_search_enabled()
finally:
    if previous_search_enabled is None:
        os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_ENABLED", None)
    else:
        os.environ["DRAFTPROOF_SOURCE_SEARCH_ENABLED"] = previous_search_enabled
    if previous_search_auto is None:
        os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_AUTO_ENABLE_WITH_KEY", None)
    else:
        os.environ["DRAFTPROOF_SOURCE_SEARCH_AUTO_ENABLE_WITH_KEY"] = previous_search_auto
    if previous_tavily_key is None:
        os.environ.pop("TAVILY_API_KEY", None)
    else:
        os.environ["TAVILY_API_KEY"] = previous_tavily_key
assert_test(
    key_only_enabled is False and explicit_enabled is True,
    "source search is opt-in and does not start spending Tavily calls just because a key exists",
)
previous_search_enabled = os.environ.get("DRAFTPROOF_SOURCE_SEARCH_ENABLED")
previous_search_cap = os.environ.get("DRAFTPROOF_SOURCE_SEARCH_MAX_CALLS_PER_RUN")
previous_tavily_key = os.environ.get("TAVILY_API_KEY")
try:
    os.environ["DRAFTPROOF_SOURCE_SEARCH_ENABLED"] = "1"
    os.environ["DRAFTPROOF_SOURCE_SEARCH_MAX_CALLS_PER_RUN"] = "9"
    os.environ.pop("TAVILY_API_KEY", None)
    capped_source_layer = _build_source_grounding_search_layer(
        (
            "Teachers guide students online. Students compare websites. AI tools affect drafting. "
            "Online courses change revision. Social media shapes trust. Search engines create overload. "
            "Classroom tests still matter. Feedback helps students judge sources."
        ),
        {"ai_risk_badge": {"writing_components": {"source_grounding_risk": 95.0}}},
        max_queries=9,
        max_results=1,
    )
finally:
    if previous_search_enabled is None:
        os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_ENABLED", None)
    else:
        os.environ["DRAFTPROOF_SOURCE_SEARCH_ENABLED"] = previous_search_enabled
    if previous_search_cap is None:
        os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_MAX_CALLS_PER_RUN", None)
    else:
        os.environ["DRAFTPROOF_SOURCE_SEARCH_MAX_CALLS_PER_RUN"] = previous_search_cap
    if previous_tavily_key is None:
        os.environ.pop("TAVILY_API_KEY", None)
    else:
        os.environ["TAVILY_API_KEY"] = previous_tavily_key
assert_test(
    capped_source_layer.get("budget", {}).get("hard_max_calls_per_run") == 5
    and capped_source_layer.get("budget", {}).get("max_calls_per_run") == 5
    and len(capped_source_layer.get("claim_targets") or []) <= 5,
    "source grounding search is hard-capped at five Tavily calls per run",
)
previous_source_depth = os.environ.get("DRAFTPROOF_SOURCE_SEARCH_DEPTH")
try:
    os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_DEPTH", None)
    advanced_depth = _source_search_depth_status(
        {"ai_risk_badge": {"writing_components": {"source_grounding_risk": 95.0}}},
        1,
    )
    basic_depth = _source_search_depth_status(
        {"ai_risk_badge": {"writing_components": {"source_grounding_risk": 95.0}}},
        4,
    )
finally:
    if previous_source_depth is None:
        os.environ.pop("DRAFTPROOF_SOURCE_SEARCH_DEPTH", None)
    else:
        os.environ["DRAFTPROOF_SOURCE_SEARCH_DEPTH"] = previous_source_depth
assert_test(
    advanced_depth.get("search_depth") == "advanced"
    and advanced_depth.get("chunks_per_source") == 3
    and basic_depth.get("search_depth") == "basic",
    "source search uses adaptive advanced retrieval only for severe low-target grounding gaps",
)
pruning_candidates = _content_pruning_candidates(
    pruning_source,
    {"rewrite_edit_briefs": []},
    limit=3,
)
assert_test(
    pruning_candidates
    and any(meta.get("operation") in {"delete_paragraph", "compress_paragraph"} for _s, _c, meta in pruning_candidates)
    and all("Technology is changing education rapidly" not in candidate or meta.get("operation") == "compress_paragraph" for _s, candidate, meta in pruning_candidates),
    "content pruning generates deletion/compression candidates for generic score-drag paragraphs",
)
heading_target_text = (
    "Title\n\n"
    "Introduction\n\n"
    "This paragraph gives enough body text for the opening section to be treated as prose rather than a heading.\n\n"
    "Conclusion\n\n"
    "This review has discussed inclusive learning design in Certificate III Hairdressing. Demonstration alone does not build competency. Learners need a process they can follow and repeat."
)
heading_targets = _paragraph_component_targets(heading_target_text, {}, limit=4)
assert_test(
    _ai_candidate_quality_reject_reason("Title\n\nConclusion") == "orphan_heading:Conclusion"
    and all(t.get("paragraph") != "Conclusion" for t in heading_targets)
    and any(str(t.get("paragraph") or "").startswith("This review has discussed") for t in heading_targets),
    "paragraph targeting skips heading-only conclusion targets and quality gate rejects orphan headings",
)
score_drag_status = _score_drag_removal_status(
    authenticity_status={
        "human_delta": 0.0,
        "ai_authorship_delta": 1.0,
        "ai_transformation_delta": 0.0,
    },
    human_shift={"score": -3.92},
    ai_delta=0.5,
    finding_delta=-3,
    review_burden_delta=-1,
    weighted_severity_delta=-4,
    critical_high_delta=0,
    ai_score_regressed=False,
)
assert_test(
    not score_drag_status.get("allowed")
    and score_drag_status.get("cleanup_only")
    and score_drag_status.get("ignored_negative_human_shift")
    and score_drag_status.get("finding_drop") == 3,
    "score-drag removal is cleanup-only when Human movement is not material",
)
score_drag_mitigation_status = _score_drag_removal_status(
    authenticity_status={
        "human_delta": 6.0,
        "ai_authorship_delta": 1.0,
        "ai_transformation_delta": 3.0,
    },
    human_shift={"score": 8.0},
    ai_delta=0.5,
    finding_delta=-3,
    review_burden_delta=-1,
    weighted_severity_delta=-4,
    critical_high_delta=0,
    ai_score_regressed=False,
)
assert_test(
    not score_drag_mitigation_status.get("allowed"),
    "score-drag mitigation remains disabled unless explicitly enabled",
)
valid_anchor_answers, rejected_anchor_answers = _validate_author_evidence_answers(
    intake_layer,
    [
        {
            "anchor_id": "anchor_1",
            "answer": "In my class, two students copied short online source explanations but could not explain why they trusted those sources.",
            "confidence": "confirmed",
            "permission_to_use": True,
        },
        {
            "anchor_id": "anchor_1",
            "answer": "Maybe something like that happened.",
            "confidence": "uncertain",
            "permission_to_use": True,
        },
    ],
)
irrelevant_anchor_check = _author_answer_relevance(
    {
        "target_preview": "Real learning takes time, mistakes, practice, confusion, feedback, and trying again."
    },
    "In one class discussion, students named online platforms but did not explain which source they trusted.",
)
integration_prompt = _author_evidence_integration_prompt(
    "Students are surrounded by too much information.",
    intake_layer["questions"][0],
    valid_anchor_answers[0]["answer"],
)
deterministic_integrated, deterministic_reason = _deterministic_author_anchor_paragraph(
    "Students are surrounded by too much information. Some of it is helpful, and some of it is wrong.",
    intake_layer["questions"][0],
    valid_anchor_answers[0]["answer"],
)
clean_integrated, clean_integrated_reason = _clean_author_evidence_integrated_paragraph(
    "Students are surrounded by too much information. In my class, two students copied short online explanations but could not explain the steps when I asked them to talk through their work.",
    "Students are surrounded by too much information.",
)
spliced_anchor_text = _splice_author_evidence_paragraph(
    "Intro paragraph.\n\nStudents are surrounded by too much information.\n\nEnd paragraph.",
    1,
    clean_integrated,
)
assert_test(
    len(valid_anchor_answers) == 1
    and rejected_anchor_answers
    and not irrelevant_anchor_check["accepted"]
    and "Use only the confirmed answer" in integration_prompt
    and not deterministic_reason
    and "two students copied" in deterministic_integrated
    and not clean_integrated_reason
    and "two students copied" in spliced_anchor_text,
    "confirmed author evidence answers validate, prompt, clean, and splice into the target paragraph",
)
scope_summary = _scan_scope_summary({
    "predictability": {
        "sentences": [{"sentence": "one"}, {"sentence": "two"}],
        "all_sentences": [{"sentence": "one"}, {"sentence": "two"}],
        "score_derivation": {
            "included_sentence_count": 2,
            "raw_mean": 0.34851,
        },
    }
})
assert_test(
    scope_summary == {
        "predictability_scored_sentences": 2,
        "predictability_total_sentences": 2,
        "predictability_included_sentence_count": 2,
        "predictability_raw_mean": 0.3485,
    },
    "AI search records predictability scan scope for scored candidates",
)
original_allow_env = os.environ.get("DRAFTPROOF_AI_SEARCH_ALLOW_LLM_AFTER_DETERMINISTIC")
os.environ.pop("DRAFTPROOF_AI_SEARCH_ALLOW_LLM_AFTER_DETERMINISTIC", None)
assert_test(
    _allow_ai_search_llm_after_deterministic(),
    "AI search allows LLM fallback after failed deterministic candidates by default",
)
os.environ["DRAFTPROOF_AI_SEARCH_ALLOW_LLM_AFTER_DETERMINISTIC"] = "0"
assert_test(
    not _allow_ai_search_llm_after_deterministic(),
    "AI search deterministic-only mode remains available by env override",
)
if original_allow_env is None:
    os.environ.pop("DRAFTPROOF_AI_SEARCH_ALLOW_LLM_AFTER_DETERMINISTIC", None)
else:
    os.environ["DRAFTPROOF_AI_SEARCH_ALLOW_LLM_AFTER_DETERMINISTIC"] = original_allow_env
with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as env_file:
    env_file.write("OPENROUTER_API_KEY=test-key-from-file\n")
    env_file.write("LLM_MODEL=test-model-from-file\n")
    env_file.write("# ignored comment\n")
    env_path = env_file.name
original_key_env = os.environ.get("OPENROUTER_API_KEY")
original_model_env = os.environ.get("LLM_MODEL")
os.environ.pop("OPENROUTER_API_KEY", None)
os.environ.pop("LLM_MODEL", None)
loaded_env_keys = _load_local_env(env_path)
assert_test(
    os.environ.get("OPENROUTER_API_KEY") == "test-key-from-file"
    and os.environ.get("LLM_MODEL") == "test-model-from-file"
    and "OPENROUTER_API_KEY" in loaded_env_keys,
    "rewrite pipeline loads local .env keys when shell export is missing",
)
os.environ["OPENROUTER_API_KEY"] = "already-exported"
loaded_again = _load_local_env(env_path)
assert_test(
    os.environ.get("OPENROUTER_API_KEY") == "already-exported"
    and "OPENROUTER_API_KEY" not in loaded_again,
    "rewrite pipeline .env loader does not override exported keys",
)
if original_key_env is None:
    os.environ.pop("OPENROUTER_API_KEY", None)
else:
    os.environ["OPENROUTER_API_KEY"] = original_key_env
if original_model_env is None:
    os.environ.pop("LLM_MODEL", None)
else:
    os.environ["LLM_MODEL"] = original_model_env
try:
    os.unlink(env_path)
except OSError:
    pass

model_env_names = [
    "DRAFTPROOF_PLANNER_MODEL",
    "DRAFTPROOF_GENERATOR_MODEL",
    "DRAFTPROOF_RETRY_MODEL",
    "DRAFTPROOF_REWRITE_MODEL_LOCK",
    "DRAFTPROOF_RETRY_MODEL_ENABLED",
    "DRAFTPROOF_RETRY_MODEL_MAX_CALLS",
    "PLANNER_MODEL",
    "GENERATOR_MODEL",
    "RETRY_MODEL",
    "RETRY_MODEL_ENABLED",
    "RETRY_MODEL_MAX_CALLS",
    "planner_model",
    "generator_model",
    "retry_model",
    "retry_model_enabled",
    "retry_model_max_calls",
]
saved_model_env = {name: os.environ.get(name) for name in model_env_names}
for name in model_env_names:
    os.environ.pop(name, None)
os.environ["DRAFTPROOF_REWRITE_MODEL_LOCK"] = "0"
os.environ["DRAFTPROOF_PLANNER_MODEL"] = "openai/gpt-4.1-mini"
os.environ["DRAFTPROOF_GENERATOR_MODEL"] = "openai/gpt-5-mini"
os.environ["DRAFTPROOF_RETRY_MODEL"] = "openai/gpt-5.2"
roles_disabled = _llm_role_config("fallback-model")
assert_test(
    roles_disabled["planner_model"] == "openai/gpt-4.1-mini"
    and roles_disabled["generator_model"] == "openai/gpt-5-mini"
    and roles_disabled["retry_model"] == "openai/gpt-5.2"
    and roles_disabled["retry_model_enabled"] is False
    and roles_disabled["retry_model_max_calls"] == 0,
    "LLM role config resolves planner/generator/retry models with retry kill switch off by default",
)
assert_test(
    not _retry_model_enabled(),
    "retry model kill switch defaults off",
)
os.environ["DRAFTPROOF_RETRY_MODEL_ENABLED"] = "1"
os.environ["DRAFTPROOF_RETRY_MODEL_MAX_CALLS"] = "2"
roles_enabled = _llm_role_config("fallback-model")
assert_test(
    roles_enabled["retry_model_enabled"] is True
    and roles_enabled["retry_model_max_calls"] == 2,
    "retry model kill switch enables bounded retry-model calls",
)
os.environ.pop("DRAFTPROOF_PLANNER_MODEL", None)
os.environ.pop("DRAFTPROOF_GENERATOR_MODEL", None)
os.environ.pop("DRAFTPROOF_RETRY_MODEL", None)
os.environ["DRAFTPROOF_REWRITE_MODEL_LOCK"] = "openai/gpt-4.1-mini"
roles_locked = _llm_role_config("fallback-model")
assert_test(
    roles_locked["planner_model"] == "openai/gpt-4.1-mini"
    and roles_locked["generator_model"] == "openai/gpt-4.1-mini"
    and roles_locked["retry_model"] == "openai/gpt-4.1-mini",
    "rewrite model lock forces all LLM roles to the approved model",
)
os.environ["DRAFTPROOF_REWRITE_MODEL_LOCK"] = "0"
os.environ.pop("DRAFTPROOF_RETRY_MODEL_ENABLED", None)
os.environ.pop("DRAFTPROOF_RETRY_MODEL_MAX_CALLS", None)
os.environ["planner_model"] = "openai/gpt-4.1-mini"
os.environ["generator_model"] = "openai/gpt-5-mini"
os.environ["retry_model"] = "openai/gpt-5.2"
os.environ["retry_model_enabled"] = "1"
os.environ["retry_model_max_calls"] = "3"
roles_lowercase = _llm_role_config("fallback-model")
assert_test(
    roles_lowercase["planner_model"] == "fallback-model"
    and roles_lowercase["generator_model"] == "fallback-model"
    and roles_lowercase["retry_model"] == "fallback-model"
    and roles_lowercase["retry_model_enabled"] is False
    and roles_lowercase["retry_model_max_calls"] == 0,
    "LLM role config ignores lowercase env names; Koyeb env keys must be uppercase",
)
for name, value in saved_model_env.items():
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value

stale_ai_summary = {
    "rollback_applied": True,
    "rollback_reason": "density batch AI gate failed",
    "attempted_final_text": "Old attempted text",
    "detect_scan_attempted": {"ai_risk_badge": {"ai_likelihood_score": 59.73}},
}
_clear_stale_rollback_for_kept_ai_mitigation(
    stale_ai_summary,
    "AI mitigation search",
)
assert_test(
    stale_ai_summary.get("rollback_applied") is False,
    "kept AI mitigation clears stale density rollback",
)
assert_test(
    "rollback_reason" not in stale_ai_summary
    and "detect_scan_attempted" not in stale_ai_summary,
    "kept AI mitigation removes stale attempted/original-preserved fields",
)
assert_test(
    _ai_search_fast_accept_reason(58.12, 44.14),
    "AI search stops early once deterministic candidate strongly mitigates AI",
)
assert_test(
    _ai_search_fast_accept_reason(57.78, 48.89),
    "AI search accepts deterministic candidate that crosses below 50 percent",
)
assert_test(
    not _ai_search_fast_accept_reason(58.12, 57.88),
    "AI search does not stop early for tiny AI movement",
)

ai_first_report = render_rewrite_report(
    summary={
        "rollback_applied": False,
        "converged": True,
        "ai_first_mitigation": {
            "kept": True,
            "reference_ai": 73.43,
            "rewritten_ai": 57.33,
            "ai_delta": 16.10,
            "soft_followups": ["writing_quality 55.11->70.21"],
        },
        "detect_scan_original_saved": {
            "ai_risk_badge": {"ai_likelihood_score": 73.43, "writing_quality_score": 55.11},
            "findings": {"critical": [], "high": [], "medium": [{}] * 20, "low": [{}] * 44},
        },
        "detect_scan_rewritten": {
            "ai_risk_badge": {"ai_likelihood_score": 57.33, "writing_quality_score": 70.21},
            "findings": {"critical": [], "high": [], "medium": [{}] * 2, "low": [{}] * 3},
        },
    },
    sentence_comparison=[],
    ai_findings=[],
)
assert_test("**AI Mitigated**" in ai_first_report, "AI-first report labels kept AI mitigation")
assert_test(
    "Writing-quality or lower-severity changes are follow-up work" in ai_first_report,
    "AI-first report explains quality follow-up instead of rollback",
)

comparison_mp = SimpleNamespace(
    original_text="A one. B two. C three. D four.",
    final_text="A one. B two rewritten with C three combined. D four.",
    original_metrics=SimpleNamespace(sentence_details=[]),
    final_metrics=SimpleNamespace(sentence_details=[]),
)
comparison_rows = _build_aligned_sentence_comparison(comparison_mp)
changed_rows = [
    row for row in comparison_rows
    if row.get("orig_sentence") != row.get("new_sentence")
]
assert_test(
    all(row.get("new_sentence") for row in changed_rows),
    "sentence comparison groups unequal replace blocks without blank rewritten cells",
)
large_replace_mp = SimpleNamespace(
    original_text="A one. B two. C three. D four. E five.",
    final_text="First replacement. Second replacement. Third replacement. Fourth replacement. Fifth replacement.",
    original_metrics=SimpleNamespace(sentence_details=[]),
    final_metrics=SimpleNamespace(sentence_details=[]),
)
large_replace_rows = _build_aligned_sentence_comparison(large_replace_mp)
assert_test(
    len(large_replace_rows) == 5
    and all(len(row.get("new_sentence", "").split(".")) <= 2 for row in large_replace_rows),
    "sentence comparison keeps large replace blocks sentence-level instead of one giant new sentence",
)
heading_comparison_mp = SimpleNamespace(
    original_text=(
        "The standard stays the same, but the learning path must be more transparent.\n"
        "Conclusion\n"
        "This review has discussed inclusive learning design.\n"
        "Demonstration alone does not build competency."
    ),
    final_text=(
        "The standard stays the same, but the learning path must be more transparent.\n"
        "Conclusion\n"
        "This review has discussed inclusive learning design.\n"
        "Demonstration alone does not build competency."
    ),
    original_metrics=SimpleNamespace(sentence_details=[]),
    final_metrics=SimpleNamespace(sentence_details=[]),
)
heading_rows = _build_aligned_sentence_comparison(heading_comparison_mp)
assert_test(
    any(row.get("orig_sentence") == "Conclusion" for row in heading_rows)
    and all("Conclusion This review" not in row.get("orig_sentence", "") for row in heading_rows),
    "sentence comparison keeps newline headings separate from conclusion prose",
)

driver_plan = build_mitigation_plan(
    plan=None,
    raw_json={
        "ai_risk_badge": {
            "ai_components": {"unsupported_claim_risk": 90.0},
            "writing_components": {"source_grounding_risk": 70.0},
        },
        "rewrite_plan": {
            "auto_target_context": [{
                "finding": {
                    "finding_id": "f001",
                    "title": "medium_predictability",
                    "sentence_id": "s001",
                    "evidence": "School is no longer the only place to acquire knowledge.",
                    "rewrite_context": {
                        "paragraph_id": "p001",
                        "previous_sentence": "",
                        "next_sentence": "Online information now fills students' daily lives.",
                        "paragraph_excerpt": (
                            "School is no longer the only place to acquire knowledge. "
                            "Online information now fills students' daily lives."
                        ),
                        "signal_instruction": "Break the common-word path and ground the claim in nearby context.",
                    },
                }
            }]
        },
    },
)
assert_test(driver_plan["counts"]["needs_source_or_example"] == 2, "component drivers produce evidence guidance counts")
assert_test(driver_plan["primary_mode"] == "guided_revision", "component drivers set guided revision mode")
assert_test(driver_plan["score_mitigation_targets"][0]["component"] == "unsupported_claim_risk", "score mitigation targets prioritize largest evidence driver")
assert_test(driver_plan["risk_mitigation_actions"][0]["action_type"] == "soften_or_support_claim", "risk actions turn score drivers into concrete mitigation actions")
assert_test(
    any(item["action_type"] == "connect_source_to_claim" for item in driver_plan["risk_mitigation_actions"]),
    "risk actions include source-to-claim mitigation",
)
marked = driver_plan["marked_content_suggestions"]
assert_test(marked[0]["auto_apply"] is False, "marked content suggestions are not auto-applied")
assert_test("[ADD SOURCE OR EXAMPLE]" in marked[0]["suggested_addition"], "marked suggestions visibly bracket new content")
assert_test("School is no longer" in marked[0]["target_text"], "marked suggestions point to concrete target text")
assert_test("Sentence s001" in marked[0]["where"], "marked suggestions include concrete sentence location")
assert_test(
    any(item["action_type"] == "add_source_bridge" for item in marked),
    "marked suggestions include source bridge structure",
)
assert_test(driver_plan["reference_patterns"][0]["component"] == "unsupported_claim_risk", "guided reference patterns prioritize evidence drivers")
density_plan = build_mitigation_plan(
    plan=None,
    raw_json={
        "ai_risk_badge": {
            "ai_components": {
                "qualifying_text_ai_density": 77.78,
                "generic_assertion_risk": 90.0,
            },
            "writing_components": {
                "unsupported_claim_risk": 80.0,
                "source_grounding_risk": 70.0,
            },
        },
        "rewrite_plan": {
            "auto_target_context": [{
                "finding": {
                    "finding_id": "f001",
                    "title": "medium_predictability",
                    "sentence_id": "s001",
                    "evidence": "This does not reduce the complexity of the task; rather, it clarifies the pathway toward competency.",
                    "rewrite_context": {
                        "paragraph_id": "p001",
                        "paragraph_excerpt": (
                            "The learner may keep cutting, but they cannot trace whether the problem came from the diagonal parting. "
                            "This does not reduce the complexity of the task; rather, it clarifies the pathway toward competency."
                        ),
                    },
                }
            }]
        },
    },
)
assert_test(
    any(item["component"] == "qualifying_text_ai_density" and item["bucket"] == "paragraph_rebuild" for item in density_plan["component_drivers"]),
    "qualifying text density becomes paragraph rebuild driver",
)
assert_test(
    any(item["action_type"] == "paragraph_density_rebuild" for item in density_plan["risk_mitigation_actions"]),
    "density driver produces paragraph rebuild action",
)
assert_test(
    any(item["action_type"] == "rebuild_paragraph_density" for item in density_plan["marked_content_suggestions"]),
    "density driver produces marked paragraph rebuild suggestion",
)
density_suggestion = next(
    item for item in density_plan["marked_content_suggestions"]
    if item["action_type"] == "rebuild_paragraph_density"
)
assert_test(
    "The learner may keep cutting" in density_suggestion["target_text"],
    "density suggestion uses paragraph context instead of narrow sentence",
)
assert_test(
    not density_suggestion["target_text"].endswith("c"),
    "density suggestion does not quote truncated sentence fragments",
)
guided_report = render_rewrite_report(
    summary={
        "no_text_change": True,
        "converged": False,
        "mitigation_plan": driver_plan,
        "detect_scan_original_saved": {
            "ai_risk_badge": {"ai_likelihood_score": 40.0, "writing_quality_score": 50.0},
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
        },
        "detect_scan_rewritten": {
            "ai_risk_badge": {"ai_likelihood_score": 40.0, "writing_quality_score": 50.0},
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
        },
    },
    sentence_comparison=[],
    ai_findings=[],
)
assert_test("## Guided Revision Checklist" in guided_report, "guided report highlights revision checklist")
assert_test("automatic sentence edits cannot safely supply" in guided_report, "guided report explains why original is preserved")
assert_test("## Risk Score Mitigation Targets" in guided_report, "guided report shows score mitigation targets")
assert_test("## Risk Mitigation Actions" in guided_report, "guided report shows concrete mitigation actions")
assert_test("## Suggested Additions For Review" in guided_report, "guided report shows marked content suggestions")
assert_test("[ADD SOURCE OR EXAMPLE]" in guided_report, "guided report highlights bracketed suggested content")
assert_test("School is no longer the only place" in guided_report, "guided report shows concrete target text for suggestions")

partial_ok, partial_reason, partial_delta = _target_predictability_acceptance(
    {"risk": 0.5083, "label": "medium"},
    {"risk": 0.4612, "label": "medium"},
)
assert_test(partial_ok, "predictability gate accepts meaningful medium-band reduction")
assert_test(partial_reason == "target_reduced_but_still_medium", "predictability gate records partial reduction reason")
assert_test(partial_delta > 0.04, "predictability gate exposes local reduction amount")
tiny_ok, tiny_reason, _ = _target_predictability_acceptance(
    {"risk": 0.5083, "label": "medium"},
    {"risk": 0.5000, "label": "medium"},
)
assert_test(not tiny_ok and "target_reduction_too_small" in tiny_reason, "predictability gate still rejects tiny no-op reductions")

throttle_text = (
    "Students use the chart to plan the cut. "
    "Teachers use the chart to explain the next section."
)
throttle_findings = [
    Finding(
        finding_type="medium_predictability",
        risk_level="medium",
        evidence_strength="moderate",
        detail="predictable",
        evidence="Students use the chart to plan the cut.",
        recommendation="rewrite",
        suggested_action_type="auto",
        actionability="auto_fixable",
        location={"sentence_index": 0, "sentence_id": "s001"},
        metadata={"finding_id": "f001", "scanner": "predictability"},
    ),
    Finding(
        finding_type="medium_predictability",
        risk_level="medium",
        evidence_strength="moderate",
        detail="predictable",
        evidence="Teachers use the chart to explain the next section.",
        recommendation="rewrite",
        suggested_action_type="auto",
        actionability="auto_fixable",
        location={"sentence_index": 1, "sentence_id": "s002"},
        metadata={"finding_id": "f002", "scanner": "predictability"},
    ),
]
throttle_context = DetectJSONContext(
    detect_results=[make_detect_result(throttle_findings)],
    input_text=throttle_text,
    raw_json={
        "ai_risk_badge": {
            "ai_likelihood_score": 40.0,
            "ai_components": {"unsupported_claim_risk": 90.0},
            "writing_components": {"source_grounding_risk": 70.0},
        }
    },
)
throttle_calls = []
throttle_result = run_rewrite(
    throttle_text,
    [make_detect_result(throttle_findings)],
    rewrite_fn=lambda text, prompt: throttle_calls.append((text, prompt)) or "1. Students check the chart before planning the cut.",
    config=RewriteConfig(max_auto_targets=6, max_llm_calls=6),
    rewrite_context=throttle_context,
    ai_only=False,
)
assert_test(len(throttle_calls) <= 3, "guided revision throttles automatic rewrite calls")
assert_test(
    throttle_result.summary.get("rewrite_effective_config", {}).get("effective_auto_target_limit") == 2,
    "guided revision records effective auto-target limit",
)

suggestion_only_context = DetectJSONContext(
    detect_results=[make_detect_result(throttle_findings[:1])],
    input_text=throttle_text,
    raw_json={
        "ai_risk_badge": {
            "ai_likelihood_score": 40.0,
            "ai_components": {"unsupported_claim_risk": 90.0},
            "writing_components": {"source_grounding_risk": 70.0},
        }
    },
)
suggestion_only_result = run_rewrite(
    throttle_text,
    [make_detect_result(throttle_findings[:1])],
    rewrite_fn=lambda text, prompt: "1. Students apply structured guidance materials during repeated practice activities.",
    config=RewriteConfig(max_auto_targets=1, max_llm_calls=1),
    rewrite_context=suggestion_only_context,
    ai_only=False,
)
assert_test(
    suggestion_only_result.summary.get("outcome") == "suggestion_only",
    "marked suggestions change no-kept-edit outcome to suggestion_only",
)

pipeline_no_change = run_rewrite_pipeline(
    detect_json={
        "input_text": "Students use the chart.",
        "ai_risk_badge": {"ai_likelihood_score": 40.0, "writing_quality_score": 50.0},
        "findings": {"critical": [], "high": [], "medium": [], "low": []},
    },
    output_dir=tempfile.mkdtemp(prefix="draftproof-test-"),
)
assert_test(pipeline_no_change["status"] == "clean", "pipeline skips when no rewrite is needed")

guided_pipeline_text = (
    "Inclusive learning design in adult VET hairdressing training is part of teaching technical skills. "
    "This shows that learning is important and can improve outcomes for learners. "
    "When learners pause, they may need a short moment to process what they have just practised. "
) * 6
_previous_ai_search = os.environ.get("DRAFTPROOF_AI_MITIGATION_SEARCH")
os.environ["DRAFTPROOF_AI_MITIGATION_SEARCH"] = "0"
try:
    guided_pipeline = run_rewrite_pipeline(
        detect_json={
            "input_text": guided_pipeline_text,
            "ai_risk_badge": {
                "ai_likelihood_score": 57.78,
                "writing_quality_score": 54.12,
                "ai_components": {
                    "topk_pattern": 67.7,
                    "generic_assertion_risk": 90.0,
                    "qualifying_text_ai_density": 69.88,
                },
                "writing_components": {
                    "unsupported_claim_risk": 70.0,
                    "source_grounding_risk": 70.0,
                },
            },
            "rewrite_decision": {"run_rewrite": True, "mode": "targeted"},
            "rewrite_plan": {
                "mode": "targeted",
                "overall_action": "predictability_revision",
                "auto_fixable": [{"finding_id": "f001"}],
            },
            "findings": {
                "critical": [],
                "high": [],
                "medium": [{
                    "finding_id": "f001",
                    "category": "predictability",
                    "scanner": "predictability",
                    "title": "medium_predictability",
                    "adjusted_risk": "medium",
                    "actionability": "auto_fixable",
                    "sentence_id": "s002",
                    "evidence": "This shows that learning is important and can improve outcomes for learners.",
                    "recommendation": "Rewrite with concrete classroom detail.",
                    "score": 0.52,
                }],
                "low": [],
            },
        },
        output_dir=tempfile.mkdtemp(prefix="draftproof-test-guided-"),
        api_key=None,
    )
finally:
    if _previous_ai_search is None:
        os.environ.pop("DRAFTPROOF_AI_MITIGATION_SEARCH", None)
    else:
        os.environ["DRAFTPROOF_AI_MITIGATION_SEARCH"] = _previous_ai_search
guided_summary = guided_pipeline["result"].summary
assert_test(
    guided_summary.get("ai_mitigation_blocked_auto_rewrite") is False,
    "pipeline does not block automatic mitigation for author-evidence gaps by default",
)
assert_test(
    guided_summary.get("ai_mitigation_search", {}).get("reason") != "requires_author_input",
    "pipeline no longer labels author-evidence gaps as an AI search blocker",
)
assert_test(
    guided_summary.get("ai_mitigation_search", {}).get("enabled") is not True,
    "test disables AI search while preserving guidance output",
)
marked_mitigation_rewrite = guided_summary.get("marked_mitigation_rewrite") or {}
assert_test(
    marked_mitigation_rewrite.get("draft_text") and "[[ADD VERIFIED DETAIL:" in marked_mitigation_rewrite.get("draft_text"),
    "guided mitigation produces marked rewrite content",
)
assert_test(
    marked_mitigation_rewrite.get("auto_apply") is False,
    "marked mitigation rewrite is not auto-applied",
)


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("16. DENSITY PARAGRAPH MITIGATION")
print("=" * 70)

density_text = (
    "The learner practises consultation skills before cutting hair. The teacher checks the plan.\n\n"
    "Inclusive Learning Design in Certificate III Hairdressing is important because it helps all learners succeed "
    "in a modern learning environment. This shows that teachers should provide support and guidance while using "
    "technology and different methods. Overall, the approach creates better outcomes for students while still "
    "covering the required salon skills."
)
density_context = DetectJSONContext(
    detect_results=[make_detect_result([])],
    input_text=density_text,
    raw_json={
        "ai_risk_badge": {
            "ai_components": {
                "qualifying_text_ai_density": 82.0,
                "generic_assertion_risk": 76.0,
            }
        },
        "domain_profile": {
            "matched_domain_terms": [
                "Certificate III Hairdressing",
                "consultation",
                "salon skills",
                "teacher",
                "learners",
            ]
        },
        "rewrite_edit_briefs": [
            {
                "finding_id": "f-density",
                "paragraph_excerpt": (
                    "Inclusive Learning Design in Certificate III Hairdressing is important because it helps all learners succeed "
                    "in a modern learning environment."
                ),
                "signals": {"finding_type": "medium_predictability"},
            }
        ],
    },
)
density_plan = build_mitigation_plan(
    RewritePlan(
        actions=[],
        auto_fixable=[],
        manual_required=[],
        protected=[],
        review_only=[],
        total_weighted_risk=0.0,
        auto_risk=0.0,
        manual_risk=0.0,
        protected_risk=0.0,
        rewritable_risk=0.0,
    ),
    density_context.raw_json,
)
density_idx, density_para, density_meta = _select_density_paragraph(
    density_text,
    density_context,
    density_plan,
)
assert_test(density_idx == 1, "density pass selects high-signal paragraph")
assert_test(density_meta.get("matched_items", 0) >= 1, "density selection uses scan pointers")

density_prompt = _density_paragraph_prompt(density_para, density_context, density_plan)
assert_test("<TARGET>" in density_prompt and "</TARGET>" in density_prompt, "density prompt marks target paragraph")
assert_test("Output exactly one replacement paragraph" in density_prompt, "density prompt requests one paragraph")
assert_test("qualifying_text_ai_density=82.0%" in density_prompt, "density prompt includes component score")
assert_test("Certificate III Hairdressing" in density_prompt, "density prompt includes domain anchors")
assert_test("Named entities that must remain unchanged" in density_prompt, "density prompt preserves named entities")
assert_test("digital landscape" in density_prompt, "density prompt includes anti-polish examples")
assert_test("sentence total reconstruction" in density_prompt.lower(), "density prompt asks for total reconstruction")
assert_test("Change at least 70% of sentence openings" in density_prompt, "density prompt requires changed sentence openings")
assert_test("challenge, phase, setting" in density_prompt, "density prompt forbids weak polished substitutions")
density_repair_prompt = _density_repair_prompt(
    density_para,
    "Inclusive Learning Design in Certificate III Hairdressing is taught inside the haircutting task.",
    "semantic_drift lost_named_entity: 'Box Hill Institute'",
    density_context,
    density_plan,
)
assert_test("Previous rejected candidate" in density_repair_prompt, "density repair prompt includes rejected candidate")
assert_test("Box Hill Institute" in density_repair_prompt, "density repair prompt repeats lost named entity")
assert_test("Repair only the safety failure" in density_repair_prompt, "density repair prompt limits retry scope")
assert_test(
    _density_entity_only_drift("semantic_drift lost_named_entity: 'Box Hill Institute'"),
    "density entity-only drift can be downgraded after repair",
)
assert_test(
    not _density_entity_only_drift("semantic_drift number_changed: '2024'"),
    "density entity downgrade does not allow number drift",
)
near_copy_density_candidate = density_para.replace(
    "should not sit outside technical skill teaching",
    "should not be separate from technical skill teaching",
)
assert_test(
    _density_transformation_too_small(density_para, near_copy_density_candidate),
    "density transformation guard catches weak paragraph polishing",
)
assert_test(
    _density_paragraph_reject_reason(density_para, near_copy_density_candidate, density_context)
    == "density_transformation_too_small",
    "density guard rejects weak paragraph polishing",
)
transformation_repair_prompt = _density_repair_prompt(
    density_para,
    near_copy_density_candidate,
    "density_transformation_too_small",
    density_context,
    density_plan,
)
assert_test("sentence footprint" in transformation_repair_prompt, "density repair prompt handles weak transformation")


class FakeDensityScanner:
    def scan_text(self, text):
        if "bad candidate" in text:
            risk, top10 = 0.72, 0.80
        elif "strong candidate" in text:
            risk, top10 = 0.44, 0.48
        else:
            risk, top10 = 0.68, 0.76
        sent = SimpleNamespace(
            predictability_risk=risk,
            top_10_ratio=top10,
            avg_surprisal=3.0,
            risk_label="medium",
            sentence=text,
        )
        return {"sentences": [sent]}


local_ok, local_reason, local_signal = _density_local_signal_acceptance(
    FakeDensityScanner(),
    "original density paragraph",
    "strong candidate density paragraph",
)
assert_test(local_ok, "density local signal gate accepts detector-improving paragraph")
assert_test(local_signal["risk_delta"] > 0, "density local signal records risk reduction")
local_bad_ok, local_bad_reason, _ = _density_local_signal_acceptance(
    FakeDensityScanner(),
    "original density paragraph",
    "bad candidate density paragraph",
)
assert_test(not local_bad_ok, "density local signal gate rejects detector-regressing paragraph")
assert_test("density_local_signal_regressed" in local_bad_reason, "density local signal rejection is explicit")
class SlightRiskImprovementScanner:
    def scan_text(self, text):
        if "candidate" in text:
            risk, top10 = 0.3480, 0.4306
        else:
            risk, top10 = 0.3583, 0.4447
        sent = SimpleNamespace(
            predictability_risk=risk,
            top_10_ratio=top10,
            avg_surprisal=3.0,
            risk_label="medium",
            sentence=text,
        )
        return {"sentences": [sent]}


slight_ok, slight_reason, slight_signal = _density_local_signal_acceptance(
    SlightRiskImprovementScanner(),
    "original density paragraph",
    "candidate density paragraph",
)
assert_test(slight_ok, "density local signal gate accepts measurable AI-risk improvement")
assert_test(slight_signal["risk_delta"] > 0, "density local signal keeps measurable risk delta")


class BestDensityCandidateScanner:
    def scan_text(self, text):
        if "During the training room task" in text:
            risk, top10 = 0.44, 0.48
        elif "technical accuracy issue" in text:
            risk, top10 = 0.72, 0.80
        else:
            risk, top10 = 0.68, 0.76
        sent = SimpleNamespace(
            predictability_risk=risk,
            top_10_ratio=top10,
            avg_surprisal=3.0,
            risk_label="medium",
            sentence=text,
        )
        return {"sentences": [sent]}


best_density_original = (
    "The learner may keep cutting in the training room while the teacher watches each step "
    "and checks the section."
)
weak_density_candidate = (
    "This technical accuracy issue is visible in the learning framework and digital landscape."
)
strong_density_candidate = (
    "During the training room task, the learner cuts each section while the teacher watches "
    "and checks each step."
)
best_density_text, best_density_reason, best_density_evals = _select_best_density_candidate(
    BestDensityCandidateScanner(),
    best_density_original,
    [
        ("primary", weak_density_candidate),
        ("ai_stronger", strong_density_candidate),
    ],
    None,
)
assert_test(best_density_text == strong_density_candidate, "density selector keeps strongest AI-reducing candidate")
assert_test(best_density_reason == "", "density selector returns no rejection for accepted best candidate")
assert_test(len(best_density_evals) == 2, "density selector records both candidate evaluations")
assert_test(
    any(item["label"] == "ai_stronger" and not item["rejection_reason"] for item in best_density_evals),
    "density selector evaluation records accepted stronger candidate",
)
rescue_prompt = _density_rescue_prompt(density_text, density_context, density_plan)
assert_test("AI-first density rescue pass" in rescue_prompt, "density rescue prompt identifies AI-first rescue")
assert_test("break the repeated long-form AI-style pattern" in rescue_prompt, "density rescue prompt targets document pattern")
assert_test("synonym swaps and light polishing" in rescue_prompt, "density rescue prompt rejects light polishing")
assert_test("Output the complete rewritten draft only" in rescue_prompt, "density rescue prompt requires full draft output")
assert_test("qualifying_text_ai_density=82.0%" in rescue_prompt, "density rescue prompt includes density driver")
assert_test("Certificate III Hairdressing" in rescue_prompt, "density rescue prompt preserves named entities")
assert_test(
    rescue_prompt.rfind("Return the complete rewritten draft only") > rescue_prompt.rfind("</TARGET_DOCUMENT>"),
    "density rescue final output instruction appears after target document",
)
rescue_total_prompt = _density_rescue_prompt(
    density_text,
    density_context,
    density_plan,
    rescue_mode="total_reconstruction",
)
assert_test("Rescue mode: total reconstruction" in rescue_total_prompt, "density rescue supports total reconstruction mode")
rescue_structure_prompt = _density_rescue_prompt(
    density_text,
    density_context,
    density_plan,
    rescue_mode="paragraph_restructure",
)
assert_test("Rescue mode: paragraph restructuring" in rescue_structure_prompt, "density rescue supports paragraph restructuring mode")
rescue_process_prompt = _density_rescue_prompt(
    density_text,
    density_context,
    density_plan,
    rescue_mode="process_voice",
)
assert_test("Rescue mode: concrete process voice" in rescue_process_prompt, "density rescue supports process voice mode")
cleaned_rescue = _clean_density_rescue_output(
    "Final draft:\n\nFirst rewritten paragraph.\n\n\nSecond rewritten paragraph.",
    "Original paragraph.",
)
assert_test(
    cleaned_rescue == "First rewritten paragraph.\n\nSecond rewritten paragraph.",
    "density rescue cleaner preserves paragraph breaks",
)
retry_rescue_prompt = _density_rescue_retry_prompt(
    rescue_prompt,
    "candidate_too_short 190<2900",
    density_text,
)
assert_test("previous answer was rejected" in retry_rescue_prompt, "density rescue retry explains invalid output")
assert_test("complete rewritten draft" in retry_rescue_prompt, "density rescue retry requires full draft")
assert_test("Final instruction: output only" in retry_rescue_prompt, "density rescue retry ends with output-only instruction")
local_repair_prompt = _density_repair_prompt(
    density_para,
    "bad candidate density paragraph",
    local_bad_reason,
    density_context,
    density_plan,
)
assert_test("local GPT-2 detector check" in local_repair_prompt, "density repair prompt handles local detector rejection")
polish_repair_prompt = _density_repair_prompt(
    density_para,
    "This is especially true when learners move from observing a demonstration to cutting a controlled haircut themselves.",
    "generic_polish_increase",
    density_context,
    density_plan,
)
assert_test("polish/formality" in polish_repair_prompt, "density repair prompt handles polish rejection")
assert_test("especially true" in polish_repair_prompt, "density repair prompt forbids latest polished phrasing")

bad_density_candidate = (
    "Inclusive Learning Design in Certificate III Hairdressing provides a visible learning framework in a digital landscape, "
    "preserving technical rigor while learners encounter operational obstacles across complex outcomes."
)
assert_test(
    bool(_density_paragraph_reject_reason(density_para, bad_density_candidate, density_context)),
    "density guard rejects polished abstraction patterns",
)

single_paragraph_doc = (
    "Sentence one starts the document. Sentence two is the dense target. "
    "Sentence three is also part of the target. Sentence four must remain. "
    "Sentence five must also remain."
)
density_window = "Sentence two is the dense target. Sentence three is also part of the target."
spliced_density = _splice_density_candidate(
    single_paragraph_doc,
    0,
    density_window.replace("  ", " "),
    "Replacement sentence A. Replacement sentence B.",
    {"region_type": "sentence_window", "sentence_start": 1, "sentence_count": 2},
)
assert_test("Sentence one starts the document." in spliced_density, "density splice preserves text before sentence window")
assert_test("Sentence four must remain." in spliced_density, "density splice preserves text after sentence window")
assert_test("Replacement sentence A." in spliced_density, "density splice inserts replacement window")

long_density_text = (
    "Opening note before the dense section.\n\n"
    + " ".join(
        f"Background sentence {i} keeps the earlier section moving without much scanner relevance."
        for i in range(1, 18)
    )
    + " Inclusive Learning Design in Certificate III Hairdressing is important because it helps all learners succeed "
      "in a modern learning environment. This shows that teachers should provide support and guidance while using "
      "technology and different methods. Overall, the approach creates better outcomes for students while still "
      "covering the required salon skills. "
    + " ".join(
        f"Later sentence {i} continues a different part of the essay without needing the first mitigation pass."
        for i in range(1, 18)
    )
)
long_density_idx, long_density_region, long_density_meta = _select_density_paragraph(
    long_density_text,
    density_context,
    density_plan,
)
assert_test(
    len(long_density_region.split()) < 330,
    "density selector uses bounded window inside oversized paragraph",
)
assert_test(
    long_density_meta.get("region_type") == "sentence_window",
    "density selector records sentence-window region type",
)
assert_test(
    "Inclusive Learning Design in Certificate III Hairdressing" in long_density_region,
    "density window keeps scan-matched paragraph content",
)

density_first_finding = Finding(
    finding_type="generic_phrase",
    risk_level="medium",
    evidence_strength="moderate",
    detail="generic phrase",
    evidence="modern learning environment",
    recommendation="rewrite",
    suggested_action_type="auto",
    actionability="auto_fixable",
    location={"sentence_index": 2, "sentence_id": "s003"},
    metadata={"finding_id": "density-first", "scanner": "ai_generation"},
)
density_first_calls = []


def density_first_rewrite_fn(text, prompt):
    density_first_calls.append((text, prompt))
    if "Density paragraph mitigation pass" in prompt:
        return (
            "Inclusive Learning Design in Certificate III Hairdressing helps learners connect support, technology, "
            "and salon skills in the same lesson. The teacher can check how each learner uses the method while still "
            "covering the required salon skills."
        )
    return "1. The class uses technology and support while covering the required salon skills."


density_first_result = run_rewrite(
    density_text,
    [make_detect_result([density_first_finding])],
    rewrite_fn=density_first_rewrite_fn,
    config=RewriteConfig(max_auto_targets=6, max_llm_calls=1),
    rewrite_context=density_context,
    ai_only=False,
)
assert_test(
    density_first_calls and "Density paragraph mitigation pass" in density_first_calls[0][1],
    "density paragraph mitigation runs before sentence rewrites",
)
assert_test(
    density_first_result.summary.get("density_paragraph_pass", {}).get("phase") == "before_sentence_rewrites",
    "density paragraph pass records first-priority phase",
)
assert_test(
    density_first_result.summary.get("rewrite_effective_config", {}).get("density_mitigation_priority") == "before_sentence_rewrites",
    "effective config records density-first priority",
)

near_threshold_density_context = DetectJSONContext(
    detect_results=[make_detect_result([])],
    input_text=density_text,
    raw_json={
        "ai_risk_badge": {
            "ai_likelihood_score": 59.73,
            "ai_components": {
                "qualifying_text_ai_density": 69.93,
                "generic_assertion_risk": 76.0,
            },
        },
        "domain_profile": density_context.raw_json["domain_profile"],
        "rewrite_edit_briefs": density_context.raw_json["rewrite_edit_briefs"],
    },
)
near_threshold_calls = []


def near_threshold_rewrite_fn(text, prompt):
    near_threshold_calls.append((text, prompt))
    if "Density paragraph mitigation pass" in prompt:
        return (
            "In Certificate III Hairdressing, learners practise support, technology, and salon skills together. "
            "The teacher checks how the method is used while the required salon skills are still covered."
        )
    return "1. The class uses support while covering salon skills."


near_threshold_result = run_rewrite(
    density_text,
    [make_detect_result([density_first_finding])],
    rewrite_fn=near_threshold_rewrite_fn,
    config=RewriteConfig(max_auto_targets=0, max_llm_calls=1),
    rewrite_context=near_threshold_density_context,
    ai_only=False,
)
assert_test(
    near_threshold_calls and "Density paragraph mitigation pass" in near_threshold_calls[0][1],
    "AI-risk near-threshold density still triggers paragraph mitigation",
)
assert_test(
    near_threshold_result.summary.get("density_paragraph_pass", {}).get("density_threshold") == 60.0,
    "AI-risk density threshold lowers to 60 for mitigation",
)

short_budget_text = "Short mitigation paragraph. " * 20
long_budget_text = "Long mitigation paragraph. " * 240
assert_test(
    _adaptive_budget_default(short_budget_text, 2, 6) == "2",
    "adaptive budget uses short-document default below threshold",
)
assert_test(
    _adaptive_budget_default(long_budget_text, 2, 6) == "6",
    "adaptive budget keeps long-document default above threshold",
)

paragraph_target_text = (
    "I watch learners in SHBHCUT006 when they move from sectioning to projection. "
    "In my class I can see the guide disappear, and my correction usually starts with comb angle, elbow position, and tension. "
    "I slow the section down, check the mannequin, and ask learners to show where the next subsection should sit before cutting.\n\n"
    "Inclusive learning design is important because it supports students and helps create better outcomes. "
    "This shows that the framework is significant and can improve access for different learners in a range of settings. "
    "The issue is important because learners need support and guidance across different classroom situations. "
    "This can help learning because students require clear structures, suitable adjustments, and useful pathways for progress.\n\n"
    "This review has discussed inclusive learning design. "
    "It is important for teachers to support learners and create better outcomes."
)
paragraph_targets = _paragraph_component_targets(
    paragraph_target_text,
    {"rewrite_edit_briefs": []},
    limit=5,
)
paragraph_roles = [target.get("role") for target in paragraph_targets]
assert_test(
    paragraph_roles
    and paragraph_roles[0] in {"generic_claim_heavy", "conclusion_template_risk"}
    and (
        "human_anchor_rich" not in paragraph_roles
        or paragraph_roles.index("human_anchor_rich") > 0
    ),
    "paragraph-component targeting prioritizes generic/conclusion blockers over human-anchor-rich paragraphs",
)
short_paragraph_target_text = (
    "Students now use websites, AI tools, social media, and short videos when they study.\n\n"
    "I think the harder part is deciding what to trust because not every answer shows its source or context.\n\n"
    "Teachers still matter because they can slow the process down and ask students why an answer is useful.\n\n"
    "Assessment also needs attention because polished work can hide weak understanding."
)
previous_adaptive_short_targets = os.environ.get("DRAFTPROOF_ADAPTIVE_SHORT_PARAGRAPH_TARGETS")
try:
    os.environ["DRAFTPROOF_ADAPTIVE_SHORT_PARAGRAPH_TARGETS"] = "1"
    short_paragraph_targets = _paragraph_component_targets(
        short_paragraph_target_text,
        {
            "ai_risk_badge": {
                "writing_components": {"unsupported_claim_risk": 80.0, "source_grounding_risk": 70.0},
                "ai_components": {"generic_assertion_risk": 65.0},
            }
        },
        limit=4,
    )
finally:
    if previous_adaptive_short_targets is None:
        os.environ.pop("DRAFTPROOF_ADAPTIVE_SHORT_PARAGRAPH_TARGETS", None)
    else:
        os.environ["DRAFTPROOF_ADAPTIVE_SHORT_PARAGRAPH_TARGETS"] = previous_adaptive_short_targets
assert_test(
    len(short_paragraph_targets) >= 2
    and all(len((target.get("paragraph") or "").split()) < 45 for target in short_paragraph_targets[:2]),
    "paragraph-component targeting has opt-in short-paragraph adaptation instead of defaulting to expensive broad targeting",
)
code_anchor_loss_reason = _ai_search_protected_loss_reason(
    "The HBB26 group uses SHBHCUT006 with 6 learners.",
    "The group uses SHBHCUT006 with 6 learners.",
    detect_protected_spans("The HBB26 group uses SHBHCUT006 with 6 learners."),
)
assert_test(
    code_anchor_loss_reason == "code_anchor_lost:hbb26",
    "AI-search protected-span guard preserves course/intake code anchors",
)

target_push_text = (
    "Opening paragraph keeps the topic clear for the reader.\n\n"
    "The real challenge now is knowing what to trust. "
    "Students can use many online tools, and this is important because it helps learning in many ways. "
    "This shows that education needs to change for the modern world. "
    "The issue is significant because students need support and guidance. "
    "This can create better outcomes because the system can help students learn more effectively over time.\n\n"
    "This is why teachers still play an important role. "
    "Teachers can help students ask questions and understand information. "
    "This is useful because it improves learning and creates better outcomes. "
    "Schools should focus on skills and knowledge for the future. "
    "This means assessment should support learning, feedback, and improvement in a broad educational context.\n\n"
    "The conclusion is that education should prepare students for change."
)
target_push_report = {
    "integrity_layers": {
        "layers": {
            "human_contribution_signal": {"score": 55.0},
            "ai_transformation_risk": {"score": 45.0},
            "ai_authorship_risk": {"score": 56.0},
        }
    },
    "ai_risk_badge": {
        "ai_components": {"generic_assertion_risk": 90.0, "predictability": 75.0},
        "writing_components": {"unsupported_claim_risk": 90.0, "broad_claim_risk": 80.0},
    },
}
target_push_candidates = _post_safe_win_target_push_candidates(
    target_push_text,
    target_push_report,
    limit=4,
)
assert_test(
    bool(target_push_candidates)
    and all(strategy.startswith("post_safe_target_push_") for strategy, _candidate, _meta in target_push_candidates),
    "post-safe-win target push generates bounded deterministic candidates below target",
)
construction_candidates = _human_signal_construction_candidates(
    target_push_text,
    target_push_report,
    limit=2,
)
assert_test(
    bool(construction_candidates)
    and any(
        meta.get("operation") == "human_signal_construction"
        and meta.get("changed_sentence_frames", 0) >= 3
        for _strategy, _candidate, meta in construction_candidates
    ),
    "human signal construction builds section-level author-density candidates",
)
anchor_contract = _human_anchor_driver_contract(
    {
        "ai_risk_badge": {
            "writing_components": {
                "lived_detail_risk": 80.0,
                "domain_grounding_strength": 100.0,
            },
            "transformation_classification": {
                "features": {
                    "human_anchor_score": 0.36,
                    "rewrite_smoothness": 0.62,
                    "source_similarity": 0.0,
                    "surface_similarity": 0.0,
                },
            },
        }
    },
    text=target_push_text,
)
assert_test(
    anchor_contract["next_lived_detail_band"]["risk"] == 65.0
    and anchor_contract["required_anchor_sentences_for_next_band"] >= 1
    and anchor_contract["human_raw_formula"]["before"]["anchor_component"] == 16.2,
    "human anchor driver contract exposes lived-detail band and human_raw contribution",
)
anchor_candidates = _human_anchor_amplifier_candidates(
    target_push_text,
    target_push_report,
    limit=3,
)
anchor_candidate_text = "\n".join(candidate for _strategy, candidate, _meta in anchor_candidates)
assert_test(
    bool(anchor_candidates)
    and all(meta.get("scope") == "implied_context_only" for _strategy, _candidate, meta in anchor_candidates)
    and any(meta.get("changed_sentence_frames", 0) >= 2 for _strategy, _candidate, meta in anchor_candidates),
    "human anchor amplifier creates bounded implied-context candidates",
)
assert_test(
    not _synthetic_meta_anchor_artifact_reason(anchor_candidate_text)
    and "When this is applied in practice" not in anchor_candidate_text
    and "I would narrow the point this way" not in anchor_candidate_text,
    "human anchor amplifier avoids generic meta/process filler phrases",
)
strict_anchor_report = make_footprint_report(
    ai_authorship=50,
    human=67,
    ai_transformation=33,
    grounding=68,
    human_anchor=60,
    smoothness=48,
    semantic_uniformity=54,
    ai_likelihood=50,
    topk_pattern=97,
    topk_calibrated_risk=92,
    generic_assertion_risk=65,
    unsupported_claim_risk=90,
    broad_claim_risk=75,
    discourse=27,
    expansion=65,
    signal_agreement=47,
)
weak_anchor_burden_gate = _human_anchor_positive_burden_gate_status(
    {"positive_ai_burden": {"drop": 3.8}},
    strict_anchor_report,
)
strong_anchor_burden_gate = _human_anchor_positive_burden_gate_status(
    {"positive_ai_burden": {"drop": 4.2}},
    strict_anchor_report,
)
assert_test(
    not weak_anchor_burden_gate["accepted"]
    and weak_anchor_burden_gate["reason"] == "positive_ai_burden_drop_too_small"
    and strong_anchor_burden_gate["accepted"],
    "human anchor candidates need meaningful positive AI-burden movement while core AI drivers remain high",
)
portfolio_candidates = _formula_portfolio_candidates(
    target_push_text,
    target_push_report,
    topk_route_candidates=[
        (
            "topk_route_optimizer_sample",
            anchor_candidates[0][1],
            {"operation": "topk_route_rebuild"},
        )
    ],
    blocker_operation_candidates=_blocker_operation_candidates(target_push_text, target_push_report, limit=1),
    generic_assertion_candidates=_generic_assertion_compiler_candidates(target_push_text, target_push_report, limit=1),
    pruning_candidates=_content_pruning_candidates(target_push_text, target_push_report, limit=1),
    limit=4,
)
assert_test(
    bool(portfolio_candidates)
    and all(meta.get("formula_portfolio_candidate") for _strategy, _candidate, meta in portfolio_candidates)
    and any(
        "human_anchor_suppression" in (meta.get("targeted_drivers") or [])
        for _strategy, _candidate, meta in portfolio_candidates
    )
    and any(
        set(meta.get("targeted_drivers") or []) & {"ai_likelihood", "topk_calibrated_risk", "semantic_uniformity", "patchwork_expansion"}
        for _strategy, _candidate, meta in portfolio_candidates
    ),
    "formula portfolio generator composes AI-driver reduction with Human Anchor suppression candidates",
)
block_driver_map = _formula_block_driver_map(
    (
        "Education systems can be important for society and technology because they support many people.\n\n"
        "References\n\n"
        "https://example.com/source"
    ),
    turnitin_formula_original,
)
assert_test(
    block_driver_map["blocks"][0]["action"] in {"compress", "rebuild", "remove_candidate"}
    and "human_anchor_deficit" in block_driver_map["blocks"][0]
    and "suppression_gain_potential" in block_driver_map["blocks"][0]
    and "recommended_portfolio_action" in block_driver_map["blocks"][0]
    and all(
        row["action"] == "preserve"
        for row in block_driver_map["blocks"][1:]
    )
    and block_driver_map["top_blocks"],
    "formula block driver map targets generic prose while preserving reference blocks",
)
frontier_text = (
    "Students can use AI tools for homework. This creates a challenge for schools because answers can look complete.\n\n"
    "Teachers need to understand the process. The education system must respond to these changes."
)
frontier_report = make_footprint_report(
    ai_authorship=50,
    human=50,
    ai_transformation=50,
    grounding=45,
    human_anchor=22,
    smoothness=50,
    semantic_uniformity=55,
    ai_likelihood=60,
    topk_pattern=60,
    topk_calibrated_risk=40,
    generic_assertion_risk=55,
    qualifying_text_ai_density=55,
    unsupported_claim_risk=20,
    broad_claim_risk=30,
    discourse=35,
    expansion=30,
    section_style=30,
    signal_agreement=40,
)
frontier_map = _formula_block_driver_map(frontier_text, frontier_report)
frontier = _human_anchor_suppression_frontier(frontier_text, frontier_report, frontier_map)
frontier_candidates = _human_anchor_suppression_frontier_candidates(
    frontier_text,
    frontier_report,
    frontier_map,
    limit=4,
)
assert_test(
    frontier["suppression_headroom"] > 0
    and frontier["candidate_blocks"]
    and bool(frontier_candidates)
    and all(meta.get("human_anchor_suppression_frontier") for _strategy, _candidate, meta in frontier_candidates)
    and any(
        "human_anchor_suppression" in (meta.get("targeted_drivers") or [])
        for _strategy, _candidate, meta in frontier_candidates
    ),
    "human anchor suppression frontier exposes headroom and creates portfolio candidates",
)
geometry_text = (
    "Furthermore, students can use AI tools because answers can look complete. "
    "Teachers still guide the process."
)
geometry_report = json.loads(json.dumps(frontier_report))
geometry_report["predictability"] = {
    "all_sentences": [
        {
            "sentence_id": "s001",
            "sentence_index": 0,
            "sentence": "Furthermore, students can use AI tools because answers can look complete.",
            "top10_ratio": 0.82,
            "top50_ratio": 0.91,
            "predictability_risk": 0.76,
        },
        {
            "sentence_id": "s002",
            "sentence_index": 1,
            "sentence": "Teachers still guide the process.",
            "top10_ratio": 0.42,
            "top50_ratio": 0.62,
            "predictability_risk": 0.38,
        },
    ]
}
feasibility = _formula_feasibility_estimator(geometry_report)
geometry_map = _geometry_risk_map(geometry_text, geometry_report, limit=3)
geometry_candidates = _coordinated_micro_perturbation_candidates(
    geometry_text,
    geometry_report,
    geometry_map,
    limit=3,
)
assert_test(
    feasibility["mode"] in {"geometry_mode", "safe_portfolio_mode"}
    and "ai_likelihood" in feasibility["dominant_drivers"]
    and feasibility["estimated_safe_floor"] >= 0,
    "formula feasibility estimator exposes safe/aggressive floor and dominant weighted drivers",
)
assert_test(
    bool(geometry_map["sentence_hotspots"])
    and geometry_map["sentence_hotspots"][0]["sentence_index"] == 0
    and geometry_map["sentence_hotspots"][0]["drivers"]["connector_risk"] > 0,
    "geometry risk map ranks predictable connector/cadence hotspots",
)
assert_test(
    bool(geometry_candidates)
    and all(meta.get("coordinated_micro_perturbation") for _strategy, _candidate, meta in geometry_candidates)
    and any("Furthermore" not in candidate for _strategy, candidate, _meta in geometry_candidates),
    "coordinated micro perturbation creates deterministic geometry candidates",
)
convergence_current_text = (
    "Schools can feel predictable when quick answers hide work. "
    "Students need judgement in class. "
    "Teachers still guide the process."
)
convergence_bad_text = (
    "Schools can feel predictable when quick answers hide work. "
    "Students need judgement in class. "
    "Teachers still guide the process, overall."
)
convergence_better_text = (
    "Schools can feel predictable. "
    "Students need judgement in class when quick answers hide work. "
    "Teachers still guide the process."
)
convergence_current_report = make_footprint_report(
    ai_authorship=50,
    human=50,
    ai_transformation=50,
    grounding=45,
    human_anchor=30,
    smoothness=50,
    semantic_uniformity=50,
    ai_likelihood=60,
    topk_pattern=60,
    topk_calibrated_risk=30,
    generic_assertion_risk=45,
    qualifying_text_ai_density=45,
    unsupported_claim_risk=30,
    broad_claim_risk=30,
    discourse=30,
    expansion=40,
    section_style=40,
    signal_agreement=50,
)
convergence_better_report = make_footprint_report(
    ai_authorship=48,
    human=52,
    ai_transformation=48,
    grounding=45,
    human_anchor=35,
    smoothness=48,
    semantic_uniformity=45,
    ai_likelihood=55,
    topk_pattern=55,
    topk_calibrated_risk=25,
    generic_assertion_risk=42,
    qualifying_text_ai_density=42,
    unsupported_claim_risk=30,
    broad_claim_risk=30,
    discourse=28,
    expansion=35,
    section_style=35,
    signal_agreement=45,
)
convergence_bad_report = make_footprint_report(
    ai_authorship=48,
    human=48,
    ai_transformation=48,
    grounding=45,
    human_anchor=10,
    smoothness=70,
    semantic_uniformity=70,
    ai_likelihood=70,
    topk_pattern=40,
    topk_calibrated_risk=10,
    generic_assertion_risk=70,
    qualifying_text_ai_density=70,
    unsupported_claim_risk=30,
    broad_claim_risk=30,
    discourse=50,
    expansion=70,
    section_style=70,
    signal_agreement=70,
)
anti_smoothing_ok = _anti_smoothing_guard_status(convergence_current_report, convergence_better_report, strict=True)
anti_smoothing_bad = _anti_smoothing_guard_status(convergence_current_report, convergence_bad_report, strict=True)
assert_test(
    anti_smoothing_ok["accepted"]
    and not anti_smoothing_bad["accepted"]
    and anti_smoothing_bad["backfires"],
    "anti-smoothing guard accepts full formula texture gains and rejects component backfire",
)
scan_reports = {
    convergence_bad_text: convergence_bad_report,
    convergence_better_text: convergence_better_report,
}
planner_scores_seen = []

def fake_formula_convergence_scan(scan_text):
    return scan_reports[scan_text]

def fake_formula_convergence_candidates(text, report, pass_index, block_map):
    planner_scores_seen.append(_turnitin_like_ai_profile(report)["score"])
    if pass_index == 1:
        return [
            ("one_signal_backfire", convergence_bad_text, {}),
            ("portfolio_gain", convergence_better_text, {}),
        ]
    return []

def fake_formula_convergence_drift(_a, _b, **_kwargs):
    return SimpleNamespace(accepted=True, similarity=0.99, reasons=[])

convergence_result = _formula_convergence_controller(
    convergence_current_text,
    convergence_current_report,
    convergence_current_report,
    {"max_passes": 2, "max_scans": 4, "max_llm_calls": 0},
    scan_func=fake_formula_convergence_scan,
    candidate_builder=fake_formula_convergence_candidates,
    drift_checker=fake_formula_convergence_drift,
)
assert_test(
    convergence_result["selected"]
    and convergence_result["selected_text"] == convergence_better_text
    and convergence_result["score_after"] < convergence_result["score_before"]
    and convergence_result["formula_convergence_passes"][0]["selected"]
    and convergence_result["scans_used"] == 2
    and convergence_result["llm_calls_used"] == 0
    and convergence_result["phase_budget_used"]["scans"] == 2
    and planner_scores_seen[0] == _turnitin_like_ai_profile(convergence_current_report)["score"],
    "formula convergence controller replans from current best, records budget usage, and keeps the best safe formula drop",
)
assert_test(
    any(
        row.get("strategy") == "one_signal_backfire"
        and row.get("reason") == "no_safe_formula_drop"
        for row in convergence_result["candidates"]
    ),
    "formula convergence controller rejects one-signal movement when total formula score worsens",
)
unsupported_regression_text = (
    "Schools can feel predictable. "
    "Students need judgement in class when quick answers hide work. "
    "Teachers still guide the process."
)
unsupported_regression_report = make_footprint_report(
    ai_authorship=48,
    human=52,
    ai_transformation=48,
    grounding=45,
    human_anchor=42,
    smoothness=45,
    semantic_uniformity=45,
    ai_likelihood=52,
    topk_pattern=55,
    topk_calibrated_risk=24,
    generic_assertion_risk=42,
    qualifying_text_ai_density=42,
    unsupported_claim_risk=44,
    broad_claim_risk=30,
    discourse=28,
    expansion=30,
    section_style=30,
    signal_agreement=40,
)
unsupported_result = _formula_convergence_controller(
    convergence_current_text,
    convergence_current_report,
    convergence_current_report,
    {"max_passes": 1, "max_scans": 2, "max_llm_calls": 0},
    scan_func=lambda _text: unsupported_regression_report,
    candidate_builder=lambda _text, _report, _pass, _map: [
        (
            "anchor_unsupported_backfire",
            unsupported_regression_text,
            {"human_anchor_suppression_frontier": True},
        )
    ],
    drift_checker=fake_formula_convergence_drift,
)
assert_test(
    not unsupported_result["selected"]
    and unsupported_result["candidates"][0]["reason"] == "unsupported_claim_risk_regressed",
    "formula convergence rejects Human Anchor candidates that regress unsupported-claim risk",
)

class FakeFormulaGateway:
    def chat(self, *_args, **_kwargs):
        return SimpleNamespace(content=json.dumps({
            "candidates": [
                {
                    "reason": "replace one high-drag paragraph",
                    "patches": [
                        {
                            "operation_type": "LIKELIHOOD_TEXTURE_REBUILD",
                            "target_paragraph_index": 0,
                            "expected_driver": "ai_likelihood",
                            "replacement": "Schools can feel uneven. In class, students need judgement when a quick answer hides the work. Teachers still guide the process."
                        }
                    ],
                }
            ]
        }))

llm_patch_candidates = _formula_convergence_llm_patch_candidates(
    convergence_current_text,
    convergence_current_report,
    _formula_block_driver_map(convergence_current_text, convergence_current_report),
    FakeFormulaGateway(),
    max_candidates=2,
)
assert_test(
    bool(llm_patch_candidates)
    and llm_patch_candidates[0][0].startswith("formula_convergence_llm_block_recreate")
    and llm_patch_candidates[0][2].get("formula_convergence_llm_block_recreate")
    and "quick answer hides the work" in llm_patch_candidates[0][1],
    "formula convergence LLM block recreate returns JSON paragraph patch candidates",
)
stance_candidates = _author_stance_thread_candidates(
    target_push_text,
    target_push_report,
    limit=3,
)
assert_test(
    stance_candidates == [],
    "author stance threading does not synthesize first-person judgement without source stance material",
)
target_push_report_at_goal = json.loads(json.dumps(target_push_report))
target_push_report_at_goal["integrity_layers"]["layers"]["human_contribution_signal"]["score"] = 82.0
assert_test(
    _post_safe_win_target_push_candidates(target_push_text, target_push_report_at_goal, limit=4) == [],
    "post-safe-win target push skips when Human target is already reached",
)

safe_stale_blocker_override = _dominant_blocker_safe_progress_override(
    {
        "required": True,
        "cleared": False,
        "active_keys": ["unsupported_claim_risk"],
        "drops": {"unsupported_claim_risk": 0.0},
    },
    {
        "human_delta": 5.0,
        "ai_authorship_delta": 2.0,
        "ai_transformation_delta": 5.0,
        "ai_authorship_regression_blocked": False,
        "critical_high_regressed": False,
        "review_burden_regressed": False,
        "weighted_severity_regressed": False,
    },
    {"active_regression": 0.0},
    ai_score_regressed=False,
    finding_delta=-1,
    review_burden_delta=-1,
    weighted_severity_delta=-2,
    critical_high_delta=0,
)
assert_test(
    safe_stale_blocker_override.get("allowed") is False
    and safe_stale_blocker_override.get("dominant_drop_allowed") is False,
    "dominant blocker gate blocks stale unsupported-claim progress without dominant movement",
)
dominant_moving_blocker_override = _dominant_blocker_safe_progress_override(
    {
        "required": True,
        "cleared": False,
        "active_keys": ["unsupported_claim_risk"],
        "drops": {"unsupported_claim_risk": 2.0},
    },
    {
        "human_delta": 18.0,
        "ai_authorship_delta": 1.0,
        "ai_transformation_delta": 18.0,
        "ai_authorship_regression_blocked": False,
        "critical_high_regressed": False,
        "review_burden_regressed": False,
        "weighted_severity_regressed": False,
    },
    {"active_drop": 30.0, "active_regression": 15.0},
    ai_score_regressed=False,
    finding_delta=-1,
    review_burden_delta=-6,
    weighted_severity_delta=-10,
    critical_high_delta=-1,
)
assert_test(
    dominant_moving_blocker_override.get("allowed") is True,
    "dominant blocker gate allows safe progress when dominant blocker moves",
)
topk_moving_blocker_override = _dominant_blocker_safe_progress_override(
    {
        "required": True,
        "cleared": False,
        "active_keys": ["unsupported_claim_risk", "source_grounding_risk", "broad_claim_risk", "topk_calibrated_risk"],
        "drops": {
            "unsupported_claim_risk": 0.0,
            "source_grounding_risk": 0.0,
            "broad_claim_risk": 0.0,
            "topk_calibrated_risk": 0.75,
        },
    },
    {
        "human_delta": 5.0,
        "ai_authorship_delta": 2.0,
        "ai_transformation_delta": 5.0,
        "ai_authorship_regression_blocked": False,
        "critical_high_regressed": False,
        "review_burden_regressed": False,
        "weighted_severity_regressed": False,
    },
    {"active_drop": 0.75, "active_regression": 0.0},
    ai_score_regressed=False,
    finding_delta=-1,
    review_burden_delta=-1,
    weighted_severity_delta=-2,
    critical_high_delta=0,
)
assert_test(
    topk_moving_blocker_override.get("allowed") is True
    and topk_moving_blocker_override.get("dominant_drop_allowed") is True,
    "dominant blocker gate allows safe progress when top-k moves while source/unsupported remain pinned",
)
target_breakthrough_stale_blocker_override = _dominant_blocker_safe_progress_override(
    {
        "required": True,
        "cleared": False,
        "active_keys": ["unsupported_claim_risk"],
        "drops": {"unsupported_claim_risk": 0.0},
    },
    {
        "human_delta": 32.0,
        "ai_authorship_delta": 1.0,
        "ai_transformation_delta": 20.0,
        "crosses_target_human": True,
        "ai_authorship_regression_blocked": False,
        "critical_high_regressed": False,
        "review_burden_regressed": False,
        "weighted_severity_regressed": False,
    },
    {"active_drop": 30.0, "active_regression": 0.0},
    ai_score_regressed=False,
    finding_delta=-1,
    review_burden_delta=-6,
    weighted_severity_delta=-10,
    critical_high_delta=-1,
)
assert_test(
    target_breakthrough_stale_blocker_override.get("allowed") is True
    and target_breakthrough_stale_blocker_override.get("target_breakthrough") is True,
    "dominant blocker gate allows target breakthrough even when unsupported blocker is pinned",
)

unsafe_stale_blocker_override = _dominant_blocker_safe_progress_override(
    {
        "required": True,
        "cleared": False,
        "active_keys": ["unsupported_claim_risk"],
        "drops": {"unsupported_claim_risk": 0.0},
    },
    {
        "human_delta": 5.0,
        "ai_authorship_delta": -1.0,
        "ai_transformation_delta": 5.0,
        "ai_authorship_regression_blocked": True,
    },
    {"active_regression": 0.0},
    ai_score_regressed=False,
    finding_delta=-1,
    review_burden_delta=-1,
    weighted_severity_delta=-2,
    critical_high_delta=0,
)
assert_test(
    unsafe_stale_blocker_override.get("allowed") is False,
    "dominant blocker gate still blocks authorship regression",
)

old_auth_budget = os.environ.get("DRAFTPROOF_AUTHENTICITY_MAX_LLM_CALLS")
old_search_budget = os.environ.get("DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS")
try:
    os.environ.pop("DRAFTPROOF_AUTHENTICITY_MAX_LLM_CALLS", None)
    os.environ["DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS"] = "0"
    assert_test(
        _resolve_stage_llm_budget(
            "DRAFTPROOF_AUTHENTICITY_MAX_LLM_CALLS",
            "DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS",
            default=4,
        ) == 0,
        "authenticity generation honors shared AI-search LLM kill switch",
    )
    os.environ["DRAFTPROOF_AUTHENTICITY_MAX_LLM_CALLS"] = "2"
    assert_test(
        _resolve_stage_llm_budget(
            "DRAFTPROOF_AUTHENTICITY_MAX_LLM_CALLS",
            "DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS",
            default=4,
        ) == 2,
        "stage-specific LLM budget overrides shared fallback budget",
    )
finally:
    if old_auth_budget is None:
        os.environ.pop("DRAFTPROOF_AUTHENTICITY_MAX_LLM_CALLS", None)
    else:
        os.environ["DRAFTPROOF_AUTHENTICITY_MAX_LLM_CALLS"] = old_auth_budget
    if old_search_budget is None:
        os.environ.pop("DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS", None)
    else:
        os.environ["DRAFTPROOF_AI_SEARCH_MAX_LLM_CALLS"] = old_search_budget


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("17. POST-SELECTION AI-DENSITY BREAKER ADD-ON")
print("=" * 70)

assert_test(
    _ai_density_breaker_canonical_fact_sentence(
        "The United States was founded in 1776 after the American colonies declared independence from Britain."
    )
    is True,
    "density breaker preserves canonical factual sentences with protected years",
)

route_sentence, route_ops = _ai_density_breaker_sentence_route(
    "One of the biggest strengths of the United States is its economic power."
)
assert_test(
    route_sentence != "One of the biggest strengths of the United States is its economic power."
    and "economic power" in route_sentence
    and route_ops,
    "density breaker uses operation-specific route edits for generic expansion sentences",
)

density_source = (
    "The United States was founded in 1776 after the American colonies declared independence from Britain. "
    "One of the biggest strengths of the United States is its economic power. "
    "At the same time, diversity has also created challenges related to inequality and social tension."
)
density_report = {
    "ai_risk_badge": {
        "ai_components": {"topk_pattern_raw": 100, "topk_calibrated_risk": 92},
        "transformation_classification": {
            "features": {
                "human_anchor_score": 0.20,
                "ai_likelihood": 0.55,
                "semantic_uniformity_risk": 0.60,
                "rewrite_smoothness": 0.50,
                "outline_to_text_expansion": 0.30,
                "section_style_variance": 0.20,
                "signal_agreement_score": 0.60,
                "human_anchor_discount": 0.15,
            }
        },
        "writing_components": {
            "lived_detail_risk": 0.80,
            "domain_grounding_strength": 0.70,
        },
    },
    "predictability": {
        "all_sentences": [
            {"sentence": "The United States was founded in 1776 after the American colonies declared independence from Britain.", "top10_ratio": 0.95, "top50_ratio": 1.0},
            {"sentence": "One of the biggest strengths of the United States is its economic power.", "top10_ratio": 0.90, "top50_ratio": 1.0},
            {"sentence": "At the same time, diversity has also created challenges related to inequality and social tension.", "top10_ratio": 0.80, "top50_ratio": 1.0},
        ]
    },
}
density_candidates = _post_selection_ai_density_breaker_candidates(density_source, density_report, limit=4)
assert_test(
    bool(density_candidates)
    and all("1776" in candidate for _strategy, candidate, _meta in density_candidates),
    "density breaker candidates keep canonical factual anchors while editing generic routes",
)
assert_test(
    bool(density_candidates)
    and density_candidates[0][0].startswith("density_window_patch")
    and all(
        not strategy.startswith("density_route_patch_top4")
        and not strategy.startswith("density_route_patch_top6")
        for strategy, _candidate, _meta in density_candidates
    ),
    "density breaker uses bounded density windows instead of scattered top-4/top-6 sentence edits",
)
assert_test(
    bool(density_candidates)
    and all(
        (_meta.get("patchwork_budget") or {}).get("accepted") is True
        and int(_meta.get("edited_sentence_count") or 0)
        <= int((_meta.get("edit_budget") or {}).get("max_edited_sentences") or 0)
        for _strategy, _candidate, _meta in density_candidates
        if _meta.get("operation") == "coordinated_density_window_patch"
    ),
    "density breaker candidates carry and obey the patchwork edit budget",
)

base_acceptance_report = {
    "ai_risk_badge": {
        "ai_components": {"topk_pattern_raw": 98, "topk_calibrated_risk": 90, "qualifying_text_ai_density": 60},
        "transformation_classification": {
            "features": {
                "ai_likelihood": 0.50,
                "semantic_uniformity_risk": 0.58,
                "rewrite_smoothness": 0.48,
                "outline_to_text_expansion": 0.40,
                "section_style_variance": 0.20,
                "signal_agreement_score": 0.50,
                "human_anchor_discount": 0.18,
            }
        },
    },
    "integrity_layers": {
        "layers": {
            "ai_authorship_risk": {"score": 49},
            "ai_transformation_risk": {"score": 45},
            "human_contribution_signal": {"score": 55},
        }
    },
    "findings": {"critical": [], "high": [], "medium": [], "low": []},
}
candidate_acceptance_report = {
    "ai_risk_badge": {
        "ai_components": {"topk_pattern_raw": 96, "topk_calibrated_risk": 88, "qualifying_text_ai_density": 57},
        "transformation_classification": {
            "features": {
                "ai_likelihood": 0.48,
                "semantic_uniformity_risk": 0.57,
                "rewrite_smoothness": 0.48,
                "outline_to_text_expansion": 0.39,
                "section_style_variance": 0.20,
                "signal_agreement_score": 0.49,
                "human_anchor_discount": 0.18,
            }
        },
    },
    "integrity_layers": {
        "layers": {
            "ai_authorship_risk": {"score": 48},
            "ai_transformation_risk": {"score": 44},
            "human_contribution_signal": {"score": 56},
        }
    },
    "findings": {"critical": [], "high": [], "medium": [], "low": []},
}
accepted_density = _post_selection_ai_density_breaker_acceptance(
    base_acceptance_report,
    candidate_acceptance_report,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
)
assert_test(
    accepted_density.get("selectable") is True
    and accepted_density.get("driver_drops", {}).get("qualifying_text_ai_density") == 3.0,
    "density breaker accepts only rescanned formula/external/density improvement without safety regression",
)
assert_test(
    accepted_density.get("positive_ai_burden_drop", 0) > 0,
    "density breaker acceptance records positive AI-burden movement when present",
)
micro_density_report = {
    "ai_risk_badge": {
        "ai_components": {"topk_pattern_raw": 98, "topk_calibrated_risk": 90, "qualifying_text_ai_density": 60},
        "transformation_classification": {
            "features": {
                "ai_likelihood": 0.499,
                "semantic_uniformity_risk": 0.58,
                "rewrite_smoothness": 0.48,
                "outline_to_text_expansion": 0.40,
                "section_style_variance": 0.20,
                "signal_agreement_score": 0.50,
                "human_anchor_discount": 0.18,
            }
        },
    },
    "integrity_layers": {
        "layers": {
            "ai_authorship_risk": {"score": 49},
            "ai_transformation_risk": {"score": 45},
            "human_contribution_signal": {"score": 55},
        }
    },
    "findings": {"critical": [], "high": [], "medium": [], "low": []},
}
accepted_micro_density = _post_selection_ai_density_breaker_acceptance(
    base_acceptance_report,
    micro_density_report,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
)
assert_test(
    accepted_micro_density.get("selectable") is True
    and 0 < accepted_micro_density.get("formula_score_drop", 0) < 0.25,
    "density breaker keeps safety-clean micro formula drops instead of applying a minimum-drop cliff",
)

auto_repair_source = (
    "The United States was founded in 1776 after the American colonies declared independence from Britain. "
    "The Constitution later set out a federal structure with separate branches of government. "
    "One of the biggest strengths of the United States is its economic power. "
    "Universities, technology firms, sports leagues, and films all contribute to its global presence. "
    "Apple, Microsoft, Google, and Tesla are often named in discussions about American business influence. "
    "NASA is also associated with space research and national scientific ambition. "
    "At the same time, diversity has also created challenges related to inequality and social tension. "
    "These tensions do not erase the country's influence, but they complicate how that influence is understood."
)
auto_repair_base = make_footprint_report(
    ai_authorship=55,
    human=53,
    ai_transformation=47,
    grounding=45,
    human_anchor=25,
    smoothness=48,
    semantic_uniformity=56,
    ai_likelihood=55,
    topk_pattern=100,
    topk_calibrated_risk=100,
    generic_assertion_risk=70,
    qualifying_text_ai_density=65,
    unsupported_claim_risk=30,
    broad_claim_risk=45,
    discourse=40,
    expansion=35,
    section_style=20,
    signal_agreement=50,
)
auto_repair_improved = make_footprint_report(
    ai_authorship=55,
    human=54,
    ai_transformation=46,
    grounding=45,
    human_anchor=25,
    smoothness=47,
    semantic_uniformity=55,
    ai_likelihood=54,
    topk_pattern=100,
    topk_calibrated_risk=100,
    generic_assertion_risk=68,
    qualifying_text_ai_density=64,
    unsupported_claim_risk=30,
    broad_claim_risk=45,
    discourse=40,
    expansion=35,
    section_style=20,
    signal_agreement=49,
)

def _simple_report_finding_total(report):
    findings = (report or {}).get("findings") or {}
    return sum(len(findings.get(tier, [])) for tier in ("critical", "high", "medium", "low"))

def _simple_report_review_burden(report):
    findings = (report or {}).get("findings") or {}
    return sum(len(findings.get(tier, [])) for tier in ("critical", "high", "medium"))

def _simple_report_weighted_severity(report):
    findings = (report or {}).get("findings") or {}
    weights = {"critical": 8, "high": 5, "medium": 2, "low": 1}
    return sum(len(findings.get(tier, [])) * weight for tier, weight in weights.items())

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
    drift_checker=lambda _source, _candidate, **_kwargs: SimpleNamespace(accepted=True, similarity=0.99, reasons=[]),
    scan_func=lambda candidate: auto_repair_improved if "One strength of" in candidate or "At the same time" not in candidate else auto_repair_base,
    turnitin_profile=_turnitin_like_ai_profile,
    turnitin_gate_status=_turnitin_like_ai_gate_status,
    strict_safe_status=_strict_ai_safe_band_status,
    contribution_scores=_contribution_scores,
    integrity_scores=_integrity_scores,
    badge_ai=lambda report: ((report or {}).get("ai_risk_badge") or {}).get("ai_likelihood_score"),
    finding_total=_simple_report_finding_total,
    review_burden=_simple_report_review_burden,
    weighted_severity=_simple_report_weighted_severity,
    critical_high_count=lambda report: len(((report or {}).get("findings") or {}).get("critical", []))
    + len(((report or {}).get("findings") or {}).get("high", [])),
)
auto_repair_plan = auto_repair_compile_plan(auto_repair_source, auto_repair_base, auto_repair_deps, max_windows=2)
assert_test(
    any(row.get("canonical_fact_preserve") for row in auto_repair_plan.get("risk_map") or [])
    and any(row.get("operators") for row in auto_repair_plan.get("selected_windows") or []),
    "auto-repair compiler preserves canonical facts while compiling editable operator windows",
)
auto_repair_pool = auto_repair_candidate_pool(auto_repair_source, auto_repair_plan, auto_repair_deps, limit=6)
assert_test(
    bool(auto_repair_pool)
    and all((meta.get("operator_contract") or {}).get("must_not_add_claims") is True for _s, _c, meta in auto_repair_pool),
    "auto-repair compiler generates operator-bound patch candidates instead of open-ended rewrites",
)
auto_repair_result = run_auto_repair_controller(
    auto_repair_source,
    auto_repair_base,
    auto_repair_base,
    auto_repair_deps,
    max_rounds=2,
    max_scans=4,
    target_human=80,
)
assert_test(
    auto_repair_result.get("selected") is True
    and auto_repair_result.get("accepted_rounds") >= 1
    and (auto_repair_result.get("score_drop") or 0) > 0,
    "auto-repair controller rescans bounded candidates and keeps safe Pareto progress",
)

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
    drift_checker=lambda _source, _candidate, **_kwargs: SimpleNamespace(accepted=True, similarity=0.99, reasons=[]),
    scan_func=lambda candidate: auto_repair_improved if "One strength of" in candidate or "At the same time" not in candidate else auto_repair_base,
    turnitin_profile=_turnitin_like_ai_profile,
    turnitin_gate_status=_turnitin_like_ai_gate_status,
    strict_safe_status=_strict_ai_safe_band_status,
    contribution_scores=_contribution_scores,
    integrity_scores=_integrity_scores,
    badge_ai=lambda report: ((report or {}).get("ai_risk_badge") or {}).get("ai_likelihood_score"),
    finding_total=_simple_report_finding_total,
    review_burden=_simple_report_review_burden,
    weighted_severity=_simple_report_weighted_severity,
    critical_high_count=lambda report: len(((report or {}).get("findings") or {}).get("critical", []))
    + len(((report or {}).get("findings") or {}).get("high", [])),
)
compiler_plan = compiler_build_plan(auto_repair_source, auto_repair_base, compiler_deps, max_windows=4)
assert_test(
    any(row.get("classification") == "canonical_fact_preserve" and "1776" in row.get("sentence", "") for row in compiler_plan.get("sentence_risk_map") or [])
    and any(row.get("classification") == "generic_expansion_target" for row in compiler_plan.get("selected_windows") or []),
    "rewrite compiler preserves canonical facts and targets generic expansion/template sentences",
)
fact_loss_candidate = auto_repair_source.replace("1776", "the founding year", 1)
fact_loss_validation = compiler_validate_candidate(
    auto_repair_source,
    fact_loss_candidate,
    {"operator": "COMPRESS_ABSTRACT_CLAIM"},
    compiler_deps,
)
assert_test(
    fact_loss_validation.get("passed") is False
    and any("lost" in reason for reason in fact_loss_validation.get("reject_reasons") or []),
    "rewrite compiler validator blocks fact/anchor loss",
)
overcompressed_quality = compiler_evaluate_quality(
    auto_repair_source,
    "The United States has influence.",
    {"operator": "REMOVE_LOW_VALUE_GENERIC_BLOCK"},
    compiler_deps,
)
assert_test(
    overcompressed_quality.get("passed") is False
    and "over_compressed_document" in (overcompressed_quality.get("reject_reasons") or []),
    "rewrite compiler evaluator rejects over-compressed output",
)
malformed_quality = compiler_evaluate_quality(
    auto_repair_source,
    auto_repair_source.replace(
        "One of the biggest strengths of the United States is its economic power.",
        "Mixed has led to mixed opinions. its economic power.",
    ),
    {"operator": "REDUCE_SYMMETRIC_CADENCE"},
    compiler_deps,
)
assert_test(
    malformed_quality.get("passed") is False
    and any("artifact" in reason or "orphan" in reason for reason in malformed_quality.get("reject_reasons") or []),
    "rewrite compiler evaluator rejects malformed sentence artifacts before scan selection",
)
budget_check = RewriteRunBudget(max_seconds=30, max_scans=8, max_llm_calls=2, started_at=0)
budget_check.record_stage("ai_mitigation_search", seconds=29.0, scans=8, llm_calls=2)
assert_test(
    budget_check.can_run(min_seconds=5, min_scans=1, min_llm_calls=1) is False
    and budget_check.skip_reason("rewrite_compiler", min_seconds=5, min_scans=1, min_llm_calls=1).get("reason") in {
        "global_time_budget_exhausted",
        "global_scan_budget_exhausted",
        "global_llm_budget_exhausted",
    },
    "global rewrite budget blocks late phases after the shared cap is exhausted",
)

def _ledger_record(stage, source_text, current_text, candidate_text, current_report, candidate_report):
    metrics = {
        "turnitin_profile": _turnitin_like_ai_profile(candidate_report),
        "current_turnitin_profile": _turnitin_like_ai_profile(current_report),
        "original_turnitin_profile": _turnitin_like_ai_profile(current_report),
        "strict_safe": _strict_ai_safe_band_status(candidate_report),
        "footprint": _ai_footprint_profile(candidate_report),
        "current_footprint": _ai_footprint_profile(current_report),
        "ai_authorship": _integrity_scores(candidate_report).get("ai_authorship"),
        "current_ai_authorship": _integrity_scores(current_report).get("ai_authorship"),
        "ai_transformation": _contribution_scores(candidate_report).get("ai_transformation"),
        "current_ai_transformation": _contribution_scores(current_report).get("ai_transformation"),
        "review_burden": _simple_report_review_burden(candidate_report),
        "current_review_burden": _simple_report_review_burden(current_report),
        "weighted_severity": _simple_report_weighted_severity(candidate_report),
        "current_weighted_severity": _simple_report_weighted_severity(current_report),
        "critical_high": 0,
        "current_critical_high": 0,
        "finding_total": _simple_report_finding_total(candidate_report),
        "current_finding_total": _simple_report_finding_total(current_report),
        "changed_sentence_ratio": 0.05,
    }
    return build_candidate_record(
        stage=stage,
        strategy=stage,
        text=candidate_text,
        report=candidate_report,
        original_text=source_text,
        original_report=current_report,
        current_text=current_text,
        current_report=current_report,
        metrics=metrics,
    )

pinned_tiny_report = make_footprint_report(
    ai_authorship=55,
    human=54,
    ai_transformation=46,
    grounding=45,
    human_anchor=25,
    smoothness=51,
    semantic_uniformity=55,
    ai_likelihood=54.8,
    topk_pattern=100,
    topk_calibrated_risk=100,
    generic_assertion_risk=68,
    qualifying_text_ai_density=64,
    unsupported_claim_risk=30,
    broad_claim_risk=45,
    discourse=40,
    expansion=35,
    section_style=20,
    signal_agreement=49,
)
ledger = CandidateLedger(min_formula_drop=0.001, min_late_formula_drop_when_pinned=1.0)
ledger.seed(_ledger_record("seed", auto_repair_source, auto_repair_source, auto_repair_source, auto_repair_base, auto_repair_base))
tiny_decision = ledger.consider(_ledger_record(
    "rewrite_compiler",
    auto_repair_source,
    auto_repair_source,
    auto_repair_source.replace("One of the biggest strengths of", "One strength of"),
    auto_repair_base,
    pinned_tiny_report,
))
assert_test(
    tiny_decision.get("accepted") is False
    and "pinned_topk" in str(tiny_decision.get("reason")),
    "global selector rejects tiny late gains when Top-k stays pinned and smoothness regresses",
)
compiler_candidate = auto_repair_source.replace(
    "One of the biggest strengths of the United States is its economic power.",
    "One strength of the United States is its economic power.",
)
compiler_validation = compiler_validate_candidate(
    auto_repair_source,
    compiler_candidate,
    {"operator": "COMPRESS_ABSTRACT_CLAIM"},
    compiler_deps,
)
compiler_quality = compiler_evaluate_quality(
    auto_repair_source,
    compiler_candidate,
    {"operator": "COMPRESS_ABSTRACT_CLAIM"},
    compiler_deps,
)
compiler_candidate_eval = compiler_evaluate_scanned_candidate(
    auto_repair_source,
    auto_repair_base,
    compiler_candidate,
    auto_repair_improved,
    auto_repair_base,
    compiler_deps,
    validation=compiler_validation,
    quality=compiler_quality,
)
assert_test(
    compiler_candidate_eval.get("accepted") is True
    and compiler_candidate_eval.get("outcome_class") == "unsafe_partial_improvement"
    and (compiler_candidate_eval.get("strict_ai_safe_band") or {}).get("achieved") is False,
    "rewrite compiler preserves useful formula drops but labels unsafe detector drivers honestly",
)
compiler_result = run_rewrite_compiler(
    auto_repair_source,
    auto_repair_base,
    auto_repair_base,
    compiler_deps,
    config=CompilerConfig(mode="compiler_strict", max_rounds=1, max_scans=4, candidate_pool_limit=8, shortlist_limit=4),
)
assert_test(
    compiler_result.get("selected") is True
    and compiler_result.get("llm_calls_used") == 0
    and (compiler_result.get("score_drop") or 0) > 0,
    "rewrite compiler strict mode selects deterministic scanned progress without LLM calls",
)

density_smoothness_slippage_report = {
    "ai_risk_badge": {
        "ai_components": {"topk_pattern_raw": 97.69, "topk_calibrated_risk": 89.154, "qualifying_text_ai_density": 61.06},
        "transformation_classification": {
            "features": {
                "ai_likelihood": 0.485,
                "semantic_uniformity_risk": 0.5821,
                "rewrite_smoothness": 0.5074,
                "outline_to_text_expansion": 0.39,
                "section_style_variance": 0.20,
                "signal_agreement_score": 0.50,
                "human_anchor_discount": 0.18,
            }
        },
    },
    "integrity_layers": {
        "layers": {
            "ai_authorship_risk": {"score": 47},
            "ai_transformation_risk": {"score": 43},
            "human_contribution_signal": {"score": 57},
        }
    },
    "findings": {"critical": [], "high": [], "medium": [], "low": []},
}
accepted_density_slippage = _post_selection_ai_density_breaker_acceptance(
    base_acceptance_report,
    density_smoothness_slippage_report,
    review_burden_delta=-2,
    weighted_severity_delta=-3,
    critical_high_delta=0,
)
assert_test(
    accepted_density_slippage.get("selectable") is True
    and accepted_density_slippage.get("component_slippage_accepted") is True
    and "rewrite_smoothness" in (accepted_density_slippage.get("component_slippage") or {}),
    "density breaker preserves bounded smoothness slippage when total formula and core AI drivers improve",
)

anchor_probe_candidates = _post_density_human_anchor_probe_candidates(
    density_source,
    density_report,
    limit=3,
)
assert_test(
    bool(anchor_probe_candidates)
    and all("1776" in candidate for _strategy, candidate, _meta in anchor_probe_candidates)
    and all(" I " not in f" {candidate} " for _strategy, candidate, _meta in anchor_probe_candidates),
    "post-density Human Anchor probe keeps canonical facts and avoids personal-voice injection",
)
assert_test(
    bool(anchor_probe_candidates)
    and all(
        (_meta.get("patchwork_budget") or {}).get("accepted") is True
        and int(_meta.get("changed_sentence_frames") or 0) <= 3
        for _strategy, _candidate, _meta in anchor_probe_candidates
    ),
    "post-density Human Anchor probe uses a tiny bounded edit budget",
)

anchor_probe_base = make_footprint_report(
    ai_authorship=51,
    human=49,
    ai_transformation=44,
    grounding=40,
    human_anchor=20,
    smoothness=48,
    semantic_uniformity=58,
    ai_likelihood=50,
    topk_pattern=96,
    topk_calibrated_risk=90,
    generic_assertion_risk=70,
    qualifying_text_ai_density=60,
    unsupported_claim_risk=30,
    broad_claim_risk=45,
    discourse=40,
    expansion=35,
    section_style=20,
    signal_agreement=50,
)
anchor_probe_good = make_footprint_report(
    ai_authorship=51,
    human=50,
    ai_transformation=44,
    grounding=40,
    human_anchor=32,
    smoothness=48,
    semantic_uniformity=58,
    ai_likelihood=50,
    topk_pattern=96,
    topk_calibrated_risk=90,
    generic_assertion_risk=70,
    qualifying_text_ai_density=60,
    unsupported_claim_risk=30,
    broad_claim_risk=45,
    discourse=40,
    expansion=35,
    section_style=20,
    signal_agreement=50,
)
anchor_probe_bad = make_footprint_report(
    ai_authorship=52,
    human=50,
    ai_transformation=44,
    grounding=40,
    human_anchor=35,
    smoothness=48,
    semantic_uniformity=58,
    ai_likelihood=50,
    topk_pattern=98,
    topk_calibrated_risk=94,
    generic_assertion_risk=70,
    qualifying_text_ai_density=60,
    unsupported_claim_risk=30,
    broad_claim_risk=45,
    discourse=40,
    expansion=35,
    section_style=20,
    signal_agreement=50,
)
accepted_anchor_probe = _post_density_human_anchor_probe_acceptance(
    anchor_probe_base,
    anchor_probe_good,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
)
rejected_anchor_probe = _post_density_human_anchor_probe_acceptance(
    anchor_probe_base,
    anchor_probe_bad,
    review_burden_delta=0,
    weighted_severity_delta=0,
    critical_high_delta=0,
)
assert_test(
    accepted_anchor_probe.get("selectable") is True
    and accepted_anchor_probe.get("human_anchor_suppression_gain", 0) > 0
    and accepted_anchor_probe.get("formula_score_drop", 0) > 0,
    "post-density Human Anchor probe accepts measured suppression gain only when total formula drops",
)
assert_test(
    rejected_anchor_probe.get("selectable") is False
    and rejected_anchor_probe.get("driver_drops", {}).get("topk_calibrated_risk", 0) < 0,
    "post-density Human Anchor probe rejects anchor gain when protected AI drivers backfire",
)

repo_root_for_worker_checks = os.path.join(os.path.dirname(__file__), "..")
worker_tasks_source = open(
    os.path.join(repo_root_for_worker_checks, "worker", "app", "tasks.py")
).read()
worker_entrypoint_source = open(
    os.path.join(repo_root_for_worker_checks, "worker", "entrypoint.sh")
).read()
assert_test(
    '"post_density_human_anchor_probe"' in worker_tasks_source
    and '"selected_human_anchor_probe_strategy"' in worker_tasks_source,
    "worker debug export includes post-density Human Anchor probe details",
)
assert_test(
    '"debug_export_version": REWRITE_DEBUG_EXPORT_VERSION' in worker_tasks_source
    and '"runtime_code_fingerprint": runtime_fingerprint' in worker_tasks_source
    and '"worker_tasks_sha256_12"' in worker_tasks_source
    and '"rewrite_compiler"' in worker_tasks_source
    and '"deterministic_rewrite_compiler"' in worker_tasks_source
    and '"selected_rewrite_compiler_strategy"' in worker_tasks_source
    and '"llm_calls_used": stage.get("llm_calls")' in worker_tasks_source,
    "worker debug export includes compiler details plus actual runtime source fingerprint",
)
assert_test(
    "rm -rf /app/worker/app" in worker_entrypoint_source
    and "cp -a /tmp/draftproof-repo/worker/app /app/worker/app" in worker_entrypoint_source,
    "worker entrypoint replaces the baked worker app instead of nesting latest app under app/app",
)

llm_summary_probe = {}
_record_rewrite_llm_calls(llm_summary_probe, "ai_search_llm_calls_used", 2)
_record_rewrite_llm_calls(llm_summary_probe, "formula_convergence_llm_calls_used", "4")
assert_test(
    llm_summary_probe.get("llm_calls_used") == 6
    and llm_summary_probe.get("ai_search_llm_calls_used") == 2
    and llm_summary_probe.get("formula_convergence_llm_calls_used") == 4,
    "rewrite summary aggregates phase LLM call counts instead of reporting zero after budgeted calls",
)

formula_gap_budget = formula_gap_budget_contract()
assert_test(
    formula_gap_budget.get("deterministic_probe_scans") == 2
    and formula_gap_budget.get("llm_candidate_calls") == 5
    and formula_gap_budget.get("finalist_scans") == 5
    and formula_gap_budget.get("total_scan_cap") == 10,
    "formula-gap orchestrator reserves LLM budget and caps deterministic probe scans",
)
formula_families = formula_gap_portfolio_families(5)
assert_test(
    formula_families == [
        "STATISTICAL_TEXTURE_REBUILD",
        "SEMANTIC_VARIANCE_RESTRUCTURE",
        "HUMAN_ANCHOR_SUPPRESSION_GAIN",
        "HYBRID_TEXTURE_ANCHOR",
        "LOW_VALUE_COMPRESS_REMOVE",
    ],
    "formula-gap orchestrator exposes the five portfolio LLM candidate families",
)
valid_formula_payload = json.dumps({
    "strategy": "STATISTICAL_TEXTURE_REBUILD",
    "targeted_drivers": ["ai_likelihood", "rewrite_smoothness"],
    "changed_blocks": [1],
    "fact_inventory_preserved": True,
    "core_claims_preserved_or_merged": True,
    "protected_anchors_preserved": True,
    "unsupported_new_facts": False,
    "candidate_text": "The United States began with independence in 1776. Its later growth mixed political power, industry, immigration, and cultural influence.",
})
valid_patch_payload = json.dumps({
    "strategy": "STATISTICAL_TEXTURE_REBUILD",
    "targeted_drivers": ["ai_likelihood", "rewrite_smoothness"],
    "changed_blocks": [1],
    "fact_inventory_preserved": True,
    "core_claims_preserved_or_merged": True,
    "protected_anchors_preserved": True,
    "unsupported_new_facts": False,
    "patches": [
        {
            "block_index": 1,
            "operation": "replace",
            "replacement_text": "Culture still carries weight, but not in a neat list. Films, sport, music, and technology pull in different directions, and that uneven mix is part of the point.",
        }
    ],
})
accepted_payload, accepted_reason = extract_formula_gap_candidate_payload(valid_formula_payload)
accepted_patch_payload, accepted_patch_reason = extract_formula_gap_candidate_payload(valid_patch_payload)
rejected_payload, rejected_reason = extract_formula_gap_candidate_payload(
    valid_formula_payload.replace('"unsupported_new_facts": false', '"unsupported_new_facts": true')
)
assert_test(
    accepted_payload is not None
    and accepted_patch_payload is not None
    and not accepted_reason
    and not accepted_patch_reason
    and rejected_payload is None
    and rejected_reason == "unsupported_new_facts_declared",
    "formula-gap LLM JSON contract accepts full candidates or block patches and rejects declared unsupported facts",
)
fake_formula_report = {
    "ai_risk_badge": {
        "ai_components": {
            "ai_likelihood": 50,
            "topk_calibrated_risk": 40,
            "semantic_uniformity_risk": 30,
            "rewrite_smoothness": 30,
            "outline_to_text_expansion": 20,
            "section_style_variance": 10,
            "signal_agreement_score": 50,
        },
        "human_anchor_discount": 12,
    }
}
block_source_probe = (
    "The United States was founded in 1776 after the colonies declared independence from Britain.\n\n"
    "One of the most important features of the country is its economy and culture. It has become influential in many areas and plays a major role around the world.\n\n"
    "In conclusion, the country remains powerful and important because of its history, economy, and culture."
)
block_tasks_probe = formula_gap_block_portfolio_tasks(block_source_probe, fake_formula_report, limit=5)
assembled_patch_text, applied_patch_rows, assembly_reason = assemble_formula_gap_candidate(
    block_source_probe,
    accepted_patch_payload or {},
)
formula_plan_probe = formula_gap_plan(fake_formula_report)
prompt_probe = formula_gap_candidate_prompt(
    block_source_probe,
    fake_formula_report,
    "LOW_VALUE_COMPRESS_REMOVE",
    protected_anchors=[{"text": "1776", "reason": "numeric"}],
    block_task=block_tasks_probe[-1] if block_tasks_probe else None,
)
entity_probe = formula_gap_named_entity_inventory(
    "The United States joined NATO work with New York City, Hollywood, and The Civil Rights Movement as examples."
)
assert_test(
    formula_plan_probe.get("target_score") == 20.0
    and "remaining_weighted_drivers" in formula_plan_probe
    and "LOW_VALUE_COMPRESS_REMOVE" in prompt_probe
    and "1776" in prompt_probe,
    "formula-gap plan and prompt expose weighted drivers, strict target, and protected anchors",
)
assert_test(
    len(block_tasks_probe) == 5
    and all(task.get("block_indexes") for task in block_tasks_probe)
    and "patches" in prompt_probe
    and "Selected source blocks" in prompt_probe
    and "assembler will copy unchanged blocks exactly" in prompt_probe,
    "formula-gap orchestrator divides five LLM calls into block-scoped patch tasks",
)
assert_test(
    applied_patch_rows
    and not assembly_reason
    and assembled_patch_text.startswith("The United States was founded in 1776")
    and "Culture still carries weight" in assembled_patch_text
    and assembled_patch_text.endswith("history, economy, and culture."),
    "formula-gap assembler applies JSON patches while preserving untouched source blocks",
)
assert_test(
    "The United States" in entity_probe
    and "New York City" in entity_probe
    and "The Civil Rights Movement" in entity_probe,
    "formula-gap prompt inventory preserves named entities that semantic drift would otherwise reject",
)
span_density_text = (
    "The United States was founded in 1776 after the colonies declared independence from Britain. "
    "One of the most important features of the country is its strong global influence. "
    "Furthermore, it plays a major role in politics, economics, culture, and education. "
    "This shows that the country has become a significant force in the modern world. "
    "In conclusion, the country remains important because of its history and culture."
)
span_density_report = {
    "ai_risk_badge": {
        "ai_likelihood_score": 62,
        "ai_components": {
            "topk_calibrated_risk": 88,
            "qualifying_text_ai_density": 72,
        },
        "writing_components": {
            "lived_detail_risk": 80,
            "domain_grounding_strength": 30,
        },
    },
    "predictability": {
        "all_sentences": [
            {"sentence_id": f"s{idx + 1:03d}", "sentence": sentence, "top10_ratio": 0.74, "top50_ratio": 0.9, "predictability_risk": 0.55}
            for idx, sentence in enumerate([s for s in span_density_text.split(". ") if s])
        ]
    },
}
span_density_candidate_text = (
    "The United States was founded in 1776 after the colonies declared independence from Britain. "
    "Its global influence is broad, but the effects do not land evenly. "
    "Politics, economics, culture, and education each show that influence in different ways."
)
span_density_candidate_report = {
    **span_density_report,
    "predictability": {
        "all_sentences": [
            {"sentence_id": "s001", "sentence": "The United States was founded in 1776 after the colonies declared independence from Britain.", "top10_ratio": 0.74, "top50_ratio": 0.9, "predictability_risk": 0.55},
            {"sentence_id": "s002", "sentence": "Its global influence is broad, but the effects do not land evenly.", "top10_ratio": 0.42, "top50_ratio": 0.65, "predictability_risk": 0.25},
            {"sentence_id": "s003", "sentence": "Politics, economics, culture, and education each show that influence in different ways.", "top10_ratio": 0.45, "top50_ratio": 0.7, "predictability_risk": 0.22},
        ]
    },
}
span_density_contract = build_eligible_span_density_contract(span_density_text, span_density_report)
span_density_comparison = compare_eligible_span_density(
    span_density_text,
    span_density_report,
    span_density_candidate_text,
    span_density_candidate_report,
)
span_unsafe_class = classify_ai_search_candidate({
    "selectable": True,
    "turnitin_like_mitigation": True,
    "turnitin_like_ai_gate": {"safe_band": True},
    "formula_gap_contract": {"target_met": True},
    "eligible_span_density_gate": {"safe": False, "improved": False},
})
span_safe_class = classify_ai_search_candidate({
    "selectable": True,
    "turnitin_like_mitigation": True,
    "turnitin_like_ai_gate": {"safe_band": True},
    "formula_gap_contract": {"target_met": True},
    "eligible_span_density_gate": {"safe": True, "improved": True},
})
assert_test(
    span_density_contract["safe"] is False
    and span_density_contract["top_sentence_targets"][0]["sentence_index"] != 0
    and span_density_contract["needs_author_context"] is True,
    "eligible span density map preserves canonical facts and flags unsafe generic prose needing author context",
)
assert_test(
    span_density_comparison["improved"] is True
    and span_density_comparison["safe"] is True
    and span_density_comparison["unsafe_eligible_word_ratio_drop"] > 0,
    "eligible span density comparison measures material unsafe-span reduction",
)
assert_test(
    span_unsafe_class["class"] != "detector_safe"
    and span_safe_class["class"] == "detector_safe",
    "selector cannot label detector-safe while eligible span density remains unsafe",
)
segment_window_text = (
    "The United States was founded in 1776 after the colonies declared independence from Britain. "
    "The Constitution later created a system of government with separate branches. "
    "One of the most important features of the country is its strong global influence. "
    "Furthermore, it plays a major role in politics, economics, culture, and education. "
    "This shows that the country has become a significant force in the modern world. "
    "Another important feature is its economic strength and global companies. "
    "Apple, Microsoft, Google, Tesla, and NASA are often used as examples. "
    "Overall, the United States remains influential because of its history and culture."
)
segment_window_report = {
    "ai_risk_badge": {
        "ai_likelihood_score": 64,
        "ai_components": {
            "topk_calibrated_risk": 92,
            "qualifying_text_ai_density": 75,
        },
        "writing_components": {
            "lived_detail_risk": 80,
            "domain_grounding_strength": 30,
        },
        "transformation_classification": {
            "features": {
                "semantic_uniformity_risk": 0.58,
                "rewrite_smoothness": 0.52,
                "outline_to_text_expansion": 0.40,
                "section_style_variance": 0.45,
                "signal_agreement_score": 0.50,
            }
        },
    },
    "predictability": {
        "all_sentences": [
            {
                "sentence_id": f"s{idx + 1:03d}",
                "sentence_index": idx,
                "sentence": sentence,
                "top10_ratio": 0.78,
                "top50_ratio": 0.91,
                "predictability_risk": 0.60,
            }
            for idx, sentence in enumerate([s.strip() for s in re.split(r"(?<=[.!?])\s+", segment_window_text) if s.strip()])
        ]
    },
}
segment_windows = build_segment_density_windows(segment_window_text, segment_window_report, limit=3)
segment_tasks = segment_window_tasks(segment_window_text, segment_window_report, limit=3)
segment_prompt = segment_window_candidate_prompt(segment_window_text, segment_window_report, segment_tasks[0])
segment_payload, segment_payload_reason = extract_segment_window_payload(json.dumps({
    "strategy": "WINDOW_TEXTURE_REBUILD",
    "targeted_drivers": ["ai_likelihood", "topk_calibrated_risk"],
    "fact_inventory_preserved": True,
    "protected_anchors_preserved": True,
    "unsupported_new_facts": False,
    "sentence_patches": [
        {
            "sentence_index": 2,
            "replacement_text": "Its global influence is broad, but the effects do not land in one neat pattern."
        },
        {
            "sentence_index": 0,
            "replacement_text": "The founding date changed."
        },
    ],
}))
segment_candidate_text, segment_applied, segment_assembly_reason = assemble_segment_window_candidate(
    segment_window_text,
    segment_payload or {},
    segment_tasks[0],
)
assert_test(
    segment_windows
    and 5 <= segment_windows[0]["sentence_count"] <= 10
    and segment_windows[0]["editable_sentence_count"] >= 1,
    "segment-window controller ranks overlapping 5-10 sentence density windows",
)
assert_test(
    segment_window_is_canonical_fact_sentence("The United States was founded in 1776 after independence from Britain.")
    and "canonical_fact_preserve" in segment_prompt
    and "No personal voice" in segment_prompt,
    "segment-window prompt preserves canonical facts and forbids personal voice",
)
assert_test(
    segment_payload is not None
    and not segment_payload_reason
    and "1776" in segment_candidate_text
    and "The founding date changed" not in segment_candidate_text
    and segment_applied
    and not segment_assembly_reason,
    "segment-window assembler applies only editable sentence patches and skips canonical facts",
)
assert_test(
    segment_patchwork_budget(segment_window_text, segment_candidate_text, segment_applied)["accepted"] is True,
    "segment-window patchwork budget accepts scoped window edits",
)
decision_probe = build_candidate_decision(
    {
        "selectable": True,
        "reason": "accepted_partial_turnitin_like_mitigation",
        "turnitin_like_ai_gate": {"improved": True, "safety_clean": True, "score_drop": 4.5},
        "formula_gap_contract": {"score_drop": 4.5, "target_met": False},
        "reference_ai": 60,
    },
    {"formula_gap_contract": {"score_drop": 4.5, "target_met": False}},
    candidate_ai=55,
)
assert_test(
    decision_probe.selectable
    and decision_probe.formula_score_drop == 4.5
    and decision_probe.headline_ai_drop == 5.0
    and decision_probe.to_dict().get("rank"),
    "candidate selector exposes CandidateDecision with visible rank and headline drops",
)
resolved_global_seconds = resolve_global_rewrite_seconds(
    legacy_seconds=30,
    controller_policy_seconds=210,
    env_seconds=None,
)
medium_reserve_seconds = post_ai_search_reserve_seconds(888)
capped_ai_search_seconds = cap_phase_seconds_for_reserve(
    max_seconds=210,
    remaining_seconds=205,
    reserve_seconds=medium_reserve_seconds,
)
tight_ai_search_seconds = cap_phase_seconds_for_reserve(
    max_seconds=210,
    remaining_seconds=40,
    reserve_seconds=medium_reserve_seconds,
)
assert_test(
    resolved_global_seconds == 210
    and medium_reserve_seconds == 55
    and capped_ai_search_seconds == 150
    and 0 < tight_ai_search_seconds < 40,
    "global rewrite budget uses controller policy and reserves time for post-search phases",
)


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("18. FULL PLANNER→GUARD→SCORE PIPELINE")
print("=" * 70)

# Simulate the full flow
dr = make_detect_result()
plan = planner.plan([dr])
fixability_map = {ftype: route["fixability"] for ftype, route in FINDING_ROUTING.items()}

# Only rewrite auto-fixable findings
auto_findings = [a.finding for a in plan.auto_fixable]
manual_findings = [a.finding for a in plan.manual_required]
protected_findings = [a.finding for a in plan.protected]

print(f"  Auto-fixable: {len(auto_findings)} findings")
print(f"  Manual: {len(manual_findings)} findings")
print(f"  Protected: {len(protected_findings)} findings")

# Simulate improved post-rewrite findings (some auto findings resolved)
post_rewrite_findings = auto_findings[2:]  # first 2 resolved by rewrite

protected_spans = detect_protected_spans(SAMPLE_TEXT)
rewrite_candidate = SAMPLE_TEXT.replace("Furthermore", "—").replace("In conclusion, ", "")

# Transactional apply with voice guard
voice = analyze_voice(SAMPLE_TEXT)
tx = transactional_apply(
    SAMPLE_TEXT, rewrite_candidate, protected_spans,
    config, voice_guard=VoiceGuard(),
)

# Score
drift = check_semantic_drift(SAMPLE_TEXT, rewrite_candidate, threshold=0.85)
score = score_candidate(
    original_findings=auto_findings,
    candidate_findings=post_rewrite_findings,
    original_text=SAMPLE_TEXT,
    candidate_text=rewrite_candidate,
    drift_check=drift,
    fixability_map=fixability_map,
)

print(f"  Transactional: accepted={tx.accepted}, reason='{tx.reason}'")
print(f"  Drift: accepted={drift.accepted}, similarity={drift.similarity:.3f}")
print(f"  Score: total={score.total}, finding_reduction={score.finding_reduction}, voice={score.voice_preservation}")

assert_test(tx.accepted or not tx.accepted, "pipeline completes without error")  # just checking no crash
assert_test(score.total >= 0, f"score is non-negative")


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RESULTS")
print("=" * 70)
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Total:  {passed + failed}")

if failed == 0:
    print("\nAll tests passed!")
else:
    print(f"\n{failed} tests FAILED")
    sys.exit(1)
