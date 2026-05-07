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
    _allow_ai_search_llm_after_deterministic,
    _load_local_env,
    _repair_candidate_source_damage,
    _source_repair_drift_false_positive,
    _ai_search_protected_loss_reason,
    _ai_search_drift_false_positive,
    _ai_search_entity_drift_scan_allowed,
    _reconstruction_drift_scan_allowed,
    _scan_scope_summary,
    _human_shift_score,
    _authenticity_gate_status,
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
    _sentence_texture_risk_map,
    _micro_texture_window,
    _splice_sentence_window,
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
    _paragraph_role,
    _human_signal_amplification_prompt,
    _score_human_amplification_candidate,
    _build_author_evidence_completion_layer,
    _build_mitigation_ceiling_diagnostics,
    _build_author_evidence_intake_layer,
    _blocked_human_candidate_repair_prompt,
    _blocking_finding_targets,
    _finding_local_repair_prompt,
    _extract_finding_local_patches,
    _apply_finding_local_patches,
    _extract_paragraph_component_candidates,
    _clean_paragraph_component_candidate,
    _splice_paragraph,
)
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
meaningful_search_status = _ai_search_candidate_selection_status(57.78, 52.50, True)
assert_test(
    meaningful_search_status["selectable"],
    "AI search selects candidates only after the required AI drop is met",
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
                    "semantic_uniformity_risk": semantic_uniformity / 100,
                }
            }
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
    human=68,
    ai_transformation=32,
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
        human=68,
        ai_transformation=32,
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
    and negative_shift_gate["reason"] == "human_shift_score_too_low",
    "authenticity gate rejects AI drops that are outweighed by human-side regressions",
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
frozen_payload = _freeze_anchor_payload(
    {"anchors": ["SHBHCUT002"], "body": "SHBHCUT003 appears in context."},
    anchor_mapping,
)
assert_test(
    json.dumps(frozen_payload).count("SHBHCUT") == 0,
    "anchor lock freezes nested section context payloads",
)
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
    and "Do not replace author-owned classroom reasoning" in reconstruction_prompt,
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
assert_test(
    _ai_candidate_quality_reject_reason("Introduction Inclusive learning design starts here."),
    "AI search rejects merged heading text",
)
assert_test(
    _ai_candidate_quality_reject_reason(
        "I encourage open discussion so learners can With only six learners in my current HBB26 intake, smaller classes help. "
        "The rest of the document continues normally with enough words to form a candidate."
    ) == "dangling_sentence_fragment_join",
    "AI search rejects dangling sentence-fragment joins",
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
    "Maintaining standards while improving access Inclusive learning design does not lower the standard. "
    "ith only six learners, I can watch the technique more closely. "
    "This does not simplify the work, but it clarifies the learning steps. "
    "This does not simplify the work, but it clarifies the learning steps."
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
    "Introduction Inclusive learning design starts here. "
    "ith only six learners, I can observe closely. "
    "This does not simplify the work, but it clarifies the learning steps. "
    "This does not simplify the work, but it clarifies the learning steps. "
    "Conclusion This review ends here."
)
assert_test(
    not re.search(r"\bith only\b", repaired_candidate, re.I)
    and "Introduction Inclusive" not in repaired_candidate
    and "Conclusion This" not in repaired_candidate,
    "AI search repairs inherited source damage before candidate gates",
)
assert_test(
    "fixed_broken_with_fragment" in repair_notes
    and "normalized_with_only_phrase" in repair_notes
    and any(note.startswith("split_merged_heading") for note in repair_notes),
    "AI search records source damage repairs on candidates",
)
assert_test(
    "removed_duplicate_sentences:1" in repair_notes
    and repaired_candidate.count("This does not simplify the work") == 1,
    "AI search removes repeated exact sentences before candidate gates",
)
overlap_damaged, overlap_notes = _repair_candidate_source_damage(
    "With only six learners in my current HBB26 intake, smaller class sizes let me observe technique. "
    "I encourage open discussion so learners can With only six learners in my current HBB26 intake, smaller class sizes let me observe technique. "
    "I encourage open discussion so learners can describe how they perceive the haircut shape. "
    "describe how they perceive the haircut shape. "
    "Learners gain confidence when their ideas are acknowledged. "
    "A competent learner can explain the steps, identify the guide, check balance, adjust Learners gain confidence when their ideas are acknowledged. "
    "A competent learner can explain the steps, identify the guide, check balance, adjust projection, and apply the technique to real clients. "
    "projection, and apply the technique to real clients."
)
assert_test(
    "learners can With only" not in overlap_damaged
    and "adjust Learners gain" not in overlap_damaged
    and not re.search(r"(?<!can )describe how they perceive", overlap_damaged)
    and overlap_damaged.count("projection, and apply") == 1,
    "AI search repairs overlapping fragment damage before quality gates",
)
assert_test(
    any(note.startswith("removed_dangling_prefix") for note in overlap_notes)
    and any(note.startswith("removed_duplicate_fragments") for note in overlap_notes),
    "AI search records overlapping fragment repairs",
)
latest_selected_damage, latest_selected_notes = _repair_candidate_source_damage(
    "Inclusive Learning Design in Certificate III Hairdressing Introduction\n\n"
    "Inclusive learning design starts here. Competency should not depend on chance. "
    "When learners start to get lost\n\n"
    "The challenge begins here. They must practise, receive corrections, and repeat the skill. "
    "Showing the haircut clearly\n\n"
    "A demonstration reveals the educator's actions. "
    "The sources address different aspects of the classroom challenge. "
    "CESE and Chandler and Sweller focus on cognitive overload. "
    "Billett and Kirschner et al. Billett and Kirschner et al. "
    "CAST and Jwad et al. describe multiple learning pathways. "
    "DEWR defines the boundary for reasonable adjustment and maintaining assessment integrity. "
    "multiple learning pathways. Competency should not depend on chance in practice."
)
assert_test(
    "Hairdressing Introduction" not in latest_selected_damage
    and "chance.\n\nWhen learners start to get lost" in latest_selected_damage
    and "skill.\n\nShowing the haircut clearly" in latest_selected_damage,
    "AI search repairs selected-candidate heading placement before final output",
)
assert_test(
    "Billett and Kirschner et al. Billett and Kirschner et al." not in latest_selected_damage
    and "guided practice over discovery learning" in latest_selected_damage
    and "integrity. multiple learning pathways." not in latest_selected_damage,
    "AI search repairs selected-candidate conclusion source fragments",
)
assert_test(
    "split_title_from_introduction" in latest_selected_notes
    and any(
        note.startswith("split_sentence_before_heading")
        or note.startswith("split_orphaned_heading")
        for note in latest_selected_notes
    )
    and "repaired_conclusion_fragment:guided_practice" in latest_selected_notes,
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
    not _ai_search_entity_drift_scan_allowed(
        "Certificate III Hairdressing content.",
        ["lost_named_entity: 'Box Hill Institute'"],
        0.95,
    ),
    "AI search still blocks scoring candidates that lose critical entities",
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
paragraph_prompt = _paragraph_component_prompt(
    paragraph_targets[0],
    paragraph_search_json,
    1,
    reference_ai=57.78,
    required_ai_drop=5.0,
    target_ai_score=52.78,
    candidate_count=3,
)
assert_test(
    "TARGET PARAGRAPH" in paragraph_prompt
    and "generic_assertion_risk=90.00%" in paragraph_prompt
    and "target AI<=52.78" in paragraph_prompt
    and "Return exactly 3 alternative replacement paragraphs" in paragraph_prompt,
    "paragraph component prompt passes score drivers and scoped rewrite instruction",
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
    "In my practice I noticed that my learners handled sectioning, projection, guide control, "
    "parting, comb tension, wrist position, elbow height, scissor angle, client consultation, "
    "mannequin practice, and subsection checks more confidently after I slowed the demonstration.",
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
)
assert_test(
    "HUMAN_SIGNAL_AMPLIFICATION_REPAIR" in amplification_prompt
    and "Controlled operation: add a source-to-practice bridge" in amplification_prompt
    and "Human Contribution must increase by at least 2" in amplification_prompt
    and "AI Authorship must not increase" in amplification_prompt
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
                "target_paragraph_preview": "Students are surrounded by too much information.",
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
os.environ.pop("DRAFTPROOF_RETRY_MODEL_ENABLED", None)
os.environ.pop("DRAFTPROOF_RETRY_MODEL_MAX_CALLS", None)
os.environ["planner_model"] = "openai/gpt-4.1-mini"
os.environ["generator_model"] = "openai/gpt-5-mini"
os.environ["retry_model"] = "openai/gpt-5.2"
os.environ["retry_model_enabled"] = "1"
os.environ["retry_model_max_calls"] = "3"
roles_lowercase = _llm_role_config("fallback-model")
assert_test(
    roles_lowercase["planner_model"] == "openai/gpt-4.1-mini"
    and roles_lowercase["generator_model"] == "openai/gpt-5-mini"
    and roles_lowercase["retry_model"] == "openai/gpt-5.2"
    and roles_lowercase["retry_model_enabled"] is True
    and roles_lowercase["retry_model_max_calls"] == 3,
    "LLM role config accepts lowercase Koyeb model env names",
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
educational_rewrite = guided_summary.get("educational_mitigation_rewrite") or {}
assert_test(
    educational_rewrite.get("draft_text") and "[[ADD VERIFIED DETAIL:" in educational_rewrite.get("draft_text"),
    "guided mitigation produces educational marked rewrite content",
)
assert_test(
    educational_rewrite.get("auto_apply") is False,
    "educational mitigation rewrite is not auto-applied",
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


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("17. FULL PLANNER→GUARD→SCORE PIPELINE")
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
