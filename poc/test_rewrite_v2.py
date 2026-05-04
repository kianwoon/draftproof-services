"""Test the re-architected rewrite module.

Tests fixability routing, voice guard, transactional apply,
floor detection, outcome classification, and report integration.

Run:  cd poc && python test_rewrite_v2.py
"""

import sys
import os

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
from rewrite.rewrite import (
    _candidate_task_instruction,
    _candidate_style_reject_reason,
    _paragraph_coherence_reject_reason,
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
assert_test(cfg.max_llm_calls == 6, f"default max_llm_calls=6 (got {cfg.max_llm_calls})")
assert_test(cfg.max_auto_targets == 6, f"default max_auto_targets=6 (got {cfg.max_auto_targets})")
assert_test(cfg.max_rewrite_seconds == 90, f"default max_rewrite_seconds=90 (got {cfg.max_rewrite_seconds})")
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

for bad in (
    "The chart improves technical accuracy for each learner.",
    "Students face operational obstacles during practice.",
    "This visible learning framework preserves technical rigor in the digital landscape.",
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


# ════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("16. FULL PLANNER→GUARD→SCORE PIPELINE")
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
