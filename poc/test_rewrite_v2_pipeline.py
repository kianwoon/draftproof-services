"""Focused tests for the scan-driven rewrite pipeline V2."""

from __future__ import annotations

import json
import os
import tempfile

from rewrite.guards import check_semantic_drift, detect_protected_spans, protected_spans_preserved
from rewrite_v2 import run_rewrite_pipeline_v2
from rewrite_v2.pipeline import (
    _author_stance_thesis_filter_failures,
    _author_strategy_semantic_override_allowed,
    _academic_all_section_filter_failures,
    _academic_assignment_sections,
    _all_section_compact_allowed,
    _normalize_academic_all_section_candidate,
    _parse_academic_all_section_variants,
    _academic_section_filter_failures,
    _academic_section_targets,
    _compose_academic_sections,
    _build_author_stance_texture_pass_prompt,
    _build_author_stance_thesis_reframe_prompt,
    _cluster_text_from_gate,
    _compose_full_doc_delta_winners,
    _expected_full_reconstruction_paragraph_count,
    _paragraph_inventory_for_full_reconstruction,
    _paragraph_tactics,
    _paragraph_target_map,
    _patch_filter_failures,
    _has_rewrite_meta_text,
    _select_best_v2_frontier,
    _survey_style_failures,
    _required_entities_for_full_reconstruction,
    _restore_required_anchor_forms,
    _replace_once_flexible,
    _strip_rewrite_meta_text,
)
from rewrite_v2.goal_contract import RewriteGoalStatus, evaluate_rewrite_goal, needs_author_context
from rewrite_v2.selection import CandidateLane, decide_candidate, select_best_applicable_candidate
from rewrite_v2.strategy import StrategyKind, classify_content_route, route_strategies


def assert_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


scan_json = {
    "input_text": "The United States is often described as one of the most influential countries in modern history.",
    "ai_risk_badge": {
        "ai_likelihood_score": 54.62,
        "writing_quality_score": 61.0,
        "ai_components": {
            "topk_calibrated_risk": 80,
            "qualifying_text_ai_density": 72,
            "external_ai_flag_risk": 55,
        },
    },
    "integrity_layers": {
        "layers": {
            "ai_authorship": {"score": 60},
            "ai_transformation": {"score": 55},
        },
    },
    "findings": {
        "critical": [],
        "high": [{"id": "f001", "text": "AI likelihood remains high."}],
        "medium": [],
        "low": [],
    },
    "sentence_map": {
        "s001": {
            "text": "The United States is often described as one of the most influential countries in modern history.",
        },
    },
}
replay_candidates = [
    {
        "strategy": "weak_candidate",
        "text": "The United States has had influence in modern history.",
        "report": {
            "ai_risk_badge": {
                "ai_likelihood_score": 53.0,
                "writing_quality_score": 60.0,
                "ai_components": {"topk_calibrated_risk": 70},
            },
            "integrity_layers": {"layers": {"ai_authorship": {"score": 55}, "ai_transformation": {"score": 50}}},
            "findings": {"critical": [], "high": [{"id": "f001"}], "medium": [], "low": []},
        },
    },
    {
        "strategy": "strong_frontier_candidate",
        "text": "The United States has influenced politics, technology, and culture.",
        "report": {
            "ai_risk_badge": {
                "ai_likelihood_score": 50.2,
                "writing_quality_score": 59.0,
                "ai_components": {"topk_calibrated_risk": 58},
            },
            "integrity_layers": {"layers": {"ai_authorship": {"score": 48}, "ai_transformation": {"score": 44}}},
            "findings": {"critical": [], "high": [{"id": "f001"}], "medium": [], "low": []},
        },
    },
]

strategies = route_strategies(scan_json, full_rewrite_allowed=True)
assert_test(
    any(strategy.kind == StrategyKind.FULL_REWRITE for strategy in strategies),
    "V2 router allows full rewrite when no rewrite briefs exist",
)
broad_route = classify_content_route(
    "\n\n".join([
        "The United States has a large role in world affairs and public debate.",
        "Its founding history still shapes how people talk about rights and power.",
        "The economy gives the country unusual weight across markets and companies.",
        "Culture also matters because films, music, and sports travel far outside its borders.",
        "Diversity is part of the country's strength, but inequality remains visible.",
        "Politics can make the country feel divided even when institutions remain strong.",
        "Healthcare and education show the same mixture of ambition and uneven access.",
        "Foreign policy gives the country influence, but it also creates criticism.",
        "Technology keeps the country close to global change and new risks.",
        "That mix explains why the country remains difficult to judge simply.",
    ]),
    scan_json,
)
assert_test(
    broad_route.content_mode == "broad_explanatory_essay"
    and "author_stance_thesis_reframe" in broad_route.allowed_strategy_families,
    "V2 content router allows author-thesis for broad explanatory essays",
)
academic_route = classify_content_route(
    "Studies suggest feedback matters for learning (Smith, 2021). Other work reached a similar result [2]. References show the field is still divided.",
    scan_json,
)
assert_test(
    academic_route.content_mode == "academic_cited_text"
    and "author_stance_thesis_reframe" in academic_route.blocked_strategy_families,
    "V2 content router blocks author-thesis for citation-heavy academic text",
)
assert_test(
    "academic_cited_section_density_resolver" in academic_route.allowed_strategy_families,
    "V2 content router enables section-level resolver for academic cited text",
)
assert_test(
    "academic_all_section_compact_reconstruction" in academic_route.allowed_strategy_families,
    "V2 content router enables compact all-section academic reconstruction for assignment-shaped cited text",
)
academic_strategies = route_strategies(scan_json, full_rewrite_allowed=True, content_route=academic_route)
assert_test(
    not any(strategy.kind == StrategyKind.FULL_REWRITE for strategy in academic_strategies),
    "V2 strategy router blocks full-document strategies for academic cited text",
)
academic_assignment = """Question 1
Hai Di Lao is a people-processing service (Blog, 2025).

Question 2
Reviews shown in Figure 1 to 4 describe slow staff responses. During the service encounter stage, customers judge whether staff notice their needs. A recent study indicates that attentiveness affects service experience (Gao et al., 2025). Azemi et al. (2019) states that negative word-of-mouth often follows poor service encounters.

Question 3
The outlet could use a visual call-light system (Rizzo, 2021). It could also use a 10-minute mandatory first check-in and a monthly mystery shopper programme (Kinch, 2025)."""
academic_assignment_scan = {
    **scan_json,
    "generation_handoff": {
        "document_profile": {"document_type": "reflective_or_analytical_submission"},
        "section_generation_units": [],
    },
}
assert_test(
    _all_section_compact_allowed(academic_assignment, academic_assignment_scan),
    "V2 compact academic layer is allowed for assignment-shaped cited sections",
)
non_assignment_academic = "Research shows the effect remains contested (Smith, 2021). Later work reached another result (Jones, 2022)."
assert_test(
    not _all_section_compact_allowed(non_assignment_academic, scan_json),
    "V2 compact academic layer does not run on unsectioned citation-heavy prose",
)
all_section_variants = _parse_academic_all_section_variants(
    """===VARIANT 1===
Question 1
Hai Di Lao is a people-processing service (Blog, 2025).

Question 2
Reviews shown in Figure 1 to 4 still point to slow staff responses (Gao et al., 2025).
===END==="""
)
assert_test(
    len(all_section_variants) == 1 and all_section_variants[0]["candidate_id"] == "academic_all_section_variant_1",
    "V2 compact academic layer parses delimited variants",
)
normalized_all_section = _normalize_academic_all_section_candidate(
    '**Question 1**\nThis aligns with "renting rather than owning the service as a whole." (Blog, 2025)\n\n**Question 2**\nReviews shown in Figure 1 to 4 remain relevant. Gao et al. (2025) notes the service issue.',
    [
        {
            "heading": "Question 1",
            "text": 'Question 1\nThis aligns with “renting rather than owning the service as a whole.” (Blog, 2025)',
        },
        {
            "heading": "Question 2",
            "text": "Question 2\nReviews shown in Figure 1 to 4 remain relevant (Gao et al., 2025).",
            "citations": ["(Gao et al., 2025)"],
        },
    ],
)
assert_test(
    "Question 1" in normalized_all_section
    and "“renting rather than owning the service as a whole.”" in normalized_all_section,
    "V2 compact academic layer normalizes markdown headings and exact quote forms",
)
assert_test(
    "(Gao et al., 2025)" in normalized_all_section
    and "Gao et al. (2025)" not in normalized_all_section,
    "V2 compact academic layer restores exact protected citation forms",
)
normalized_visual_reference = _normalize_academic_all_section_candidate(
    "Question 2\nReviews still point to slow staff responses (Gao et al., 2025).",
    [{
        "heading": "Question 2",
        "text": "Question 2\nReviews shown in Figure 1 to 4 remain relevant (Gao et al., 2025).",
        "citations": ["(Gao et al., 2025)"],
    }],
)
assert_test(
    "Figure 1 to 4" in normalized_visual_reference,
    "V2 compact academic layer restores protected visual evidence references",
)
academic_targets = _academic_section_targets(academic_assignment, scan_json, limit=3)
assert_test(
    len(academic_targets) >= 2
    and any("(Gao et al., 2025)" in target.get("citations", []) for target in academic_targets),
    "V2 academic resolver selects citation-bearing sections as rewrite targets",
)
section_patches = [
    {
        "section_id": academic_targets[0]["section_id"],
        "rewritten_section": (
            f"{academic_targets[0]['heading']}\n"
            "Reviews shown in Figure 1 to 4 still point to slow staff responses. "
            "During the service encounter stage, customers judge whether staff notice their needs. "
            "A recent study indicates that attentiveness affects service experience (Gao et al., 2025). "
            "Azemi et al. (2019) states that negative word-of-mouth often follows poor service encounters."
        ),
    },
    {
        "section_id": academic_targets[1]["section_id"],
        "rewritten_section": (
            f"{academic_targets[1]['heading']}\n"
            "The outlet could use a visual call-light system (Rizzo, 2021). "
            "It could also keep the 10-minute mandatory first check-in and a monthly mystery shopper programme (Kinch, 2025)."
        ),
    },
]
assert_test(
    not _academic_section_filter_failures(academic_targets[:2], section_patches),
    "V2 academic section filter accepts heading and citation preserving patches",
)
composed_academic, composed_academic_patches = _compose_academic_sections(
    academic_assignment,
    academic_targets[:2],
    section_patches,
)
assert_test(
    "(Gao et al., 2025)" in composed_academic
    and "Azemi et al. (2019)" in composed_academic
    and len([row for row in composed_academic_patches if row.get("applied")]) == 2,
    "V2 academic section composition preserves required citations in the full document",
)
bad_section_patches = [{**section_patches[0], "rewritten_section": section_patches[0]["rewritten_section"].replace("(Gao et al., 2025)", "Gao et al. (2025)")}]
assert_test(
    any("citation_lost:(Gao et al., 2025)" in reason for reason in _academic_section_filter_failures(academic_targets[:1], bad_section_patches)),
    "V2 academic section filter rejects citation form drift",
)
all_sections = _academic_assignment_sections(academic_assignment, academic_assignment_scan)
all_section_candidate = """Question 1
Hai Di Lao is a people-processing service (Blog, 2025).

Question 2
Reviews shown in Figure 1 to 4 still point to slow staff responses. A recent study indicates that attentiveness affects service experience (Gao et al., 2025). Azemi et al. (2019) states that negative word-of-mouth often follows poor service encounters.

Question 3
The outlet could use a visual call-light system (Rizzo, 2021). It could also keep the 10-minute mandatory first check-in and a monthly mystery shopper programme (Kinch, 2025).
"""
assert_test(
    not _academic_all_section_filter_failures(all_sections, all_section_candidate),
    "V2 compact academic layer accepts all-section candidates preserving headings and citations",
)
assert_test(
    any(
        "citation_lost:(Kinch, 2025)" in reason
        for reason in _academic_all_section_filter_failures(all_sections, all_section_candidate.replace("(Kinch, 2025)", "Kinch 2025"))
    ),
    "V2 compact academic layer rejects citation loss",
)
quote_route = classify_content_route(
    'The interviewee said “I did not know where to begin.” A second student said “the instructions felt unclear.” A teacher added “support arrived too late.”',
    scan_json,
)
assert_test(
    quote_route.content_mode == "quote_heavy"
    and "entity_locked_full_reconstruction" in quote_route.blocked_strategy_families,
    "V2 content router blocks full reconstruction for quote-heavy text",
)
structured_route = classify_content_route(
    "1. Create the report\n2. Review every finding\n3. Export the JSON\n4. Send the result",
    scan_json,
)
assert_test(
    structured_route.content_mode == "structured_list_table"
    and structured_route.allowed_strategy_families == ["targeted_paragraph_reconstruction"],
    "V2 content router keeps structured list/table content on targeted patches only",
)
technical_route = classify_content_route(
    "The API endpoint returns JSON when the database repository raises an exception in the deployment.",
    scan_json,
)
assert_test(
    technical_route.content_mode == "technical_content"
    and "author_stance_thesis_reframe" in technical_route.blocked_strategy_families,
    "V2 content router blocks thesis rewrite for technical content",
)
regulated_route = classify_content_route(
    "The policy states that users must not disclose patient records, and compliance review shall document every exception.",
    scan_json,
)
assert_test(
    regulated_route.content_mode == "regulated_policy_content"
    and "entity_locked_full_reconstruction" in regulated_route.blocked_strategy_families,
    "V2 content router limits regulated policy content to minimal strategies",
)
short_route = classify_content_route("This paragraph is too short to safely rebuild without more context.", scan_json)
assert_test(
    short_route.content_mode == "short_text"
    and short_route.allowed_strategy_families == ["targeted_paragraph_reconstruction"],
    "V2 content router limits very short text candidate budget",
)
reflection_route = classify_content_route(
    "I think the lesson changed how I read evidence because my first answer was too quick and my later notes were more careful.",
    scan_json,
)
assert_test(
    reflection_route.content_mode == "personal_reflection"
    and "author_stance_thesis_reframe" in reflection_route.allowed_strategy_families,
    "V2 content router allows author voice only when personal stance already exists",
)
broad_selector_rows = [
    {
        "strategy": "scan_entity_locked_full_reconstruction",
        "strategy_kind": "entity_locked_full_reconstruction",
        "candidate_ai": 31.22,
        "decision": {
            "lane": CandidateLane.SAFE_NEAR_MISS.value,
            "required_drop_met": True,
            "quality_safe": True,
            "semantic_safe": True,
            "rank": [2, 1, 0, 0, 38.48, 39.0, 1, 1, -1],
        },
    },
    {
        "strategy": "scan_author_stance_thesis_reframe",
        "strategy_kind": "author_stance_thesis_reframe",
        "candidate_ai": 32.54,
        "decision": {
            "lane": CandidateLane.SAFE_NEAR_MISS.value,
            "required_drop_met": True,
            "quality_safe": True,
            "semantic_safe": True,
            "rank": [2, 1, 0, 0, 37.16, 37.16, 1, 1, -4],
        },
    },
    {
        "strategy": "unsafe_cluster_rescue",
        "strategy_kind": "unsafe_cluster_rescue",
        "candidate_ai": 30.7,
        "decision": {
            "lane": CandidateLane.SAFE_NEAR_MISS.value,
            "required_drop_met": True,
            "quality_safe": True,
            "semantic_safe": True,
            "rank": [2, 1, 0, 0, 39.0, 39.0, 1, 1, -7],
        },
    },
]
broad_selector_best = _select_best_v2_frontier(broad_selector_rows, content_route=broad_route)
assert_test(
    broad_selector_best and broad_selector_best.get("strategy") == "scan_author_stance_thesis_reframe",
    "V2 broad essay selector prefers author-thesis over lower-internal-score rescue near-miss",
)
localized_scan = {
    **scan_json,
    "ai_risk_badge": {
        **(scan_json.get("ai_risk_badge") or {}),
        "ai_components": {
            **((scan_json.get("ai_risk_badge") or {}).get("ai_components") or {}),
            "topk_calibrated_risk": 100,
        },
    },
    "rewrite_edit_briefs": [{
        "finding_id": "f001",
        "target_sentence": "The United States is often described as one of the most influential countries in modern history.",
        "signals": {"predictable_token_spans": ["described as one of the most"]},
    }],
}
localized_strategies = route_strategies(localized_scan, full_rewrite_allowed=True)
assert_test(
    localized_strategies and localized_strategies[0].kind == StrategyKind.TARGETED,
    "V2 router starts with targeted resolution when rewrite briefs exist",
)
assert_test(
    not any(strategy.kind == StrategyKind.FULL_REWRITE for strategy in localized_strategies),
    "V2 router does not feed full content before targeted resolution",
)
previous_allow_full = os.environ.get("DRAFTPROOF_REWRITE_V2_ALLOW_FULL_AFTER_TARGETED")
os.environ["DRAFTPROOF_REWRITE_V2_ALLOW_FULL_AFTER_TARGETED"] = "1"
try:
    full_enabled_strategies = route_strategies(localized_scan, full_rewrite_allowed=True)
finally:
    if previous_allow_full is None:
        os.environ.pop("DRAFTPROOF_REWRITE_V2_ALLOW_FULL_AFTER_TARGETED", None)
    else:
        os.environ["DRAFTPROOF_REWRITE_V2_ALLOW_FULL_AFTER_TARGETED"] = previous_allow_full
assert_test(
    len(full_enabled_strategies) >= 2 and any(strategy.kind == StrategyKind.FULL_REWRITE for strategy in full_enabled_strategies),
    "V2 can enable full rewrite only after targeted resolution by flag",
)
assert_test(
    all(strategy.targeted_drivers for strategy in localized_strategies),
    "V2 strategies declare targeted drivers",
)

with tempfile.TemporaryDirectory() as tmpdir:
    result = run_rewrite_pipeline_v2(
        detect_json=scan_json,
        output_dir=tmpdir,
        replay_candidate_records=replay_candidates,
    )

summary = result["result"].summary
candidate_trace = summary.get("candidate_trace") or []
assert_test(
    result["status"] == "safe_partial_mitigation_applied",
    "V2 replay applies safe partial mitigation without claiming strict success",
)
assert_test(
    summary["rewrite_goal_status"]["status"] == RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value,
    "V2 summary exposes strict failed goal status",
)
assert_test(
    all((row.get("decision") or {}).get("lane") != CandidateLane.GOAL_MET.value for row in candidate_trace),
    "V2 replay does not classify unsafe partial candidates as goal met",
)
assert_test(
    len(summary.get("stage_timings") or []) == 1
    and summary["stage_timings"][0]["stage"] == "rewrite_v2_scan_driven",
    "V2 replay does not run the old post-selection controller cascade",
)
assert_test(
    (summary.get("selected_candidate") or {}).get("strategy") == "strong_frontier_candidate",
    "V2 typed selector chooses the strongest replay frontier candidate",
)

context_scan = {
    **scan_json,
    "input_text": "A draft that needs author evidence before safe AI mitigation can be completed.",
    "ai_mitigation": {"note": "Mitigation requires author context and evidence anchors before rewriting."},
}
with tempfile.TemporaryDirectory() as tmpdir:
    previous_fail_fast = os.environ.get("DRAFTPROOF_REWRITE_V2_FAIL_FAST_AUTHOR_CONTEXT")
    os.environ["DRAFTPROOF_REWRITE_V2_FAIL_FAST_AUTHOR_CONTEXT"] = "1"
    try:
        context_result = run_rewrite_pipeline_v2(detect_json=context_scan, output_dir=tmpdir)
    finally:
        if previous_fail_fast is None:
            os.environ.pop("DRAFTPROOF_REWRITE_V2_FAIL_FAST_AUTHOR_CONTEXT", None)
        else:
            os.environ["DRAFTPROOF_REWRITE_V2_FAIL_FAST_AUTHOR_CONTEXT"] = previous_fail_fast
context_summary = context_result["result"].summary
assert_test(
    context_result["status"] == RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT.value,
    "V2 stops with needs_author_context when scan says missing context blocks mitigation",
)
assert_test(
    len(context_summary.get("candidate_trace") or []) == 0,
    "V2 does not spend candidate budget after author-context fail-fast",
)
counter_only_scan = {
    **scan_json,
    "ai_mitigation": {
        "counts": {"needs_author_context": 3, "needs_author_evidence": 3},
        "readiness": {"requires_user_input": True},
    },
}
assert_test(
    not needs_author_context(counter_only_scan),
    "V2 does not treat author-context counters as hard rewrite blockers",
)

original_report = {
    "ai_risk_badge": {"ai_likelihood_score": 54.62, "ai_components": {"topk_calibrated_risk": 80}},
    "integrity_layers": {"layers": {"ai_authorship": {"score": 60}, "ai_transformation": {"score": 55}}},
    "findings": {"critical": [], "high": [], "medium": [], "low": []},
}
safe_report = {
    "ai_risk_badge": {
        "ai_likelihood_score": 18.0,
        "ai_components": {
            "topk_calibrated_risk": 10,
            "qualifying_text_ai_density": 10,
            "external_ai_flag_risk": 10,
        },
    },
    "integrity_layers": {"layers": {"ai_authorship": {"score": 10}, "ai_transformation": {"score": 10}}},
    "findings": {"critical": [], "high": [], "medium": [], "low": []},
}
goal = evaluate_rewrite_goal(
    original_text="A specific classroom note with 2026 evidence.",
    candidate_text="A specific classroom note with 2026 evidence.",
    original_report=original_report,
    candidate_report=safe_report,
)
decision = decide_candidate(
    goal=goal,
    original_report=original_report,
    candidate_report=safe_report,
    reference_ai=54.62,
    required_ai_drop=5.0,
    target_ai_score=49.62,
)
assert_test(goal.status == RewriteGoalStatus.AI_MITIGATED, "V2 strict goal contract recognizes safe candidates")
assert_test(decision.lane == CandidateLane.GOAL_MET, "V2 candidate decision selects only strict goal-met candidates as success")

review_decision = decide_candidate(
    goal=goal,
    original_report=original_report,
    candidate_report=safe_report,
    reference_ai=54.62,
    required_ai_drop=5.0,
    target_ai_score=49.62,
    semantic_safe=False,
)
assert_test(
    review_decision.lane != CandidateLane.GOAL_MET,
    "V2 does not classify detector-safe but semantic-review candidates as final success",
)

risky_external_report = {
    "ai_risk_badge": {
        "ai_likelihood_score": 30.7,
        "ai_components": {
            "topk_calibrated_risk": 26.114,
            "generic_assertion_risk": 90.0,
            "qualifying_text_ai_density": 0.0,
        },
        "writing_components": {
            "unsupported_claim_risk": 55.0,
            "broad_claim_risk": 35.0,
            "source_grounding_risk": 100.0,
        },
        "transformation_classification": {
            "features": {
                "rewrite_smoothness": 0.301,
                "semantic_uniformity_risk": 0.485,
                "discourse_regularity_risk": 0.3775,
            },
        },
    },
    "integrity_layers": {
        "layers": {
            "ai_authorship_risk": {"score": 31.0},
            "ai_transformation_risk": {"score": 18.0},
        },
    },
    "findings": {"critical": [], "high": [], "medium": [], "low": []},
}
risky_external_goal = evaluate_rewrite_goal(
    original_text=scan_json["input_text"],
    candidate_text=(
        "The speed of educational change now puts pressure on conventional school models. "
        "For years, teaching relied heavily on classrooms, textbooks, and instructors guiding lessons. "
        "Students navigate a world saturated with information from YouTube, TikTok, and countless online resources. "
        "Schools still try to balance old expectations with new tools.\n\n"
        "Teachers now have to help students judge information instead of simply delivering it. "
        "They need to explain concepts, question sources, and guide students through uncertainty. "
        "Yet schools often cling to outdated practices like high-stakes testing and rigid grading. "
        "That tension makes reform difficult.\n\n"
        "Technology creates useful options but also new risks. "
        "AI-generated tools can support practice, feedback, and explanation. "
        "They can also hide weak understanding or widen gaps between students. "
        "The result is a system that looks modern but remains uneven.\n\n"
        "Education therefore needs more than new devices or software. "
        "It needs clearer judgment about what learning is for. "
        "Without that, schools will keep reacting to change instead of shaping it."
    ),
    original_report=original_report,
    candidate_report=risky_external_report,
)
risky_external_decision = decide_candidate(
    goal=risky_external_goal,
    original_report=original_report,
    candidate_report=risky_external_report,
    reference_ai=69.7,
    required_ai_drop=5.0,
    target_ai_score=64.7,
)
assert_test(
    risky_external_goal.external_detector_proxy
    and not risky_external_goal.external_detector_proxy.get("safe"),
    "V2 external detector proxy flags top-k/generic smooth near-miss risk",
)
assert_test(
    risky_external_decision.lane == CandidateLane.PARTIAL_DIAGNOSTIC
    and risky_external_decision.reason == "external_detector_proxy_not_safe",
    "V2 selector does not apply unsafe external-proxy candidates as safe near-miss",
)

unsafe_low_ai = {
    "strategy": "unsafe_low_ai",
    "candidate_ai": 34.0,
    "decision": {
        "lane": CandidateLane.PARTIAL_DIAGNOSTIC.value,
        "quality_safe": True,
        "semantic_safe": False,
        "required_drop_met": True,
        "ai_target_gap": 0.0,
        "rank": [1, 1, 0, 0, 20, 20, 1, 0, -1],
    },
}
safe_close_partial = {
    "strategy": "safe_close_partial",
    "candidate_ai": 50.0,
    "decision": {
        "lane": CandidateLane.PARTIAL_DIAGNOSTIC.value,
        "quality_safe": True,
        "semantic_safe": True,
        "required_drop_met": False,
        "ai_target_gap": 0.38,
        "rank": [1, 0, 0, -0.38, 4.62, 4.62, 1, 1, -2],
    },
}
assert_test(
    select_best_applicable_candidate([unsafe_low_ai, safe_close_partial], close_partial_max_gap=1.0)["strategy"] == "safe_close_partial",
    "V2 selector prefers applicable safe frontiers over lower-scoring semantic-unsafe diagnostics",
)

academic_guard = check_semantic_drift(
    "Consumers' experience is affected when staff miss requests. Hence, service quality declines.",
    "Customer experience is affected when staff miss requests. Service quality therefore declines.",
    threshold=0.15,
)
assert_test(
    academic_guard.accepted,
    "Semantic guard does not hard-fail generic capitalized academic terms",
)

compose_text, compose_patches = _compose_full_doc_delta_winners(
    "Paragraph one about Apple.\n\nParagraph two about Tesla.",
    [
        {
            "paragraph_id": "p001",
            "candidate_ai": 40.0,
            "decision": {"quality_safe": True, "semantic_safe": False},
            "semantic_safe": False,
            "protected_anchors_safe": True,
            "patches": [{
                "applied": True,
                "target_paragraph": "Paragraph one about Apple.",
                "rewritten_paragraph": "Paragraph one without Apple.",
            }],
        },
        {
            "paragraph_id": "p002",
            "candidate_ai": 45.0,
            "decision": {"quality_safe": True, "semantic_safe": True},
            "semantic_safe": True,
            "protected_anchors_safe": True,
            "patches": [{
                "applied": True,
                "target_paragraph": "Paragraph two about Tesla.",
                "rewritten_paragraph": "Tesla appears in the second paragraph.",
            }],
        },
    ],
    54.62,
)
assert_test(
    "Paragraph one about Apple." in compose_text and len(compose_patches) == 1,
    "V2 composition skips semantic-unsafe paragraph winners",
)

with tempfile.TemporaryDirectory() as tmpdir:
    near_miss_result = run_rewrite_pipeline_v2(
        detect_json=scan_json,
        output_dir=tmpdir,
        replay_candidate_records=[{
            "strategy": "safe_near_miss_score_target",
            "text": "The United States has influenced politics, technology, and culture.",
            "report": {
                "ai_risk_badge": {
                    "ai_likelihood_score": 49.0,
                    "writing_quality_score": 58.0,
                    "ai_components": {"topk_calibrated_risk": 20},
                },
                "integrity_layers": {"layers": {"ai_authorship_risk": {"score": 30}, "ai_transformation_risk": {"score": 40}}},
                "findings": {"critical": [], "high": [{"id": "f001"}], "medium": [], "low": []},
            },
        }],
    )
near_miss_summary = near_miss_result["result"].summary
assert_test(
    near_miss_summary["final_text"] != scan_json["input_text"],
    "V2 applies score-target safe near-miss candidates as rewritten output",
)
assert_test(
    near_miss_result["status"] == "safe_near_miss_applied",
    "V2 exposes applied safe near-miss separately from strict success",
)
assert_test(
    near_miss_summary["rewrite_goal_status"]["status"] == RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value,
    "V2 keeps strict goal status failed for applied safe near-miss candidates",
)

with tempfile.TemporaryDirectory() as tmpdir:
    close_partial_result = run_rewrite_pipeline_v2(
        detect_json=scan_json,
        output_dir=tmpdir,
        replay_candidate_records=[{
            "strategy": "close_partial_frontier",
            "text": "The United States has influenced politics, technology, and culture.",
            "report": {
                "ai_risk_badge": {
                    "ai_likelihood_score": 50.0,
                    "writing_quality_score": 58.0,
                    "ai_components": {"topk_calibrated_risk": 58},
                },
                "integrity_layers": {"layers": {"ai_authorship": {"score": 48}, "ai_transformation": {"score": 44}}},
                "findings": {"critical": [], "high": [{"id": "f001"}], "medium": [], "low": []},
            },
        }],
    )
close_partial_summary = close_partial_result["result"].summary
assert_test(
    close_partial_summary["final_text"] != scan_json["input_text"],
    "V2 applies close safe partial candidates instead of preserving original text",
)
assert_test(
    close_partial_summary["rewrite_goal_status"]["reason"] == "close_score_frontier_applied_but_target_not_met",
    "V2 reports close partial application without calling it target success",
)
assert_test(
    close_partial_result["status"] == "safe_partial_mitigation_applied",
    "V2 top-level status no longer says no safe candidate when safe partial text is applied",
)
assert_test(
    close_partial_summary.get("rewrite_effective_config", {}).get("apply_partial_max_gap") == 2.0,
    "V2 records effective close-partial tolerance in rewrite summary",
)

previous_partial_gap = os.environ.get("DRAFTPROOF_REWRITE_V2_APPLY_PARTIAL_MAX_GAP")
os.environ["DRAFTPROOF_REWRITE_V2_APPLY_PARTIAL_MAX_GAP"] = "2.0"
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        production_gap_result = run_rewrite_pipeline_v2(
            detect_json=scan_json,
            output_dir=tmpdir,
            replay_candidate_records=[{
                "strategy": "production_safe_partial_gap",
                "text": "The United States has influenced politics, technology, culture, education, and business.",
                "report": {
                    "ai_risk_badge": {
                        "ai_likelihood_score": 51.48,
                        "writing_quality_score": 56.88,
                        "ai_components": {"topk_calibrated_risk": 60},
                    },
                    "integrity_layers": {"layers": {"ai_authorship": {"score": 51}, "ai_transformation": {"score": 44}}},
                    "findings": {"critical": [], "high": [{"id": "f001"}], "medium": [], "low": []},
                },
            }],
        )
finally:
    if previous_partial_gap is None:
        os.environ.pop("DRAFTPROOF_REWRITE_V2_APPLY_PARTIAL_MAX_GAP", None)
    else:
        os.environ["DRAFTPROOF_REWRITE_V2_APPLY_PARTIAL_MAX_GAP"] = previous_partial_gap
production_gap_summary = production_gap_result["result"].summary
assert_test(
    production_gap_summary["final_text"] != scan_json["input_text"],
    "V2 applies safe production-like partial frontier with target gap under 2 points",
)
assert_test(
    production_gap_summary["rewrite_goal_status"]["reason"] == "close_score_frontier_applied_but_target_not_met",
    "V2 reports production-like safe partial as applied but not strict success",
)
assert_test(
    production_gap_result["status"] == "safe_partial_mitigation_applied",
    "V2 production-like safe partial does not return misleading no-safe-candidate status",
)

with tempfile.TemporaryDirectory() as tmpdir:
    coverage_result = run_rewrite_pipeline_v2(
        detect_json=scan_json,
        output_dir=tmpdir,
        replay_candidate_records=[
            {
                "strategy": "scan_targeted_driver_mitigation",
                "strategy_kind": "targeted",
                "paragraph_id": "p003",
                "applied_patch_count": 1,
                "text": "Single paragraph rewrite.",
                "report": {
                    "ai_risk_badge": {
                        "ai_likelihood_score": 50.96,
                        "writing_quality_score": 58.84,
                        "ai_components": {"topk_calibrated_risk": 60},
                    },
                    "integrity_layers": {"layers": {"ai_authorship": {"score": 51}, "ai_transformation": {"score": 44}}},
                    "findings": {"critical": [], "high": [{"id": "f001"}], "medium": [], "low": []},
                },
            },
            {
                "strategy": "scan_targeted_composed_full_doc_delta_winners",
                "strategy_kind": "targeted_composition",
                "composed_patches": [{"paragraph_id": "p001"}, {"paragraph_id": "p002"}, {"paragraph_id": "p003"}, {"paragraph_id": "p004"}],
                "text": "Four paragraph composition rewrite.",
                "report": {
                    "ai_risk_badge": {
                        "ai_likelihood_score": 52.36,
                        "writing_quality_score": 56.06,
                        "ai_components": {"topk_calibrated_risk": 62},
                    },
                    "integrity_layers": {"layers": {"ai_authorship": {"score": 52}, "ai_transformation": {"score": 45}}},
                    "findings": {"critical": [], "high": [{"id": "f001"}], "medium": [], "low": []},
                },
            },
        ],
    )
coverage_summary = coverage_result["result"].summary
assert_test(
    (coverage_summary.get("selected_candidate") or {}).get("strategy") == "scan_targeted_composed_full_doc_delta_winners",
    "V2 prefers safe multi-paragraph composition over one-paragraph frontier within bounded AI penalty",
)
assert_test(
    coverage_summary["final_text"] == "Four paragraph composition rewrite.",
    "V2 applies the coverage-preferred composed rewrite",
)

fragment_failures = _patch_filter_failures([{
    "target_paragraph": "The United States was founded in 1776 after the American colonies declared independence from Britain.",
    "rewritten_paragraph": "The United States declared independence. From Britain. In 1776. The American Revolutionary War followed. Designed to balance power.",
}])
assert_test(
    any(str(item).startswith("surface_quality:") for item in fragment_failures),
    "V2 rejects fragment-heavy rewrites that can trigger external AI detectors",
)
previous_tactics = os.environ.get("DRAFTPROOF_REWRITE_V2_TACTICS")
os.environ["DRAFTPROOF_REWRITE_V2_TACTICS"] = "minimal_carrier,broken_choppy,choppy_analytic"
try:
    filtered_tactics = _paragraph_tactics()
finally:
    if previous_tactics is None:
        os.environ.pop("DRAFTPROOF_REWRITE_V2_TACTICS", None)
    else:
        os.environ["DRAFTPROOF_REWRITE_V2_TACTICS"] = previous_tactics
assert_test(
    "broken_choppy" not in filtered_tactics,
    "V2 disables fragment-prone broken_choppy tactic by default",
)

with tempfile.TemporaryDirectory() as tmpdir:
    no_candidate_result = run_rewrite_pipeline_v2(
        detect_json=scan_json,
        output_dir=tmpdir,
        api_key=None,
    )
no_candidate_summary = no_candidate_result["result"].summary
assert_test(
    no_candidate_summary["rewrite_goal_status"]["reason"] == "candidate_generation_failed_no_candidates",
    "V2 distinguishes zero candidate generation from unsafe candidate selection",
)
assert_test(
    no_candidate_summary["candidate_generation_status"]["generated_count"] == 0,
    "V2 records zero generated candidates in summary diagnostics",
)

paragraph_map = _paragraph_target_map(
    {"rewrite_edit_briefs": [{"paragraph_id": "p002", "paragraph_excerpt": "truncated paragraph"}]},
    "First full paragraph.\n\nSecond full paragraph with exact source text.",
)
assert_test(
    paragraph_map["p002"] == "Second full paragraph with exact source text.",
    "V2 prefers real document paragraphs over truncated paragraph excerpts",
)

reconstruction_expected_count = _expected_full_reconstruction_paragraph_count(
    {"rewrite_edit_briefs": [{"paragraph_id": "p001"}, {"paragraph_id": "p002"}, {"paragraph_id": "p003"}]},
    "Flattened sentence-map text without blank-line paragraphs.",
)
assert_test(
    reconstruction_expected_count == 3,
    "V2 full reconstruction guard uses scan paragraph ids instead of flattened sentence-map text",
)

required_entities = _required_entities_for_full_reconstruction(
    "The United States declared independence from Britain. This led to the American Revolutionary War."
)
assert_test(
    "Britain. This" not in required_entities and "American Revolutionary War. The" not in required_entities,
    "V2 full reconstruction entity lock rejects sentence-boundary entity artifacts",
)
generic_entities = _required_entities_for_full_reconstruction(
    "AlphaCom released NovaAI in 2025. Later, Dr. Maria Chen tested it near Singapore."
)
assert_test(
    "AlphaCom" in generic_entities and "NovaAI" in generic_entities and "Singapore" in generic_entities,
    "V2 full reconstruction entity lock is content-agnostic and preserves dynamic anchors",
)
paragraph_inventory = _paragraph_inventory_for_full_reconstruction(
    {},
    "AlphaCom released NovaAI in 2025.\n\nDr. Maria Chen tested it near Singapore.",
)
assert_test(
    len(paragraph_inventory) == 2 and paragraph_inventory[0]["paragraph_id"] == "p001",
    "V2 builds a dynamic paragraph inventory for full reconstruction prompts",
)
cleaned_meta = _strip_rewrite_meta_text(
    "Here's a rewritten version:\n\n---\n\nParagraph one.\n\nParagraph two.\n\n---\n\nChanges made:\n- varied wording"
)
assert_test(
    cleaned_meta == "Paragraph one.\n\nParagraph two.",
    "V2 strips LLM wrapper/meta text from generated candidates",
)
anchor_repaired = _restore_required_anchor_forms(
    "Hollywood and the National Basketball Association appear here. The UN and NATO are involved.",
    ["The National Basketball Association", "United Nations", "North Atlantic Treaty Organization"],
)
assert_test(
    "The National Basketball Association" in anchor_repaired
    and "United Nations" in anchor_repaired
    and "North Atlantic Treaty Organization" in anchor_repaired,
    "V2 restores generic acronym and article anchor forms before rescoring",
)
author_prompt = _build_author_stance_thesis_reframe_prompt(
    original_text="AlphaCom released NovaAI in 2025. It created debate about schools.",
    required_entities=["AlphaCom", "NovaAI"],
    paragraph_inventory=[{"paragraph_id": "p001", "keywords": ["released", "debate", "schools"]}],
    target_paragraph_count=4,
)
assert_test(
    "4 paragraphs" in author_prompt and "Do not preserve the broad one-topic-per-paragraph survey shape" in author_prompt,
    "V2 author-stance thesis prompt requests merged argument structure instead of survey shape",
)
author_filter_failures = _author_stance_thesis_filter_failures(
    candidate_text=(
        "I find AlphaCom hard to judge because NovaAI looks useful and risky.\n\n"
        "I think the school debate matters because a tool can help work without settling trust.\n\n"
        "The stronger point is not that technology is good or bad. It is that people still need rules.\n\n"
        "I do not see NovaAI as a simple success story, because adoption is easier than judgment."
    ),
    required_entities=["AlphaCom", "NovaAI"],
    min_paragraphs=4,
    max_paragraphs=4,
)
assert_test(
    not author_filter_failures,
    "V2 author-stance thesis filter accepts anchored four-paragraph argument candidates",
)
assert_test(
    "powerhouse" in author_prompt and "Do not open paragraphs with category labels" in author_prompt,
    "V2 author-stance thesis prompt discourages formal survey phrases without blocking rescoring",
)
anchor_coverage_failures = _author_stance_thesis_filter_failures(
    candidate_text=(
        "I find AlphaCom difficult to judge because NovaAI solves one problem and creates another.\n\n"
        "I think the useful part is clear, but the public cost is harder to settle.\n\n"
        "That picture is incomplete without asking who controls the tool.\n\n"
        "I do not see it as a simple success story."
    ),
    required_entities=["AlphaCom", "NovaAI", "tool", "success", "Singapore"],
    min_paragraphs=4,
    max_paragraphs=4,
)
assert_test(
    not anchor_coverage_failures,
    "V2 author-stance thesis filter allows high anchor coverage before semantic rescoring",
)
texture_prompt = _build_author_stance_texture_pass_prompt(
    source_text="AlphaCom released NovaAI in 2025. Schools debated its use.",
    draft_text=(
        "Economically, AlphaCom is undeniably powerful because NovaAI changed schools.\n\n"
        "Culturally, the tool complicates its legacy.\n\n"
        "Globally, people debated the same question.\n\n"
        "I do not see this as simple."
    ),
    required_entities=["AlphaCom", "NovaAI"],
    target_paragraph_count=4,
)
assert_test(
    "texture pass, not a new essay" in texture_prompt
    and "Hard rule: no paragraph may begin with a category label" in texture_prompt
    and "Use only facts already present" in texture_prompt,
    "V2 author texture pass preserves facts while targeting formal survey rhythm",
)
texture_filter_failures = _author_stance_thesis_filter_failures(
    candidate_text=(
        "AlphaCom is hard to judge because NovaAI solves one problem and creates another.\n\n"
        "The useful part is clear, but the public cost is harder to settle.\n\n"
        "That picture is incomplete without asking who controls the tool.\n\n"
        "It is not a simple success story."
    ),
    required_entities=["AlphaCom", "NovaAI", "tool", "success"],
    min_paragraphs=4,
    max_paragraphs=4,
    require_author_stance_marker=False,
)
assert_test(
    not texture_filter_failures,
    "V2 author texture pass can rescore plain-stance candidates without first-person markers",
)
assert_test(
    not _has_rewrite_meta_text("Power here is both creative and uneven."),
    "V2 meta-text detector does not reject normal prose containing here is",
)
assert_test(
    _has_rewrite_meta_text("Here is the rewritten essay:\n\nAlphaCom changed the debate."),
    "V2 meta-text detector still rejects wrapper text",
)
survey_failures = _survey_style_failures(
    "Economically, AlphaCom is a powerhouse.\n\nCulturally, NovaAI has massive impact."
)
assert_test(
    any(str(item).startswith("survey_opening:") for item in survey_failures)
    and any(str(item).startswith("survey_phrase:") for item in survey_failures),
    "V2 survey-style filter catches category openings and polished detector-risk phrases",
)
assert_test(
    _author_strategy_semantic_override_allowed(
        strategy_kind="author_stance_texture_pass",
        generated_candidate={"required_entities": ["AlphaCom", "NovaAI", "schools", "Singapore"]},
        candidate_text="AlphaCom and NovaAI changed schools, but the question stayed local in Singapore.",
        semantic_similarity=0.78,
        anchors_safe=True,
    ),
    "V2 author texture pass can use high-similarity anchor coverage semantic override",
)
assert_test(
    not _author_strategy_semantic_override_allowed(
        strategy_kind="author_stance_texture_pass",
        generated_candidate={"required_entities": ["AlphaCom", "NovaAI", "schools", "Singapore"]},
        candidate_text="AlphaCom changed the debate.",
        semantic_similarity=0.78,
        anchors_safe=True,
    ),
    "V2 author texture semantic override rejects low anchor coverage",
)

cluster_text = _cluster_text_from_gate(
    "First sentence. Second sentence. Third sentence. Fourth sentence.",
    {"start_sentence": 1, "end_sentence": 2},
)
assert_test(
    cluster_text == "Second sentence. Third sentence.",
    "V2 extracts unsafe cluster text by sentence window",
)
rewritten_cluster_text, cluster_applied = _replace_once_flexible(
    "First sentence.\n\nSecond sentence.   Third sentence.\n\nFourth sentence.",
    "Second sentence. Third sentence.",
    "Second sentence changed. Third sentence changed.",
)
assert_test(
    cluster_applied and "Second sentence changed. Third sentence changed." in rewritten_cluster_text,
    "V2 replaces unsafe cluster text across whitespace differences",
)

entity_start_drift = check_semantic_drift(
    "The entertainment industry in Hollywood has become a major export.",
    "Hollywood stands out as a major export.",
    threshold=0.15,
)
assert_test(
    entity_start_drift.accepted,
    "V2 semantic guard preserves entities moved to sentence starts",
)
quantifier_drift = check_semantic_drift(
    "Many Customers disagreed with the policy. AlphaCom revised it in 2025.",
    "Customers disagreed with the policy. AlphaCom revised it in 2025.",
    threshold=0.15,
)
assert_test(
    quantifier_drift.accepted,
    "V2 semantic guard does not treat quantifier-prefixed noun phrases as named entities",
)
soft_quote_original = 'Education should focus on “what students know,” but also on “how students think.”'
soft_quote_candidate = "Education should focus on knowledge, but also on student reasoning."
soft_quote_protected = detect_protected_spans(soft_quote_original)
soft_quote_drift = check_semantic_drift(soft_quote_original, soft_quote_candidate, threshold=0.15)
assert_test(
    all(span.reason != "direct_quote" for span in soft_quote_protected)
    and protected_spans_preserved(soft_quote_original, soft_quote_candidate, soft_quote_protected)
    and all(not reason.startswith("quote_lost") for reason in soft_quote_drift.reasons),
    "V2 quote guard treats short conceptual emphasis quotes as soft semantic content",
)
hard_quote_original = 'A learner may say “I do not know where to begin” during assessment.'
hard_quote_candidate = "A learner may say they are unsure during assessment."
hard_quote_protected = detect_protected_spans(hard_quote_original)
hard_quote_drift = check_semantic_drift(hard_quote_original, hard_quote_candidate, threshold=0.15)
assert_test(
    any(span.reason == "direct_quote" for span in hard_quote_protected)
    and not protected_spans_preserved(hard_quote_original, hard_quote_candidate, hard_quote_protected)
    and any(reason.startswith("quote_lost") for reason in hard_quote_drift.reasons),
    "V2 quote guard still hard-protects attributed direct quotes",
)

print("All rewrite V2 pipeline tests passed.")
