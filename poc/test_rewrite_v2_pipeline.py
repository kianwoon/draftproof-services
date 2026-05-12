"""Focused tests for the scan-driven rewrite pipeline V2."""

from __future__ import annotations

import json
import os
import tempfile

from rewrite.guards import check_semantic_drift
from rewrite_v2 import run_rewrite_pipeline_v2
from rewrite_v2.pipeline import _paragraph_target_map
from rewrite_v2.goal_contract import RewriteGoalStatus, evaluate_rewrite_goal, needs_author_context
from rewrite_v2.selection import CandidateLane, decide_candidate
from rewrite_v2.strategy import StrategyKind, route_strategies


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
    result["status"] == RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value,
    "V2 replay reports failed mitigation instead of unsafe partial success",
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
                    "ai_components": {"topk_calibrated_risk": 58},
                },
                "integrity_layers": {"layers": {"ai_authorship": {"score": 48}, "ai_transformation": {"score": 44}}},
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
    near_miss_result["status"] == RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value,
    "V2 does not label applied safe near-miss candidates as strict success",
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

paragraph_map = _paragraph_target_map(
    {"rewrite_edit_briefs": [{"paragraph_id": "p002", "paragraph_excerpt": "truncated paragraph"}]},
    "First full paragraph.\n\nSecond full paragraph with exact source text.",
)
assert_test(
    paragraph_map["p002"] == "Second full paragraph with exact source text.",
    "V2 prefers real document paragraphs over truncated paragraph excerpts",
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

print("All rewrite V2 pipeline tests passed.")
