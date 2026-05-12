from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class IterativeTopkRouteOptimizerDeps:
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    best_ai_search_selectable: Callable[[], bool]
    safe_topk_calibrated_limit: Callable[[], float]
    verified_candidate_scans_used: Callable[[], int]
    extend_candidate_scan_budget: Callable[[dict, int, int], None]
    search_budget_exhausted: Callable[[str], bool]
    ai_footprint_gate_status: Callable[..., dict]
    review_burden: Callable[[dict | None], float]
    weighted_severity: Callable[[dict | None], float]
    critical_high_count: Callable[[dict | None], int]
    topk_route_optimizer_candidates: Callable[[str, dict | None], list[tuple[str, str, dict]]]
    evaluate_ai_search_candidate: Callable[..., None]
    get_best_text: Callable[[], str]
    get_best_report: Callable[[], dict | None]
    get_best_strategy: Callable[[], str]
    get_best_selection_status: Callable[[], dict | None]


def run_iterative_topk_route_optimizer(
    trigger_phase: str,
    *,
    search_summary: dict,
    search_budget: dict,
    adaptive_stop_reason: str,
    original_report_dict: dict | None,
    original_review_burden: float,
    original_severity: float,
    saved_critical_high: int,
    deps: IterativeTopkRouteOptimizerDeps,
) -> str:
    """Run bounded deterministic top-k route rounds against the current best candidate."""
    if not deps.env_flag("DRAFTPROOF_ITERATIVE_TOPK_ROUTE_OPTIMIZER", True):
        search_summary["iterative_topk_route_optimizer"] = {
            "enabled": False,
            "reason": "disabled",
            "trigger_phase": trigger_phase,
        }
        return adaptive_stop_reason
    if not deps.best_ai_search_selectable() or not isinstance(deps.get_best_report(), dict):
        search_summary["iterative_topk_route_optimizer"] = {
            "enabled": True,
            "skipped": True,
            "reason": "no_selectable_base",
            "trigger_phase": trigger_phase,
        }
        return adaptive_stop_reason
    try:
        max_rounds = max(1, int(deps.float_env("DRAFTPROOF_ITERATIVE_TOPK_ROUTE_ROUNDS", 3.0)))
    except (TypeError, ValueError):
        max_rounds = 3
    target_drop = deps.float_env("DRAFTPROOF_AI_FOOTPRINT_SATURATED_MIN_TOPK_DROP", 8.0)
    safe_topk = deps.safe_topk_calibrated_limit()
    active_threshold = deps.float_env("DRAFTPROOF_AI_FOOTPRINT_ACTIVE_TOPK_THRESHOLD", 90.0)
    reserve = max(
        0,
        int(deps.float_env(
            "DRAFTPROOF_ITERATIVE_TOPK_ROUTE_SCAN_RESERVE",
            float(max_rounds * 4),
        )),
    )
    if reserve > 0:
        current_scans = deps.verified_candidate_scans_used()
        deps.extend_candidate_scan_budget(search_budget, current_scans, reserve)
    if str(adaptive_stop_reason or "").startswith("budget_exhausted"):
        adaptive_stop_reason = ""
    summary = {
        "enabled": True,
        "trigger_phase": trigger_phase,
        "max_rounds": max_rounds,
        "target_drop": target_drop,
        "safe_topk": safe_topk,
        "active_threshold": active_threshold,
        "scan_reserve_added": reserve,
        "base_strategy": deps.get_best_strategy(),
        "rounds": [],
    }
    for round_index in range(1, max_rounds + 1):
        best_report = deps.get_best_report()
        if not isinstance(best_report, dict):
            break
        gate_before_round = deps.ai_footprint_gate_status(
            original_report_dict,
            best_report,
            review_burden_delta=deps.review_burden(best_report) - original_review_burden,
            weighted_severity_delta=deps.weighted_severity(best_report) - original_severity,
            critical_high_delta=deps.critical_high_count(best_report) - saved_critical_high,
            ai_score_regressed=False,
        )
        topk_before = (
            (gate_before_round.get("after") or {})
            .get("authorship_footprint", {})
            .get("topk_calibrated_risk")
        )
        topk_drop_before = (gate_before_round.get("drops") or {}).get("topk_calibrated_risk")
        before_strategy = deps.get_best_strategy()
        round_summary = {
            "round": round_index,
            "base_strategy": before_strategy,
            "topk_before": topk_before,
            "topk_drop_before": topk_drop_before,
            "candidate_count": 0,
            "selected_strategy_before": before_strategy,
        }
        if (
            isinstance(topk_before, (int, float))
            and float(topk_before) < safe_topk
        ):
            round_summary["skipped"] = True
            round_summary["reason"] = "topk_target_reached"
            summary["rounds"].append(round_summary)
            break
        round_candidates = deps.topk_route_optimizer_candidates(deps.get_best_text(), best_report)
        round_summary["candidate_count"] = len(round_candidates)
        if not round_candidates:
            round_summary["reason"] = "no_route_candidates"
            summary["rounds"].append(round_summary)
            break
        before_gate = gate_before_round
        for candidate_index, (strategy, candidate, meta) in enumerate(round_candidates, start=1):
            if deps.search_budget_exhausted("iterative_topk_route_optimizer"):
                round_summary["reason"] = adaptive_stop_reason or "budget_exhausted"
                break
            deps.evaluate_ai_search_candidate(
                f"iterative_{strategy}_r{round_index}_c{candidate_index}",
                candidate,
                deterministic=True,
                extra={
                    **(meta or {}),
                    "iterative_topk_route_optimizer": True,
                    "topk_route_round": round_index,
                    "base_strategy": before_strategy,
                },
            )
        best_selection_status = deps.get_best_selection_status()
        gate_after_round = (
            best_selection_status.get("ai_footprint_gate")
            if isinstance(best_selection_status, dict) else None
        )
        after_strategy = deps.get_best_strategy()
        round_summary.update({
            "selected_strategy_after": after_strategy,
            "selected_changed": after_strategy != before_strategy,
            "topk_drop_after": (
                (gate_after_round.get("drops") or {}).get("topk_calibrated_risk")
                if isinstance(gate_after_round, dict) else None
            ),
            "topk_after": (
                ((gate_after_round.get("after") or {}).get("authorship_footprint") or {}).get("topk_calibrated_risk")
                if isinstance(gate_after_round, dict) else None
            ),
        })
        summary["rounds"].append(round_summary)
        if after_strategy == before_strategy:
            break
        if before_gate == gate_after_round:
            break
    best_selection_status = deps.get_best_selection_status()
    summary["selected_strategy_after"] = deps.get_best_strategy()
    summary["selected_gate_after"] = (
        best_selection_status.get("ai_footprint_gate")
        if isinstance(best_selection_status, dict) else None
    )
    search_summary["iterative_topk_route_optimizer"] = summary
    return adaptive_stop_reason
