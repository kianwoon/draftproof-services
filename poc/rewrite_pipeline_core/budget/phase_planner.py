"""Shared phase-budget planning for the rewrite pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def rewrite_phase_budget_plan(
    source_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    max_scans: int = 14,
    max_llm_calls: int = 10,
    ai_search_policy: dict | None = None,
    formula_gap_budget: dict | None = None,
    text_word_count: Callable[[str], int],
    formula_convergence_budget: Callable[[str], dict],
    turnitin_like_ai_profile: Callable[[dict | None], dict],
    eligible_span_density_contract: Callable[[str, dict | None], dict],
    env_flag: Callable[[str, bool], bool],
    safe_topk_calibrated_limit: Callable[[], float],
    verified_candidate_scan_budget: Callable[[str, dict | None], dict],
    float_env: Callable[[str, float], float],
) -> dict:
    """Allocate the shared rewrite budget before high-cost phases start.

    The global ledger records what happened after each phase. This planner
    decides what each major controller is allowed to spend before AI search can
    consume the whole run-level budget.
    """

    total_scans = max(0, int(max_scans or 0))
    total_llm = max(0, int(max_llm_calls or 0))
    ai_policy = ai_search_policy if isinstance(ai_search_policy, dict) else {}
    formula_contract = formula_gap_budget if isinstance(formula_gap_budget, dict) else {}
    formula_default = formula_convergence_budget(source_text)
    profile = turnitin_like_ai_profile(current_report if isinstance(current_report, dict) else original_report or {})
    density = eligible_span_density_contract(source_text, current_report if isinstance(current_report, dict) else {})
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    topk_risk = float(components.get("topk_calibrated_risk") or 0.0)
    density_unsafe = not bool(density.get("safe"))
    target_unmet = not bool(profile.get("target_met"))
    segment_needed = bool(
        env_flag("DRAFTPROOF_SEGMENT_WINDOW_DENSITY_CONTROLLER", True)
        and target_unmet
        and (density_unsafe or topk_risk >= safe_topk_calibrated_limit())
    )

    try:
        ai_desired_scans = int(ai_policy.get("max_candidate_scans") or 0)
    except (TypeError, ValueError):
        ai_desired_scans = 0
    if ai_desired_scans <= 0:
        ai_desired_scans = int(verified_candidate_scan_budget(source_text, original_report).get("max_candidate_scans") or 0)
    try:
        ai_desired_llm = int(ai_policy.get("max_llm_calls") or 0)
    except (TypeError, ValueError):
        ai_desired_llm = 0
    if ai_desired_llm <= 0:
        ai_desired_llm = min(total_llm or 10, int(formula_contract.get("llm_candidate_calls") or 5))

    if segment_needed:
        formula_scans = min(int(formula_default.get("max_scans") or 0), 2)
        formula_llm = min(int(formula_default.get("max_llm_calls") or 0), 1)
        segment_scans = min(3, total_scans)
        segment_llm = min(3, total_llm)
        post_segment_scans = min(1, total_scans)
        post_segment_llm = 0
        unallocated_scan_reserve = 2 if total_scans >= 12 else 1 if total_scans >= 8 else 0
        unallocated_llm_reserve = 1 if total_llm >= 8 else 0
        base_scan_contract = 14
        base_llm_contract = 10
        extra_scans = max(0, total_scans - base_scan_contract)
        extra_llm = max(0, total_llm - base_llm_contract)
        if extra_scans > 0:
            segment_goal_scan_cap = max(
                segment_scans,
                int(float_env("DRAFTPROOF_SEGMENT_WINDOW_GOAL_MAX_SCANS", 12.0)),
            )
            formula_goal_scan_cap = max(
                formula_scans,
                int(float_env("DRAFTPROOF_FORMULA_CONVERGENCE_GOAL_MAX_SCANS", 6.0)),
            )
            post_goal_scan_cap = max(
                post_segment_scans,
                int(float_env("DRAFTPROOF_POST_SEGMENT_GOAL_MAX_SCANS", 3.0)),
            )
            segment_extra = min(extra_scans, max(0, segment_goal_scan_cap - segment_scans))
            segment_scans += segment_extra
            extra_scans -= segment_extra
            formula_extra = min(extra_scans, max(0, formula_goal_scan_cap - formula_scans))
            formula_scans += formula_extra
            extra_scans -= formula_extra
            post_extra = min(extra_scans, max(0, post_goal_scan_cap - post_segment_scans))
            post_segment_scans += post_extra
        if extra_llm > 0:
            segment_goal_llm_cap = max(
                segment_llm,
                int(float_env("DRAFTPROOF_SEGMENT_WINDOW_GOAL_MAX_LLM_CALLS", 10.0)),
            )
            formula_goal_llm_cap = max(
                formula_llm,
                int(float_env("DRAFTPROOF_FORMULA_CONVERGENCE_GOAL_MAX_LLM_CALLS", 3.0)),
            )
            segment_llm_extra = min(extra_llm, max(0, segment_goal_llm_cap - segment_llm))
            segment_llm += segment_llm_extra
            extra_llm -= segment_llm_extra
            formula_llm_extra = min(extra_llm, max(0, formula_goal_llm_cap - formula_llm))
            formula_llm += formula_llm_extra
    else:
        formula_scans = min(int(formula_default.get("max_scans") or 0), 3)
        formula_llm = min(int(formula_default.get("max_llm_calls") or 0), 1)
        segment_scans = 0
        segment_llm = 0
        post_segment_scans = 0
        post_segment_llm = 0
        unallocated_scan_reserve = 1 if total_scans >= 8 else 0
        unallocated_llm_reserve = 0

    reserved_scans = formula_scans + segment_scans + post_segment_scans + unallocated_scan_reserve
    reserved_llm = formula_llm + segment_llm + post_segment_llm + unallocated_llm_reserve
    ai_max_scans = max(0, total_scans - reserved_scans) if total_scans else ai_desired_scans
    ai_max_llm = max(0, total_llm - reserved_llm) if total_llm else ai_desired_llm
    if segment_needed and total_scans >= 10:
        ai_max_scans = min(ai_max_scans, 6)
    if segment_needed and total_llm >= 8:
        ai_max_llm = min(ai_max_llm, 5)

    ai_scans = max(0, min(ai_desired_scans, ai_max_scans))
    ai_llm = max(0, min(ai_desired_llm, ai_max_llm))
    used_scans = ai_scans + formula_scans + segment_scans + post_segment_scans
    used_llm = ai_llm + formula_llm + segment_llm + post_segment_llm

    return {
        "version": "phase_budget_plan_v1",
        "enabled": True,
        "reason": (
            "segment_density_or_topk_goal_first_extended"
            if segment_needed and (total_scans > 14 or total_llm > 10)
            else "segment_density_or_topk_priority"
            if segment_needed
            else "standard_formula_priority"
        ),
        "total": {
            "max_scans": total_scans,
            "max_llm_calls": total_llm,
        },
        "drivers": {
            "turnitin_like_target_met": bool(profile.get("target_met")),
            "eligible_span_density_safe": bool(density.get("safe")),
            "topk_calibrated_risk": round(topk_risk, 3),
            "segment_window_needed": segment_needed,
        },
        "phases": {
            "ai_mitigation_search": {
                "max_scans": ai_scans,
                "max_llm_calls": ai_llm,
                "desired_scans": ai_desired_scans,
                "desired_llm_calls": ai_desired_llm,
            },
            "formula_convergence_controller": {
                "max_scans": formula_scans,
                "max_llm_calls": formula_llm,
            },
            "segment_window_density_controller": {
                "max_scans": segment_scans,
                "max_llm_calls": segment_llm,
                "needed": segment_needed,
            },
            "post_segment_followup": {
                "max_scans": post_segment_scans,
                "max_llm_calls": post_segment_llm,
            },
            "unallocated_reserve": {
                "max_scans": max(0, total_scans - used_scans) if total_scans else 0,
                "max_llm_calls": max(0, total_llm - used_llm) if total_llm else 0,
                "planned_floor_scans": unallocated_scan_reserve,
                "planned_floor_llm_calls": unallocated_llm_reserve,
            },
        },
    }
