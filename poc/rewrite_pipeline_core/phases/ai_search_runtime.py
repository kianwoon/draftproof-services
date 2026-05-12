"""Runtime accounting helpers for the AI-search phase."""

from __future__ import annotations

import time
from typing import Callable


def verified_candidate_scans_used(search_summary: dict) -> int:
    try:
        return int(search_summary.get("full_candidate_scans") or 0)
    except (TypeError, ValueError):
        return 0


def record_verified_candidate_scan(search_summary: dict) -> None:
    search_summary["full_candidate_scans"] = verified_candidate_scans_used(search_summary) + 1
    controller = search_summary.get("candidate_scoring_controller")
    if isinstance(controller, dict):
        controller["full_scans_used"] = search_summary["full_candidate_scans"]
        controller["candidate_records"] = len(search_summary.get("candidates", [])) + 1


def phase_budget_can_spend(
    phase: str,
    *,
    calls: int = 1,
    phase_budget_used: dict,
    phase_budget_contract: dict,
    search_summary: dict,
    env_flag: Callable[..., bool],
) -> bool:
    if phase not in phase_budget_used:
        return True
    if (
        phase == "topk_safe_band_rebuild"
        and env_flag("DRAFTPROOF_TOPK_CAN_BORROW_UNUSED_PHASE_BUDGET", False)
        and bool(search_summary.get("topk_safe_band_priority"))
        and int(search_summary.get("llm_calls") or 0) + int(calls)
        <= int(phase_budget_contract.get("total_llm_hard_cap") or 0)
    ):
        return True
    return (
        int(phase_budget_used.get(phase) or 0) + int(calls)
        <= int(phase_budget_contract.get(phase) or 0)
        and int(search_summary.get("llm_calls") or 0) + int(calls)
        <= int(phase_budget_contract.get("total_llm_hard_cap") or 0)
    )


def record_phase_llm_call(
    phase: str,
    *,
    calls: int = 1,
    phase_budget_used: dict,
    phase_budget_contract: dict,
    search_summary: dict,
) -> None:
    if phase not in phase_budget_used:
        return
    phase_limit = int(phase_budget_contract.get(phase) or 0)
    previous_used = int(phase_budget_used.get(phase) or 0)
    phase_budget_used[phase] = int(phase_budget_used.get(phase) or 0) + int(calls)
    search_summary["phase_budget_used"] = dict(phase_budget_used)
    if (
        phase == "topk_safe_band_rebuild"
        and phase_budget_used[phase] > phase_limit
        and bool(search_summary.get("topk_safe_band_priority"))
    ):
        borrowed = max(0, phase_budget_used[phase] - max(phase_limit, previous_used))
        if borrowed:
            search_summary["topk_phase_budget_borrowed_calls"] = (
                int(search_summary.get("topk_phase_budget_borrowed_calls") or 0) + borrowed
            )


def phase_budget_block_record(
    phase: str,
    summary: dict,
    *,
    calls: int = 1,
    phase_budget_used: dict,
    phase_budget_contract: dict,
    search_summary: dict,
    env_flag: Callable[..., bool],
) -> bool:
    if phase_budget_can_spend(
        phase,
        calls=calls,
        phase_budget_used=phase_budget_used,
        phase_budget_contract=phase_budget_contract,
        search_summary=search_summary,
        env_flag=env_flag,
    ):
        return False
    summary.update({
        "skipped": True,
        "reason": "phase_llm_budget_exhausted",
        "phase": phase,
        "requested_calls": calls,
        "phase_budget_contract": phase_budget_contract,
        "phase_budget_used": dict(phase_budget_used),
        "total_llm_calls": int(search_summary.get("llm_calls") or 0),
    })
    return True


def search_budget_exhausted_record(
    *,
    phase: str,
    before_llm: bool,
    search_started: float,
    search_budget: dict,
    search_summary: dict,
    adaptive_stop_reason: str,
    best_strategy: str | None,
    best_selection_status: dict,
    llm_call_budget_exhausted_before_send: Callable[[int, int], bool],
) -> tuple[bool, str]:
    if adaptive_stop_reason and str(adaptive_stop_reason).startswith("budget_exhausted"):
        return True, adaptive_stop_reason
    elapsed = time.time() - search_started
    reason = ""
    phase_name = str(phase or "")
    llm_bound_phase = any(
        token in phase_name
        for token in ("llm", "topk", "post_topk", "texture", "score_feedback")
    )
    if elapsed >= float(search_budget["max_seconds"]):
        reason = "budget_exhausted_time"
    elif before_llm and llm_call_budget_exhausted_before_send(
        int(search_summary.get("llm_calls") or 0),
        int(search_budget["max_llm_calls"]),
    ):
        reason = "budget_exhausted_llm_calls"
    elif (
        not before_llm
        and llm_bound_phase
        and int(search_summary.get("llm_calls") or 0) >= int(search_budget["max_llm_calls"])
    ):
        reason = "budget_exhausted_llm_calls"
    elif verified_candidate_scans_used(search_summary) >= int(search_budget["max_candidate_scans"]):
        reason = "budget_exhausted_candidate_scans"
    if not reason:
        return False, adaptive_stop_reason
    adaptive_stop_reason = reason
    search_summary["budget_exhausted"] = {
        "phase": phase,
        "reason": reason,
        "seconds": round(elapsed, 3),
        "llm_calls": int(search_summary.get("llm_calls") or 0),
        "candidate_scans": verified_candidate_scans_used(search_summary),
        "candidate_records": len(search_summary.get("candidates", [])),
        **search_budget,
        "selected_strategy": best_strategy,
        "has_selectable_candidate": bool(best_strategy and best_selection_status.get("selectable")),
    }
    search_summary["adaptive_stop"] = {
        "phase": phase,
        "reason": reason,
        "candidate_count_scanned": verified_candidate_scans_used(search_summary),
        "candidate_records": len(search_summary.get("candidates", [])),
        "selected_strategy": best_strategy,
        "selection_status": best_selection_status,
    }
    return True, adaptive_stop_reason


def apply_budget_gateway(gateway, phase: str, *, search_summary: dict, search_budget_exhausted: Callable[..., bool]):
    original_chat = gateway.chat

    def chat_with_budget(*args, **kwargs):
        if search_budget_exhausted(phase, before_llm=True):
            search_summary["llm_calls"] = max(0, int(search_summary.get("llm_calls") or 0) - 1)
            budget_record = search_summary.get("budget_exhausted")
            if isinstance(budget_record, dict):
                budget_record["llm_calls"] = int(search_summary.get("llm_calls") or 0)
            raise RuntimeError(search_summary["budget_exhausted"]["reason"])
        return original_chat(*args, **kwargs)

    gateway.chat = chat_with_budget  # type: ignore[method-assign]
    return gateway


def adaptive_stop_record(
    *,
    phase: str,
    adaptive_stop_reason: str,
    search_summary: dict,
    best_strategy: str | None,
    best_selection_status: dict,
    ai_search_adaptive_stop_reason: Callable[..., str],
    short_document: bool,
) -> tuple[bool, str]:
    if adaptive_stop_reason:
        return True, adaptive_stop_reason
    adaptive_stop_reason = ai_search_adaptive_stop_reason(
        best_selection_status,
        phase=phase,
        short_document=short_document,
    )
    if adaptive_stop_reason:
        search_summary["adaptive_stop"] = {
            "phase": phase,
            "reason": adaptive_stop_reason,
            "candidate_count_scanned": verified_candidate_scans_used(search_summary),
            "candidate_records": len(search_summary.get("candidates", [])),
            "selected_strategy": best_strategy,
            "selection_status": best_selection_status,
        }
        return True, adaptive_stop_reason
    return False, adaptive_stop_reason
