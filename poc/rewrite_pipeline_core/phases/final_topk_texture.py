"""Final Top-k texture repair phase orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os
import time

from llm.gateway import LLMConfig, LLMGateway


@dataclass(frozen=True)
class FinalTopkTextureRepairDeps:
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    adaptive_budget_default: Callable[[str, int, int], int]
    best_ai_search_selectable: Callable[[], bool]
    blocker_scores: Callable[[dict | None], dict]
    safe_topk_calibrated_limit: Callable[[], float]
    verified_candidate_scans_used: Callable[[], int]
    final_topk_texture_scan_reserve: Callable[[str], int]
    extend_candidate_scan_budget: Callable[[dict, int, int], None]
    phase_budget_block_record: Callable[[str, dict], bool]
    llm_call_budget_exhausted_before_send: Callable[[int, int], bool]
    record_phase_llm_call: Callable[[str], None]
    topk_texture_repair_prompt: Callable[..., str]
    phase_chat_sampling_kwargs: Callable[..., dict]
    extract_paragraph_component_candidates: Callable[[str, int], list[str]]
    clean_full_document_candidate: Callable[[str, str], str]
    evaluate_ai_search_candidate: Callable[..., None]
    get_best_text: Callable[[], str]
    get_best_report: Callable[[], dict | None]
    get_best_strategy: Callable[[], str]


def run_final_topk_texture_repair(
    trigger_phase: str,
    *,
    search_summary: dict,
    search_budget: dict,
    adaptive_stop_reason: str,
    formula_gap_orchestrator_completed: bool,
    effective_key: str | None,
    gateway: LLMGateway,
    base_url: str,
    generator_model: str,
    hard_llm_cap: int,
    search_started: float,
    deps: FinalTopkTextureRepairDeps,
) -> str:
    """Run final scoped texture repair when selected text still has Top-k pressure."""
    if formula_gap_orchestrator_completed:
        search_summary["final_topk_texture_repair"] = {
            "enabled": bool(deps.env_flag("DRAFTPROOF_FINAL_TOPK_TEXTURE_REPAIR", True)),
            "skipped": True,
            "reason": "formula_gap_candidate_orchestrator_completed",
            "trigger_phase": trigger_phase,
        }
        return adaptive_stop_reason
    if not deps.env_flag("DRAFTPROOF_FINAL_TOPK_TEXTURE_REPAIR", True):
        search_summary["final_topk_texture_repair"] = {
            "enabled": False,
            "reason": "disabled",
            "trigger_phase": trigger_phase,
        }
        return adaptive_stop_reason
    existing_summary = search_summary.get("final_topk_texture_repair")
    if (
        isinstance(existing_summary, dict)
        and existing_summary.get("enabled")
        and not existing_summary.get("skipped")
    ):
        return adaptive_stop_reason
    if not effective_key:
        search_summary["final_topk_texture_repair"] = {
            "enabled": False,
            "reason": "no_llm_key",
            "trigger_phase": trigger_phase,
        }
        return adaptive_stop_reason
    if not deps.best_ai_search_selectable() or not isinstance(deps.get_best_report(), dict):
        search_summary["final_topk_texture_repair"] = {
            "enabled": True,
            "skipped": True,
            "reason": "no_selectable_base",
            "trigger_phase": trigger_phase,
        }
        return adaptive_stop_reason

    best_text = deps.get_best_text()
    best_report = deps.get_best_report()
    best_strategy = deps.get_best_strategy()
    blockers = deps.blocker_scores(best_report)
    min_topk = deps.float_env("DRAFTPROOF_FINAL_TOPK_TEXTURE_MIN_TOPK", deps.safe_topk_calibrated_limit())
    min_predictability = deps.float_env("DRAFTPROOF_FINAL_TOPK_TEXTURE_MIN_PREDICTABILITY", 75.0)
    min_generic = deps.float_env("DRAFTPROOF_FINAL_TOPK_TEXTURE_MIN_GENERIC_ASSERTION", 70.0)
    active = {
        "topk_calibrated_risk": float(blockers.get("topk_calibrated_risk") or 0.0),
        "topk_pattern_raw": float(blockers.get("topk_pattern_raw", blockers.get("topk_pattern")) or 0.0),
        "predictability": float(blockers.get("predictability") or 0.0),
        "generic_assertion_risk": float(blockers.get("generic_assertion_risk") or 0.0),
    }
    should_run = (
        active["topk_calibrated_risk"] >= min_topk
        or active["predictability"] >= min_predictability
        or active["generic_assertion_risk"] >= min_generic
    )
    try:
        candidate_limit = max(
            1,
            int(deps.float_env(
                "DRAFTPROOF_FINAL_TOPK_TEXTURE_CANDIDATES",
                float(deps.adaptive_budget_default(best_text, 3, 4)),
            )),
        )
    except (TypeError, ValueError):
        candidate_limit = 1
    summary = {
        "enabled": True,
        "trigger_phase": trigger_phase,
        "candidate_limit": candidate_limit,
        "base_strategy": best_strategy,
        "blockers_before": blockers,
        "thresholds": {
            "topk_calibrated_risk": min_topk,
            "predictability": min_predictability,
            "generic_assertion_risk": min_generic,
        },
    }
    if not should_run:
        summary.update({"skipped": True, "reason": "texture_below_threshold"})
        search_summary["final_topk_texture_repair"] = summary
        return adaptive_stop_reason

    elapsed_before_final = time.time() - search_started
    final_min_seconds = deps.float_env("DRAFTPROOF_FINAL_TOPK_TEXTURE_MIN_SECONDS_REMAINING", 25.0)
    seconds_remaining = float(search_budget.get("max_seconds") or 0.0) - elapsed_before_final
    if seconds_remaining < final_min_seconds:
        summary.update({
            "skipped": True,
            "reason": "insufficient_time_budget_for_final_texture_repair",
            "seconds_remaining": round(seconds_remaining, 3),
            "min_seconds_remaining": final_min_seconds,
        })
        search_summary["final_topk_texture_repair"] = summary
        return adaptive_stop_reason

    scan_reserve_reason = str(adaptive_stop_reason or "")
    if (
        scan_reserve_reason != "budget_exhausted_candidate_scans"
        and deps.verified_candidate_scans_used() >= int(search_budget.get("max_candidate_scans") or 0)
    ):
        scan_reserve_reason = "budget_exhausted_candidate_scans"
    scan_reserve = deps.final_topk_texture_scan_reserve(scan_reserve_reason)
    if scan_reserve > 0:
        previous_max = int(search_budget.get("max_candidate_scans") or 0)
        current_scans = deps.verified_candidate_scans_used()
        deps.extend_candidate_scan_budget(search_budget, current_scans, scan_reserve)
        adaptive_stop_reason = ""
        summary["scan_reserve"] = {
            "enabled": True,
            "previous_max_candidate_scans": previous_max,
            "candidate_scans_before_reserve": current_scans,
            "reserve_added": scan_reserve,
            "max_candidate_scans": search_budget["max_candidate_scans"],
        }
    search_summary["final_topk_texture_repair"] = summary
    try:
        current_llm_calls = int(search_summary.get("llm_calls") or 0)
        if current_llm_calls >= hard_llm_cap:
            summary.update({
                "skipped": True,
                "reason": "llm_hard_cap_reached",
                "llm_calls": current_llm_calls,
                "hard_llm_cap": hard_llm_cap,
            })
            return adaptive_stop_reason
        if deps.phase_budget_block_record("final_texture_proxy_repair", summary):
            return adaptive_stop_reason
        next_llm_call_exceeds_budget = deps.llm_call_budget_exhausted_before_send(
            current_llm_calls + 1,
            int(search_budget.get("max_llm_calls") or 0),
        )
        if next_llm_call_exceeds_budget and int(search_budget.get("max_llm_calls") or 0) < hard_llm_cap:
            search_budget["max_llm_calls"] = min(hard_llm_cap, current_llm_calls + 1)
            next_llm_call_exceeds_budget = False
        use_budget_override_gateway = bool(
            (
                str(adaptive_stop_reason or "").startswith("budget_exhausted")
                or next_llm_call_exceeds_budget
            )
            and deps.env_flag("DRAFTPROOF_FINAL_TOPK_TEXTURE_AFTER_BUDGET", True)
            and effective_key
            and current_llm_calls < hard_llm_cap
        )
        prompt = deps.topk_texture_repair_prompt(
            best_text,
            best_report,
            candidate_count=candidate_limit,
        )
        search_summary["llm_calls"] += 1
        deps.record_phase_llm_call("final_texture_proxy_repair")
        topk_gateway = gateway
        if use_budget_override_gateway:
            topk_gateway = LLMGateway(LLMConfig(
                api_key=effective_key,
                model=generator_model,
                base_url=base_url,
                timeout=int(os.environ.get("DRAFTPROOF_AI_SEARCH_TIMEOUT", "120")),
                max_retries=int(os.environ.get("DRAFTPROOF_AI_SEARCH_RETRIES", "1")),
                max_tokens=int(os.environ.get(
                    "DRAFTPROOF_FINAL_TOPK_TEXTURE_MAX_TOKENS",
                    os.environ.get("DRAFTPROOF_TOPK_TEXTURE_MAX_TOKENS", "4800"),
                )),
                temperature=float(os.environ.get(
                    "DRAFTPROOF_FINAL_TOPK_TEXTURE_TEMPERATURE",
                    os.environ.get("DRAFTPROOF_TOPK_TEXTURE_TEMPERATURE", "0.78"),
                )),
            ))
            summary["ran_after_search_budget"] = True
            summary["search_budget_reason"] = (
                adaptive_stop_reason
                or "final_topk_texture_llm_call_reserve"
            )
        response = topk_gateway.chat(
            prompt,
            system=(
                "You are DraftProof's final top-k texture repair engine. "
                "Patch only predictable phrasing in the already selected candidate. "
                "Do not add facts. Return only tagged full-document candidates."
            ),
            **deps.phase_chat_sampling_kwargs(
                "DRAFTPROOF_FINAL_TOPK_TEXTURE",
                temperature_env="DRAFTPROOF_FINAL_TOPK_TEXTURE_TEMPERATURE",
                temperature_default=float(os.environ.get(
                    "DRAFTPROOF_TOPK_TEXTURE_TEMPERATURE",
                    "0.78",
                )),
                max_tokens_env="DRAFTPROOF_FINAL_TOPK_TEXTURE_MAX_TOKENS",
                max_tokens_default=int(os.environ.get(
                    "DRAFTPROOF_TOPK_TEXTURE_MAX_TOKENS",
                    "4800",
                )),
                fallback_prefix="DRAFTPROOF_TOPK_TEXTURE",
            ),
        )
        outputs = deps.extract_paragraph_component_candidates(response.content, candidate_limit)
    except Exception as exc:
        search_summary["candidates"].append({
            "strategy": "final_topk_texture_repair_batch",
            "passed_local_checks": False,
            "reason": f"llm_error {exc}",
            "topk_texture_repair": True,
            "final_topk_texture_repair": True,
            "base_strategy": deps.get_best_strategy(),
        })
        summary["reason"] = f"llm_error {exc}"
        return adaptive_stop_reason

    accepted_before = len([
        candidate
        for candidate in search_summary.get("candidates", [])
        if isinstance(candidate, dict)
        and candidate.get("final_topk_texture_repair")
        and (candidate.get("selection_status") or {}).get("selectable")
    ])
    base_strategy = deps.get_best_strategy()
    base_text = deps.get_best_text()
    for candidate_number, raw_candidate in enumerate(outputs, start=1):
        candidate = deps.clean_full_document_candidate(raw_candidate, base_text)
        strategy = f"final_topk_texture_repair_c{candidate_number}"
        if not candidate:
            search_summary["candidates"].append({
                "strategy": strategy,
                "passed_local_checks": False,
                "reason": "empty_or_unchanged_candidate",
                "topk_texture_repair": True,
                "final_topk_texture_repair": True,
                "base_strategy": base_strategy,
            })
            continue
        deps.evaluate_ai_search_candidate(
            strategy,
            candidate,
            deterministic=False,
            extra={
                "topk_texture_repair": True,
                "final_topk_texture_repair": True,
                "final_topk_texture_repair_budget_override": bool(use_budget_override_gateway),
                "base_strategy": base_strategy,
            },
        )
    accepted_after = len([
        candidate
        for candidate in search_summary.get("candidates", [])
        if isinstance(candidate, dict)
        and candidate.get("final_topk_texture_repair")
        and (candidate.get("selection_status") or {}).get("selectable")
    ])
    summary["accepted_count"] = max(0, accepted_after - accepted_before)
    summary["selected_strategy_after"] = deps.get_best_strategy()
    return adaptive_stop_reason
