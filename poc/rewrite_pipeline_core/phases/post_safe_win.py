"""Post-safe-win Human target push phase orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os

from llm.gateway import LLMConfig, LLMGateway


@dataclass(frozen=True)
class PostSafeWinTargetPushDeps:
    env_flag: Callable[[str, bool], bool]
    strict_ai_safe_band_status: Callable[[dict | None], dict]
    post_safe_target_push_allows_deterministic_after_budget: Callable[[str], bool]
    post_safe_target_push_scan_reserve: Callable[[str], int]
    verified_candidate_scans_used: Callable[[], int]
    extend_candidate_scan_budget: Callable[[dict, int, int], None]
    best_ai_search_selectable: Callable[[], bool]
    contribution_scores: Callable[[dict | None], dict]
    float_env: Callable[[str, float], float]
    adaptive_budget_default: Callable[[str, int, int], int]
    post_safe_win_target_push_candidates: Callable[..., list[tuple[str, str, dict]]]
    report_progress: Callable[[int, str], None]
    evaluate_ai_search_candidate: Callable[..., None]
    budget_gateway: Callable[[LLMGateway, str], LLMGateway]
    paragraph_component_targets: Callable[..., list[dict]]
    safe_index: Callable[[Any, int], int]
    search_budget_exhausted: Callable[[str], bool]
    human_signal_amplification_prompt: Callable[..., str]
    phase_chat_sampling_kwargs: Callable[..., dict]
    extract_paragraph_component_candidates: Callable[[str, int], list[str]]
    clean_paragraph_component_candidate: Callable[..., tuple[str, str]]
    paragraph_anchor_lock: Callable[[dict], Any]
    splice_paragraph: Callable[[str, int, str], str]
    get_best_text: Callable[[], str]
    get_best_report: Callable[[], dict | None]
    get_best_ai: Callable[[], float | int | None]
    get_best_strategy: Callable[[], str]
    get_best_selection_status: Callable[[], dict]


def run_post_safe_win_target_push(
    trigger_phase: str,
    *,
    search_summary: dict,
    search_budget: dict,
    adaptive_stop_reason: str,
    topk_safe_band_priority: bool,
    density_or_generic_priority: bool,
    effective_key: str | None,
    generator_model: str,
    base_url: str,
    original_report_dict: dict | None,
    confirmed_author_anchor_brief: str,
    deps: PostSafeWinTargetPushDeps,
) -> str:
    if (
        deps.env_flag("DRAFTPROOF_STRICT_AI_PHASE_BUDGET_ONLY", True)
        and (topk_safe_band_priority or density_or_generic_priority)
        and not deps.strict_ai_safe_band_status(deps.get_best_report()).get("achieved")
    ):
        search_summary["post_safe_win_target_push"] = {
            "enabled": bool(deps.env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH", True)),
            "skipped": True,
            "reason": "strict_ai_phase_budget_only",
            "trigger_phase": trigger_phase,
            "selected_strategy": deps.get_best_strategy(),
            "strict_ai_safe_band": deps.strict_ai_safe_band_status(deps.get_best_report()),
        }
        return adaptive_stop_reason
    resumed_after_llm_budget = False
    resumed_after_scan_budget = False
    scan_reserve_added = 0
    if str(adaptive_stop_reason).startswith("budget_exhausted"):
        if deps.post_safe_target_push_allows_deterministic_after_budget(adaptive_stop_reason):
            budget_reason = str(adaptive_stop_reason or "")
            resumed_after_llm_budget = budget_reason == "budget_exhausted_llm_calls"
            resumed_after_scan_budget = budget_reason == "budget_exhausted_candidate_scans"
            if resumed_after_scan_budget:
                scan_reserve_added = deps.post_safe_target_push_scan_reserve(budget_reason)
                if scan_reserve_added > 0:
                    previous_max = int(search_budget.get("max_candidate_scans") or 0)
                    current_scans = deps.verified_candidate_scans_used()
                    deps.extend_candidate_scan_budget(search_budget, current_scans, scan_reserve_added)
                    search_summary["post_safe_target_push_scan_reserve"] = {
                        "enabled": True,
                        "trigger_phase": trigger_phase,
                        "previous_max_candidate_scans": previous_max,
                        "candidate_scans_before_reserve": current_scans,
                        "reserve_added": scan_reserve_added,
                        "max_candidate_scans": search_budget["max_candidate_scans"],
                    }
            adaptive_stop_reason = ""
        else:
            search_summary["post_safe_win_target_push"] = {
                "enabled": False,
                "reason": adaptive_stop_reason,
                "trigger_phase": trigger_phase,
            }
            return adaptive_stop_reason
    if not deps.env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH", True):
        return adaptive_stop_reason
    if not deps.best_ai_search_selectable() or not isinstance(deps.get_best_report(), dict):
        return adaptive_stop_reason
    current_human = deps.contribution_scores(deps.get_best_report()).get("human")
    target_human = deps.float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    if not isinstance(current_human, (int, float)) or float(current_human) >= target_human:
        search_summary["post_safe_win_target_push"] = {
            "enabled": True,
            "trigger_phase": trigger_phase,
            "skipped": True,
            "reason": "human_target_already_reached",
            "current_human": current_human,
            "target_human": target_human,
        }
        return adaptive_stop_reason
    existing_target_push = search_summary.get("post_safe_win_target_push")
    if isinstance(existing_target_push, dict) and existing_target_push.get("accepted"):
        rerun_allowed = bool(
            trigger_phase == "pre_selection"
            and deps.env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_RERUN_AFTER_BETTER_BASE", True)
            and deps.get_best_strategy() != existing_target_push.get("accepted_strategy")
        )
        if not rerun_allowed:
            return adaptive_stop_reason
        prior_runs = search_summary.setdefault("post_safe_win_target_push_previous_runs", [])
        prior_runs.append(existing_target_push)
    try:
        push_limit = max(
            0,
            int(deps.float_env(
                "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_CANDIDATES",
                float(deps.adaptive_budget_default(deps.get_best_text(), 4, 8)),
            )),
        )
    except (TypeError, ValueError):
        push_limit = 4
    if push_limit <= 0:
        search_summary["post_safe_win_target_push"] = {
            "enabled": False,
            "trigger_phase": trigger_phase,
            "reason": "candidate_limit_zero",
            "current_human": current_human,
            "target_human": target_human,
        }
        return adaptive_stop_reason
    initial_strategy = deps.get_best_strategy()
    initial_human = float(current_human)
    initial_ai = deps.get_best_ai()
    try:
        max_rounds = max(
            1,
            int(deps.float_env(
                "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_ROUNDS",
                float(deps.adaptive_budget_default(deps.get_best_text(), 2, 4)),
            )),
        )
    except (TypeError, ValueError):
        max_rounds = 2
    summary = {
        "enabled": True,
        "trigger_phase": trigger_phase,
        "base_strategy": initial_strategy,
        "base_ai": initial_ai,
        "base_human": initial_human,
        "target_human": target_human,
        "resumed_after_llm_budget": resumed_after_llm_budget,
        "candidate_limit": push_limit,
        "max_rounds": max_rounds,
        "candidate_count": 0,
        "accepted": False,
        "accepted_strategy": None,
        "rounds": [],
        "resumed_after_scan_budget": resumed_after_scan_budget,
        "scan_reserve_added": scan_reserve_added,
    }
    search_summary["post_safe_win_target_push"] = summary
    min_extra_gain = deps.float_env("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_MIN_EXTRA_HUMAN_GAIN", 1.0)
    total_scanned = 0
    for round_index in range(1, max_rounds + 1):
        if not isinstance(deps.get_best_report(), dict):
            break
        round_human = deps.contribution_scores(deps.get_best_report()).get("human")
        if not isinstance(round_human, (int, float)):
            break
        if float(round_human) >= target_human:
            summary["reason"] = "human_target_reached"
            break
        round_base_strategy = deps.get_best_strategy()
        round_base_human = float(round_human)
        round_base_ai = deps.get_best_ai()
        candidates = deps.post_safe_win_target_push_candidates(
            deps.get_best_text(),
            deps.get_best_report(),
            limit=push_limit,
        )
        summary["candidate_count"] += len(candidates)
        round_summary = {
            "round": round_index,
            "base_strategy": round_base_strategy,
            "base_ai": round_base_ai,
            "base_human": round_base_human,
            "candidate_count": len(candidates),
            "accepted": False,
        }
        summary["rounds"].append(round_summary)
        if not candidates:
            round_summary["reason"] = "no_target_push_candidates"
            if not summary.get("accepted"):
                summary["reason"] = "no_target_push_candidates"
            break
        for push_index, (strategy, candidate, meta) in enumerate(candidates, start=1):
            total_scanned += 1
            deps.report_progress(
                min(89, 80 + total_scanned),
                (
                    "Trying post-safe-win target push "
                    f"{round_index}.{push_index}/{len(candidates)}"
                ),
            )
            deps.evaluate_ai_search_candidate(
                strategy,
                candidate,
                deterministic=True,
                extra={
                    **meta,
                    "post_safe_win_target_push": True,
                    "post_safe_win_target_push_round": round_index,
                    "base_strategy": round_base_strategy,
                    "trigger_phase": trigger_phase,
                },
            )
        candidate_human = (
            deps.contribution_scores(deps.get_best_report()).get("human")
            if isinstance(deps.get_best_report(), dict) else None
        )
        round_accepted = (
            deps.get_best_strategy() != round_base_strategy
            and deps.get_best_selection_status().get("selectable")
            and isinstance(candidate_human, (int, float))
            and float(candidate_human) >= round_base_human + min_extra_gain
        )
        if not round_accepted:
            round_summary["reason"] = "no_extra_safe_human_gain"
            if not summary.get("accepted"):
                summary["reason"] = "no_extra_safe_human_gain"
            break
        round_summary.update({
            "accepted": True,
            "accepted_strategy": deps.get_best_strategy(),
            "accepted_human": candidate_human,
            "accepted_ai": deps.get_best_ai(),
            "scanned": len(candidates),
        })
        summary.update({
            "accepted": True,
            "accepted_strategy": deps.get_best_strategy(),
            "accepted_human": candidate_human,
            "accepted_ai": deps.get_best_ai(),
            "scanned": total_scanned,
            "accepted_round": round_index,
        })
        adaptive_stop_reason = "adaptive_stop_after_post_safe_win_target_push"
        search_summary["adaptive_stop"] = {
            "phase": "post_safe_win_target_push",
            "reason": adaptive_stop_reason,
            "candidate_count_scanned": deps.verified_candidate_scans_used(),
            "selected_strategy": deps.get_best_strategy(),
            "selection_status": deps.get_best_selection_status(),
        }
    deterministic_plateau_after_accept = False
    if summary.get("accepted"):
        latest_human = (
            deps.contribution_scores(deps.get_best_report()).get("human")
            if isinstance(deps.get_best_report(), dict) else None
        )
        deterministic_plateau_after_accept = bool(
            deps.env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_LLM_AFTER_ACCEPT", True)
            and isinstance(latest_human, (int, float))
            and float(latest_human) < target_human
        )
        if not deterministic_plateau_after_accept:
            if not summary.get("reason"):
                summary["reason"] = "accepted_iterative_target_push"
            return adaptive_stop_reason
        summary["deterministic_plateau_after_accept"] = True
        summary["deterministic_plateau_human"] = latest_human
        summary["reason"] = "deterministic_target_push_plateau_below_target"
    if resumed_after_llm_budget:
        summary["llm_target_push"] = {
            "enabled": False,
            "reason": "llm_budget_exhausted_before_target_push",
        }
        if not summary.get("accepted"):
            adaptive_stop_reason = "budget_exhausted_llm_calls"
        return adaptive_stop_reason
    if (
        (not summary.get("accepted") or deterministic_plateau_after_accept)
        and deps.env_flag("DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_LLM", True)
        and effective_key
        and not resumed_after_llm_budget
    ):
        try:
            llm_candidate_limit = max(
                1,
                int(deps.float_env(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_LLM_CANDIDATES",
                    float(deps.adaptive_budget_default(deps.get_best_text(), 3, 4)),
                )),
            )
            llm_target_limit = max(
                1,
                int(deps.float_env(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_LLM_TARGETS",
                    float(deps.adaptive_budget_default(deps.get_best_text(), 4, 6)),
                )),
            )
            llm_round_limit = max(
                1,
                int(deps.float_env(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_LLM_ROUNDS",
                    float(deps.adaptive_budget_default(deps.get_best_text(), 1, 2)),
                )),
            )
            llm_call_limit = max(
                1,
                int(deps.float_env(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_LLM_MAX_CALLS",
                    float(deps.adaptive_budget_default(deps.get_best_text(), 4, 6)),
                )),
            )
        except (TypeError, ValueError):
            llm_candidate_limit = 1
            llm_target_limit = 1
            llm_round_limit = 1
            llm_call_limit = 1
        summary["llm_target_push"] = {
            "enabled": True,
            "target_limit_per_round": llm_target_limit,
            "candidate_limit_per_target": llm_candidate_limit,
            "round_limit": llm_round_limit,
            "max_calls": llm_call_limit,
            "calls_used": 0,
            "target_count": 0,
            "accepted_count": 0,
            "accepted": False,
            "rounds": [],
        }
        try:
            push_gateway = LLMGateway(LLMConfig(
                api_key=effective_key,
                model=generator_model,
                base_url=base_url,
                timeout=int(os.environ.get("DRAFTPROOF_AI_SEARCH_TIMEOUT", "120")),
                max_retries=int(os.environ.get("DRAFTPROOF_AI_SEARCH_RETRIES", "1")),
                max_tokens=int(os.environ.get(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_MAX_TOKENS",
                    os.environ.get("DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_MAX_TOKENS", "2600"),
                )),
                temperature=float(os.environ.get(
                    "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_TEMPERATURE",
                    os.environ.get("DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TEMPERATURE", "0.45"),
                )),
            ))
            push_gateway = deps.budget_gateway(push_gateway, "post_safe_win_target_push_llm")
        except Exception as exc:
            summary["llm_target_push"]["reason"] = f"gateway_error {exc}"
            push_gateway = None
        if push_gateway:
            llm_base_text = deps.get_best_text()
            llm_base_strategy = deps.get_best_strategy()
            llm_base_ai = deps.get_best_ai()
            llm_base_human = (
                deps.contribution_scores(deps.get_best_report()).get("human")
                if isinstance(deps.get_best_report(), dict) else initial_human
            )
            llm_attempted_indexes: set[int] = set()
            llm_calls_used = 0
            for llm_round in range(1, llm_round_limit + 1):
                if llm_calls_used >= llm_call_limit:
                    summary["llm_target_push"]["reason"] = "llm_call_limit_reached"
                    break
                if isinstance(llm_base_human, (int, float)) and float(llm_base_human) >= target_human:
                    summary["llm_target_push"]["reason"] = "human_target_reached"
                    break
                llm_target_pool = deps.paragraph_component_targets(
                    llm_base_text,
                    deps.get_best_report() if isinstance(deps.get_best_report(), dict) else original_report_dict,
                    limit=max(llm_target_limit * 4, llm_target_limit + len(llm_attempted_indexes)),
                )
                llm_targets = [
                    target for target in llm_target_pool
                    if deps.safe_index(target.get("index"), -1) not in llm_attempted_indexes
                ][: max(0, min(llm_target_limit, llm_call_limit - llm_calls_used))]
                llm_round_summary = {
                    "round": llm_round,
                    "base_strategy": llm_base_strategy,
                    "base_ai": llm_base_ai,
                    "base_human": llm_base_human,
                    "target_count": len(llm_targets),
                    "targets": [
                        {
                            "paragraph_index": target.get("index"),
                            "role": target.get("role"),
                            "score": target.get("score"),
                        }
                        for target in llm_targets
                    ],
                    "accepted": False,
                }
                summary["llm_target_push"]["rounds"].append(llm_round_summary)
                summary["llm_target_push"]["target_count"] += len(llm_targets)
                if not llm_targets:
                    llm_round_summary["reason"] = "no_unattempted_llm_targets"
                    if not summary["llm_target_push"].get("reason"):
                        summary["llm_target_push"]["reason"] = "no_unattempted_llm_targets"
                    break
                round_start_strategy = deps.get_best_strategy()
                for target_number, target in enumerate(llm_targets, start=1):
                    if llm_calls_used >= llm_call_limit:
                        summary["llm_target_push"]["reason"] = "llm_call_limit_reached"
                        break
                    target_index = deps.safe_index(target.get("index"), -1)
                    if target_index >= 0:
                        llm_attempted_indexes.add(target_index)
                    deps.report_progress(
                        min(92, 84 + target_number),
                        (
                            "Trying post-safe-win LLM target push "
                            f"{llm_round}.{target_number}/{len(llm_targets)}"
                        ),
                    )
                    try:
                        if deps.search_budget_exhausted("post_safe_win_target_push_llm"):
                            raise RuntimeError("budget_exhausted_llm_calls")
                        prompt = deps.human_signal_amplification_prompt(
                            target,
                            original_report_dict,
                            target_number,
                            candidate_count=llm_candidate_limit,
                            confirmed_author_anchors=confirmed_author_anchor_brief,
                        )
                        search_summary["llm_calls"] += 1
                        response = push_gateway.chat(
                            prompt,
                            system=(
                                "You are DraftProof's post-safe-win target push engine. "
                                "Increase Human Contribution only if Authorship, Transformation, "
                                "review burden, severity, anchors, and meaning remain safe. "
                                "Return only tagged replacement paragraphs."
                            ),
                            **deps.phase_chat_sampling_kwargs(
                                "DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH",
                                temperature_env="DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_TEMPERATURE",
                                temperature_default=float(os.environ.get(
                                    "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_TEMPERATURE",
                                    "0.45",
                                )),
                                max_tokens_env="DRAFTPROOF_POST_SAFE_WIN_TARGET_PUSH_MAX_TOKENS",
                                max_tokens_default=int(os.environ.get(
                                    "DRAFTPROOF_HUMAN_SIGNAL_AMPLIFICATION_MAX_TOKENS",
                                    "2600",
                                )),
                            ),
                        )
                        llm_calls_used += 1
                        summary["llm_target_push"]["calls_used"] = llm_calls_used
                        outputs = deps.extract_paragraph_component_candidates(
                            response.content,
                            llm_candidate_limit,
                        )
                    except Exception as exc:
                        search_summary["candidates"].append({
                            "strategy": (
                                f"post_safe_target_push_llm_p{int(target.get('index', 0)) + 1}"
                                "_batch"
                            ),
                            "passed_local_checks": False,
                            "reason": f"llm_error {exc}",
                            "post_safe_win_target_push": True,
                            "post_safe_win_target_push_llm": True,
                            "paragraph_index": target.get("index"),
                            "paragraph_role": target.get("role"),
                        })
                        continue
                    for candidate_number, raw_paragraph_candidate in enumerate(outputs, start=1):
                        strategy = (
                            f"post_safe_target_push_llm_p{int(target.get('index', 0)) + 1}"
                            f"_c{candidate_number}"
                        )
                        paragraph_candidate, paragraph_reject = deps.clean_paragraph_component_candidate(
                            raw_paragraph_candidate,
                            target.get("paragraph") or "",
                            deps.paragraph_anchor_lock(target),
                        )
                        if paragraph_reject:
                            search_summary["candidates"].append({
                                "strategy": strategy,
                                "passed_local_checks": False,
                                "reason": paragraph_reject,
                                "post_safe_win_target_push": True,
                                "post_safe_win_target_push_llm": True,
                                "paragraph_index": target.get("index"),
                                "paragraph_role": target.get("role"),
                            })
                            continue
                        patched_candidate = deps.splice_paragraph(
                            llm_base_text,
                            int(target.get("index", 0)),
                            paragraph_candidate,
                        )
                        previous_best_strategy = deps.get_best_strategy()
                        deps.evaluate_ai_search_candidate(
                            strategy,
                            patched_candidate,
                            deterministic=False,
                            extra={
                                "post_safe_win_target_push": True,
                                "post_safe_win_target_push_llm": True,
                                "human_signal_amplification": True,
                                "paragraph_component": True,
                                "paragraph_index": target.get("index"),
                                "paragraph_role": target.get("role"),
                                "paragraph_driver_score": target.get("score"),
                                "paragraph_drivers": target.get("drivers"),
                                "base_strategy": llm_base_strategy,
                                "trigger_phase": trigger_phase,
                            },
                        )
                        candidate_human = (
                            deps.contribution_scores(deps.get_best_report()).get("human")
                            if isinstance(deps.get_best_report(), dict) else None
                        )
                        llm_extra_human_gain = bool(
                            isinstance(candidate_human, (int, float))
                            and isinstance(llm_base_human, (int, float))
                            and float(candidate_human) >= float(llm_base_human) + min_extra_gain
                        )
                        llm_same_human_quality_gain = bool(
                            isinstance(candidate_human, (int, float))
                            and isinstance(llm_base_human, (int, float))
                            and float(candidate_human) >= float(llm_base_human)
                            and isinstance(deps.get_best_ai(), (int, float))
                            and isinstance(llm_base_ai, (int, float))
                            and float(deps.get_best_ai()) < float(llm_base_ai) - 0.05
                        )
                        if (
                            deps.get_best_strategy() == strategy
                            and previous_best_strategy != strategy
                            and deps.get_best_selection_status().get("selectable")
                            and (llm_extra_human_gain or llm_same_human_quality_gain)
                        ):
                            summary.update({
                                "accepted": True,
                                "accepted_strategy": deps.get_best_strategy(),
                                "accepted_human": candidate_human,
                                "accepted_ai": deps.get_best_ai(),
                                "reason": (
                                    "accepted_llm_target_push"
                                    if llm_extra_human_gain
                                    else "accepted_llm_target_push_quality_gain"
                                ),
                            })
                            summary["llm_target_push"].update({
                                "accepted": True,
                                "accepted_strategy": deps.get_best_strategy(),
                                "accepted_human": candidate_human,
                            })
                            summary["llm_target_push"]["accepted_count"] += 1
                            llm_round_summary.update({
                                "accepted": True,
                                "accepted_strategy": deps.get_best_strategy(),
                                "accepted_human": candidate_human,
                                "accepted_ai": deps.get_best_ai(),
                            })
                            llm_base_text = deps.get_best_text()
                            llm_base_strategy = deps.get_best_strategy()
                            llm_base_ai = deps.get_best_ai()
                            llm_base_human = candidate_human
                            adaptive_stop_reason = "adaptive_stop_after_post_safe_win_target_push"
                            search_summary["adaptive_stop"] = {
                                "phase": "post_safe_win_target_push",
                                "reason": adaptive_stop_reason,
                                "candidate_count_scanned": deps.verified_candidate_scans_used(),
                                "selected_strategy": deps.get_best_strategy(),
                                "selection_status": deps.get_best_selection_status(),
                            }
                if deps.get_best_strategy() == round_start_strategy:
                    llm_round_summary["reason"] = "no_safe_gain_in_round"
                elif isinstance(llm_base_human, (int, float)) and float(llm_base_human) >= target_human:
                    summary["llm_target_push"]["reason"] = "human_target_reached"
                    break
        elif "llm_target_push" not in summary:
            summary["llm_target_push"] = {
                "enabled": False,
                "reason": "no_llm_key",
            }
    return adaptive_stop_reason
