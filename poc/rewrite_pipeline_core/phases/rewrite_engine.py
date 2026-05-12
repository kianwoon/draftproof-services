"""Rewrite-engine phase orchestration for the DraftProof rewrite pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os
import time


@dataclass(frozen=True)
class RewriteEnginePhaseDeps:
    sanitize_text: Callable[[str], str]
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    run_full_scan_report_dict: Callable[[str], dict]
    ensure_ai_mitigation_contract: Callable[[dict | None], dict]
    ai_mitigation_requires_user_input: Callable[[dict | None], bool]
    radar_goal_controller_status: Callable[[dict | None], dict]
    radar_blocker_option_matrix: Callable[[dict | None], dict]
    rewrite_config_factory: Callable[..., Any]
    run_rewrite: Callable[..., Any]
    build_marked_mitigation_rewrite: Callable[[str, dict], str]
    manual_summary_from_ai_mitigation: Callable[[dict], list[dict]]


def run_rewrite_engine_phase(
    *,
    text: str,
    ctx: Any,
    api_key: str | None,
    generator_model: str | None,
    base_url: str | None,
    max_passes: int,
    target_top10: float,
    max_detect_loops: int,
    output_dir: str | None,
    ai_only: bool,
    llm_roles: dict,
    report_progress: Callable[[int, str], None],
    deps: RewriteEnginePhaseDeps,
) -> dict:
    """Run the legacy rewrite engine prepass and return its phase state."""
    # Sanitize input text before rewrite (fix mojibake from PDF/docx extraction).
    text = deps.sanitize_text(text)

    pre_rewrite_stage_timings: list[dict] = []
    baseline_report_dict = ctx.raw_json
    if deps.env_flag("DRAFTPROOF_FRESH_ORIGINAL_BASELINE", True):
        report_progress(40, "Running baseline scan for rewrite controller")
        scan_t0 = time.time()
        baseline_report_dict = deps.run_full_scan_report_dict(text)
        pre_rewrite_stage_timings.append({
            "stage": "fresh_original_scan",
            "seconds": round(time.time() - scan_t0, 3),
        })

    pre_rewrite_badge = (baseline_report_dict or {}).get("ai_risk_badge") or {}
    pre_rewrite_ai = pre_rewrite_badge.get("ai_likelihood_score")
    ai_mitigation_contract = deps.ensure_ai_mitigation_contract(baseline_report_dict)
    ai_mitigation_needs_author = deps.ai_mitigation_requires_user_input(ai_mitigation_contract)
    allow_auto_with_author_gaps = deps.env_flag("DRAFTPROOF_ALLOW_AUTO_WITH_AUTHOR_GAPS", True)
    radar_goal_controller = deps.radar_goal_controller_status(baseline_report_dict)
    radar_option_matrix = radar_goal_controller.get("option_matrix") or deps.radar_blocker_option_matrix(baseline_report_dict)
    ai_search_first = (
        (
            os.environ.get("DRAFTPROOF_AI_SEARCH_FIRST", "1") != "0"
            and isinstance(pre_rewrite_ai, (int, float))
            and pre_rewrite_ai >= deps.float_env("DRAFTPROOF_AI_FIRST_REQUIRED_MIN_AI", 50.0)
        )
        or bool(radar_goal_controller.get("execute_before_local_rewrite"))
    ) and (not ai_mitigation_needs_author or allow_auto_with_author_gaps)
    if radar_goal_controller.get("execute_before_local_rewrite"):
        report_progress(40, "Radar goal ladder selected before local sentence repair")
        print(
            "  Radar goal-first mode: "
            f"Human gap {radar_goal_controller.get('human_gap_to_target')} "
            f"with direct blockers {radar_goal_controller.get('direct_gain_blockers')}"
        )
    rewrite_config = None
    if ai_search_first:
        rewrite_config = deps.rewrite_config_factory(
            max_llm_calls=0,
            max_density_passes=0,
            max_rewrite_seconds=30,
        )
    elif ai_mitigation_needs_author and not allow_auto_with_author_gaps:
        rewrite_config = deps.rewrite_config_factory(
            max_llm_calls=0,
            max_density_passes=0,
            max_rewrite_seconds=30,
        )

    t0 = time.time()
    report_progress(41, "Preparing rewrite plan from scan findings")
    result = deps.run_rewrite(
        content=text,
        detect_results=ctx.detect_results,
        api_key=api_key,
        model=generator_model,
        base_url=base_url,
        max_passes=max_passes,
        target_top10=target_top10,
        max_detect_loops=max_detect_loops,
        output_dir=output_dir,
        rewrite_context=ctx,
        ai_only=ai_only,
        config=rewrite_config,
        progress_callback=report_progress,
    )
    result.summary["llm_model_roles"] = llm_roles
    result.summary["ai_mitigation"] = ai_mitigation_contract
    result.summary["radar_blocker_option_matrix"] = radar_option_matrix
    result.summary["radar_goal_controller"] = {
        key: value
        for key, value in radar_goal_controller.items()
        if key != "option_matrix"
    }
    if ai_mitigation_needs_author:
        result.summary["ai_mitigation_blocked_auto_rewrite"] = not allow_auto_with_author_gaps
        if not allow_auto_with_author_gaps:
            result.summary["outcome"] = "suggestion_only"
        marked_rewrite = deps.build_marked_mitigation_rewrite(text, ai_mitigation_contract)
        if marked_rewrite:
            result.summary["marked_mitigation_rewrite"] = marked_rewrite
        suggestions = result.summary.setdefault("manual_suggestions", [])
        existing_keys = {
            (
                item.get("component"),
                item.get("suggested_sentence"),
                item.get("user_input_needed"),
            )
            for item in suggestions
            if isinstance(item, dict)
        }
        for suggestion in deps.manual_summary_from_ai_mitigation(ai_mitigation_contract):
            key = (
                suggestion.get("component"),
                suggestion.get("suggested_sentence"),
                suggestion.get("user_input_needed"),
            )
            if key not in existing_keys:
                suggestions.append(suggestion)
                existing_keys.add(key)
    if ai_search_first:
        result.summary["rewrite_engine_mode"] = "ai_search_first_skip_rewrite_prepass"
        result.summary.setdefault("saved_contract_notes", []).append(
            "Skipped costly density/sentence LLM prepass because AI mitigation search is the first objective."
        )
    elif ai_mitigation_needs_author and not allow_auto_with_author_gaps:
        result.summary["rewrite_engine_mode"] = "guided_authenticity_requires_author_input"
        result.summary.setdefault("saved_contract_notes", []).append(
            "Skipped automatic sentence/density rewrite because AI-Mitigation requires author-supplied grounding."
        )
    engine_elapsed = time.time() - t0
    stage_timings = pre_rewrite_stage_timings + [
        {"stage": "rewrite_engine", "seconds": round(engine_elapsed, 3)}
    ]
    report_progress(74, "Building rewrite comparison")

    return {
        "text": text,
        "result": result,
        "baseline_report_dict": baseline_report_dict,
        "ai_mitigation_contract": ai_mitigation_contract,
        "ai_mitigation_needs_author": ai_mitigation_needs_author,
        "allow_auto_with_author_gaps": allow_auto_with_author_gaps,
        "radar_goal_controller": radar_goal_controller,
        "radar_option_matrix": radar_option_matrix,
        "ai_search_first": ai_search_first,
        "rewrite_config": rewrite_config,
        "rewrite_engine_started_at": t0,
        "rewrite_engine_elapsed": engine_elapsed,
        "stage_timings": stage_timings,
    }
