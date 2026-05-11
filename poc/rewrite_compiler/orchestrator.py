"""Bounded deterministic rewrite compiler orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from . import evaluator, operators, planner, selector, signals, validator


@dataclass
class CompilerConfig:
    mode: str = "compiler_strict"
    max_rounds: int = 1
    max_scans: int = 4
    candidate_pool_limit: int = 14
    shortlist_limit: int = 4
    max_llm_calls: int = 0


@dataclass
class CompilerDependencies:
    split_sentences: Callable[[str], list[str]]
    text_word_count: Callable[[str], int]
    geometry_risk_map: Callable[..., dict]
    is_canonical_fact_sentence: Callable[[str], bool]
    splice_sentence: Callable[[str, int, str], str]
    repair_aggression_score: Callable[[str, str], dict]
    locality_score: Callable[[str, str], dict]
    detect_protected_spans: Callable[[str], Any]
    protected_loss_reason: Callable[[str, str, Any], str]
    concept_origin_reject_reason: Callable[[str, str], str]
    drift_checker: Callable[..., Any]
    scan_func: Callable[[str], dict]
    turnitin_profile: Callable[[dict | None], dict]
    turnitin_gate_status: Callable[..., dict]
    strict_safe_status: Callable[[dict | None], dict]
    contribution_scores: Callable[[dict | None], dict]
    integrity_scores: Callable[[dict | None], dict]
    badge_ai: Callable[[dict | None], float | None]
    finding_total: Callable[[dict | None], int]
    review_burden: Callable[[dict | None], int]
    weighted_severity: Callable[[dict | None], int]
    critical_high_count: Callable[[dict | None], int]


def _mode_llm_budget(mode: str) -> int:
    normalized = str(mode or "compiler_strict").strip().lower()
    if normalized == "compiler_balanced":
        return 1
    if normalized == "compiler_experimental":
        return 3
    return 0


def _pre_rank(candidate: dict, validation: dict, quality: dict) -> tuple:
    return (
        1 if validation.get("passed") else 0,
        1 if quality.get("passed") else 0,
        -float((validation.get("locality") or {}).get("changed_sentence_ratio") or 0.0),
        -float((validation.get("repair_aggression") or {}).get("score") or 0.0),
        float(quality.get("term_preservation_ratio") or 0.0),
        -len(validation.get("reject_reasons") or []),
        -len(quality.get("reject_reasons") or []),
    )


def run_rewrite_compiler(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    deps: CompilerDependencies,
    *,
    config: CompilerConfig | None = None,
) -> dict:
    config = config or CompilerConfig()
    mode = str(config.mode or "compiler_strict").strip().lower()
    if mode not in {"compiler_strict", "compiler_balanced", "compiler_experimental"}:
        mode = "compiler_strict"
    max_rounds = max(0, int(config.max_rounds or 0))
    max_scans = max(0, int(config.max_scans or 0))
    max_llm_calls = max(0, int(config.max_llm_calls if config.max_llm_calls is not None else _mode_llm_budget(mode)))
    if mode == "compiler_strict":
        max_llm_calls = 0
    current_text_value = str(current_text or "")
    current_report_value = current_report
    start_snapshot = signals.formula_snapshot(current_text_value, current_report_value, deps)
    summary = {
        "enabled": True,
        "version": "deterministic_rewrite_compiler_v1",
        "mode": mode,
        "selected": False,
        "selected_text": current_text_value,
        "selected_report": current_report_value,
        "selected_strategy": None,
        "selected_candidate": None,
        "operator_catalog": operators.OPERATOR_CATALOG,
        "llm_calls_used": 0,
        "llm_call_budget": max_llm_calls,
        "scans_used": 0,
        "max_scans": max_scans,
        "max_rounds": max_rounds,
        "rounds": [],
        "candidates": [],
        "score_before": start_snapshot.get("score"),
        "score_after": start_snapshot.get("score"),
        "strict_detector_safe": bool((start_snapshot.get("strict_safe_band") or {}).get("achieved")),
        "reason": "no_candidate_selected",
    }
    if not current_text_value.strip() or not isinstance(current_report_value, dict):
        summary["enabled"] = False
        summary["reason"] = "missing_current_text_or_report"
        return summary
    if max_rounds <= 0 or max_scans <= 0:
        summary["reason"] = "budget_zero"
        return summary

    cumulative_aggression = 0.0
    cumulative_locality = 0.0
    accepted_rounds = 0
    for round_id in range(1, max_rounds + 1):
        if int(summary["scans_used"] or 0) >= max_scans:
            summary["reason"] = "scan_budget_exhausted"
            break
        current_snapshot = signals.formula_snapshot(current_text_value, current_report_value, deps)
        if bool(current_snapshot.get("target_met")) and bool((current_snapshot.get("strict_safe_band") or {}).get("achieved")):
            summary["reason"] = "target_reached"
            break
        plan = planner.build_plan(
            current_text_value,
            current_report_value,
            deps,
            max_windows=8 if signals.split_sentences(current_text_value) and len(signals.split_sentences(current_text_value)) <= 40 else 10,
        )
        pool = operators.generate_candidates(
            current_text_value,
            plan,
            deps,
            limit=max(1, int(config.candidate_pool_limit or 1)),
        )
        round_summary = {
            "round_id": round_id,
            "score_before": current_snapshot.get("score"),
            "strict_detector_safe_before": bool((current_snapshot.get("strict_safe_band") or {}).get("achieved")),
            "plan": {
                "dominant_drivers": plan.get("dominant_drivers"),
                "canonical_fact_preserved_count": plan.get("canonical_fact_preserved_count"),
                "generic_blocks_targeted": plan.get("generic_blocks_targeted"),
                "template_blocks_targeted": plan.get("template_blocks_targeted"),
                "selected_windows": plan.get("selected_windows"),
                "block_risk_map": plan.get("block_risk_map"),
            },
            "candidate_count": len(pool),
            "shortlisted": 0,
            "scanned": 0,
            "selected": False,
        }
        if not pool:
            round_summary["reason"] = "no_operator_candidates"
            summary["rounds"].append(round_summary)
            summary["reason"] = "ceiling_reached_no_operator_candidates"
            break

        prechecked: list[dict] = []
        for candidate in pool:
            candidate_text = str(candidate.get("text") or "")
            validation = validator.validate_candidate(
                current_text_value,
                candidate_text,
                candidate,
                deps,
                cumulative_aggression=cumulative_aggression,
                cumulative_locality=cumulative_locality,
            )
            quality = evaluator.evaluate_quality(current_text_value, candidate_text, candidate, deps)
            row = {
                **{key: value for key, value in candidate.items() if key != "text"},
                "validation": validation,
                "quality": quality,
                "pre_rank": _pre_rank(candidate, validation, quality),
            }
            if not validation.get("passed") or not quality.get("passed"):
                row["reason"] = (
                    (validation.get("reject_reasons") or quality.get("reject_reasons") or ["precheck_failed"])[0]
                )
                summary["candidates"].append(row)
                continue
            prechecked.append({"candidate": candidate, "validation": validation, "quality": quality, "row": row})
        prechecked.sort(key=lambda item: item["row"]["pre_rank"], reverse=True)
        shortlist = prechecked[: max(1, min(int(config.shortlist_limit or 1), max_scans - int(summary["scans_used"] or 0)))]
        round_summary["shortlisted"] = len(shortlist)
        best_eval = None
        best_row = None
        best_text = current_text_value
        best_report = current_report_value
        for item in shortlist:
            candidate = item["candidate"]
            candidate_text = str(candidate.get("text") or "")
            row = item["row"]
            if int(summary["scans_used"] or 0) >= max_scans:
                row["reason"] = "scan_budget_exhausted"
                summary["candidates"].append(row)
                break
            scan_t0 = time.time()
            try:
                candidate_report = deps.scan_func(candidate_text)
            except Exception as exc:
                row["reason"] = f"candidate_scan_error {exc}"
                summary["candidates"].append(row)
                continue
            summary["scans_used"] += 1
            round_summary["scanned"] += 1
            scan_eval = selector.evaluate_scanned_candidate(
                current_text_value,
                current_report_value,
                candidate_text,
                candidate_report,
                original_report,
                deps,
                validation=item["validation"],
                quality=item["quality"],
            )
            row.update({
                "scan_seconds": round(time.time() - scan_t0, 3),
                "acceptance": scan_eval,
                "accepted": scan_eval.get("accepted"),
                "reason": scan_eval.get("reason"),
                "outcome_class": scan_eval.get("outcome_class"),
                "formula_score": scan_eval.get("formula_score_after"),
                "formula_score_drop": scan_eval.get("formula_score_drop"),
                "ai_score_drop": scan_eval.get("ai_score_drop"),
                "ai_authorship_drop": scan_eval.get("ai_authorship_drop"),
                "ai_transformation_drop": scan_eval.get("ai_transformation_drop"),
                "strict_ai_safe_band": scan_eval.get("strict_ai_safe_band"),
            })
            summary["candidates"].append(row)
            if selector.better_candidate(scan_eval, best_eval):
                best_eval = dict(scan_eval)
                best_text = candidate_text
                best_report = candidate_report
                best_eval["strategy"] = row.get("strategy")
                best_eval["operator"] = row.get("operator")
                best_row = row
        if not best_eval:
            round_summary["reason"] = "no_safe_compiler_candidate"
            summary["rounds"].append(round_summary)
            summary["reason"] = "ceiling_reached_no_safe_compiler_candidate"
            break
        current_text_value = best_text
        current_report_value = best_report
        accepted_rounds += 1
        validation = ((best_row or {}).get("validation") or {})
        cumulative_aggression = float(validation.get("cumulative_aggression_after") or cumulative_aggression)
        cumulative_locality = float(validation.get("cumulative_locality_after") or cumulative_locality)
        round_summary.update({
            "selected": True,
            "selected_strategy": best_eval.get("strategy"),
            "selected_operator": best_eval.get("operator"),
            "score_after": best_eval.get("formula_score_after"),
            "score_drop": best_eval.get("formula_score_drop"),
            "outcome_class": best_eval.get("outcome_class"),
            "strict_detector_safe_after": bool((best_eval.get("strict_ai_safe_band") or {}).get("achieved")),
            "selected_candidate": {
                key: best_eval.get(key)
                for key in (
                    "strategy",
                    "operator",
                    "outcome_class",
                    "formula_score_before",
                    "formula_score_after",
                    "formula_score_drop",
                    "ai_score_drop",
                    "ai_authorship_drop",
                    "ai_transformation_drop",
                    "strict_ai_safe_band",
                    "reason",
                )
            },
        })
        summary["rounds"].append(round_summary)

    final_snapshot = signals.formula_snapshot(current_text_value, current_report_value, deps)
    final_strict = final_snapshot.get("strict_safe_band") or {}
    score_drop = round(float(start_snapshot.get("score") or 0.0) - float(final_snapshot.get("score") or 0.0), 3)
    if bool(final_snapshot.get("target_met")) and bool(final_strict.get("achieved")):
        outcome = "ai_mitigated"
    elif score_drop > 0.001 and not bool(final_strict.get("achieved")):
        outcome = "unsafe_partial_improvement"
    elif score_drop > 0.001:
        outcome = "partially_ai_mitigated"
    elif accepted_rounds > 0:
        outcome = "cleanup_only"
    else:
        outcome = "ceiling_reached"
    summary.update({
        "selected": accepted_rounds > 0,
        "selected_text": current_text_value,
        "selected_report": current_report_value,
        "selected_strategy": (
            ((summary.get("rounds") or [])[-1] or {}).get("selected_strategy")
            if accepted_rounds > 0 else None
        ),
        "selected_candidate": (
            ((summary.get("rounds") or [])[-1] or {}).get("selected_candidate")
            if accepted_rounds > 0 else None
        ),
        "accepted_rounds": accepted_rounds,
        "score_after": final_snapshot.get("score"),
        "score_drop": score_drop,
        "target_met": bool(final_snapshot.get("target_met")),
        "strict_detector_safe": bool(final_strict.get("achieved")),
        "outcome_class": outcome,
        "remaining_detector_drivers": final_strict.get("remaining") or [],
        "cumulative_aggression": round(cumulative_aggression, 3),
        "cumulative_locality": round(cumulative_locality, 3),
    })
    if accepted_rounds > 0 and (
        str(summary.get("reason") or "").startswith("ceiling_reached")
        or summary.get("reason") == "no_candidate_selected"
    ):
        summary["reason"] = "accepted_compiler_progress"
    if accepted_rounds <= 0 and mode in {"compiler_balanced", "compiler_experimental"} and max_llm_calls > 0:
        summary["llm_fallback_status"] = "available_but_not_attached"
    return summary
