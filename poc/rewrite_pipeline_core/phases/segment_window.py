"""Segment-window density optimization phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rewrite_controller.eligible_span_density import (
    build_eligible_span_density_contract as _eligible_span_density_contract,
    compare_eligible_span_density as _eligible_span_density_comparison,
)
from rewrite_controller.segment_window_density import (
    SEGMENT_WINDOW_CONTROLLER_VERSION as _SEGMENT_WINDOW_CONTROLLER_VERSION,
    assemble_segment_window_candidate as _assemble_segment_window_candidate,
    build_segment_density_windows as _segment_density_windows,
    extract_segment_window_payload as _extract_segment_window_payload,
    segment_patchwork_budget as _segment_window_patchwork_budget,
    segment_window_candidate_prompt as _segment_window_candidate_prompt,
    segment_window_tasks as _segment_window_tasks,
)
from rewrite_pipeline_core.scoring.profiles import _turnitin_like_ai_profile


@dataclass(frozen=True)
class SegmentWindowDensityControllerDeps:
    env_flag: Callable[..., bool]
    float_env: Callable[[str, float], float]
    safe_topk_calibrated_limit: Callable[[], float]
    run_full_scan_report_dict: Callable[[str], dict]
    check_semantic_drift: Callable[..., Any]
    detect_protected_spans: Callable[[str], Any]
    clean_full_document_candidate: Callable[[str, str], str]
    ai_candidate_quality_reject_reason: Callable[[str], str]
    evaluate_text_quality_regression: Callable[..., dict]
    ai_search_protected_loss_reason: Callable[..., str]
    report_review_burden: Callable[[dict | None], int | float]
    report_weighted_severity: Callable[[dict | None], int | float]
    critical_high_count: Callable[[dict | None], int | float]
    segment_window_density_acceptance: Callable[..., dict]
    phase_chat_sampling_kwargs: Callable[..., dict]


def run_segment_window_density_controller(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    gateway: LLMGateway | None = None,
    scan_func=None,
    drift_checker=None,
    max_scans: int | None = None,
    max_llm_calls: int | None = None,
    deps: SegmentWindowDensityControllerDeps | None = None,
) -> dict:
    """Run scoped 5-10 sentence density-window LLM patches."""
    if deps is None:
        raise ValueError("SegmentWindowDensityControllerDeps is required")
    drift_checker = drift_checker or deps.check_semantic_drift
    if not deps.env_flag("DRAFTPROOF_SEGMENT_WINDOW_DENSITY_CONTROLLER", True):
        return {"enabled": False, "reason": "disabled"}
    if not isinstance(current_text, str) or not current_text.strip() or not isinstance(current_report, dict):
        return {"enabled": False, "reason": "missing_current_selection"}
    scan_func = scan_func or deps.run_full_scan_report_dict
    max_scans = max(
        0,
        int(max_scans if isinstance(max_scans, int) else deps.float_env("DRAFTPROOF_SEGMENT_WINDOW_MAX_SCANS", 3.0)),
    )
    max_llm_calls = max(
        0,
        int(max_llm_calls if isinstance(max_llm_calls, int) else deps.float_env("DRAFTPROOF_SEGMENT_WINDOW_MAX_LLM_CALLS", 3.0)),
    )
    current_profile = _turnitin_like_ai_profile(current_report)
    density_before = _eligible_span_density_contract(current_text, current_report)
    components = current_profile.get("components") if isinstance(current_profile.get("components"), dict) else {}
    topk_after = float(components.get("topk_calibrated_risk") or 0.0)
    if bool(current_profile.get("target_met")) and bool(density_before.get("safe")):
        return {
            "enabled": True,
            "selected": False,
            "reason": "already_turnitin_and_density_safe",
            "segment_density_windows": _segment_density_windows(current_text, current_report),
        }
    if bool(density_before.get("safe")) and topk_after < deps.safe_topk_calibrated_limit():
        return {
            "enabled": True,
            "selected": False,
            "reason": "density_and_topk_already_safe",
            "segment_density_windows": _segment_density_windows(current_text, current_report),
        }
    if gateway is None or max_llm_calls <= 0:
        return {
            "enabled": True,
            "selected": False,
            "reason": "no_llm_budget_or_gateway",
            "segment_density_windows": _segment_density_windows(current_text, current_report),
            "eligible_span_density_before": density_before,
        }
    if max_scans <= 0:
        return {"enabled": True, "selected": False, "reason": "scan_budget_zero"}

    tasks = _segment_window_tasks(current_text, current_report, limit=max_llm_calls)
    summary = {
        "enabled": True,
        "version": _SEGMENT_WINDOW_CONTROLLER_VERSION,
        "selected": False,
        "selected_text": current_text,
        "selected_report": current_report,
        "selected_strategy": None,
        "llm_calls": 0,
        "scans_used": 0,
        "candidate_count": 0,
        "segment_density_windows": _segment_density_windows(current_text, current_report),
        "eligible_span_density_before": density_before,
        "base_summary": {
            "turnitin_like_ai_score": current_profile.get("score"),
            "target_met": current_profile.get("target_met"),
            "topk_calibrated_risk": topk_after,
        },
        "candidate_frontier": [],
        "reason": "no_candidate_selected",
    }
    if not tasks:
        summary["reason"] = "no_segment_window_tasks"
        return summary

    protected = deps.detect_protected_spans(current_text)
    best_eval = None
    best_rank = None
    best_text = current_text
    best_report = current_report
    seen = {current_text.strip()}
    for task_index, task in enumerate(tasks, start=1):
        if summary["llm_calls"] >= max_llm_calls or summary["scans_used"] >= max_scans:
            break
        strategy = f"segment_window_density_{str(task.get('family') or 'window').lower()}_{task_index}"
        candidate_eval = {
            "strategy": strategy,
            "task": task,
            "passed_local_checks": False,
        }
        try:
            prompt = _segment_window_candidate_prompt(current_text, current_report, task)
            summary["llm_calls"] += 1
            response = gateway.chat(
                prompt,
                system=(
                    "You are DraftProof's segment-window density controller. "
                    "Return only JSON sentence patches for the selected window."
                ),
                **deps.phase_chat_sampling_kwargs(
                    "DRAFTPROOF_SEGMENT_WINDOW_DENSITY",
                    temperature_env="DRAFTPROOF_SEGMENT_WINDOW_TEMPERATURE",
                    temperature_default=0.42,
                    max_tokens_env="DRAFTPROOF_SEGMENT_WINDOW_MAX_TOKENS",
                    max_tokens_default=2600,
                ),
            )
            payload, payload_reason = _extract_segment_window_payload(response.content)
        except Exception as exc:
            candidate_eval["reason"] = f"llm_error {exc}"
            summary["candidate_frontier"].append(candidate_eval)
            continue
        if not payload:
            candidate_eval["reason"] = payload_reason or "invalid_segment_window_payload"
            summary["candidate_frontier"].append(candidate_eval)
            continue
        assembled, applied, assembly_reason = _assemble_segment_window_candidate(current_text, payload, task)
        candidate_text = deps.clean_full_document_candidate(assembled, current_text)
        candidate_eval.update({
            "payload_strategy": payload.get("strategy"),
            "applied_sentence_patches": applied,
            "targeted_drivers": payload.get("targeted_drivers"),
        })
        if not candidate_text:
            candidate_eval["reason"] = assembly_reason or "empty_or_unchanged_candidate"
            summary["candidate_frontier"].append(candidate_eval)
            continue
        if candidate_text.strip() in seen:
            candidate_eval["reason"] = "duplicate_candidate"
            summary["candidate_frontier"].append(candidate_eval)
            continue
        seen.add(candidate_text.strip())
        local_reason = deps.ai_candidate_quality_reject_reason(candidate_text)
        if local_reason:
            candidate_eval["reason"] = local_reason
            summary["candidate_frontier"].append(candidate_eval)
            continue
        patchwork = _segment_window_patchwork_budget(current_text, candidate_text, applied)
        candidate_eval["patchwork_budget"] = patchwork
        if not patchwork.get("accepted"):
            candidate_eval["reason"] = patchwork.get("reason") or "patchwork_edit_budget_exceeded"
            summary["candidate_frontier"].append(candidate_eval)
            continue
        quality_gate = deps.evaluate_text_quality_regression(
            current_text,
            candidate_text,
            changed_sentence_ratio=patchwork.get("edited_sentence_ratio"),
        )
        candidate_eval["quality_gate"] = quality_gate
        if not quality_gate.get("passed"):
            candidate_eval["reason"] = (quality_gate.get("reject_reasons") or ["quality_gate_failed"])[0]
            summary["candidate_frontier"].append(candidate_eval)
            continue
        protected_loss = deps.ai_search_protected_loss_reason(current_text, candidate_text, protected)
        if protected_loss:
            candidate_eval["reason"] = "protected_span_lost " + protected_loss
            summary["candidate_frontier"].append(candidate_eval)
            continue
        try:
            drift = drift_checker(current_text, candidate_text, threshold=0.80)
        except TypeError:
            drift = drift_checker(current_text, candidate_text)
        candidate_eval["drift_similarity"] = round(float(getattr(drift, "similarity", 1.0)), 3)
        if not bool(getattr(drift, "accepted", True)):
            candidate_eval["reason"] = "semantic_drift " + "; ".join(list(getattr(drift, "reasons", []) or [])[:3])
            candidate_eval["drift_reasons"] = list(getattr(drift, "reasons", []) or [])[:10]
            summary["candidate_frontier"].append(candidate_eval)
            continue
        try:
            candidate_report = scan_func(candidate_text)
        except Exception as exc:
            candidate_eval["reason"] = f"candidate_scan_error {exc}"
            summary["candidate_frontier"].append(candidate_eval)
            continue
        summary["scans_used"] += 1
        review_delta = deps.report_review_burden(candidate_report) - deps.report_review_burden(current_report)
        severity_delta = deps.report_weighted_severity(candidate_report) - deps.report_weighted_severity(current_report)
        critical_delta = deps.critical_high_count(candidate_report) - deps.critical_high_count(current_report)
        acceptance = deps.segment_window_density_acceptance(
            current_text,
            current_report,
            candidate_text,
            candidate_report,
            review_burden_delta=review_delta,
            weighted_severity_delta=severity_delta,
            critical_high_delta=critical_delta,
        )
        candidate_eval.update({
            "passed_local_checks": True,
            "acceptance": acceptance,
            "selectable": acceptance.get("selectable"),
            "reason": acceptance.get("reason"),
            "formula_score": acceptance.get("formula_score_after"),
            "formula_score_drop": acceptance.get("formula_score_drop"),
            "eligible_span_density_gate": acceptance.get("eligible_span_density_gate"),
            "driver_drops": acceptance.get("driver_drops"),
        })
        summary["candidate_frontier"].append(candidate_eval)
        if not acceptance.get("selectable"):
            continue
        density_gate = acceptance.get("eligible_span_density_gate") or {}
        rank = (
            1 if acceptance.get("target_met") and density_gate.get("safe") else 0,
            float(acceptance.get("formula_score_drop") or 0.0),
            float(density_gate.get("unsafe_eligible_word_ratio_drop") or 0.0),
            float(density_gate.get("longest_unsafe_span_words_drop") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("topk_calibrated_risk") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("ai_likelihood") or 0.0),
            -float((acceptance.get("formula_score_after") or 100.0)),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_eval = candidate_eval
            best_text = candidate_text
            best_report = candidate_report

    summary["candidate_count"] = len(summary.get("candidate_frontier") or [])
    if best_eval:
        best_eval["selected"] = True
        final_density = _eligible_span_density_comparison(
            current_text,
            current_report,
            best_text,
            best_report,
        )
        summary.update({
            "selected": True,
            "selected_text": best_text,
            "selected_report": best_report,
            "selected_strategy": best_eval.get("strategy"),
            "selected_candidate": {
                key: best_eval.get(key)
                for key in (
                    "strategy",
                    "formula_score",
                    "formula_score_drop",
                    "eligible_span_density_gate",
                    "driver_drops",
                    "patchwork_budget",
                    "drift_similarity",
                    "applied_sentence_patches",
                    "acceptance",
                )
            },
            "eligible_span_density_after": final_density.get("after"),
            "eligible_span_density_drop": final_density.get("unsafe_eligible_word_ratio_drop"),
            "longest_unsafe_span_words_drop": final_density.get("longest_unsafe_span_words_drop"),
            "reason": best_eval.get("reason"),
            "final_summary": {
                "turnitin_like_ai_score": _turnitin_like_ai_profile(best_report).get("score"),
                "target_met": _turnitin_like_ai_profile(best_report).get("target_met"),
                "eligible_span_density_safe": final_density.get("safe"),
            },
        })
    elif summary["candidate_frontier"]:
        summary["reason"] = "ceiling_reached"
        summary["why_not_below_20"] = "No segment-window candidate reduced both formula score and eligible-span density without safety regression."
    return summary
