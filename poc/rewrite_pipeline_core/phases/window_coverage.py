"""Window-coverage density optimization phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rewrite_controller.eligible_span_density import (
    build_eligible_span_density_contract as _eligible_span_density_contract,
    compare_eligible_span_density as _eligible_span_density_comparison,
)
from rewrite_controller.window_coverage_density import (
    WINDOW_COVERAGE_CONTROLLER_VERSION as _WINDOW_COVERAGE_CONTROLLER_VERSION,
    assemble_window_coverage_candidate as _assemble_window_coverage_candidate,
    build_window_coverage_map as _window_coverage_map,
    compare_window_coverage_density as _window_coverage_comparison,
    extract_window_coverage_payload as _extract_window_coverage_payload,
    window_coverage_ablation_candidates as _window_coverage_ablation_candidates,
    window_coverage_candidate_prompt as _window_coverage_candidate_prompt,
    window_coverage_deterministic_variants as _window_coverage_deterministic_variants,
    window_coverage_patchwork_budget as _window_coverage_patchwork_budget,
    window_coverage_portfolio_candidates as _window_coverage_portfolio_candidates,
    window_coverage_tasks as _window_coverage_tasks,
)
from rewrite_pipeline_core.scoring.profiles import _turnitin_like_ai_profile


@dataclass(frozen=True)
class WindowCoverageOptimizerDeps:
    env_flag: Callable[..., bool]
    float_env: Callable[[str, float], float]
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
    window_coverage_density_acceptance: Callable[..., dict]
    phase_chat_sampling_kwargs: Callable[..., dict]


def run_window_coverage_density_optimizer(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    *,
    gateway: LLMGateway | None = None,
    scan_func=None,
    drift_checker=None,
    max_scans: int | None = None,
    max_llm_calls: int | None = None,
    deps: WindowCoverageOptimizerDeps | None = None,
) -> dict:
    """Run portfolio patches for high-leverage unsafe sliding-window coverage."""
    if deps is None:
        raise ValueError("WindowCoverageOptimizerDeps is required")
    if not deps.env_flag("DRAFTPROOF_WINDOW_COVERAGE_DENSITY_OPTIMIZER", True):
        return {"enabled": False, "reason": "disabled"}
    if not isinstance(current_text, str) or not current_text.strip() or not isinstance(current_report, dict):
        return {"enabled": False, "reason": "missing_current_selection"}
    scan_func = scan_func or deps.run_full_scan_report_dict
    drift_checker = drift_checker or deps.check_semantic_drift
    max_scans = max(
        0,
        int(max_scans if isinstance(max_scans, int) else deps.float_env("DRAFTPROOF_WINDOW_COVERAGE_MAX_SCANS", 5.0)),
    )
    max_llm_calls = max(
        0,
        int(max_llm_calls if isinstance(max_llm_calls, int) else deps.float_env("DRAFTPROOF_WINDOW_COVERAGE_MAX_LLM_CALLS", 5.0)),
    )
    current_profile = _turnitin_like_ai_profile(current_report)
    coverage_map = _window_coverage_map(current_text, current_report)
    density_before = coverage_map.get("eligible_span_density") or _eligible_span_density_contract(current_text, current_report)
    if bool(current_profile.get("target_met")) and bool(density_before.get("safe")):
        return {
            "enabled": True,
            "selected": False,
            "reason": "already_turnitin_and_density_safe",
            "window_coverage_map": coverage_map,
        }
    if bool(density_before.get("safe")) and int(coverage_map.get("unsafe_window_count") or 0) <= 0:
        return {
            "enabled": True,
            "selected": False,
            "reason": "density_and_window_coverage_already_safe",
            "window_coverage_map": coverage_map,
        }
    if max_scans <= 0:
        return {"enabled": True, "selected": False, "reason": "scan_budget_zero", "window_coverage_map": coverage_map}

    summary = {
        "enabled": True,
        "version": _WINDOW_COVERAGE_CONTROLLER_VERSION,
        "window_coverage_portfolio_optimizer": True,
        "selected": False,
        "selected_text": current_text,
        "selected_report": current_report,
        "selected_strategy": None,
        "llm_calls": 0,
        "scans_used": 0,
        "candidate_count": 0,
        "window_coverage_map": coverage_map,
        "top_coverage_sentences": coverage_map.get("top_coverage_sentences"),
        "eligible_span_density_before": density_before,
        "unsafe_window_count_before": coverage_map.get("unsafe_window_count"),
        "ai_sentence_vote_ratio_before": coverage_map.get("ai_sentence_vote_ratio"),
        "base_summary": {
            "turnitin_like_ai_score": current_profile.get("score"),
            "target_met": current_profile.get("target_met"),
        },
        "deterministic_variant_frontier": [],
        "portfolio_candidate_frontier": [],
        "patch_ablation_frontier": [],
        "window_coverage_passes": [],
        "candidate_frontier": [],
        "reason": "no_candidate_selected",
    }

    final_eval = None
    final_text = current_text
    final_report = current_report
    seen = {current_text.strip()}
    protected_cache: dict[int, Any] = {}

    def protected_for(text: str):
        key = id(text)
        if key not in protected_cache:
            protected_cache[key] = deps.detect_protected_spans(text)
        return protected_cache[key]

    def candidate_rank(acceptance: dict) -> tuple:
        coverage_gate = acceptance.get("window_coverage_gate") or {}
        density_gate = acceptance.get("eligible_span_density_gate") or {}
        patchwork = acceptance.get("patchwork_budget") or {}
        return (
            1 if acceptance.get("target_met") and coverage_gate.get("safe") and density_gate.get("safe") else 0,
            float(acceptance.get("formula_score_drop") or 0.0),
            float(coverage_gate.get("unsafe_window_count_drop") or 0.0),
            float(coverage_gate.get("ai_sentence_vote_ratio_drop") or 0.0),
            float(density_gate.get("unsafe_eligible_word_ratio_drop") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("topk_calibrated_risk") or 0.0),
            float((acceptance.get("driver_drops") or {}).get("ai_likelihood") or 0.0),
            -int(patchwork.get("edited_sentence_count") or 0),
            -float(acceptance.get("formula_score_after") or 100.0),
        )

    def evaluate_candidate(
        base_text: str,
        base_report: dict,
        candidate_text: str,
        applied: list[dict],
        candidate_eval: dict,
    ) -> tuple[dict, dict | None]:
        candidate_text = deps.clean_full_document_candidate(candidate_text, base_text)
        if not candidate_text:
            candidate_eval["reason"] = candidate_eval.get("reason") or "empty_or_unchanged_candidate"
            summary["candidate_frontier"].append(candidate_eval)
            return candidate_eval, None
        if candidate_text.strip() in seen:
            candidate_eval["reason"] = "duplicate_candidate"
            summary["candidate_frontier"].append(candidate_eval)
            return candidate_eval, None
        seen.add(candidate_text.strip())
        candidate_eval["applied_sentence_patches"] = applied
        local_reason = deps.ai_candidate_quality_reject_reason(candidate_text)
        if local_reason:
            candidate_eval["reason"] = local_reason
            summary["candidate_frontier"].append(candidate_eval)
            return candidate_eval, None
        patchwork = _window_coverage_patchwork_budget(base_text, candidate_text, applied)
        candidate_eval["patchwork_budget"] = patchwork
        if not patchwork.get("accepted"):
            candidate_eval["reason"] = patchwork.get("reason") or "window_coverage_patchwork_budget_exceeded"
            summary["candidate_frontier"].append(candidate_eval)
            return candidate_eval, None
        quality_gate = deps.evaluate_text_quality_regression(
            base_text,
            candidate_text,
            changed_sentence_ratio=patchwork.get("edited_sentence_ratio"),
        )
        candidate_eval["quality_gate"] = quality_gate
        if not quality_gate.get("passed"):
            candidate_eval["reason"] = (quality_gate.get("reject_reasons") or ["quality_gate_failed"])[0]
            summary["candidate_frontier"].append(candidate_eval)
            return candidate_eval, None
        protected_loss = deps.ai_search_protected_loss_reason(base_text, candidate_text, protected_for(base_text))
        if protected_loss:
            candidate_eval["reason"] = "protected_span_lost " + protected_loss
            summary["candidate_frontier"].append(candidate_eval)
            return candidate_eval, None
        try:
            drift = drift_checker(base_text, candidate_text, threshold=0.80)
        except TypeError:
            drift = drift_checker(base_text, candidate_text)
        candidate_eval["drift_similarity"] = round(float(getattr(drift, "similarity", 1.0)), 3)
        if not bool(getattr(drift, "accepted", True)):
            candidate_eval["reason"] = "semantic_drift " + "; ".join(list(getattr(drift, "reasons", []) or [])[:3])
            candidate_eval["drift_reasons"] = list(getattr(drift, "reasons", []) or [])[:10]
            summary["candidate_frontier"].append(candidate_eval)
            return candidate_eval, None
        try:
            candidate_report = scan_func(candidate_text)
        except Exception as exc:
            candidate_eval["reason"] = f"candidate_scan_error {exc}"
            summary["candidate_frontier"].append(candidate_eval)
            return candidate_eval, None
        summary["scans_used"] += 1
        review_delta = deps.report_review_burden(candidate_report) - deps.report_review_burden(base_report)
        severity_delta = deps.report_weighted_severity(candidate_report) - deps.report_weighted_severity(base_report)
        critical_delta = deps.critical_high_count(candidate_report) - deps.critical_high_count(base_report)
        acceptance = deps.window_coverage_density_acceptance(
            base_text,
            base_report,
            candidate_text,
            candidate_report,
            review_burden_delta=review_delta,
            weighted_severity_delta=severity_delta,
            critical_high_delta=critical_delta,
        )
        acceptance["patchwork_budget"] = patchwork
        candidate_eval.update({
            "passed_local_checks": True,
            "acceptance": acceptance,
            "selectable": acceptance.get("selectable"),
            "reason": acceptance.get("reason"),
            "formula_score": acceptance.get("formula_score_after"),
            "formula_score_drop": acceptance.get("formula_score_drop"),
            "window_coverage_gate": acceptance.get("window_coverage_gate"),
            "eligible_span_density_gate": acceptance.get("eligible_span_density_gate"),
            "driver_drops": acceptance.get("driver_drops"),
        })
        summary["candidate_frontier"].append(candidate_eval)
        return candidate_eval, candidate_report

    pass_text = current_text
    pass_report = current_report
    ablation_scans_used = 0
    max_ablation_scans = min(2, max_scans)
    for pass_index in range(1, 3):
        if summary["scans_used"] >= max_scans:
            break
        pass_summary = {
            "pass": pass_index,
            "base_score": _turnitin_like_ai_profile(pass_report).get("score"),
            "selected": False,
        }
        variants = _window_coverage_deterministic_variants(pass_text, pass_report, sentence_limit=8, variant_limit=32)
        portfolios = _window_coverage_portfolio_candidates(
            pass_text,
            pass_report,
            variants=variants,
            portfolio_limit=10,
        )
        if pass_index == 1:
            summary["deterministic_variant_frontier"] = variants[:24]
            summary["portfolio_candidate_frontier"] = [
                {
                    "strategy": item.get("strategy"),
                    "source": item.get("source"),
                    "predicted_rank": item.get("predicted_rank"),
                    "applied_sentence_patches": item.get("applied_sentence_patches"),
                }
                for item in portfolios[:10]
            ]

        pass_best_eval = None
        pass_best_text = None
        pass_best_report = None
        pass_best_rank = None
        finalist_limit = max(0, min(6, max_scans - summary["scans_used"]))
        for portfolio in portfolios[:finalist_limit]:
            candidate_eval = {
                "strategy": portfolio.get("strategy"),
                "source": portfolio.get("source"),
                "pass": pass_index,
                "predicted_rank": portfolio.get("predicted_rank"),
                "passed_local_checks": False,
            }
            evaluated, candidate_report = evaluate_candidate(
                pass_text,
                pass_report,
                str(portfolio.get("candidate_text") or ""),
                list(portfolio.get("applied_sentence_patches") or []),
                candidate_eval,
            )
            if evaluated.get("selectable"):
                rank = candidate_rank(evaluated.get("acceptance") or {})
                if pass_best_rank is None or rank > pass_best_rank:
                    pass_best_rank = rank
                    pass_best_eval = evaluated
                    pass_best_text = str(portfolio.get("candidate_text") or "")
                    pass_best_report = candidate_report
                continue
            if (
                candidate_report is not None
                and ablation_scans_used < max_ablation_scans
                and summary["scans_used"] < max_scans
                and evaluated.get("reason") in {
                    "formula_score_not_reduced",
                    "headline_ai_score_regressed",
                    "ai_authorship_regressed",
                    "ai_transformation_regressed",
                    "topk_calibrated_risk_regressed",
                    "ai_likelihood_regressed",
                    "external_ai_flag_risk_regressed",
                    "review_burden_regressed",
                    "weighted_severity_regressed",
                    "critical_high_regressed",
                }
            ):
                for ablation in _window_coverage_ablation_candidates(
                    pass_text,
                    evaluated.get("applied_sentence_patches") or [],
                    limit=max_ablation_scans - ablation_scans_used,
                ):
                    if summary["scans_used"] >= max_scans or ablation_scans_used >= max_ablation_scans:
                        break
                    ablation_scans_used += 1
                    summary["patch_ablation_frontier"].append({
                        "strategy": ablation.get("strategy"),
                        "parent_strategy": evaluated.get("strategy"),
                        "dropped_patch": ablation.get("dropped_patch"),
                    })
                    ablation_eval = {
                        "strategy": ablation.get("strategy"),
                        "source": ablation.get("source"),
                        "parent_strategy": evaluated.get("strategy"),
                        "pass": pass_index,
                        "predicted_rank": ablation.get("predicted_rank"),
                        "dropped_patch": ablation.get("dropped_patch"),
                        "passed_local_checks": False,
                    }
                    ablated, ablated_report = evaluate_candidate(
                        pass_text,
                        pass_report,
                        str(ablation.get("candidate_text") or ""),
                        list(ablation.get("applied_sentence_patches") or []),
                        ablation_eval,
                    )
                    if ablated.get("selectable"):
                        rank = candidate_rank(ablated.get("acceptance") or {})
                        if pass_best_rank is None or rank > pass_best_rank:
                            pass_best_rank = rank
                            pass_best_eval = ablated
                            pass_best_text = str(ablation.get("candidate_text") or "")
                            pass_best_report = ablated_report

        if not pass_best_eval and gateway is not None and summary["llm_calls"] < max_llm_calls:
            tasks = _window_coverage_tasks(pass_text, pass_report, limit=max_llm_calls - summary["llm_calls"])
            for task_index, task in enumerate(tasks, start=1):
                if summary["llm_calls"] >= max_llm_calls or summary["scans_used"] >= max_scans:
                    break
                strategy = f"window_coverage_{str(task.get('family') or 'coverage').lower()}_{task_index}"
                candidate_eval = {
                    "strategy": strategy,
                    "source": "llm_fallback",
                    "task": task,
                    "pass": pass_index,
                    "passed_local_checks": False,
                }
                try:
                    prompt = _window_coverage_candidate_prompt(pass_text, pass_report, task)
                    summary["llm_calls"] += 1
                    response = gateway.chat(
                        prompt,
                        system=(
                            "You are DraftProof's window-coverage density optimizer. "
                            "Return only JSON sentence patches for the selected high-coverage sentences."
                        ),
                        **deps.phase_chat_sampling_kwargs(
                            "DRAFTPROOF_WINDOW_COVERAGE_DENSITY",
                            temperature_env="DRAFTPROOF_WINDOW_COVERAGE_TEMPERATURE",
                            temperature_default=0.42,
                            max_tokens_env="DRAFTPROOF_WINDOW_COVERAGE_MAX_TOKENS",
                            max_tokens_default=3200,
                        ),
                    )
                    payload, payload_reason = _extract_window_coverage_payload(response.content)
                except Exception as exc:
                    candidate_eval["reason"] = f"llm_error {exc}"
                    summary["candidate_frontier"].append(candidate_eval)
                    continue
                if not payload:
                    candidate_eval["reason"] = payload_reason or "invalid_window_coverage_payload"
                    summary["candidate_frontier"].append(candidate_eval)
                    continue
                assembled, applied, assembly_reason = _assemble_window_coverage_candidate(pass_text, payload, task)
                candidate_eval.update({
                    "payload_strategy": payload.get("strategy"),
                    "targeted_drivers": payload.get("targeted_drivers"),
                    "assembly_reason": assembly_reason,
                })
                evaluated, candidate_report = evaluate_candidate(
                    pass_text,
                    pass_report,
                    assembled,
                    applied,
                    candidate_eval,
                )
                if evaluated.get("selectable"):
                    rank = candidate_rank(evaluated.get("acceptance") or {})
                    if pass_best_rank is None or rank > pass_best_rank:
                        pass_best_rank = rank
                        pass_best_eval = evaluated
                        pass_best_text = assembled
                        pass_best_report = candidate_report

        if pass_best_eval and pass_best_text and pass_best_report:
            pass_best_eval["selected"] = True
            final_eval = pass_best_eval
            final_text = pass_best_text
            final_report = pass_best_report
            pass_text = pass_best_text
            pass_report = pass_best_report
            pass_summary.update({
                "selected": True,
                "source": pass_best_eval.get("source"),
                "strategy": pass_best_eval.get("strategy"),
                "formula_score_after": pass_best_eval.get("formula_score"),
                "formula_score_drop": pass_best_eval.get("formula_score_drop"),
            })
            summary["window_coverage_passes"].append(pass_summary)
            if bool(_turnitin_like_ai_profile(pass_report).get("target_met")):
                break
            continue

        pass_summary["reason"] = "no_selectable_window_coverage_candidate"
        summary["window_coverage_passes"].append(pass_summary)
        break

    summary["candidate_count"] = len(summary.get("candidate_frontier") or [])
    if final_eval:
        final_coverage = _window_coverage_comparison(
            current_text,
            current_report,
            final_text,
            final_report,
        )
        final_density = _eligible_span_density_comparison(
            current_text,
            current_report,
            final_text,
            final_report,
        )
        initial_profile = _turnitin_like_ai_profile(current_report)
        final_profile = _turnitin_like_ai_profile(final_report)
        selected_drop = round(float(initial_profile.get("score") or 0.0) - float(final_profile.get("score") or 0.0), 3)
        edited_count = int(((final_eval.get("patchwork_budget") or {}).get("edited_sentence_count") or 0))
        coverage_efficiency = round(selected_drop / max(1, edited_count), 3)
        summary.update({
            "selected": True,
            "selected_text": final_text,
            "selected_report": final_report,
            "selected_strategy": final_eval.get("strategy"),
            "selected_candidate": {
                key: final_eval.get(key)
                for key in (
                    "strategy",
                    "source",
                    "formula_score",
                    "formula_score_drop",
                    "window_coverage_gate",
                    "eligible_span_density_gate",
                    "driver_drops",
                    "patchwork_budget",
                    "drift_similarity",
                    "applied_sentence_patches",
                    "acceptance",
                )
            },
            "selected_window_coverage_portfolio": {
                "strategy": final_eval.get("strategy"),
                "source": final_eval.get("source"),
                "formula_score_after": final_profile.get("score"),
                "formula_score_drop": selected_drop,
                "coverage_efficiency": coverage_efficiency,
                "applied_sentence_patches": final_eval.get("applied_sentence_patches"),
            },
            "coverage_efficiency": coverage_efficiency,
            "window_coverage_after": final_coverage.get("after"),
            "unsafe_window_count_after": (final_coverage.get("after") or {}).get("unsafe_window_count"),
            "unsafe_window_count_drop": final_coverage.get("unsafe_window_count_drop"),
            "ai_sentence_vote_ratio_after": (final_coverage.get("after") or {}).get("ai_sentence_vote_ratio"),
            "ai_sentence_vote_ratio_drop": final_coverage.get("ai_sentence_vote_ratio_drop"),
            "eligible_span_density_after": final_density.get("after"),
            "eligible_span_density_drop": final_density.get("unsafe_eligible_word_ratio_drop"),
            "reason": final_eval.get("reason"),
            "final_summary": {
                "turnitin_like_ai_score": final_profile.get("score"),
                "target_met": final_profile.get("target_met"),
                "eligible_span_density_safe": final_density.get("safe"),
                "window_coverage_safe": final_coverage.get("safe"),
            },
        })
    elif summary["candidate_frontier"]:
        summary["reason"] = "ceiling_reached"
        summary["why_not_below_20"] = "No window-coverage candidate reduced total formula score and unsafe window coverage without safety regression."
    return summary

