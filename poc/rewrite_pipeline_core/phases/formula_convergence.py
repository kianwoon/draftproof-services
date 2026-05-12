"""Formula-convergence controller phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math
import time

from rewrite_pipeline_core.scoring.profiles import (
    _ai_footprint_profile,
    _blocker_scores,
    _contribution_scores,
    _formula_gap_candidate_rank,
    _formula_gap_contract,
    _formula_portfolio_plan,
    _integrity_scores,
    _remaining_turnitin_like_drivers,
    _turnitin_like_ai_gate_status,
    _turnitin_like_ai_profile,
)


@dataclass(frozen=True)
class FormulaConvergenceControllerDeps:
    env_flag: Callable[..., bool]
    run_full_scan_report_dict: Callable[[str], dict]
    check_semantic_drift: Callable[..., Any]
    formula_convergence_budget: Callable[..., dict]
    report_review_burden: Callable[[dict | None], int | float]
    report_weighted_severity: Callable[[dict | None], int | float]
    critical_high_count: Callable[[dict | None], int | float]
    formula_block_driver_map: Callable[..., dict]
    human_anchor_suppression_frontier: Callable[..., dict]
    formula_feasibility_estimator: Callable[..., dict]
    geometry_risk_map: Callable[..., dict]
    formula_convergence_candidate_batch: Callable[..., list]
    formula_convergence_llm_patch_candidates: Callable[..., list]
    ai_candidate_quality_reject_reason: Callable[[str], str]
    ai_search_protected_loss_reason: Callable[..., str]
    detect_protected_spans: Callable[[str], Any]
    candidate_concept_origin_reject_reason: Callable[..., str]
    anti_smoothing_guard_status: Callable[..., dict]
    formula_convergence_primary_burden_gate_status: Callable[..., dict]
    strict_ai_safe_band_status: Callable[[dict | None], dict]
    report_badge_ai: Callable[[dict | None], Any]
    report_finding_total: Callable[[dict | None], int]


def _formula_convergence_candidate_public(row: dict | None) -> dict:
    row = row if isinstance(row, dict) else {}
    contract = row.get("formula_gap_contract") if isinstance(row.get("formula_gap_contract"), dict) else {}
    gate = row.get("turnitin_like_ai_gate") if isinstance(row.get("turnitin_like_ai_gate"), dict) else {}
    status = row.get("selection_status") if isinstance(row.get("selection_status"), dict) else {}
    return {
        "pass_index": row.get("pass_index"),
        "strategy": row.get("strategy"),
        "selectable": status.get("selectable", row.get("selectable")),
        "selected": row.get("selected", False),
        "reason": row.get("reason") or status.get("reason"),
        "score_before": contract.get("score_before"),
        "score_after": contract.get("score_after"),
        "score_drop": contract.get("score_drop"),
        "target_met": contract.get("target_met"),
        "remaining_formula_gap": contract.get("remaining_formula_gap"),
        "weighted_driver_drops": contract.get("weighted_driver_drops"),
        "formula_gap_contract": {
            "score_before": contract.get("score_before"),
            "score_after": contract.get("score_after"),
            "score_drop": contract.get("score_drop"),
            "target_met": contract.get("target_met"),
            "weighted_driver_drops": contract.get("weighted_driver_drops"),
        },
        "turnitin_like_ai_gate": {
            "safety_clean": gate.get("safety_clean"),
            "safe_band": gate.get("safe_band"),
            "score_drop": gate.get("score_drop"),
            "outcome_class": gate.get("outcome_class"),
            "component_drops": gate.get("component_drops"),
        },
        "anti_smoothing_guard": row.get("anti_smoothing_guard"),
        "primary_burden_gate": row.get("primary_burden_gate"),
        "concept_origin_guard": row.get("concept_origin_guard"),
        "applied_formula_convergence_patches": row.get("applied_formula_convergence_patches"),
        "selection_status": {
            "selectable": status.get("selectable", row.get("selectable")),
            "reason": row.get("reason") or status.get("reason"),
        },
        "turnitin_like_outcome_class": gate.get("outcome_class"),
        "ai_authorship": row.get("ai_authorship"),
        "ai_transformation": row.get("ai_transformation"),
        "review_burden": row.get("review_burden"),
        "weighted_severity": row.get("weighted_severity"),
        "critical_high_findings": row.get("critical_high_findings"),
    }


def run_formula_convergence_controller(
    current_text: str,
    current_report: dict | None,
    original_report: dict | None,
    budget: dict | None = None,
    *,
    scan_func=None,
    candidate_builder=None,
    drift_checker=None,
    llm_gateway: LLMGateway | None = None,
    deps: FormulaConvergenceControllerDeps | None = None,
) -> dict:
    """Iteratively close the Turnitin-like formula gap from the current best.

    This is deliberately stateful: each pass plans against the best rescanned
    candidate from the previous pass. It never rolls back safe partial formula
    progress just because the strict <20 target is not reached.
    """
    if deps is None:
        raise ValueError("FormulaConvergenceControllerDeps is required")
    if not deps.env_flag("DRAFTPROOF_FORMULA_CONVERGENCE_CONTROLLER", True):
        return {
            "enabled": False,
            "reason": "formula_convergence_controller_disabled",
            "selected": False,
        }
    scan_func = scan_func or deps.run_full_scan_report_dict
    drift_checker = drift_checker or deps.check_semantic_drift
    resolved_budget = deps.formula_convergence_budget(current_text, budget)
    max_passes = int(resolved_budget.get("max_passes") or 0)
    max_scans = int(resolved_budget.get("max_scans") or 0)
    if max_passes <= 0 or max_scans <= 0:
        return {
            "enabled": True,
            "selected": False,
            "reason": "formula_convergence_budget_zero",
            "phase_budget_contract": resolved_budget,
        }

    best_text = str(current_text or "")
    best_report = current_report if isinstance(current_report, dict) else {}
    start_profile = _turnitin_like_ai_profile(best_report)
    best_profile = start_profile
    original_profile = _turnitin_like_ai_profile(original_report)
    selected_any = False
    selected_strategy = None
    selected_eval = None
    scans_used = 0
    llm_calls_used = 0
    passes: list[dict] = []
    candidates: list[dict] = []
    best_frontier: list[dict] = []
    last_block_map: dict | None = None

    def num(value, default=0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else float(default)

    def report_snapshot(report_dict: dict | None) -> dict:
        profile = _turnitin_like_ai_profile(report_dict)
        integrity = _integrity_scores(report_dict)
        contribution = _contribution_scores(report_dict)
        return {
            "turnitin_like_ai_score": profile.get("score"),
            "target_gap": profile.get("target_gap"),
            "target_met": profile.get("target_met"),
            "human_anchor_suppression": profile.get("human_anchor_suppression"),
            "positive_ai_burden": profile.get("raw_positive_score"),
            "ai_authorship": integrity.get("ai_authorship"),
            "ai_transformation": contribution.get("ai_transformation"),
            "external_ai_flag_risk": _ai_footprint_profile(report_dict).get("external_ai_flag_risk"),
            "review_burden": deps.report_review_burden(report_dict),
            "weighted_severity": deps.report_weighted_severity(report_dict),
            "critical_high_findings": deps.critical_high_count(report_dict),
        }

    stop_reason = ""
    for pass_index in range(1, max_passes + 1):
        best_profile = _turnitin_like_ai_profile(best_report)
        if bool(best_profile.get("target_met")):
            stop_reason = "turnitin_like_target_met"
            break
        if scans_used >= max_scans:
            stop_reason = "scan_budget_exhausted"
            break
        block_map = deps.formula_block_driver_map(best_text, best_report)
        last_block_map = block_map
        anchor_frontier = deps.human_anchor_suppression_frontier(best_text, best_report, block_map)
        feasibility = deps.formula_feasibility_estimator(best_report, observed_candidates=candidates)
        geometry_map = deps.geometry_risk_map(best_text, best_report)
        remaining_scans = max(0, max_scans - scans_used)
        batch_limit = max(1, min(remaining_scans, int(math.ceil(max_scans / max(1, max_passes)))))
        if candidate_builder:
            raw_batch = candidate_builder(best_text, best_report, pass_index, block_map)
        else:
            deterministic_limit = batch_limit
            if (
                llm_gateway is not None
                and llm_calls_used < int(resolved_budget.get("max_llm_calls") or 0)
                and batch_limit > 2
            ):
                deterministic_limit = max(1, batch_limit - 2)
            raw_batch = deps.formula_convergence_candidate_batch(
                best_text,
                best_report,
                block_map,
                limit=deterministic_limit,
            )
            if (
                llm_gateway is not None
                and llm_calls_used < int(resolved_budget.get("max_llm_calls") or 0)
                and len(raw_batch) < remaining_scans
            ):
                llm_calls_used += 1
                llm_candidates = deps.formula_convergence_llm_patch_candidates(
                    best_text,
                    best_report,
                    block_map,
                    llm_gateway,
                    max_candidates=min(3, max(1, remaining_scans - len(raw_batch))),
                )
                raw_batch = list(raw_batch or []) + llm_candidates
        batch: list[tuple[str, str, dict]] = []
        seen = {best_text.strip()}
        for item in raw_batch or []:
            if not item or len(item) < 2:
                continue
            strategy = str(item[0] or f"candidate_{len(batch)+1}")
            candidate_text = str(item[1] or "")
            meta = item[2] if len(item) > 2 and isinstance(item[2], dict) else {}
            normalized = candidate_text.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            batch.append((strategy, candidate_text, meta))
            if len(batch) >= remaining_scans:
                break
        pass_summary = {
            "pass_index": pass_index,
            "score_before": best_profile.get("score"),
            "target_gap_before": best_profile.get("target_gap"),
            "block_driver_map": {
                key: block_map.get(key)
                for key in ("version", "block_count", "formula_score", "target_score", "remaining_gap", "dominant_formula_drivers", "top_blocks")
            },
            "feasibility_estimator": feasibility,
            "geometry_risk_map": {
                "version": geometry_map.get("version"),
                "sentence_count": geometry_map.get("sentence_count"),
                "median_sentence_words": geometry_map.get("median_sentence_words"),
                "dominant_weighted_drivers": geometry_map.get("dominant_weighted_drivers"),
                "top_sentence_hotspots": (geometry_map.get("sentence_hotspots") or [])[:6],
                "top_paragraph_hotspots": (geometry_map.get("paragraph_hotspots") or [])[:4],
            },
            "human_anchor_suppression_frontier": anchor_frontier,
            "candidate_count": len(batch),
            "generated_candidates": len(raw_batch or []),
            "llm_calls_used": llm_calls_used,
            "scanned": 0,
            "selected": False,
        }
        if not batch:
            pass_summary["reason"] = "no_candidate_batch_generated"
            passes.append(pass_summary)
            stop_reason = "no_candidate_batch_generated"
            break

        pass_best = None
        pass_best_rank = None
        for strategy, candidate_text, meta in batch:
            if scans_used >= max_scans:
                stop_reason = "scan_budget_exhausted"
                break
            candidate_eval = {
                "pass_index": pass_index,
                "strategy": strategy,
                "formula_convergence_candidate": True,
                "formula_convergence_controller": True,
                **(meta or {}),
            }
            local_reason = deps.ai_candidate_quality_reject_reason(candidate_text)
            if local_reason:
                candidate_eval["reason"] = local_reason
                candidate_eval["selection_status"] = {"selectable": False, "reason": local_reason}
                candidates.append(candidate_eval)
                continue
            protected_loss = deps.ai_search_protected_loss_reason(best_text, candidate_text, deps.detect_protected_spans(best_text))
            if protected_loss:
                reason = "protected_span_lost " + protected_loss
                candidate_eval["reason"] = reason
                candidate_eval["selection_status"] = {"selectable": False, "reason": reason}
                candidates.append(candidate_eval)
                continue
            concept_origin_reason = deps.candidate_concept_origin_reject_reason(best_text, candidate_text)
            if concept_origin_reason:
                candidate_eval["reason"] = concept_origin_reason
                candidate_eval["concept_origin_guard"] = {
                    "accepted": False,
                    "reason": concept_origin_reason,
                }
                candidate_eval["selection_status"] = {
                    "selectable": False,
                    "reason": concept_origin_reason,
                }
                candidates.append(candidate_eval)
                continue
            try:
                drift = drift_checker(best_text, candidate_text, threshold=0.15)
            except TypeError:
                drift = drift_checker(best_text, candidate_text)
            candidate_eval["drift_similarity"] = round(float(getattr(drift, "similarity", 1.0)), 3)
            if not bool(getattr(drift, "accepted", True)):
                reason = "semantic_drift " + "; ".join(list(getattr(drift, "reasons", []) or [])[:3])
                candidate_eval["reason"] = reason
                candidate_eval["drift_reasons"] = list(getattr(drift, "reasons", []) or [])[:10]
                candidate_eval["selection_status"] = {"selectable": False, "reason": reason}
                candidates.append(candidate_eval)
                continue
            try:
                scan_t0 = time.time()
                candidate_report = scan_func(candidate_text)
                scan_seconds = round(time.time() - scan_t0, 3)
            except Exception as exc:
                reason = f"candidate_scan_error {exc}"
                candidate_eval["reason"] = reason
                candidate_eval["selection_status"] = {"selectable": False, "reason": reason}
                candidates.append(candidate_eval)
                continue
            scans_used += 1
            pass_summary["scanned"] += 1

            candidate_profile = _turnitin_like_ai_profile(candidate_report)
            contract_vs_current = _formula_gap_contract(
                best_report,
                candidate_report,
                source_text=best_text,
                candidate_text=candidate_text,
            )
            contract_vs_original = _formula_gap_contract(
                original_report,
                candidate_report,
                source_text=current_text,
                candidate_text=candidate_text,
            )
            review_delta = deps.report_review_burden(candidate_report) - deps.report_review_burden(best_report)
            severity_delta = deps.report_weighted_severity(candidate_report) - deps.report_weighted_severity(best_report)
            critical_delta = deps.critical_high_count(candidate_report) - deps.critical_high_count(best_report)
            current_blockers = _blocker_scores(best_report)
            candidate_blockers = _blocker_scores(candidate_report)
            unsupported_claim_delta = (
                float(candidate_blockers.get("unsupported_claim_risk") or 0.0)
                - float(current_blockers.get("unsupported_claim_risk") or 0.0)
            )
            turnitin_gate = _turnitin_like_ai_gate_status(
                best_report,
                candidate_report,
                review_burden_delta=review_delta,
                weighted_severity_delta=severity_delta,
                critical_high_delta=critical_delta,
                ai_score_regressed=False,
            )
            current_integrity = _integrity_scores(best_report)
            candidate_integrity = _integrity_scores(candidate_report)
            current_contribution = _contribution_scores(best_report)
            candidate_contribution = _contribution_scores(candidate_report)
            current_authorship = current_integrity.get("ai_authorship")
            candidate_authorship = candidate_integrity.get("ai_authorship")
            current_transformation = current_contribution.get("ai_transformation")
            candidate_transformation = candidate_contribution.get("ai_transformation")
            reject_reasons: list[str] = []
            anti_smoothing = deps.anti_smoothing_guard_status(
                best_report,
                candidate_report,
                strict=bool(candidate_eval.get("coordinated_micro_perturbation") or candidate_eval.get("geometry_mode")),
            )
            primary_burden_gate = deps.formula_convergence_primary_burden_gate_status(
                best_report,
                candidate_report,
                contract_vs_current,
            )
            if (
                candidate_eval.get("coordinated_micro_perturbation")
                or candidate_eval.get("geometry_mode")
            ) and not anti_smoothing.get("accepted"):
                reject_reasons.append(str(anti_smoothing.get("reason") or "anti_smoothing_guard_failed"))
            if num(contract_vs_current.get("score_drop"), 0.0) <= 0.05:
                reject_reasons.append("no_safe_formula_drop")
            if not primary_burden_gate.get("accepted"):
                reject_reasons.append(str(primary_burden_gate.get("reason") or "primary_burden_gate_failed"))
            if review_delta > 0:
                reject_reasons.append("review_burden_regressed")
            if severity_delta > 0:
                reject_reasons.append("weighted_severity_regressed")
            if critical_delta > 0:
                reject_reasons.append("critical_high_regressed")
            if (
                isinstance(current_authorship, (int, float))
                and isinstance(candidate_authorship, (int, float))
                and float(candidate_authorship) > float(current_authorship) + 0.001
            ):
                reject_reasons.append("ai_authorship_regressed")
            if (
                isinstance(current_transformation, (int, float))
                and isinstance(candidate_transformation, (int, float))
                and float(candidate_transformation) > float(current_transformation) + 0.001
            ):
                reject_reasons.append("ai_transformation_regressed")
            if (
                candidate_eval.get("human_anchor_suppression_frontier")
                or candidate_eval.get("human_anchor_amplifier")
                or candidate_eval.get("portfolio_operation") in {
                    "human_anchor_suppression_gain",
                    "human_anchor_plus_texture_rebuild",
                    "low_value_remove_plus_human_anchor",
                }
            ) and unsupported_claim_delta > 3.0:
                reject_reasons.append("unsupported_claim_risk_regressed")

            selectable = not reject_reasons
            reason = "accepted_formula_convergence_step" if selectable else reject_reasons[0]
            candidate_eval.update({
                "scan_seconds": scan_seconds,
                "ai": deps.report_badge_ai(candidate_report),
                "human_contribution": candidate_contribution.get("human"),
                "ai_transformation": candidate_contribution.get("ai_transformation"),
                "ai_authorship": candidate_integrity.get("ai_authorship"),
                "external_ai_flag_risk": _ai_footprint_profile(candidate_report).get("external_ai_flag_risk"),
                "findings": deps.report_finding_total(candidate_report),
                "review_burden": deps.report_review_burden(candidate_report),
                "weighted_severity": deps.report_weighted_severity(candidate_report),
                "critical_high_findings": deps.critical_high_count(candidate_report),
                "unsupported_claim_risk_delta": round(unsupported_claim_delta, 3),
                "formula_gap_contract": contract_vs_current,
                "formula_gap_contract_vs_original": contract_vs_original,
                "anti_smoothing_guard": anti_smoothing,
                "primary_burden_gate": primary_burden_gate,
                "concept_origin_guard": {
                    "accepted": True,
                    "reason": "accepted",
                },
                "turnitin_like_ai_gate": turnitin_gate,
                "strict_ai_safe_band": deps.strict_ai_safe_band_status(candidate_report),
                "selection_status": {
                    "selectable": selectable,
                    "success": selectable,
                    "reason": reason,
                    "formula_convergence_controller": True,
                    "turnitin_like_ai_gate": turnitin_gate,
                    "formula_gap_contract": contract_vs_current,
                    "formula_gap_contract_vs_original": contract_vs_original,
                    "formula_gap_rank": list(_formula_gap_candidate_rank(contract_vs_current, turnitin_gate)),
                    "target_met": bool(candidate_profile.get("target_met")),
                    "partial_turnitin_like_mitigation": bool(selectable and not candidate_profile.get("target_met")),
                    "turnitin_like_mitigation": bool(selectable and candidate_profile.get("target_met")),
                },
                "reason": reason,
            })
            candidates.append(candidate_eval)
            public_eval = _formula_convergence_candidate_public(candidate_eval)
            best_frontier.append(public_eval)
            best_frontier = sorted(
                best_frontier,
                key=lambda row: (
                    1 if row.get("selectable") else 0,
                    1 if row.get("target_met") else 0,
                    float(row.get("score_drop") or 0.0),
                    -float(row.get("score_after") if isinstance(row.get("score_after"), (int, float)) else 100.0),
                ),
                reverse=True,
            )[:12]
            if selectable:
                rank = (
                    1 if candidate_profile.get("target_met") else 0,
                    _formula_gap_candidate_rank(contract_vs_current, turnitin_gate),
                    -num(candidate_profile.get("score"), 100.0),
                )
                if pass_best_rank is None or rank > pass_best_rank:
                    pass_best_rank = rank
                    pass_best = {
                        "strategy": strategy,
                        "text": candidate_text,
                        "report": candidate_report,
                        "eval": candidate_eval,
                        "rank": rank,
                    }
        if pass_best:
            pass_best["eval"]["selected"] = True
            best_text = pass_best["text"]
            best_report = pass_best["report"]
            best_profile = _turnitin_like_ai_profile(best_report)
            selected_any = True
            selected_strategy = pass_best["strategy"]
            selected_eval = pass_best["eval"]
            pass_summary.update({
                "selected": True,
                "selected_strategy": selected_strategy,
                "score_after": best_profile.get("score"),
                "score_drop": round(
                    num(pass_summary.get("score_before"), 0.0) - num(best_profile.get("score"), 0.0),
                    3,
                ),
                "target_gap_after": best_profile.get("target_gap"),
                "selected_candidate": _formula_convergence_candidate_public(selected_eval),
            })
            if bool(best_profile.get("target_met")):
                stop_reason = "turnitin_like_target_met"
                passes.append(pass_summary)
                break
        else:
            pass_summary["reason"] = "no_safe_formula_movement"
            passes.append(pass_summary)
            if (
                pass_index < max_passes
                and scans_used < max_scans
                and llm_gateway is not None
                and llm_calls_used < int(resolved_budget.get("max_llm_calls") or 0)
            ):
                stop_reason = ""
                continue
            stop_reason = "no_safe_formula_movement"
            break
        passes.append(pass_summary)
    else:
        if not stop_reason:
            stop_reason = "pass_budget_exhausted"

    final_profile = _turnitin_like_ai_profile(best_report)
    final_contract = _formula_gap_contract(
        original_report,
        best_report,
        source_text=current_text,
        candidate_text=best_text,
    )
    public_candidates = [_formula_convergence_candidate_public(row) for row in candidates]
    selected_public = _formula_convergence_candidate_public(selected_eval) if selected_eval else None
    why_not = (
        "turnitin-like formula target achieved"
        if bool(final_profile.get("target_met"))
        else (
            f"{final_profile.get('score')} is not below "
            f"{final_profile.get('target_score')}; remaining drivers: "
            + ", ".join(
                str(row.get("driver"))
                for row in _remaining_turnitin_like_drivers(final_profile)[:4]
                if isinstance(row, dict) and row.get("driver")
            )
        )
    )
    return {
        "enabled": True,
        "version": "formula_convergence_controller_v1",
        "selected": selected_any,
        "selected_text": best_text,
        "selected_report": best_report,
        "selected_strategy": selected_strategy,
        "selected_formula_portfolio_candidate": selected_public,
        "score_before": start_profile.get("score"),
        "score_after": final_profile.get("score"),
        "score_drop": round(num(start_profile.get("score")) - num(final_profile.get("score")), 3),
        "target_score": final_profile.get("target_score"),
        "target_met": bool(final_profile.get("target_met")),
        "remaining_formula_gap": final_profile.get("target_gap"),
        "why_not_below_20": why_not,
        "stop_reason": stop_reason,
        "phase_budget_contract": resolved_budget,
        "scans_used": scans_used,
        "llm_calls_used": llm_calls_used,
        "phase_budget_used": {
            "passes": len(passes),
            "scans": scans_used,
            "llm_calls": llm_calls_used,
        },
        "original_snapshot": report_snapshot(original_report),
        "start_snapshot": report_snapshot(current_report),
        "final_snapshot": report_snapshot(best_report),
        "feasibility_estimator": deps.formula_feasibility_estimator(
            best_report,
            observed_candidates=candidates,
        ),
        "geometry_risk_map": deps.geometry_risk_map(best_text, best_report),
        "block_driver_map": last_block_map,
        "human_anchor_suppression_frontier": deps.human_anchor_suppression_frontier(
            best_text,
            best_report,
            last_block_map,
        ),
        "formula_convergence_passes": passes,
        "candidates": public_candidates,
        "best_formula_frontier": best_frontier,
        "formula_gap_contract": final_contract,
        "formula_portfolio_plan": _formula_portfolio_plan(
            original_report,
            best_report,
            observed_candidates=candidates,
        ),
    }
