"""Candidate scan evaluation and selection."""

from __future__ import annotations

from typing import Any

from .signals import formula_snapshot


def _score_value(report: dict | None, deps: Any) -> float:
    profile = deps.turnitin_profile(report)
    value = profile.get("score")
    return float(value) if isinstance(value, (int, float)) else 100.0


def evaluate_scanned_candidate(
    current_text: str,
    current_report: dict | None,
    candidate_text: str,
    candidate_report: dict | None,
    original_report: dict | None,
    deps: Any,
    *,
    validation: dict,
    quality: dict,
) -> dict:
    current_snapshot = formula_snapshot(current_text, current_report, deps)
    candidate_snapshot = formula_snapshot(candidate_text, candidate_report, deps)
    formula_drop = round(float(current_snapshot.get("score") or 0.0) - float(candidate_snapshot.get("score") or 0.0), 3)
    review_delta = deps.review_burden(candidate_report) - deps.review_burden(current_report)
    severity_delta = deps.weighted_severity(candidate_report) - deps.weighted_severity(current_report)
    critical_delta = deps.critical_high_count(candidate_report) - deps.critical_high_count(current_report)
    findings_delta = deps.finding_total(candidate_report) - deps.finding_total(current_report)
    current_integrity = deps.integrity_scores(current_report)
    candidate_integrity = deps.integrity_scores(candidate_report)
    current_contribution = deps.contribution_scores(current_report)
    candidate_contribution = deps.contribution_scores(candidate_report)
    ai_authorship_drop = round(
        float(current_integrity.get("ai_authorship") or 0.0) - float(candidate_integrity.get("ai_authorship") or 0.0),
        3,
    )
    ai_transformation_drop = round(
        float(current_contribution.get("ai_transformation") or 0.0)
        - float(candidate_contribution.get("ai_transformation") or 0.0),
        3,
    )
    ai_score_drop = round(float(deps.badge_ai(current_report) or 0.0) - float(deps.badge_ai(candidate_report) or 0.0), 3)
    turnitin_gate = deps.turnitin_gate_status(
        current_report,
        candidate_report,
        review_burden_delta=review_delta,
        weighted_severity_delta=severity_delta,
        critical_high_delta=critical_delta,
        ai_score_regressed=ai_score_drop < -0.001,
    )
    strict = deps.strict_safe_status(candidate_report)
    reject_reasons: list[str] = []
    if not validation.get("passed"):
        reject_reasons.extend(validation.get("reject_reasons") or ["validation_failed"])
    if not quality.get("passed"):
        reject_reasons.extend(quality.get("reject_reasons") or ["quality_guard_failed"])
    if review_delta > 0:
        reject_reasons.append("review_burden_regressed")
    if severity_delta > 0:
        reject_reasons.append("weighted_severity_regressed")
    if critical_delta > 0:
        reject_reasons.append("critical_high_regressed")
    if findings_delta > 0:
        reject_reasons.append("findings_regressed")
    if ai_authorship_drop < -0.001:
        reject_reasons.append("ai_authorship_regressed")
    if ai_transformation_drop < -0.001:
        reject_reasons.append("ai_transformation_regressed")
    if formula_drop <= 0.001 and not (review_delta < 0 or severity_delta < 0):
        reject_reasons.append("no_formula_or_cleanup_progress")
    target_met = bool(candidate_snapshot.get("score", 100.0) < 20.0)
    strict_safe = bool(strict.get("achieved"))
    if target_met and strict_safe:
        outcome_class = "ai_mitigated"
    elif formula_drop > 0.001 and not strict_safe:
        outcome_class = "unsafe_partial_improvement"
    elif formula_drop > 0.001:
        outcome_class = "partially_ai_mitigated"
    elif review_delta < 0 or severity_delta < 0:
        outcome_class = "cleanup_only"
    else:
        outcome_class = "original_preserved"
    quality_penalty = 0.0 if quality.get("passed") else 100.0
    patchwork = float((validation.get("locality") or {}).get("changed_sentence_ratio") or 0.0)
    rank = (
        1 if not reject_reasons else 0,
        1 if outcome_class == "ai_mitigated" else 0,
        formula_drop,
        max(0.0, ai_score_drop),
        max(0.0, ai_authorship_drop),
        max(0.0, ai_transformation_drop),
        -patchwork,
        -quality_penalty,
        -_score_value(candidate_report, deps),
    )
    return {
        "accepted": not reject_reasons,
        "reason": "accepted_compiler_candidate" if not reject_reasons else reject_reasons[0],
        "reject_reasons": reject_reasons,
        "outcome_class": outcome_class,
        "formula_score_before": current_snapshot.get("score"),
        "formula_score_after": candidate_snapshot.get("score"),
        "formula_score_drop": formula_drop,
        "turnitin_like_ai_gate": turnitin_gate,
        "strict_ai_safe_band": strict,
        "ai_score_drop": ai_score_drop,
        "ai_authorship_drop": ai_authorship_drop,
        "ai_transformation_drop": ai_transformation_drop,
        "findings_delta": findings_delta,
        "review_burden_delta": review_delta,
        "weighted_severity_delta": severity_delta,
        "critical_high_delta": critical_delta,
        "rank": rank,
    }


def better_candidate(candidate_eval: dict | None, best_eval: dict | None) -> bool:
    if candidate_eval is None or not candidate_eval.get("accepted"):
        return False
    if best_eval is None:
        return True
    return tuple(candidate_eval.get("rank") or ()) > tuple(best_eval.get("rank") or ())
