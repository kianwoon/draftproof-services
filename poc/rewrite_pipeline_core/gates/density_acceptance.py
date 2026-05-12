from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DensityAcceptanceDeps:
    turnitin_like_ai_profile: Callable[[dict | None], dict]
    eligible_span_density_comparison: Callable[[str, dict | None, str, dict | None], dict]
    ai_footprint_profile: Callable[[dict | None], dict]
    ai_footprint_flatten: Callable[[dict | None], dict]
    report_badge_ai: Callable[[dict | None], float | int | None]
    window_coverage_comparison: Callable[[str, dict | None, str, dict | None], dict] | None = None


def segment_window_density_acceptance(
    current_text: str,
    current_report: dict | None,
    candidate_text: str,
    candidate_report: dict | None,
    deps: DensityAcceptanceDeps,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    """Acceptance policy for scoped 5-10 sentence density-window candidates."""
    base_profile = deps.turnitin_like_ai_profile(current_report)
    candidate_profile = deps.turnitin_like_ai_profile(candidate_report)
    formula_drop = round(float(base_profile.get("score") or 0.0) - float(candidate_profile.get("score") or 0.0), 3)
    density_gate = deps.eligible_span_density_comparison(
        current_text,
        current_report,
        candidate_text,
        candidate_report,
    )
    base_flat = deps.ai_footprint_flatten(deps.ai_footprint_profile(current_report))
    candidate_flat = deps.ai_footprint_flatten(deps.ai_footprint_profile(candidate_report))

    def drop(key: str) -> float:
        return round(float(base_flat.get(key) or 0.0) - float(candidate_flat.get(key) or 0.0), 3)

    driver_drops = {
        key: drop(key)
        for key in (
            "topk_calibrated_risk",
            "qualifying_text_ai_density",
            "external_ai_flag_risk",
            "ai_likelihood",
            "rewrite_smoothness",
            "ai_authorship",
            "ai_transformation",
            "semantic_uniformity",
        )
    }
    current_ai = deps.report_badge_ai(current_report)
    candidate_ai = deps.report_badge_ai(candidate_report)
    headline_ai_drop = (
        round(float(current_ai) - float(candidate_ai), 3)
        if isinstance(current_ai, (int, float)) and isinstance(candidate_ai, (int, float))
        else 0.0
    )
    reject_reasons: list[str] = []
    if formula_drop <= 0.001:
        reject_reasons.append("formula_score_not_reduced")
    unsafe_ratio_drop = float(density_gate.get("unsafe_eligible_word_ratio_drop") or 0.0)
    longest_span_drop = float(density_gate.get("longest_unsafe_span_words_drop") or 0.0)
    density_non_worsening = bool(
        density_gate.get("safe")
        or (unsafe_ratio_drop >= -0.001 and longest_span_drop >= -0.001)
    )
    density_positive = bool(
        density_gate.get("safe")
        or density_gate.get("improved")
        or unsafe_ratio_drop > 0.001
        or longest_span_drop > 0.001
    )
    if not density_positive:
        reject_reasons.append("eligible_span_density_not_improved")
    elif not density_non_worsening:
        reject_reasons.append("eligible_span_density_regressed")
    if headline_ai_drop < -0.001:
        reject_reasons.append("headline_ai_score_regressed")
    for key in ("ai_authorship", "ai_transformation", "external_ai_flag_risk", "ai_likelihood"):
        if driver_drops.get(key, 0.0) < -0.001:
            reject_reasons.append(f"{key}_regressed")
    if float(review_burden_delta or 0.0) > 0.0:
        reject_reasons.append("review_burden_regressed")
    if float(weighted_severity_delta or 0.0) > 0.0:
        reject_reasons.append("weighted_severity_regressed")
    if float(critical_high_delta or 0.0) > 0.0:
        reject_reasons.append("critical_high_regressed")
    selectable = not reject_reasons
    return {
        "version": "segment_window_density_acceptance_v1",
        "selectable": selectable,
        "reason": "accepted_segment_window_density_improvement" if selectable else reject_reasons[0],
        "formula_score_before": base_profile.get("score"),
        "formula_score_after": candidate_profile.get("score"),
        "formula_score_drop": formula_drop,
        "target_met": bool(candidate_profile.get("target_met")),
        "headline_ai_drop": headline_ai_drop,
        "driver_drops": driver_drops,
        "eligible_span_density_gate": density_gate,
        "density_positive": density_positive,
        "density_non_worsening": density_non_worsening,
        "unsafe_eligible_word_ratio_drop": density_gate.get("unsafe_eligible_word_ratio_drop"),
        "longest_unsafe_span_words_drop": density_gate.get("longest_unsafe_span_words_drop"),
        "review_burden_delta": review_burden_delta,
        "weighted_severity_delta": weighted_severity_delta,
        "critical_high_delta": critical_high_delta,
    }



def remaining_cluster_density_acceptance(
    current_text: str,
    current_report: dict | None,
    candidate_text: str,
    candidate_report: dict | None,
    deps: DensityAcceptanceDeps,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    """Acceptance policy for remaining unsafe-cluster candidates."""
    base_profile = deps.turnitin_like_ai_profile(current_report)
    candidate_profile = deps.turnitin_like_ai_profile(candidate_report)
    formula_drop = round(float(base_profile.get("score") or 0.0) - float(candidate_profile.get("score") or 0.0), 3)
    density_gate = deps.eligible_span_density_comparison(
        current_text,
        current_report,
        candidate_text,
        candidate_report,
    )
    base_flat = deps.ai_footprint_flatten(deps.ai_footprint_profile(current_report))
    candidate_flat = deps.ai_footprint_flatten(deps.ai_footprint_profile(candidate_report))

    def drop(key: str) -> float:
        return round(float(base_flat.get(key) or 0.0) - float(candidate_flat.get(key) or 0.0), 3)

    driver_drops = {
        key: drop(key)
        for key in (
            "topk_calibrated_risk",
            "qualifying_text_ai_density",
            "external_ai_flag_risk",
            "ai_likelihood",
            "rewrite_smoothness",
            "ai_authorship",
            "ai_transformation",
            "semantic_uniformity",
        )
    }
    current_ai = deps.report_badge_ai(current_report)
    candidate_ai = deps.report_badge_ai(candidate_report)
    headline_ai_drop = (
        round(float(current_ai) - float(candidate_ai), 3)
        if isinstance(current_ai, (int, float)) and isinstance(candidate_ai, (int, float))
        else 0.0
    )
    unsafe_ratio_drop = float(density_gate.get("unsafe_eligible_word_ratio_drop") or 0.0)
    longest_span_drop = float(density_gate.get("longest_unsafe_span_words_drop") or 0.0)
    reject_reasons: list[str] = []
    if formula_drop <= 0.001:
        reject_reasons.append("formula_score_not_reduced")
    if unsafe_ratio_drop < -0.001 or longest_span_drop < -0.001:
        reject_reasons.append("eligible_span_density_regressed")
    if headline_ai_drop < -0.001:
        reject_reasons.append("headline_ai_score_regressed")
    for key in (
        "ai_authorship",
        "ai_transformation",
        "topk_calibrated_risk",
        "ai_likelihood",
        "external_ai_flag_risk",
    ):
        if driver_drops.get(key, 0.0) < -0.001:
            reject_reasons.append(f"{key}_regressed")
    if float(review_burden_delta or 0.0) > 0.0:
        reject_reasons.append("review_burden_regressed")
    if float(weighted_severity_delta or 0.0) > 0.0:
        reject_reasons.append("weighted_severity_regressed")
    if float(critical_high_delta or 0.0) > 0.0:
        reject_reasons.append("critical_high_regressed")
    selectable = not reject_reasons
    return {
        "version": "remaining_cluster_density_acceptance_v1",
        "selectable": selectable,
        "reason": "accepted_remaining_cluster_formula_density_improvement" if selectable else reject_reasons[0],
        "formula_score_before": base_profile.get("score"),
        "formula_score_after": candidate_profile.get("score"),
        "formula_score_drop": formula_drop,
        "target_met": bool(candidate_profile.get("target_met")),
        "headline_ai_drop": headline_ai_drop,
        "driver_drops": driver_drops,
        "eligible_span_density_gate": density_gate,
        "unsafe_eligible_word_ratio_drop": density_gate.get("unsafe_eligible_word_ratio_drop"),
        "longest_unsafe_span_words_drop": density_gate.get("longest_unsafe_span_words_drop"),
        "review_burden_delta": review_burden_delta,
        "weighted_severity_delta": weighted_severity_delta,
        "critical_high_delta": critical_high_delta,
    }



def window_coverage_density_acceptance(
    current_text: str,
    current_report: dict | None,
    candidate_text: str,
    candidate_report: dict | None,
    deps: DensityAcceptanceDeps,
    *,
    review_burden_delta: int | float,
    weighted_severity_delta: int | float,
    critical_high_delta: int | float,
) -> dict:
    """Acceptance policy for sliding-window coverage candidates."""
    base_profile = deps.turnitin_like_ai_profile(current_report)
    candidate_profile = deps.turnitin_like_ai_profile(candidate_report)
    formula_drop = round(float(base_profile.get("score") or 0.0) - float(candidate_profile.get("score") or 0.0), 3)
    density_gate = deps.eligible_span_density_comparison(
        current_text,
        current_report,
        candidate_text,
        candidate_report,
    )
    coverage_gate = deps.window_coverage_comparison(
        current_text,
        current_report,
        candidate_text,
        candidate_report,
    )
    base_flat = deps.ai_footprint_flatten(deps.ai_footprint_profile(current_report))
    candidate_flat = deps.ai_footprint_flatten(deps.ai_footprint_profile(candidate_report))

    def drop(key: str) -> float:
        return round(float(base_flat.get(key) or 0.0) - float(candidate_flat.get(key) or 0.0), 3)

    driver_drops = {
        key: drop(key)
        for key in (
            "topk_calibrated_risk",
            "qualifying_text_ai_density",
            "external_ai_flag_risk",
            "ai_likelihood",
            "rewrite_smoothness",
            "ai_authorship",
            "ai_transformation",
            "semantic_uniformity",
        )
    }
    current_ai = deps.report_badge_ai(current_report)
    candidate_ai = deps.report_badge_ai(candidate_report)
    headline_ai_drop = (
        round(float(current_ai) - float(candidate_ai), 3)
        if isinstance(current_ai, (int, float)) and isinstance(candidate_ai, (int, float))
        else 0.0
    )
    unsafe_ratio_drop = float(density_gate.get("unsafe_eligible_word_ratio_drop") or 0.0)
    longest_span_drop = float(density_gate.get("longest_unsafe_span_words_drop") or 0.0)
    unsafe_window_drop = float(coverage_gate.get("unsafe_window_count_drop") or 0.0)
    vote_ratio_drop = float(coverage_gate.get("ai_sentence_vote_ratio_drop") or 0.0)
    reject_reasons: list[str] = []
    if formula_drop <= 0.001:
        reject_reasons.append("formula_score_not_reduced")
    if unsafe_window_drop < -0.001 or vote_ratio_drop < -0.001:
        reject_reasons.append("window_coverage_regressed")
    if unsafe_ratio_drop < -0.001 or longest_span_drop < -0.001:
        reject_reasons.append("eligible_span_density_regressed")
    if not (unsafe_window_drop > 0.001 or vote_ratio_drop > 0.001 or unsafe_ratio_drop > 0.001):
        reject_reasons.append("unsafe_density_not_improved")
    if headline_ai_drop < -0.001:
        reject_reasons.append("headline_ai_score_regressed")
    for key in (
        "ai_authorship",
        "ai_transformation",
        "topk_calibrated_risk",
        "ai_likelihood",
        "external_ai_flag_risk",
    ):
        if driver_drops.get(key, 0.0) < -0.001:
            reject_reasons.append(f"{key}_regressed")
    if float(review_burden_delta or 0.0) > 0.0:
        reject_reasons.append("review_burden_regressed")
    if float(weighted_severity_delta or 0.0) > 0.0:
        reject_reasons.append("weighted_severity_regressed")
    if float(critical_high_delta or 0.0) > 0.0:
        reject_reasons.append("critical_high_regressed")
    selectable = not reject_reasons
    return {
        "version": "window_coverage_density_acceptance_v1",
        "selectable": selectable,
        "reason": "accepted_window_coverage_formula_density_improvement" if selectable else reject_reasons[0],
        "formula_score_before": base_profile.get("score"),
        "formula_score_after": candidate_profile.get("score"),
        "formula_score_drop": formula_drop,
        "target_met": bool(candidate_profile.get("target_met")),
        "headline_ai_drop": headline_ai_drop,
        "driver_drops": driver_drops,
        "eligible_span_density_gate": density_gate,
        "window_coverage_gate": coverage_gate,
        "unsafe_window_count_drop": coverage_gate.get("unsafe_window_count_drop"),
        "ai_sentence_vote_ratio_drop": coverage_gate.get("ai_sentence_vote_ratio_drop"),
        "unsafe_eligible_word_ratio_drop": density_gate.get("unsafe_eligible_word_ratio_drop"),
        "longest_unsafe_span_words_drop": density_gate.get("longest_unsafe_span_words_drop"),
        "review_burden_delta": review_burden_delta,
        "weighted_severity_delta": weighted_severity_delta,
        "critical_high_delta": critical_high_delta,
    }

