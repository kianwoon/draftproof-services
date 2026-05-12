"""Authenticity acceptance gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AuthenticityGateDeps:
    contribution_scores: Callable[[dict | None], dict]
    integrity_scores: Callable[[dict | None], dict]
    float_env: Callable[[str, float], float]
    critical_high_count: Callable[[dict | None], int]
    human_shift_score: Callable[..., dict]


def authenticity_gate_status(
    original_report: dict,
    candidate_report: dict,
    text_changed: bool,
    *,
    original_review_burden: int,
    candidate_review_burden: int,
    original_weighted_severity: int,
    candidate_weighted_severity: int,
    min_human_gain: float = 2.0,
    min_ai_transformation_drop: float = 2.0,
    drift_similarity: float | None = None,
    deps: AuthenticityGateDeps | None = None,
) -> dict:
    if deps is None:
        raise ValueError("AuthenticityGateDeps is required")
    original = deps.contribution_scores(original_report)
    candidate = deps.contribution_scores(candidate_report)
    original_integrity = deps.integrity_scores(original_report)
    candidate_integrity = deps.integrity_scores(candidate_report)
    original_human = original.get("human")
    candidate_human = candidate.get("human")
    original_ai_transform = original.get("ai_transformation")
    candidate_ai_transform = candidate.get("ai_transformation")
    original_ai_authorship = original_integrity.get("ai_authorship")
    candidate_ai_authorship = candidate_integrity.get("ai_authorship")
    human_delta = (
        candidate_human - original_human
        if isinstance(original_human, (int, float)) and isinstance(candidate_human, (int, float))
        else None
    )
    ai_transform_delta = (
        original_ai_transform - candidate_ai_transform
        if (
            isinstance(original_ai_transform, (int, float))
            and isinstance(candidate_ai_transform, (int, float))
        )
        else None
    )
    ai_authorship_delta = (
        original_ai_authorship - candidate_ai_authorship
        if (
            isinstance(original_ai_authorship, (int, float))
            and isinstance(candidate_ai_authorship, (int, float))
        )
        else None
    )
    crosses_human_side = bool(
        isinstance(candidate_human, (int, float))
        and isinstance(candidate_ai_transform, (int, float))
        and candidate_human > candidate_ai_transform
        and (
            not isinstance(original_human, (int, float))
            or not isinstance(original_ai_transform, (int, float))
            or original_human <= original_ai_transform
        )
    )
    moves_toward_human = bool(
        (isinstance(human_delta, (int, float)) and human_delta >= min_human_gain)
        or (
            isinstance(ai_transform_delta, (int, float))
            and ai_transform_delta >= min_ai_transformation_drop
        )
        or crosses_human_side
    )
    reduces_ai_authorship = bool(
        isinstance(ai_authorship_delta, (int, float))
        and ai_authorship_delta >= deps.float_env("DRAFTPROOF_AUTHENTICITY_MIN_AI_AUTHORSHIP_DROP", 2.0)
    )
    ai_authorship_regression_tolerance = deps.float_env(
        "DRAFTPROOF_AUTHENTICITY_AI_AUTHORSHIP_REGRESSION_TOLERANCE",
        0.0,
    )
    ai_authorship_regressed = bool(
        isinstance(ai_authorship_delta, (int, float))
        and ai_authorship_delta < -ai_authorship_regression_tolerance
    )
    major_human_threshold = deps.float_env("DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_THRESHOLD", 80.0)
    major_human_gain = deps.float_env("DRAFTPROOF_AUTHENTICITY_MAJOR_HUMAN_GAIN", 50.0)
    major_human_breakthrough = bool(
        isinstance(candidate_human, (int, float))
        and isinstance(human_delta, (int, float))
        and candidate_human >= major_human_threshold
        and human_delta >= major_human_gain
    )
    ai_authorship_regression_blocked = ai_authorship_regressed
    target_human = deps.float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    strong_accept_min_human_gain = deps.float_env("DRAFTPROOF_AUTHENTICITY_STRONG_ACCEPT_MIN_HUMAN_GAIN", 20.0)
    strong_accept_min_transform_drop = deps.float_env("DRAFTPROOF_AUTHENTICITY_STRONG_ACCEPT_MIN_AI_TRANSFORM_DROP", 20.0)
    strong_accept_min_shift = deps.float_env("DRAFTPROOF_AUTHENTICITY_STRONG_ACCEPT_MIN_SHIFT", 45.0)
    crosses_target_human = bool(
        isinstance(candidate_human, (int, float))
        and candidate_human >= target_human
    )
    critical_high_regressed = deps.critical_high_count(candidate_report) > deps.critical_high_count(original_report)
    review_regressed = candidate_review_burden > original_review_burden
    severity_regressed = candidate_weighted_severity > original_weighted_severity
    human_shift = deps.human_shift_score(
        original_report,
        candidate_report,
        drift_similarity=drift_similarity,
        review_burden_delta=candidate_review_burden - original_review_burden,
        weighted_severity_delta=candidate_weighted_severity - original_weighted_severity,
    )
    human_shift_score = human_shift.get("score")
    authorship_cost_per_human_gain = (
        round(max(0.0, -float(ai_authorship_delta)) / max(float(human_delta), 1.0), 3)
        if isinstance(ai_authorship_delta, (int, float)) and isinstance(human_delta, (int, float))
        else None
    )
    human_gain_with_authorship_regression = bool(
        isinstance(human_delta, (int, float))
        and human_delta > 0
        and ai_authorship_regressed
    )
    min_human_shift_score = deps.float_env("DRAFTPROOF_AUTHENTICITY_MIN_HUMAN_SHIFT_SCORE", 3.0)
    clears_human_shift_score = bool(
        isinstance(human_shift_score, (int, float))
        and human_shift_score >= min_human_shift_score
    )
    positive_human_shift = bool(
        isinstance(human_shift_score, (int, float))
        and human_shift_score > 0
        and (moves_toward_human or reduces_ai_authorship)
    )
    strong_below_target_accept = bool(
        isinstance(candidate_human, (int, float))
        and candidate_human < target_human
        and isinstance(human_delta, (int, float))
        and human_delta >= strong_accept_min_human_gain
        and isinstance(ai_transform_delta, (int, float))
        and ai_transform_delta >= strong_accept_min_transform_drop
        and isinstance(human_shift_score, (int, float))
        and human_shift_score >= strong_accept_min_shift
        and not ai_authorship_regressed
    )
    target_accept = crosses_target_human or strong_below_target_accept
    target_gap_progress = bool(
        crosses_target_human
        or strong_below_target_accept
        or (
            isinstance(original_human, (int, float))
            and float(original_human) < target_human
            and (
                (
                    isinstance(human_delta, (int, float))
                    and human_delta >= min_human_gain
                )
                or (
                    isinstance(ai_transform_delta, (int, float))
                    and ai_transform_delta >= min_ai_transformation_drop
                )
            )
        )
        or not (isinstance(original_human, (int, float)) and float(original_human) < target_human)
    )
    human_target_regressed = bool(
        isinstance(original_human, (int, float))
        and original_human < target_human
        and isinstance(human_delta, (int, float))
        and human_delta < 0
    )
    ai_transformation_target_regressed = bool(
        isinstance(original_human, (int, float))
        and original_human < target_human
        and isinstance(ai_transform_delta, (int, float))
        and ai_transform_delta < 0
    )
    candidate_progress = bool(
        text_changed
        and (clears_human_shift_score or positive_human_shift)
        and not ai_authorship_regression_blocked
        and not human_target_regressed
        and not ai_transformation_target_regressed
        and target_gap_progress
        and not critical_high_regressed
        and not review_regressed
        and not severity_regressed
    )
    success = bool(
        candidate_progress
        and target_accept
    )
    reason = ""
    if success:
        reason = "accepted"
    elif not text_changed:
        reason = "unchanged_candidate"
    elif ai_authorship_regression_blocked:
        reason = "ai_authorship_regressed"
    elif human_target_regressed:
        reason = "human_target_regressed"
    elif ai_transformation_target_regressed:
        reason = "ai_transformation_target_regressed"
    elif not target_gap_progress:
        reason = "no_human_target_progress"
    elif not (clears_human_shift_score or positive_human_shift):
        reason = "human_shift_score_too_low"
    elif critical_high_regressed:
        reason = "critical_high_regressed"
    elif review_regressed:
        reason = "review_burden_regressed"
    elif severity_regressed:
        reason = "weighted_severity_regressed"
    elif candidate_progress:
        reason = "candidate_progress_below_target"
    return {
        "success": success,
        "reason": reason,
        "original_human": original_human,
        "candidate_human": candidate_human,
        "human_delta": human_delta,
        "original_ai_transformation": original_ai_transform,
        "candidate_ai_transformation": candidate_ai_transform,
        "ai_transformation_delta": ai_transform_delta,
        "original_ai_authorship": original_ai_authorship,
        "candidate_ai_authorship": candidate_ai_authorship,
        "ai_authorship_delta": ai_authorship_delta,
        "reduces_ai_authorship": reduces_ai_authorship,
        "ai_authorship_regressed": ai_authorship_regressed,
        "ai_authorship_regression_blocked": ai_authorship_regression_blocked,
        "ai_authorship_regression_tolerance": ai_authorship_regression_tolerance,
        "human_gain_with_authorship_regression": human_gain_with_authorship_regression,
        "false_positive_improvement": human_gain_with_authorship_regression,
        "false_positive_improvement_reason": (
            "human_gain_with_authorship_regression"
            if human_gain_with_authorship_regression
            else ""
        ),
        "major_human_breakthrough": major_human_breakthrough,
        "target_human": target_human,
        "crosses_target_human": crosses_target_human,
        "strong_below_target_accept": strong_below_target_accept,
        "strong_accept_min_human_gain": strong_accept_min_human_gain,
        "strong_accept_min_ai_transform_drop": strong_accept_min_transform_drop,
        "strong_accept_min_shift": strong_accept_min_shift,
        "target_accept": target_accept,
        "target_gap_progress": target_gap_progress,
        "candidate_progress": candidate_progress,
        "human_target_regressed": human_target_regressed,
        "ai_transformation_target_regressed": ai_transformation_target_regressed,
        "crosses_human_side": crosses_human_side,
        "human_shift_score": human_shift_score,
        "human_shift_components": human_shift.get("components"),
        "authorship_cost_per_human_gain": authorship_cost_per_human_gain,
        "human_shift_weights": human_shift.get("weights"),
        "min_human_shift_score": min_human_shift_score,
        "clears_human_shift_score": clears_human_shift_score,
        "positive_human_shift": positive_human_shift,
        "min_human_gain": min_human_gain,
        "min_ai_transformation_drop": min_ai_transformation_drop,
        "critical_high_regressed": critical_high_regressed,
        "review_burden_regressed": review_regressed,
        "weighted_severity_regressed": severity_regressed,
    }
