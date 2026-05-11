"""Shared progress and final-label policy for rewrite controllers.

This module keeps product semantics out of the large pipeline orchestration
file.  A rewrite can improve cleanup/review burden without being meaningful AI
mitigation; these helpers keep those lanes separate.
"""

from __future__ import annotations

from typing import Any


MEANINGFUL_AI_PROGRESS_THRESHOLDS = {
    "turnitin_like_ai_score_drop": 2.0,
    "ai_score_drop": 2.0,
    "ai_authorship_drop": 2.0,
    "ai_transformation_drop": 2.0,
    "positive_ai_burden_drop": 2.0,
    "ai_window_vote_ratio_drop": 5.0,
    "unsafe_eligible_density_drop": 3.0,
}

CLEANUP_PROGRESS_THRESHOLDS = {
    "findings_drop": 1.0,
    "review_burden_drop": 1.0,
    "weighted_severity_drop": 1.0,
}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _progress_rows(values: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, threshold in thresholds.items():
        value = _num(values.get(name))
        if value is None:
            continue
        rows.append({
            "driver": name,
            "value": round(value, 3),
            "threshold": round(float(threshold), 3),
            "met": bool(value >= float(threshold)),
        })
    return rows


def meaningful_ai_progress_gate(**values: Any) -> dict[str, Any]:
    """Return whether a candidate has product-meaningful AI-risk movement."""

    rows = _progress_rows(values, MEANINGFUL_AI_PROGRESS_THRESHOLDS)
    passed = [row for row in rows if row.get("met")]
    return {
        "meaningful": bool(passed),
        "passed": bool(passed),
        "reason": "meaningful_ai_progress" if passed else "no_meaningful_ai_progress",
        "drivers": passed,
        "observed": rows,
        "thresholds": dict(MEANINGFUL_AI_PROGRESS_THRESHOLDS),
    }


def cleanup_progress_gate(**values: Any) -> dict[str, Any]:
    """Return whether a candidate only made review/cleanup progress."""

    rows = _progress_rows(values, CLEANUP_PROGRESS_THRESHOLDS)
    passed = [row for row in rows if row.get("met")]
    return {
        "cleanup": bool(passed),
        "passed": bool(passed),
        "reason": "cleanup_progress" if passed else "no_cleanup_progress",
        "drivers": passed,
        "observed": rows,
        "thresholds": dict(CLEANUP_PROGRESS_THRESHOLDS),
    }


def final_rewrite_outcome_label(
    *,
    detector_safe: bool,
    text_changed: bool,
    meaningful_ai_progress: bool,
    cleanup_progress: bool,
    current_outcome: str | None = None,
) -> str:
    """Choose the final user-facing rewrite label from shared policy lanes."""

    if detector_safe:
        return "ai_mitigated"
    if text_changed and meaningful_ai_progress:
        return "unsafe_partial_improvement"
    if text_changed and cleanup_progress:
        return "cleanup_only"
    if text_changed:
        return "ceiling_reached"
    if current_outcome == "ceiling_reached":
        return "ceiling_reached"
    return "original_preserved"
