"""Typed candidate selection for rewrite pipeline V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .goal_contract import RewriteGoalEvaluation, RewriteGoalStatus


class CandidateLane(str, Enum):
    GOAL_MET = "GOAL_MET"
    SAFE_NEAR_MISS = "SAFE_NEAR_MISS"
    PARTIAL_DIAGNOSTIC = "PARTIAL_DIAGNOSTIC"
    REJECT = "REJECT"


@dataclass(frozen=True)
class CandidateDecision:
    lane: CandidateLane
    selected_as_success: bool
    goal_met: bool
    ai_target_gap: float | None
    required_drop_met: bool
    quality_safe: bool
    semantic_safe: bool
    reason: str
    rank: tuple

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lane"] = self.lane.value
        payload["rank"] = list(self.rank)
        return payload


def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _badge_ai(report: dict | None) -> float | None:
    score = ((report or {}).get("ai_risk_badge") or {}).get("ai_likelihood_score")
    return float(score) if isinstance(score, (int, float)) else None


def decide_candidate(
    *,
    goal: RewriteGoalEvaluation,
    original_report: dict | None,
    candidate_report: dict | None,
    reference_ai: float | None,
    required_ai_drop: float = 5.0,
    target_ai_score: float | None = None,
    semantic_safe: bool = True,
    quality_safe: bool = True,
    cost: int | float = 0,
) -> CandidateDecision:
    candidate_ai = _badge_ai(candidate_report)
    ai_drop = (
        float(reference_ai) - float(candidate_ai)
        if isinstance(reference_ai, (int, float)) and isinstance(candidate_ai, (int, float))
        else 0.0
    )
    if target_ai_score is None and isinstance(reference_ai, (int, float)):
        target_ai_score = float(reference_ai) - float(required_ai_drop)
    ai_target_gap = (
        max(0.0, float(candidate_ai) - float(target_ai_score))
        if isinstance(candidate_ai, (int, float)) and isinstance(target_ai_score, (int, float))
        else None
    )
    required_drop_met = bool(ai_drop >= float(required_ai_drop or 0.0))
    footprint = goal.ai_footprint_gate or {}
    turnitin = goal.turnitin_like_gate or {}
    detector_movement = max(
        0.0,
        _num((footprint.get("drops") or {}).get("ai_likelihood")),
        _num((footprint.get("drops") or {}).get("topk_calibrated_risk")),
        _num((footprint.get("drops") or {}).get("ai_authorship")),
        _num((footprint.get("drops") or {}).get("ai_transformation")),
        _num(turnitin.get("score_drop")),
    )
    preservation_safe = bool(quality_safe and semantic_safe)
    if goal.status == RewriteGoalStatus.AI_MITIGATED and preservation_safe:
        lane = CandidateLane.GOAL_MET
        reason = "strict_goal_met"
    elif goal.status == RewriteGoalStatus.AI_MITIGATED:
        lane = CandidateLane.SAFE_NEAR_MISS
        reason = "strict_detector_goal_met_but_preservation_review_required"
    elif quality_safe and semantic_safe and (required_drop_met or goal.detector_safe):
        lane = CandidateLane.SAFE_NEAR_MISS
        reason = "safe_near_miss"
    elif detector_movement > 0.0 or ai_drop > 0.0:
        lane = CandidateLane.PARTIAL_DIAGNOSTIC
        reason = "partial_progress_not_success"
    else:
        lane = CandidateLane.REJECT
        reason = "no_goal_progress"
    lane_priority = {
        CandidateLane.REJECT: 0,
        CandidateLane.PARTIAL_DIAGNOSTIC: 1,
        CandidateLane.SAFE_NEAR_MISS: 2,
        CandidateLane.GOAL_MET: 3,
    }[lane]
    gap_rank = -float(ai_target_gap) if ai_target_gap is not None else -9999.0
    rank = (
        lane_priority,
        1 if required_drop_met else 0,
        1 if goal.detector_safe else 0,
        gap_rank,
        ai_drop,
        detector_movement,
        1 if quality_safe else 0,
        1 if semantic_safe else 0,
        -float(cost or 0.0),
    )
    return CandidateDecision(
        lane=lane,
        selected_as_success=lane == CandidateLane.GOAL_MET,
        goal_met=goal.goal_met,
        ai_target_gap=round(ai_target_gap, 3) if ai_target_gap is not None else None,
        required_drop_met=required_drop_met,
        quality_safe=quality_safe,
        semantic_safe=semantic_safe,
        reason=reason,
        rank=rank,
    )


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    selectable = [
        row for row in candidates
        if isinstance(row.get("decision"), dict)
        and str(row["decision"].get("lane")) != CandidateLane.REJECT.value
    ]
    if not selectable:
        return None
    return max(selectable, key=lambda row: tuple(row["decision"].get("rank") or ()))
