"""Frontier selection policy for rewrite V2 candidates."""

from __future__ import annotations

from typing import Any

from .partial_policy import close_partial_candidate_allowed, partial_application_policy
from .selection import CandidateLane, select_best_safe_progress_candidate


def candidate_patch_coverage(candidate: dict[str, Any] | None) -> int:
    if not isinstance(candidate, dict):
        return 0
    composed = candidate.get("composed_patches")
    if isinstance(composed, list):
        return len(composed)
    count = candidate.get("applied_patch_count")
    if isinstance(count, (int, float)):
        return int(count)
    patches = candidate.get("patches")
    if isinstance(patches, list):
        return sum(1 for patch in patches if isinstance(patch, dict) and patch.get("applied"))
    return 0


def _content_mode_value(content_route: Any | None) -> str:
    if content_route is None:
        return ""
    if isinstance(content_route, dict):
        return str(content_route.get("content_mode") or "")
    return str(getattr(content_route, "content_mode", "") or "")


def _candidate_lane(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    decision = candidate.get("decision") if isinstance(candidate.get("decision"), dict) else {}
    return str(decision.get("lane") or "")


def candidate_applicable_by_policy(candidate: dict[str, Any] | None, *, policy: dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    decision = candidate.get("decision") if isinstance(candidate.get("decision"), dict) else {}
    if not decision.get("quality_safe") or not decision.get("semantic_safe"):
        return False
    lane = str(decision.get("lane") or "")
    if lane == CandidateLane.GOAL_MET.value:
        return True
    if lane == CandidateLane.SAFE_NEAR_MISS.value:
        return bool(decision.get("required_drop_met"))
    if lane == CandidateLane.PARTIAL_DIAGNOSTIC.value:
        return close_partial_candidate_allowed(candidate, policy=policy)
    return False


def select_best_applicable_by_policy(
    candidates: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    applicable = [
        row for row in candidates
        if candidate_applicable_by_policy(row, policy=policy)
    ]
    if not applicable:
        return None
    return max(applicable, key=lambda row: tuple((row.get("decision") or {}).get("rank") or ()))


def _prefer_author_stance_frontier(
    best: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    *,
    content_route: Any | None = None,
) -> dict[str, Any]:
    if _content_mode_value(content_route) != "broad_explanatory_essay":
        return best
    if _candidate_lane(best) == CandidateLane.GOAL_MET.value:
        return best
    if str(best.get("strategy") or "") == "scan_author_stance_thesis_reframe":
        return best
    author_candidates = [
        row for row in candidate_rows
        if str(row.get("strategy") or "") == "scan_author_stance_thesis_reframe"
        and _candidate_lane(row) in {CandidateLane.SAFE_NEAR_MISS.value, CandidateLane.GOAL_MET.value}
        and ((row.get("decision") or {}).get("quality_safe"))
        and ((row.get("decision") or {}).get("semantic_safe"))
    ]
    if not author_candidates:
        return best
    preferred = max(author_candidates, key=lambda row: tuple((row.get("decision") or {}).get("rank") or ()))
    if _candidate_lane(preferred) == CandidateLane.GOAL_MET.value:
        return preferred
    preferred["strategy_preferred_over"] = {
        "strategy": best.get("strategy"),
        "strategy_kind": best.get("strategy_kind"),
        "candidate_ai": best.get("candidate_ai"),
        "reason": "broad_explanatory_essay_prefers_author_stance_over_rescue_or_reconstruction",
    }
    return preferred


def select_best_v2_frontier(
    candidate_rows: list[dict[str, Any]],
    *,
    content_route: Any | None = None,
    partial_policy: dict[str, Any] | None = None,
    close_partial_max_gap: float = 2.0,
    composition_partial_max_gap: float = 3.0,
    composition_ai_penalty_max: float = 2.0,
) -> dict[str, Any] | None:
    policy = partial_policy or partial_application_policy(
        content_route,
        close_partial_max_gap=close_partial_max_gap,
        composition_partial_max_gap=composition_partial_max_gap,
    )
    best = select_best_applicable_by_policy(candidate_rows, policy=policy) or select_best_safe_progress_candidate(candidate_rows)
    if not best:
        return None
    best = _prefer_author_stance_frontier(best, candidate_rows, content_route=content_route)
    best_ai = best.get("candidate_ai")
    if not isinstance(best_ai, (int, float)):
        return best
    best_coverage = candidate_patch_coverage(best)
    composition_candidates = [
        row for row in candidate_rows
        if row.get("strategy") == "scan_targeted_composed_full_doc_delta_winners"
        and candidate_patch_coverage(row) >= max(2, best_coverage + 2)
        and isinstance(row.get("candidate_ai"), (int, float))
        and close_partial_candidate_allowed(row, policy=policy)
    ]
    if not composition_candidates:
        return best
    composition = min(composition_candidates, key=lambda row: float(row.get("candidate_ai") or 999.0))
    ai_penalty = float(composition.get("candidate_ai") or 999.0) - float(best_ai)
    if ai_penalty <= float(composition_ai_penalty_max):
        composition["coverage_preferred_over"] = {
            "strategy": best.get("strategy"),
            "paragraph_id": best.get("paragraph_id"),
            "candidate_ai": best.get("candidate_ai"),
            "coverage": best_coverage,
            "ai_penalty": round(ai_penalty, 3),
            "reason": "safe_composition_covers_more_paragraphs_with_bounded_ai_penalty",
        }
        return composition
    return best
