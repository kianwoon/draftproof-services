"""Mode-aware partial application policy for rewrite V2."""

from __future__ import annotations

import os
from typing import Any

from .selection import CandidateLane


DEFAULT_CLOSE_PARTIAL_MODES = {
    "broad_explanatory_essay",
    "generic_expository",
    "short_text",
}


def _content_mode(content_route: Any | None) -> str:
    if content_route is None:
        return ""
    if isinstance(content_route, dict):
        return str(content_route.get("content_mode") or "")
    return str(getattr(content_route, "content_mode", "") or "")


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _configured_modes() -> set[str]:
    raw = os.environ.get("DRAFTPROOF_REWRITE_V2_CLOSE_PARTIAL_MODES")
    if raw is None:
        return set(DEFAULT_CLOSE_PARTIAL_MODES)
    return {item.strip() for item in raw.split(",") if item.strip()}


def partial_application_policy(
    content_route: Any | None,
    *,
    close_partial_max_gap: float,
    composition_partial_max_gap: float,
) -> dict[str, Any]:
    """Return the enforced policy for applying close partial candidates.

    Partial candidates are still diagnostics, not success. Applying them is only
    acceptable for document shapes where a bounded imperfect improvement is less
    risky than preserving the original. Citation-heavy, regulated, technical,
    list, quote-heavy, and personal modes default to preservation unless they
    reach SAFE_NEAR_MISS or GOAL_MET. Short text is allowed because its V2 route
    already caps candidate spend and often needs bounded expansion rather than a
    full-document rewrite.
    """
    mode = _content_mode(content_route) or "unknown"
    global_enabled = _env_enabled("DRAFTPROOF_REWRITE_V2_APPLY_CLOSE_PARTIAL", True)
    allowed_modes = _configured_modes()
    mode_allowed = mode in allowed_modes
    apply_close_partial = bool(global_enabled and mode_allowed)
    if not global_enabled:
        reason = "close_partial_globally_disabled"
    elif mode_allowed:
        reason = "content_mode_allows_bounded_partial_application"
    else:
        reason = "content_mode_requires_strict_or_near_miss_candidate"
    return {
        "version": "rewrite_v2_partial_application_policy_v1",
        "content_mode": mode,
        "apply_close_partial": apply_close_partial,
        "allowed_modes": sorted(allowed_modes),
        "blocked_by_mode": bool(global_enabled and not mode_allowed),
        "close_partial_max_gap": float(close_partial_max_gap),
        "composition_partial_max_gap": float(composition_partial_max_gap),
        "reason": reason,
    }


def close_partial_candidate_allowed(
    candidate: dict[str, Any] | None,
    *,
    policy: dict[str, Any],
) -> bool:
    if not isinstance(candidate, dict) or not policy.get("apply_close_partial"):
        return False
    decision = candidate.get("decision") if isinstance(candidate.get("decision"), dict) else {}
    if decision.get("lane") != CandidateLane.PARTIAL_DIAGNOSTIC.value:
        return False
    if not decision.get("quality_safe") or not decision.get("semantic_safe"):
        return False
    gap = decision.get("ai_target_gap")
    if not isinstance(gap, (int, float)):
        return False
    close_gap = float(policy.get("close_partial_max_gap") or 0.0)
    composition_gap = float(policy.get("composition_partial_max_gap") or close_gap)
    patch_coverage = _candidate_patch_coverage(candidate)
    return float(gap) <= close_gap or (patch_coverage >= 2 and float(gap) <= composition_gap)


def _candidate_patch_coverage(candidate: dict[str, Any] | None) -> int:
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
