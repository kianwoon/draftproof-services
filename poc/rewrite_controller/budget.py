"""Global runtime budget for rewrite controller phases."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def resolve_global_rewrite_seconds(
    *,
    legacy_seconds: float = 0.0,
    controller_policy_seconds: float = 0.0,
    env_seconds: float | None = None,
    default_seconds: float = 90.0,
) -> float:
    """Resolve the run-level controller budget.

    `RewriteConfig.max_rewrite_seconds` is a legacy pre-controller setting.  It
    can be lower than the policy needed by the modern multi-stage AI mitigation
    controller, so it must not silently starve downstream phases.
    """

    candidates = [
        float(default_seconds or 0.0),
        float(legacy_seconds or 0.0),
        float(controller_policy_seconds or 0.0),
    ]
    if env_seconds is not None:
        candidates.append(float(env_seconds or 0.0))
    return max(candidates)


def post_ai_search_reserve_seconds(word_count: int | float) -> float:
    """Time reserve for post-selection repair/density phases.

    The first AI-search phase is allowed to spend most of the runtime, but not
    all of it.  These reserves keep at least one downstream controller phase
    alive after the initial LLM candidate set is scanned.
    """

    words = max(0, int(word_count or 0))
    if words <= 700:
        return 35.0
    if words <= 1800:
        return 55.0
    return 75.0


def cap_phase_seconds_for_reserve(
    *,
    max_seconds: float,
    remaining_seconds: float,
    reserve_seconds: float,
    min_phase_seconds: float = 20.0,
) -> float:
    """Cap one phase so later phases keep a usable time reserve."""

    phase_limit = max(0.0, float(max_seconds or 0.0))
    remaining = max(0.0, float(remaining_seconds or 0.0))
    reserve = max(0.0, float(reserve_seconds or 0.0))
    minimum = max(0.0, float(min_phase_seconds or 0.0))
    if phase_limit <= 0.0 or remaining <= 0.0 or reserve <= 0.0:
        return phase_limit
    if remaining > reserve:
        capped = remaining - reserve
    else:
        # Budget is already tight. Split the remaining time instead of letting
        # the current phase consume it all.
        capped = remaining * 0.55
    if capped >= minimum:
        return min(phase_limit, capped)
    if remaining > minimum + reserve:
        return min(phase_limit, minimum)
    return min(phase_limit, capped)


@dataclass
class RewriteRunBudget:
    """One shared budget across all post-rewrite phases.

    The older pipeline treated every phase as if it owned a fresh local budget.
    This object makes the configured rewrite timeout meaningful across the
    whole controller path.
    """

    max_seconds: float
    max_scans: int = 0
    max_llm_calls: int = 0
    started_at: float = field(default_factory=time.time)
    scans_used: int = 0
    llm_calls_used: int = 0
    stages: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def elapsed(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def remaining_seconds(self) -> float:
        if self.max_seconds <= 0:
            return 0.0
        return max(0.0, self.max_seconds - self.elapsed())

    def remaining_scans(self) -> int | None:
        if self.max_scans <= 0:
            return None
        return max(0, self.max_scans - self.scans_used)

    def remaining_llm_calls(self) -> int | None:
        if self.max_llm_calls <= 0:
            return None
        return max(0, self.max_llm_calls - self.llm_calls_used)

    def can_run(self, *, min_seconds: float = 0.0, min_scans: int = 0, min_llm_calls: int = 0) -> bool:
        if self.max_seconds > 0 and self.remaining_seconds() < max(0.0, float(min_seconds)):
            return False
        remaining_scans = self.remaining_scans()
        if remaining_scans is not None and remaining_scans < max(0, int(min_scans)):
            return False
        remaining_llm = self.remaining_llm_calls()
        if remaining_llm is not None and remaining_llm < max(0, int(min_llm_calls)):
            return False
        return True

    def skip_reason(self, stage: str, *, min_seconds: float = 0.0, min_scans: int = 0, min_llm_calls: int = 0) -> dict:
        reason = "global_budget_exhausted"
        if self.max_seconds > 0 and self.remaining_seconds() < max(0.0, float(min_seconds)):
            reason = "global_time_budget_exhausted"
        elif self.remaining_scans() is not None and self.remaining_scans() < max(0, int(min_scans)):
            reason = "global_scan_budget_exhausted"
        elif self.remaining_llm_calls() is not None and self.remaining_llm_calls() < max(0, int(min_llm_calls)):
            reason = "global_llm_budget_exhausted"
        payload = {
            "stage": stage,
            "reason": reason,
            "required": {
                "min_seconds": round(float(min_seconds), 3),
                "min_scans": int(min_scans),
                "min_llm_calls": int(min_llm_calls),
            },
            "budget": self.snapshot(include_skipped=False),
        }
        self.skipped.append(payload)
        return payload

    def record_stage(self, stage: str, *, seconds: float = 0.0, scans: int | None = None, llm_calls: int | None = None) -> None:
        scans_used = max(0, int(scans or 0))
        llm_used = max(0, int(llm_calls or 0))
        self.scans_used += scans_used
        self.llm_calls_used += llm_used
        self.stages.append({
            "stage": stage,
            "seconds": round(float(seconds or 0.0), 3),
            "scans": scans_used,
            "llm_calls": llm_used,
            "elapsed_total": round(self.elapsed(), 3),
            "remaining_seconds": round(self.remaining_seconds(), 3),
        })

    def snapshot(self, *, include_skipped: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "max_seconds": round(float(self.max_seconds or 0.0), 3),
            "elapsed_seconds": round(self.elapsed(), 3),
            "remaining_seconds": round(self.remaining_seconds(), 3),
            "max_scans": int(self.max_scans or 0),
            "scans_used": int(self.scans_used),
            "remaining_scans": self.remaining_scans(),
            "max_llm_calls": int(self.max_llm_calls or 0),
            "llm_calls_used": int(self.llm_calls_used),
            "remaining_llm_calls": self.remaining_llm_calls(),
            "stages": list(self.stages),
        }
        if include_skipped:
            payload["skipped"] = list(self.skipped)
        return payload

    def summary(self) -> dict[str, Any]:
        return self.snapshot(include_skipped=True)
