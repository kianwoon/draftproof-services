"""Shared runtime budget helpers for rewrite V2."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class RewriteV2RuntimeBudget:
    started_at: float
    max_runtime_seconds: int
    generation_budget_seconds: int

    @property
    def absolute_deadline(self) -> float:
        return self.started_at + max(1, int(self.max_runtime_seconds or 1))

    @property
    def generation_deadline(self) -> float:
        return self.started_at + max(1, int(self.generation_budget_seconds or 1))

    def elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def remaining_seconds(self) -> float:
        return max(0.0, self.absolute_deadline - time.time())

    def can_start(self, required_seconds: float = 0.0) -> bool:
        return self.remaining_seconds() > max(0.0, float(required_seconds or 0.0))

    def skip_reason(self, required_seconds: float = 0.0) -> str:
        return "runtime_budget_exhausted" if not self.can_start(required_seconds) else "not_applicable"

    def to_dict(self) -> dict[str, float | int]:
        return {
            "max_runtime_seconds": int(self.max_runtime_seconds),
            "generation_budget_seconds": int(self.generation_budget_seconds),
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "remaining_seconds": round(self.remaining_seconds(), 3),
        }
