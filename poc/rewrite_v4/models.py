"""Small typed containers for the V4 experiment path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RepairBrief:
    normalizer: str
    paragraph_role: str
    repair_tasks: tuple[str, ...]
    constraints: tuple[str, ...]
    avoid: tuple[str, ...]
    rejected_tasks: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    parse_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalizer": self.normalizer,
            "paragraph_role": self.paragraph_role,
            "repair_tasks": list(self.repair_tasks),
            "constraints": list(self.constraints),
            "avoid": list(self.avoid),
            "rejected_tasks": list(self.rejected_tasks),
            "parse_diagnostics": self.parse_diagnostics,
        }


@dataclass(frozen=True)
class CandidateVariant:
    variant_id: str
    text: str
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "text": self.text,
            "word_count": self.word_count,
        }
