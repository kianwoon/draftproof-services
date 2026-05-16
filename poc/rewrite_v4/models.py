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
    repair_mode: str = "source_preserving_repair"
    tutor_diagnosis: str = ""
    student_explanation: str = ""
    source_examples: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    repair_assignment: str = ""
    coverage_hint: str = "paragraph"
    rejected_tasks: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    parse_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalizer": self.normalizer,
            "repair_mode": self.repair_mode,
            "paragraph_role": self.paragraph_role,
            "tutor_diagnosis": self.tutor_diagnosis,
            "student_explanation": self.student_explanation,
            "source_examples": list(self.source_examples),
            "repair_assignment": self.repair_assignment,
            "coverage_hint": self.coverage_hint,
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
