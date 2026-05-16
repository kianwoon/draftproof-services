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
    mitigation_strategy: dict[str, Any] = field(default_factory=dict)
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
            "mitigation_strategy": self.mitigation_strategy,
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


@dataclass(frozen=True)
class ClusterRepairUnit:
    cluster_id: str
    start_sentence: int
    end_sentence: int
    start_char: int
    end_char: int
    text: str
    before_context: str
    after_context: str
    sentence_count: int
    word_count: int
    risk_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def group_id(self) -> str:
        return self.cluster_id

    @property
    def unit_id(self) -> str:
        return self.cluster_id

    @property
    def source_text(self) -> str:
        return self.text

    @property
    def operation(self) -> str:
        return "bounded_cluster_patch"

    @property
    def targets(self) -> tuple[dict[str, Any], ...]:
        return ({
            "target_id": self.cluster_id,
            "scope_level": "sentence_cluster",
            "dominant_drivers": [
                {"key": "unsafe_cluster_density", "score": self.risk_score},
            ],
        },)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "start_sentence": self.start_sentence,
            "end_sentence": self.end_sentence,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
            "before_context": self.before_context,
            "after_context": self.after_context,
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "risk_score": self.risk_score,
            "metadata": self.metadata,
        }
