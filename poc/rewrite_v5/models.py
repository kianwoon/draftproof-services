"""Typed containers for V5 controlled reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SectionUnit:
    section_id: str
    heading: str
    text: str
    start_char: int
    end_char: int
    paragraph_count: int
    word_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "paragraph_count": self.paragraph_count,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FactMap:
    section_id: str
    section_role: str
    fixed_facts: tuple[str, ...]
    personal_observations: tuple[str, ...]
    citations: tuple[str, ...]
    protected_terms: tuple[str, ...]
    current_route: tuple[str, ...]
    better_route: tuple[str, ...]
    writing_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_role": self.section_role,
            "fixed_facts": list(self.fixed_facts),
            "personal_observations": list(self.personal_observations),
            "citations": list(self.citations),
            "protected_terms": list(self.protected_terms),
            "current_route": list(self.current_route),
            "better_route": list(self.better_route),
            "writing_issues": list(self.writing_issues),
        }


@dataclass(frozen=True)
class RecompositionVariant:
    variant_id: str
    text: str
    word_count: int
    author_proxy_provenance: list[dict[str, Any]] = field(default_factory=list)
    author_review_items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "text": self.text,
            "word_count": self.word_count,
            "author_proxy_provenance": self.author_proxy_provenance,
            "author_review_items": self.author_review_items,
        }
