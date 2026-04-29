"""Shared types for criteria modules."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CriterionScore:
    """Result from a single criterion evaluation."""
    name: str                                  # criterion identifier
    value: float                               # 0.0–1.0 normalised score
    label: str                                 # "low" | "medium" | "high"
    details: Dict[str, float] = field(default_factory=dict)
    flagged_excerpts: List[str] = field(default_factory=list)
