"""Deterministic residual-pattern detectors for the QC reviewer (no LLM, no network).

The per-paragraph writer (direct_rewrite) is blind across paragraphs, so it can trade one AI
signal (generic assertion) for another (repetitive structure) -- e.g. 7/8 paragraphs opening
"In my classroom". These pure functions measure such patterns across the FULL rewritten document
and return evidence the QC reviewer must fix. They never edit text; they only report.

Content-agnostic structural/linguistic measures only (frame repetition, sentence-length variance,
closed-class connectives) -- not a banned-phrase or domain list. Aligns with the existing
_POLARITY_MARKERS / _CITATION_MARKERS closed-set precedent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ResidualIssue:
    rule: str                                   # e.g. "opener_monoculture"
    trick_ids: list[int]                        # guideline numbers, e.g. [19, 2]
    evidence: str                               # human-readable, e.g. "paragraphs 1,2,3,7 open 'In my'"
    target_sentences: list[str] = field(default_factory=list)  # exact sentences QC must fix


def detect_residual_patterns(text: str) -> list[ResidualIssue]:
    """Run every detector over the full document; return all fired issues (may be empty)."""
    issues: list[ResidualIssue] = []
    # Detectors are added in later sub-tasks.
    return issues
