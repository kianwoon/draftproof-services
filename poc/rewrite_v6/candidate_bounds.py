"""Word-count bounds for a per-paragraph rewrite candidate.

Split out of direct_rewrite.py to keep that file under the repo's 1500-LOC cap. Bounds are ratios
of the source paragraph's word count, env-tunable so a length-guard incident doesn't need a
redeploy. `word_bounds_status` distinguishes the two failure directions: "under" is stub territory
(too little content -- direct_rewrite.py always hard-rejects it) while "over" means the writer
added grounding content beyond the usual ratio -- direct_rewrite.py's `_clean_candidate` ships that
on the final attempt with a review flag instead of falling back to source_preserved (rewrite
objective: annotate, don't suppress).
"""
from __future__ import annotations

import os

from .text import Paragraph


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def candidate_word_bounds(paragraph: Paragraph) -> tuple[int, int]:
    source_words = max(1, len(str(paragraph.text or "").split()))
    min_ratio = _float_env("DRAFTPROOF_V6_PARAGRAPH_MIN_WORD_RATIO", 0.55)
    max_ratio = _float_env("DRAFTPROOF_V6_PARAGRAPH_MAX_WORD_RATIO", 1.75)
    min_ratio = max(0.25, min(0.95, min_ratio))
    max_ratio = max(1.0, min(2.5, max_ratio))
    return max(8, int(source_words * min_ratio)), max(12, int(source_words * max_ratio) + 1)


def word_bounds_status(candidate: str, paragraph: Paragraph) -> str:
    """"under" (stub) / "over" (added content beyond the ratio) / "ok"."""
    words = len(str(candidate or "").split())
    lower, upper = candidate_word_bounds(paragraph)
    return "under" if words < lower else "over" if words > upper else "ok"
