"""Draft evolution criterion — detects surface-level rewrites vs substantive revision.

Compares sentence-level structure between submitted text and common rewrite patterns
to estimate whether the text shows evidence of iterative revision (good) or
surface-level paraphrasing (risk indicator).

Current status: stub — returns zero risk pending implementation.
"""

from ._types import CriterionScore


def score(content: str, **kwargs) -> CriterionScore:
    """Score draft evolution signal.

    Returns zero risk until the full cross-draft comparison is implemented.
    The weight in ALL_CRITERIA (0.15) means this contributes nothing until activated.
    """
    return CriterionScore(
        name="draft_evolution",
        value=0.0,
        label="low",
        details={},
        flagged_excerpts=[],
    )
