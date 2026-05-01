"""Structural reuse criterion — detects recycled paragraph/sentence structures.

Identifies when text reuses organizational templates or structural patterns
commonly produced by AI writing assistants (e.g., repeated "Topic sentence →
evidence → analysis" scaffolding with no variation).

Current status: stub — returns zero risk pending implementation.
"""

from ._types import CriterionScore


def score(content: str, **kwargs) -> CriterionScore:
    """Score structural reuse signal.

    Returns zero risk until the structural pattern analysis is implemented.
    The weight in ALL_CRITERIA (0.10) means this contributes nothing until activated.
    """
    return CriterionScore(
        name="structural_reuse",
        value=0.0,
        label="low",
        details={},
        flagged_excerpts=[],
    )
