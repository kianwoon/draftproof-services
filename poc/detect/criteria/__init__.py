"""AI-generation criteria — one module per detection criterion.

Each criterion module exposes a ``score(content, **kwargs) -> CriterionScore``
function that is pure and stateless.  The composite AIGenerationSignalDetector
in ``detect/ai_generation.py`` calls every criterion and combines results with
configurable weights.
"""

from ._types import CriterionScore

from .surprisal import score as surprisal_score
from .topk_ratio import score as topk_ratio_score
from .burstiness import score as burstiness_score
from .repetitive_structure import score as repetitive_structure_score
from .generic_phrases import score as generic_phrases_score
from .specificity import score as specificity_score
from .polish_vs_grounding import score as polish_vs_grounding_score
from .paragraph_uniformity import score as paragraph_uniformity_score
from .style_shift import score as style_shift_score


# Ordered list used by the composite classifier to iterate criteria.
ALL_CRITERIA = [
    ("low_surprisal", surprisal_score, 0.20),
    ("topk_predictability", topk_ratio_score, 0.15),
    ("low_burstiness", burstiness_score, 0.15),
    ("generic_phrase_density", generic_phrases_score, 0.15),
    ("low_specificity", specificity_score, 0.15),
    ("repetitive_structure", repetitive_structure_score, 0.10),
    ("style_shift", style_shift_score, 0.05),
    ("citation_grounding_gap", polish_vs_grounding_score, 0.05),
    # paragraph_uniformity is diagnostic only (weight 0) until StructuralDetector
    ("paragraph_uniformity", paragraph_uniformity_score, 0.00),
]

__all__ = [
    "CriterionScore",
    "ALL_CRITERIA",
    "surprisal_score",
    "topk_ratio_score",
    "burstiness_score",
    "repetitive_structure_score",
    "generic_phrases_score",
    "specificity_score",
    "polish_vs_grounding_score",
    "paragraph_uniformity_score",
    "style_shift_score",
]
