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
from .source_grounding import score as source_grounding_score
from .draft_evolution import score as draft_evolution_score
from .structural_reuse import score as structural_reuse_score


# Ordered list used by the composite classifier to iterate criteria.
#
# SCOPE NOTE (2026-07-10): tuning these weights changes AIGenerationSignalDetector's
# own composite (per-finding metadata + a rewrite-pipeline fallback only) — it does
# NOT affect the primary user-facing ai_risk_badge score/tier, which is computed by
# an independent formula in detect/layer3_scoring.py:Layer3Scorer. See
# ai_generation.py's module docstring for the full consumer list.
#
# Weight philosophy (aligned with evidence-strength table):
#   Major signals (Strong):   polished_ungrounded, draft_evolution, structural_reuse
#   Moderate signals:         low_specificity, paragraph_uniformity
#   Supporting only (Weak):   surprisal, topk, burstiness, generic_phrases,
#                             repetitive_structure, style_shift
#   Diagnostic only:          source_grounding (used by authorship concern scoring)
ALL_CRITERIA = [
    # ── Supporting only (Weak–Moderate) ──
    ("low_surprisal", surprisal_score, 0.10),
    ("topk_predictability", topk_ratio_score, 0.10),
    ("low_burstiness", burstiness_score, 0.08),
    ("repetitive_structure", repetitive_structure_score, 0.08),
    ("generic_phrase_density", generic_phrases_score, 0.12),
    # ── Moderate (Review trigger) ──
    ("low_specificity", specificity_score, 0.15),
    ("paragraph_uniformity", paragraph_uniformity_score, 0.07),
    # ── Needs context ──
    ("style_shift", style_shift_score, 0.05),
    # ── Major signals (Strong) ──
    ("citation_grounding_gap", polish_vs_grounding_score, 0.20),
    ("draft_evolution", draft_evolution_score, 0.15),
    ("structural_reuse", structural_reuse_score, 0.10),
    # ── Diagnostic only ──
    ("source_grounding", source_grounding_score, 0.00),
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
    "source_grounding_score",
    "draft_evolution_score",
    "structural_reuse_score",
]
