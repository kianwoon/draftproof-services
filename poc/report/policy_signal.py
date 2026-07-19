"""Pure derivation of the AI ``llm_patterning`` signal that feeds the additive
grounding/policy composers.

``report/builder.py`` writes ``ai_components``'s ``predictability`` /
``topk_pattern`` / ``topk_pattern_raw`` / ``topk_calibrated_risk`` keys, which
``detect/grounding_diagnosis.py`` re-groups into the ``llm_patterning`` bucket
and ``detect/policy_risk.py`` reads as ``surface_ai_text_signal``. Those keys
must reflect the SAME authoritative AI-likelihood the badge shows.

The bug this module fixes: the override keyed off the EARLY DeBERTa CPU pass
(``authoritative_score.available`` + ``_deb_score``) and ignored the already-
computed, more-authoritative V7 fused/deep-scan value — so a document the badge
called 67%/Critical fed a stale ~5-9 into ``policy_risk`` (shown as 26-34/
Moderate). This module routes the decision through the fused value when the V7
tier-authority fusion actually applied, and otherwise falls back byte-identically
to the legacy early-DeBERTa logic.

Kept as a small, pure, dependency-light module (no ML stack) so it is directly
unit-testable and so ``builder.py`` stays focused.
"""
from __future__ import annotations

from typing import Optional

# Below this AI-likelihood %, the AI text is too thin to count as a policy /
# llm-patterning concern (Turnitin-style green band) -> suppress the
# grounding-diagnosis llm_patterning inputs to 0.0. This is the 0-100-scale
# equivalent of the early-DeBERTa path's 0.20 (0-1 fraction) threshold: the fused
# authority score is 0-100 while ``_deb_score`` is a 0-1 fraction, so
# 0.20 * 100 == 20.0. The two stay in lock-step (units-correct suppression).
LOW_AI_SIGNAL_SUPPRESSION_PCT = 20.0


def policy_signal_pct_for_llm_patterning(
    *,
    deberta_available: bool,
    deberta_score_fraction: Optional[float],
    deberta_pct: Optional[float],
    tier_authority_applied: bool,
    fused_ai_likelihood_pct: Optional[float],
) -> Optional[float]:
    """Return the value to write to ``ai_components``'s
    ``predictability`` / ``topk_pattern`` / ``topk_pattern_raw`` /
    ``topk_calibrated_risk`` keys, or ``None`` to leave them untouched (the
    perplexity fallback).

    Parameters
    ----------
    deberta_available:
        Whether the early DeBERTa CPU pass produced a score
        (``authoritative_score.get("available")``).
    deberta_score_fraction:
        The early DeBERTa ``ai_likelihood_score`` on a 0-1 scale (``_deb_score``),
        or ``None`` when unavailable.
    deberta_pct:
        The early DeBERTa score on a 0-100 scale (``_deb_pct``), or ``None``.
    tier_authority_applied:
        Whether the V7 tier-authority fusion actually applied (flag on AND
        ``tier_authority_status["applied"]``).
    fused_ai_likelihood_pct:
        The fused ``authoritative_ai_likelihood`` on a 0-100 scale — the same
        value the badge shows — or ``None``.

    Semantics
    ---------
    - **Fusion applied**: use the FUSED value (0-100). Below
      ``LOW_AI_SIGNAL_SUPPRESSION_PCT`` suppress to 0.0. This is the fix — the
      llm_patterning inputs track the badge, not the stale early-DeBERTa read.
    - **Fusion not applied** (flag off / below-floor / unavailable / error):
      byte-identical legacy early-DeBERTa logic — ``_deb_score < 0.20`` (0-1)
      suppresses to 0.0, else ``_deb_pct``.
    - **Neither active**: ``None`` (leave the perplexity keys alone).
    """
    if tier_authority_applied and fused_ai_likelihood_pct is not None:
        if float(fused_ai_likelihood_pct) < LOW_AI_SIGNAL_SUPPRESSION_PCT:
            return 0.0
        return round(float(fused_ai_likelihood_pct), 2)
    if deberta_available and deberta_score_fraction is not None and deberta_pct is not None:
        _fraction_threshold = LOW_AI_SIGNAL_SUPPRESSION_PCT / 100.0
        return 0.0 if deberta_score_fraction < _fraction_threshold else deberta_pct
    return None
