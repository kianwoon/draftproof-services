"""V7 category scoring — combines the 14 adapter signals plus the fused
``calibrated_detector_score`` into the 4-category Authorship Clarity
Breakdown (student_owned / ai_assisted_polished / ai_paraphrased /
ai_generated_like), per ``docs/draftproof_v7_authorship_clarity_spec.md``
§6 and the v2 tech-spec formulas embedded in ``weights.json``.

No weight/threshold literal lives in this file — every number comes from
``detect_v7/weights.json`` via ``config.py`` accessors (CLAUDE.md
no-hardcode rule). Missing signals are never silently treated as 0: a
missing (``None``) signal contributes 0 to its category's raw weighted sum
AND is recorded in the returned ``missing_signals`` list, so callers know
the raw score is an underestimate.

``degraded`` is stricter than ``missing_signals``: it is True only when a
missing signal's adapter status says the signal is BUILT but its inputs were
absent for this run (``"unavailable"``, or an unknown/absent status —
conservative). Signals that are absent BY DESIGN (adapter status
``"not_implemented"`` — the unbuilt Phase-1C/2 rows — or
``"unavailable_no_comparison_text"``) still land in ``missing_signals`` but
do NOT set ``degraded``: they are roadmap facts, identical for every
document, and counting them made ``degraded`` a constant True that the
degraded_display / primary_category_reliable guards could never use
(measured 2026-07-06: 100% of both human and AI docs degraded).
"""
from __future__ import annotations

from typing import Any, Optional

from . import config

_CATEGORY_NAMES: tuple[str, ...] = (
    "student_owned",
    "ai_assisted_polished",
    "ai_paraphrased",
    "ai_generated_like",
)


def _midrange(value: float, low: float, high: float) -> float:
    """Triangular plateau shape: 1.0 for value in [low, high], tapering
    linearly to 0.0 at the band's midpoint-distance outside the band.

    Exact formula:
    - low <= value <= high            -> 1.0 (full plateau inside the band)
    - value < low                     -> max(0.0, value / low) if low > 0
                                          else 0.0 for value < low == 0
    - value > high                    -> max(0.0, (1.0 - value) / (1.0 - high))
                                          if high < 1.0 else 0.0 for value > high == 1
    This tapers linearly from the plateau edge down to 0.0 at the axis
    endpoint (0.0 or 1.0), giving a trapezoid/triangular shape centered on
    the band rather than a hard cutoff — values just outside the band still
    score partially, values far outside score near 0.
    """
    if low <= value <= high:
        return 1.0
    if value < low:
        if low <= 0.0:
            return 0.0
        return max(0.0, value / low)
    # value > high
    if high >= 1.0:
        return 0.0
    return max(0.0, (1.0 - value) / (1.0 - high))


def _resolve_signal_value(
    signal_name: str,
    v7_signals: dict[str, Any],
    calibrated_detector_score: float,
) -> Optional[float]:
    if signal_name == "calibrated_detector_score":
        return calibrated_detector_score
    return v7_signals.get(signal_name)


def _apply_direction(
    signal_name: str,
    value: float,
    direction: str,
    band: dict[str, float],
) -> float:
    if direction == "direct":
        return value
    if direction == "inverse":
        return 1.0 - value
    if direction == "midrange":
        return _midrange(value, band["low"], band["high"])
    raise ValueError(
        f"Unknown direction {direction!r} for signal {signal_name!r} — "
        f"expected 'direct', 'inverse', or 'midrange'."
    )


def _score_category_raw(
    category: str,
    v7_signals: dict[str, Any],
    calibrated_detector_score: float,
    has_comparison_text: bool,
    band: dict[str, float],
    missing_signals: set[str],
) -> float:
    entries = config.get_category_weights(category, has_comparison_text)
    total = 0.0
    for signal_name, weight, direction in entries:
        value = _resolve_signal_value(signal_name, v7_signals, calibrated_detector_score)
        if value is None:
            missing_signals.add(signal_name)
            continue
        resolved = _apply_direction(signal_name, value, direction, band)
        total += weight * resolved
    return total


def score_paragraph(
    v7_signals: dict[str, Any],
    calibrated_detector_score: float,
    has_comparison_text: bool = False,
    esl_score: Optional[float] = None,
) -> dict[str, Any]:
    """Score one paragraph's Authorship Clarity Breakdown.

    Parameters
    ----------
    v7_signals: output of ``signal_adapter.adapt_paragraph_signals`` (14
        signal keys + ``signal_status``).
    calibrated_detector_score: output of
        ``detector_fusion.compute_calibrated_detector_score``.
    has_comparison_text: selects the ai_paraphrased with/without-comparison
        weight variant (spec D9).
    esl_score: optional document/paragraph-level ESL-likelihood score in
        [0, 1]; when provided and above the configured threshold, the ESL
        guard (weights.json ``esl_guard``) damps ``ai_generated_like`` and
        may cap confidence.

    Returns
    -------
    dict with the 4 normalized category shares (sum to 1.0), plus
    ``confidence`` ("low" or None), ``presentation`` ("mixed_signals" or
    None), ``primary_category``, ``degraded`` (bool), and
    ``missing_signals`` (list[str]).
    """
    band = config.get_ai_assisted_polished_band()
    missing_signals: set[str] = set()

    raw_scores: dict[str, float] = {}
    for category in _CATEGORY_NAMES:
        raw_scores[category] = _score_category_raw(
            category,
            v7_signals,
            calibrated_detector_score,
            has_comparison_text,
            band,
            missing_signals,
        )

    # Degraded accounting: only BUILT-but-absent signals count (see module
    # docstring). Status values come from signal_adapter; an unknown/absent
    # status is treated as degrading (conservative — never assume health).
    signal_status = v7_signals.get("signal_status") or {}
    _BY_DESIGN_ABSENT = ("not_implemented", "unavailable_no_comparison_text")
    degraded = any(
        signal_status.get(name) not in _BY_DESIGN_ABSENT
        for name in missing_signals
        if name != "calibrated_detector_score"
    )

    # --- ESL guard: damp ai_generated_like's raw share BEFORE normalization ---
    esl_guard_cfg = config.get_esl_guard_config()
    esl_triggered = esl_score is not None and esl_score > esl_guard_cfg["esl_high_threshold"]
    if esl_triggered:
        raw_scores["ai_generated_like"] *= esl_guard_cfg["ai_generated_damping"]

    total_raw = sum(raw_scores.values())
    if total_raw <= 0.0:
        equal_share = 1.0 / len(_CATEGORY_NAMES)
        normalized = {category: equal_share for category in _CATEGORY_NAMES}
        degraded = True
    else:
        normalized = {
            category: raw_scores[category] / total_raw for category in _CATEGORY_NAMES
        }

    # --- flatness check (confidence / presentation) ---
    flatness_cfg = config.get_flatness_thresholds()
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    primary_category, top_share = ranked[0]
    second_share = ranked[1][1] if len(ranked) > 1 else 0.0

    confidence: Optional[str] = None
    if (top_share - second_share) < flatness_cfg["confidence_low_gap"]:
        confidence = "low"

    presentation: Optional[str] = None
    if top_share < flatness_cfg["mixed_signals_max_category"]:
        presentation = "mixed_signals"

    # --- ESL guard: confidence cap ---
    # Rule (weights.json _notes.esl_guard): disagreement above
    # detector_disagreement_confidence_cap_threshold ALONE caps confidence.
    # esl_score above esl_disagreement_cotrigger_threshold AND the
    # disagreement trigger firing together FORCE confidence to
    # confidence_cap_value regardless of other confidence signals.
    detector_disagreement = v7_signals.get("detector_disagreement")
    disagreement_triggered = (
        detector_disagreement is not None
        and detector_disagreement > esl_guard_cfg["detector_disagreement_confidence_cap_threshold"]
    )
    if disagreement_triggered:
        confidence = esl_guard_cfg["confidence_cap_value"]
    if (
        esl_score is not None
        and esl_score > esl_guard_cfg["esl_disagreement_cotrigger_threshold"]
        and disagreement_triggered
    ):
        confidence = esl_guard_cfg["confidence_cap_value"]

    return {
        "student_owned": normalized["student_owned"],
        "ai_assisted_polished": normalized["ai_assisted_polished"],
        "ai_paraphrased": normalized["ai_paraphrased"],
        "ai_generated_like": normalized["ai_generated_like"],
        "confidence": confidence,
        "presentation": presentation,
        "primary_category": primary_category,
        "degraded": degraded,
        "missing_signals": sorted(missing_signals),
    }
