"""Grouped external-detector proxy scoring.

This module is intentionally separate from Layer 3 scoring: it powers the
external-facing estimate only and keeps DraftProof's own AI-likelihood badge
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Iterable


# v2: down-weight probability_shape (was 0.35, the largest). Across a corpus of
# scans it is a flat-to-INVERTED fluent-text floor — it scores as high (or higher)
# on green/human docs as on the one genuinely-AI red doc, so it carries no
# AI-discriminating information and only inflated the external estimate on polished
# human writing (the documented intrinsic over-flag floor). Weight shifts to the
# signals that actually separate human from AI and are user-actionable: writing
# patterns (generic/unanchored claims) and grounding. This is a known-false-positive
# floor removal — NOT validated against third-party detectors (N=1 Turnitin sample).
# Additive-only: does not affect tier, ai_likelihood, or any rewrite gate.
# v3 (2026-06-15): add a grounding-strength discount (EXTERNAL_GROUNDING_DISCOUNT_ALPHA). A
# well-grounded document (high author/context/citation strength) is less likely machine-generated,
# so the proxy is scaled down proportional to grounding strength. Calibrated α=0.30 on the labeled
# corpus: pulls genuine well-grounded human docs out of the "elevated" band (e.g. a real
# practitioner's grounded essay 25.9 -> 19.9, "low") while leaking ~1/17 AI docs and barely moving
# AI-vs-human AUC (0.898 -> 0.878). HYPOTHESIS tuned on a SMALL (32-doc) set — revisit α when the
# labeled corpus grows; do NOT treat 0.30 as validated.
MODEL_VERSION = "external_grouped_v3"
GROUP_WEIGHTS = {
    "probability_shape_risk": 0.15,
    "detector_agreement_risk": 0.25,
    "writing_pattern_risk": 0.40,
    "grounding_gap_risk": 0.20,
}

PROBABILITY_SIGNAL_WEIGHTS = {
    "nll_perplexity_risk": 8.0,
    "token_rank_alignment": 6.0,
    "topk_concentration": 5.0,
    "entropy_low_uncertainty": 5.0,
    "burstiness_risk": 7.0,
    "sentence_probability_uniformity": 4.0,
}

DETECTOR_SIGNAL_WEIGHTS = {
    "fast_detectgpt_curvature": 10.0,
    "fine_tuned_classifier": 7.0,
    "cross_model_agreement": 3.0,
}

WRITING_SIGNAL_WEIGHTS = {
    "generic_opener_broad_claim": 5.0,
    "template_phrase_density": 5.0,
    # v2: predictable_paragraph_route + packed_list_compression are the strongest
    # human/AI discriminators in this group, so they absorb the weight pulled from
    # overpolish_register_mismatch.
    "predictable_paragraph_route": 6.5,
    "sentence_rhythm_regularity": 4.0,
    "packed_list_compression": 5.5,
    "repetition_structural_reuse": 3.0,
    # v2: overpolish penalizes polished register, but polished prose is also how
    # skilled HUMANS write — it reads ~47 on green/human docs and barely climbs on
    # the AI doc (weakest discriminator in the group). Demoted 4.0 -> 1.0 so the
    # group's higher weight (0.40) does not amplify this false-positive floor.
    "overpolish_register_mismatch": 1.0,
}

GROUNDING_SIGNAL_WEIGHTS = {
    "author_anchor_strength": 5.0,
    "context_anchor_strength": 5.0,
    "source_citation_integrity": 5.0,
}

HIGH_BAND_MIN = 50.0
ELEVATED_BAND_MIN = 20.0
STRONG_GROUNDING_MIN = 65.0
WEAK_DETECTOR_AVAILABLE_MAX = 1
LOW_COVERAGE_MIN = 0.45
HIGH_CAP_SCORE = 49.9
# Grounding-strength discount (v3): final proxy *= (1 - alpha * grounding_strength/100).
# alpha=0.30 calibrated on the 32-doc labeled corpus (see MODEL_VERSION note). Single source of
# truth; set 0.0 to disable. Tunable hypothesis, NOT third-party-validated.
EXTERNAL_GROUNDING_DISCOUNT_ALPHA = 0.30


@dataclass(frozen=True)
class _Signal:
    key: str
    group: str
    weight: float
    value: float | None
    direction: str
    available: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {
            "key": self.key,
            "group": self.group,
            "weight": self.weight,
            "direction": self.direction,
            "available": self.available,
        }
        if self.available and self.value is not None:
            out["value"] = round(float(self.value), 3)
        if self.note:
            out["note"] = self.note
        return out


def _percent(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if math.isnan(numeric):
        return None
    if abs(numeric) <= 1.0:
        numeric *= 100.0
    return max(0.0, min(100.0, numeric))


def _risk01(value: Any) -> float | None:
    pct = _percent(value)
    return None if pct is None else pct / 100.0


def _criterion_value(criterion_scores: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(criterion_scores, dict) or key not in criterion_scores:
        return None
    score = criterion_scores[key]
    if hasattr(score, "value"):
        return _percent(getattr(score, "value"))
    if isinstance(score, dict):
        return _percent(score.get("value"))
    return None


def _sentences_rows(sentences: Iterable[Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sentences or []:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _mean_numeric(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
                break
    if not values:
        return None
    return sum(values) / len(values)


def _sentence_uniformity_risk(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        value = row.get("avg_surprisal")
        if not isinstance(value, (int, float)):
            value = row.get("risk")
        if isinstance(value, (int, float)):
            values.append(float(value))
    if len(values) < 3:
        return None
    mean = statistics.mean(values)
    if mean <= 0:
        return None
    cv = statistics.pstdev(values) / mean
    return max(0.0, min(100.0, (1.0 - min(cv, 0.75) / 0.75) * 100.0))


def _surprisal_to_risk(avg_surprisal: float | None) -> float | None:
    if avg_surprisal is None:
        return None
    return max(0.0, min(100.0, (4.5 - float(avg_surprisal)) / 3.5 * 100.0))


def _avg_probability_to_risk(avg_probability: float | None) -> float | None:
    if avg_probability is None:
        return None
    pct = _percent(avg_probability)
    if pct is None:
        return None
    return max(0.0, min(100.0, pct * 4.0))


def _group_score(signals: list[_Signal], group: str) -> tuple[float, int, int]:
    rows = [s for s in signals if s.group == group]
    available = [s for s in rows if s.available and s.value is not None]
    total = sum(s.weight for s in available)
    if total <= 0:
        return 0.0, 0, len(rows)
    score = sum(float(s.value or 0.0) * s.weight for s in available) / total
    return max(0.0, min(100.0, score)), len(available), len(rows)


def _band(score: float) -> str:
    if score >= HIGH_BAND_MIN:
        return "high"
    if score >= ELEVATED_BAND_MIN:
        return "elevated"
    return "low"


def _confidence(available: int, total: int, group_available: int) -> str:
    coverage = available / max(total, 1)
    if coverage >= 0.70 and group_available >= 3:
        return "high"
    if coverage >= LOW_COVERAGE_MIN and group_available >= 2:
        return "medium"
    return "low"


def grouped_external_score_from_groups(
    *,
    probability_shape_risk: float,
    detector_agreement_risk: float,
    writing_pattern_risk: float,
    grounding_gap_risk: float,
) -> float:
    """Compute the public grouped formula from already-normalized group scores."""
    return round(
        GROUP_WEIGHTS["probability_shape_risk"] * max(0.0, min(100.0, probability_shape_risk))
        + GROUP_WEIGHTS["detector_agreement_risk"] * max(0.0, min(100.0, detector_agreement_risk))
        + GROUP_WEIGHTS["writing_pattern_risk"] * max(0.0, min(100.0, writing_pattern_risk))
        + GROUP_WEIGHTS["grounding_gap_risk"] * max(0.0, min(100.0, grounding_gap_risk)),
        3,
    )


def estimate_external_grouped_score(
    *,
    sentences: Iterable[Any] | None = None,
    ai_components: dict[str, Any] | None = None,
    writing_components: dict[str, Any] | None = None,
    transformation_features: dict[str, Any] | None = None,
    criterion_scores: dict[str, Any] | None = None,
    legacy_segment_fraction: dict[str, Any] | None = None,
    legacy_likelihood: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the external-detector proxy estimate in the existing badge shape."""
    rows = _sentences_rows(sentences)
    ai = ai_components or {}
    writing = writing_components or {}
    features = transformation_features or {}

    avg_surprisal = _mean_numeric(rows, "avg_surprisal")
    avg_probability = _mean_numeric(rows, "avg_probability")
    avg_top10 = _mean_numeric(rows, "top10_ratio", "top10")
    avg_top50 = _mean_numeric(rows, "top50_ratio", "top50")
    topk_raw = _percent(ai.get("topk_pattern_raw", ai.get("topk_pattern")))
    topk_calibrated = _percent(ai.get("topk_calibrated_risk"))

    nll_value = _surprisal_to_risk(avg_surprisal)
    if nll_value is None:
        nll_value = _percent(ai.get("predictability"))
    token_rank_value = _percent(avg_top50)
    if token_rank_value is None:
        token_rank_value = topk_raw
    topk_value = _percent(avg_top10)
    if topk_value is None:
        topk_value = topk_raw
    burstiness_value = _percent(ai.get("burstiness_risk"))
    if burstiness_value is None:
        burstiness_value = _criterion_value(criterion_scores, "low_burstiness")
    sentence_uniformity_value = _sentence_uniformity_risk(rows)
    cross_agreement_value = _percent(features.get("signal_agreement_score"))
    if cross_agreement_value is None:
        cross_agreement_value = _percent(ai.get("ai_likelihood"))
    template_value = _percent(ai.get("generic_phrase_density"))
    if template_value is None:
        template_value = _criterion_value(criterion_scores, "generic_phrase_density")
    rhythm_value = _percent(writing.get("paragraph_uniformity_risk"))
    if rhythm_value is None:
        rhythm_value = _percent(ai.get("repeated_sentence_structure_risk"))
    reuse_value = _percent(ai.get("repeated_sentence_structure_risk"))
    if reuse_value is None:
        reuse_value = _criterion_value(criterion_scores, "structural_reuse")
    overpolish_value = _percent(features.get("rewrite_smoothness"))
    if overpolish_value is None:
        overpolish_value = _criterion_value(criterion_scores, "citation_grounding_gap")

    signals = [
        _Signal(
            "nll_perplexity_risk", "probability_shape_risk",
            PROBABILITY_SIGNAL_WEIGHTS["nll_perplexity_risk"],
            nll_value,
            "risk_up", nll_value is not None,
        ),
        _Signal(
            "token_rank_alignment", "probability_shape_risk",
            PROBABILITY_SIGNAL_WEIGHTS["token_rank_alignment"],
            token_rank_value,
            "risk_up", token_rank_value is not None,
        ),
        _Signal(
            "topk_concentration", "probability_shape_risk",
            PROBABILITY_SIGNAL_WEIGHTS["topk_concentration"],
            topk_value,
            "risk_up", topk_value is not None,
        ),
        _Signal(
            "entropy_low_uncertainty", "probability_shape_risk",
            PROBABILITY_SIGNAL_WEIGHTS["entropy_low_uncertainty"],
            _avg_probability_to_risk(avg_probability),
            "risk_up", avg_probability is not None,
        ),
        _Signal(
            "burstiness_risk", "probability_shape_risk",
            PROBABILITY_SIGNAL_WEIGHTS["burstiness_risk"],
            burstiness_value,
            "risk_up", burstiness_value is not None,
        ),
        _Signal(
            "sentence_probability_uniformity", "probability_shape_risk",
            PROBABILITY_SIGNAL_WEIGHTS["sentence_probability_uniformity"],
            sentence_uniformity_value,
            "risk_up", sentence_uniformity_value is not None,
        ),
        _Signal(
            "fast_detectgpt_curvature", "detector_agreement_risk",
            DETECTOR_SIGNAL_WEIGHTS["fast_detectgpt_curvature"],
            None, "risk_up", False, "unavailable_in_v1",
        ),
        _Signal(
            "fine_tuned_classifier", "detector_agreement_risk",
            DETECTOR_SIGNAL_WEIGHTS["fine_tuned_classifier"],
            None, "risk_up", False, "unavailable_in_v1",
        ),
        _Signal(
            "cross_model_agreement", "detector_agreement_risk",
            DETECTOR_SIGNAL_WEIGHTS["cross_model_agreement"],
            cross_agreement_value,
            "risk_up",
            cross_agreement_value is not None,
        ),
        _Signal(
            "generic_opener_broad_claim", "writing_pattern_risk",
            WRITING_SIGNAL_WEIGHTS["generic_opener_broad_claim"],
            _percent(writing.get("broad_claim_risk")),
            "risk_up", _percent(writing.get("broad_claim_risk")) is not None,
        ),
        _Signal(
            "template_phrase_density", "writing_pattern_risk",
            WRITING_SIGNAL_WEIGHTS["template_phrase_density"],
            template_value,
            "risk_up", template_value is not None,
        ),
        _Signal(
            "predictable_paragraph_route", "writing_pattern_risk",
            WRITING_SIGNAL_WEIGHTS["predictable_paragraph_route"],
            _percent(writing.get("paragraph_progression_risk")),
            "risk_up", _percent(writing.get("paragraph_progression_risk")) is not None,
        ),
        _Signal(
            "sentence_rhythm_regularity", "writing_pattern_risk",
            WRITING_SIGNAL_WEIGHTS["sentence_rhythm_regularity"],
            rhythm_value,
            "risk_up", rhythm_value is not None,
        ),
        _Signal(
            "packed_list_compression", "writing_pattern_risk",
            WRITING_SIGNAL_WEIGHTS["packed_list_compression"],
            _percent(ai.get("qualifying_text_ai_density")),
            "risk_up", _percent(ai.get("qualifying_text_ai_density")) is not None,
        ),
        _Signal(
            "repetition_structural_reuse", "writing_pattern_risk",
            WRITING_SIGNAL_WEIGHTS["repetition_structural_reuse"],
            reuse_value,
            "risk_up", reuse_value is not None,
        ),
        _Signal(
            "overpolish_register_mismatch", "writing_pattern_risk",
            WRITING_SIGNAL_WEIGHTS["overpolish_register_mismatch"],
            overpolish_value,
            "risk_up", overpolish_value is not None,
        ),
    ]

    author_anchor_strength = max(
        0.0,
        100.0 - (_percent(writing.get("lived_detail_risk")) or 100.0),
        _percent(features.get("human_anchor_score")) or 0.0,
    )
    context_anchor_strength = max(
        _percent(writing.get("domain_grounding_strength")) or 0.0,
        _percent(writing.get("source_grounding_strength")) or 0.0,
    )
    source_integrity = max(
        0.0,
        100.0 - (_percent(writing.get("citation_weakness_risk")) or 100.0),
        100.0 - (_percent(writing.get("unsupported_claim_risk")) or 100.0),
        100.0 - (_percent(writing.get("source_grounding_risk")) or 100.0),
    )
    grounding_signals = [
        _Signal("author_anchor_strength", "grounding_strength", GROUNDING_SIGNAL_WEIGHTS["author_anchor_strength"], author_anchor_strength, "risk_down", True),
        _Signal("context_anchor_strength", "grounding_strength", GROUNDING_SIGNAL_WEIGHTS["context_anchor_strength"], context_anchor_strength, "risk_down", True),
        _Signal("source_citation_integrity", "grounding_strength", GROUNDING_SIGNAL_WEIGHTS["source_citation_integrity"], source_integrity, "risk_down", True),
    ]
    signals.extend(grounding_signals)

    probability, probability_available, probability_total = _group_score(signals, "probability_shape_risk")
    detector, detector_available, detector_total = _group_score(signals, "detector_agreement_risk")
    writing_score, writing_available, writing_total = _group_score(signals, "writing_pattern_risk")
    grounding_strength, grounding_available, grounding_total = _group_score(signals, "grounding_strength")
    grounding_gap = max(0.0, min(100.0, 100.0 - grounding_strength))

    score = grouped_external_score_from_groups(
        probability_shape_risk=probability,
        detector_agreement_risk=detector,
        writing_pattern_risk=writing_score,
        grounding_gap_risk=grounding_gap,
    )

    caps: list[dict[str, Any]] = []
    group_available = sum(1 for n in [probability_available, detector_available, writing_available, grounding_available] if n > 0)
    available = probability_available + detector_available + writing_available + grounding_available
    total = probability_total + detector_total + writing_total + grounding_total
    coverage_ratio = available / max(total, 1)

    # v3 grounding-strength discount: reward concrete grounding (a well-grounded doc is less likely
    # machine-generated). Applied to the base weighted score before the high-band caps.
    if EXTERNAL_GROUNDING_DISCOUNT_ALPHA > 0.0 and grounding_strength > 0.0:
        discounted = score * (1.0 - EXTERNAL_GROUNDING_DISCOUNT_ALPHA * grounding_strength / 100.0)
        caps.append({
            "code": "grounding_strength_discount",
            "alpha": EXTERNAL_GROUNDING_DISCOUNT_ALPHA,
            "grounding_strength": round(grounding_strength, 3),
            "before": round(score, 3),
            "after": round(discounted, 3),
        })
        score = discounted

    if grounding_strength >= STRONG_GROUNDING_MIN and detector_available <= WEAK_DETECTOR_AVAILABLE_MAX and score >= HIGH_BAND_MIN:
        caps.append({
            "code": "strong_grounding_weak_detector_coverage",
            "before": round(score, 3),
            "after": HIGH_CAP_SCORE,
        })
        score = min(score, HIGH_CAP_SCORE)
    if coverage_ratio < LOW_COVERAGE_MIN and score >= HIGH_BAND_MIN:
        caps.append({
            "code": "low_signal_coverage_high_band_cap",
            "before": round(score, 3),
            "after": HIGH_CAP_SCORE,
        })
        score = min(score, HIGH_CAP_SCORE)
    if probability >= HIGH_BAND_MIN and max(detector, writing_score) < 35.0 and score >= HIGH_BAND_MIN:
        caps.append({
            "code": "probability_family_not_enough_alone",
            "before": round(score, 3),
            "after": HIGH_CAP_SCORE,
        })
        score = min(score, HIGH_CAP_SCORE)

    score = round(max(0.0, min(100.0, score)), 3)
    return {
        "score": score,
        "band": _band(score),
        "model": MODEL_VERSION,
        "confidence": _confidence(available, total, group_available),
        "note": (
            "DraftProof external-detector proxy based on grouped probability-shape, detector-agreement, "
            "writing-pattern, and grounding/authorship evidence signals. This is an estimate, not a "
            "Turnitin result or authorship verdict."
        ),
        "groups": {
            "probability_shape_risk": round(probability, 3),
            "detector_agreement_risk": round(detector, 3),
            "writing_pattern_risk": round(writing_score, 3),
            "grounding_gap_risk": round(grounding_gap, 3),
            "grounding_strength": round(grounding_strength, 3),
        },
        "weights": dict(GROUP_WEIGHTS),
        "signals": [signal.as_dict() for signal in signals],
        "coverage": {
            "available_signals": available,
            "total_signals": total,
            "ratio": round(coverage_ratio, 3),
            "available_groups": group_available,
            "probability_signals": {"available": probability_available, "total": probability_total},
            "detector_signals": {"available": detector_available, "total": detector_total},
            "writing_signals": {"available": writing_available, "total": writing_total},
            "grounding_signals": {"available": grounding_available, "total": grounding_total},
        },
        "caps": caps,
        "alternates": {
            "legacy_segment_fraction": legacy_segment_fraction,
            "legacy_likelihood": legacy_likelihood,
        },
    }
