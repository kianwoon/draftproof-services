"""Grounding-gap diagnosis — additive 4-bucket AI-style profile.

A parallel, ADDITIVE diagnosis layer. It re-groups signals DraftProof already
computes into four user-facing buckets and surfaces the dominant *driver*
(e.g. "grounding gap"), so the report can lead with what to FIX rather than a
bare risk number.

This does NOT affect the tier, ai_likelihood_score, or any rewrite/credit gate.
It mirrors the renormalize-over-available-signals mechanics of
``detect.external_grouped_scoring`` but with its own buckets, weights, and bands.

Buckets (100 marks):
    concrete_grounding 29 | authorship_trace 13 | llm_patterning 40 | language_texture 18
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MODEL_VERSION = "grounding_diagnosis_v1"

BUCKET_WEIGHTS = {
    "concrete_grounding": 29,
    "authorship_trace": 13,
    "llm_patterning": 40,
    "language_texture": 18,
}

# Driver -> (user-facing label, suggested action). SINGLE SOURCE OF TRUTH for
# these strings. KEEP IN SYNC with poc/report/render.py (_grounding_diagnosis_lead)
# and the frontend i18n keys report.groundingDiagnosis.* in
# draftproof-frontend/src/i18n/en/report.js + zh/report.js.
DRIVER_LABELS = {
    "concrete_grounding": ("grounding gap", "add concrete anchors, named evidence, specifics"),
    "authorship_trace": ("authorship trace", "show your decision trail / draft process"),
    "llm_patterning": ("LLM patterning", "vary structure and phrasing"),
    "language_texture": ("language texture", "cut padding, add concrete meaning"),
}

LIKELY_HUMAN_MAX = 20.0
SOME_TEXTURE_MAX = 40.0
MODERATE_MAX = 60.0
STRONG_MAX = 80.0

# Below this many scored sentences, grounding signals over-fire on short text;
# flag the diagnosis low_coverage rather than suppress the driver outright.
LOW_COVERAGE_MIN_SENTENCES = 5


@dataclass(frozen=True)
class _Signal:
    key: str
    bucket: str
    value: float | None
    available: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"key": self.key, "bucket": self.bucket, "available": self.available}
        if self.available and self.value is not None:
            out["value"] = round(float(self.value), 3)
        if self.note:
            out["note"] = self.note
        return out


def _percent(value: Any) -> float | None:
    """Coerce a 0-1 or 0-100 number to a clamped 0-100 float; None if not numeric."""
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if math.isnan(numeric):
        return None
    if abs(numeric) <= 1.0:
        numeric *= 100.0
    return max(0.0, min(100.0, numeric))


def _gap_from_strength(value: Any) -> float | None:
    """Invert a 'strength' signal into a 0-100 gap/risk value."""
    pct = _percent(value)
    return None if pct is None else max(0.0, min(100.0, 100.0 - pct))


def _maxp(*values: Any) -> float | None:
    """Max over the available _percent() values; None if none available."""
    ps = [p for p in (_percent(v) for v in values) if p is not None]
    return max(ps) if ps else None


def _band(score: float) -> str:
    if score <= LIKELY_HUMAN_MAX:
        return "likely_human"
    if score <= SOME_TEXTURE_MAX:
        return "some_texture"
    if score <= MODERATE_MAX:
        return "moderate"
    if score <= STRONG_MAX:
        return "strong"
    return "very_strong"


def _bucket_signals(
    ai: dict[str, Any], writing: dict[str, Any], features: dict[str, Any]
) -> list[_Signal]:
    """Map existing components onto the 4 buckets as 0-100 gap/risk values."""
    # Re-derive anchors from existing signals (derived, not badge keys). Unlike
    # external_grouped_scoring, a MISSING input stays unavailable rather than
    # defaulting to zero strength (which would read as a maximal gap on no data).
    _lived = _percent(writing.get("lived_detail_risk"))
    _human = _percent(features.get("human_anchor_score"))
    _anchor_parts = [v for v in ((100.0 - _lived) if _lived is not None else None, _human) if v is not None]
    author_anchor_strength = max(_anchor_parts) if _anchor_parts else None
    context_anchor_strength = _maxp(
        writing.get("domain_grounding_strength"), writing.get("source_grounding_strength")
    )

    signals: list[_Signal] = []

    def add(key: str, bucket: str, value: float | None, note: str = "") -> None:
        signals.append(_Signal(key, bucket, value, value is not None, note))

    # Concrete grounding (specificity + context + named evidence)
    add("specificity_gap", "concrete_grounding", _percent(writing.get("lived_detail_risk")))
    add("broad_claim_risk", "concrete_grounding", _percent(writing.get("broad_claim_risk")))
    add("evidence_gap", "concrete_grounding",
        _maxp(writing.get("source_grounding_risk"), writing.get("citation_weakness_risk")))
    add("context_gap", "concrete_grounding",
        None if context_anchor_strength is None else max(0.0, 100.0 - context_anchor_strength))

    # Authorship trace (decision trail / draft process / personal judgement)
    add("author_anchor_gap", "authorship_trace",
        None if author_anchor_strength is None else max(0.0, 100.0 - author_anchor_strength))
    add("human_anchor_gap", "authorship_trace", _gap_from_strength(features.get("human_anchor_score")))

    # LLM patterning (predictability + rhythm + template flow + over-balance)
    add("predictability", "llm_patterning", _percent(ai.get("predictability")))
    add("topk_pattern", "llm_patterning", _percent(ai.get("topk_pattern")))
    add("structure_reuse", "llm_patterning", _percent(ai.get("repeated_sentence_structure_risk")))
    add("rhythm_regularity", "llm_patterning",
        _maxp(writing.get("paragraph_uniformity_risk"), writing.get("repeated_starter_risk")))
    add("template_flow", "llm_patterning",
        _maxp(writing.get("signpost_paragraph_risk"), writing.get("formulaic_conclusion_risk"),
              writing.get("paragraph_progression_risk")))
    add("over_balance", "llm_patterning", _percent(ai.get("balanced_hedging_risk")))

    # Language texture (semantic padding + paraphrase residue + [abstraction: unmeasured])
    add("semantic_padding", "language_texture", _percent(ai.get("qualifying_text_ai_density")))
    add("paraphrase_residue", "language_texture",
        _maxp(features.get("paraphrase_transformation_risk"), features.get("rewrite_smoothness")))
    signals.append(_Signal("abstraction", "language_texture", None, False,
                           "abstract_noun_density_unmeasured"))
    return signals


def _bucket_score(signals: list[_Signal], bucket: str) -> tuple[float, int, int]:
    """Mean of available signal values in a bucket -> (score, available, total)."""
    rows = [s for s in signals if s.bucket == bucket]
    avail = [s.value for s in rows if s.available and s.value is not None]
    if not avail:
        return 0.0, 0, len(rows)
    return max(0.0, min(100.0, sum(avail) / len(avail))), len(avail), len(rows)


def diagnose_grounding_gap(
    *,
    ai_components: dict[str, Any] | None = None,
    writing_components: dict[str, Any] | None = None,
    transformation_features: dict[str, Any] | None = None,
    scored_sentence_count: int | None = None,
) -> dict[str, Any]:
    """Return the 4-bucket grounding diagnosis (additive; never gates anything)."""
    ai = ai_components or {}
    writing = writing_components or {}
    features = transformation_features or {}

    signals = _bucket_signals(ai, writing, features)

    buckets: dict[str, dict[str, Any]] = {}
    for name, weight in BUCKET_WEIGHTS.items():
        score, available, total = _bucket_score(signals, name)
        buckets[name] = {
            "score": round(score, 2),
            "weight": weight,
            "available": available,
            "total": total,
        }

    active = {n: b for n, b in buckets.items() if b["available"] > 0}
    weight_sum = sum(b["weight"] for b in active.values())
    if weight_sum > 0:
        final = sum(b["weight"] * b["score"] for b in active.values()) / weight_sum
    else:
        final = 0.0
    final = round(max(0.0, min(100.0, final)), 2)

    # Contributions (renormalized so they sum to the final score) -> primary driver.
    for name, b in buckets.items():
        b["contribution"] = round(b["weight"] * b["score"] / weight_sum, 2) if (weight_sum and b["available"]) else 0.0
    worst_dimension = max(active, key=lambda n: active[n]["score"], default=None)
    primary_driver = max(active, key=lambda n: buckets[n]["contribution"], default=None)

    band = _band(final)
    low_coverage = scored_sentence_count is not None and scored_sentence_count < LOW_COVERAGE_MIN_SENTENCES

    # Driver visibility: only suppress when genuinely clean (likely_human). At
    # "some_texture" (the target-doc case ~34) we DO surface the driver.
    caveat = ""
    if band == "likely_human" or primary_driver is None:
        primary_driver = None
        primary_driver_label = "well-grounded — no significant driver"
        primary_driver_action = ""
    else:
        primary_driver_label, primary_driver_action = DRIVER_LABELS[primary_driver]
        if low_coverage:
            caveat = "limited text — diagnosis tentative"

    return {
        "score": final,
        "band": band,
        "model": MODEL_VERSION,
        "primary_driver": primary_driver,
        "primary_driver_label": primary_driver_label,
        "primary_driver_action": primary_driver_action,
        "worst_dimension": worst_dimension,
        "caveat": caveat,
        "low_coverage": low_coverage,
        "buckets": buckets,
        "weights": dict(BUCKET_WEIGHTS),
        "signals": [s.as_dict() for s in signals],
        "coverage": {
            "available_signals": sum(1 for s in signals if s.available),
            "total_signals": len(signals),
            "active_buckets": len(active),
        },
        "limitations": ["abstract_noun_density_unmeasured"],
        "note": (
            "Additive grounding diagnosis: re-groups existing DraftProof signals into "
            "concrete-grounding / authorship / LLM-patterning / language-texture to surface "
            "the primary thing to fix. Does not affect tier, ai_likelihood, or any gate."
        ),
    }
