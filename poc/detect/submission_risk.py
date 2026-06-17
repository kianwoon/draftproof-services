"""Submission risk — deterministic, additive 3-layer re-presentation of the scan.

A parallel, ADDITIVE composer (the third in the family alongside
``detect.grounding_diagnosis`` and ``detect.critical_thinking``). It re-groups
signals DraftProof ALREADY computes into a student-facing "Submission risk"
headline with a multi-axis breakdown, so the report can lead with *"can you
defend this as your own work?"* instead of a bare AI-likelihood number.

Why: Singapore's universities (and Turnitin itself) now treat AI-likelihood as
a non-conclusive *trigger*, not a verdict. The offence is un-defensible /
undeclared work, not AI-resemblance. So we demote the detector percentage to a
single "text-pattern" axis (kept visible as the external trigger) and lead with
ownership + grounding — the axes the student is actually accountable for.

This does NOT affect the tier, ai_likelihood_score, external estimate, or any
rewrite/credit gate. It only adds the ``submission_risk`` block to the badge.

Three layers (weights are tunable config, not content lists):
    L1 surface (text pattern)     ~22%  <- ai_likelihood tier / detector signal
    L2 ownership (the thinking)   ~50%  <- critical_thinking_control score
    L3 academic (citation+defence)~28%  <- axis_scores citation + ownership dims

Text-UNDETECTABLE axes (AI declaration, course-policy match, group
accountability, oral-defence readiness, passive reliance) are NOT in the text.
``policy_declaration`` is therefore always "unknown — self-declare" and is
EXCLUDED from the weighted blend; the rest of that set is deferred to a future
interactive Defence Check. We never fabricate a value the text cannot support.
"""
from __future__ import annotations

from typing import Any

MODEL_VERSION = "submission_risk_v1"

# Layer weights (tunable). Midpoints of the proposed ranges
# (surface 20-25%, ownership 45-55%, academic 25-35%). These are NUMERIC blend
# weights, not a content/phrase list. Renormalised over AVAILABLE layers.
LAYER_WEIGHTS = {
    "surface": 0.22,
    "ownership": 0.50,
    "academic": 0.28,
}

# Risk-level bands on a 0-100 risk scale (higher = more risk). Tunable.
LOW_RISK_MAX = 34.0      # risk < 34  -> low
MEDIUM_RISK_MAX = 62.0   # 34 <= risk < 62 -> medium ; >= 62 -> high

# Severity ranking for the categorical levels, and the minimum risk value that
# represents each level (used when flooring to the document tier).
_LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}
_LEVEL_FLOOR_RISK = {"low": 0.0, "medium": LOW_RISK_MAX, "high": MEDIUM_RISK_MAX}

# Submission risk must never UNDERSTATE the document's own overall tier — a
# "high"-tier report (many flagged sections) reading as "Low submission risk"
# is contradictory. Floor the overall level to the document tier. The report
# Tier enum is critical/high/medium/low/clean (poc/report/report.py).
TIER_FLOOR_LEVEL = {
    "critical": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "clean": "low",
}

# Representative 0-100 risk value for each categorical detector tier. The
# text-pattern axis intentionally rides on the calibrated badge tier (not raw
# topk), so it reflects what an external detector would actually flag.
TIER_RISK = {
    "GREEN": 12.0,
    "AMBER": 48.0,
    "ORANGE": 76.0,
    "RED": 90.0,
}

# axis_scores[...] categorical level -> 0-100 risk value.
AXIS_LEVEL_RISK = {
    "clear": 15.0,
    "review": 50.0,
    "attention": 80.0,
}

# allow-hardcode: presentation strings (labels + plain-language reasons), NOT a
# scoring/matching oracle — never compared against document text. Single source
# of truth for non-i18n consumers (PDF render, logs); KEEP IN SYNC with frontend
# i18n keys report.submissionRisk.* in draftproof-frontend/src/i18n/en/report.js
# + zh/report.js.
AXIS_LABELS = {
    "text_pattern": "Text-pattern risk",
    "ownership": "Ownership risk",
    "citation": "Citation risk",
    "defence_readiness": "Defence-readiness risk",
    "policy_declaration": "Policy / declaration risk",
}

LEVEL_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "unknown": "Unknown",
}

OVERALL_LABELS = {
    "low": "Low submission risk",
    "medium": "Medium submission risk",
    "high": "High submission risk",
    "unknown": "Not enough text to assess",
}

# allow-hardcode: presentation strings (plain-language "main reason" per axis),
# NOT a scoring/matching oracle — never compared against document text. Keyed by
# axis code; the frontend localizes via report.submissionRisk.reasons.*.
# `flagged_findings` is used when the level is floored up to the document tier
# (the risk comes from flagged sections, not a single axis).
AXIS_REASONS = {
    "ownership": "weak ownership of the thinking — thin judgement, reasoning, or grounding",
    "citation": "claims that are not clearly tied to a source",
    "defence_readiness": "hard to defend as your own work in an interview",
    "text_pattern": "AI-like text patterns that may trigger a detector",
    "flagged_findings": "flagged sections that need review before submission",
}

POLICY_NOTE = "Unknown — self-declare (your AI use, course policy, and group contribution)"

# Below this many scored sentences the grounding signals over-fire; mark the
# diagnosis low_coverage. Mirrors critical_thinking.LOW_COVERAGE_MIN_SENTENCES.
LOW_COVERAGE_MIN_SENTENCES = 5

_NOTE = (
    "Additive Submission-risk view: re-groups existing DraftProof signals into a "
    "3-layer breakdown (text pattern / ownership / academic) and leads with whether "
    "the work is defensible as the student's own. The AI-likelihood percentage is "
    "demoted to the text-pattern axis (an external trigger, not a verdict). "
    "Declaration / policy / group / oral-defence are not in the text and stay "
    "'unknown — self-declare'. Does not affect tier, ai_likelihood, or any gate."
)


def _level_from_risk(risk: float) -> str:
    if risk < LOW_RISK_MAX:
        return "low"
    if risk < MEDIUM_RISK_MAX:
        return "medium"
    return "high"


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def _ownership_risk(ct: dict[str, Any]) -> float | None:
    """Risk = inverse of the Critical Thinking Control score (0-100, higher=better
    control). None when the control diagnosis abstained (insufficient data)."""
    score = (ct or {}).get("score")
    if not isinstance(score, (int, float)):
        return None
    return max(0.0, min(100.0, 100.0 - float(score)))


def _defence_readiness_risk(ct: dict[str, Any]) -> float | None:
    """"Could you defend this in an interview?" — built from the dimensions that
    show the thinking is yours: student_judgement, reasoning_trail,
    evidence_grounding (the anchorable, accountable dims). Optionally tempered by
    the Phase-2 LLM reflection / alternative_comparison flags when present. None
    when no contributing dimension has data."""
    dims = (ct or {}).get("dimensions") or {}
    controls = []
    for name in ("student_judgement", "reasoning_trail", "evidence_grounding"):
        d = dims.get(name) or {}
        if isinstance(d.get("control"), (int, float)):
            controls.append(float(d["control"]))

    llm = (ct or {}).get("llm_dimensions") or {}
    for name in ("reflection", "alternative_comparison"):
        d = llm.get(name) if isinstance(llm, dict) else None
        if isinstance(d, dict) and isinstance(d.get("score"), (int, float)):
            controls.append(float(d["score"]))

    mean_control = _mean(controls)
    if mean_control is None:
        return None
    return max(0.0, min(100.0, 100.0 - mean_control))


def score_submission_risk(
    *,
    ai_likelihood_score: float | None = None,
    ai_tier: str | None = None,
    critical_thinking_control: dict[str, Any] | None = None,
    axis_scores: dict[str, str] | None = None,
    document_tier: str | None = None,
    scored_sentence_count: int | None = None,
) -> dict[str, Any]:
    """Return the additive Submission-risk diagnosis. Pure function over signals
    that are already computed upstream; never gates anything.

    ``document_tier`` (the report overall_tier: critical/high/medium/low/clean)
    floors the overall level so submission risk never reads lower than the
    document's own tier."""
    ct = critical_thinking_control or {}
    axes_in = axis_scores or {}

    # --- L1 surface: text-pattern (the demoted detector signal) ---
    tier_key = str(ai_tier or "").upper()
    text_risk = TIER_RISK.get(tier_key)
    text_axis: dict[str, Any] = {
        "level": _level_from_risk(text_risk) if text_risk is not None else "unknown",
        "risk": round(text_risk, 2) if text_risk is not None else None,
        # the percentage stays visible here (demoted, not hidden)
        "display_score": round(float(ai_likelihood_score), 2)
        if isinstance(ai_likelihood_score, (int, float)) else None,
        "label": AXIS_LABELS["text_pattern"],
    }

    # --- L2 ownership: the thinking ---
    own_risk = _ownership_risk(ct)
    ownership_axis = {
        "level": _level_from_risk(own_risk) if own_risk is not None else "unknown",
        "risk": round(own_risk, 2) if own_risk is not None else None,
        "label": AXIS_LABELS["ownership"],
    }

    # --- L3 academic: citation + defence-readiness ---
    cit_level = str(axes_in.get("citation") or "")
    cit_risk = AXIS_LEVEL_RISK.get(cit_level)
    citation_axis = {
        "level": _level_from_risk(cit_risk) if cit_risk is not None else "unknown",
        "risk": round(cit_risk, 2) if cit_risk is not None else None,
        "label": AXIS_LABELS["citation"],
    }

    def_risk = _defence_readiness_risk(ct)
    defence_axis = {
        "level": _level_from_risk(def_risk) if def_risk is not None else "unknown",
        "risk": round(def_risk, 2) if def_risk is not None else None,
        "label": AXIS_LABELS["defence_readiness"],
    }

    # Text cannot reveal declaration / policy / group accountability -> never guess.
    policy_axis = {
        "level": "unknown",
        "risk": None,
        "label": AXIS_LABELS["policy_declaration"],
        "note": POLICY_NOTE,
    }

    axes = {
        "text_pattern": text_axis,
        "ownership": ownership_axis,
        "citation": citation_axis,
        "defence_readiness": defence_axis,
        "policy_declaration": policy_axis,
    }

    # --- overall: weighted blend over AVAILABLE layers (policy excluded) ---
    academic_risk = _mean([cit_risk, def_risk] if cit_risk is not None or def_risk is not None
                          else [])
    layer_risk = {
        "surface": text_risk,
        "ownership": own_risk,
        "academic": academic_risk,
    }
    contributions = {
        layer: (LAYER_WEIGHTS[layer], risk)
        for layer, risk in layer_risk.items()
        if risk is not None
    }

    low_coverage = (
        scored_sentence_count is not None
        and scored_sentence_count < LOW_COVERAGE_MIN_SENTENCES
    )

    # Floor the overall level to the document's own tier so submission risk never
    # reads lower than the report tier (a "high"-tier report can't say "Low").
    tier_floor = TIER_FLOOR_LEVEL.get(str(document_tier or "").lower())

    if not contributions:
        # No axis data. If the document tier still signals risk, reflect it rather
        # than abstaining to "unknown" on a tiered report.
        if tier_floor and _LEVEL_RANK[tier_floor] > 0:
            return {
                "overall": {
                    "level": tier_floor,
                    "label": OVERALL_LABELS[tier_floor],
                    "risk": _LEVEL_FLOOR_RISK[tier_floor],
                    "main_reason": AXIS_REASONS["flagged_findings"],
                    "main_reason_code": "flagged_findings",
                    "floored": True,
                },
                "axes": axes,
                "weights": dict(LAYER_WEIGHTS),
                "model": MODEL_VERSION,
                "low_coverage": low_coverage,
                "note": _NOTE,
            }
        return {
            "overall": {
                "level": "unknown",
                "label": OVERALL_LABELS["unknown"],
                "risk": None,
                "main_reason": "",
                "main_reason_code": "",
                "floored": False,
            },
            "axes": axes,
            "weights": dict(LAYER_WEIGHTS),
            "model": MODEL_VERSION,
            "low_coverage": low_coverage,
            "note": _NOTE,
        }

    weight_sum = sum(w for w, _ in contributions.values())
    overall_risk = sum(w * r for w, r in contributions.values()) / weight_sum
    overall_level = _level_from_risk(overall_risk)

    floored = False
    if tier_floor and _LEVEL_RANK[tier_floor] > _LEVEL_RANK[overall_level]:
        overall_level = tier_floor
        overall_risk = max(overall_risk, _LEVEL_FLOOR_RISK[tier_floor])
        floored = True

    # main_reason: when floored up to the document tier the risk comes from flagged
    # sections (not a single axis); otherwise the largest weighted contributor
    # (the academic layer attributes to its worse of citation / defence-readiness).
    nag = overall_level != "low"
    if not nag:
        reason_axis = ""
    elif floored:
        reason_axis = "flagged_findings"
    else:
        reason_axis = _dominant_reason_axis(contributions, cit_risk, def_risk)
    main_reason = AXIS_REASONS.get(reason_axis, "")

    return {
        "overall": {
            "level": overall_level,
            "label": OVERALL_LABELS[overall_level],
            "risk": round(overall_risk, 2),
            # English prose for PDF/logs; *_code lets the frontend localize (mirrors
            # grounding_diagnosis.primary_driver).
            "main_reason": main_reason,
            "main_reason_code": reason_axis,
            "floored": floored,
        },
        "axes": axes,
        "weights": dict(LAYER_WEIGHTS),
        "model": MODEL_VERSION,
        "low_coverage": low_coverage,
        "note": _NOTE,
    }


def _dominant_reason_axis(
    contributions: dict[str, tuple[float, float]],
    cit_risk: float | None,
    def_risk: float | None,
) -> str:
    """Pick the axis name driving the headline. Layers map to a user-facing axis;
    the academic layer resolves to whichever of citation / defence-readiness is
    worse so the reason is specific."""
    weighted = {layer: w * r for layer, (w, r) in contributions.items()}
    top_layer = max(weighted, key=lambda k: weighted[k])
    if top_layer == "surface":
        return "text_pattern"
    if top_layer == "ownership":
        return "ownership"
    # academic
    if (def_risk or -1) > (cit_risk or -1):
        return "defence_readiness"
    return "citation"
