"""Policy risk — two policy-interpreted scores from one shared signal engine.

Schools differ: some allow AI for research IF declared and controlled; others don't.
A single "AI likelihood" number can't serve both and reads as an accusation. This
additive composer re-weights the SAME writing signals two ways:

    AI-Allowed     "Does this look like acceptable AI-assisted work?"
                   -> grounding / judgment / specificity matter; surface AI text is light.
    AI-Restricted  "Could this be treated as undeclared / AI-written?"
                   -> surface AI text + authorship voice weigh more.

It re-groups signals DraftProof already computes (grounding_diagnosis buckets +
critical_thinking dimensions). It does NOT change the detector (ai_likelihood, tier)
or any gate — it only re-presents signals (the fourth additive composer alongside
grounding_diagnosis, critical_thinking, submission_risk).

Two of the proposed base signals — declaration_gap and process_defensibility_gap —
are NOT in the text (we can't know if AI use was declared or whether drafts exist).
They are OMITTED from the baseline and the remaining weights are renormalised to sum
to 1; each is surfaced as a "confirm yourself" factor whose ``confirm_delta`` shows
how much the score drops if the student confirms it (the omitted factor at gap 0).
We never fabricate a value the text can't support, and never assert "AI used / not used".
"""
from __future__ import annotations

from typing import Any

MODEL_VERSION = "policy_risk_v1"

# Original weights (the user's model). The text-UNDERIVABLE factors are omitted from
# the baseline and renormalised away; they remain here only to size the confirm nudge.
WEIGHTS_ALLOWED = {
    "grounding_gap": 0.30,
    "judgment_gap": 0.25,
    "prompt_specificity_gap": 0.20,
    "declaration_gap": 0.15,          # underivable -> confirm-yourself factor
    "surface_ai_text_signal": 0.10,
}
WEIGHTS_RESTRICTED = {
    "surface_ai_text_signal": 0.30,
    "authorship_voice_gap": 0.25,
    "grounding_gap": 0.20,
    "prompt_specificity_gap": 0.15,
    "process_defensibility_gap": 0.10,  # underivable -> confirm-yourself factor
}
# Not in the text. Each policy has exactly one (declaration for allowed, process for restricted).
UNDERIVABLE = {"declaration_gap", "process_defensibility_gap"}
CONFIRM_FACTOR = {"ai_allowed": "declaration_gap", "ai_restricted": "process_defensibility_gap"}

# Bands per policy (the user's thresholds). Restricted is stricter.
ALLOWED_BANDS = ((25.0, "low"), (50.0, "moderate"), (75.0, "high"), (float("inf"), "severe"))
RESTRICTED_BANDS = ((20.0, "low"), (45.0, "moderate"), (70.0, "high"), (float("inf"), "severe"))

# allow-hardcode: a single human-readable transparency note for non-i18n consumers
# (PDF/logs). NOT a scoring/matching oracle — never compared against document text.
# The frontend uses its own localized copy (report.policyRisk.*).
_NOTE = (
    "Policy risk re-weights the same writing signals two ways: 'AI allowed' judges whether the "
    "work stays grounded and student-controlled; 'AI restricted' weighs surface AI-text patterns "
    "more. Declaration and process aren't in the text -- they're confirm-yourself factors. These "
    "scores do not prove AI use; they estimate how the draft may read under different school policies."
)


def _band(score: float, bands) -> str:
    for threshold, label in bands:
        if score <= threshold:
            return label
    return bands[-1][1]


def _bucket(buckets: dict, key: str) -> float | None:
    b = buckets.get(key) if isinstance(buckets, dict) else None
    v = (b or {}).get("score") if isinstance(b, dict) else None
    return float(v) if isinstance(v, (int, float)) else None


def _dim_gap(dims: dict, name: str) -> float | None:
    d = dims.get(name) if isinstance(dims, dict) else None
    g = (d or {}).get("gap") if isinstance(d, dict) else None
    return float(g) if isinstance(g, (int, float)) else None


def _score_policy(weights: dict, base_signals: dict, omit_key: str, bands) -> dict:
    """Weighted average over the available DERIVABLE signals (underivable + None dropped),
    renormalised to sum 1. confirm_delta = how much the score drops if the omitted factor is
    confirmed (included at gap 0)."""
    avail = {
        k: w for k, w in weights.items()
        if k not in UNDERIVABLE and isinstance(base_signals.get(k), (int, float))
    }
    weight_sum = sum(avail.values())
    if weight_sum <= 0:
        return {"score": None, "level": "unknown", "main_issue": "",
                "confirm_factor": omit_key, "confirm_delta": 0.0, "confirm_level": "unknown"}

    score = sum(w * base_signals[k] for k, w in avail.items()) / weight_sum
    main_issue = max(avail, key=lambda k: avail[k] * base_signals[k])

    # Confirming the omitted factor includes it at gap 0 -> pulls the weighted mean down.
    w_omit = weights.get(omit_key, 0.0)
    confirmed = score * weight_sum / (weight_sum + w_omit) if w_omit > 0 else score
    delta = score - confirmed

    return {
        "score": round(score, 2),
        "level": _band(score, bands),
        "main_issue": main_issue,
        "confirm_factor": omit_key,
        "confirm_delta": round(delta, 2),
        "confirm_level": _band(confirmed, bands),
    }


def _floor_restricted_to_allowed(ai_allowed: dict, ai_restricted: dict) -> dict:
    """A stricter (AI-restricted) policy must never read as LOWER risk than a more
    permissive (AI-allowed) policy for the SAME document -- that inverts the ordinary
    reading of the two numbers side by side ("stricter policy = same-or-worse risk").
    The two composites weight genuinely different signals (Allowed leans on
    grounding/judgment/specificity gaps; Restricted leans on surface-AI-text/voice
    gaps) and CAN organically invert when a document's worst signals happen to be the
    ones Allowed weights heaviest (2026-07-20 owner review, confirmed via hand
    arithmetic on a live report: grounding_gap=43.75 + prompt_specificity_gap=52.5
    dominate Allowed's weighting while staying below surface_ai_text_signal's weight
    in Restricted, producing ai_allowed=36.2 > ai_restricted=33.39).

    Fix: when ai_restricted's organic score is lower than ai_allowed's, floor it up to
    match. ``main_issue`` and ``confirm_delta`` are left untouched -- they still
    correctly explain what drove the ORGANIC restricted computation. ``confirm_level``
    is rebased onto the floored score so it never shows a lower band than the
    (now-floored) headline level: confirmed_absolute = floored_score - confirm_delta
    (confirm_delta is an absolute point amount, so this is just re-anchoring it to the
    new floor, not recomputing it). ``pre_floor_score``/``floored_to_ai_allowed`` keep
    the organic value inspectable -- we never silently overwrite a computed value
    without leaving a trace of what it was before."""
    a_score = ai_allowed.get("score")
    r_score = ai_restricted.get("score")
    ai_restricted["floored_to_ai_allowed"] = False
    ai_restricted["pre_floor_score"] = None
    if not isinstance(a_score, (int, float)) or not isinstance(r_score, (int, float)):
        return ai_restricted
    if r_score >= a_score:
        return ai_restricted

    confirm_delta = ai_restricted.get("confirm_delta") or 0.0
    ai_restricted["pre_floor_score"] = r_score
    ai_restricted["floored_to_ai_allowed"] = True
    ai_restricted["score"] = a_score
    ai_restricted["level"] = _band(a_score, RESTRICTED_BANDS)
    ai_restricted["confirm_level"] = _band(a_score - confirm_delta, RESTRICTED_BANDS)
    return ai_restricted


def score_policy_risk(
    *,
    grounding_diagnosis: dict[str, Any] | None = None,
    critical_thinking_control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the additive two-score policy risk. Pure function over already-computed
    signals; never gates anything."""
    buckets = (grounding_diagnosis or {}).get("buckets") or {}
    dims = (critical_thinking_control or {}).get("dimensions") or {}

    base_signals = {
        "surface_ai_text_signal": _bucket(buckets, "llm_patterning"),
        "grounding_gap": _bucket(buckets, "concrete_grounding"),
        "authorship_voice_gap": _bucket(buckets, "authorship_trace"),
        "judgment_gap": _dim_gap(dims, "student_judgement"),
        "prompt_specificity_gap": _dim_gap(dims, "specific_context"),
    }

    ai_allowed = _score_policy(WEIGHTS_ALLOWED, base_signals, "declaration_gap", ALLOWED_BANDS)
    ai_restricted = _score_policy(WEIGHTS_RESTRICTED, base_signals, "process_defensibility_gap", RESTRICTED_BANDS)
    ai_restricted = _floor_restricted_to_allowed(ai_allowed, ai_restricted)

    return {
        "ai_allowed": ai_allowed,
        "ai_restricted": ai_restricted,
        "base_signals": {k: (round(v, 2) if isinstance(v, (int, float)) else None) for k, v in base_signals.items()},
        "weights": {"ai_allowed": dict(WEIGHTS_ALLOWED), "ai_restricted": dict(WEIGHTS_RESTRICTED)},
        "model": MODEL_VERSION,
        "note": _NOTE,
    }
