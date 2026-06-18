"""Critical Thinking Control — deterministic, additive student-control diagnosis.

A parallel, ADDITIVE diagnosis layer. It re-groups signals DraftProof already
computes (via ``detect.grounding_diagnosis``) into student-facing *control*
dimensions and surfaces the weakest one, so the report can lead with what the
student should THINK harder about — judgement, comparison, decision-making,
ownership — rather than a bare AI-risk number.

This does NOT affect the tier, ai_likelihood_score, or any rewrite/credit gate.

Higher score = STRONGER human control (the inverse of a grounding *gap*). The
headline number is fully deterministic: it consumes the per-signal *gap* values
already produced by ``diagnose_grounding_gap`` and re-aggregates them into the
5 reusable dimensions below. The two genuinely-new dimensions
(``alternative_comparison``, ``reflection``) are NON-deterministic LLM-judged
flags handled separately in ``critical_thinking_llm`` and are NOT folded into
this number.

Deterministic dimensions (75 nominal marks, renormalised over available data):
    specific_context 20 | student_judgement 20 | reasoning_trail 15
    ai_dependency 10 | evidence_grounding 10
"""
from __future__ import annotations

from typing import Any

MODEL_VERSION = "critical_thinking_v1"

# Nominal weights for the 5 DETERMINISTIC dimensions (the 2 LLM dimensions
# alternative_comparison=15 / reflection=10 are surfaced as qualitative flags,
# never summed into this score). Renormalised over available dimensions.
DIMENSION_WEIGHTS = {
    "specific_context": 20,
    "student_judgement": 20,
    "reasoning_trail": 15,
    "ai_dependency": 10,
    "evidence_grounding": 10,
}

# Each control dimension re-aggregates these grounding_diagnosis signal keys
# (every value is a 0-100 *gap/risk*, higher = weaker control). The mean of the
# available signals is the dimension's gap; control = 100 - gap. These are
# SIGNAL KEYS (machine identifiers from grounding_diagnosis), not content text.
#
# DELIBERATELY EXCLUDED from ai_dependency: predictability + topk_pattern. Those
# measure raw FLUENCY, which this project has repeatedly proven cannot distinguish
# fluent-HUMAN from AI text (the un-mitigable topk floor sits ~70-80 even on
# genuine human writing). Including them made ai_dependency the weakest dimension
# on well-grounded human essays -> it became the lead coaching ("you're too
# AI-dependent") = a FALSE ACCUSATION (the cardinal sin). ai_dependency keeps only
# over_balance (a targeted both-sides-hedging tell) and is lead-ineligible below.
DIMENSION_SIGNALS = {
    "specific_context": ("specificity_gap", "broad_claim_risk"),
    "student_judgement": ("author_anchor_gap", "human_anchor_gap"),
    "reasoning_trail": ("template_flow", "structure_reuse"),
    "ai_dependency": ("over_balance",),
    "evidence_grounding": ("evidence_gap", "context_gap"),
}

# Dimensions that may render as an informational bar but must NEVER become the
# headline coaching. ai_dependency rides on non-anchorable stylometry (over_balance
# is a shape signal, not a passage you can point a student at), so per precision-
# first the lead is always one of the 4 anchorable grounding/judgement dimensions.
LEAD_INELIGIBLE_DIMENSIONS = frozenset({"ai_dependency"})

# allow-hardcode: machine finding-type / signal-category identifiers routed to a
# dimension code — signal routing, NOT a content/text matcher (mirrors
# detect.base._SIGNAL_CATEGORY_MAP). No document text is ever compared.
#
# PRIMARY: map a finding's precise ``finding_type`` (the report-level Finding.title)
# to a control dimension. This is finer than signal_category, which collapses
# citation gaps and surface grammar/spelling into one "writing_quality" bucket --
# mapping the whole bucket to evidence_grounding would mislabel a grammar-flagged
# paragraph as "connect this claim to a source". Surface writing types
# (grammar_issue / fragment_sentence / spelling_issue / punctuation_issue) are
# intentionally UNMAPPED -> they are not a critical-thinking gap and get no tag.
FINDING_TYPE_TO_DIMENSION = {
    # specific_context — vague / broad / template, lacking concrete specifics
    "generic_phrase": "specific_context",
    "generic_policy_claim": "specific_context",
    "broad_education_claim": "specific_context",
    "template_personal_reflection": "specific_context",
    "low_specificity": "specific_context",
    "formulaic_conclusion": "specific_context",
    # evidence_grounding — citation / source gaps
    "uncited_claim": "evidence_grounding",
    "missing_citation": "evidence_grounding",
    "broken_citation": "evidence_grounding",
    "weak_source_grounding": "evidence_grounding",
    # NOTE: authorship_risk finding types (semantic_drift, similarity_overlap,
    # semantic_uniformity, discourse_regularity, *_ai_generation_likelihood) are
    # DELIBERATELY NOT mapped to student_judgement. They are AI-LIKENESS / detector
    # signals, not evidence the writer failed to take a position -- mapping them
    # would re-introduce the AI-detection-vs-thinking conflation we removed from
    # ai_dependency (it would coach "take a position" off a coherence signal).
    # student_judgement therefore has NO per-paragraph source; it stays
    # document-level only (like reasoning_trail).
    # ai_dependency — stylometry (mapped but LEAD-INELIGIBLE below)
    "high_predictability": "ai_dependency",
    "medium_predictability": "ai_dependency",
    "low_predictability": "ai_dependency",
    "review_predictability": "ai_dependency",
    "high_topk_predictability": "ai_dependency",
    "style_shift": "ai_dependency",
    "low_burstiness": "ai_dependency",
    "repetitive_sentence_structure": "ai_dependency",
}

# FALLBACK for finding types not in the precise map above. Deliberately OMITS:
#   - "writing_quality" (ambiguous citation+grammar bucket) so an unmapped surface
#     issue is never mislabeled; the precise citation types already cover the
#     legitimate writing_quality -> evidence_grounding case.
#   - "authorship_risk" -> would coach student_judgement off AI-likeness signals
#     (the conflation removed above). student_judgement + reasoning_trail have NO
#     per-paragraph source; they are document-level only.
# predictability -> ai_dependency stays LEAD-INELIGIBLE (no per-paragraph
# "too AI-dependent" false accusation), so per-paragraph tags are only ever
# specific_context or evidence_grounding -- the two genuinely anchorable gaps.
SIGNAL_CATEGORY_TO_DIMENSION = {
    "genericity": "specific_context",
    "predictability": "ai_dependency",
}


def _dimension_for_finding(finding: dict[str, Any]) -> str | None:
    """Precise finding_type mapping first; coarse signal_category fallback (never
    for the ambiguous writing_quality bucket)."""
    finding_type = str((finding or {}).get("finding_type") or "")
    dimension = FINDING_TYPE_TO_DIMENSION.get(finding_type)
    if dimension:
        return dimension
    return SIGNAL_CATEGORY_TO_DIMENSION.get(str(finding.get("signal_category") or ""))

# allow-hardcode: presentation strings, NOT a scoring/matching oracle. These map
# a dimension CODE -> (user-facing label, coaching action); they are never
# compared against document text. Mirrors the approved DRIVER_LABELS convention
# in grounding_diagnosis.py. SINGLE SOURCE OF TRUTH — KEEP IN SYNC with the
# frontend i18n keys report.criticalThinking.dimensions.* in
# draftproof-frontend/src/i18n/en/report.js + zh/report.js.
DIMENSION_LABELS = {
    "specific_context": ("specific context", "anchor claims to a real assignment, case, example, or observation"),
    "student_judgement": ("student judgement", "take a position and justify why you chose it"),
    "reasoning_trail": ("reasoning trail", "show how you moved from evidence to conclusion"),
    "ai_dependency": ("AI dependency", "challenge the first answer instead of keeping it as-is"),
    "evidence_grounding": ("evidence grounding", "connect each claim to a source, example, or data point"),
}

# Bands on the CONTROL score (higher = stronger control).
STRONG_CONTROL_MIN = 80.0
ACCEPTABLE_CONTROL_MIN = 60.0
WEAK_CONTROL_MIN = 40.0
HIGH_DEPENDENCY_MIN = 20.0

# Below this many scored sentences, grounding signals over-fire on short text;
# flag the diagnosis low_coverage rather than suppress the lead outright. Mirrors
# grounding_diagnosis.LOW_COVERAGE_MIN_SENTENCES.
LOW_COVERAGE_MIN_SENTENCES = 5

# allow-hardcode: presentation strings, NOT a detect-list. Maps a band CODE ->
# human-readable status for non-i18n consumers (PDF render, logs). Never matched
# against input text. Frontend i18n (report.criticalThinking.bands.*) is the
# localized source; KEEP IN SYNC.
BAND_STATUS = {
    "strong_control": "Strong student control",
    "acceptable_control": "Acceptable control",
    "weak_control": "Weak control",
    "high_dependency": "High dependency risk",
    "very_high_dependency": "Very high dependency risk",
    "insufficient_data": "Not enough text to assess",
}


def _band(score: float) -> str:
    if score >= STRONG_CONTROL_MIN:
        return "strong_control"
    if score >= ACCEPTABLE_CONTROL_MIN:
        return "acceptable_control"
    if score >= WEAK_CONTROL_MIN:
        return "weak_control"
    if score >= HIGH_DEPENDENCY_MIN:
        return "high_dependency"
    return "very_high_dependency"


def _signal_gap_map(grounding_diagnosis: dict[str, Any]) -> dict[str, float]:
    """Extract {signal_key: gap_value} for the AVAILABLE signals only."""
    out: dict[str, float] = {}
    for sig in grounding_diagnosis.get("signals", []) or []:
        if sig.get("available") and isinstance(sig.get("value"), (int, float)):
            out[sig["key"]] = max(0.0, min(100.0, float(sig["value"])))
    return out


def score_critical_thinking(
    *,
    grounding_diagnosis: dict[str, Any] | None = None,
    scored_sentence_count: int | None = None,
) -> dict[str, Any]:
    """Return the deterministic Critical Thinking Control diagnosis.

    Consumes the already-computed ``diagnose_grounding_gap`` output so the math
    has a single source of truth. Additive; never gates anything.
    """
    diag = grounding_diagnosis or {}
    gaps = _signal_gap_map(diag)

    dimensions: dict[str, dict[str, Any]] = {}
    for name, weight in DIMENSION_WEIGHTS.items():
        keys = DIMENSION_SIGNALS[name]
        vals = [gaps[k] for k in keys if k in gaps]
        label, action = DIMENSION_LABELS[name]
        if vals:
            gap = sum(vals) / len(vals)
            control = max(0.0, min(100.0, 100.0 - gap))
            dimensions[name] = {
                "control": round(control, 2),
                "gap": round(gap, 2),
                "weight": weight,
                "available": len(vals),
                "total": len(keys),
                "label": label,
                "action": action,
            }
        else:
            dimensions[name] = {
                "control": None,
                "gap": None,
                "weight": weight,
                "available": 0,
                "total": len(keys),
                "label": label,
                "action": action,
            }

    active = {n: d for n, d in dimensions.items() if d["available"] > 0}
    weight_sum = sum(d["weight"] for d in active.values())

    low_coverage = (
        scored_sentence_count is not None
        and scored_sentence_count < LOW_COVERAGE_MIN_SENTENCES
    )

    if weight_sum <= 0:
        # No usable signals -> abstain rather than emit a false "very high
        # dependency" 0 (precision-first: do not invent a verdict on no data).
        return {
            "score": None,
            "band": "insufficient_data",
            "status": BAND_STATUS["insufficient_data"],
            "model": MODEL_VERSION,
            "dimensions": dimensions,
            "lead_dimension": None,
            "lead_dimension_label": "",
            "lead_dimension_action": "",
            "weakest_dimension": None,
            "weights": dict(DIMENSION_WEIGHTS),
            "low_coverage": low_coverage,
            "caveat": "insufficient signal coverage to assess student control",
            "coverage": {"available_dimensions": 0, "total_dimensions": len(dimensions)},
            "llm_dimensions": None,
            "highlights": [],
            "note": _NOTE,
        }

    weighted_gap = sum(d["weight"] * (100.0 - d["control"]) for d in active.values()) / weight_sum
    score = round(max(0.0, min(100.0, 100.0 - weighted_gap)), 2)
    band = _band(score)

    # Weakest available dimension (lowest control) -- reported for transparency.
    weakest = min(active, key=lambda n: active[n]["control"])

    # The LEAD coaching is the weakest *lead-eligible* dimension (excludes the
    # non-anchorable ai_dependency, so we never headline a false "too AI-dependent"
    # accusation on fluent human writing). Falls back to None if nothing eligible.
    lead_eligible = {n: d for n, d in active.items() if n not in LEAD_INELIGIBLE_DIMENSIONS}
    lead_candidate = (
        min(lead_eligible, key=lambda n: lead_eligible[n]["control"])
        if lead_eligible else None
    )

    # Only frame a "work on this" lead when control isn't already strong. At
    # strong_control we still expose the weakest dimension for transparency, but
    # without nagging the student to fix a well-controlled draft.
    if band == "strong_control" or lead_candidate is None:
        lead_dimension = None
        lead_label = ""
        lead_action = ""
    else:
        lead_dimension = lead_candidate
        lead_label, lead_action = DIMENSION_LABELS[lead_candidate]

    caveat = ""
    if low_coverage and band != "strong_control":
        caveat = "limited text — diagnosis tentative"

    return {
        "score": score,
        "band": band,
        "status": BAND_STATUS[band],
        "model": MODEL_VERSION,
        "dimensions": dimensions,
        "lead_dimension": lead_dimension,
        "lead_dimension_label": lead_label,
        "lead_dimension_action": lead_action,
        "weakest_dimension": weakest,
        "weights": dict(DIMENSION_WEIGHTS),
        "low_coverage": low_coverage,
        "caveat": caveat,
        "coverage": {
            "available_dimensions": len(active),
            "total_dimensions": len(dimensions),
        },
        # Filled by detect.critical_thinking_llm (Phase 2, behind kill-switch).
        # These are qualitative review flags, NOT folded into ``score``.
        "llm_dimensions": None,
        "highlights": [],
        "note": _NOTE,
    }


_NOTE = (
    "Additive Critical Thinking Control diagnosis: re-groups existing DraftProof "
    "grounding signals into student-control dimensions (specific context / student "
    "judgement / reasoning trail / AI dependency / evidence grounding) and surfaces "
    "the weakest one to work on. Higher = stronger human control. Does not affect "
    "tier, ai_likelihood, or any gate."
)


def score_critical_thinking_per_paragraph(
    findings_by_paragraph: dict[str, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Tag each flagged paragraph with its weakest LEAD-ELIGIBLE control dimension.

    Deterministic and additive. ``findings_by_paragraph`` maps a paragraph_id to a
    list of that paragraph's flagged findings, each a dict with ``finding_type``
    (the precise detector type), ``signal_category`` (coarse fallback), and an
    optional numeric ``score``. For each paragraph we accumulate finding weight per
    dimension via ``_dimension_for_finding`` and emit the most-flagged lead-eligible
    one (ai_dependency is excluded so a paragraph is never falsely headlined
    "too AI-dependent"; surface grammar/spelling findings map to nothing). Returns
    one row per tagged paragraph:
    ``{paragraph_id, dimension, label, action, weight}`` (paragraphs with no
    lead-eligible flagged dimension are omitted).
    """
    out: list[dict[str, Any]] = []
    for pid, findings in (findings_by_paragraph or {}).items():
        if not pid or not findings:
            continue
        weights: dict[str, float] = {}
        for finding in findings:
            dimension = _dimension_for_finding(finding)
            if not dimension:
                continue
            raw = (finding or {}).get("score")
            weight = float(raw) if isinstance(raw, (int, float)) else 1.0
            weights[dimension] = weights.get(dimension, 0.0) + max(0.0, weight)

        eligible = {
            d: w for d, w in weights.items()
            if d not in LEAD_INELIGIBLE_DIMENSIONS and w > 0
        }
        if not eligible:
            continue
        dimension = max(eligible, key=lambda d: eligible[d])
        label, action = DIMENSION_LABELS[dimension]
        out.append({
            "paragraph_id": str(pid),
            "dimension": dimension,
            "label": label,
            "action": action,
            "weight": round(eligible[dimension], 2),
        })
    out.sort(key=lambda row: row["paragraph_id"])
    return out
