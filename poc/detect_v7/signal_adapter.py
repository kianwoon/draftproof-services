"""V7 signal adapter — maps existing ``poc/detect/`` per-paragraph signals onto
the 14 V7 signal names declared in ``weights.json``.

This module performs RENAMING/DERIVATION ONLY. It never introduces new scoring
weights, thresholds, or blend ratios of its own — every number that flows
through here comes from an existing ``poc/detect/`` signal. Where a signal
must be *derived* from more than one input (e.g. ``predictable_structure``,
``sentence_smoothness``, ``author_voice_absence``), the exact arithmetic and
the reasoning for it are documented inline; these are structural choices
(e.g. "mean of N equally-informative existing risk signals"), never tuned
weights (per CLAUDE.md's no-hardcode-weights rule).

Input shape (``raw_signals``): a per-paragraph dict that mirrors the existing
production shapes already used across ``poc/detect/``:
    - ``ai_components``: dict shaped like ``layer3_scoring.WritingComponentInputs``
      text-pattern fields (e.g. ``generic_assertion_risk``, ``repeated_sentence_structure_risk``).
    - ``writing_components``: dict shaped like the grounding/structure fields of
      the same dataclass (e.g. ``lived_detail_risk``, ``citation_weakness_risk``,
      ``formulaic_conclusion_risk``, ``signpost_paragraph_risk``).
    - ``transformation_features``: dict carrying ``human_anchor_score`` (see
      ``authorship_windows.HUMAN_SIGNAL_WEIGHTS``) and other derived features.
    - ``criterion_scores``: dict of the ``poc/detect/criteria/*.py`` ``score()``
      outputs (``CriterionScore`` objects or their ``.details``/``.value`` dict
      equivalents), keyed by criterion name: ``low_burstiness``, ``low_surprisal``,
      ``low_specificity``, ``style_shift``, ``repetitive_structure``.
    - ``semantic_shape``: optional dict from ``semantic_shape.SemanticShapeDetector``
      (``semantic_drift_risk`` etc.) — only meaningful with comparison text (see
      the ``semantic_drift`` mapping below).
    - ``has_comparison_text``: bool, explicit flag for whether a draft-evolution/
      comparison text was supplied for this paragraph.

All values are read defensively (``raw_signals`` sub-dicts may be missing or
None); a signal whose required inputs are absent maps to ``None`` with
``signal_status`` reflecting why.
"""
from __future__ import annotations

from typing import Any, Optional


# The 14 V7 signal names, verbatim from detect_v7/weights.json category_weights
# (deduplicated across all category formulas) minus calibrated_detector_score
# (owned by the not-yet-built detector_fusion.py), plus esl_false_positive_risk
# (spec §5, not part of any category formula — consumed separately by the ESL
# guard in category_scoring.py).
V7_SIGNAL_NAMES: tuple[str, ...] = (
    "specificity_score",
    "grounding_gap",
    "sentence_variance",
    "generic_density",
    "author_voice_absence",
    "sentence_smoothness",
    "local_style_shift",
    "semantic_drift",
    "paraphrase_pattern_score",
    "meaning_preservation_score",
    "predictable_structure",
    "citation_gap",
    "detector_disagreement",
    "esl_false_positive_risk",
)

_STATUS_OK = "ok"
_STATUS_NOT_IMPLEMENTED = "not_implemented"
_STATUS_NO_COMPARISON = "unavailable_no_comparison_text"
# A BUILT signal whose inputs were absent for THIS document/run. Distinct from
# "not_implemented" (signal doesn't exist in the codebase yet — a roadmap fact,
# not a data-quality fact): only "unavailable" should degrade the breakdown's
# trust accounting (category_scoring counts it toward `degraded`), otherwise
# the two unbuilt Phase-1C signals force degraded=True on EVERY document and
# the degraded_display / primary_category_reliable guards can never
# discriminate (measured 2026-07-06: degraded_display fired on 100% of both
# human and AI classes).
_STATUS_UNAVAILABLE = "unavailable"


def _get(d: Optional[dict[str, Any]], key: str) -> Optional[float]:
    """Read a numeric field out of a possibly-missing/None dict."""
    if not isinstance(d, dict):
        return None
    value = d.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _get_criterion_value(criterion_scores: Optional[dict[str, Any]], name: str) -> Optional[float]:
    """Read a criteria/*.py ``score()`` output's ``.value`` (0-1), whether it's a
    ``CriterionScore`` dataclass instance or an already-dict-ified equivalent.
    """
    if not isinstance(criterion_scores, dict) or name not in criterion_scores:
        return None
    entry = criterion_scores[name]
    if hasattr(entry, "value"):
        value = getattr(entry, "value")
    elif isinstance(entry, dict):
        value = entry.get("value")
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def adapt_paragraph_signals(raw_signals: dict[str, Any]) -> dict[str, Any]:
    """Map one paragraph's existing poc/detect/ signals onto the 14 V7 names.

    Returns a dict with all 14 V7 signal names as keys (float in [0,1] or
    None), plus a companion ``signal_status`` dict keyed the same way with
    values "ok" / "unavailable" (built signal, inputs absent this run) /
    "not_implemented" (signal not built yet — only the Phase-1C/2 rows) /
    "unavailable_no_comparison_text".
    """
    raw_signals = raw_signals or {}
    ai = raw_signals.get("ai_components") or {}
    writing = raw_signals.get("writing_components") or {}
    features = raw_signals.get("transformation_features") or {}
    criterion_scores = raw_signals.get("criterion_scores") or {}
    semantic_shape = raw_signals.get("semantic_shape") or {}
    has_comparison_text = bool(raw_signals.get("has_comparison_text"))

    out: dict[str, Optional[float]] = {}
    status: dict[str, str] = {}

    # 1. grounding_gap ← direct: writing_components.source_grounding_risk if present,
    #    else the citation_grounding_gap meta-criterion (polish_vs_grounding.py), which
    #    is itself already a 0-1 "gap" (high = ungrounded). No new weighting: pick the
    #    first available direct signal, in the order the spec names it (source_grounding
    #    risk is the primary grounding-gap field on WritingComponentInputs; the
    #    citation_grounding_gap criterion is the documented fallback proxy).
    grounding_gap = _get(writing, "source_grounding_risk")
    if grounding_gap is None:
        grounding_gap = _get_criterion_value(criterion_scores, "citation_grounding_gap")
    out["grounding_gap"] = _clamp01(grounding_gap)
    status["grounding_gap"] = _STATUS_OK if out["grounding_gap"] is not None else _STATUS_UNAVAILABLE

    # 2. generic_density ← direct: ai_components.generic_assertion_risk (layer3_scoring's
    #    primary generic-content signal) else the generic_phrase_density criterion value.
    generic_density = _get(ai, "generic_assertion_risk")
    if generic_density is None:
        generic_density = _get(ai, "generic_phrase_density")
    if generic_density is None:
        generic_density = _get_criterion_value(criterion_scores, "generic_phrase_density")
    out["generic_density"] = _clamp01(generic_density)
    status["generic_density"] = _STATUS_OK if out["generic_density"] is not None else _STATUS_UNAVAILABLE

    # 3. predictable_structure ← MEAN of repetitive_structure (criteria/repetitive_structure.py
    #    .value), repeated_sentence_structure_risk, formulaic_conclusion_risk, and
    #    signpost_paragraph_risk (all WritingComponentInputs/ai_components fields already
    #    0-1 "risk_up" signals). MEAN, not max, per task spec: these four are independent
    #    structural cues (starter repetition, sentence-level reuse, conclusion formula,
    #    signposting) that each partially indicate formulaic structure; averaging treats
    #    them as equally-informative evidence rather than letting one strong signal (e.g.
    #    a single very-high signpost_paragraph_risk on an otherwise varied paragraph)
    #    dominate the composite the way max() would. This is a structural aggregation
    #    choice over an existing signal set, not a new tunable weight.
    structure_parts = [
        v for v in (
            _get_criterion_value(criterion_scores, "repetitive_structure"),
            _get(ai, "repeated_sentence_structure_risk"),
            _get(writing, "formulaic_conclusion_risk"),
            _get(writing, "signpost_paragraph_risk"),
        ) if v is not None
    ]
    predictable_structure = _mean(structure_parts)
    out["predictable_structure"] = _clamp01(predictable_structure)
    status["predictable_structure"] = _STATUS_OK if out["predictable_structure"] is not None else _STATUS_UNAVAILABLE

    # 4. sentence_smoothness ← derived from low_burstiness + low_surprisal criteria.
    #    VERIFIED POLARITY (read criteria/burstiness.py and criteria/surprisal.py source):
    #    - low_burstiness.value: "Low std = low burstiness = more AI-like" — the returned
    #      `value` is ALREADY the smooth/AI-like direction (high value = smooth/uniform
    #      rhythm), not a "burstiness" value that needs inverting.
    #    - low_surprisal.value: "higher = more AI-like" (statistically safe/predictable
    #      word choices) — also ALREADY the smooth/AI-like direction.
    #    So "sentence_smoothness" = mean(low_burstiness.value, low_surprisal.value) directly
    #    — NO (1 - x) inversion, despite the task's naive "inverse of burstiness+surprisal"
    #    framing, because both source fields are risk-framed (higher raw risk = higher
    #    smoothness), not variance-framed. Mean of the two available inputs is the
    #    documented aggregation (equal-weight blend of the two smoothness-flavored
    #    predictability criteria), not a new tunable weight.
    smoothness_parts = [
        v for v in (
            _get_criterion_value(criterion_scores, "low_burstiness"),
            _get_criterion_value(criterion_scores, "low_surprisal"),
        ) if v is not None
    ]
    sentence_smoothness = _mean(smoothness_parts)
    out["sentence_smoothness"] = _clamp01(sentence_smoothness)
    status["sentence_smoothness"] = _STATUS_OK if out["sentence_smoothness"] is not None else _STATUS_UNAVAILABLE

    # 5. author_voice_absence ← blend of lived_detail_risk (writing_components; already an
    #    "absence of lived/concrete author detail" RISK, i.e. direct — high = more absent)
    #    and human_anchor_score (transformation_features; a STRENGTH signal, so inverted via
    #    1 - x to become an absence/gap). Mirrors the exact author_anchor_strength derivation
    #    already used in external_grouped_scoring.py/grounding_diagnosis.py
    #    (`max(100 - lived_detail_risk, human_anchor_score)` there is a STRENGTH; V7 wants
    #    absence, so we take the mean of the gap-framed components instead of reusing their
    #    max()-based strength blend, to avoid importing that module's own tuned max() choice
    #    as if it were ours). Mean of the two available absence-framed inputs — no new weight.
    lived_detail_risk = _get(writing, "lived_detail_risk")
    human_anchor_score = _get(features, "human_anchor_score")
    human_anchor_absence = None if human_anchor_score is None else max(0.0, min(1.0, 1.0 - human_anchor_score))
    voice_parts = [v for v in (lived_detail_risk, human_anchor_absence) if v is not None]
    author_voice_absence = _mean(voice_parts)
    out["author_voice_absence"] = _clamp01(author_voice_absence)
    status["author_voice_absence"] = _STATUS_OK if out["author_voice_absence"] is not None else _STATUS_UNAVAILABLE

    # 6. citation_gap ← direct: writing_components.citation_weakness_risk (primary field
    #    on WritingComponentInputs) else the citation_grounding_gap meta-criterion as a
    #    fallback proxy (same fallback poc/detect/citation.py's overall_risk is not a
    #    per-paragraph 0-1 "gap" signal — it's a whole-document bibliography cross-check —
    #    so it is intentionally NOT used here; citation_weakness_risk is the per-paragraph
    #    field the rest of poc/detect/ already threads through writing_components).
    citation_gap = _get(writing, "citation_weakness_risk")
    if citation_gap is None:
        citation_gap = _get_criterion_value(criterion_scores, "citation_grounding_gap")
    out["citation_gap"] = _clamp01(citation_gap)
    status["citation_gap"] = _STATUS_OK if out["citation_gap"] is not None else _STATUS_UNAVAILABLE

    # 7. semantic_drift ← needs comparison text (spec: draft_evolution criterion + semantic_shape
    #    output, comparing this draft against a prior/source draft). Note: semantic_shape.py's
    #    own semantic_drift_risk is computed WITHIN a single document (adjacent-sentence drift),
    #    not against a comparison text — but the V7 spec's semantic_drift signal (§5, feeding the
    #    ai_assisted_polished formula) is explicitly the DRAFT-COMPARISON notion (draft_evolution_
    #    jump_risk on WritingComponentInputs is comparison-dependent). Per the task instructions,
    #    honor has_comparison_text as the explicit gate: without it, return None rather than
    #    silently reusing the intra-document semantic_shape signal, which would conflate two
    #    different notions of "drift".
    if has_comparison_text:
        semantic_drift = _get(writing, "draft_evolution_jump_risk")
        if semantic_drift is None:
            semantic_drift = _get(semantic_shape, "semantic_drift_risk")
        out["semantic_drift"] = _clamp01(semantic_drift)
        status["semantic_drift"] = _STATUS_OK if out["semantic_drift"] is not None else _STATUS_UNAVAILABLE
    else:
        out["semantic_drift"] = None
        status["semantic_drift"] = _STATUS_NO_COMPARISON

    # 8. sentence_variance ← direct: burstiness IS sentence-length/predictability variance
    #    (criteria/burstiness.py docstring: "Burstiness = std(sentence_surprisal_scores)").
    #    Same source field as sentence_smoothness's burstiness term — this is intentional
    #    (spec maps burstiness onto both smoothness's constituent parts and this direct
    #    signal); no inversion, straight pass-through of low_burstiness.value.
    sentence_variance = _get_criterion_value(criterion_scores, "low_burstiness")
    out["sentence_variance"] = _clamp01(sentence_variance)
    status["sentence_variance"] = _STATUS_OK if out["sentence_variance"] is not None else _STATUS_UNAVAILABLE

    # 9. specificity_score ← direct pass-through of criteria/specificity.py score().value.
    specificity_score = _get_criterion_value(criterion_scores, "low_specificity")
    out["specificity_score"] = _clamp01(specificity_score)
    status["specificity_score"] = _STATUS_OK if out["specificity_score"] is not None else _STATUS_UNAVAILABLE

    # 10. local_style_shift ← direct pass-through of criteria/style_shift.py score().value.
    local_style_shift = _get_criterion_value(criterion_scores, "style_shift")
    out["local_style_shift"] = _clamp01(local_style_shift)
    status["local_style_shift"] = _STATUS_OK if out["local_style_shift"] is not None else _STATUS_UNAVAILABLE

    # 11. detector_disagreement ← direct pass-through of external_grouped_scoring's
    #     detector_agreement_risk GROUP score (0-100 in that module; normalized to 0-1 here).
    #     Callers may pass either the raw 0-100 group score or an already-normalized 0-1
    #     value under raw_signals["detector_agreement_risk"]; normalize defensively.
    detector_disagreement = raw_signals.get("detector_agreement_risk")
    if detector_disagreement is None:
        detector_disagreement = _get(ai, "detector_agreement_risk")
    if isinstance(detector_disagreement, bool) or not isinstance(detector_disagreement, (int, float)):
        detector_disagreement = None
    else:
        detector_disagreement = float(detector_disagreement)
        if detector_disagreement > 1.0:
            detector_disagreement = detector_disagreement / 100.0
    out["detector_disagreement"] = _clamp01(detector_disagreement)
    status["detector_disagreement"] = _STATUS_OK if out["detector_disagreement"] is not None else _STATUS_UNAVAILABLE

    # 12. paraphrase_pattern_score — NOT YET BUILT (Phase 1C, MiniLM structural-reuse
    #     pattern signal per weights.json _notes.ai_paraphrased_variants). Do not
    #     approximate from structural_reuse_risk — that field is a different, coarser
    #     signal and substituting it would silently misrepresent an unbuilt detector.
    out["paraphrase_pattern_score"] = None
    status["paraphrase_pattern_score"] = _STATUS_NOT_IMPLEMENTED

    # 13. meaning_preservation_score — NOT YET BUILT (Phase 1C, requires comparison text).
    out["meaning_preservation_score"] = None
    status["meaning_preservation_score"] = _STATUS_NOT_IMPLEMENTED

    # 14. esl_false_positive_risk — NOT YET BUILT per-paragraph (Phase 2 ESL guard is
    #     document-level in weights.json's esl_guard config, not a per-paragraph V7 signal
    #     yet). Included for completeness per spec §5.
    out["esl_false_positive_risk"] = None
    status["esl_false_positive_risk"] = _STATUS_NOT_IMPLEMENTED

    # Assert we produced exactly the declared 14 keys — a structural self-check, not a
    # runtime weight/threshold.
    assert set(out.keys()) == set(V7_SIGNAL_NAMES), (
        f"adapt_paragraph_signals output keys {sorted(out.keys())} != "
        f"V7_SIGNAL_NAMES {sorted(V7_SIGNAL_NAMES)}"
    )

    out["signal_status"] = status
    return out
