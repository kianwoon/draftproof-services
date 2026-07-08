"""V7 pipeline bridge — wires the (already-built and tested) V7 Authorship
Clarity Breakdown modules onto the REAL shape ``poc/report/builder.py``
produces, behind an env-var kill switch. Strictly additive: never touches
tier, ai_likelihood, the badge, or any existing report field.

## Traced real input shape (2026-07-04, read directly from source — not
guessed; see ``poc/report/builder.py`` ~L1267-1420 and ``poc/detect/run.py``
``_collect_criterion_scores``):

- ``poc/report/builder.py`` builds ``ai_components`` and ``writing_components``
  as **document-level, flat dicts on a 0-100 scale** (``{k: round(v * 100, 2)
  for k, v in layer3.ai_phase.components.items()}`` / ``...writing_phase...``).
  There is NO per-paragraph ``ai_components``/``writing_components`` dict
  anywhere in the production pipeline — Layer3Scorer produces one score per
  document, not per paragraph.
- ``criterion_scores`` (``poc/detect/run.py::_collect_criterion_scores``,
  threaded into builder.py as ``self._summaries.get("criterion_scores")``) is
  also **document-level**: a dict of ``CriterionScore(name, value)`` objects
  (``value`` already 0-1), keyed by the exact criteria/*.py ``name=`` literals:
  ``low_burstiness``, ``low_surprisal``, ``low_specificity``, ``style_shift``,
  ``repetitive_structure``, ``generic_phrase_density``, ``structural_reuse``,
  ``source_grounding``, ``citation_grounding_gap``, ``draft_evolution``,
  ``paragraph_uniformity``, ``topk_predictability`` — these match
  ``signal_adapter.py``'s expected ``criterion_scores`` keys exactly.
- ``transformation.features`` (``poc/detect/transformation.py``, via
  ``classify_transformation_from_scan``) is likewise document-level.
- The calibrated composite/deberta detector score used by ``detector_fusion.py``
  is NOT separately available as an isolated raw "fakespot score" object in
  builder.py; the closest existing calibrated score is
  ``ai_risk_badge["ai_likelihood_score"]`` (0-100 scale; already the
  authoritative/calibrated composite — DeBERTa-authoritative when that path
  fires, else perplexity Layer3). This bridge treats that single composite as
  the "composite" fusion input for Phase 1A (quick-scan, 1-detector fusion) —
  it is the only calibrated score builder.py actually exposes at this call
  site (the fusion key was renamed from the historically mislabeled
  "fakespot" — there never was an isolated raw fakespot-detector score at
  this call site, only the badge composite). Using a second, genuinely
  separate detector (e.g. true fakespot raw vs. deberta_large) for
  2-detector fusion is future scope (Modal/deep-scan), NOT implemented here
  — this is stated explicitly rather than fabricated.

## Granularity gap (documented honestly, not silently smoothed over)

Because builder.py has no per-paragraph signal breakdown, this bridge treats
the WHOLE DOCUMENT as a single "paragraph" unit for Phase-1A V7 scoring — one
call to ``adapt_paragraph_signals`` / ``score_paragraph``, word-count-weighted
aggregation over exactly one unit. This is an intentional, documented
simplification: true per-paragraph Authorship Clarity Breakdown requires
per-paragraph ``ai_components``/``writing_components``/``criterion_scores``,
which do not exist in the current pipeline. Do not present this bridge's
output as genuinely per-paragraph; ``breakdown_composer``'s
``paragraph_count`` will be 1.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from poc.detect.deberta_windowing import split_sentences

from . import aggregate, breakdown_composer, category_scoring, config, detector_fusion, modal_client, signal_adapter

logger = logging.getLogger(__name__)

_ENV_VAR = "DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"
_DEEP_SCAN_ENV_VAR = "DRAFTPROOF_V7_DEEP_SCAN"
_TIER_AUTHORITY_ENV_VAR = "DRAFTPROOF_V7_TIER_AUTHORITY"
_TRUTHY = {"1", "true"}

_UNCERTAINTY_FLAG_DEEP_SCAN_UNCALIBRATED = "deep_scan_uncalibrated"
_UNCERTAINTY_FLAG_DEEP_SCAN_BELOW_FLOOR = "deep_scan_below_reliability_floor"
_UNCERTAINTY_FLAG_ESL_GUARD_UNAVAILABLE = "esl_guard_unavailable"
_UNCERTAINTY_FLAG_TIER_CATEGORY_CONTRADICTION = "tier_category_contradiction"
_MIXED_SIGNALS_PRESENTATION = "mixed_signals"
_STUDENT_OWNED_CATEGORY = "student_owned"


def _deep_scan_band(proportion: float) -> str:
    """Map the deep-scan sentence proportion to a display band using the
    weights.json cutoffs (config.get_deep_scan_display_bands — no literals
    here per the no-hardcode rule). Mirrors poc/detect/deberta_signal.py's
    ``_band_for`` philosophy: there is NO "green" band — below the
    reliability floor is "insufficient evidence", never "clean".
    """
    bands = config.get_deep_scan_display_bands()
    if proportion < bands["insufficient_below"]:
        return "insufficient"
    if proportion < bands["orange_min"]:
        return "amber"
    if proportion < bands["red_min"]:
        return "orange"
    return "red"


def is_v7_enabled() -> bool:
    """Kill switch for the V7 Authorship Clarity Breakdown pipeline step.

    Reads ``DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN``. Default is OFF (falsy) —
    the env var must be exactly ``"1"`` or ``"true"`` (case-insensitive,
    surrounding whitespace stripped) to enable. Any other value, including
    unset, empty string, ``"0"``, ``"false"``, or an unrecognized string,
    resolves to disabled. This mirrors the accepted-values contract other V7
    kill switches use (see ``docs/draftproof_v7_authorship_clarity_spec.md``
    §2) but is intentionally stricter than
    ``poc/detect/authenticity_dashboard.py``'s broader truthy set
    (``{"1","true","yes","on"}``) per this task's explicit "1"/"true" spec.
    """
    raw = os.getenv(_ENV_VAR, "")
    return raw.strip().lower() in _TRUTHY


def is_deep_scan_enabled() -> bool:
    """Kill switch for the V7 deep-scan (2-detector, Modal-backed) fusion path.

    Reads ``DRAFTPROOF_V7_DEEP_SCAN`` with the exact same strict "1"/"true"
    truthy contract as ``is_v7_enabled()`` (see that function's docstring).
    Default is OFF. The Modal desklib/ai-text-detector-academic-v1.01
    checkpoint was SCoCESLE-calibrated 2026-07-04 via sentence
    threshold-proportion (see ``weights.json``'s
    ``deep_scan_calibration._provenance``); this env var is now an
    operational kill switch (cost/latency of the paid Modal call), not a
    calibration gate — the runtime per-scan ``uncalibrated`` flag
    (``get_deep_scan_proportion``'s return value) reflects whether the
    Modal endpoint reports the calibrated checkpoint tag for that call.
    """
    raw = os.getenv(_DEEP_SCAN_ENV_VAR, "")
    return raw.strip().lower() in _TRUTHY


def is_tier_authority_enabled() -> bool:
    """Kill switch for the V7 fused-score tier-authority re-base.

    Reads ``DRAFTPROOF_V7_TIER_AUTHORITY`` with the exact same strict
    "1"/"true" truthy contract as ``is_v7_enabled()``/``is_deep_scan_enabled()``
    (case-insensitive, whitespace-stripped; anything else, including unset,
    resolves to disabled). Default is OFF — when off, the badge's tier and
    ai_likelihood_score are computed exactly as before this feature existed
    (byte-identical output).
    """
    raw = os.getenv(_TIER_AUTHORITY_ENV_VAR, "")
    return raw.strip().lower() in _TRUTHY


def _tier_for_fused_score(fused_score: float, cutoffs: dict[str, float]) -> str:
    if fused_score >= cutoffs["red"]:
        return "red"
    if fused_score >= cutoffs["orange"]:
        return "orange"
    if fused_score >= cutoffs["amber"]:
        return "amber"
    return "green"


def compute_fused_authority(composite_0_100: float, proportion_0_1: float) -> dict[str, Any]:
    """Pure function: fuse the composite badge score with the deep-scan
    sentence proportion into the V7 tier-authority score + tier.

    ``fused_score = weights.composite * composite_0_100 +
    weights.deep_scan_proportion * proportion_0_1 * 100`` per
    ``weights.json``'s ``tier_authority`` section (poc/calibration/
    v7_fused_gate_result.json — GATE PASS 2026-07-04). Tier is derived from
    the SAME cutoffs (32/48/65) already used for the composite-only scale;
    the cutoff sweep validated they carry over unchanged on the fused scale.

    Parameters
    ----------
    composite_0_100: the existing composite ``ai_likelihood_score`` (0-100).
    proportion_0_1: the deep-scan sentence-level flagged proportion (0-1).

    Returns
    -------
    ``{"fused_score": float (0-100, 2dp), "tier": "green"|"amber"|"orange"|"red"}``

    Raises ``ValueError`` if either input is not a finite number in its
    expected range — this function never silently clamps out-of-range
    inputs into a fake tier; the caller (``run_v7_breakdown``) is
    responsible for fail-open behavior (skip the override, not call this
    function with malformed inputs).
    """
    if isinstance(composite_0_100, bool) or not isinstance(composite_0_100, (int, float)):
        raise ValueError(f"composite_0_100 must be a number, got {composite_0_100!r}")
    if isinstance(proportion_0_1, bool) or not isinstance(proportion_0_1, (int, float)):
        raise ValueError(f"proportion_0_1 must be a number, got {proportion_0_1!r}")
    if not (0.0 <= float(composite_0_100) <= 100.0):
        raise ValueError(f"composite_0_100 must be in [0, 100], got {composite_0_100!r}")
    if not (0.0 <= float(proportion_0_1) <= 1.0):
        raise ValueError(f"proportion_0_1 must be in [0, 1], got {proportion_0_1!r}")

    tier_authority = config.get_tier_authority_config()
    weights = tier_authority["weights"]
    cutoffs = tier_authority["cutoffs"]

    fused_score = (
        weights["composite"] * float(composite_0_100)
        + weights["deep_scan_proportion"] * float(proportion_0_1) * 100.0
    )
    fused_score = round(fused_score, 2)
    tier = _tier_for_fused_score(fused_score, cutoffs)
    return {"fused_score": fused_score, "tier": tier}


def get_deep_scan_proportion(detection_result: Any) -> Optional[dict[str, Any]]:
    """Best-effort deep-scan sentence-proportion lookup, shared by
    ``run_v7_breakdown`` (2-detector fusion input) and the builder-level
    fused tier-authority override (``compute_fused_authority`` needs this
    SAME proportion so the two never disagree).

    Returns ``None`` when deep scan is disabled, document text is
    unavailable, sentence splitting produces nothing, or the Modal call
    fails/returns a malformed/unavailable response — fail-open, never
    raises. On success returns a dict with the fields ``run_v7_breakdown``
    and the builder override both need: ``proportion`` (0-1),
    ``uncalibrated`` (bool), ``below_floor`` (bool), ``payload`` (the
    display-ready deep_scan dict).
    """
    # Short-circuit: the builder's tier-authority block runs first and hands its
    # Modal result through as _precomputed_deep_scan — reuse it so a scan with
    # both flags on pays for ONE deep-scan call, not two identical ones.
    if isinstance(detection_result, dict):
        precomputed = detection_result.get("_precomputed_deep_scan")
        if isinstance(precomputed, dict) and "proportion" in precomputed:
            return precomputed
    if not is_deep_scan_enabled():
        return None
    document_text = _extract_document_text(detection_result)
    if not document_text:
        logger.info(
            "detect_v7.pipeline_bridge: deep scan enabled but no document text "
            "available on detection_result; skipping deep-scan proportion."
        )
        return None
    sentences = [s for s in split_sentences(document_text) if s.strip()]
    if not sentences:
        logger.info(
            "detect_v7.pipeline_bridge: deep scan enabled but document text "
            "produced no sentences; skipping deep-scan proportion."
        )
        return None
    modal_response = modal_client.call_deep_scan(sentences)
    chunk_scores = modal_response.get("chunk_scores") if isinstance(modal_response, dict) else None
    if not (
        isinstance(modal_response, dict)
        and modal_response.get("available") is True
        and isinstance(chunk_scores, list)
        and len(chunk_scores) == len(sentences)
        and all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in chunk_scores)
    ):
        logger.info(
            "detect_v7.pipeline_bridge: Modal deep scan unavailable/failed/malformed; "
            "skipping deep-scan proportion."
        )
        return None

    calibration = config.get_deep_scan_calibration()
    sent_threshold = calibration["sent_threshold"]
    doc_floor = calibration["doc_floor"]
    flagged = sum(1 for s in chunk_scores if s >= sent_threshold)
    proportion = flagged / len(chunk_scores)
    deberta_score = max(0.0, min(1.0, float(proportion)))
    payload = {
        "proportion": deberta_score,
        "band": _deep_scan_band(deberta_score),
        "calibrated": modal_response.get("calibrated") is True,
    }
    # Additive per-paragraph proportions: pure post-processing of the SAME
    # per-sentence Modal scores (zero extra Modal cost, document math above
    # untouched). Fail-open: a mapping failure just omits the key.
    paragraph_rows = _per_paragraph_proportions(
        document_text, sentences, chunk_scores, sent_threshold
    )
    if paragraph_rows:
        payload["paragraphs"] = paragraph_rows
        # Surfaces explain "insufficient evidence" with the actual arithmetic
        # ("1 of 4 flagged · 25%, below the 30% reliability floor") — the
        # floor must reach them as DATA (weights.json doc_floor), never as a
        # literal in frontend/PDF code (no-hardcode rule).
        payload["reliability_floor"] = doc_floor
    return {
        "proportion": deberta_score,
        "uncalibrated": modal_response.get("calibrated") is not True,
        "below_floor": proportion < doc_floor,
        "payload": payload,
    }


def _per_paragraph_proportions(
    document_text: str,
    sentences: list[str],
    chunk_scores: list,
    sent_threshold: float,
) -> Optional[list[dict[str, Any]]]:
    """Group the existing per-sentence deep-scan scores by paragraph.

    Mapping construction (deterministic, no re-splitting): ``split_sentences``
    normalizes ALL whitespace to single spaces before splitting, and
    ``split_paragraphs`` (poc/detect/layer3_scoring.py) produces the same
    normalization per paragraph block — so ``" ".join(paragraphs)``
    reconstructs exactly the normalized string the sentences were split from.
    Each sentence is located in that normalized string with a forward cursor
    (sentences appear in order) and assigned to the paragraph whose
    normalized char range contains its start. The document-level proportion
    is NOT recomputed here — this is presentation-side grouping only.

    Bands reuse the document display-band cutoffs (weights.json — no new
    constants); short paragraphs are naturally noisy, so ``sentence_count``
    rides along for consumers to judge reliability. ``sentence_count`` is the
    CANONICAL per-paragraph count (structured_sentence_segments — the same
    source the report's sentence_map and on-page paragraph card use), so the
    deep-scan table matches the card; ``flagged_count`` (and thus each row's
    ``proportion``) are the deep-scan detector's own reading, re-based onto
    that canonical denominator and clamped to it. Returns None (key omitted)
    when there are fewer than 2 paragraphs or any mapping inconsistency —
    fail-open, never raises.

    Heading blocks are EXCLUDED from the emitted rows (their sentences still
    count in the document-level proportion, unchanged): a standalone title
    like "Critical Analysis: ..." is one blank-line block, so without this it
    surfaced as "Paragraph 1 — 100% (1 sentence)" and shifted every body
    paragraph's number by one versus what the user sees as paragraphs
    (observed on a live report 2026-07-06: panel showed 4 paragraphs for a
    3-paragraph essay with a title). Heading detection reuses the SAME
    heuristic as the report's display segmentation
    (poc/detect/document_structure._looks_like_heading), applied only to
    single-line blocks. ``index`` is therefore the body-paragraph ordinal
    (0-based), matching how a reader counts paragraphs.
    """
    try:
        import re as _re

        from poc.detect.document_structure import (
            _looks_like_heading,
            structured_sentence_segments,
        )
        from poc.detect.layer3_scoring import split_paragraphs

        paragraphs = split_paragraphs(document_text)
        if len(paragraphs) < 2 or len(sentences) != len(chunk_scores):
            return None
        # Canonical per-paragraph sentence counts — the SAME segmentation the
        # report's sentence_map and the on-page paragraph card use
        # (structured_sentence_segments). The Modal deep-scan detector runs its
        # OWN sentence splitter (deberta_windowing.split_sentences), which
        # over-segments some paragraphs (a lowercase name after a period, an
        # abbreviation, etc.), so the DISPLAY denominators drifted away from the
        # card — a 3-sentence paragraph surfaced as "1 of 4". These counts,
        # aligned block-for-block to split_paragraphs (both segment the same
        # blank-line blocks in document order), re-base ONLY the displayed
        # sentence_count so the deep-scan table matches the card. The
        # document-level proportion computed by the caller is NOT touched.
        # Fail-open: any shape mismatch keeps the detector's own counts.
        canonical_counts = None
        try:
            from collections import OrderedDict as _OrderedDict

            _groups = _OrderedDict()
            for _seg in structured_sentence_segments(document_text):
                _pid = _seg.get("paragraph_id") or ""
                _groups[_pid] = _groups.get(_pid, 0) + 1
            _counts = list(_groups.values())
            if len(_counts) == len(paragraphs):
                canonical_counts = _counts
        except Exception:
            canonical_counts = None
        # Raw blocks via the SAME split regex as split_paragraphs, same
        # filtering — indices stay aligned with the normalized list. A block
        # is a heading only if it is a single raw line that passes the
        # display segmentation's heading heuristic.
        raw_blocks = [b for b in _re.split(r"\n\s*\n+", document_text.strip()) if b.strip()]
        is_heading = [
            "\n" not in b.strip() and _looks_like_heading(b.strip())
            for b in raw_blocks
        ] if len(raw_blocks) == len(paragraphs) else [False] * len(paragraphs)
        norm = " ".join(paragraphs)
        # Paragraph k covers norm[start_k : start_k + len(p)] (+1 joiner space).
        ranges = []
        pos = 0
        for p in paragraphs:
            ranges.append((pos, pos + len(p)))
            pos += len(p) + 1
        per_paragraph: list[list[float]] = [[] for _ in paragraphs]
        cursor = 0
        para_idx = 0
        for sentence, score in zip(sentences, chunk_scores):
            start = norm.find(sentence, cursor)
            if start < 0:
                return None  # mapping inconsistency — omit rather than guess
            cursor = start + len(sentence)
            while para_idx < len(ranges) - 1 and start >= ranges[para_idx][1]:
                para_idx += 1
            target = para_idx
            if is_heading[target]:
                # An unpunctuated title merges with the first body sentence in
                # the normalized sentence stream (split_sentences collapses the
                # paragraph break, and there is no terminal punctuation to
                # split on). Such a sentence STARTS in the heading block but
                # extends past it — it is body content and must count toward
                # the next body paragraph, or the paragraph's flagged count
                # contradicts the sentence underlines (observed live
                # 2026-07-06: paragraph 1 showed '0 of 4 flagged' while its
                # first sentence was visibly flagged). A sentence fully inside
                # the heading range is title-only text: dropped from rows
                # (still counted in the document-level proportion above).
                if start + len(sentence) > ranges[target][1]:
                    while target < len(ranges) - 1:
                        target += 1
                        if not is_heading[target]:
                            break
                    if is_heading[target]:
                        continue  # headings all the way down — drop from rows
                else:
                    continue  # title-only sentence — not a body row member
            per_paragraph[target].append(float(score))
        rows = []
        body_ordinal = 0
        for i, scores in enumerate(per_paragraph):
            if not scores:
                continue
            if is_heading[i]:
                continue  # titles/headings are not paragraphs — see docstring
            p_flagged = sum(1 for s in scores if s >= sent_threshold)
            # Prefer the canonical (card) sentence count as the denominator so
            # the deep-scan table lines up with the on-page paragraph card;
            # fall back to the detector's own count when the two segmentations
            # don't line up (fail-open). Clamp flagged to the denominator so the
            # "X of Y" display can never exceed Y after re-basing. Only the
            # display is re-based — the document-level proportion is unchanged.
            denom = len(scores)
            if canonical_counts is not None and canonical_counts[i] > 0:
                denom = canonical_counts[i]
            p_flagged = min(p_flagged, denom)
            p_prop = p_flagged / denom
            rows.append(
                {
                    "index": body_ordinal,
                    "sentence_count": denom,
                    "flagged_count": p_flagged,
                    "proportion": round(p_prop, 4),
                    "band": _deep_scan_band(p_prop),
                }
            )
            body_ordinal += 1
        # A single body paragraph (e.g. title + one paragraph) is not a
        # breakdown — same rationale as the <2-paragraph early return.
        return rows if len(rows) >= 2 else None
    except Exception:
        logger.exception(
            "detect_v7.pipeline_bridge: per-paragraph deep-scan grouping failed; "
            "omitting paragraphs (additive, non-fatal)."
        )
        return None


def _extract_document_text(detection_result: Any) -> Optional[str]:
    """Best-effort extraction of raw document text for the Modal deep-scan
    call. The ``ai_risk_badge`` dict this bridge normally receives (see
    module docstring) does NOT carry raw text at this call site — it is
    document-level scores only. This checks the small set of plausible keys
    a caller might attach; if none are present, deep scan is skipped
    (fail-open, not fatal — falls back to the quick-scan path).
    """
    candidates = ("text", "document_text", "source_text", "full_text")
    if isinstance(detection_result, dict):
        for key in candidates:
            value = detection_result.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None
    for key in candidates:
        value = getattr(detection_result, key, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_ai_components(detection_result: Any) -> Optional[dict[str, Any]]:
    if isinstance(detection_result, dict):
        return detection_result.get("ai_components")
    return getattr(detection_result, "ai_components", None)


def _extract_writing_components(detection_result: Any) -> Optional[dict[str, Any]]:
    if isinstance(detection_result, dict):
        return detection_result.get("writing_components")
    return getattr(detection_result, "writing_components", None)


def _extract_criterion_scores(detection_result: Any) -> Optional[dict[str, Any]]:
    if isinstance(detection_result, dict):
        return detection_result.get("criterion_scores")
    return getattr(detection_result, "criterion_scores", None)


def _extract_transformation_features(detection_result: Any) -> Optional[dict[str, Any]]:
    if isinstance(detection_result, dict):
        tc = detection_result.get("transformation_classification") or {}
        return tc.get("features")
    tc = getattr(detection_result, "transformation_classification", None)
    if isinstance(tc, dict):
        return tc.get("features")
    return getattr(tc, "features", None) if tc is not None else None


def _extract_tier(detection_result: Any) -> Optional[str]:
    """Best-effort extraction of the badge's ``tier`` string (e.g.
    ``"clean"``/``"acceptable"``/``"concerning"``/``"strong"`` — see
    ``poc/report/builder.py``'s ``ai_risk_badge["tier"]``). Same dict/attr
    fallback pattern as the other ``_extract_*`` helpers in this module.
    Returns ``None`` when absent/malformed — callers that gate on tier must
    fail-open (no guard action) rather than raise.
    """
    if isinstance(detection_result, dict):
        value = detection_result.get("tier")
    else:
        value = getattr(detection_result, "tier", None)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def _extract_calibrated_score(detection_result: Any) -> Optional[float]:
    """Pull the composite ``ai_likelihood_score`` (0-100 scale in builder.py's
    ``ai_risk_badge``) and normalize to 0-1 for ``detector_fusion.py``.
    """
    if isinstance(detection_result, dict):
        value = detection_result.get("ai_likelihood_score")
    else:
        value = getattr(detection_result, "ai_likelihood_score", None)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _extract_qualifying_word_count(detection_result: Any) -> Optional[int]:
    """Best-effort qualifying word count for the aggregation weight. Falls
    back to 1 (single unit, uniform weight) when not resolvable — the
    document-level-only aggregation this bridge does (see module docstring)
    only has one unit anyway, so the exact count does not change the output,
    only whether an honest word count is threaded through for future
    multi-paragraph support.
    """
    if isinstance(detection_result, dict):
        count = detection_result.get("qualifying_word_count") or detection_result.get("word_count")
    else:
        count = getattr(detection_result, "qualifying_word_count", None) or getattr(
            detection_result, "word_count", None
        )
    if isinstance(count, bool) or not isinstance(count, (int, float)) or count <= 0:
        return None
    return int(count)


def _normalize_components_to_unit_scale(components: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """builder.py's ``ai_components``/``writing_components`` are 0-100 scale;
    ``signal_adapter.py`` expects 0-1. Normalize defensively (values already
    <=1 are left as-is, on the assumption they're already unit-scale — this
    only rescales values that are clearly percentage-scale).
    """
    if not isinstance(components, dict):
        return None
    normalized: dict[str, Any] = {}
    for key, value in components.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        normalized[key] = value / 100.0 if value > 1.0 else value
    return normalized


def _build_raw_signals(detection_result: Any) -> dict[str, Any]:
    """Reshape the real (document-level) builder.py signal shapes into the
    per-unit ``raw_signals`` dict ``signal_adapter.adapt_paragraph_signals``
    expects. See module docstring for the traced field names and the
    documented single-unit granularity gap.
    """
    return {
        "ai_components": _normalize_components_to_unit_scale(_extract_ai_components(detection_result)),
        "writing_components": _normalize_components_to_unit_scale(_extract_writing_components(detection_result)),
        "transformation_features": _extract_transformation_features(detection_result),
        "criterion_scores": _extract_criterion_scores(detection_result),
        "semantic_shape": None,  # not threaded through builder.py at this call site
        "has_comparison_text": False,  # Phase 1A quick-scan path has no comparison text
    }


def _compose_display_fallback(breakdown: dict[str, Any]) -> None:
    """V8 three-way display fallback (owner decision 2026-07-08, evidence
    ``calibration/v12_validation/v8_frontier_result.json``): ``ai_paraphrased``
    is measurably inseparable from ``ai_generated_like`` on single-document
    evidence, so the user-facing breakdown MERGES the two indistinguishable AI
    indicators into one ``ai_transformed`` display category.

    Strictly ADDITIVE + presentation-only. Mutates ``breakdown`` in place to
    add three fields — ``display_taxonomy``, ``display_shares``,
    ``display_primary`` — computed from the existing four-way
    ``document_breakdown_raw`` shares. The four-way fields
    (``document_breakdown_raw``/``document_breakdown_bands``/
    ``primary_category``) are NEVER touched; they are retained for audit and
    the V8b paired-draft feature family that may later recover the separation.

    Runs BEFORE ``_apply_tier_consistency_guard`` so the guard can inspect
    ``display_primary`` (see that function). Config-driven, no hardcode:
    ``config.get_display_fallback_config()`` supplies the merge spec and the
    on/off ``mode``. When ``mode != "three_way"`` (e.g. the ``"four_way"``
    off-switch) NO ``display_*`` fields are emitted — a forward-compatible
    kill switch that leaves the four-way breakdown byte-identical.
    """
    fallback = config.get_display_fallback_config()
    if fallback["mode"] != "three_way":
        return

    merged_from = fallback["merged_from"]
    merged_category = fallback["merged_display_category"]
    raw_shares = breakdown["document_breakdown_raw"]

    display_shares: dict[str, float] = {
        category: share
        for category, share in raw_shares.items()
        if category not in merged_from
    }
    display_shares[merged_category] = sum(raw_shares[c] for c in merged_from)

    # argmax of the merged shares; ties break on first-seen order, matching
    # aggregate.aggregate_document's max(...) convention for primary_category.
    display_primary = max(display_shares.items(), key=lambda item: item[1])[0]

    breakdown["display_taxonomy"] = fallback["mode"]
    breakdown["display_shares"] = display_shares
    breakdown["display_primary"] = display_primary


def _apply_tier_consistency_guard(breakdown: dict[str, Any], tier: Optional[str]) -> None:
    """Owner-approved tier-consistency display guard (companion to the
    2026-07-08 category_weights re-tune, see ``weights.json``'s
    ``display_consistency_guard._notes``): a badge tier in the configured
    trigger set (``concerning``/``strong`` by default) paired with a
    ``student_owned`` primary read is a contradictory display — red tier +
    "the writing looks student-owned". Per the project's "guards ANNOTATE,
    never SUPPRESS" alignment principle, this NEVER touches
    ``document_breakdown_raw``/``document_breakdown_bands``/
    ``primary_category`` — it only reuses the existing mixed_signals
    presentation + ``primary_category_reliable=False`` mechanism
    (``breakdown_composer``'s flatness guard already uses this same
    ``"mixed_signals"`` value) and appends an uncertainty flag.

    The "student_owned primary read" it challenges is EITHER the four-way
    ``primary_category`` OR — once the V8 three-way display fallback has run
    (``_compose_display_fallback``, called just before this guard) — the
    ``display_primary`` shown to the user. The merge can only ever move the
    top category AWAY from ``student_owned`` (it pools the two AI indicators
    into ``ai_transformed``), so ``display_primary == student_owned`` implies
    ``primary_category == student_owned`` too; the extra ``display_primary``
    check is defensive and future-proof (it fires even if a later merge spec
    changes that relationship) rather than strictly additive coverage today.

    Fail-open by construction: mutates ``breakdown`` in place only when
    ``tier`` is a non-``None`` string in the configured trigger set AND the
    four-way primary OR the display primary is ``"student_owned"``; a
    missing/unresolved tier (e.g. the caller's ``detection_result`` never
    carried one) silently does nothing, matching this bridge's overall
    fail-open contract.
    """
    if tier is None:
        return
    student_owned_read = (
        breakdown.get("primary_category") == _STUDENT_OWNED_CATEGORY
        or breakdown.get("display_primary") == _STUDENT_OWNED_CATEGORY
    )
    if not student_owned_read:
        return
    guard_config = config.get_display_consistency_guard_config()
    trigger_tiers = guard_config["student_owned_contradiction_tiers"]
    if tier not in trigger_tiers:
        return
    breakdown["presentation"] = _MIXED_SIGNALS_PRESENTATION
    breakdown["primary_category_reliable"] = False
    flags = breakdown.setdefault("uncertainty_flags", [])
    if _UNCERTAINTY_FLAG_TIER_CATEGORY_CONTRADICTION not in flags:
        flags.append(_UNCERTAINTY_FLAG_TIER_CATEGORY_CONTRADICTION)


def run_v7_breakdown(detection_result: Any) -> Optional[dict[str, Any]]:
    """Compute the V7 Authorship Clarity Breakdown for one scan's detection
    result, or return ``None`` if disabled/unavailable/failed.

    Parameters
    ----------
    detection_result: the object/dict produced by the detection pipeline at
        the point the caller has ``ai_components``, ``writing_components``,
        ``criterion_scores``, and ``ai_likelihood_score`` available (in
        practice: the ``ai_risk_badge`` dict assembled in
        ``poc/report/builder.py``, or any dict/object exposing the same
        attribute/key names — see module docstring for the exact traced
        shape). Accepts both a plain dict and an attribute-bearing object.

    Returns
    -------
    ``compose_authorship_breakdown()``'s output dict, or ``None`` when:
    - the kill switch (``is_v7_enabled()``) is off (fail-fast, no work done);
    - there isn't enough signal to compute a calibrated detector score or any
      V7 signals (insufficient input);
    - ANY exception occurs anywhere in the computation.

    The returned breakdown always carries ``"granularity": "document"``:
    per-paragraph V7 scoring is not yet implemented (the whole document is
    treated as one "paragraph" unit — see module docstring's "Granularity
    gap" section), so this value is hardwired until true per-paragraph
    scoring exists.

    The returned breakdown's ``uncertainty_flags`` always includes
    ``"esl_guard_unavailable"``: no per-document ESL-likelihood estimator
    exists yet (unbuilt Phase-2 work), so ``esl_score`` is always passed as
    ``None`` to ``category_scoring.score_paragraph`` and the esl_guard
    damping/co-trigger logic in ``weights.json`` can never fire.

    Deliberate exception-swallowing: this function wraps its entire body in a
    broad ``try/except Exception`` and returns ``None`` instead of raising,
    logging the failure via the standard ``logging`` module. This is
    justified ONLY because the V7 breakdown is strictly additive, optional,
    and kill-switched — a bug here must never break or delay the scan report
    it decorates. This is not a general-purpose error-hiding pattern; it
    mirrors the same fail-open contract used by
    ``poc/detect/authenticity_dashboard.py::maybe_attach`` and
    ``poc/detect/submission_risk.py``'s composer entry points.
    """
    if not is_v7_enabled():
        return None

    try:
        calibrated_score = _extract_calibrated_score(detection_result)
        if calibrated_score is None:
            logger.info(
                "detect_v7.pipeline_bridge: no calibrated ai_likelihood_score on "
                "detection_result; skipping V7 breakdown."
            )
            return None

        deep_scan_uncalibrated = False
        deep_scan_below_floor = False
        deep_scan_payload: Optional[dict[str, Any]] = None
        detector_scores = {"composite": calibrated_score}
        deep_scan_result = get_deep_scan_proportion(detection_result)
        if deep_scan_result is not None:
            detector_scores = {"composite": calibrated_score, "deberta_large": deep_scan_result["proportion"]}
            deep_scan_uncalibrated = deep_scan_result["uncalibrated"]
            deep_scan_below_floor = deep_scan_result["below_floor"]
            deep_scan_payload = deep_scan_result["payload"]

        fused_score, _fusion_detail = detector_fusion.compute_calibrated_detector_score(detector_scores)

        raw_signals = _build_raw_signals(detection_result)
        # Thread the fused calibrated detector score into raw_signals so the
        # adapter can derive the detector-gated specificity split
        # (specificity_student_evidence / specificity_ai_evidence). fused_score
        # is computed just above, before this point — the order is intentional
        # and the capture_signals replica mirrors it exactly (offline == e2e).
        raw_signals["calibrated_detector_score"] = fused_score
        if not raw_signals.get("ai_components") and not raw_signals.get("writing_components"):
            logger.info(
                "detect_v7.pipeline_bridge: no ai_components/writing_components on "
                "detection_result; skipping V7 breakdown."
            )
            return None

        v7_signals = signal_adapter.adapt_paragraph_signals(raw_signals)
        # esl_score is always None: no per-document ESL-likelihood estimator
        # exists in this codebase yet (unbuilt Phase-2 work, spec §5, row
        # `esl_false_positive_risk`). ESL protection currently lives in
        # detector-level corpus calibration (fakespot isotonic + deep-scan
        # SCoCESLE thresholds), not in this per-paragraph esl_guard damping/
        # co-trigger logic, which can therefore never fire in practice. The
        # esl_guard_unavailable uncertainty flag (appended below) surfaces
        # that honestly instead of silently no-oping.
        paragraph_result = category_scoring.score_paragraph(
            v7_signals,
            fused_score,
            has_comparison_text=bool(raw_signals.get("has_comparison_text")),
            esl_score=None,
        )

        word_count = _extract_qualifying_word_count(detection_result) or 1
        document_aggregate = aggregate.aggregate_document([paragraph_result], [word_count])

        breakdown = breakdown_composer.compose_authorship_breakdown(document_aggregate, [paragraph_result])
        # Granularity honesty: this bridge treats the whole document as ONE
        # paragraph unit (see module docstring's "Granularity gap" section) --
        # per-paragraph V7 scoring is not yet implemented, so this value is
        # hardwired to "document" until it is. "document" is a factual
        # descriptive label, not a tunable numeric constant.
        breakdown["granularity"] = "document"
        if deep_scan_uncalibrated:
            flags = breakdown.setdefault("uncertainty_flags", [])
            if _UNCERTAINTY_FLAG_DEEP_SCAN_UNCALIBRATED not in flags:
                flags.append(_UNCERTAINTY_FLAG_DEEP_SCAN_UNCALIBRATED)
        if deep_scan_below_floor:
            flags = breakdown.setdefault("uncertainty_flags", [])
            if _UNCERTAINTY_FLAG_DEEP_SCAN_BELOW_FLOOR not in flags:
                flags.append(_UNCERTAINTY_FLAG_DEEP_SCAN_BELOW_FLOOR)
        if deep_scan_payload is not None:
            # Additive, same pattern as the uncertainty_flags appends above:
            # only present when the deep scan actually succeeded (frontend
            # null-checks the key's absence for disabled/failed/no-text).
            breakdown["deep_scan"] = deep_scan_payload
        # esl_score is always None today (see comment at the score_paragraph
        # call site above) -- always append this flag so callers are never
        # given a false impression that the ESL guard is protecting them.
        flags = breakdown.setdefault("uncertainty_flags", [])
        if _UNCERTAINTY_FLAG_ESL_GUARD_UNAVAILABLE not in flags:
            flags.append(_UNCERTAINTY_FLAG_ESL_GUARD_UNAVAILABLE)
        # V8 three-way display fallback: compose display_* fields from the
        # four-way shares BEFORE the guard runs, so the extended guard can
        # challenge a student_owned display_primary too (additive; four-way
        # fields untouched — see _compose_display_fallback).
        _compose_display_fallback(breakdown)
        _apply_tier_consistency_guard(breakdown, _extract_tier(detection_result))
        return breakdown
    except Exception:
        logger.exception("detect_v7.pipeline_bridge: run_v7_breakdown failed; returning None (additive, non-fatal).")
        return None
