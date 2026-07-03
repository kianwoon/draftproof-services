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
- The calibrated fakespot/deberta detector score used by ``detector_fusion.py``
  is NOT separately available as an isolated "fakespot score" object in
  builder.py; the closest existing calibrated score is
  ``ai_risk_badge["ai_likelihood_score"]`` (0-100 scale; already the
  authoritative/calibrated composite — DeBERTa-authoritative when that path
  fires, else perplexity Layer3). This bridge treats that single composite as
  the "fakespot" fusion input for Phase 1A (quick-scan, 1-detector fusion) —
  it is the only calibrated score builder.py actually exposes at this call
  site. Using a second, genuinely separate detector (e.g. true fakespot raw
  vs. deberta_large) for 2-detector fusion is future scope (Modal/deep-scan),
  NOT implemented here — this is stated explicitly rather than fabricated.

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
_TRUTHY = {"1", "true"}

_UNCERTAINTY_FLAG_DEEP_SCAN_UNCALIBRATED = "deep_scan_uncalibrated"
_UNCERTAINTY_FLAG_DEEP_SCAN_BELOW_FLOOR = "deep_scan_below_reliability_floor"


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
    Default is OFF. MUST stay OFF in production until the Modal
    ``deberta_large`` checkpoint is SCoCESLE-calibrated (see
    ``modal_endpoints/README.md`` "Remaining Task 1B.0 work") — this function
    only reads the env var, it never hardcodes a default-on value.
    """
    raw = os.getenv(_DEEP_SCAN_ENV_VAR, "")
    return raw.strip().lower() in _TRUTHY


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
        detector_scores = {"fakespot": calibrated_score}
        if is_deep_scan_enabled():
            document_text = _extract_document_text(detection_result)
            if document_text:
                sentences = [s for s in split_sentences(document_text) if s.strip()]
                if not sentences:
                    logger.info(
                        "detect_v7.pipeline_bridge: deep scan enabled but document text "
                        "produced no sentences; falling back to 1-detector quick-scan fusion."
                    )
                else:
                    modal_response = modal_client.call_deep_scan(sentences)
                    chunk_scores = (
                        modal_response.get("chunk_scores") if isinstance(modal_response, dict) else None
                    )
                    if (
                        isinstance(modal_response, dict)
                        and modal_response.get("available") is True
                        and isinstance(chunk_scores, list)
                        and len(chunk_scores) == len(sentences)
                        and all(
                            isinstance(s, (int, float)) and not isinstance(s, bool) for s in chunk_scores
                        )
                    ):
                        calibration = config.get_deep_scan_calibration()
                        sent_threshold = calibration["sent_threshold"]
                        doc_floor = calibration["doc_floor"]
                        flagged = sum(1 for s in chunk_scores if s >= sent_threshold)
                        proportion = flagged / len(chunk_scores)
                        deberta_score = max(0.0, min(1.0, float(proportion)))
                        detector_scores = {"fakespot": calibrated_score, "deberta_large": deberta_score}
                        deep_scan_uncalibrated = modal_response.get("calibrated") is not True
                        deep_scan_below_floor = proportion < doc_floor
                        deep_scan_payload = {
                            "proportion": deberta_score,
                            "band": _deep_scan_band(deberta_score),
                            "calibrated": modal_response.get("calibrated") is True,
                        }
                    else:
                        logger.info(
                            "detect_v7.pipeline_bridge: Modal deep scan unavailable/failed/malformed; "
                            "falling back to 1-detector quick-scan fusion."
                        )
            else:
                logger.info(
                    "detect_v7.pipeline_bridge: deep scan enabled but no document text "
                    "available on detection_result; falling back to 1-detector quick-scan fusion."
                )

        fused_score, _fusion_detail = detector_fusion.compute_calibrated_detector_score(detector_scores)

        raw_signals = _build_raw_signals(detection_result)
        if not raw_signals.get("ai_components") and not raw_signals.get("writing_components"):
            logger.info(
                "detect_v7.pipeline_bridge: no ai_components/writing_components on "
                "detection_result; skipping V7 breakdown."
            )
            return None

        v7_signals = signal_adapter.adapt_paragraph_signals(raw_signals)
        paragraph_result = category_scoring.score_paragraph(
            v7_signals,
            fused_score,
            has_comparison_text=bool(raw_signals.get("has_comparison_text")),
            esl_score=None,
        )

        word_count = _extract_qualifying_word_count(detection_result) or 1
        document_aggregate = aggregate.aggregate_document([paragraph_result], [word_count])

        breakdown = breakdown_composer.compose_authorship_breakdown(document_aggregate, [paragraph_result])
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
        return breakdown
    except Exception:
        logger.exception("detect_v7.pipeline_bridge: run_v7_breakdown failed; returning None (additive, non-fatal).")
        return None
