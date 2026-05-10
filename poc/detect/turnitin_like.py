"""Shared Turnitin-like AI footprint scoring.

This module is the single source for the rewrite/scanner aligned score:

  0.45 * AI likelihood
+ 0.20 * calibrated Top-k risk
+ 0.12 * semantic uniformity
+ 0.10 * rewrite smoothness
+ 0.08 * patchwork / expansion
+ 0.05 * signal agreement
- human-anchor suppression

Inputs may be decimals or percentages. Output components are percentages.
"""

from __future__ import annotations

from typing import Any, Dict

from detect.topk_calibration import calibrate_topk_risk


TURNITIN_LIKE_SCORE_VERSION = "turnitin_like_v1"
TURNITIN_LIKE_SAFE_BAND = 35.0
TURNITIN_LIKE_COMPONENT_WEIGHTS = {
    "ai_likelihood": 0.45,
    "topk_calibrated_risk": 0.20,
    "semantic_uniformity": 0.12,
    "rewrite_smoothness": 0.10,
    "patchwork_expansion": 0.08,
    "signal_agreement": 0.05,
}


def _percent(value: Any, default: float = 0.0) -> float:
    if not isinstance(value, (int, float)):
        return float(default)
    numeric = float(value)
    if abs(numeric) <= 1.0:
        numeric *= 100.0
    return max(0.0, min(100.0, numeric))


def turnitin_like_ai_profile(
    *,
    features: Dict[str, Any] | None = None,
    ai_components: Dict[str, Any] | None = None,
    eligible_sentence_count: int = 3,
) -> Dict[str, Any]:
    """Compute the shared Turnitin-like score from scanner feature maps."""
    features = features or {}
    ai_components = ai_components or {}
    topk_raw = _percent(
        ai_components.get("topk_pattern_raw", ai_components.get("topk_pattern")),
        default=0.0,
    )
    topk_calibrated = _percent(ai_components.get("topk_calibrated_risk"), default=-1.0)
    if topk_calibrated < 0.0:
        topk_calibrated = float(
            calibrate_topk_risk(
                topk_raw,
                eligible_sentence_count=eligible_sentence_count,
            ).get("topk_calibrated_risk", topk_raw)
        )

    patchwork_expansion = max(
        _percent(features.get("outline_to_text_expansion")),
        _percent(features.get("section_style_variance")),
    )
    components = {
        "ai_likelihood": _percent(
            features.get("ai_likelihood"),
            default=_percent(ai_components.get("ai_likelihood_score")),
        ),
        "topk_calibrated_risk": topk_calibrated,
        "semantic_uniformity": _percent(features.get("semantic_uniformity_risk")),
        "rewrite_smoothness": _percent(features.get("rewrite_smoothness")),
        "patchwork_expansion": patchwork_expansion,
        "signal_agreement": _percent(features.get("signal_agreement_score")),
    }
    human_anchor_suppression = _percent(features.get("human_anchor_discount"), default=-1.0)
    if human_anchor_suppression <= 0.0:
        human_anchor_suppression = min(45.0, _percent(features.get("human_anchor_score")) * 0.45)

    weighted_components = {
        key: round(float(components[key]) * float(weight), 3)
        for key, weight in TURNITIN_LIKE_COMPONENT_WEIGHTS.items()
    }
    raw_positive_score = sum(weighted_components.values())
    score = max(0.0, min(100.0, raw_positive_score - human_anchor_suppression))
    return {
        "version": TURNITIN_LIKE_SCORE_VERSION,
        "score": round(score, 3),
        "components": {key: round(float(value), 3) for key, value in components.items()},
        "weighted_components": weighted_components,
        "human_anchor_suppression": round(float(human_anchor_suppression), 3),
        "raw_positive_score": round(float(raw_positive_score), 3),
        "weights": dict(TURNITIN_LIKE_COMPONENT_WEIGHTS),
        "safe_band": score <= TURNITIN_LIKE_SAFE_BAND,
        "safe_band_threshold": TURNITIN_LIKE_SAFE_BAND,
    }


def turnitin_like_ai_profile_from_report(report_dict: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(report_dict, dict):
        empty = {key: 0.0 for key in TURNITIN_LIKE_COMPONENT_WEIGHTS}
        return {
            "version": TURNITIN_LIKE_SCORE_VERSION,
            "score": 0.0,
            "components": empty,
            "weighted_components": {key: 0.0 for key in TURNITIN_LIKE_COMPONENT_WEIGHTS},
            "human_anchor_suppression": 0.0,
            "raw_positive_score": 0.0,
            "weights": dict(TURNITIN_LIKE_COMPONENT_WEIGHTS),
            "safe_band": True,
            "safe_band_threshold": TURNITIN_LIKE_SAFE_BAND,
        }
    badge = report_dict.get("ai_risk_badge") or {}
    transformation = badge.get("transformation_classification") or {}
    return turnitin_like_ai_profile(
        features=transformation.get("features") or {},
        ai_components=badge.get("ai_components") or {},
    )
