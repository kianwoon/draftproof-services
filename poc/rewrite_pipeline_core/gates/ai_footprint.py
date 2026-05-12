"""AI-footprint acceptance gates for rewrite candidates."""

from __future__ import annotations

import os

from detect.topk_calibration import TOPK_CALIBRATED_SAFE_LIMIT
from rewrite_pipeline_core.scoring.profiles import (
    _ai_footprint_flatten,
    _ai_footprint_profile,
)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _ai_footprint_gate_status(
    original_report: dict | None,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float = 0,
    weighted_severity_delta: int | float = 0,
    critical_high_delta: int | float = 0,
    ai_score_regressed: bool = False,
) -> dict:
    """Classify whether a candidate moved real AI-footprint drivers.

    Grounding is present in the proxy because external detectors often react to
    broad unsupported prose, but grounding alone never grants mitigation status.
    Authorship/texture movement must exist for partial or full AI mitigation.
    """
    before = _ai_footprint_profile(original_report)
    after = _ai_footprint_profile(candidate_report)

    before_flat = _ai_footprint_flatten(before)
    after_flat = _ai_footprint_flatten(after)
    driver_keys = [
        "ai_authorship",
        "ai_transformation",
        "ai_likelihood",
        "topk_calibrated_risk",
        "topk_pattern",
        "topk_pattern_raw",
        "rewrite_smoothness",
        "semantic_uniformity",
        "qualifying_text_ai_density",
        "discourse_regularity",
        "generic_assertion_risk",
        "unsupported_claim_risk",
        "broad_claim_risk",
        "external_ai_flag_risk",
    ]
    drops = {
        key: round(float(before_flat.get(key, 0.0)) - float(after_flat.get(key, 0.0)), 3)
        for key in driver_keys
    }
    primary_keys = [
        "ai_authorship",
        "ai_transformation",
        "ai_likelihood",
        "topk_calibrated_risk",
        "rewrite_smoothness",
        "semantic_uniformity",
        "qualifying_text_ai_density",
        "discourse_regularity",
    ]
    thresholds = {
        "ai_authorship": _float_env("DRAFTPROOF_AI_FOOTPRINT_MIN_AUTHORSHIP_DROP", 1.0),
        "ai_transformation": _float_env("DRAFTPROOF_AI_FOOTPRINT_MIN_TRANSFORMATION_DROP", 1.0),
        "ai_likelihood": _float_env("DRAFTPROOF_AI_FOOTPRINT_MIN_LIKELIHOOD_DROP", 1.0),
        "topk_calibrated_risk": _float_env("DRAFTPROOF_AI_FOOTPRINT_MIN_TOPK_DROP", 2.0),
        "rewrite_smoothness": _float_env("DRAFTPROOF_AI_FOOTPRINT_MIN_SMOOTHNESS_DROP", 1.0),
        "semantic_uniformity": _float_env("DRAFTPROOF_AI_FOOTPRINT_MIN_SEMANTIC_DROP", 1.0),
        "qualifying_text_ai_density": _float_env("DRAFTPROOF_AI_FOOTPRINT_MIN_QUALIFYING_DENSITY_DROP", 1.0),
        "discourse_regularity": _float_env("DRAFTPROOF_AI_FOOTPRINT_MIN_DISCOURSE_DROP", 1.0),
        "external_ai_flag_risk": _float_env("DRAFTPROOF_EXTERNAL_FLAG_PROXY_MIN_DROP", 1.5),
    }
    active_topk_threshold = _float_env("DRAFTPROOF_AI_FOOTPRINT_ACTIVE_TOPK_THRESHOLD", 90.0)
    if before_flat.get("topk_pattern_raw", before_flat.get("topk_pattern", 0.0)) >= active_topk_threshold:
        thresholds["topk_calibrated_risk"] = max(
            thresholds["topk_calibrated_risk"],
            _float_env("DRAFTPROOF_AI_FOOTPRINT_SATURATED_MIN_TOPK_DROP", 8.0),
        )
    safe_topk_limit = TOPK_CALIBRATED_SAFE_LIMIT
    material_primary = [
        key for key in primary_keys
        if drops.get(key, 0.0) >= thresholds.get(key, 1.0)
    ]
    material_proxy = drops.get("external_ai_flag_risk", 0.0) >= thresholds["external_ai_flag_risk"]
    texture_blockers = []
    if (
        before_flat.get("topk_pattern_raw", before_flat.get("topk_pattern", 0.0)) >= active_topk_threshold
        and drops.get("topk_calibrated_risk", 0.0) < thresholds["topk_calibrated_risk"]
    ):
        texture_blockers.append({
            "driver": "topk_calibrated_risk",
            "reason": "active_topk_pattern_not_reduced",
            "before": round(float(before_flat.get("topk_calibrated_risk", 0.0)), 3),
            "after": round(float(after_flat.get("topk_calibrated_risk", 0.0)), 3),
            "raw_before": round(float(before_flat.get("topk_pattern_raw", before_flat.get("topk_pattern", 0.0))), 3),
            "raw_after": round(float(after_flat.get("topk_pattern_raw", after_flat.get("topk_pattern", 0.0))), 3),
            "drop": drops.get("topk_calibrated_risk", 0.0),
            "required_drop": thresholds["topk_calibrated_risk"],
        })
    if float(after_flat.get("topk_calibrated_risk", 0.0)) > safe_topk_limit:
        texture_blockers.append({
            "driver": "topk_calibrated_risk",
            "reason": "topk_calibrated_above_safe_level",
            "before": round(float(before_flat.get("topk_calibrated_risk", 0.0)), 3),
            "after": round(float(after_flat.get("topk_calibrated_risk", 0.0)), 3),
            "raw_before": round(float(before_flat.get("topk_pattern_raw", before_flat.get("topk_pattern", 0.0))), 3),
            "raw_after": round(float(after_flat.get("topk_pattern_raw", after_flat.get("topk_pattern", 0.0))), 3),
            "required_max": round(float(safe_topk_limit), 3),
        })
    smoothness_regression_limit = _float_env("DRAFTPROOF_AI_FOOTPRINT_MAX_SMOOTHNESS_REGRESSION", 1.0)
    if (
        before_flat.get("topk_pattern_raw", before_flat.get("topk_pattern", 0.0)) >= active_topk_threshold
        and drops.get("rewrite_smoothness", 0.0) < -smoothness_regression_limit
    ):
        texture_blockers.append({
            "driver": "rewrite_smoothness",
            "reason": "smoothness_regressed_while_topk_pinned",
            "before": round(float(before_flat.get("rewrite_smoothness", 0.0)), 3),
            "after": round(float(after_flat.get("rewrite_smoothness", 0.0)), 3),
            "drop": drops.get("rewrite_smoothness", 0.0),
            "max_regression": smoothness_regression_limit,
        })
    safety_clean = bool(
        not ai_score_regressed
        and float(review_burden_delta or 0.0) <= 0.0
        and float(weighted_severity_delta or 0.0) <= 0.0
        and float(critical_high_delta or 0.0) <= 0.0
        and drops.get("ai_authorship", 0.0) >= 0.0
        and drops.get("ai_transformation", 0.0) >= 0.0
    )
    safe_band_thresholds = {
        "external_ai_flag_risk": _float_env("DRAFTPROOF_EXTERNAL_FLAG_PROXY_SAFE_BAND", 35.0),
        "ai_authorship": _float_env("DRAFTPROOF_AI_FOOTPRINT_SAFE_AUTHORSHIP", 35.0),
        "ai_transformation": _float_env("DRAFTPROOF_AI_FOOTPRINT_SAFE_TRANSFORMATION", 35.0),
        "qualifying_text_ai_density": _float_env("DRAFTPROOF_QUALIFYING_AI_DENSITY_SAFE_BAND", 35.0),
        "topk_calibrated_risk": safe_topk_limit,
        "rewrite_smoothness": _float_env("DRAFTPROOF_AI_FOOTPRINT_SAFE_SMOOTHNESS", 55.0),
    }
    safe_band = bool(
        safety_clean
        and all(float(after_flat.get(key, 0.0)) <= limit for key, limit in safe_band_thresholds.items())
    )
    material_driver_moved = bool(material_primary and material_proxy and safety_clean and not texture_blockers)
    if safe_band and material_primary:
        outcome_class = "ai_mitigated"
    elif material_driver_moved:
        outcome_class = "partially_ai_mitigated"
    elif material_primary and material_proxy and safety_clean and texture_blockers:
        outcome_class = "ai_footprint_blocked_by_texture"
    elif safety_clean and (
        float(review_burden_delta or 0.0) < 0.0
        or float(weighted_severity_delta or 0.0) < 0.0
        or drops.get("generic_assertion_risk", 0.0) > 0.0
        or drops.get("unsupported_claim_risk", 0.0) > 0.0
        or drops.get("broad_claim_risk", 0.0) > 0.0
    ):
        outcome_class = "cleanup_improved"
    else:
        outcome_class = "no_ai_footprint_improvement"
    remaining = [
        {
            "driver": key,
            "value": round(float(after_flat.get(key, 0.0)), 3),
            "safe_band": round(float(limit), 3),
        }
        for key, limit in safe_band_thresholds.items()
        if float(after_flat.get(key, 0.0)) > float(limit)
    ]
    return {
        "before": before,
        "after": after,
        "drops": drops,
        "material_primary_drivers": material_primary,
        "material_proxy_drop": material_proxy,
        "texture_blockers": texture_blockers,
        "material_driver_moved": material_driver_moved,
        "safety_clean": safety_clean,
        "safe_band": safe_band,
        "outcome_class": outcome_class,
        "remaining_ai_footprint_drivers": remaining,
        "thresholds": thresholds,
        "safe_band_thresholds": safe_band_thresholds,
    }
