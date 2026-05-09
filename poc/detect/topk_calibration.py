"""Calibrate raw GPT-2 Top-k predictability into a prose risk band.

Raw Top-k from the current scanner is a diagnostic token-route signal. It is
not calibrated to the product safe band directly: normal prose can sit well
above 25 on the raw GPT-2 scale. This module provides a fixed v1 monotonic
mapping used by scanner output and rewrite gates.
"""

from __future__ import annotations

from typing import Any, Dict


TOPK_CALIBRATION_VERSION = "gpt2-v1"
TOPK_CALIBRATED_SAFE_LIMIT = 25.0
TOPK_MIN_ELIGIBLE_SENTENCES = 3
TOP10_NORMALISATION_THRESHOLD = 0.65


def _percent(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if numeric <= 1.0:
        numeric *= 100.0
    return max(0.0, min(100.0, numeric))


def _raw_from_avg_top10(avg_top10_ratio: Any) -> float | None:
    if not isinstance(avg_top10_ratio, (int, float)):
        return None
    ratio = max(0.0, min(1.0, float(avg_top10_ratio)))
    return max(0.0, min(100.0, (ratio / TOP10_NORMALISATION_THRESHOLD) * 100.0))


def calibrate_topk_risk(
    raw_topk_pattern: Any = None,
    *,
    avg_top10_ratio: Any = None,
    eligible_sentence_count: Any = None,
) -> Dict[str, Any]:
    """Return fixed v1 calibrated Top-k risk fields.

    Bands:
      raw <= 45   -> low risk band
      raw 45-60   -> watch band
      raw 60-75   -> elevated band
      raw >= 75   -> high/saturated band

    The curve intentionally maps raw 67.37 to roughly 24.8, matching the
    current GPT-2 scanner behavior where raw Top-k above 25 is common prose.
    """
    raw = _percent(raw_topk_pattern)
    if raw is None:
        raw = _raw_from_avg_top10(avg_top10_ratio)

    try:
        eligible = int(eligible_sentence_count)
    except (TypeError, ValueError):
        eligible = 0

    calibration_eligible = eligible >= TOPK_MIN_ELIGIBLE_SENTENCES
    if raw is None:
        raw = 0.0

    if not calibration_eligible:
        calibrated = 100.0
        band = "insufficient_prose"
    elif raw <= 45.0:
        calibrated = raw * 0.25
        band = "low"
    elif raw <= 60.0:
        calibrated = 11.25 + ((raw - 45.0) * 0.45)
        band = "watch"
    elif raw <= 75.0:
        calibrated = 18.0 + ((raw - 60.0) * 0.92)
        band = "elevated"
    else:
        calibrated = 31.8 + ((raw - 75.0) * 2.728)
        band = "high_saturated"

    calibrated = max(0.0, min(100.0, calibrated))
    return {
        "topk_pattern_raw": round(raw, 3),
        "topk_calibrated_risk": round(calibrated, 3),
        "topk_safe_band": bool(calibration_eligible and calibrated < TOPK_CALIBRATED_SAFE_LIMIT),
        "topk_calibration_version": TOPK_CALIBRATION_VERSION,
        "topk_calibration_band": band,
        "topk_calibration_eligible": calibration_eligible,
        "topk_eligible_sentence_count": eligible,
    }
