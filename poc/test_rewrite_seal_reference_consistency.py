"""Rewrite seal verdict guard (render_rewrite.py).

History:
  1. Seal showed "GOOD" (calibrated risk) -> read by users as "Turnitin-safe" -> false promise.
  2. Switched the verdict to track the external/Turnitin band ("Still flagged by detectors" ...).
  3. INTERIM (current): a real Turnitin report proved our external band OVER-FLAGS reality
     (+62.5 pts on a doc Turnitin cleared at 0% -- see poc/calibration/). So while
     EXTERNAL_ESTIMATE_DISPLAY_ENABLED is False we make NO detector claim at all: the seal is a
     neutral grounding verdict, and no external % / band leaks onto any surface.

These tests pin the interim contract. When the flag flips True (post-calibration), they skip and
the band-driven behaviour is expected again.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from report.render import EXTERNAL_ESTIMATE_DISPLAY_ENABLED  # noqa: E402
from report.render_rewrite import render_rewrite_report  # noqa: E402

_DEMOTED = not EXTERNAL_ESTIMATE_DISPLAY_ENABLED
_skip_if_enabled = pytest.mark.skipif(
    not _DEMOTED, reason="external estimate re-enabled; band-driven verdict expected instead")


def _summary(external_band: str = "high", ext_score: int = 52) -> dict:
    return {
        "outcome": "partial_candidate_not_strict_safe",
        "status": "partial_candidate_not_strict_safe",
        "detect_scores": {"original_ai_authorship": 70, "rewritten_ai_authorship": 32},
        "detect_scan_original": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 42, "writing_quality_score": 65},
        },
        "detect_scan_rewritten": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {
                "ai_likelihood_score": 32,
                "writing_quality_score": 70,
                "external_detector_estimate": {"score": ext_score, "band": external_band},
            },
            "scan_intelligence": {"transformation": {"contribution": {
                "human_contribution_ratio": 97, "ai_transformation_ratio": 3, "calibrated_ai_risk": 16,
            }}},
        },
    }


@_skip_if_enabled
def test_demoted_seal_makes_no_detector_claim():
    # Even with a "high" external band, the demoted seal must NOT assert a detector outcome
    # (neither "still flagged" nor a reassuring "safe"/"GOOD"/"low"), and must not leak a % .
    report = render_rewrite_report(_summary("high", 52), [], [], original_text="O.", final_text="R.")
    up = report.upper()
    assert "GROUNDED DRAFT" in up, report
    assert "STILL FLAGGED BY DETECTORS" not in up
    assert "DETECTOR RISK LOW" not in up
    assert "~52%" not in report
    assert "below 20% reference" not in report
    assert "Turnitin / external" not in report


@_skip_if_enabled
def test_demoted_seal_credits_grounding_and_directs_finish():
    report = render_rewrite_report(_summary("high", 52), [], [], original_text="O.", final_text="R.")
    assert "grounding improved" in report.lower()
    assert "finish in your own words" in report.lower()


if __name__ == "__main__":
    test_demoted_seal_makes_no_detector_claim()
    test_demoted_seal_credits_grounding_and_directs_finish()
    print("ok")
