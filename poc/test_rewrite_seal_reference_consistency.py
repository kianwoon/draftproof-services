"""Rewrite seal verdict guard (render_rewrite.py).

Charter decision (user, 2026-05-31): users read a "GOOD" seal as "Turnitin-safe".
But the rewrite cannot reach Turnitin-safe (the external/fluency estimate has a floor
no rewrite removes), so a reassuring verdict next to a "~52% likely to be flagged" card
is a FALSE promise. Therefore the seal verdict must TRACK THE DETECTOR REALITY:

  * external band "high"     -> "Still flagged by detectors"   (never reads as safe)
  * external band "elevated" -> "Detector risk remains"
  * external band "low"      -> "Detector risk low"            (the only ~safe verdict)

and the subtitle credits the grounding work + directs the user to finish in their own
words -- it must NOT show a calibrated-risk % with a "below 20% reference" safety phrase.

Imports ONLY report.render_rewrite to stay decoupled from the rewrite pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from report.render_rewrite import render_rewrite_report  # noqa: E402


def _summary(external_band: str, ext_score: int, headline: int = 32, calibrated: int = 16) -> dict:
    return {
        "outcome": "partial_candidate_not_strict_safe",
        "status": "partial_candidate_not_strict_safe",
        "detect_scores": {
            "original_ai_authorship": 70,
            "rewritten_ai_authorship": headline,
            "original_ai_transformation": 34,
            "rewritten_ai_transformation": 3,
            "rewritten_human_contribution": 97,
        },
        "detect_scan_original": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 42, "writing_quality_score": 65},
        },
        "detect_scan_rewritten": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {
                "ai_likelihood_score": headline,
                "writing_quality_score": 70,
                "external_detector_estimate": {"score": ext_score, "band": external_band},
            },
            "scan_intelligence": {"transformation": {"contribution": {
                "human_contribution_ratio": 97,
                "ai_transformation_ratio": 3,
                "calibrated_ai_risk": calibrated,
            }}},
        },
    }


def test_high_external_band_never_claims_safe():
    # Detectors still flag it (~52% high) while the internal calibrated risk is a rosy 16%.
    # The seal must reflect the DETECTOR reality, not the rosy number.
    report = render_rewrite_report(_summary("high", 52), [], [], original_text="O.", final_text="R.")
    assert "Still flagged by detectors".upper() in report.upper(), report
    # No false-safety / no cherry-picked calibrated-risk-with-reference on the seal verdict line.
    assert "below 20% reference" not in report, "seal must not imply a Turnitin pass"
    assert "16% calibrated risk · below 20% reference" not in report
    # Credits the work + directs the user to finish it themselves.
    assert "finish in your own words" in report.lower(), report


def test_low_external_band_is_the_only_safe_verdict():
    report = render_rewrite_report(_summary("low", 14), [], [], original_text="O.", final_text="R.")
    assert "Detector risk low".upper() in report.upper(), report
    assert "Still flagged by detectors".upper() not in report.upper(), report


def test_elevated_external_band_says_risk_remains():
    report = render_rewrite_report(_summary("elevated", 40), [], [], original_text="O.", final_text="R.")
    assert "Detector risk remains".upper() in report.upper(), report


if __name__ == "__main__":
    test_high_external_band_never_claims_safe()
    test_low_external_band_is_the_only_safe_verdict()
    test_elevated_external_band_says_risk_remains()
    print("ok")
