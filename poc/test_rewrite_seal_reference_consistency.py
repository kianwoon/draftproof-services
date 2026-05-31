"""Seal-subtitle consistency guard (render_rewrite.py).

The rewritten-rating seal subtitle is "{X}% calibrated risk · {reference suffix}".
Two invariants the user paid for:
  1. X is the CALIBRATED authorship risk (not the conservative headline AI-likelihood),
     so the page and the PDF show the same number.
  2. The reference suffix is computed from that SAME displayed X — so a sub-20%
     calibrated risk reads "below 20% reference", never "review threshold exceeded".

Imports ONLY report.render_rewrite to stay decoupled from the rewrite pipeline
(the broader content-agnostic test module fails to import on an unrelated symbol).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from report.render_rewrite import render_rewrite_report  # noqa: E402


def _summary(headline: int, calibrated: int) -> dict:
    return {
        "outcome": "partial_candidate_not_strict_safe",
        "status": "partial_candidate_not_strict_safe",
        "detect_scores": {
            "original_ai_authorship": 70,
            "rewritten_ai_authorship": headline,
            "rewritten_human_contribution": 90,
            "rewritten_ai_transformation": 10,
        },
        "detect_scan_original": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 42, "writing_quality_score": 65},
        },
        "detect_scan_rewritten": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": headline, "writing_quality_score": 70},
            "scan_intelligence": {"transformation": {"contribution": {
                "human_contribution_ratio": 90,
                "ai_transformation_ratio": 10,
                "calibrated_ai_risk": calibrated,
            }}},
        },
    }


def test_seal_shows_calibrated_value_with_below_reference_when_calibrated_under_20():
    # Headline 32 (>=20) but calibrated 16 (<20): the seal must lead with the calibrated 16,
    # and the suffix must agree with THAT number -> "below 20% reference".
    report = render_rewrite_report(_summary(32, 16), [], [], original_text="O.", final_text="R.")
    assert "16% calibrated risk · below 20% reference" in report, report
    assert "32% calibrated risk" not in report, "seal must not mislabel the headline as calibrated risk"
    assert "16% calibrated risk · review threshold exceeded" not in report, "sub-20% must never read 'threshold exceeded'"


def test_seal_reads_threshold_exceeded_when_calibrated_at_or_above_20():
    report = render_rewrite_report(_summary(48, 25), [], [], original_text="O.", final_text="R.")
    assert "25% calibrated risk · review threshold exceeded" in report, report
    assert "below 20% reference" not in report, report


if __name__ == "__main__":
    test_seal_shows_calibrated_value_with_below_reference_when_calibrated_under_20()
    test_seal_reads_threshold_exceeded_when_calibrated_at_or_above_20()
    print("ok")
