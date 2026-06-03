from __future__ import annotations

from poc.report.render_rewrite import render_rewrite_report
from poc.rewrite_pipeline_core.scoring.profiles import _contribution_scores


def test_rewrite_report_normalizes_impossible_contribution_pairs():
    summary = {
        "outcome": "partial_candidate_not_strict_safe",
        "status": "partial_candidate_not_strict_safe",
        "detect_scores": {
            "original_ai_authorship": 42,
            "rewritten_ai_authorship": 33,
            "original_human_contribution": 66,
            "original_ai_transformation": 34,
            "rewritten_human_contribution": 99,
            "rewritten_ai_transformation": 100,
        },
        "detect_scan_original": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 42, "writing_quality_score": 65},
        },
        "detect_scan_rewritten": {
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
            "ai_risk_badge": {"ai_likelihood_score": 33, "writing_quality_score": 70},
        },
    }

    report = render_rewrite_report(summary, [], [], original_text="Original.", final_text="Rewritten.")

    assert "| **Human Contribution** | `66%` | `99%` | `+33%` |" in report
    assert "| **AI Transformation** | `34%` | `1%` | `-33%` |" in report
    assert "`100%` | `+66%`" not in report


def test_footprint_profile_normalizes_impossible_contribution_pairs():
    scores = _contribution_scores(
        {
            "integrity_layers": {
                "layers": {
                    "human_contribution_signal": {"score": 99},
                    "ai_transformation_risk": {"score": 100},
                }
            }
        }
    )

    assert scores == {"human": 99.0, "ai_transformation": 1.0}
