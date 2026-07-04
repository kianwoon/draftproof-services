"""Regression test for the PDF Authorship Clarity Breakdown panel
(poc/report/render_panels.py::render_authorship_breakdown), added when the PDF
was aligned to the V7 scan page. Guards: the panel renders the fused headline +
deep-scan estimate + disclaimer, and fails open (empty string) when the badge
carries no authorship_breakdown."""
from report.render_panels import render_authorship_breakdown

_BADGE = {
    "tier": "green",
    "ai_likelihood_score": 24.35,
    "tier_authority": {
        "source": "v7_fused", "fused_score": 24.35,
        "composite_score": 14.0, "proportion": 0.31, "flag_line": 32,
    },
    "authorship_breakdown": {
        "document_breakdown_raw": {
            "student_owned": 0.37, "ai_assisted_polished": 0.21,
            "ai_paraphrased": 0.14, "ai_generated_like": 0.28,
        },
        "document_breakdown_bands": {
            "student_owned": "Some", "ai_assisted_polished": "Little",
            "ai_paraphrased": "Little", "ai_generated_like": "Some",
        },
        "primary_category": "student_owned", "confidence": "low",
        "deep_scan": {"proportion": 0.31, "band": "amber", "calibrated": False},
        "disclaimer": "DraftProof provides authorship clarity signals. It does not determine misconduct.",
    },
}


def test_renders_fused_headline_deepscan_and_disclaimer():
    html = render_authorship_breakdown({"ai_risk_badge": _BADGE})
    assert html  # non-empty
    assert "24" in html          # fused AI-likelihood headline
    assert "31" in html          # deep-scan estimate
    assert "misconduct" in html  # disclaimer verbatim


def test_fail_open_when_breakdown_absent():
    # Older reports / flag-off carry no authorship_breakdown -> empty panel.
    assert render_authorship_breakdown({"ai_risk_badge": {}}) == ""
    assert render_authorship_breakdown({}) == ""


def test_deep_scan_only_when_no_tier_authority():
    # Deep scan present but tier authority didn't fire: panel still renders
    # (deep-scan-only fallback), never raises.
    badge = {k: v for k, v in _BADGE.items() if k != "tier_authority"}
    html = render_authorship_breakdown({"ai_risk_badge": badge})
    assert html
    assert "31" in html
