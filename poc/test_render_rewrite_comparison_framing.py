"""Regression test for the rewrite-comparison PDF's verdict-authority + composition-
interpretation framing (poc/report/render_rewrite.py::_cmp_column /
_executive_comparison_html), added for parity with the scan-page framing shipped in
b35c6327 (poc/report/render_panels.py::render_merged_authorship_risk). KEEP-IN-SYNC
with poc/test_render_authorship_breakdown.py::test_merged_render_shows_owner_mandated_framing_labels.
"""
from report.render_rewrite import _executive_comparison_html

_BREAKDOWN = {
    "document_breakdown_raw": {
        "student_owned": 0.37, "ai_assisted_polished": 0.21,
        "ai_paraphrased": 0.14, "ai_generated_like": 0.28,
    },
    "document_breakdown_bands": {
        "student_owned": "Some", "ai_assisted_polished": "Little",
        "ai_paraphrased": "Little", "ai_generated_like": "Some",
    },
}


def _badge(fused_score, tier="amber"):
    return {
        "tier": tier,
        "ai_likelihood_score": fused_score,
        "tier_authority": {
            "source": "v7_fused", "fused_score": fused_score,
            "composite_score": fused_score - 5, "proportion": 0.31, "flag_line": 32,
        },
        "authorship_breakdown": _BREAKDOWN,
    }


def _scan(fused_score, tier="amber"):
    return {"ai_risk_badge": _badge(fused_score, tier)}


def _summary(orig_score=45.0, new_score=18.0):
    return {
        "detect_scan_original_saved": _scan(orig_score, tier="amber"),
        "detect_scan_rewritten": _scan(new_score, tier="green"),
    }


def test_both_columns_carry_verdict_authority_and_composition_interpretation_framing():
    html = _executive_comparison_html(_summary(), review_required=False)
    assert html
    # Verdict-authority line appears once per column (2 total), not zero, not duplicated
    # beyond one-per-scan.
    assert html.count("dp-verdict-framing") == 2
    assert html.count("AI-risk assessment") == 2
    assert html.count("does not override it") == 2
    # Composition subhead carries the extended interpretation copy, once per column.
    assert html.count("interpretive breakdown of writing character") == 2
    assert html.count("not the AI-risk verdict above") == 2
    # Ordering: the verdict framing precedes the composition bars in document order.
    assert html.index("dp-verdict-framing") < html.index("dp-abd-bars")


def test_legacy_scans_without_breakdown_render_unchanged():
    # No authorship_breakdown on either scan -> no framing, no bars, no crash
    # (byte-identical to pre-change legacy behavior for old rewrite jobs).
    summary = {
        "detect_scan_original_saved": {"ai_risk_badge": {
            "tier": "amber", "tier_authority": {"fused_score": 45.0},
        }},
        "detect_scan_rewritten": {"ai_risk_badge": {
            "tier": "green", "tier_authority": {"fused_score": 18.0},
        }},
    }
    html = _executive_comparison_html(summary, review_required=False)
    assert "dp-verdict-framing" not in html
    assert "dp-abd-bars" not in html


def test_no_scans_returns_empty_string():
    assert _executive_comparison_html({}, review_required=False) == ""


def test_three_way_display_fallback_applies_per_column():
    # V8 three-way display fallback (poc/detect_v7/pipeline_bridge.py::
    # _compose_display_fallback) reuses render_panels._authorship_bars_html via
    # _cmp_column, so the merged ai_transformed share renders identically in each
    # of the two rewrite-comparison columns.
    three_way_breakdown = {
        **_BREAKDOWN,
        "display_taxonomy": "three_way",
        "display_shares": {"student_owned": 0.37, "ai_assisted_polished": 0.21, "ai_transformed": 0.42},
        "display_primary": "ai_transformed",
    }
    summary = {
        "detect_scan_original_saved": {"ai_risk_badge": {**_badge(45.0, "amber"), "authorship_breakdown": three_way_breakdown}},
        "detect_scan_rewritten": {"ai_risk_badge": {**_badge(18.0, "green"), "authorship_breakdown": three_way_breakdown}},
    }
    html = _executive_comparison_html(summary, review_required=False)
    assert html.count("AI-transformed") >= 2
    assert "AI-paraphrased" not in html
    assert "AI-generated-like" not in html
