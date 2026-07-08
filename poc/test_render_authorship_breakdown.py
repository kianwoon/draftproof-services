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


def test_merged_render_shows_ai_likelihood_once():
    from report.render_panels import render_merged_authorship_risk
    # Minimal report with both an authorship breakdown (fused) and a submission-risk section.
    report = _make_report_with_breakdown_and_sr()  # helper below
    data = {"document_context": {"word_count": 400}, "overall_tier_reason": ""}
    html = render_merged_authorship_risk(report, data)
    # The composition bars are present (authorship co-located).
    assert "dp-abd-bars" in html
    # PAGE PARITY (owner rule 2026-07-06): the merged scan lead mirrors the
    # page's card — axis levels + scale, WITHOUT the rewrite-only policy
    # cards / plain-English verdict / priority fixes.
    assert "1. Where the risk sits" in html
    assert "Submission and policy view" not in html
    assert "AI allowed with declaration" not in html
    assert "Plain-English verdict" not in html
    assert "Priority fixes" not in html
    assert "What do these numbers mean?" in html
    # The AI-likelihood % appears exactly once (de-duped): count the fused headline phrase.
    assert html.count("DraftProof AI-likelihood") == 1


def test_merged_render_shows_owner_mandated_framing_labels():
    # Owner-mandated framing (2026-07-08): the verdict headline is labeled the
    # AI-risk AUTHORITY, and the composition bars are labeled a display
    # INTERPRETATION. KEEP-IN-SYNC with MergedAuthorshipRisk.jsx's
    # .merged-verdict-framing paragraph + report.merged.compositionLensNote.
    from report.render_panels import render_merged_authorship_risk
    report = _make_report_with_breakdown_and_sr()
    data = {"document_context": {"word_count": 400}, "overall_tier_reason": ""}
    html = render_merged_authorship_risk(report, data)
    assert "dp-verdict-framing" in html
    assert "AI-risk assessment" in html
    assert "does not override it" in html
    assert "How the writing reads" in html
    assert "interpretive breakdown of writing character" in html
    # The framing line must appear before the composition bars in document order.
    assert html.index("dp-verdict-framing") < html.index("dp-abd-bars")


def test_mixed_signals_caveat_renders_when_presentation_set():
    # Tier-consistency guard (poc/detect_v7/pipeline_bridge.py::_apply_tier_consistency_guard)
    # sets breakdown["presentation"] = "mixed_signals" — PDF must show the same caveat as
    # MergedAuthorshipRisk.jsx's composition section.
    badge = {**_BADGE, "authorship_breakdown": {**_BADGE["authorship_breakdown"], "presentation": "mixed_signals"}}
    html = render_authorship_breakdown({"ai_risk_badge": badge})
    assert "dp-abd-caveat" in html
    assert "Mixed signals" in html
    assert "conflict with this document" in html


def test_mixed_signals_caveat_absent_when_presentation_missing():
    # Legacy/non-triggered reports carry no "presentation" field -> no caveat, byte-identical
    # to pre-change behavior.
    html = render_authorship_breakdown({"ai_risk_badge": _BADGE})
    assert "dp-abd-caveat" not in html
    assert "Mixed signals" not in html


def test_three_way_bars_render_when_display_fields_present():
    # V8 three-way display fallback (poc/detect_v7/pipeline_bridge.py::
    # _compose_display_fallback): display_taxonomy == "three_way" + display_shares ->
    # 3 merged bars (student_owned / ai_assisted_polished / ai_transformed), the
    # explainer caveat, and NO four-way ai_paraphrased/ai_generated_like rows.
    breakdown = {
        **_BADGE["authorship_breakdown"],
        "display_taxonomy": "three_way",
        "display_shares": {"student_owned": 0.37, "ai_assisted_polished": 0.21, "ai_transformed": 0.42},
        "display_primary": "ai_transformed",
    }
    html = render_authorship_breakdown({"ai_risk_badge": {**_BADGE, "authorship_breakdown": breakdown}})
    assert "AI-transformed" in html
    assert "42" in html  # merged share percentage
    assert "reports them as one AI-transformed share" in html
    assert "AI-paraphrased" not in html
    assert "AI-generated-like" not in html


def test_four_way_bars_unchanged_when_display_fields_absent():
    # Legacy payloads (mode "four_way" / older reports) carry no display_* fields ->
    # byte-identical four-way rendering, no three-way explainer.
    html = render_authorship_breakdown({"ai_risk_badge": _BADGE})
    assert "AI-paraphrased" in html
    assert "AI-generated-like" in html
    assert "AI-transformed" not in html
    assert "reports them as one AI-transformed share" not in html


def _make_report_with_breakdown_and_sr():
    from types import SimpleNamespace
    badge = {
        "tier": "green",
        "authorship_breakdown": {
            "document_breakdown_raw": {"student_owned": 0.34, "ai_assisted_polished": 0.23, "ai_paraphrased": 0.14, "ai_generated_like": 0.29},
            "document_breakdown_bands": {"student_owned": "Some", "ai_assisted_polished": "Little", "ai_paraphrased": "Little", "ai_generated_like": "Some"},
            "disclaimer": "DraftProof provides authorship clarity signals.",
        },
        "tier_authority": {"fused_score": 24, "composite_score": 14, "proportion": 0.31},
        "submission_risk": {"overall": {"level": "low"}, "axes": {"text_pattern": {"level": "low", "display_score": 24}}},
        "ai_likelihood_score": 24,
    }
    return SimpleNamespace(ai_risk_badge=badge)
