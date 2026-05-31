"""Dual-headline AI-likelihood rendering (page + PDF share this band table)."""
from report.render import _ai_likelihood_bands


def test_draftproof_tier_maps_to_color():
    out = _ai_likelihood_bands({"ai_likelihood_score": 42.0, "tier": "AMBER"})
    assert out["draftproof"] == {"score": 42, "tier": "AMBER", "color": "#d97706"}


def test_all_draftproof_tiers():
    colors = {"GREEN": "#16a34a", "AMBER": "#d97706", "ORANGE": "#ea580c", "RED": "#dc2626"}
    for tier, color in colors.items():
        out = _ai_likelihood_bands({"ai_likelihood_score": 50, "tier": tier})
        assert out["draftproof"]["color"] == color


def test_external_bands_map_to_label_and_color():
    cases = {
        "low": ("unlikely to be flagged", "#16a34a"),
        "elevated": ("possibly flagged", "#d97706"),
        "high": ("likely to be flagged", "#dc2626"),
    }
    for band, (label, color) in cases.items():
        out = _ai_likelihood_bands({
            "ai_likelihood_score": 42, "tier": "AMBER",
            "external_detector_estimate": {"score": 59.8, "band": band},
        })
        assert out["external"]["label"] == label
        assert out["external"]["color"] == color
        assert out["external"]["score"] == 60  # rounded


def test_missing_external_returns_none():
    out = _ai_likelihood_bands({"ai_likelihood_score": 42, "tier": "AMBER"})
    assert out["external"] is None
    assert out["draftproof"]["score"] == 42


def test_missing_score_returns_none_draftproof():
    assert _ai_likelihood_bands({})["draftproof"] is None
    assert _ai_likelihood_bands(None)["draftproof"] is None


from report.render import render_markdown
from report.report import DraftReport, PredictabilitySummary, Tier


def _report_with_badge(ext=True):
    badge = {
        "ai_likelihood_score": 42.0, "tier": "AMBER",
        "authorship_rating_label": "Possible AI-Assisted",
        "transformation_classification": {"label": "AI-stitched / patchwork", "confidence": "high"},
    }
    if ext:
        badge["external_detector_estimate"] = {"score": 59.8, "band": "high", "note": "x"}
    return DraftReport(
        overall_tier=Tier.MEDIUM, finding_count=0, findings_by_tier={},
        original_text="Some essay text about education and technology today.",
        predictability=PredictabilitySummary(
            overall_risk=0.42, risk_distribution={}, sentences=[], style_shifts=[], generic_phrases_found=[]),
        ai_risk_badge=badge,
    )


def test_markdown_shows_both_numbers_and_ordering():
    md = render_markdown(_report_with_badge(ext=True))
    assert "AI Likelihood" in md
    assert "42%" in md and "~60%" in md
    assert "likely to be flagged" in md
    assert md.index("AI Likelihood") < md.index("How DraftProof calibrates this")


def test_markdown_external_unavailable_fallback():
    md = render_markdown(_report_with_badge(ext=False))
    assert "External estimate unavailable" in md
    assert "42%" in md


def test_display_ai_score_is_not_halved():
    # GUARD: the legacy 0.5 "display multiplier" was removed. The displayed AI score must
    # equal the badge ai_likelihood_score (the single canonical number), NOT half of it.
    from report.render import _display_ai_score
    assert _display_ai_score(42) == 42
    assert _display_ai_score(0.42) == 42  # 0-1 scale normalizes to percent
    assert _display_ai_score(None) is None


def test_markdown_score_is_badge_value_no_legacy_halving():
    # GUARD: with badge ai_likelihood_score=42, the PDF/markdown must show 42% (header +
    # AI Likelihood block + Original Scan) and NEVER the legacy halved 21% anywhere.
    md = render_markdown(_report_with_badge(ext=True))
    assert "42%" in md
    assert "21%" not in md
