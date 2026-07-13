"""Tests for report.headline_confidence — the deterministic badge annotation that
reconciles a clean-looking headline (green / 0%) with contrary evidence elsewhere
on the same report (second-opinion tile, raw perplexity tier, low-confidence
verdict, below-floor deep scan). Annotate-not-suppress: the field only ADDS a
qualifier; it never changes tier or score.

Motivating live case (scan d449aca9, 2026-07-13): headline 0% green while the
desklib second-opinion tile showed 5/15 sentences >=0.99 (33%, amber) and the
raw perplexity tier was HIGH.
"""

from report.headline_confidence import compute_headline_confidence


def _green_badge(**overrides):
    badge = {
        "tier": "green",
        "ai_likelihood_score": 0.0,
        "verdict_low_confidence": False,
    }
    badge.update(overrides)
    return badge


def test_second_opinion_amber_tile_on_green_badge_fires_divergence():
    badge = _green_badge(
        ai_signal_deberta={"band": "amber", "signal_pct": 33, "available": True}
    )
    out = compute_headline_confidence(badge, raw_tier="low")
    assert out is not None
    assert out["level"] == "low"
    assert "second_opinion_divergence" in out["reasons"]
    assert out["detail"]["second_opinion_pct"] == 33


def test_raw_high_tier_on_green_badge_fires_divergence():
    out = compute_headline_confidence(_green_badge(), raw_tier="high")
    assert out is not None
    assert "raw_tier_divergence" in out["reasons"]
    assert out["detail"]["raw_tier"] == "high"


def test_verdict_low_confidence_reason():
    out = compute_headline_confidence(
        _green_badge(verdict_low_confidence=True), raw_tier="low"
    )
    assert out is not None
    assert "verdict_low_confidence" in out["reasons"]


def test_below_floor_deep_scan_reason_requires_fused_source():
    badge = _green_badge(
        signal_source="v7_fused",
        authorship_breakdown={
            "uncertainty_flags": ["deep_scan_below_reliability_floor"]
        },
    )
    out = compute_headline_confidence(badge, raw_tier="low")
    assert out is not None
    assert "deep_scan_below_floor" in out["reasons"]
    # same flags WITHOUT the fused source do not fire this reason
    badge2 = _green_badge(
        authorship_breakdown={
            "uncertainty_flags": ["deep_scan_below_reliability_floor"]
        },
    )
    assert compute_headline_confidence(badge2, raw_tier="low") is None


def test_clean_green_badge_with_no_contrary_evidence_returns_none():
    badge = _green_badge(
        ai_signal_deberta={"band": "insufficient", "signal_pct": 7, "available": True}
    )
    assert compute_headline_confidence(badge, raw_tier="low") is None


def test_non_low_risk_tier_returns_none_even_with_divergent_tile():
    badge = _green_badge(
        tier="amber",
        ai_signal_deberta={"band": "red", "signal_pct": 80, "available": True},
    )
    assert compute_headline_confidence(badge, raw_tier="high") is None


def test_live_case_d449aca9_all_three_reasons_fire():
    badge = _green_badge(
        signal_source="v7_fused",
        verdict_low_confidence=True,
        ai_signal_deberta={"band": "amber", "signal_pct": 33, "available": True},
        authorship_breakdown={
            "uncertainty_flags": [
                "paraphrase_without_original_draft",
                "deep_scan_below_reliability_floor",
                "esl_guard_unavailable",
            ]
        },
    )
    out = compute_headline_confidence(badge, raw_tier="high")
    assert out is not None
    assert out["level"] == "low"
    assert set(out["reasons"]) == {
        "second_opinion_divergence",
        "raw_tier_divergence",
        "verdict_low_confidence",
        "deep_scan_below_floor",
    }


def test_pdf_verdict_renders_headline_confidence_qualifier():
    from report.render import _ai_signal_verdict

    badge = _green_badge(
        headline_confidence={
            "level": "low",
            "reasons": ["second_opinion_divergence", "raw_tier_divergence"],
            "detail": {"second_opinion_pct": 33, "raw_tier": "high"},
        }
    )
    verdict = _ai_signal_verdict(badge)
    assert "low-confidence" in verdict.lower()
    assert "33%" in verdict
    # the richer qualifier replaces the generic low-confidence sentence
    assert "band boundary" not in verdict


class _FakeReport:
    def __init__(self, badge):
        self.ai_risk_badge = badge


def test_pdf_merged_panel_renders_headline_confidence_qualifier(monkeypatch):
    # The ACTIVE PDF lead is render_panels.render_merged_authorship_risk, NOT
    # render.py's legacy _render_ai_likelihood_headline fallback — the qualifier
    # must appear at the panel level or most PDFs never show it (on both the
    # fused-hero path and the no-breakdown/lead-only path).
    from report import render_panels

    monkeypatch.setattr(render_panels, "render_scan_lead", lambda *a, **k: "<div>lead</div>")
    badge = _green_badge(
        tier_authority={"fused_score": 0.0, "composite_score": 0.0, "proportion": 0.0},
        headline_confidence={
            "level": "low",
            "reasons": ["second_opinion_divergence"],
            "detail": {"second_opinion_pct": 33},
        },
    )
    # no-breakdown (lead-only) path
    html = render_panels.render_merged_authorship_risk(_FakeReport(badge), {})
    assert "low-confidence" in html and "33%" in html
    # breakdown (fused hero) path
    badge_bd = dict(badge)
    badge_bd["authorship_breakdown"] = {"document_breakdown_raw": {}, "document_breakdown_bands": {}}
    html = render_panels.render_merged_authorship_risk(_FakeReport(badge_bd), {})
    assert "low-confidence" in html and "33%" in html


def test_pdf_merged_panel_without_headline_confidence_unchanged(monkeypatch):
    from report import render_panels

    monkeypatch.setattr(render_panels, "render_scan_lead", lambda *a, **k: "<div>lead</div>")
    badge = _green_badge(
        tier_authority={"fused_score": 0.0, "composite_score": 0.0, "proportion": 0.0},
    )
    html = render_panels.render_merged_authorship_risk(_FakeReport(badge), {})
    assert html and "low-confidence" not in html


def test_pdf_verdict_without_headline_confidence_keeps_legacy_sentence():
    from report.render import _ai_signal_verdict

    badge = _green_badge(verdict_low_confidence=True)
    verdict = _ai_signal_verdict(badge)
    assert "band boundary" in verdict


def test_detect_pipeline_markdown_carries_annotation_when_json_does(tmp_path):
    """detect_pipeline.run_detect must build the report JSON BEFORE rendering the
    markdown/PDF: headline_confidence (and the deberta tile sync) are attached
    during report_to_dict, so rendering first ships a PDF without the qualifier
    the JSON carries. Uses a short human-ish text (green + verdict_low_confidence
    fires on thin samples) so the annotation is present in the JSON."""
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path

    _repo = _Path(__file__).resolve().parent.parent
    if str(_repo) not in _sys.path:
        _sys.path.insert(0, str(_repo))
    from poc.detect_pipeline import run_detect

    text = (
        "The sun rose over the quiet town. People walked to the market as the "
        "shops opened slowly."
    )
    result = run_detect(text, str(tmp_path), verbose=True)
    data = _json.loads(_Path(result["json_path"]).read_text())
    hc = (data.get("ai_risk_badge") or {}).get("headline_confidence")
    if not hc:
        import pytest as _pytest
        _pytest.skip("annotation did not fire on this text — precondition unmet")
    md = _Path(result["md_path"]).read_text()
    assert "Treat the headline percentage as low-confidence" in md


def test_malformed_badge_fails_open():
    assert compute_headline_confidence({}, raw_tier=None) is None
    assert compute_headline_confidence(None, raw_tier="high") is None
    # tile present but unavailable/malformed must not fire
    badge = _green_badge(ai_signal_deberta={"available": False, "band": "red"})
    assert compute_headline_confidence(badge, raw_tier="low") is None
