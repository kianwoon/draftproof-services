"""PDF-panel tests for the Phase-0 Authorship Evidence Level renderer."""
from __future__ import annotations

from report.render_panels import render_authorship_evidence_levels


_AEL = {
    "level": 2,
    "max_level_assessable": 2,
    "lenses": {
        "ai_pattern": {"band": "amber", "assessment_confidence": "high"},
        "grounding": {"band": "moderate", "assessment_confidence": "high"},
        "citation": {"level": "high", "assessment_confidence": "moderate"},
        "reasoning": {"band": "acceptable_control", "assessment_confidence": "moderate"},
    },
    "assessment_confidence": "moderate",
    "confidence_reasons": ["no_assignment_context"],
    "coverage": [
        {"type": "generator_window", "value": "through_2026_07"},
        {"type": "context_availability", "value": "assignment_brief_absent"},
    ],
    "limitations": ["no_assignment_context"],
    "lifecycle": {"status": "advisory", "scoring_enabled": False},
}


def test_renders_level_and_lens_rows():
    html = render_authorship_evidence_levels({"ai_risk_badge": {"authorship_evidence_levels": _AEL}})
    assert "Level <strong>2 of 5</strong>" in html
    assert "AI-pattern signal" in html
    assert "Grounding" in html
    assert "Source traceability" in html
    assert "Reasoning development" in html


def test_coverage_and_limitations_lines_removed():
    # Owner 2026-07-15: the §C assessment-confidence / coverage / limitations
    # lines were dropped from the rendered panel as noise (data stays in the
    # badge JSON). This asserts they stay OUT — web + PDF in sync.
    html = render_authorship_evidence_levels({"ai_risk_badge": {"authorship_evidence_levels": _AEL}})
    assert "through_2026_07" not in html
    assert "assignment_brief_absent" not in html
    assert "Limitations" not in html
    # The advisory beta chip stays.
    assert "advisory" in html.lower()


def test_renders_lens_anchor_detail():
    ael = dict(_AEL, lenses={
        "ai_pattern": {"band": "amber", "assessment_confidence": "high",
                       "anchor": {"headline": "Reads as AI-generated — 11 of 26 sentences",
                                  "example": 'And once that happens, "years of experience" becomes weak.',
                                  "fix": None}},
        "grounding": {"band": "moderate", "assessment_confidence": "high",
                      "anchor": {"headline": "Weak grounding — claims without concrete anchors",
                                 "example": "Make sure only a handful of people know the edge cases.",
                                 "fix": "Name a specific example, source, or moment."}},
        "citation": {"level": "high", "assessment_confidence": "moderate"},
        "reasoning": {"band": "acceptable_control", "assessment_confidence": "moderate"},
    })
    html = render_authorship_evidence_levels({"ai_risk_badge": {"authorship_evidence_levels": ael}})
    assert "Reads as AI-generated — 11 of 26 sentences" in html
    assert "years of experience" in html  # verbatim flagged sentence surfaced
    assert "Name a specific example" in html  # fix direction surfaced
    assert "dp-ael-anchor" in html
    # A lens with no anchor still renders (no crash, band chip only).
    assert "Source traceability" in html


def test_empty_when_field_absent():
    assert render_authorship_evidence_levels({"ai_risk_badge": {}}) == ""
    assert render_authorship_evidence_levels({}) == ""
    assert render_authorship_evidence_levels({"ai_risk_badge": {"authorship_evidence_levels": {}}}) == ""


def test_fail_open_on_partial_lenses():
    ael = dict(_AEL, lenses={"ai_pattern": {"band": "green", "assessment_confidence": "high"},
                             "grounding": {"available": False}})
    html = render_authorship_evidence_levels({"ai_risk_badge": {"authorship_evidence_levels": ael}})
    assert html != ""
    assert "Not assessed" in html
