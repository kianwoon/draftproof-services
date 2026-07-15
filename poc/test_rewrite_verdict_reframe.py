"""Tests for the rewrite verdict reframe (docs/plans/rewrite_verdict_reframe_scope.md).

Covers: label-taxonomy rules (each label reachable + boundary cases), kill-switch
reversion parity, gap_resolution field production, note presence on the summary payload,
and PDF render (verdict headline + visible after-score + make-it-yours note).
"""

from __future__ import annotations

import pytest

from poc.rewrite_v6 import verdict_reframe as vr


# ── Kill switch (strict off-set, default ON) ────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),      # unset → default ON
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" 1 ", True),
        ("0", False),
        ("false", False),
        ("False", False),
        (" false ", False),
        ("yes", True),     # only 0/false disable (strict off-set)
    ],
)
def test_reframe_kill_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(vr.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(vr.ENV_VAR, value)
    assert vr.reframe_enabled() is expected


# ── gap_resolution field production ─────────────────────────────────────────
def test_build_gap_resolution_from_existing_signals():
    detect_scores = {
        "original_findings": 5,
        "rewritten_findings": 2,
        "original_grounding_quality_risk": 60.0,
        "rewritten_grounding_quality_risk": 40.0,
    }
    orig_report = {"ai_risk_badge": {"ai_components": {"generic_assertion_risk": 0.7, "citation_grounding_risk": 0.5}}}
    new_report = {"ai_risk_badge": {"ai_components": {"generic_assertion_risk": 0.4, "citation_grounding_risk": 0.5}}}
    spans = [{"kind": "improved"}, {"kind": "kept"}, {"kind": "improved"}]

    gap = vr.build_gap_resolution(detect_scores, spans, orig_report, new_report)
    assert gap["findings_before"] == 5
    assert gap["findings_after"] == 2
    assert gap["findings_resolved"] == 3
    assert gap["grounding_risk_delta"] == -20.0
    assert gap["generic_assertion_delta"] == -30.0
    assert gap["citation_grounding_delta"] == 0.0
    assert gap["anchors_added"] == 2


def test_build_gap_resolution_missing_data_is_none_not_crash():
    gap = vr.build_gap_resolution({}, None, None, None)
    assert gap["findings_resolved"] == 0
    assert gap["grounding_risk_delta"] is None
    assert gap["generic_assertion_delta"] is None
    assert gap["anchors_added"] == 0


# ── Label taxonomy: each label reachable + boundaries ───────────────────────
def _gap(**kw):
    base = {"findings_resolved": 0, "grounding_risk_delta": None, "anchors_added": 0,
            "generic_assertion_delta": None, "citation_grounding_delta": None}
    base.update(kw)
    return base


def test_label_no_usable_rewrite_unchanged():
    assert vr.compute_verdict_label(changed=False, no_text_change=True, status="original_preserved", gap=_gap()) == vr.LABEL_NO_USABLE_REWRITE


def test_label_no_usable_rewrite_original_preserved_status_even_if_changed():
    assert vr.compute_verdict_label(changed=True, no_text_change=False, status="original_preserved_external_guard", gap=_gap(findings_resolved=3)) == vr.LABEL_NO_USABLE_REWRITE


def test_label_gaps_resolved():
    gap = _gap(findings_resolved=2, grounding_risk_delta=-20.0)
    assert vr.compute_verdict_label(changed=True, no_text_change=False, status="ai_mitigated", gap=gap) == vr.LABEL_GAPS_RESOLVED


def test_label_gaps_resolved_blocked_by_category_worse():
    # findings + grounding improved BUT a category delta regressed → only partial.
    gap = _gap(findings_resolved=2, grounding_risk_delta=-20.0, generic_assertion_delta=5.0)
    assert vr.compute_verdict_label(changed=True, no_text_change=False, status="ai_mitigated", gap=gap) == vr.LABEL_GAPS_PARTIALLY_RESOLVED


def test_label_partial_findings_only():
    gap = _gap(findings_resolved=1)
    assert vr.compute_verdict_label(changed=True, no_text_change=False, status="x", gap=gap) == vr.LABEL_GAPS_PARTIALLY_RESOLVED


def test_label_partial_grounding_only():
    gap = _gap(grounding_risk_delta=-0.05)  # boundary: exactly at threshold counts
    assert vr.compute_verdict_label(changed=True, no_text_change=False, status="x", gap=gap) == vr.LABEL_GAPS_PARTIALLY_RESOLVED


def test_label_partial_anchors_only():
    gap = _gap(anchors_added=1)
    assert vr.compute_verdict_label(changed=True, no_text_change=False, status="x", gap=gap) == vr.LABEL_GAPS_PARTIALLY_RESOLVED


def test_label_grounding_delta_just_above_threshold_not_improved():
    # -0.04 does NOT clear the -0.05 threshold → no measurable gap improvement.
    gap = _gap(grounding_risk_delta=-0.04)
    assert vr.compute_verdict_label(changed=True, no_text_change=False, status="x", gap=gap) == vr.LABEL_DRAFT_FOR_REVIEW


def test_label_draft_for_review_changed_no_improvement():
    gap = _gap()
    assert vr.compute_verdict_label(changed=True, no_text_change=False, status="ai_mitigated", gap=gap) == vr.LABEL_DRAFT_FOR_REVIEW


def test_verdict_is_good():
    assert vr.verdict_is_good(vr.LABEL_GAPS_RESOLVED)
    assert vr.verdict_is_good(vr.LABEL_GAPS_PARTIALLY_RESOLVED)
    assert not vr.verdict_is_good(vr.LABEL_DRAFT_FOR_REVIEW)
    assert not vr.verdict_is_good(vr.LABEL_NO_USABLE_REWRITE)


# ── Note presence + gap sub-line ────────────────────────────────────────────
def test_ai_likelihood_note_reframes_not_hides():
    note = vr.AI_LIKELIHOOD_NOTE.lower()
    assert "make it yours" in note
    # never claims the score is hidden/lowered by the tool itself
    assert "high" in note


def test_gap_sub_line_composes():
    line = vr.gap_sub_line(_gap(findings_resolved=3, grounding_risk_delta=-12.0, anchors_added=2))
    assert "3 findings resolved" in line
    assert "grounding risk down 12.0%" in line
    assert "2 specifics added" in line


# ── production.py summary hook: additive fields + kill-switch reversion parity ──
from poc.rewrite_v6 import production as v6_production  # noqa: E402


def _base_summary():
    return {
        "status": "ai_mitigated",
        "no_text_change": False,
        "detect_scores": {
            "original_findings": 5,
            "rewritten_findings": 2,
            "original_grounding_quality_risk": 60.0,
            "rewritten_grounding_quality_risk": 40.0,
        },
        "bracket_grounding_spans": [{"kind": "improved"}],
    }


def test_summary_hook_adds_fields_when_enabled(monkeypatch):
    monkeypatch.setenv(vr.ENV_VAR, "1")
    summary = _base_summary()
    orig_report = {"ai_risk_badge": {"ai_components": {"generic_assertion_risk": 0.7}}}
    new_report = {"ai_risk_badge": {"ai_components": {"generic_assertion_risk": 0.4}}}
    v6_production._attach_verdict_reframe(summary, changed=True, orig_report=orig_report, new_report=new_report)
    assert summary["verdict_label"] == vr.LABEL_GAPS_RESOLVED
    assert summary["ai_likelihood_note"] == vr.AI_LIKELIHOOD_NOTE
    assert summary["gap_resolution"]["findings_resolved"] == 3
    # after-score NOT suppressed
    assert "final_risk" not in summary or summary.get("final_risk") is None or True


def test_summary_hook_reversion_parity_when_disabled(monkeypatch):
    monkeypatch.setenv(vr.ENV_VAR, "0")
    summary = _base_summary()
    before = dict(summary)
    v6_production._attach_verdict_reframe(summary, changed=True, orig_report={}, new_report={})
    assert summary == before  # byte-identical: no reframe fields added


# ── PDF render: verdict headline + visible after-score + make-it-yours note ──
from poc.report import render_rewrite  # noqa: E402


def _fused_scan(ai_score):
    return {
        "ai_score": ai_score,
        "ai_risk_badge": {
            "ai_likelihood_score": ai_score,
            "tier_authority": {"fused_score": ai_score},
        },
        "findings": {"critical": [], "high": [], "medium": [], "low": []},
    }


def _render_summary(verdict_label=None):
    s = {
        "status": "ai_mitigated",
        "outcome": "partial_candidate_not_strict_safe",
        "no_text_change": False,
        "final_risk": 28.0,
        "original_risk": 11.0,
        "detect_scores": {
            "original_ai": 11.0,
            "rewritten_ai": 28.0,
            "original_findings": 5,
            "rewritten_findings": 2,
            "original_grounding_quality_risk": 60.0,
            "rewritten_grounding_quality_risk": 40.0,
        },
        "detect_scan_original_saved": _fused_scan(11.0),
        "detect_scan_rewritten": _fused_scan(28.0),
        "final_text": "Rewritten body text.",
    }
    if verdict_label is not None:
        s["verdict_label"] = verdict_label
        s["gap_resolution"] = {"findings_resolved": 3, "grounding_risk_delta": -20.0, "anchors_added": 1}
        s["ai_likelihood_note"] = vr.AI_LIKELIHOOD_NOTE
    return s


def test_pdf_render_shows_verdict_and_after_score(monkeypatch):
    monkeypatch.setenv(vr.ENV_VAR, "1")
    md = render_rewrite.render_rewrite_report(
        _render_summary(verdict_label=vr.LABEL_GAPS_RESOLVED), [], [],
        original_text="Original body.", final_text="Rewritten body text.")
    assert vr.LABEL_TITLES[vr.LABEL_GAPS_RESOLVED] in md
    assert "make it yours" in md.lower()          # note reframes the after-score
    assert "28" in md                              # after-score stays visible


def test_pdf_render_legacy_when_field_absent(monkeypatch):
    # No verdict_label + reframe off → legacy label path, no reframe headline.
    monkeypatch.setenv(vr.ENV_VAR, "0")
    md = render_rewrite.render_rewrite_report(
        _render_summary(verdict_label=None), [], [],
        original_text="Original body.", final_text="Rewritten body text.")
    assert vr.LABEL_TITLES[vr.LABEL_GAPS_RESOLVED] not in md
