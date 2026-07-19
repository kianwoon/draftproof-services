"""Regression test: the PDF's policy-risk panel (AI-Allowed / AI-Restricted,
poc/report/render.py::_render_policy_risk) was only ever called from a dead
`elif report.ai_risk_badge:` branch — identical condition to the `if` above it,
so Python could never take it. The panel has never rendered in any PDF for any
report. Fixed 2026-07-20 by moving the call into the live branch, right after
the claim-graph panel.

Gated on authorship_breakdown's PRESENCE (independent-review finding,
2026-07-20): render_merged_authorship_risk()->render_scan_lead() already
renders its OWN legacy "Policy view" cards (render_panels.py ~895) whenever
authorship_breakdown is ABSENT. The fixture below therefore MUST carry an
authorship_breakdown, or this test would silently exercise the legacy-cards
path instead of the new panel (the mistake in the first version of this test —
its "36"/"33" assertions passed against the legacy cards' numeric score, not
against _render_policy_risk's own output).

2026-07-20 follow-up: owner found the panel level-only ("Moderate", no "36")
on a real downloaded PDF while the web PolicyRiskView and the legacy cards
both show the number. _policy_row now appends the rounded score ("Moderate
36"), matching the legacy cards' `f"{level} {round(score)}"` convention --
see the score assertions below, which now assert PRESENCE (the opposite of
the original test's design, which deliberately avoided asserting scores
because this panel didn't render them yet)."""
from report.report import DraftReport, Tier
from report.render import render_report

_POLICY_RISK = {
    "ai_allowed": {
        "score": 36.2, "level": "moderate", "main_issue": "grounding_gap",
        "confirm_factor": "declaration_gap", "confirm_delta": 5.43, "confirm_level": "moderate",
    },
    "ai_restricted": {
        "score": 36.2, "level": "moderate", "main_issue": "surface_ai_text_signal",
        "confirm_factor": "process_defensibility_gap", "confirm_delta": 3.34, "confirm_level": "moderate",
        "floored_to_ai_allowed": True, "pre_floor_score": 33.39,
    },
}

_AUTHORSHIP_BREAKDOWN = {
    "document_breakdown_raw": {
        "student_owned": 0.20, "ai_assisted_polished": 0.30,
        "ai_paraphrased": 0.15, "ai_generated_like": 0.35,
    },
    "document_breakdown_bands": {
        "student_owned": "Little", "ai_assisted_polished": "Some",
        "ai_paraphrased": "Little", "ai_generated_like": "Some",
    },
    "primary_category": "ai_generated_like", "confidence": "medium",
    "deep_scan": {"proportion": 1.0, "band": "red", "calibrated": False},
    "disclaimer": "DraftProof provides authorship clarity signals. It does not determine misconduct.",
}

_BADGE = {
    "tier": "red",
    "ai_likelihood_score": 66.75,
    "tier_authority": {
        "source": "v7_fused", "fused_score": 66.75,
        "composite_score": 5.0, "proportion": 1.0, "flag_line": 32,
    },
    "authorship_breakdown": _AUTHORSHIP_BREAKDOWN,
    "submission_risk": {
        "overall": {"level": "high", "main_reason": "text pattern"},
        "axes": {
            "text_pattern": {"level": "high", "label": "Text-pattern risk"},
            "ownership": {"level": "low", "label": "Ownership risk"},
            "citation": {"level": "medium", "label": "Citation risk"},
            "defence_readiness": {"level": "low", "label": "Defence-readiness risk"},
            "policy_declaration": {"level": "unknown", "label": "Policy / declaration risk"},
        },
    },
    "policy_risk": _POLICY_RISK,
}


def _report(badge):
    return DraftReport(
        overall_tier=Tier.CRITICAL,
        finding_count=0,
        findings_by_tier={},
        ai_risk_badge=badge,
        generated_at="2026-07-20 01:00 SGT",
        original_text="Some submitted text.",
    )


def test_policy_risk_panel_reaches_the_rendered_pdf_markdown():
    md = render_report(_report(_BADGE))
    assert "Policy risk" in md
    assert "AI is allowed" in md
    assert "AI is not allowed" in md
    # _render_policy_risk's own content (render.py::_policy_row) — level + main-issue
    # text uniquely identifies THIS panel vs the legacy cards.
    assert "weak source grounding" in md
    assert "polished, AI-like surface style" in md


def test_policy_risk_panel_shows_the_score_number_not_just_the_level():
    # Owner found this panel showing "Moderate" with no number on a real downloaded
    # PDF, while the web PolicyRiskView and the legacy cards both show it. Both rows
    # in the fixture are 36.2 (post-floor) -- "Moderate 36" should appear once per
    # row, proving the score is attached to EACH row, not a one-off coincidence.
    md = render_report(_report(_BADGE))
    assert md.count("Moderate 36") == 2


def test_policy_risk_panel_absent_when_diagnosis_abstained():
    badge = {k: v for k, v in _BADGE.items() if k != "policy_risk"}
    md = render_report(_report(badge))
    assert "AI is allowed" not in md
    assert "AI is not allowed" not in md


def test_policy_risk_panel_absent_when_authorship_breakdown_missing():
    """The legacy "Policy view" cards inside render_merged_authorship_risk already
    cover this path (render_panels.py ~895) -- the new panel must stay silent here
    or the PDF would show policy content twice."""
    badge = {k: v for k, v in _BADGE.items() if k != "authorship_breakdown"}
    md = render_report(_report(badge))
    assert "Policy risk — how this draft may read under your school's AI policy" not in md


def test_dead_elif_branch_is_provably_unreachable():
    """Documents the root cause directly: `if X: ... elif X: ...` can never take
    the elif for any value of X. Line-exact match (not substring) -- a naive
    `"if report.ai_risk_badge:" in src` check is trivially satisfied by the elif
    line itself ("elif ..." contains "if ..." as a substring starting at index 2),
    which is why the first version of this test didn't actually prove anything."""
    import inspect
    from report import render as render_module

    src_lines = [line.strip() for line in inspect.getsource(render_module.render_report).splitlines()]
    if_count = sum(1 for line in src_lines if line == "if report.ai_risk_badge:")
    elif_count = sum(1 for line in src_lines if line == "elif report.ai_risk_badge:")
    assert if_count >= 1
    assert elif_count >= 1
