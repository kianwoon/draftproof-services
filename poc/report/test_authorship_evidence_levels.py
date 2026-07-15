"""Tests for the Phase-0 Authorship Evidence Level composer.

Governing spec: docs/plans/credible_authorship_assessment_v2.md
(§H Phase-0 row, §10 Evidence Levels, §C confidence/coverage contract,
§B signal lifecycle). Phase 0 = Evidence Levels 0-2 only, display-layer only,
lifecycle status ADVISORY (fusion_weight = 0 — must NOT touch any scoring path).
"""
from __future__ import annotations

import os
import subprocess
import sys

from report.authorship_evidence_levels import (
    compute_evidence_level,
    evidence_levels_enabled,
)


# ── Kill-switch (default ON; "0"/"false" reverts byte-identically) ─────────

def test_kill_switch_default_on():
    os.environ.pop("DRAFTPROOF_AUTHORSHIP_EVIDENCE_LEVELS", None)
    assert evidence_levels_enabled() is True


def test_kill_switch_off_values():
    for val in ("0", "false", "FALSE", "off", "no"):
        os.environ["DRAFTPROOF_AUTHORSHIP_EVIDENCE_LEVELS"] = val
        try:
            assert evidence_levels_enabled() is False, val
        finally:
            del os.environ["DRAFTPROOF_AUTHORSHIP_EVIDENCE_LEVELS"]


def test_kill_switch_on_values():
    for val in ("1", "true", "on", "yes"):
        os.environ["DRAFTPROOF_AUTHORSHIP_EVIDENCE_LEVELS"] = val
        try:
            assert evidence_levels_enabled() is True, val
        finally:
            del os.environ["DRAFTPROOF_AUTHORSHIP_EVIDENCE_LEVELS"]


def _fused_badge(**overrides):
    """A badge that fired the v7 fused statistical path with content lenses."""
    badge = {
        "tier": "amber",
        "signal_source": "v7_fused",
        "ai_signal_deberta": {"available": True, "band": "amber", "signal_pct": 33},
        "grounding_diagnosis": {"band": "moderate", "low_coverage": False,
                                "primary_driver_label": "generic phrasing"},
        "critical_thinking_control": {"band": "acceptable_control", "score": 0.6},
        "submission_risk": {"overall": {"level": "medium"},
                            "axes": {"citation": {"level": "high"}}},
        "ai_components": {"generic_assertion_risk": 40},
    }
    badge.update(overrides)
    return badge


# ── Level mapping ──────────────────────────────────────────────────────────

def test_level_2_when_statistical_and_content_lenses_present():
    out = compute_evidence_level(_fused_badge(), {})
    assert out["level"] == 2
    assert out["max_level_assessable"] == 2
    assert set(out["lenses"]) == {"ai_pattern", "grounding", "citation", "reasoning"}


def test_level_1_when_only_statistical_indication():
    badge = {"tier": "amber", "signal_source": "v7_fused",
             "ai_signal_deberta": {"available": True, "band": "amber", "signal_pct": 20}}
    out = compute_evidence_level(badge, {})
    assert out["level"] == 1


def test_level_0_when_nothing_assessable():
    out = compute_evidence_level({"tier": ""}, {})
    assert out["level"] == 0


# ── Lens re-presentation (no new numbers) ──────────────────────────────────

def test_grounding_lens_reflects_existing_band():
    out = compute_evidence_level(_fused_badge(), {})
    assert out["lenses"]["grounding"]["band"] == "moderate"


def test_citation_lens_reflects_submission_risk_axis():
    out = compute_evidence_level(_fused_badge(), {})
    assert out["lenses"]["citation"]["level"] == "high"


def test_reasoning_lens_capped_at_moderate_confidence():
    # LLM-judged lens (project memory: rating over-judges fluent human text)
    # must never claim "high" assessment confidence.
    badge = _fused_badge(critical_thinking_control={"band": "strong_control", "score": 0.95})
    out = compute_evidence_level(badge, {})
    assert out["lenses"]["reasoning"]["assessment_confidence"] in ("low", "moderate")


def test_ai_pattern_low_confidence_when_headline_contradicted():
    badge = _fused_badge(headline_confidence={"level": "low", "reasons": ["second_opinion_divergence"]})
    out = compute_evidence_level(badge, {})
    assert out["lenses"]["ai_pattern"]["assessment_confidence"] == "low"


# ── Confidence + coverage contract (§C) ────────────────────────────────────

def test_assessment_confidence_banded_not_percentage():
    out = compute_evidence_level(_fused_badge(), {})
    assert out["assessment_confidence"] in ("high", "moderate", "low")
    assert isinstance(out["confidence_reasons"], list)


def test_coverage_is_typed_objects():
    out = compute_evidence_level(_fused_badge(), {})
    types = {c["type"] for c in out["coverage"]}
    assert "generator_window" in types
    assert "context_availability" in types


def test_context_availability_reports_brief_absent():
    out = compute_evidence_level(_fused_badge(), {})
    ctx = [c for c in out["coverage"] if c["type"] == "context_availability"][0]
    assert ctx["value"] == "assignment_brief_absent"


def test_limitations_include_no_assignment_context():
    out = compute_evidence_level(_fused_badge(), {})
    assert "no_assignment_context" in out["limitations"]


def test_short_paragraph_limitation_when_low_coverage():
    badge = _fused_badge(grounding_diagnosis={"band": "moderate", "low_coverage": True})
    out = compute_evidence_level(badge, {})
    assert "short_paragraphs_low_confidence" in out["limitations"]


# ── generator_window derivation (§C — DATA-derived, never hardcoded) ───────

def test_generator_window_derived_from_env_checkpoint(monkeypatch=None):
    os.environ["DRAFTPROOF_MODAL_CHECKPOINT"] = "finetune-v1"
    try:
        out = compute_evidence_level(_fused_badge(), {})
        gw = [c for c in out["coverage"] if c["type"] == "generator_window"][0]
        # The WINDOW is the provenance DATE (user-meaningful coverage claim);
        # the env checkpoint tag rides along as `model` — 2026-07-15 live fix:
        # the slug was displayed where a date belonged.
        assert gw["value"] is not None and str(gw["value"]).startswith("through ")
        assert gw.get("model") == "finetune-v1"
    finally:
        del os.environ["DRAFTPROOF_MODAL_CHECKPOINT"]


def test_generator_window_derived_from_weights_provenance_when_no_env():
    os.environ.pop("DRAFTPROOF_MODAL_CHECKPOINT", None)
    out = compute_evidence_level(_fused_badge(), {})
    gw = [c for c in out["coverage"] if c["type"] == "generator_window"][0]
    # weights.json deep_scan_calibration provenance carries a YYYY-MM-DD; the
    # derived window must reflect that, never a hardcoded literal. Either a real
    # derived value or an honest null+note — never a fabricated date string.
    assert "value" in gw
    if gw["value"] is None:
        assert gw.get("note") == "provenance unavailable"


# ── Lifecycle (§B — ADVISORY, fusion_weight 0) ─────────────────────────────

def test_lifecycle_is_advisory_and_not_scoring():
    out = compute_evidence_level(_fused_badge(), {})
    lc = out["lifecycle"]
    assert lc["status"] == "advisory"
    assert lc["scoring_enabled"] is False
    assert lc["calibration_version"] is None
    assert lc["fairness_gate_passed"] is None


# ── Fail-open (annotator must never break a report build) ──────────────────

def test_fail_open_on_non_dict_badge():
    assert compute_evidence_level(None, {}) is None
    assert compute_evidence_level("garbage", {}) is None


def test_fail_open_on_malformed_lens_fields():
    badge = {"tier": "amber", "signal_source": "v7_fused",
             "grounding_diagnosis": "not-a-dict",
             "submission_risk": 12,
             "critical_thinking_control": []}
    out = compute_evidence_level(badge, {})
    assert out is not None
    assert out["level"] >= 1


# ── Specific per-lens anchors (display enrichment, NO FABRICATION) ─────────

def _real_report_fields():
    """A /tmp/dp_r2b.json-shaped subset: real flagged highlight_segments + the
    three driver findings the anchors read from."""
    return {
        "highlight_segments": [
            {"sentence_id": "u_s001", "text": "SOPs can now be turned into workflows.",
             "highlight": {"enabled": False}, "primary_signal": None},
            {"sentence_id": "s001",
             "text": 'And once that happens, "years of experience" becomes a pretty weak metric.',
             "highlight": {"enabled": True, "label": "High-confidence AI signal"},
             "primary_signal": {"score": 100}},
            {"sentence_id": "s002",
             "text": "Someone might have 20 years under their belt, but in reality it repeats.",
             "highlight": {"enabled": True}, "primary_signal": {"score": 90}},
        ],
        "findings": {
            "critical": [], "high": [
                {"category": "semantic_shape", "title": "semantic_drift",
                 "evidence": "SOPs can now be turned into workflows. Experience can be turned into skills.",
                 "detail": "Some adjacent sentence jumps show semantic drift. Score: 82%."},
            ],
            "medium": [
                {"category": "ai_generation", "title": "low_specificity",
                 "evidence": {"type": "document_level",
                              "example_sentences": [
                                  "Make sure only a handful of people know how to handle the edge cases.",
                                  "That stuff is getting codified, automated, or straight-up handled by AI."]}},
            ],
            "low": [],
        },
        "citation": {"in_text_count": 0, "bib_entry_count": 0},
    }


def test_ai_pattern_anchor_counts_and_strongest_example():
    out = compute_evidence_level(_fused_badge(), _real_report_fields())
    a = out["lenses"]["ai_pattern"]["anchor"]
    assert a["count"] == {"flagged": 2, "total": 3}
    assert a["headline"] == "Reads as AI-generated — 2 of 3 sentences"
    # strongest = highest primary_signal.score (s001 @ 100).
    assert a["example"] == 'And once that happens, "years of experience" becomes a pretty weak metric.'


def test_grounding_anchor_from_low_specificity_finding():
    out = compute_evidence_level(_fused_badge(), _real_report_fields())
    a = out["lenses"]["grounding"]["anchor"]
    assert "grounding" in a["headline"].lower()
    assert a["example"] == "Make sure only a handful of people know how to handle the edge cases."
    assert a["fix"]


def test_citation_anchor_reports_no_sources():
    out = compute_evidence_level(_fused_badge(), _real_report_fields())
    a = out["lenses"]["citation"]["anchor"]
    assert "No sources cited" in a["headline"]
    assert a["fix"]


def test_reasoning_anchor_from_semantic_drift_finding():
    out = compute_evidence_level(_fused_badge(), _real_report_fields())
    a = out["lenses"]["reasoning"]["anchor"]
    assert "jump" in a["headline"].lower()
    assert a["example"].startswith("SOPs can now be turned into workflows.")
    assert a["fix"]


def test_no_fabrication_every_example_is_a_substring_of_input():
    """The whole point: an anchor example must be VERBATIM report data, never
    invented. Collect every candidate sentence in the input and assert each
    rendered example (minus a truncation ellipsis) is a substring of one."""
    rf = _real_report_fields()
    corpus = [s["text"] for s in rf["highlight_segments"]]
    for sev in rf["findings"].values():
        for it in sev:
            ev = it.get("evidence")
            if isinstance(ev, str):
                corpus.append(ev)
            elif isinstance(ev, dict):
                corpus.extend(ev.get("example_sentences") or [])
    out = compute_evidence_level(_fused_badge(), rf)
    seen_example = False
    for lens in out["lenses"].values():
        a = lens.get("anchor") if isinstance(lens, dict) else None
        ex = a.get("example") if isinstance(a, dict) else None
        if not ex:
            continue
        seen_example = True
        probe = ex[:-1] if ex.endswith("…") else ex  # drop truncation marker
        assert any(probe in c for c in corpus), f"fabricated example: {ex!r}"
    assert seen_example, "fixture should exercise at least one example"


def test_example_is_end_trimmed_to_max_chars():
    rf = _real_report_fields()
    long = "x" * 400 + " tail-should-be-cut"
    rf["findings"]["high"][0]["evidence"] = long
    out = compute_evidence_level(_fused_badge(), rf)
    ex = out["lenses"]["reasoning"]["anchor"]["example"]
    assert len(ex) <= 141 and ex.endswith("…")  # 140 + ellipsis
    assert long.startswith(ex[:-1])  # verbatim prefix, nothing invented


def test_anchors_absent_when_no_report_data_falls_back_to_band_only():
    """Older reports / absent data -> no anchor key on any lens; panel renders as
    before (band only). Report never breaks."""
    out = compute_evidence_level(_fused_badge(), {})
    for lens in out["lenses"].values():
        assert "anchor" not in lens
    # And a wholly empty report_fields is fine too.
    out2 = compute_evidence_level(_fused_badge(), None)
    for lens in out2["lenses"].values():
        assert "anchor" not in lens


def test_anchor_derivation_never_raises_on_malformed_data():
    bad = {"highlight_segments": "nope", "findings": 12, "citation": []}
    out = compute_evidence_level(_fused_badge(), bad)
    assert out is not None
    for lens in out["lenses"].values():
        assert "anchor" not in lens


# ── Circular import guard (report/* must not import rewrite_v6/*) ───────────

def test_no_circular_import_under_worker_import_order():
    """The composer must import cleanly in a FRESH interpreter without pulling
    rewrite_v6/* (mirrors test_no_circular_import_under_worker_import_order in
    poc/test_rewrite_verdict_reframe.py — this class of bug crashed every prod
    rewrite task, commit 2a409b88)."""
    import pathlib
    poc = pathlib.Path(__file__).resolve().parent.parent
    # Worker import order (mirror poc/test_rewrite_verdict_reframe.py): import the
    # composer standalone, then render_rewrite — a fresh interpreter surfaces any
    # partially-initialized-module cycle the composer might close.
    code = (
        "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');\n"
        "from report.authorship_evidence_levels import compute_evidence_level\n"
        "assert not any('rewrite_v6' in k for k in sys.modules), "
        "sorted(k for k in sys.modules if 'rewrite_v6' in k)\n"
        "from report.render_rewrite import render_rewrite_report\n"
        "print('IMPORT_OK')" % (str(poc), str(poc.parent))
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert "IMPORT_OK" in proc.stdout, f"stderr tail: {proc.stderr[-800:]}"
