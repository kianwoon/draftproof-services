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
        assert gw["value"] is not None
        assert "finetune-v1" in str(gw["value"])
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
