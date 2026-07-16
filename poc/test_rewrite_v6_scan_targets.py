"""Tests for wiring enhanced-scan findings into the rewrite writer (P1/P2/P3).

P1 (per-sentence grounding/reasoning), P2 (claim-graph entailment), P3 (finding-aware verification).
Each feature is additive and kill-switched; these tests pin the join logic, the verdict filtering, the
prompt injection, the telemetry, and that every kill switch yields the pre-change behaviour.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest

from rewrite_v6.direct_prompts import apply_scan_target_instructions
from rewrite_v6.finding_verification import (
    apply_finding_verification,
    verify_finding_resolution,
)
from rewrite_v6.report_contracts import (
    _paragraph_flagged_sentences,
    _paragraph_unsupported_claims,
    extract_paragraph_diagnoses,
    paragraph_diagnoses_context,
)


class _P:
    def __init__(self, pid: str):
        self.id = pid


def _report_with_grounding_finding():
    """Minimal report: a grounding finding on s011 that lives in paragraph p004 (via highlight_segments)."""
    return {
        "highlight_segments": [
            {"sentence_id": "s011", "paragraph_id": "p004", "text": "When students work together they grow."},
            {"sentence_id": "s012", "paragraph_id": "p004", "text": "That is important."},
        ],
        "findings": {
            "medium": [
                {"title": "low_specificity", "sentence_id": "s011", "recommendation": "Add a concrete anchor."},
                {"title": "medium_predictability", "sentence_id": "s012"},  # excluded (not grounding/reasoning)
            ]
        },
    }


# ── P1 ──────────────────────────────────────────────────────────────────────

def test_p1_joins_grounding_finding_to_its_paragraph_and_text():
    out = _paragraph_flagged_sentences(_report_with_grounding_finding())
    assert list(out) == ["p004"]
    row = out["p004"][0]
    assert row["issue"] == "grounding"
    assert "students work together" in row["text"]
    assert row["fix"] == "Add a concrete anchor."


def test_p1_drops_unanchored_and_non_allowlisted_findings():
    report = {
        "highlight_segments": [{"sentence_id": "s1", "paragraph_id": "p1", "text": "x"}],
        "findings": {"medium": [
            {"title": "semantic_drift"},  # no sentence_id -> dropped (never paragraph-guessed)
            {"title": "high_topk_predictability", "sentence_id": "s1"},  # not in allowlist
        ]},
    }
    assert _paragraph_flagged_sentences(report) == {}


def test_p1_kill_switch_off_is_byte_identical(monkeypatch):
    report = _report_with_grounding_finding()
    monkeypatch.setenv("DRAFTPROOF_V6_FLAGGED_SENTENCES", "0")
    diags = extract_paragraph_diagnoses(report)
    assert all("flagged_sentences" not in d for d in diags.values())


def test_p1_default_on_attaches_to_diagnosis():
    diags = extract_paragraph_diagnoses(_report_with_grounding_finding())
    assert diags.get("p004", {}).get("flagged_sentences")


# ── P2 ──────────────────────────────────────────────────────────────────────

def _report_with_claim_graph():
    return {
        "authorship_evidence": {
            "claim_graph": {
                "claims": [
                    {"id": "c1", "node_type": "CLAIM", "text": "outcomes improve 30 percent",
                     "source": {"paragraph_id": "p002"}},
                    {"id": "c2", "node_type": "CLAIM", "text": "collaboration deepens learning",
                     "source": {"paragraph_id": "p003"}},
                    {"id": "c3", "node_type": "CLAIM", "text": "a verified claim",
                     "source": {"paragraph_id": "p003"}},
                ],
                "evidence": [
                    {"claim_ids": ["c1"], "detail": {"resolution": {
                        "status": "resolved", "locator": "https://x", "locator_type": "url",
                        "entailment": {"c1": {"status": "contradicted", "entailment_score": 0.12}}}}},
                    {"claim_ids": ["c2"], "detail": {"resolution": {"status": "paywalled"}}},
                    {"claim_ids": ["c3"], "detail": {"resolution": {
                        "status": "resolved",
                        "entailment": {"c3": {"status": "verified", "entailment_score": 0.95}}}}},
                ],
            }
        }
    }


def test_p2_keeps_non_verified_claims_and_skips_verified():
    out = _paragraph_unsupported_claims(_report_with_claim_graph())
    assert out["p002"][0]["verdict"] == "contradicted"
    assert out["p002"][0]["entailment_score"] == 0.12
    assert out["p003"][0]["verdict"] == "paywalled"
    verdicts = [r["verdict"] for rows in out.values() for r in rows]
    assert "verified" not in verdicts


def test_p2_untruncated_claim_text_and_why():
    out = _paragraph_unsupported_claims(_report_with_claim_graph())
    row = out["p002"][0]
    assert row["claim"] == "outcomes improve 30 percent"
    assert "soften" in row["why"] or "correct" in row["why"]


def test_p2_absent_claim_graph_is_empty():
    assert _paragraph_unsupported_claims({"findings": {}}) == {}


def test_p2_kill_switch_off(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_UNSUPPORTED_CLAIMS", "0")
    diags = extract_paragraph_diagnoses(_report_with_claim_graph())
    assert all("unsupported_claims" not in d for d in diags.values())


# ── Prompt injection ─────────────────────────────────────────────────────────

def test_apply_scan_target_instructions_injects_both():
    payload = {"instructions": []}
    diagnosis = {
        "flagged_sentences": [{"text": "x", "issue": "grounding"}],
        "unsupported_claims": [{"claim": "c", "verdict": "contradicted", "why": "w"}],
    }
    apply_scan_target_instructions(payload, diagnosis)
    assert payload["flagged_sentences"] and payload["unsupported_claims"]
    joined = " ".join(payload["instructions"])
    assert "flagged_sentences are the EXACT" in joined
    assert "unsupported_claims are claims" in joined


def test_apply_scan_target_instructions_noop_when_absent():
    payload = {"instructions": ["existing"]}
    apply_scan_target_instructions(payload, {})
    assert payload == {"instructions": ["existing"]}


# ── P3 ──────────────────────────────────────────────────────────────────────

def _final_report(still_grounding: bool):
    title = "low_specificity" if still_grounding else "medium_predictability"
    return {
        "highlight_segments": [{"sentence_id": "s11", "paragraph_id": "p004"}],
        "findings": {"medium": [{"title": title, "sentence_id": "s11"}]},
    }


def test_p3_counts_resolved_vs_persisted():
    paras = [_P("p004"), _P("p002")]
    diags = {
        "p004": {"flagged_sentences": [{"text": "x", "issue": "grounding"}]},
        "p002": {"unsupported_claims": [{"claim": "c", "verdict": "contradicted"}]},
    }
    with paragraph_diagnoses_context(diags):
        persisted = verify_finding_resolution(paras, _final_report(still_grounding=True))
        resolved = verify_finding_resolution(paras, _final_report(still_grounding=False))
    assert persisted["persisted"] == 1 and persisted["resolved"] == 0 and persisted["resolution_rate"] == 0.0
    assert resolved["resolved"] == 1 and resolved["persisted"] == 0 and resolved["resolution_rate"] == 1.0
    assert persisted["claim_targets_not_locally_verifiable"] == 1


@dataclass
class _Doc:
    rewritten_text: str = "text"
    pass_trace: list = field(default_factory=list)


def test_p3_wrapper_appends_pass_trace_row_and_review_flag():
    paras = _P("p004")

    class _Scan:
        paragraphs = [paras]

    diags = {"p004": {"flagged_sentences": [{"text": "x", "issue": "grounding"}]}}
    with paragraph_diagnoses_context(diags):
        doc = apply_finding_verification(_Doc(), _Scan(), lambda _t: _final_report(still_grounding=True))
    rows = [r for r in doc.pass_trace if r.get("selected_source") == "finding_verification"]
    assert rows and rows[0]["persisted"] == 1
    assert rows[0]["review_flags"][0]["added"] == "a grounding gap may remain"


def test_p3_kill_switch_off_appends_nothing(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_FINDING_VERIFICATION", "0")
    import importlib

    import rewrite_v6.finding_verification as fv
    importlib.reload(fv)

    class _Scan:
        paragraphs = [_P("p004")]

    try:
        doc = fv.apply_finding_verification(_Doc(), _Scan(), lambda _t: _final_report(True))
        assert doc.pass_trace == []
    finally:
        monkeypatch.delenv("DRAFTPROOF_V6_FINDING_VERIFICATION", raising=False)
        importlib.reload(fv)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
