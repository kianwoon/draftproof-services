"""Report-level kill-switch parity for the Phase-1 claim-graph attach seam.

Carry-in from the M1 review: prove that with DRAFTPROOF_CLAIM_GRAPH OFF the
report dict is BYTE-IDENTICAL to unset (the experimental graph is a pure no-op),
and that with the flag ON the on-path attach populates the graph.

Follows the report-level test patterns in poc/test_headline_confidence.py.
full report_to_dict is heavy (imports the ML-adjacent detect stack); per the M1
review note, this uses the LIGHTER targeted approach — it drives the REAL attach
function ``report.report.attach_claim_graph_experimental`` with a realistic
result dict. That function IS the production seam (report_to_dict calls exactly
it), so the parity guarantee is on the real code path, not a reimplementation.
"""
from __future__ import annotations

import json

from report.report import attach_claim_graph_experimental


_TEXT = (
    "The intervention reduced processing time by 35 percent. "
    "It also improved satisfaction across every cohort we surveyed."
)


def _result():
    # Realistic post-report shape: the graph attaches UNDER authorship_evidence.
    return {
        "ai_risk_badge": {"tier": "green", "ai_likelihood_score": 0.0},
        "authorship_evidence": {"authorship_trace": {}, "evidence_levels": []},
        "findings": [],
    }


def test_off_vs_unset_byte_identical(monkeypatch):
    # unset
    monkeypatch.delenv("DRAFTPROOF_CLAIM_GRAPH", raising=False)
    unset = attach_claim_graph_experimental(_result(), _TEXT)
    # explicit off
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH", "0")
    off = attach_claim_graph_experimental(_result(), _TEXT)
    assert json.dumps(unset, sort_keys=True) == json.dumps(off, sort_keys=True)
    # and neither attached anything (byte-identical to the untouched input)
    assert json.dumps(unset, sort_keys=True) == json.dumps(_result(), sort_keys=True)
    assert "claim_graph" not in unset["authorship_evidence"]


def test_on_path_presence_empty_container_without_gateway(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH", "1")
    # No LLM keys in the unit env → the seam attaches the empty-but-valid
    # plumbing container (on-path presence without any network call).
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    out = attach_claim_graph_experimental(_result(), _TEXT)
    cg = out["authorship_evidence"]["claim_graph"]
    assert cg["schema_version"] == "cg-1"
    assert cg["status"] == "experimental"
    assert cg["kill_switch"] == "DRAFTPROOF_CLAIM_GRAPH"
    assert cg["claims"] == [] and cg["edges"] == []


def test_missing_authorship_evidence_is_non_fatal(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH", "1")
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    result = {"findings": []}  # no authorship_evidence object to attach under
    out = attach_claim_graph_experimental(result, _TEXT)
    assert out is result  # fail-open: returned unchanged, no exception
    assert "claim_graph" not in out
