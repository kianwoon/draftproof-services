"""Phase-2 N0 CI guard — phase-conditional verification-state legality.

Governing plan: docs/plans/phase2_entailment_grounding_scope.md §3/§5 (N0 row),
kickoff decision 3 (``corroborated`` stays FORBIDDEN this phase).

The N0 deliverable is a governance guard, not a signal: it proves the schema
UNLOCKS ``verified`` only when ``DRAFTPROOF_ENTAILMENT`` is on, that
``corroborated`` stays rejected regardless, and that the Phase-1 invariant is
byte-identical when the flag is off. No LLM, no network — pure validators.
"""
from __future__ import annotations

import pytest

from claim_graph import entailment_enabled, ENTAILMENT_KILL_SWITCH_ENV
from claim_graph import schema
from claim_graph.build import build_claim_graph


ENV = ENTAILMENT_KILL_SWITCH_ENV


def _segments():
    s0 = "The intervention reduced processing time by 35%."
    return [
        {"sentence_id": "s001", "paragraph_id": "p001", "start_char": 100,
         "end_char": 100 + len(s0), "sentence": s0},
    ]


def _claim(status):
    return {"sentence_id": "s001", "quote": "reduced processing time by 35%",
            "node_type": "CLAIM", "claim_type": "factual",
            "verification_status": status}


def _build(status):
    return build_claim_graph("t", _segments(), {"claims": [_claim(status)]})


# ── Flag default OFF ─────────────────────────────────────────────────────────
def test_entailment_default_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert entailment_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_entailment_on_values(monkeypatch, val):
    monkeypatch.setenv(ENV, val)
    assert entailment_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
def test_entailment_off_values(monkeypatch, val):
    monkeypatch.setenv(ENV, val)
    assert entailment_enabled() is False


# ── (a) flag OFF → verified AND corroborated both REJECTED (Phase-1 invariant) ─
def test_flag_off_verified_rejected(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _build("verified")["claims"] == []


def test_flag_off_corroborated_rejected(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _build("corroborated")["claims"] == []


# ── (b) flag ON → verified ACCEPTED, corroborated STILL rejected ─────────────
def test_flag_on_verified_accepted(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    graph = _build("verified")
    assert len(graph["claims"]) == 1
    assert graph["claims"][0]["verification_status"] == "verified"


def test_flag_on_corroborated_still_rejected(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert _build("corroborated")["claims"] == []


def test_phase2_forbidden_states_constant():
    # corroborated is the only Phase-2 forbidden state (kickoff decision 3).
    assert schema.PHASE2_FORBIDDEN_STATES == ("corroborated",)
    # Phase-1 constant is untouched (byte-identical Phase-1 path).
    assert schema.PHASE1_FORBIDDEN_STATES == ("verified", "corroborated")


# ── (c) the existing scoring_enabled ⇒ fairness_gate guard still holds ────────
def test_scoring_promotion_ci_guard_still_holds():
    """N0 must not disturb the M5 §B promotion guard (test_signals mirror)."""
    from claim_graph import signals
    from claim_graph.schema import ClaimNode

    assert signals._META_BASE["scoring_enabled"] is False

    class _FakeEmbedder:
        available = True

        def encode(self, texts):
            import math
            return [[float(len(t) % 7), math.pi] for t in texts]

    claims = [ClaimNode(id="c_001", node_type="CLAIM",
                        text="The 2019 trial cut cost by 35%.",
                        origins=["external_source"])]
    sigs, _ = signals.compute_signals(claims, [], text="doc", embedder=_FakeEmbedder())
    assert sigs
    for s in sigs:
        if s.get("scoring_enabled"):
            assert s.get("fairness_gate_passed") is True
            assert s.get("calibration_version")
