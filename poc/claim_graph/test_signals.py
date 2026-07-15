"""M3 tests: the three EXPERIMENTAL graph signals.

All signals are deterministic given their inputs (substitutability given the
embedder). The embedding model is INJECTED as a fake in band/determinism tests
so bands are reachable without MiniLM in CI; a dedicated test covers the
MiniLM-unavailable fail-open path.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from claim_graph import signals
from claim_graph.schema import ClaimNode, Edge


# ── Builders ─────────────────────────────────────────────────────────────────
def _claim(cid, text, *, status="unverified", origins=None, primary=None,
           node_type="CLAIM", char_start=0):
    origins = origins if origins is not None else []
    src = None
    if node_type == "CLAIM":
        src = {"paragraph_id": "p001", "sentence_id": "s%03d" % char_start,
               "char_start": char_start, "char_end": char_start + len(text)}
    return ClaimNode(id=cid, node_type=node_type, text=text, claim_type="factual",
                     source=src, verification_status=status,
                     origins=list(origins),
                     primary_origin=primary or (origins[0] if origins else None))


class _FakeEmbedder:
    """Returns a deterministic unit vector; cosine controllable via `sim_map`.

    For a pair [orig, neutralised] we return vectors whose cosine equals the
    value looked up from `sim_map` keyed by the ORIGINAL text (default 1.0)."""

    available = True

    def __init__(self, sim_map=None):
        self.sim_map = sim_map or {}
        self.calls = 0

    def encode(self, texts):
        # texts == [original, neutralised]; emit 2D unit vectors with the
        # requested cosine (theta from the map).
        import math
        self.calls += 1
        orig = texts[0]
        c = self.sim_map.get(orig, 1.0)
        theta = math.acos(max(-1.0, min(1.0, c)))
        return [[1.0, 0.0], [math.cos(theta), math.sin(theta)]]


# ── Interrogatability ────────────────────────────────────────────────────────
def test_interrogatability_low_band_generic_unverified():
    claims = [_claim("c_001", "This is important and things matter a lot.",
                     status="unverified")]
    sig, questions = signals.compute_interrogatability(claims, [])
    assert sig["signal"] == "interrogatability"
    assert sig["band"] == "low"
    assert sig["status"] == "experimental"
    assert sig["scoring_enabled"] is False
    assert set(sig["components"]) == {
        "specificity_presence", "causal_depth",
        "contextual_anchoring", "evidence_traceability"}


def test_interrogatability_high_band_rich_traceable():
    claims = [
        _claim("c_001", "The 2019 Stanford trial reduced latency by 35% in Boston.",
               status="internally_supported", origins=["external_source"], char_start=0),
        _claim("c_002", "The 2021 Oxford cohort of 4200 patients improved by 12%.",
               status="internally_supported", origins=["original_analysis"], char_start=100),
    ]
    edges = [Edge("x_001", "causes", "c_001", "c_002"),
             Edge("x_002", "explains", "c_002", "c_001")]
    sig, _ = signals.compute_interrogatability(claims, edges)
    assert sig["band"] == "high"
    assert sig["components"]["causal_depth"] > 0


def test_interrogatability_emits_question_nodes_for_unverified_specifics():
    claims = [_claim("c_001", "The 2019 Stanford trial cut cost by 35%.",
                     status="unverified", char_start=0)]
    sig, questions = signals.compute_interrogatability(claims, [])
    assert questions, "expected QUESTION nodes for unverified specifics"
    assert all(q.node_type == "QUESTION" for q in questions)
    assert all(q.source is None for q in questions)
    assert all(q.references == "c_001" for q in questions)
    assert sig["value"]["unverified_specific_rate"] == 1.0


def test_interrogatability_verified_specifics_emit_no_questions():
    claims = [_claim("c_001", "The 2019 Stanford trial cut cost by 35%.",
                     status="internally_supported", origins=["external_source"])]
    _, questions = signals.compute_interrogatability(claims, [])
    assert questions == []


# ── Substitutability ─────────────────────────────────────────────────────────
def test_substitutability_high_band_when_meaning_unchanged():
    claims = [_claim("c_001", "This clearly matters for everyone involved today.")]
    emb = _FakeEmbedder()  # cosine 1.0 -> maximally substitutable
    sig = signals.compute_substitutability(claims, emb)
    assert sig["signal"] == "substitutability"
    assert sig["band"] == "high"
    assert sig["reconciles_with"] == ["generic_assertion_risk"]


def test_substitutability_low_band_when_swap_changes_meaning():
    txt = "The 2019 Stanford trial reduced latency by 35%."
    claims = [_claim("c_001", txt)]
    emb = _FakeEmbedder(sim_map={txt: 0.55})
    sig = signals.compute_substitutability(claims, emb)
    assert sig["band"] == "low"
    assert sig["value"] < 0.8


def test_substitutability_fail_open_when_model_unavailable():
    claims = [_claim("c_001", "Anything at all here.")]
    sig = signals.compute_substitutability(claims, None)
    assert sig["value"] is None
    assert sig["band"] is None
    assert "embedding model unavailable" in " ".join(sig["limitations"]).lower()


# ── Origin map ───────────────────────────────────────────────────────────────
def test_origin_map_multi_label_aggregation_and_primary():
    claims = [
        _claim("c_001", "A.", origins=["external_source", "interpretation"],
               primary="external_source"),
        _claim("c_002", "B.", origins=["external_source"], primary="external_source"),
        _claim("c_003", "C.", origins=["personal_observation"],
               primary="personal_observation"),
    ]
    sig = signals.compute_origin_map(claims)
    assert sig["signal"] == "origin_map"
    assert sig["value"]["primary_origin"] == "external_source"
    dist = sig["value"]["distribution"]
    # distribution normalises over LABEL occurrences: external_source = 2 of 4.
    assert dist["external_source"] == pytest.approx(2 / 4)


def test_origin_map_flags_unsupported_assertion_heavy():
    claims = [_claim("c_%03d" % i, "x.", origins=[], status="unverified")
              for i in range(4)]
    sig = signals.compute_origin_map(claims)
    assert sig["value"]["unsupported_assertion_heavy"] is True


# ── Orchestrator: fail-open isolation + lifecycle + determinism ──────────────
def test_compute_signals_all_three_present_with_lifecycle():
    claims = [_claim("c_001", "The 2019 trial cut cost by 35%.", origins=["external_source"])]
    sigs, questions = signals.compute_signals(claims, [], text="doc", embedder=_FakeEmbedder())
    names = {s["signal"] for s in sigs}
    assert names == {"interrogatability", "substitutability", "origin_map"}
    for s in sigs:
        assert s["status"] == "experimental"
        assert s["scoring_enabled"] is False
        assert s["calibration_version"] is None
        assert s["fairness_gate_passed"] is None


def test_per_signal_fail_open_isolation():
    class _Boom:
        available = True

        def encode(self, texts):
            raise RuntimeError("embed boom")

    claims = [_claim("c_001", "The 2019 trial cut cost by 35%.", origins=["external_source"])]
    sigs, _ = signals.compute_signals(claims, [], text="doc", embedder=_Boom())
    by = {s["signal"]: s for s in sigs}
    assert by["substitutability"]["value"] is None  # failed open
    assert by["interrogatability"]["value"] is not None  # unaffected
    assert by["origin_map"]["value"] is not None


def test_determinism_same_input_same_output():
    claims = [_claim("c_001", "The 2019 Stanford trial cut cost by 35%.",
                     origins=["external_source"])]
    a = signals.compute_signals(claims, [], text="doc", embedder=_FakeEmbedder())
    b = signals.compute_signals(claims, [], text="doc", embedder=_FakeEmbedder())
    assert json.dumps(a[0], sort_keys=True) == json.dumps(b[0], sort_keys=True)


# ── Integration: extraction -> signals present ───────────────────────────────
class _FakeGateway:
    def __init__(self, responses, model="fake/gpt-oss-120b"):
        self.responses = list(responses)
        self.model = model

    def chat(self, prompt, system=None, response_format=None, **kwargs):
        item = self.responses.pop(0) if self.responses else "{}"
        return SimpleNamespace(content=item)


def test_integration_extraction_attaches_experimental_signals(monkeypatch):
    from claim_graph import extract
    from claim_graph.build import build_claim_graph_extracted
    extract.clear_cache()
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")
    s0 = "The 2019 Stanford trial reduced processing time by 35%."
    segs = [{"sentence_id": "s001", "paragraph_id": "p001", "start_char": 0,
             "end_char": len(s0), "sentence": s0}]
    body = json.dumps({"claims": [{
        "sentence_id": "s001", "quote": "reduced processing time by 35%",
        "node_type": "CLAIM", "claim_type": "factual",
        "origins": ["external_source"], "primary_origin": "external_source"}],
        "edges": []})
    graph = build_claim_graph_extracted(s0, segs, _FakeGateway([body]))
    names = {s["signal"] for s in graph["signals"]}
    assert names == {"interrogatability", "substitutability", "origin_map"}
    assert all(s["status"] == "experimental" for s in graph["signals"])
