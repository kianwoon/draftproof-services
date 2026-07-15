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


def test_scoring_promotion_ci_guard():
    """§B CI guard (M5): scoring_enabled ⇒ fairness_gate_passed and calibration_version.

    M5 owner decision 2026-07-15: NO signal promoted (M4 report — gaming set scores
    highest interrogatability; Phase-1 internally_supported = coherence, not truth).
    A signal may only flip scoring_enabled together with a real calibration_version
    and a passed fairness gate; flipping the flag alone must fail here.
    """
    assert signals._META_BASE["scoring_enabled"] is False
    claims = [_claim("c_001", "The 2019 trial cut cost by 35%.", origins=["external_source"])]
    sigs, _ = signals.compute_signals(claims, [], text="doc", embedder=_FakeEmbedder())
    assert sigs, "signals missing — guard has nothing to check"
    for s in sigs:
        if s.get("scoring_enabled"):
            assert s.get("fairness_gate_passed") is True, f"{s['signal']}: scoring without fairness gate"
            assert s.get("calibration_version"), f"{s['signal']}: scoring without calibration_version"


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


def test_question_nodes_never_evict_real_claims(monkeypatch):
    """M3 review gap (2026-07-15): QUESTION nodes are appended during signal
    computation; under cap pressure they must be evicted FIRST and never push
    a real CLAIM out. Build a container at exactly the claim cap with every
    claim unverified+specific (max QUESTION emission), then assert all CLAIM
    nodes survive and only QUESTION nodes were dropped."""
    import json
    from claim_graph import validators

    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_MAX_CLAIMS", "4")
    sents = []
    claims = []
    for i in range(1, 5):  # exactly at cap
        text = f"On day {i} the measured value was {i * 7} units exactly."
        sents.append({"sentence_id": f"s{i:03d}", "paragraph_id": "p001",
                      "start_char": (i - 1) * 60, "end_char": (i - 1) * 60 + len(text),
                      "sentence": text})
        claims.append({"sentence_id": f"s{i:03d}", "quote": text, "node_type": "CLAIM",
                       "claim_type": "personal observation",
                       "verification_status": "unverified",
                       "origins": ["personal_observation"],
                       "primary_origin": "personal_observation"})
    out = validators.validate_graph({"claims": claims, "edges": []}, sents,
                                    compute_signals=True, text=" ".join(s["sentence"] for s in sents),
                                    embedder=None)
    kept = out.get("claims") or []
    real = [c for c in kept if c.get("node_type") == "CLAIM"]
    assert len(real) == 4, f"a real claim was evicted: {json.dumps([c.get('id') for c in kept])}"
    # any dropped nodes under the cap must have been QUESTION nodes only
    qs = [c for c in kept if c.get("node_type") == "QUESTION"]
    assert all(q.get("references") for q in qs)


# ── N4: verified-only specificity credit (M4 rec (f)1 gaming fix) ─────────────
_RICH = "The 2019 Stanford trial cut cost by 35% in Boston."  # 4 specifics


def test_interrogatability_verified_only_credits_only_verified():
    """specificity_verified_only=True: a VERIFIED claim's specifics credit; the
    SAME specifics on an UNVERIFIED claim earn ZERO credit (the gaming fix)."""
    vsig, vq = signals.compute_interrogatability(
        [_claim("c_001", _RICH, status="verified")], [],
        specificity_verified_only=True)
    usig, uq = signals.compute_interrogatability(
        [_claim("c_001", _RICH, status="unverified")], [],
        specificity_verified_only=True)
    # Credit DIFFERS by verification: verified > 0, unverified == 0.
    assert vsig["components"]["specificity_presence"] > 0.0
    assert usig["components"]["specificity_presence"] == 0.0
    # Audit surface unchanged: total_specifics identical; unverified still probes.
    assert vsig["coverage"]["total_specifics"] == usig["coverage"]["total_specifics"] > 0
    assert vq == [], "verified specifics must emit NO questions"
    assert uq, "unverified specifics still emit teacher-probe questions"


def test_interrogatability_verified_only_off_credits_all_specifics():
    """Default (OFF): credit counts ALL specifics regardless of status — the
    Phase-1/M3 numerator, byte-identical. No mode marker leaks into the signal."""
    sig, _ = signals.compute_interrogatability(
        [_claim("c_001", _RICH, status="unverified")], [])
    assert sig["components"]["specificity_presence"] == 1.0  # 4 specifics / (1*2)
    assert "specificity_verified_only" not in sig


def test_verified_only_mode_recorded_only_when_on():
    on, _ = signals.compute_interrogatability(
        [_claim("c_001", _RICH, status="verified")], [],
        specificity_verified_only=True)
    off, _ = signals.compute_interrogatability(
        [_claim("c_001", _RICH, status="verified")], [])
    assert on.get("specificity_verified_only") is True
    assert "specificity_verified_only" not in off
    assert any("VERIFIED-ONLY" in lim for lim in on["limitations"])


# Frozen pre-N4 snapshot (captured from the pre-refactor validate_graph with
# compute_signals=True, embedder=None). The validators _validate_core/
# finalize_container split + the specificity_verified_only param default (False)
# must reproduce this BYTE-FOR-BYTE with entailment OFF — the critical parity
# contract Fable flagged (if the verified-only formula ever ran unconditionally,
# nothing would be verified and every specificity_presence would collapse to 0).
_PRE_N4_BASELINE = '{"claims": [{"assessment_confidence": "low", "claim_type": "factual", "evidence_level": 0, "id": "c_001", "limitations": [], "node_type": "CLAIM", "origins": ["external_source"], "primary_origin": "external_source", "references": null, "source": {"char_end": 50, "char_start": 0, "paragraph_id": "p001", "sentence_id": "s001"}, "text": "The 2019 Stanford trial cut cost by 35% in Boston.", "verification_paths": [], "verification_status": "unverified"}, {"assessment_confidence": "low", "claim_type": "factual", "evidence_level": 0, "id": "c_002", "limitations": [], "node_type": "CLAIM", "origins": [], "primary_origin": null, "references": null, "source": {"char_end": 110, "char_start": 53, "paragraph_id": "p001", "sentence_id": "s002"}, "text": "This is important and things generally matter a lot here.", "verification_paths": [], "verification_status": "unverified"}, {"assessment_confidence": "low", "claim_type": null, "evidence_level": 0, "id": "q_001", "limitations": [], "node_type": "QUESTION", "origins": [], "primary_origin": null, "references": "c_001", "source": null, "text": "What is the source or basis for \\"2019\\"?", "verification_paths": [], "verification_status": "unverified"}, {"assessment_confidence": "low", "claim_type": null, "evidence_level": 0, "id": "q_002", "limitations": [], "node_type": "QUESTION", "origins": [], "primary_origin": null, "references": "c_001", "source": null, "text": "What is the source or basis for \\"Stanford\\"?", "verification_paths": [], "verification_status": "unverified"}, {"assessment_confidence": "low", "claim_type": null, "evidence_level": 0, "id": "q_003", "limitations": [], "node_type": "QUESTION", "origins": [], "primary_origin": null, "references": "c_001", "source": null, "text": "What is the source or basis for \\"35%\\"?", "verification_paths": [], "verification_status": "unverified"}, {"assessment_confidence": "low", "claim_type": null, "evidence_level": 0, "id": "q_004", "limitations": [], "node_type": "QUESTION", "origins": [], "primary_origin": null, "references": "c_001", "source": null, "text": "What is the source or basis for \\"Boston\\"?", "verification_paths": [], "verification_status": "unverified"}], "edges": [], "evidence": [], "kill_switch": "DRAFTPROOF_CLAIM_GRAPH", "schema_version": "cg-1", "signals": [{"band": "moderate", "calibration_version": null, "components": {"causal_depth": 0.0, "contextual_anchoring": 0.5, "evidence_traceability": 0.0, "specificity_presence": 1.0}, "coverage": {"claims_considered": 2, "total_specifics": 4}, "fairness_gate_passed": null, "limitations": ["high interrogatability of an UNVERIFIED specific is an audit surface (questions), never an ownership credit (\\u00a7A)", "specific detection is a provisional heuristic, not NER"], "provisional_thresholds": {"high": 0.66, "low": 0.33, "note": "uncalibrated; Phase-4 \\u00a7B"}, "reconciles_with": ["citation", "grounding_diagnosis"], "scoring_enabled": false, "signal": "interrogatability", "status": "experimental", "value": {"composite": 0.375, "questions_emitted": 4, "unverified_specific_rate": 1.0}}, {"band": null, "calibration_version": null, "coverage": {"claims_considered": 2, "claims_embedded": 0}, "fairness_gate_passed": null, "limitations": ["embedding model unavailable"], "provisional_thresholds": {"high": 0.9, "low": 0.8, "note": "uncalibrated; Phase-4 \\u00a7B"}, "reconciles_with": ["generic_assertion_risk"], "scoring_enabled": false, "signal": "substitutability", "status": "experimental", "value": null}, {"calibration_version": null, "coverage": {"claims_considered": 2, "total_labels": 1}, "fairness_gate_passed": null, "limitations": ["origin is descriptive audit metadata, never a standalone credit; personal_observation is NEUTRAL (\\u00a7I)"], "provisional_thresholds": {"note": "uncalibrated; Phase-4 \\u00a7B", "unsupported_heavy": 0.5}, "reconciles_with": ["authorship_evidence", "critical_thinking"], "scoring_enabled": false, "signal": "origin_map", "status": "experimental", "value": {"distribution": {"external_source": 1.0}, "primary_origin": "external_source", "unsupported_assertion_heavy": false, "unsupported_assertion_rate": 0.5}}], "status": "experimental", "truncated": false, "truncated_dropped_ids": []}'


def _parity_props_segs():
    segs = [
        {"sentence_id": "s001", "paragraph_id": "p001", "start_char": 0, "end_char": 52,
         "sentence": "The 2019 Stanford trial cut cost by 35% in Boston."},
        {"sentence_id": "s002", "paragraph_id": "p001", "start_char": 53, "end_char": 110,
         "sentence": "This is important and things generally matter a lot here."},
    ]
    props = {"claims": [
        {"sentence_id": "s001", "quote": "The 2019 Stanford trial cut cost by 35% in Boston.",
         "node_type": "CLAIM", "claim_type": "factual",
         "origins": ["external_source"], "primary_origin": "external_source"},
        {"sentence_id": "s002", "quote": "This is important and things generally matter a lot here.",
         "node_type": "CLAIM", "claim_type": "factual", "origins": [], "primary_origin": None},
    ], "edges": []}
    return props, segs


def test_validate_graph_off_parity_byte_identical():
    """CRITICAL PARITY: the validators refactor + default param reproduce the
    frozen pre-N4 container byte-for-byte (entailment OFF path)."""
    import json
    from claim_graph import validators
    props, segs = _parity_props_segs()
    out = validators.validate_graph(props, segs, compute_signals=True,
                                    text="doc", embedder=None)
    assert json.dumps(out, sort_keys=True) == _PRE_N4_BASELINE


def test_finalize_container_matches_validate_graph():
    """The _validate_core + finalize_container split equals the one-shot
    validate_graph (behaviour-preserving refactor)."""
    import json
    from claim_graph import validators
    props, segs = _parity_props_segs()
    one = validators.validate_graph(props, segs, compute_signals=True,
                                    text="doc", embedder=None)
    claims, edges = validators._validate_core(props, segs)
    split = validators.finalize_container(claims, edges, compute_signals=True,
                                          text="doc", embedder=None)
    assert json.dumps(split, sort_keys=True) == json.dumps(one, sort_keys=True)
