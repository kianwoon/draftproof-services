"""Deterministic tests for M4 eval UTILITIES (the eval run itself is a
measurement, not a test — plan §7). No LLM, no network here."""
from __future__ import annotations

import os

from claim_graph.eval import analyze, build_corpus_manifest as bcm, interrog_ab


# ── manifest builder determinism ─────────────────────────────────────────────
_FAKE_PERSUADE = [
    {"essay_id": "E1", "words": 150, "sha256": "a" * 64, "text": "x"},
    {"essay_id": "E2", "words": 200, "sha256": "b" * 64, "text": "y"},
]


def test_build_manifest_deterministic():
    m1 = bcm.build_manifest(_FAKE_PERSUADE)
    m2 = bcm.build_manifest(_FAKE_PERSUADE)
    assert m1 == m2  # byte-identical structure given identical corpus
    assert m1["manifest_version"] == bcm.MANIFEST_VERSION
    assert m1["counts"]["H"] == 2
    h_docs = [d for d in m1["docs"] if d["group"] == "H"]
    assert [d["doc_id"] for d in h_docs] == ["H_00", "H_01"]
    assert h_docs[0]["text_ref"] == {"kind": "persuade_cache", "key": "E1"}


def test_group_a_selection_stable_and_bounded():
    a = bcm.select_group_a()
    assert 0 < len(a) <= bcm.A_COUNT
    paths = [r["path"] for r in a]
    assert paths == sorted(paths)  # deterministic order
    for r in a:
        assert r["path"].endswith(".json") and len(r["sha256"]) == 64
        assert r["words"] > 0


def test_resolve_text_case_json_reads_committed_case():
    a = bcm.select_group_a(n=1)[0]
    doc = {"text_ref": {"kind": "case_json", "path": a["path"]}}
    text = bcm.resolve_text(doc, persuade_cache={})
    assert isinstance(text, str) and len(text.split()) == a["words"]


# ── offline A/B recompute ────────────────────────────────────────────────────
def _claim(cid, text, status, origins=None):
    return {"id": cid, "node_type": "CLAIM", "text": text,
            "verification_status": status, "origins": origins or [],
            "primary_origin": (origins or [None])[0], "references": None}


def test_interrog_recompute_verified_excludes_unverified_specifics():
    claims = [
        _claim("c_001", "In 2019 the rate rose 42%.", "unverified"),
        _claim("c_002", "In 2020 the rate rose 15%.", "internally_supported"),
    ]
    cur = interrog_ab.recompute(claims, [], variant="current")
    ver = interrog_ab.recompute(claims, [], variant="verified")
    # current counts specifics on BOTH claims; verified only on the supported one.
    assert cur["components"]["specificity_presence"] > ver["components"]["specificity_presence"]
    assert cur["band"] in ("low", "moderate", "high")
    assert ver["composite"] <= cur["composite"]


def test_interrog_recompute_empty_graph():
    r = interrog_ab.recompute([], [], variant="current")
    assert r["composite"] == 0.0 and r["band"] == "low"


def test_interrog_recompute_matches_signals_current_formula():
    # The 'current' variant must reproduce signals.compute_interrogatability's
    # composite on the same graph (unit-parity, §B3).
    from claim_graph import schema, signals
    nodes = [
        schema.ClaimNode(id="c_001", node_type="CLAIM", text="In 2019 it rose 42%.",
                         claim_type="factual", source={"paragraph_id": "p001",
                         "sentence_id": "s001", "char_start": 0, "char_end": 20},
                         verification_status="unverified", origins=["external_source"],
                         primary_origin="external_source"),
        schema.ClaimNode(id="c_002", node_type="CLAIM", text="It was consistent.",
                         claim_type="analytical", source={"paragraph_id": "p001",
                         "sentence_id": "s002", "char_start": 0, "char_end": 18},
                         verification_status="internally_supported", origins=[],
                         primary_origin=None),
    ]
    sig, _q = signals.compute_interrogatability(nodes, [])
    dicts = [n.to_dict() for n in nodes]
    recomputed = interrog_ab.recompute(dicts, [], variant="current")
    assert recomputed["composite"] == sig["value"]["composite"]
    assert recomputed["components"] == sig["components"]


# ── analysis schema ──────────────────────────────────────────────────────────
def _row(group, comp, band, ver_comp, ver_band, subst, gen):
    return {
        "doc_id": group + "_00", "group": group, "words": 200,
        "interrogatability": {"composite": comp, "band": band,
                              "components": {}, "unverified_specific_rate": 0.5,
                              "questions_emitted": 3},
        "interrog_verified": {"composite": ver_comp, "band": ver_band, "components": {}},
        "substitutability": {"value": subst, "band": "high" if subst and subst > 0.9 else "low"},
        "generic_assertion_risk": gen,
        "origin_map": {"primary_origin": "external_source", "unsupported_assertion_rate": 0.3},
        "graph": {"claims_real": 5}, "llm_calls": 3, "error": None,
    }


def test_analyze_produces_headline_keys():
    rows = [
        _row("H", 0.6, "moderate", 0.55, "moderate", 0.75, 0.30),
        _row("A", 0.2, "low", 0.20, "low", 0.92, 0.85),
        _row("G", 0.8, "high", 0.30, "low", 0.70, 0.40),
    ]
    out = analyze.analyze(rows)
    for key in ("interrogatability_current", "interrogatability_verified",
                "false_thin_rate_H", "gaming_high_rate_G", "substitutability",
                "substitutability_vs_generic_pearson", "origin_map", "extraction"):
        assert key in out
    # verified variant knocks the gaming doc out of 'high'
    assert out["gaming_high_rate_G"]["current"] == 1.0
    assert out["gaming_high_rate_G"]["verified"] == 0.0
