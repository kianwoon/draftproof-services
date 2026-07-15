"""M1 tests for the Phase-1 claim-graph plumbing.

Governing specs: docs/plans/phase1_claim_graph_execution_plan.md (§3 validator
rules, §7 test plan); docs/plans/credible_authorship_assessment_v2.md (invariants).
Everything here is deterministic — no LLM is ever called (extraction is M2).
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from claim_graph import claim_graph_enabled, KILL_SWITCH_ENV
from claim_graph import schema, validators
from claim_graph.build import build_claim_graph, maybe_build_claim_graph


# ── Fixtures ────────────────────────────────────────────────────────────────
def _segments():
    """Two sentences with explicit spans/text (mirrors structured_sentence_segments)."""
    s0 = "The intervention reduced processing time by 35%."
    s1 = "It also improved satisfaction across every cohort."
    # start_char offsets are arbitrary-but-consistent absolute doc offsets.
    return [
        {"sentence_id": "s001", "paragraph_id": "p001", "start_char": 100,
         "end_char": 100 + len(s0), "sentence": s0},
        {"sentence_id": "s002", "paragraph_id": "p001", "start_char": 200,
         "end_char": 200 + len(s1), "sentence": s1},
    ]


def _claim(sid, quote, **kw):
    base = {"sentence_id": sid, "quote": quote, "node_type": "CLAIM",
            "claim_type": "factual"}
    base.update(kw)
    return base


# ── Kill-switch (default OFF for Phase 1) ───────────────────────────────────
def test_kill_switch_default_off(monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    assert claim_graph_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_kill_switch_on_values(monkeypatch, val):
    monkeypatch.setenv(KILL_SWITCH_ENV, val)
    assert claim_graph_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
def test_kill_switch_off_values(monkeypatch, val):
    monkeypatch.setenv(KILL_SWITCH_ENV, val)
    assert claim_graph_enabled() is False


def test_maybe_build_off_returns_empty_dict(monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    assert maybe_build_claim_graph("x", _segments()) == {}


def test_maybe_build_on_returns_valid_container(monkeypatch):
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    graph = maybe_build_claim_graph("x", _segments())
    assert graph["schema_version"] == schema.SCHEMA_VERSION
    assert graph["status"] == "experimental"
    assert graph["kill_switch"] == KILL_SWITCH_ENV
    assert graph["claims"] == [] and graph["edges"] == []


# ── Empty-graph-capable build ───────────────────────────────────────────────
def test_build_empty_graph_is_valid():
    graph = build_claim_graph("text", _segments(), proposals=None)
    assert graph == schema.empty_graph()


# ── Rule 1: span-match rejection (headline invariant, R1) ───────────────────
def test_valid_quote_accepted_and_offsets_mapped():
    graph = build_claim_graph(
        "t", _segments(),
        {"claims": [_claim("s001", "reduced processing time by 35%")]},
    )
    assert len(graph["claims"]) == 1
    src = graph["claims"][0]["source"]
    # char offsets are absolute (segment.start_char + local match offset).
    assert src["char_start"] == 100 + len("The intervention ")
    assert src["sentence_id"] == "s001"
    assert src["paragraph_id"] == "p001"


def test_hallucinated_quote_rejected():
    graph = build_claim_graph(
        "t", _segments(),
        {"claims": [_claim("s001", "a recent industry survey reports one-third")]},
    )
    assert graph["claims"] == []  # not in source → dropped


def test_claim_with_unknown_sentence_id_rejected():
    graph = build_claim_graph(
        "t", _segments(),
        {"claims": [_claim("s999", "reduced processing time by 35%")]},
    )
    assert graph["claims"] == []


# ── Rule 6: node-type purity ────────────────────────────────────────────────
def test_unknown_node_type_rejected():
    graph = build_claim_graph(
        "t", _segments(),
        {"claims": [_claim("s001", "reduced processing time by 35%", node_type="ASSERTION")]},
    )
    assert graph["claims"] == []


def test_inference_node_skips_span_check_and_has_null_source():
    graph = build_claim_graph(
        "t", _segments(),
        {"claims": [{"node_type": "INFERENCE", "text": "system interpretation"}]},
    )
    assert len(graph["claims"]) == 1
    assert graph["claims"][0]["source"] is None


# ── Rule 3: verification-status clamp (verified/corroborated unreachable) ────
@pytest.mark.parametrize("status", ["verified", "corroborated"])
def test_verified_marked_claim_rejected(status):
    graph = build_claim_graph(
        "t", _segments(),
        {"claims": [_claim("s001", "reduced processing time by 35%", verification_status=status)]},
    )
    assert graph["claims"] == []


def test_unrecognised_status_defaults_unverified():
    graph = build_claim_graph(
        "t", _segments(),
        {"claims": [_claim("s001", "reduced processing time by 35%", verification_status="bogus")]},
    )
    assert graph["claims"][0]["verification_status"] == "unverified"


# ── Rule 2: dedup by span overlap ───────────────────────────────────────────
def test_overlapping_spans_deduped():
    graph = build_claim_graph(
        "t", _segments(),
        {"claims": [
            _claim("s001", "reduced processing time by 35%"),
            _claim("s001", "reduced processing time by 35%"),
        ]},
    )
    assert len(graph["claims"]) == 1


# ── Rule 4: edge legality + suppression ─────────────────────────────────────
def _two_claim_proposals():
    return {"claims": [
        _claim("s001", "reduced processing time by 35%"),
        _claim("s002", "improved satisfaction across every cohort"),
    ]}


def test_illegal_edge_type_rejected():
    p = _two_claim_proposals()
    p["edges"] = [{"type": "totally_made_up",
                   "src_quote": "reduced processing time by 35%",
                   "dst_quote": "improved satisfaction across every cohort"}]
    graph = build_claim_graph("t", _segments(), p)
    assert graph["edges"] == []


def test_dangling_edge_endpoint_rejected():
    p = _two_claim_proposals()
    p["edges"] = [{"type": "supports",
                   "src_quote": "reduced processing time by 35%",
                   "dst_quote": "nonexistent claim text"}]
    graph = build_claim_graph("t", _segments(), p)
    assert graph["edges"] == []


def test_legal_edge_resolved_to_node_ids():
    p = _two_claim_proposals()
    p["edges"] = [{"type": "supports",
                   "src_quote": "reduced processing time by 35%",
                   "dst_quote": "improved satisfaction across every cohort"}]
    graph = build_claim_graph("t", _segments(), p)
    assert len(graph["edges"]) == 1
    e = graph["edges"][0]
    assert e["type"] == "supports"
    assert e["src"].startswith("c_") and e["dst"].startswith("c_")


def test_revises_suppresses_inconsistent_with():
    p = _two_claim_proposals()
    q0, q1 = "reduced processing time by 35%", "improved satisfaction across every cohort"
    p["edges"] = [
        {"type": "inconsistent_with", "src_quote": q0, "dst_quote": q1},
        {"type": "revises", "src_quote": q1, "dst_quote": q0},
    ]
    graph = build_claim_graph("t", _segments(), p)
    types = {e["type"] for e in graph["edges"]}
    assert "inconsistent_with" not in types  # suppressed by revises on same pair
    assert "revises" in types


def test_inconsistent_with_marks_both_endpoints_contradicted():
    p = _two_claim_proposals()
    q0, q1 = "reduced processing time by 35%", "improved satisfaction across every cohort"
    p["edges"] = [{"type": "inconsistent_with", "src_quote": q0, "dst_quote": q1}]
    graph = build_claim_graph("t", _segments(), p)
    statuses = {c["verification_status"] for c in graph["claims"]}
    assert statuses == {"contradicted"}


# ── Rule 5: cycle detection on depends_on ───────────────────────────────────
def test_depends_on_cycle_broken():
    # three claims across three sentences; A->B->C->A must drop the closing edge.
    segs = _segments() + [
        {"sentence_id": "s003", "paragraph_id": "p001", "start_char": 300,
         "end_char": 340, "sentence": "Therefore the pilot informed the rollout."}
    ]
    qa, qb, qc = ("reduced processing time by 35%",
                  "improved satisfaction across every cohort",
                  "the pilot informed the rollout")
    p = {"claims": [_claim("s001", qa), _claim("s002", qb), _claim("s003", qc)],
         "edges": [
             {"type": "depends_on", "src_quote": qa, "dst_quote": qb},
             {"type": "depends_on", "src_quote": qb, "dst_quote": qc},
             {"type": "depends_on", "src_quote": qc, "dst_quote": qa},
         ]}
    graph = build_claim_graph("t", segs, p)
    depends = [e for e in graph["edges"] if e["type"] == "depends_on"]
    assert len(depends) == 2  # closing edge dropped, DAG preserved


# ── Caps + eviction determinism ─────────────────────────────────────────────
def _many_claim_segments(n):
    segs = []
    for i in range(1, n + 1):
        text = f"Claim number {i} asserts a distinct measurable outcome here."
        segs.append({"sentence_id": f"s{i:03d}", "paragraph_id": f"p{i:03d}",
                     "start_char": i * 1000, "end_char": i * 1000 + len(text),
                     "sentence": text})
    return segs


def test_caps_eviction_deterministic_and_flags_truncation():
    n = 10
    segs = _many_claim_segments(n)
    proposals = {"claims": [_claim(f"s{i:03d}", f"Claim number {i} asserts")
                            for i in range(1, n + 1)]}
    g1 = validators.validate_graph(proposals, segs, cap_claims=4, cap_edges=300)
    g2 = validators.validate_graph(proposals, segs, cap_claims=4, cap_edges=300)
    assert len(g1["claims"]) == 4
    assert g1["truncated"] is True
    assert len(g1["truncated_dropped_ids"]) == n - 4
    # Determinism: identical inputs → identical kept ids + dropped ids.
    assert [c["id"] for c in g1["claims"]] == [c["id"] for c in g2["claims"]]
    assert g1["truncated_dropped_ids"] == g2["truncated_dropped_ids"]


def test_no_truncation_when_under_cap():
    graph = build_claim_graph("t", _segments(), _two_claim_proposals())
    assert graph["truncated"] is False
    assert graph["truncated_dropped_ids"] == []


# ── Fresh-interpreter import test (mirror test_authorship_evidence_levels) ───
def test_no_rewrite_v6_import_in_fresh_interpreter():
    import pathlib
    poc = pathlib.Path(__file__).resolve().parent.parent
    code = (
        "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');\n"
        "from claim_graph.build import build_claim_graph\n"
        "from claim_graph import schema, validators\n"
        "assert not any('rewrite_v6' in k for k in sys.modules), "
        "sorted(k for k in sys.modules if 'rewrite_v6' in k)\n"
        "assert build_claim_graph('t', []) == schema.empty_graph()\n"
        "print('IMPORT_OK')" % (str(poc), str(poc.parent))
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "IMPORT_OK" in proc.stdout, f"stderr tail: {proc.stderr[-800:]}"
