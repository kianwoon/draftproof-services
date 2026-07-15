"""M2 tests: paragraph-batched extraction + cross-batch reconciliation.

The LLM gateway is MOCKED — no network call ever happens in CI. The validators
(M1) remain the single authority on what enters the graph; these tests assert
the extraction/reconcile plumbing feeds them correctly and that stats/caching/
fail-open behave per plan §3/§5.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from claim_graph import extract, reconcile
from claim_graph.build import build_claim_graph_extracted, maybe_build_claim_graph


# ── Mock gateway ─────────────────────────────────────────────────────────────
class FakeGateway:
    """Serves a queue of canned responses. Each item is a str (content), an
    Exception instance (raised), or None (empty content)."""

    def __init__(self, responses, model="fake/gpt-oss-120b"):
        self.responses = list(responses)
        self.model = model
        self.calls = []

    def chat(self, prompt, system=None, response_format=None, **kwargs):
        self.calls.append(prompt)
        item = self.responses.pop(0) if self.responses else "{}"
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=item)


def _segments():
    s0 = "The intervention reduced processing time by 35%."
    s1 = "It also improved satisfaction across every cohort."
    return [
        {"sentence_id": "s001", "paragraph_id": "p001", "start_char": 0,
         "end_char": len(s0), "sentence": s0},
        {"sentence_id": "s002", "paragraph_id": "p002", "start_char": 100,
         "end_char": 100 + len(s1), "sentence": s1},
    ]


def _claim_json(sid, quote, node_type="CLAIM", **kw):
    d = {"sentence_id": sid, "quote": quote, "node_type": node_type,
         "claim_type": "factual", "origins": ["external_source"],
         "primary_origin": "external_source"}
    d.update(kw)
    return d


@pytest.fixture(autouse=True)
def _clear_cache():
    extract.clear_cache()
    yield
    extract.clear_cache()


# ── Valid extraction ────────────────────────────────────────────────────────
def test_valid_json_extraction_accepts_and_maps_offsets(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")  # one batch
    body = json.dumps({"claims": [_claim_json("s001", "reduced processing time by 35%")],
                       "edges": []})
    gw = FakeGateway([body])
    graph = build_claim_graph_extracted("doc", _segments(), gw)
    assert len(graph["claims"]) == 1
    src = graph["claims"][0]["source"]
    assert src["char_start"] == len("The intervention ")
    assert src["sentence_id"] == "s001"
    assert graph["extraction_stats"]["proposed"] == 1
    assert graph["extraction_stats"]["accepted"] == 1
    assert graph["lifecycle"]["status"] == "extracted"


# ── Malformed JSON → Stage-3 retry path ─────────────────────────────────────
def test_malformed_json_triggers_retry(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")
    good = json.dumps({"claims": [_claim_json("s001", "reduced processing time by 35%")]})
    gw = FakeGateway(["}{ not json at all", good])  # first parse fails, retry OK
    graph = build_claim_graph_extracted("doc", _segments(), gw)
    assert len(graph["claims"]) == 1
    assert graph["extraction_stats"]["parse_failures"] == 1
    assert graph["extraction_stats"]["retries"] == 1
    assert len(gw.calls) == 2


def test_persistent_malformed_json_fails_open_empty(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_MAX_RETRIES", "2")
    gw = FakeGateway(["garbage", "still garbage", "nope"])
    graph = build_claim_graph_extracted("doc", _segments(), gw)
    assert graph["claims"] == []  # nothing parsed, but a valid container
    assert graph["schema_version"] == "cg-1"
    assert graph["extraction_stats"]["parse_failures"] == 1


# ── Hallucinated span → rejected by validators (R1) ─────────────────────────
def test_hallucinated_span_rejected_with_stats(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")
    body = json.dumps({"claims": [
        _claim_json("s001", "a fabricated survey of one thousand firms"),
    ]})
    gw = FakeGateway([body])
    graph = build_claim_graph_extracted("doc", _segments(), gw)
    assert graph["claims"] == []
    assert graph["extraction_stats"]["rejected_by_rule"]["span_not_found"] == 1
    assert graph["extraction_stats"]["accepted"] == 0


# ── INFERENCE / QUESTION node routing ───────────────────────────────────────
def test_inference_and_question_nodes_carry_null_source(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")
    body = json.dumps({"claims": [
        {"node_type": "INFERENCE", "text": "the pilot likely generalises"},
        {"node_type": "QUESTION", "text": "which survey supports the 35% figure?"},
    ]})
    gw = FakeGateway([body])
    graph = build_claim_graph_extracted("doc", _segments(), gw)
    types = sorted(c["node_type"] for c in graph["claims"])
    assert types == ["INFERENCE", "QUESTION"]
    assert all(c["source"] is None for c in graph["claims"])


# ── Batch boundaries ────────────────────────────────────────────────────────
def test_paragraph_batches_group_by_paragraph(monkeypatch):
    segs = []
    for i in range(1, 6):  # 5 paragraphs, 1 sentence each
        t = f"Sentence {i} makes a distinct point."
        segs.append({"sentence_id": f"s{i:03d}", "paragraph_id": f"p{i:03d}",
                     "start_char": i * 100, "end_char": i * 100 + len(t), "sentence": t})
    assert len(extract.paragraph_batches(segs, paras_per_batch=4)) == 2  # 4 + 1
    assert len(extract.paragraph_batches(segs, paras_per_batch=2)) == 3  # 2 + 2 + 1
    # sentences of one paragraph never split across batches
    multi = [
        {"sentence_id": "s001", "paragraph_id": "p001", "start_char": 0, "end_char": 5, "sentence": "a b."},
        {"sentence_id": "s002", "paragraph_id": "p001", "start_char": 6, "end_char": 11, "sentence": "c d."},
    ]
    batches = extract.paragraph_batches(multi, paras_per_batch=1)
    assert len(batches) == 1 and len(batches[0]) == 2


# ── Reconciler: cross-batch edge discovery ──────────────────────────────────
def test_reconciler_discovers_cross_batch_edge(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "1")  # 2 batches
    b1 = json.dumps({"claims": [_claim_json("s001", "reduced processing time by 35%")]})
    b2 = json.dumps({"claims": [_claim_json("s002", "improved satisfaction across every cohort")]})
    rec = json.dumps({"edges": [{"type": "supports", "src": "c_001", "dst": "c_002"}]})
    gw = FakeGateway([b1, b2, rec])
    graph = build_claim_graph_extracted("doc", _segments(), gw)
    assert len(graph["claims"]) == 2
    assert [e["type"] for e in graph["edges"]] == ["supports"]
    assert graph["extraction_stats"]["reconcile_calls"] == 1
    assert graph["extraction_stats"]["cross_batch_edges_proposed"] == 1


def test_reconciler_illegal_edge_rejected(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "1")
    b1 = json.dumps({"claims": [_claim_json("s001", "reduced processing time by 35%")]})
    b2 = json.dumps({"claims": [_claim_json("s002", "improved satisfaction across every cohort")]})
    rec = json.dumps({"edges": [{"type": "totally_made_up", "src": "c_001", "dst": "c_002"}]})
    gw = FakeGateway([b1, b2, rec])
    graph = build_claim_graph_extracted("doc", _segments(), gw)
    assert graph["edges"] == []


def test_reconciler_skipped_when_fewer_than_two_claims():
    accepted = [{"id": "c_001", "node_type": "CLAIM", "text": "x", "source": {"paragraph_id": "p001"}}]
    edges, stats = reconcile.reconcile(accepted, FakeGateway([]))
    assert edges == [] and stats["reconcile_calls"] == 0


# ── Cache hit ────────────────────────────────────────────────────────────────
def test_extraction_cache_hit_avoids_second_call(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")
    body = json.dumps({"claims": [_claim_json("s001", "reduced processing time by 35%")]})
    gw = FakeGateway([body])
    p1, s1 = extract.run_extraction("same-text", _segments(), gw)
    n = len(gw.calls)
    p2, s2 = extract.run_extraction("same-text", _segments(), gw)  # cache hit
    assert len(gw.calls) == n  # no new gateway call
    assert p1 == p2 and s1 == s2


# ── Fail-open ────────────────────────────────────────────────────────────────
def test_gateway_raising_yields_empty_valid_container(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_MAX_RETRIES", "1")
    gw = FakeGateway([RuntimeError("boom"), RuntimeError("boom again")])
    graph = build_claim_graph_extracted("doc", _segments(), gw)
    assert graph["claims"] == [] and graph["schema_version"] == "cg-1"


def test_extraction_exception_records_lifecycle_note(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("extractor exploded")

    monkeypatch.setattr(extract, "run_extraction", _boom)
    graph = build_claim_graph_extracted("doc", _segments(), FakeGateway([]))
    assert graph["claims"] == []
    assert graph["lifecycle"]["status"] == "extraction_failed"
    assert "exploded" in graph["lifecycle"]["error"]


# ── maybe_build_claim_graph end-to-end with injected gateway ─────────────────
def test_maybe_build_uses_injected_gateway_when_enabled(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH", "1")
    monkeypatch.setenv("DRAFTPROOF_CLAIM_GRAPH_PARAS_PER_BATCH", "5")
    body = json.dumps({"claims": [_claim_json("s001", "reduced processing time by 35%")]})
    gw = FakeGateway([body])
    graph = maybe_build_claim_graph("doc", _segments(), gateway=gw)
    assert len(graph["claims"]) == 1
    assert graph["lifecycle"]["status"] == "extracted"


def test_maybe_build_disabled_ignores_gateway(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_CLAIM_GRAPH", raising=False)
    gw = FakeGateway([json.dumps({"claims": []})])
    assert maybe_build_claim_graph("doc", _segments(), gateway=gw) == {}
    assert gw.calls == []  # never touched
