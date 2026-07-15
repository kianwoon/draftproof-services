"""N1 tests — deterministic citation → claim linking (no network, no LLM).

Governing plan: docs/plans/phase2_entailment_grounding_scope.md §5 (N1 row).
N1 records WHICH claim cites WHAT; it resolves nothing and mints no ``verified``
(that is N2–N4). Everything here is deterministic — identical input → identical
EvidenceNodes with stable ids.
"""
from __future__ import annotations

import pytest

from claim_graph import ENTAILMENT_KILL_SWITCH_ENV, KILL_SWITCH_ENV
from claim_graph import citations
from claim_graph.schema import EvidenceNode


# ── Fixtures ────────────────────────────────────────────────────────────────
def _claims():
    """Two CLAIM-node dicts (validator output shape) with absolute doc spans.

    Doc text below is authored so the citations land inside / after these spans.
    """
    return [
        {"id": "c_001", "node_type": "CLAIM", "source": {
            "paragraph_id": "p001", "sentence_id": "s001",
            "char_start": 0, "char_end": 55}},
        {"id": "c_002", "node_type": "CLAIM", "source": {
            "paragraph_id": "p001", "sentence_id": "s002",
            "char_start": 56, "char_end": 100}},
    ]


# (Smith, 2022)@26 inside c_001 [0,55); [1]@106, doi@118, url@138 all fall AFTER
# c_002's span end (100) → nearest-preceding fallback links them to c_002.
_DOC = (
    "The trial cut cost by 35% (Smith, 2022) as reported.\n\n"  # ends near 55
    "A later review confirmed the effect across cohorts. [1]\n"  # [1] AFTER c_002 span
    "See doi:10.1234/abc.def and https://example.org/paper."
)


# ── EvidenceNode.detail round-trip ──────────────────────────────────────────
def test_evidence_node_detail_roundtrips():
    n = EvidenceNode(id="e_001", kind="external_citation", claim_ids=["c_001"],
                     detail={"marker": "(Smith, 2022)", "locator_type": "in_text"})
    d = n.to_dict()
    assert d["id"] == "e_001"
    assert d["kind"] == "external_citation"
    assert d["claim_ids"] == ["c_001"]
    assert d["detail"] == {"marker": "(Smith, 2022)", "locator_type": "in_text"}


def test_evidence_node_detail_none_omitted():
    # Backward-compat: default None → no ``detail`` key (existing callers unaffected).
    assert EvidenceNode(id="e_1", kind="k").to_dict() == {
        "id": "e_1", "kind": "k", "claim_ids": []}


# ── Extraction + linking ────────────────────────────────────────────────────
def test_link_emits_evidence_nodes():
    ev = citations.link_citations(_DOC, _claims())
    assert ev, "expected at least one citation evidence node"
    assert all(isinstance(n, EvidenceNode) for n in ev)
    assert all(n.kind == "external_citation" for n in ev)
    # stable, ordered ids
    assert [n.id for n in ev] == [f"e_{i:03d}" for i in range(1, len(ev) + 1)]


def test_determinism_same_input_same_nodes():
    a = [n.to_dict() for n in citations.link_citations(_DOC, _claims())]
    b = [n.to_dict() for n in citations.link_citations(_DOC, _claims())]
    assert a == b


def test_marker_inside_claim_span_links_to_that_claim():
    ev = citations.link_citations(_DOC, _claims())
    smith = [n for n in ev if n.detail and "Smith" in n.detail.get("marker", "")]
    assert len(smith) == 1
    assert smith[0].claim_ids == ["c_001"]  # (Smith, 2022) is inside c_001's span


def test_marker_after_span_links_to_nearest_preceding_claim():
    ev = citations.link_citations(_DOC, _claims())
    numeric = [n for n in ev if n.detail and n.detail.get("marker") == "[1]"]
    assert len(numeric) == 1
    # [1] sits AFTER c_002's span end → nearest-preceding fallback → c_002.
    assert numeric[0].claim_ids == ["c_002"]


def test_doi_and_url_extracted():
    ev = citations.link_citations(_DOC, _claims())
    types = {n.detail.get("locator_type") for n in ev if n.detail}
    assert "doi" in types
    assert "url" in types
    doi = [n for n in ev if n.detail and n.detail.get("locator_type") == "doi"][0]
    assert doi.detail.get("doi") == "10.1234/abc.def"


def test_unlinked_citation_emitted_with_empty_claim_ids():
    # A citation BEFORE the first claim span links to nothing → visible, claim_ids=[].
    claims = [{"id": "c_001", "node_type": "CLAIM", "source": {
        "paragraph_id": "p001", "sentence_id": "s001",
        "char_start": 100, "char_end": 160}}]
    doc = "(Early, 2001) precedes every claim in this document body area......" + "x" * 100
    ev = citations.link_citations(doc, claims)
    early = [n for n in ev if n.detail and "Early" in n.detail.get("marker", "")]
    assert len(early) == 1
    assert early[0].claim_ids == []  # unlinked but still emitted


def test_no_claims_all_unlinked_but_emitted():
    ev = citations.link_citations(_DOC, [])
    assert ev  # citations still surfaced
    assert all(n.claim_ids == [] for n in ev)


def test_fail_open_on_bad_input_returns_empty():
    # Non-str text must not raise — fail-open contract.
    assert citations.link_citations(None, _claims()) == []


# ── Build-path wiring (entailment-gated) ─────────────────────────────────────
def _seg(text, sid, pid, start):
    return {"sentence_id": sid, "paragraph_id": pid, "start_char": start,
            "end_char": start + len(text), "sentence": text}


class _FakeGateway:
    """Minimal gateway: returns one batch of claim proposals then edges."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "fake/gpt-oss-120b"

    def chat(self, prompt, system=None, response_format=None, **kwargs):
        from types import SimpleNamespace
        item = self.responses.pop(0) if self.responses else "{}"
        return SimpleNamespace(content=item)


def test_build_extracted_evidence_off_by_default(monkeypatch):
    """Entailment OFF → evidence stays [] (Phase-1 byte-identical)."""
    monkeypatch.delenv(ENTAILMENT_KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    from claim_graph import extract
    from claim_graph.build import build_claim_graph_extracted
    extract.clear_cache()
    text = "The trial cut cost by 35% (Smith, 2022)."
    segs = [_seg(text, "s001", "p001", 0)]
    import json
    proposals = json.dumps({"claims": [
        {"sentence_id": "s001", "quote": "trial cut cost by 35%",
         "node_type": "CLAIM", "claim_type": "factual"}]})
    gw = _FakeGateway([proposals])
    container = build_claim_graph_extracted(text, segs, gw)
    assert container["evidence"] == []


def test_build_extracted_evidence_on_when_flag_set(monkeypatch):
    """Entailment ON → citation evidence nodes populate container['evidence']."""
    monkeypatch.setenv(ENTAILMENT_KILL_SWITCH_ENV, "1")
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    from claim_graph import extract
    from claim_graph.build import build_claim_graph_extracted
    extract.clear_cache()
    text = "The trial cut cost by 35% (Smith, 2022)."
    segs = [_seg(text, "s001", "p001", 0)]
    import json
    proposals = json.dumps({"claims": [
        {"sentence_id": "s001", "quote": "trial cut cost by 35%",
         "node_type": "CLAIM", "claim_type": "factual"}]})
    gw = _FakeGateway([proposals])
    container = build_claim_graph_extracted(text, segs, gw)
    assert container["evidence"], "expected citation evidence when entailment on"
    node = container["evidence"][0]
    assert node["kind"] == "external_citation"
    # (Smith, 2022) sits inside the accepted claim's span → linked.
    assert node["claim_ids"] and node["claim_ids"][0].startswith("c_")
    # cited claim status stays unverified in N1 (verified is N4).
    for c in container["claims"]:
        assert c["verification_status"] != "verified"
