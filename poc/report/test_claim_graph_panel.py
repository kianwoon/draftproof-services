"""Tests for the claim-graph "Source grounding" display composer.

The composer re-presents the EXPERIMENTAL cg-1 claim-graph container
(``authorship_evidence.claim_graph``) as an evidentiary, advisory panel. It is
display-only: ``scoring`` is hard-False and it NEVER touches the tier or score.
Absent/empty container -> None (nothing renders; prod byte-identical).
"""
from __future__ import annotations

from report.claim_graph_panel import compose_claim_graph_display


# ── Realistic cg-1 fixture (shapes cribbed from claim_graph/schema.py,
#    sources.py::_record, entailment.py verdicts, signals.py interrogatability). ──
def _fixture_container() -> dict:
    return {
        "schema_version": "cg-1",
        "status": "experimental",
        "truncated": False,
        "claims": [
            {"id": "c_001", "node_type": "CLAIM",
             "text": "Global temperatures rose 1.1 degrees Celsius since 1880.",
             "verification_status": "verified", "references": None},
            {"id": "c_002", "node_type": "CLAIM",
             "text": "The policy reduced emissions by 40 percent in two years.",
             "verification_status": "contradicted", "references": None},
            {"id": "c_003", "node_type": "CLAIM",
             "text": "A 2019 survey covered 3000 respondents across five nations.",
             "verification_status": "unverified", "references": None},
            {"id": "c_004", "node_type": "CLAIM",
             "text": "The 2021 meta-analysis pooled twelve controlled trials.",
             "verification_status": "unverified", "references": None},
            {"id": "q_001", "node_type": "QUESTION",
             "text": "What is the source for the 40 percent emissions reduction?",
             "verification_status": "unverified", "references": "c_002"},
            {"id": "q_002", "node_type": "QUESTION",
             "text": "Which five nations did the 2019 survey cover?",
             "verification_status": "unverified", "references": "c_003"},
        ],
        "edges": [],
        "evidence": [
            {"id": "e_001", "kind": "external_citation", "claim_ids": ["c_001"],
             "detail": {"marker": "(NASA, 2020)", "locator_type": "doi",
                        "resolution": {"status": "resolved", "locator_type": "doi",
                                       "locator": "10.1029/2020GL088469",
                                       "source_text": "warming of 1.1C ...",
                                       "entailment": {"c_001": {"status": "verified",
                                                                "entailment_score": 0.9862,
                                                                "contradiction_score": 0.0016}}}}},
            {"id": "e_002", "kind": "external_citation", "claim_ids": ["c_002"],
             "detail": {"marker": "(Smith, 2022)", "locator_type": "doi",
                        "resolution": {"status": "resolved", "locator_type": "doi",
                                       "locator": "10.1177/0002764221993",
                                       "source_text": "no measurable reduction ...",
                                       "entailment": {"c_002": {"status": "contradicted",
                                                                 "entailment_score": 0.02,
                                                                 "contradiction_score": 0.91}}}}},
            {"id": "e_003", "kind": "external_citation", "claim_ids": ["c_003"],
             "detail": {"marker": "(Lee, 2019)", "locator_type": "doi",
                        "resolution": {"status": "paywalled", "locator_type": "doi",
                                       "locator": "10.1000/paywalled.doi.example.long.suffix"}}},
            {"id": "e_004", "kind": "external_citation", "claim_ids": ["c_004"],
             "detail": {"marker": "(Doe, 2021)", "locator_type": "url",
                        "resolution": {"status": "unresolved", "locator_type": "url",
                                       "locator": "https://example.org/meta"}}},
        ],
        "signals": [
            {"signal": "interrogatability", "band": "moderate", "status": "advisory",
             "value": {"composite": 0.5, "unverified_specific_rate": 0.5,
                       "questions_emitted": 2},
             "coverage": {"claims_considered": 4, "total_specifics": 6}},
        ],
        "source_coverage": {"type": "source_access", "value": ["crossref", "web"]},
        "source_limitations": ["outside_source_access"],
    }


# ── Absent / empty -> None (prod byte-identical). ──

def test_none_when_none():
    assert compose_claim_graph_display(None) is None


def test_none_when_not_dict():
    assert compose_claim_graph_display("nope") is None
    assert compose_claim_graph_display(123) is None


def test_none_when_empty_container():
    from claim_graph.schema import empty_graph
    assert compose_claim_graph_display(empty_graph()) is None


def test_none_when_no_claims_no_evidence():
    assert compose_claim_graph_display({"claims": [], "evidence": [], "signals": []}) is None


# ── Populated container -> pinned contract. ──

def test_present_and_advisory():
    out = compose_claim_graph_display(_fixture_container())
    assert out is not None
    assert out["present"] is True
    assert out["scoring"] is False          # ALWAYS advisory
    assert out["status"] == "advisory"      # from interrogatability signal status


def test_summary_counts():
    out = compose_claim_graph_display(_fixture_container())
    s = out["summary"]
    assert s["specific_claims"] == 6        # interrogatability total_specifics
    assert s["verified_to_source"] == 1     # one CLAIM verification_status verified
    assert s["need_grounding"] == 2         # questions_emitted


def test_source_check_statuses_mapped():
    out = compose_claim_graph_display(_fixture_container())
    checks = out["source_checks"]
    by_status = {c["status"] for c in checks}
    assert by_status == {"verified", "contradicted", "paywalled", "unresolved"}
    # Contradicted (the only red) surfaces first under severity ordering.
    assert checks[0]["status"] == "contradicted"
    verified_row = next(c for c in checks if c["status"] == "verified")
    assert verified_row["entailment_score"] == 0.9862
    assert verified_row["claim_excerpt"].startswith("Global temperatures rose")
    assert verified_row["locator"] == "doi 10.1029/2020GL088469"


def test_source_check_locator_truncated():
    out = compose_claim_graph_display(_fixture_container())
    pw = next(c for c in out["source_checks"] if c["status"] == "paywalled")
    assert len(pw["locator"]) <= 80
    assert pw["entailment_score"] is None   # no entailment on paywalled


def test_questions_capped_and_texts():
    out = compose_claim_graph_display(_fixture_container())
    assert out["questions"][0].startswith("What is the source")
    assert len(out["questions"]) <= 6


def test_source_checks_capped_at_8():
    c = _fixture_container()
    # Duplicate evidence/claims to exceed the cap.
    base_ev = list(c["evidence"])
    for i in range(12):
        c["claims"].append({"id": f"cx_{i}", "node_type": "CLAIM",
                            "text": f"Filler specific claim number {i}.",
                            "verification_status": "unverified", "references": None})
        c["evidence"].append({"id": f"ex_{i}", "kind": "external_citation",
                              "claim_ids": [f"cx_{i}"],
                              "detail": {"marker": f"(F{i})", "locator_type": "url",
                                         "resolution": {"status": "unresolved",
                                                        "locator_type": "url",
                                                        "locator": f"https://x/{i}"}}})
    out = compose_claim_graph_display(c)
    assert len(out["source_checks"]) <= 8
    # The single contradiction must survive the cap (severity ordering).
    assert any(x["status"] == "contradicted" for x in out["source_checks"])


def test_excerpt_trimmed_140():
    c = _fixture_container()
    c["claims"][0]["text"] = "X" * 400
    out = compose_claim_graph_display(c)
    v = next(x for x in out["source_checks"] if x["status"] == "verified")
    assert len(v["claim_excerpt"]) <= 141   # ~140 + ellipsis char


def test_coverage_and_limitations_passthrough():
    out = compose_claim_graph_display(_fixture_container())
    assert out["coverage_channels"] == ["crossref", "web"]
    assert out["limitations"] == ["outside_source_access"]


def test_status_falls_back_to_container_status():
    c = _fixture_container()
    c["signals"] = []  # no interrogatability status
    out = compose_claim_graph_display(c)
    assert out["status"] == "experimental"


def test_summary_fallbacks_without_signal():
    c = _fixture_container()
    c["signals"] = []
    out = compose_claim_graph_display(c)
    # specific_claims falls back to CLAIM-node count (4), need_grounding to
    # QUESTION-node count (2).
    assert out["summary"]["specific_claims"] == 4
    assert out["summary"]["need_grounding"] == 2


def test_fail_open_on_malformed():
    # Malformed evidence/claims must never raise -> either None or a safe dict.
    bad = {"claims": [{"id": "c", "node_type": "CLAIM"}],  # no text
           "evidence": [{"id": "e", "detail": None, "claim_ids": None}],
           "signals": [{"signal": "interrogatability", "value": None}]}
    out = compose_claim_graph_display(bad)
    assert out is None or isinstance(out, dict)


def test_verified_status_never_leaks_score_field():
    out = compose_claim_graph_display(_fixture_container())
    assert "score" not in out
    assert "tier" not in out
    assert out["scoring"] is False
