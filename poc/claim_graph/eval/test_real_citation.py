"""Deterministic tests for the N6 real-citation slice UTILITIES (no network, no
LLM). The live Crossref build + the eval run itself are measurements, not tests
(plan §7). These pin the anti-fabrication-critical determinism: finding
extraction is verbatim-from-abstract, doc assembly is DOI-linkable, and the
cross-pairing invariant (misattributed cite_doi != claim_source_doi) holds."""
from __future__ import annotations

from claim_graph.eval import build_real_citation_cases as brc
from claim_graph.eval import analyze_n6


# ── extract_finding_sentence ─────────────────────────────────────────────────
_ABSTRACT = (
    "<jats:p><jats:title>Background:</jats:title> Physical activity matters. "
    "Results: Participants who exercised daily had a 27% lower risk of "
    "cardiovascular events than sedentary controls. We conclude it helps.</jats:p>"
)


def test_extract_finding_is_verbatim_substring_of_clean_abstract():
    finding = brc.extract_finding_sentence(_ABSTRACT)
    assert finding is not None
    cleaned = brc.clean_abstract(_ABSTRACT)
    # The claim must be a verbatim slice of the cleaned abstract (anti-fabrication):
    # the label is stripped and a terminal period may be appended, so compare core.
    core = finding.rstrip(".")
    assert core in cleaned
    assert "27%" in finding  # picked the finding sentence, not the framing


def test_extract_finding_none_when_no_finding_sentence():
    # No numbers, no result cues, all too short → nothing qualifies.
    assert brc.extract_finding_sentence("It is nice. We looked. All good.") is None
    assert brc.extract_finding_sentence("") is None


def test_strip_label_removes_leading_section_marker():
    finding = brc.extract_finding_sentence(_ABSTRACT)
    assert finding is not None
    assert not finding.lower().startswith("results")


# ── assemble_doc — DOI must be linkable by citations.link_citations ──────────
def test_assemble_doc_contains_claim_and_linkable_doi():
    from claim_graph import citations
    claim = "Daily exercise cut cardiovascular risk by 27% versus controls."
    doi = "10.1234/abc.def"
    doc = brc.assemble_doc(claim, doi)
    assert claim in doc and doi in doc
    nodes = citations.link_citations(doc, [])
    dois = [n.detail.get("doi") for n in nodes if n.detail.get("locator_type") == "doi"]
    assert doi in dois  # the DOI regex/dedup keeps the bare DOI locator


# ── build_fixture schema ─────────────────────────────────────────────────────
def test_build_fixture_entailed_and_misattributed_schema():
    ent = brc.build_fixture("RC_00", brc.CASE_ENTAILED, "A 27% drop was seen here.",
                            "10.1/x", "10.1/x", "Paper X", "Paper X")
    assert ent["cite_doi"] == ent["claim_source_doi"]  # entailed: same paper
    assert ent["synthetic_adversarial"] is False
    for k in ("case_id", "case_type", "claim", "text", "cite_doi", "claim_source_doi"):
        assert k in ent

    mis = brc.build_fixture("RC_01", brc.CASE_MISATTRIBUTED, "A 27% drop was seen here.",
                            "10.2/y", "10.1/x", "Paper X", "Paper Y")
    assert mis["cite_doi"] != mis["claim_source_doi"]  # misattributed: off-source
    assert mis["synthetic_adversarial"] is True


# ── build_slice cross-pairing invariant ──────────────────────────────────────
def _cand(topic, doi, finding):
    return {"doi": doi, "title": "T-" + doi, "abstract": finding + " More.",
            "finding": finding, "topic": topic}


def test_build_slice_pairs_entailed_same_and_misattributed_off_source():
    by_topic = {
        brc.TOPICS[0]: [_cand(brc.TOPICS[0], "10.1/a1", "Finding one rose 10%."),
                        _cand(brc.TOPICS[0], "10.1/a2", "Finding two fell 20%.")],
        brc.TOPICS[1]: [_cand(brc.TOPICS[1], "10.2/b1", "Finding three grew 30%."),
                        _cand(brc.TOPICS[1], "10.2/b2", "Finding four dropped 40%.")],
    }
    fixtures = brc.build_slice(by_topic)
    ent = [f for f in fixtures if f["case_type"] == brc.CASE_ENTAILED]
    mis = [f for f in fixtures if f["case_type"] == brc.CASE_MISATTRIBUTED]
    assert len(ent) == 2 and len(mis) >= 1
    for f in ent:
        assert f["cite_doi"] == f["claim_source_doi"]
    for f in mis:
        assert f["cite_doi"] != f["claim_source_doi"]
    # deterministic ids, no dupes
    ids = [f["case_id"] for f in fixtures]
    assert ids == sorted(ids) and len(ids) == len(set(ids))


# ── analyze_n6 schema (deterministic, synthetic rows) ────────────────────────
def _gaming_row(pipe_comp, pipe_band, cur_comp, cur_band, n_verified=0):
    return {"doc_id": "G_00", "group": "G", "error": None,
            "interrogatability": {"composite": pipe_comp, "band": pipe_band},
            "interrog_current_recompute": {"composite": cur_comp, "band": cur_band},
            "entailment": {"n_verified": n_verified}}


def _real_row(doc_id, case_type, verified_fired, n_contr, ent_scores, con_scores,
              verdicts):
    return {"doc_id": doc_id, "group": "R", "case_type": case_type, "error": None,
            "entailment": {"verified_fired": verified_fired, "n_verified": int(verified_fired),
                           "n_contradicted": n_contr, "verdict_statuses": verdicts,
                           "entailment_scores": ent_scores, "contradiction_scores": con_scores}}


def test_analyze_gaming_reports_drop_and_high_elimination():
    rows = [_gaming_row(0.45, "moderate", 0.70, "high"),
            _gaming_row(0.40, "moderate", 0.72, "high")]
    out = analyze_n6.analyze_gaming(rows)
    assert out["n"] == 2
    assert out["n6_pipeline_verified_only"]["high_rate"] == 0.0
    assert out["n6_same_graph_m4_formula"]["high_rate"] == 1.0
    assert out["high_rate_eliminated"] is True
    assert out["verified_claims_in_gaming"] == 0
    assert out["drop_vs_m4_baseline_mean"] > 0  # dropped below 0.725


def test_analyze_real_separates_entailed_from_misattributed():
    rows = [
        _real_row("RC_00", "entailed", True, 0, [0.98], [0.00], ["verified"]),
        _real_row("RC_01", "entailed", True, 0, [0.97], [0.01], ["verified"]),
        _real_row("RC_08", "misattributed", False, 0, [0.03], [0.02], ["unverified"]),
        _real_row("RC_09", "misattributed", False, 0, [], [], []),  # 0-claim, untested
    ]
    out = analyze_n6.analyze_real(rows)
    assert out["entailed_verified_fire_rate"] == 1.0
    assert out["misattributed_verified_fire_rate"] == 0.0
    assert out["false_unverified_rate_entailed_tested"] == 0.0
    assert out["contradicted_fire_count_entailed"] == 0
    assert out["misattributed_tested"] == 1  # RC_09 has no verdict → untested
    assert out["misattributed_tested_all_unverified"] is True
