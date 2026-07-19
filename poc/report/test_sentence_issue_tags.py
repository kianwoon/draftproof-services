"""Tests for the sentence-issue-tag display composer (display-layer only).

Guardrails under test (owner-approved design):
  * ONLY trustworthy findings produce tags: low_specificity -> grounding (amber),
    semantic_drift -> reasoning (purple), ai_signal_deberta -> ai (red).
  * Predictability / genericity findings NEVER produce a tag (methodology soup).
  * A sentence with AI + low_specificity yields BOTH tags, in stable order.
  * semantic_drift with no sentence_id -> document_level, not a sentence.
  * Absent trustworthy findings -> None (byte-identical fallback).
  * No fabrication: tags derive only from real findings/segments.
"""
from __future__ import annotations

from report.sentence_issue_tags import compose_sentence_issue_tags


def _seg(sid, tier="high"):
    return {
        "sentence_id": sid,
        "type": "sentence",
        "signals": [{"key": "ai_signal_deberta", "tier": tier}],
    }


def _finding(title, sid, category="ai_generation"):
    return {"title": title, "category": category, "sentence_id": sid,
            "recommendation": f"rec for {title}"}


def test_none_when_no_trustworthy_findings():
    fields = {
        "findings": {
            "medium": [_finding("medium_predictability", "s009", "predictability"),
                       _finding("high_topk_predictability", "s009")],
            "low": [_finding("review_predictability", "s001", "predictability"),
                    _finding("genericity", "s002")],
        },
        "highlight_segments": [{"sentence_id": "s001", "signals": []}],
    }
    assert compose_sentence_issue_tags(fields) is None


def test_none_on_empty_or_bad_input():
    assert compose_sentence_issue_tags(None) is None
    assert compose_sentence_issue_tags({}) is None
    assert compose_sentence_issue_tags({"findings": {}, "highlight_segments": []}) is None


def test_predictability_genericity_never_tag():
    fields = {
        "findings": {
            "medium": [_finding("medium_predictability", "s009", "predictability"),
                       _finding("high_topk_predictability", "s009"),
                       _finding("low_specificity", "s009")],
            "low": [_finding("review_predictability", "s001", "predictability"),
                    _finding("genericity", "s001")],
        },
        "highlight_segments": [],
    }
    out = compose_sentence_issue_tags(fields)
    assert out is not None
    # s009 has ONLY the grounding tag — the two predictability titles are dropped.
    assert set(out["sentences"].keys()) == {"s009"}
    types = [t["type"] for t in out["sentences"]["s009"]]
    assert types == ["grounding"]
    assert out["sentences"]["s009"][0]["color"] == "amber"


def test_ai_from_high_and_amber_segments_only():
    fields = {
        "findings": {},
        "highlight_segments": [
            _seg("s001", "high"),
            _seg("s002", "amber"),
            _seg("s003", "clean"),          # not high/amber -> no tag
            {"sentence_id": "s004", "signals": [{"key": "other", "tier": "high"}]},
        ],
    }
    out = compose_sentence_issue_tags(fields)
    assert set(out["sentences"].keys()) == {"s001", "s002"}
    assert out["sentences"]["s001"][0]["type"] == "ai"
    assert out["sentences"]["s001"][0]["color"] == "red"


def test_multi_issue_sentence_stacks_ai_and_grounding_in_stable_order():
    fields = {
        "findings": {"medium": [_finding("low_specificity", "s001")]},
        "highlight_segments": [_seg("s001", "high")],
    }
    out = compose_sentence_issue_tags(fields)
    tags = out["sentences"]["s001"]
    # BOTH present, order stable: ai first, grounding second.
    assert [t["type"] for t in tags] == ["ai", "grounding"]
    assert [t["color"] for t in tags] == ["red", "amber"]


def test_semantic_drift_without_sentence_id_goes_document_level():
    fields = {
        "findings": {"high": [_finding("semantic_drift", None, "semantic_shape")]},
        "highlight_segments": [],
    }
    out = compose_sentence_issue_tags(fields)
    assert out["sentences"] == {}
    assert len(out["document_level"]) == 1
    assert out["document_level"][0]["type"] == "reasoning"
    assert out["document_level"][0]["color"] == "purple"


def test_semantic_drift_with_sentence_id_attaches_to_sentence():
    fields = {
        "findings": {"high": [_finding("semantic_drift", "s005", "semantic_shape")]},
        "highlight_segments": [],
    }
    out = compose_sentence_issue_tags(fields)
    assert out["document_level"] == []
    assert out["sentences"]["s005"][0]["type"] == "reasoning"


def test_legend_present_and_english_fallbacks_carried():
    fields = {
        "findings": {"medium": [_finding("low_specificity", "s009")]},
        "highlight_segments": [_seg("s001", "high")],
    }
    out = compose_sentence_issue_tags(fields)
    colors = [row["color"] for row in out["legend"]]
    # Legend always lists all 4 known types (unconditional, matching the
    # pre-existing red/amber/purple behavior) — citation joined 2026-07-19.
    assert colors == ["red", "amber", "purple", "blue"]
    # Every tag carries an i18n code AND an English fallback (for the PDF).
    tag = out["sentences"]["s009"][0]
    assert tag["label_code"] and tag["label_en"]
    assert tag["fix_code"] and tag["fix_en"]


def test_no_duplicate_ai_tags_for_repeated_segment():
    fields = {
        "findings": {},
        "highlight_segments": [_seg("s001", "high"), _seg("s001", "high")],
    }
    out = compose_sentence_issue_tags(fields)
    assert len(out["sentences"]["s001"]) == 1


# ── Citation (blue), added 2026-07-19 ──
#
# Citation findings use `sentence_id` exactly like grounding/reasoning — it is
# ALREADY resolved upstream (poc/report/builder.py::add_detection's "Strategy 2",
# start_char -> containing predictability sentence) by the time a finding dict
# reaches this module. There is no separate location/document_text mechanism
# here (an earlier version of this file tried that; `location` never survives
# report-layer serialization into result["findings"] in the first place, so it
# would have been silently inert on every real report — see report.py's
# `_tier_findings`, which serializes `sentence_id` but not `location`).


def test_citation_finding_with_sentence_id_is_tagged():
    fields = {"findings": {"high": [_finding("missing_from_bib", "s002", "citation")]},
              "highlight_segments": []}
    out = compose_sentence_issue_tags(fields)
    assert out is not None
    assert set(out["sentences"].keys()) == {"s002"}
    tag = out["sentences"]["s002"][0]
    assert tag["type"] == "citation"
    assert tag["color"] == "blue"


def test_uncited_claim_also_tags():
    fields = {"findings": {"medium": [_finding("uncited_claim", "s003", "citation")]},
              "highlight_segments": []}
    out = compose_sentence_issue_tags(fields)
    assert set(out["sentences"].keys()) == {"s003"}
    assert out["sentences"]["s003"][0]["type"] == "citation"


def test_uncited_in_body_never_tags_a_sentence():
    """uncited_in_body is excluded from _CITATION_TITLES entirely (module
    docstring: its evidence lives only in the bibliography, which
    CitationDetector strips before locating evidence, so it never resolves a
    sentence_id upstream either) — even WITH a sentence_id present here (e.g. a
    future upstream change that resolves one anyway), it must still never tag,
    proving the exclusion is title-based, not merely an absent-sentence_id
    side effect."""
    fields = {"findings": {"low": [_finding("uncited_in_body", "s001", "citation")]},
              "highlight_segments": []}
    assert compose_sentence_issue_tags(fields) is None


def test_citation_finding_without_sentence_id_is_skipped_not_crashed():
    """Strategy 2 upstream can fail to resolve a sentence_id (e.g. no
    predictability summary available) -- sentence_id absent/None must be
    skipped silently, never crash, never fall back to document_level (an
    unanchorable citation tag would misattribute it to the whole document)."""
    fields = {"findings": {"high": [_finding("missing_from_bib", None, "citation")]},
              "highlight_segments": []}
    out = compose_sentence_issue_tags(fields)
    assert out is None


def test_multiple_citation_findings_across_sentences_are_independent():
    fields = {
        "findings": {
            "high": [_finding("missing_from_bib", "s002", "citation")],
            "medium": [_finding("uncited_claim", "s003", "citation")],
        },
        "highlight_segments": [],
    }
    out = compose_sentence_issue_tags(fields)
    assert set(out["sentences"].keys()) == {"s002", "s003"}
    assert out["sentences"]["s002"][0]["type"] == "citation"
    assert out["sentences"]["s003"][0]["type"] == "citation"


def test_two_citation_findings_same_sentence_dedupe_to_one():
    fields = {
        "findings": {
            "high": [_finding("missing_from_bib", "s002", "citation")],
            "medium": [_finding("uncited_claim", "s002", "citation")],
        },
        "highlight_segments": [],
    }
    out = compose_sentence_issue_tags(fields)
    assert len(out["sentences"]["s002"]) == 1
    assert out["sentences"]["s002"][0]["type"] == "citation"


def test_citation_stacks_with_ai_and_grounding_in_stable_order():
    fields = {
        "findings": {
            "medium": [_finding("low_specificity", "s002"),
                       _finding("missing_from_bib", "s002", "citation")],
        },
        "highlight_segments": [_seg("s002", "high")],
    }
    out = compose_sentence_issue_tags(fields)
    tags = out["sentences"]["s002"]
    assert [t["type"] for t in tags] == ["ai", "grounding", "citation"]
    assert [t["color"] for t in tags] == ["red", "amber", "blue"]
