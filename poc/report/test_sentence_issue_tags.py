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
    assert colors == ["red", "amber", "purple"]
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
