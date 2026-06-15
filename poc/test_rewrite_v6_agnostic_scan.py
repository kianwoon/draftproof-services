"""Targeting must be driven by content-derived grounding signals, not hardcoded word lists."""
from poc.rewrite_v6.scan import (
    scan_text,
    scan_text_with_report,
    findings_for_paragraph,
)

# --- Sentences that ONLY trip the removed content-word detectors -------------------------------

def _all_tags(scan):
    return {tag for finding in scan.findings for tag in finding.tags}

def test_broad_claim_wordlist_no_longer_tags():
    # "one of the" + "the most" tripped _broad_claim; no number/name/citation => no grounding tell.
    scan = scan_text("This is one of the most significant ideas of the modern age.")
    assert "broad_claim" not in _all_tags(scan)

def test_transition_stack_wordlist_no_longer_tags():
    scan = scan_text("However, furthermore, the situation continued to develop over the period.")
    assert "transition_stack" not in _all_tags(scan)

def test_predictable_start_wordlist_no_longer_tags():
    scan = scan_text("This shows that the overall direction of the work stayed consistent throughout.")
    assert "predictable_start" not in _all_tags(scan)
    assert "context_anchor_gap" not in _all_tags(scan)

def test_evaluative_wordlist_no_longer_tags():
    # "important" + "challenge" with no first-person tripped _author_anchor_gap.
    scan = scan_text("It is important to address the challenge that the situation presents to everyone.")
    assert "author_anchor_gap" not in _all_tags(scan)
    assert "unsupported_claim_gap" not in _all_tags(scan)

# --- Structural-agnostic detectors MUST survive -----------------------------------------------

def test_list_pressure_structural_tag_survives():
    scan = scan_text("The system tracks revenue, costs, staffing, latency, and uptime across teams.")
    assert "packed_list" in _all_tags(scan)

def test_repeated_frame_structural_finding_survives():
    # allow-hardcode: this is a test fixture constructing a paragraph with a known structural
    # property (4 sentences sharing the same 3-word frame prefix), NOT a scoring/matching
    # oracle or allow-list. The assertion is on the structural tag, not the phrase content.
    # _sentence_frame() returns words[:3] for non-article starts; "Results showed clear" repeats
    # 4 times in one paragraph, which satisfies the ≥3 identical-frame threshold.
    frame = "Results showed clear"
    sentences = [
        f"{frame} improvements across the board.",
        f"{frame} differences among the groups.",
        f"{frame} patterns in the data.",
        f"{frame} trends over the period.",
    ]
    text = " ".join(sentences)
    scan = scan_text(text)
    assert "repeated_sentence_frame" in _all_tags(scan)


from poc.rewrite_v6.scan import _citation_anchor

def test_citation_anchor_is_structural_form_not_verb_list():
    # Structural citation FORM still recognized.
    assert _citation_anchor("Smith et al. (2019) reported a measurable shift.") is True
    # Bare reporting verb with no citation form is NO LONGER a citation tell.
    assert _citation_anchor("The author indicates that the result holds.") is False
    assert _citation_anchor("According to many, the trend continued.") is False


import os

def _grounding_report(sentence_text, *, paragraph_id="p001", sentence_id="s1",
                      actionability="auto_fixable", title="low_specificity", category="genericity"):
    return {
        "scan_intelligence": {"document": {"paragraphs": [{"paragraph_id": paragraph_id}]}},
        "sentence_map": {sentence_id: {"paragraph_id": paragraph_id, "text": sentence_text}},
        "highlight_segments": [
            {
                "sentence_id": sentence_id,
                "signals": [
                    {"title": title, "category": category, "score": 0.72,
                     "actionability": actionability, "finding_id": "f1"}
                ],
            }
        ],
    }

def test_grounding_finding_drives_targeting_without_env_switch(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_SCANNER_PREDICTABILITY", raising=False)
    text = "The approach generally improves outcomes for the relevant population over time."
    scan = scan_text_with_report(text, _grounding_report(text))
    pid = scan.paragraphs[0].id
    tags = {t for f in findings_for_paragraph(scan, pid) for t in f.tags}
    assert any("specificity" in t or "genericity" in t for t in tags)


def test_review_only_predictability_does_not_drive_targeting():
    text = "The framework adapts to the situation and supports the people who depend on it daily."
    report = _grounding_report(
        text, actionability="review_only", title="high_predictability", category="predictability"
    )
    scan = scan_text_with_report(text, report)
    pid = scan.paragraphs[0].id
    # No structural finding (sentence is not a packed list / overload), and the only report signal
    # is review_only predictability -> paragraph must NOT be flagged for rewrite.
    assert findings_for_paragraph(scan, pid) == []

def test_mixed_segment_still_drives_when_a_mitigable_signal_present():
    text = "The framework adapts to the situation and supports the people who depend on it daily."
    report = {
        "scan_intelligence": {"document": {"paragraphs": [{"paragraph_id": "p001"}]}},
        "sentence_map": {"s1": {"paragraph_id": "p001", "text": text}},
        "highlight_segments": [
            {"sentence_id": "s1", "signals": [
                {"title": "high_predictability", "category": "predictability",
                 "score": 0.8, "actionability": "review_only", "finding_id": "f1"},
                {"title": "low_specificity", "category": "genericity",
                 "score": 0.7, "actionability": "auto_fixable", "finding_id": "f2"},
            ]},
        ],
    }
    scan = scan_text_with_report(text, report)
    pid = scan.paragraphs[0].id
    assert findings_for_paragraph(scan, pid) != []
