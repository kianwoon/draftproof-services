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
