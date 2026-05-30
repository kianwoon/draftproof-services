"""Author-proxy-as-author: first-person grounding is encouraged, but when the proxy ADDS
first-person experience the author must confirm it -- so it is surfaced as a review flag
(annotate, never reject). Pins that detector.
"""

from __future__ import annotations

from poc.rewrite_v6.direct_rewrite import (
    _has_added_first_person_experience as added_fp,
    _ungrounded_claims,
)

# Third-person source (no authorial first person of its own).
SRC = "Teachers shape learning. Content is now plentiful. The education system must evolve."


def test_flags_added_first_person_experience():
    assert added_fp("In my classroom, I have seen students lean on AI for first drafts.", SRC) is True
    assert added_fp("When I began teaching, lessons centered on a single printed textbook.", SRC) is True


def test_no_flag_for_detached_grounding():
    assert added_fp("Teachers guide students to question sources and verify authorship.", SRC) is False


def test_no_flag_when_source_is_already_first_person():
    fp_src = "In my classroom I have always asked students to verify their sources."
    # The author already writes in first person, so the proxy isn't newly attributing experience.
    assert added_fp("In my classroom, I now ask them to check three sources before writing.", fp_src) is False


def test_ungrounded_claims_lists_still_generic_sentences():
    text = (
        "Technology is changing education rapidly. Good teaching matters in every classroom. "
        "For example, when a student drafts an essay with AI, the teacher checks each claim. "
        "Learning should be meaningful and engaging for all."
    )
    claims = _ungrounded_claims(text)
    # The two bare claims are flagged; the grounded "for example, when..." sentence is not.
    assert any("Technology is changing education" in c for c in claims)
    assert any("Good teaching matters" in c for c in claims)
    assert not any("for example" in c.lower() for c in claims)


def test_ungrounded_claims_skips_short_fragments():
    # Short sentences are not treated as substantive ungrounded claims.
    assert _ungrounded_claims("It matters. Things change. We adapt.") == []
