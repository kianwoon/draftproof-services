"""Author-proxy-as-author: first-person grounding is encouraged, but when the proxy ADDS
first-person experience the author must confirm it -- so it is surfaced as a review flag
(annotate, never reject). Pins that detector.
"""

from __future__ import annotations

from poc.rewrite_v6.direct_rewrite import _has_added_first_person_experience as added_fp

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
