"""Honest partial-degradation signal for the DEFAULT direct rewrite path.

A writer/LLM call that raises (provider outage, auth, timeout, empty content, rate-limit exhausted)
is swallowed to source_preserved -- previously indistinguishable, at the outcome layer, from a
LEGITIMATE no-change (a candidate was produced but rejected as a stub / broken grammar / meaning
flip). A provider partial outage therefore shipped a mostly-unchanged, fully-billed rewrite with no
user-facing signal. ``direct_rewrite`` now tags the error-caused fallbacks ``reject_reason=
"writer_error"``; ``production`` counts them into ``summary.writer_degraded_paragraphs`` and composes
an honest author-review card. This locks that behaviour end-to-end (no LLM required).
"""
from __future__ import annotations

from poc.rewrite_v6.direct_rewrite import (
    _paragraph_outcome,
    _record_writer_error,
    _writer_error_recorded,
    _writer_error_recording,
)
from poc.rewrite_v6.production import _writer_degraded_card, _writer_degraded_counts
from poc.rewrite_v6.text import Paragraph


def _para(pid: str = "p001") -> Paragraph:
    return Paragraph(id=pid, index=0, text="The system is efficient and robust and effective.", sentences=[])


# --- reason distinction at the outcome layer -------------------------------------------------


def test_outcome_tags_writer_error_when_recorded():
    para = _para("p001")
    with _writer_error_recording():
        _record_writer_error(para.id)
        assert _writer_error_recorded(para.id) is True
        text, row = _paragraph_outcome(0, para, None, [])
    assert text == para.text  # original preserved
    assert row["selected_source"] == "source_preserved"
    assert row["reject_reason"] == "writer_error"


def test_outcome_tags_no_clean_rewrite_when_not_recorded():
    para = _para("p002")
    with _writer_error_recording():
        # A legitimate no-change: no writer error was recorded for this paragraph.
        text, row = _paragraph_outcome(0, para, None, [])
    assert text == para.text
    assert row["reject_reason"] == "no_clean_rewrite"


def test_outcome_ignores_sink_when_candidate_delivered():
    para = _para("p003")
    with _writer_error_recording():
        _record_writer_error(para.id)  # errored on an early attempt but a retry succeeded
        text, row = _paragraph_outcome(0, para, "A grounded, human-sounding rewrite of the claim.", [])
    assert text.startswith("A grounded")
    assert row["status"] == "accepted"
    assert "reject_reason" not in row


def test_recording_is_document_scoped_and_restored():
    assert _writer_error_recorded("p001") is False  # no active sink outside a document pass
    with _writer_error_recording():
        _record_writer_error("p001")
        assert _writer_error_recorded("p001") is True
    # Sink restored on exit -> the id does not bleed into the next document pass.
    assert _writer_error_recorded("p001") is False


# --- count + total from the trace ------------------------------------------------------------


def test_counts_from_writer_error_rows():
    trace = [
        {"target_paragraph_id": "p001", "selected_source": "direct_llm", "status": "accepted"},
        {"target_paragraph_id": "p002", "selected_source": "source_preserved",
         "status": "source_preserved", "reject_reason": "writer_error"},
        {"target_paragraph_id": "p003", "selected_source": "source_preserved",
         "status": "source_preserved", "reject_reason": "no_clean_rewrite"},
    ]
    degraded, total = _writer_degraded_counts(trace)
    assert degraded == 1  # only p002 (error); p003 is a legitimate no-change
    assert total == 3


def test_counts_exclude_recovered_paragraph():
    # p002 errored in pass 1 but was rewritten by a later accepted pass (residual_fix) -> not degraded.
    trace = [
        {"target_paragraph_id": "p002", "selected_source": "source_preserved",
         "status": "source_preserved", "reject_reason": "writer_error"},
        {"target_paragraph_id": "p002", "selected_source": "residual_fix", "status": "accepted"},
    ]
    degraded, total = _writer_degraded_counts(trace)
    assert degraded == 0


def test_counts_dedupe_and_empty():
    assert _writer_degraded_counts([]) == (0, 0)
    assert _writer_degraded_counts(None) == (0, 0)
    dup = [
        {"target_paragraph_id": "p009", "selected_source": "source_preserved",
         "status": "source_preserved", "reject_reason": "writer_error"},
        {"target_paragraph_id": "p009", "selected_source": "source_preserved",
         "status": "source_preserved", "reject_reason": "writer_error"},
    ]
    assert _writer_degraded_counts(dup) == (1, 1)


# --- the honest card -------------------------------------------------------------------------


def test_card_states_exact_counts():
    card = _writer_degraded_card(2, 5)
    assert card["card_id"] == "writer-degraded"
    assert card["kind"] == "service_degradation"
    assert "2 of 5 flagged paragraphs" in card["instruction"]
    assert "temporary service issue" in card["instruction"]
    assert "Re-run the rewrite" in card["author_task"]
    # No fabricated billing/credit claim.
    assert "credit" not in card["author_task"].lower()


def test_card_singular_grammar():
    card = _writer_degraded_card(1, 4)
    assert "1 of 4 flagged paragraph could not" in card["instruction"]
