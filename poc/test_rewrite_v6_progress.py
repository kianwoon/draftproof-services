"""Progress reporting for the V6 lean direct rewrite.

Pins the work-based progress fix:
* the per-paragraph bar reflects flagged sections COMPLETED (done/total), not a
  paragraph's position in the document;
* best-of-N passes map into monotonic sub-bands so the bar never jumps backwards
  when the whole per-paragraph loop reruns.
"""

from __future__ import annotations

from poc.rewrite_v6.direct_rewrite import (
    _PROGRESS_CEIL,
    _PROGRESS_FLOOR,
    _attempt_progress,
    _section_progress,
)


def test_section_progress_is_work_based_and_bounded():
    assert _section_progress(0, 3) == _PROGRESS_FLOOR
    assert _section_progress(3, 3) == _PROGRESS_CEIL
    assert _PROGRESS_FLOOR < _section_progress(1, 3) < _section_progress(2, 3) < _PROGRESS_CEIL


def test_section_progress_handles_no_flagged_and_overflow():
    assert _section_progress(0, 0) == _PROGRESS_FLOOR   # nothing flagged
    assert _section_progress(5, 3) == _PROGRESS_CEIL     # done > total clamps


def test_section_progress_is_monotonic():
    seq = [_section_progress(d, 4) for d in range(5)]
    assert seq == sorted(seq)
    assert seq[0] == _PROGRESS_FLOOR and seq[-1] == _PROGRESS_CEIL


def test_attempt_progress_passthrough_for_single_attempt():
    cb = lambda p, m: None
    assert _attempt_progress(cb, 0, 1) is cb
    assert _attempt_progress(None, 0, 2) is None


def test_attempt_progress_maps_passes_into_monotonic_subbands():
    events: list[tuple[int, str]] = []

    def cb(percent, message):
        events.append((percent, message))

    # Two best-of-N passes, each replaying the local 40..78 sweep for 4 sections.
    for attempt in range(2):
        wrapped = _attempt_progress(cb, attempt, 2)
        for done in range(4):
            wrapped(_section_progress(done, 4), f"Rewriting section {done + 1} of 4")
        wrapped(_PROGRESS_CEIL, "Rewriting section 4 of 4")

    percents = [p for p, _ in events]
    # The bar only ever advances across BOTH passes (no reset on pass 2).
    assert percents == sorted(percents)
    assert min(percents) >= _PROGRESS_FLOOR
    assert max(percents) <= _PROGRESS_CEIL

    first_pass = events[:5]
    second_pass = events[5:]
    assert any("section 1 of 4" in m for _, m in first_pass)      # counting on pass 1
    assert all(m == "Refining rewrite" for _, m in second_pass)   # refinement on pass 2+
