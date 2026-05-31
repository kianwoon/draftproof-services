"""Domain grounding must require at least one concrete anchor.

Bug: `domain_grounding_index = strong_domain_count / (word_count/100)` maxes to 1.0 on
generic domain *vocabulary* (test_content11: "comfortably", "supporting", "situations",
"education system" -> 100% domain grounding) even with 0 named entities, 0 numbers,
0 dates, 0 quotes. That inflates `human_anchor_score` (0.20 of it is domain_anchor),
which discounts the footprint AI risk and drives the "Human anchoring dominates" framing.

Real subject-matter grounding co-occurs with at least one concrete anchor. Gate:
when there are ZERO concrete anchors anywhere, cap domain_grounding_strength low.
Docs with real anchors (entities/numbers/dates/quotes) are unaffected.
"""

from report.report import ReportBuilder


def gate(strength, anchors):
    return ReportBuilder._gate_domain_grounding(strength, anchors)


def test_no_anchors_caps_inflated_grounding():
    # content11 case: maxed grounding (1.0) but zero concrete anchors -> capped.
    assert gate(1.0, 0) == 0.30


def test_no_anchors_below_cap_is_unchanged():
    assert gate(0.20, 0) == 0.20
    assert gate(0.0, 0) == 0.0


def test_no_anchors_at_cap_boundary():
    assert gate(0.30, 0) == 0.30


def test_with_concrete_anchors_grounding_is_preserved():
    # Real grounding (entities/numbers/etc. present) must NOT be gated.
    assert gate(1.0, 1) == 1.0
    assert gate(0.75, 5) == 0.75
    assert gate(0.55, 23) == 0.55


def test_negative_anchor_count_treated_as_none():
    assert gate(1.0, -1) == 0.30
