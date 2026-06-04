"""Tests for honest register/polish coaching (poc/rewrite_v6/register_coaching.py).

The coaching must be PRECISE (anchored to real sentences), NON-REDUNDANT with grounding coaching, and
ABSTAIN (return None) when it cannot be precise. It is NOT a score lever.

allow-hardcode: the _DOC below is a TEST FIXTURE (a sample paragraph to exercise the scorer), not a
detect/scoring/matching word-list. The code under test selects sentences via the content-agnostic
register_score signal; this fixture just provides realistic input.
"""

from rewrite_v6.register_coaching import build_register_coaching, register_coaching_enabled
from rewrite_v6.direct_rewrite import _ungrounded_claims


# A doc mixing grounded-but-polished lines with plain first-person lines (content-agnostic shapes).
_DOC = (
    "Now every draft must contain an argument, receive two rounds of peer feedback, and be revised "
    "before final submission; this forces deeper thinking and clearer communication, raising overall "
    "writing quality. "
    "Right now my school stands at a crossroads. "
    "When a student types projectile motion into Google, ten or more results appear, and they vote in "
    "our class forum on which explanation feels most intuitive. "
    "Last Tuesday I watched Maya and Luis, two seniors, copying textbook definitions of the quadratic "
    "formula at 10 p.m. instead of debating how it applies. "
    "I stopped lecturing and turned the lesson into a source-tracking exercise."
)


def test_enabled_default_on():
    assert register_coaching_enabled() is True


def test_returns_structured_coaching():
    rc = build_register_coaching(_DOC)
    assert rc is not None
    # Honesty: the copy must explicitly state it will NOT lower the detector score.
    assert "will not lower the detector score" in rc["note"].lower()


def test_offenders_are_grounded_and_nonredundant():
    rc = build_register_coaching(_DOC)
    offenders = {o["text"] for o in rc["offenders"]}
    ungrounded = set(_ungrounded_claims(_DOC))
    # Register offenders must be DISJOINT from grounding coaching (genuinely additive surface).
    assert offenders.isdisjoint(ungrounded)


def test_worked_contrast_orders_polished_above_plain():
    rc = build_register_coaching(_DOC)
    wc = rc["worked_contrast"]
    assert wc["polished"]["register_score"] > wc["plain"]["register_score"]
    assert wc["polished"]["text"] != wc["plain"]["text"]


def test_abstains_on_too_short_text():
    assert build_register_coaching("Short. Too short.") is None
    assert build_register_coaching("") is None
