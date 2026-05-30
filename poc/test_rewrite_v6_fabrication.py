"""Fabricated-named-entity detection for the lean direct rewrite.

Pins the #3 guard: the writer is told not to invent named people/institutions/
places, and any that slip through are surfaced as a review flag (annotate, never
reject). This tests the detector that drives that flag.
"""

from __future__ import annotations

from poc.rewrite_v6.direct_rewrite import _has_fabricated_named_entities

SOURCE = "Students learn from teachers and many digital platforms such as YouTube and TikTok."


def test_flags_invented_honorific_name():
    assert _has_fabricated_named_entities("Mr. Patel uploaded a lesson for the class.", SOURCE) is True


def test_flags_invented_institution_phrase():
    assert _has_fabricated_named_entities("At Lincoln High School, students used new tools.", SOURCE) is True


def test_no_flag_for_generic_referents():
    assert _has_fabricated_named_entities(
        "A student in a typical classroom relies on a teacher to interpret sources.", SOURCE
    ) is False


def test_no_flag_when_proper_noun_is_in_source():
    # Multi-word proper noun that appears in the source is not fabricated.
    src = "The school partnered with Khan Academy for math practice."
    assert _has_fabricated_named_entities("Students used Khan Academy each week.", src) is False


def test_no_flag_for_sentence_initial_single_capital():
    # A single capitalized word starting a sentence is normal, not a proper-noun phrase.
    assert _has_fabricated_named_entities("Education changes quickly. Schools adapt slowly.", SOURCE) is False
