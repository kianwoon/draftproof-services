"""Safe deterministic cleanup for gpt-oss's stray sentence-period before an intended comma
("...at a high school., I keep asking" -> "...at a high school, I keep asking"). The whole point is
that it NEVER damages a valid abbreviation period ("U.S.," "etc.," "Inc.," "e.g.,"), which is why it
only fires after a >=5-letter lowercase run. No LLM, no rejection, no-op on clean text.

allow-hardcode: the sample sentences below are test fixtures (inputs to exercise the normalizer and
its abbreviation-safety), not matching logic or a phrase list.
"""
from __future__ import annotations

from poc.rewrite_v6 import direct_rewrite as dr


def test_fixes_stray_period_before_comma_lowercase_word():
    out = dr._normalize_punctuation("During the three years I taught biology at a high school., I keep asking.")
    assert "high school, I keep asking." in out
    assert "school.," not in out


def test_fixes_when_word_is_capitalized():
    # binds to the trailing lowercase letters, so a capitalized real word is still fixed
    assert dr._normalize_punctuation("School., I said.") == "School, I said."


def test_preserves_capitalized_and_dotted_abbreviations():
    for valid in (
        "I work for the U.S., which is large.",
        "We joined Acme Inc., a small firm.",
        "Use AI, e.g., a chatbot, in class.",
        "See section i.e., the intro, first.",
    ):
        assert dr._normalize_punctuation(valid) == valid  # unchanged


def test_preserves_short_lowercase_abbreviations():
    # short tokens (<5 lowercase) are exactly where abbreviations live -> never touched
    for valid in ("books, etc., are fine.", "vol., no., pp., all kept."):
        assert dr._normalize_punctuation(valid) == valid


def test_noop_on_clean_text_and_other_punctuation():
    clean = "This sentence is perfectly clean. It ends well, then continues; finally it stops."
    assert dr._normalize_punctuation(clean) == clean
    assert dr._normalize_punctuation("An ellipsis... is preserved.") == "An ellipsis... is preserved."
    assert dr._normalize_punctuation("A decimal 3.14, used here.") == "A decimal 3.14, used here."


def test_handles_empty_and_none():
    assert dr._normalize_punctuation("") == ""
    assert dr._normalize_punctuation(None) == ""
