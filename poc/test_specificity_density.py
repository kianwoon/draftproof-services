"""Regression tests for the specificity-density normalisation fix.

Bug: ``specificity = min(1.0, feature_sum) / (word_count/100)`` clamped the RAW feature
sum to 1.0 BEFORE length-normalising. For longer documents the sum routinely exceeds 1.0,
so it was capped and then divided by length -- crushing specificity for long docs and
INVERTING the signal: a heavily-grounded long essay scored as LESS specific (higher
AI-risk) than short, generic, anchorless text. Fix normalises first, clamps last.
"""
from detect.criteria import specificity

# allow-hardcode: these are TEST FIXTURE documents (sample inputs fed to the scorer), not
# a scoring/matching oracle or any detection list. They are never compared against user text.
_GROUNDED = ("In 2019, Dr. Maria Gomez at Box Hill Institute recorded a 37% improvement "
             "for 42 students using the Octagon Method across 12 weeks in Melbourne. ")
_GENERIC = ("It is widely believed that many things are important in different ways, and "
            "people often tend to think about various ideas and consider many aspects. ")


def _score(text):
    return specificity.score(text)


def test_long_grounded_doc_reads_more_specific_than_long_generic_doc():
    # Both long enough that the old cap-then-divide would have crushed the grounded one.
    grounded = _score(_GROUNDED * 6)
    generic = _score(_GENERIC * 6)
    # higher specificity_score = more concrete; the grounded doc must read more specific
    assert grounded.details["specificity_score"] > generic.details["specificity_score"]
    # ...and therefore carry LESS specificity-risk (the inversion the bug created)
    assert grounded.value < generic.value


def test_long_dense_doc_not_crushed_to_zero():
    # The bug collapsed a feature-rich long doc's specificity toward 0; it must now retain
    # real density rather than being penalised purely for length.
    assert _score(_GROUNDED * 6).details["specificity_score"] > 0.05


def test_low_specificity_finding_is_anchored_to_a_real_sentence():
    # Regression for the empty-evidence gap: the finding must point at a concrete vague
    # sentence (flagged_excerpts) instead of surfacing an empty excerpt.
    res = _score(_GENERIC * 6)
    assert res.flagged_excerpts, "expected non-empty flagged_excerpts for generic text"
    top = res.flagged_excerpts[0]
    assert len(top.split()) >= 8              # a real sentence, not a fragment
    assert "widely believed" in top.lower()   # drawn from the actual input text


def test_anchored_sentence_is_not_flagged_as_vaguest():
    # Vaguest-first ordering: a richly-anchored sentence must not appear among the flags
    # when generic ones are present.
    # allow-hardcode: test-fixture sentence (sample input), not detection logic.
    anchored = ("In 2019, Maria Gomez at Box Hill Institute measured a 37% gain for 42 "
                "students across 12 weeks in Melbourne.")
    doc = (_GENERIC + " ") * 3 + anchored
    res = _score(doc)
    assert res.flagged_excerpts
    assert all("Box Hill Institute" not in ex for ex in res.flagged_excerpts)
