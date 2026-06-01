"""The surfaced external-detector estimate must be the MORE CONSERVATIVE (higher-band) of the two
estimators, so the dual-headline never under-warns. Regression guard for the GPTZero-100%-vs-
segment-6% mismatch: the perplexity-blend ('high') must win over the segment-fraction ('low'),
not the other way around (the old `segment or likelihood` preference under-warned)."""
from poc.detect.layer3_scoring import combine_external_detector_estimates


SEGMENT_LOW = {"score": 6.3, "band": "low", "model": "segment_fraction_v1", "note": "seg"}
LIKELIHOOD_HIGH = {"score": 61.5, "band": "high", "note": "perp"}


def test_higher_band_wins_when_segment_low_likelihood_high():
    # The exact GPTZero mismatch: segment says low (6.3), perplexity says high (61.5).
    out = combine_external_detector_estimates(SEGMENT_LOW, LIKELIHOOD_HIGH)
    assert out["band"] == "high"
    assert out["score"] == 61.5
    assert out["note"] == "perp"


def test_segment_wins_when_it_is_the_higher_band():
    seg_high = {"score": 55.0, "band": "high", "note": "seg"}
    lik_elev = {"score": 40.0, "band": "elevated", "note": "perp"}
    out = combine_external_detector_estimates(seg_high, lik_elev)
    assert out["band"] == "high"
    assert out["score"] == 55.0
    assert out["note"] == "seg"


def test_tie_band_breaks_to_higher_score():
    a = {"score": 38.0, "band": "elevated", "note": "a"}
    b = {"score": 45.0, "band": "elevated", "note": "b"}
    out = combine_external_detector_estimates(a, b)
    assert out["score"] == 45.0
    assert out["note"] == "b"


def test_both_alternates_attached_for_transparency():
    out = combine_external_detector_estimates(SEGMENT_LOW, LIKELIHOOD_HIGH)
    assert out["alternates"]["segment_fraction"] == SEGMENT_LOW
    assert out["alternates"]["likelihood"] == LIKELIHOOD_HIGH


def test_falls_back_to_likelihood_when_segment_is_none():
    out = combine_external_detector_estimates(None, LIKELIHOOD_HIGH)
    assert out["band"] == "high"
    assert out["score"] == 61.5


def test_none_only_when_both_missing():
    assert combine_external_detector_estimates(None, None) is None
    assert combine_external_detector_estimates({"band": "low"}, None) is None  # no numeric score
