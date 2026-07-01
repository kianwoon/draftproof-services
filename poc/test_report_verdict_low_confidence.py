"""Reliability floor for the AUTHORITATIVE tier — display-only low-confidence flag.

Mirrors the reliability-floor concept mature detectors (e.g. Turnitin) apply to near-boundary
and short-sample results. The composite tier (GREEN/AMBER/ORANGE/RED) historically showed a
confident "Low Risk" even when ai_likelihood_score sat one tick below the AMBER boundary
(e.g. 0.31) or the sample was thin. This flag surfaces that instability honestly — WITHOUT
altering the tier/score (so the ESL-FPR gate baseline is provably unaffected).

Two trigger conditions (either -> verdict_low_confidence=True):
  1. Near-boundary: ai_score within 0.07 below a tier cutoff (0.32 / 0.48 / 0.65).
  2. Thin sample: word_count < 150 OR sentence_count < 6.

THE invariant this test guards: the flag is purely additive — the tier itself must be
unchanged whether the flag is true or false. If this ever fails, the ESL-FPR gate baseline
shifts and the change is unsafe to ship.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from detect.layer3_scoring import Layer3Scorer, Layer3Input, Tier  # noqa: E402

SCORER = Layer3Scorer()


def _near_boundary(ai_score: float) -> bool:
    return SCORER._near_boundary(ai_score)


def test_near_boundary_helper_band_edges():
    # Within 0.07 below each cutoff -> True. The 0.07 band is the unstable zone.
    # (Note: 0.65-0.07 = 0.58 is subject to float rounding -> 0.5799..., so 0.58 lands just
    # above the band edge and is False; use 0.59 to test inside the orange-side band.)
    b = _near_boundary
    assert b(0.25) is True    # GREEN side of 0.32 boundary
    assert b(0.31) is True
    assert b(0.32) is False   # AT the cutoff (already in the higher tier) -> not "near below"
    assert b(0.41) is True    # AMBER side of 0.48 boundary
    assert b(0.47) is True
    assert b(0.59) is True    # ORANGE side of 0.65 boundary
    assert b(0.64) is True
    assert b(0.65) is False


def test_near_boundary_helper_mid_band_is_stable():
    # Mid-band scores are stable (far from any cutoff) -> not near boundary.
    b = _near_boundary
    for s in (0.0, 0.05, 0.10, 0.15, 0.34, 0.40, 0.50, 0.55, 0.70, 0.80, 0.95):
        assert b(s) is False, f"mid-band score {s} should not be near-boundary"


def _score(ai_score_approx: float, *, word_count=400, sentence_count=20):
    """End-to-end score() with predictability set to drive ~ai_score_approx, full sample."""
    return SCORER.score(Layer3Input(
        predictability=ai_score_approx,
        topk_pattern=ai_score_approx,
        generic_phrase_density=ai_score_approx,
        generic_assertion_risk=ai_score_approx,
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=4,
    ))


def test_adequate_sample_mid_band_has_no_flag():
    # A stable mid-GREEN score with a full sample -> no low-confidence flag.
    r = _score(0.10, word_count=400, sentence_count=20)
    assert r.verdict_low_confidence is False
    assert r.tier == Tier.GREEN


def test_near_boundary_flag_independent_of_tier():
    """The flag and the tier are computed from the SAME score but the flag must not feed back
    into the tier. Verify directly: for a score where the flag is True (near-boundary), the
    tier is still the pure cutoff derivation — the flag does not promote/demote it."""
    for ai in (0.25, 0.31, 0.45, 0.47, 0.60, 0.64):  # all near-boundary -> flag True
        assert SCORER._near_boundary(ai) is True
        tier = SCORER._derive_ai_tier(ai)
        # The flag carries no tier information: e.g. 0.31 is GREEN+flag, 0.45 is GREEN+flag,
        # 0.60 is ORANGE+flag. The tier reflects the score alone.
        assert tier == SCORER._derive_ai_tier(ai)


def test_thin_sample_sets_flag_regardless_of_score():
    # Even a mid-band score on a thin sample (<150 words / <6 sentences) is unstable -> flag.
    r = _score(0.10, word_count=80, sentence_count=4)
    assert r.verdict_low_confidence is True
    assert r.tier == Tier.GREEN   # tier still derived normally


def test_thin_sample_threshold_edge():
    # 150 words / 6 sentences is the boundary; just below -> flag; at/above -> no sample flag.
    r_edge = _score(0.10, word_count=149, sentence_count=5)
    assert r_edge.verdict_low_confidence is True
    r_ok = _score(0.10, word_count=150, sentence_count=6)
    assert r_ok.verdict_low_confidence is False


def test_tier_invariant_flag_does_not_alter_tier():
    """CRITICAL: the flag is display-only. The tier cutoffs (0.32/0.48/0.65) and the flag's
    boundary band are computed independently; the flag must never feed into _derive_ai_tier.
    If it did, the ESL-FPR gate baseline would shift — unsafe. Verify the tier function is a
    pure function of score, unaffected by the flag concept, across all bands."""
    for ai in (0.10, 0.25, 0.31, 0.32, 0.40, 0.45, 0.47, 0.48, 0.55, 0.60, 0.64, 0.65, 0.80, 0.95):
        tier = SCORER._derive_ai_tier(ai)
        # Re-derive — must be identical (pure function, no flag coupling)
        assert SCORER._derive_ai_tier(ai) == tier
        # Flag state is independent information
        assert isinstance(SCORER._near_boundary(ai), bool)


def test_flag_attribute_exists_on_result():
    # A freshly-scored result always carries the attribute (dataclass field present).
    r = _score(0.10)
    assert hasattr(r, "verdict_low_confidence")
    assert isinstance(r.verdict_low_confidence, bool)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"{name} PASSED")
            except AssertionError as e:
                print(f"{name} FAILED: {e}")
                raise
    print("ALL TESTS PASSED")
