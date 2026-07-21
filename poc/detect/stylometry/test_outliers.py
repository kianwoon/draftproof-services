"""Tests for poc.detect.stylometry.outliers.

allow-hardcode: the ParagraphFingerprint field values below are hand-authored
synthetic TEST FIXTURE data (a uniform baseline group plus a deliberately shifted
paragraph) used to exercise detect_outliers -- they are test input data, not a
matching/scoring list consumed by production code.

Flag rule (2026-07-21): a paragraph is an outlier only when it deviates on at least
`OUTLIER_MIN_DEVIATING_FEATURES` (=3) feature dimensions, each exceeding
`OUTLIER_PER_FEATURE_Z_THRESHOLD` (=5.25) — the k-of-m rule derived by
calibration/derive_outlier_threshold.py to hold human per-paragraph FP <=5%. The
"shifted paragraph" fixtures below therefore deviate on >=3 dimensions (a genuine
multi-dimensional voice shift, which is what the rule exists to catch); a 1-2 feature
spike is deliberately NOT an outlier anymore (that was the old hair-trigger behavior).

Covers: (1) the brief's literal scenario -- 5 uniform paragraphs + 1 paragraph shifted
on sentence length (3x), function-word rate (near-zero) and word length -- the shifted
paragraph must be flagged and the uniform ones must not; (2) the same shape with realistic
jitter in the baseline (not literally identical values) to exercise the finite,
non-degenerate robust-z path as well as the degenerate (zero-MAD) path; (3) the
MIN_PARAGRAPHS floor returning no findings; (4) the MIN_WORDS_PER_PARAGRAPH floor
excluding too-short paragraphs and returning no findings once too few remain; (5) a
paragraph with flesch_reading_ease=None must not crash and must not be treated as a
0.0 outlier; (6) top-3 feature naming is stable and human-readable (no raw field
names); (7) the LOCAL_OUTLIER_FACTOR stub raises NotImplementedError.
"""
from __future__ import annotations

import math

import pytest

from detect.stylometry.features import ParagraphFingerprint
from detect.stylometry.outliers import (
    MIN_PARAGRAPHS,
    MIN_WORDS_PER_PARAGRAPH,
    OUTLIER_MIN_DEVIATING_FEATURES,
    OUTLIER_PER_FEATURE_Z_THRESHOLD,
    OUTLIER_THRESHOLD,
    OutlierResult,
    OutlierStrategy,
    detect_outliers,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_BASE_PUNCTUATION_RATES = {
    "comma": 5.0,
    "semicolon": 0.5,
    "colon": 0.3,
    "dash": 0.2,
    "paren": 0.1,
}


def _make_fingerprint(
    paragraph_id: str,
    *,
    sentence_length_mean: float = 15.0,
    sentence_length_std: float = 4.0,
    word_length_mean: float = 4.5,
    root_ttr: float = 6.0,
    punctuation_rates: dict[str, float] | None = None,
    transition_rate: float = 2.0,
    function_word_rate: float = 40.0,
    flesch_reading_ease: float | None = 60.0,
    passive_voice_rate: float = 0.2,
    subordination_rate: float = 0.5,
    lexical_density: float = 0.55,
    academic_vocab_rate: float = 3.0,
    word_count: int = 120,
) -> ParagraphFingerprint:
    return ParagraphFingerprint(
        paragraph_id=paragraph_id,
        sentence_length_mean=sentence_length_mean,
        sentence_length_std=sentence_length_std,
        word_length_mean=word_length_mean,
        root_ttr=root_ttr,
        punctuation_rates=dict(punctuation_rates or _BASE_PUNCTUATION_RATES),
        transition_rate=transition_rate,
        function_word_rate=function_word_rate,
        flesch_reading_ease=flesch_reading_ease,
        passive_voice_rate=passive_voice_rate,
        subordination_rate=subordination_rate,
        lexical_density=lexical_density,
        academic_vocab_rate=academic_vocab_rate,
        word_count=word_count,
    )


def _uniform_baseline(count: int = 5) -> list[ParagraphFingerprint]:
    return [_make_fingerprint(f"p{i:03d}") for i in range(1, count + 1)]


# ---------------------------------------------------------------------------
# Core scenario: 5 uniform + 1 shifted (brief's literal example)
# ---------------------------------------------------------------------------


def test_shifted_paragraph_flagged_others_not_identical_baseline():
    baseline = _uniform_baseline(5)
    shifted = _make_fingerprint(
        "p006",
        sentence_length_mean=45.0,  # 3x the baseline's 15.0
        function_word_rate=2.0,  # near-zero vs. baseline's 40.0
        word_length_mean=9.0,  # 2x the baseline's 4.5 -- 3rd deviating dimension
    )

    results = detect_outliers(baseline + [shifted])

    assert [r.paragraph_id for r in results] == ["p006"]
    flagged = results[0]
    assert flagged.outlier_score > OUTLIER_PER_FEATURE_Z_THRESHOLD
    # Report artifacts are serialized to JSON downstream (R2 -> frontend), which has
    # no `Infinity` literal -- the score must always be finite, even in this fully
    # degenerate (byte-identical baseline) case where MAD collapses to zero on
    # every dimension.
    assert math.isfinite(flagged.outlier_score)
    # Exactly the three shifted dimensions differ from the (byte-identical) baseline,
    # so top_deviating_features must be those three and nothing else (no padding with
    # non-deviating z==0.0 names).
    assert set(flagged.top_deviating_features) == {
        "sentence length",
        "function word rate",
        "word length",
    }


def test_shifted_paragraph_flagged_with_jittered_realistic_baseline():
    # Baseline paragraphs are close but not byte-identical -- exercises the
    # finite (non-degenerate-MAD) robust z-score path.
    jitter_sentence_lengths = [14.0, 15.0, 16.0, 15.0, 14.0]
    jitter_function_rates = [38.0, 40.0, 42.0, 39.0, 41.0]
    jitter_word_lengths = [4.4, 4.5, 4.6, 4.5, 4.4]
    baseline = [
        _make_fingerprint(
            f"p{i:03d}",
            sentence_length_mean=jitter_sentence_lengths[i - 1],
            function_word_rate=jitter_function_rates[i - 1],
            word_length_mean=jitter_word_lengths[i - 1],
        )
        for i in range(1, 6)
    ]
    shifted = _make_fingerprint(
        "p006",
        sentence_length_mean=46.0,
        function_word_rate=1.5,
        word_length_mean=9.5,  # 3rd deviating dimension vs. ~4.5 baseline
    )

    results = detect_outliers(baseline + [shifted])

    assert [r.paragraph_id for r in results] == ["p006"]
    assert math.isfinite(results[0].outlier_score)
    assert results[0].outlier_score > OUTLIER_PER_FEATURE_Z_THRESHOLD
    assert set(results[0].top_deviating_features) == {
        "sentence length",
        "function word rate",
        "word length",
    }


# ---------------------------------------------------------------------------
# MIN_PARAGRAPHS floor
# ---------------------------------------------------------------------------


def test_below_min_paragraphs_returns_no_findings():
    # baseline + shifted totals MIN_PARAGRAPHS - 1 paragraphs -- one short of the
    # floor -- so detect_outliers must fail open regardless of how extreme the
    # shifted paragraph is.
    baseline = _uniform_baseline(MIN_PARAGRAPHS - 2)
    shifted = _make_fingerprint("pshift", sentence_length_mean=999.0)

    results = detect_outliers(baseline + [shifted])

    assert results == []


# ---------------------------------------------------------------------------
# MIN_WORDS_PER_PARAGRAPH floor
# ---------------------------------------------------------------------------


def test_too_short_paragraphs_excluded_then_floor_returns_no_findings():
    # 6 paragraphs total, but only 4 meet the word-count floor -- after excluding
    # the too-short ones, only 4 remain (< MIN_PARAGRAPHS), so no findings.
    assert MIN_PARAGRAPHS == 6, "test assumes the brief's literal MIN_PARAGRAPHS=6"
    baseline = _uniform_baseline(4)
    too_short_1 = _make_fingerprint("pshort1", word_count=MIN_WORDS_PER_PARAGRAPH - 1)
    too_short_2 = _make_fingerprint(
        "pshort2", sentence_length_mean=999.0, word_count=MIN_WORDS_PER_PARAGRAPH - 1
    )

    results = detect_outliers(baseline + [too_short_1, too_short_2])

    assert results == []


def test_too_short_paragraph_itself_never_flagged_even_if_extreme():
    baseline = _uniform_baseline(5)
    extreme_but_too_short = _make_fingerprint(
        "pshort",
        sentence_length_mean=999.0,
        function_word_rate=0.0,
        word_count=MIN_WORDS_PER_PARAGRAPH - 1,
    )

    results = detect_outliers(baseline + [extreme_but_too_short])

    assert results == []
    assert all(r.paragraph_id != "pshort" for r in results)


# ---------------------------------------------------------------------------
# None handling (flesch_reading_ease can be None for degenerate paragraphs)
# ---------------------------------------------------------------------------


def test_none_flesch_reading_ease_does_not_crash_or_flag_as_outlier():
    # 5 baseline paragraphs with a real flesch score, one paragraph identical in
    # every OTHER dimension but with flesch_reading_ease=None. If None were ever
    # silently coerced to 0.0, this paragraph would look like an extreme outlier
    # (0.0 vs. a baseline median of 60.0) and get wrongly flagged.
    baseline = _uniform_baseline(5)
    none_flesch = _make_fingerprint("p006", flesch_reading_ease=None)

    results = detect_outliers(baseline + [none_flesch])

    assert results == []


def test_none_flesch_reading_ease_on_baseline_paragraphs_does_not_crash():
    # All baseline paragraphs degenerate (flesch=None); the shifted paragraph has a
    # real numeric flesch value. Must not crash, and the shift on other dimensions
    # must still be detected.
    baseline = [
        _make_fingerprint(f"p{i:03d}", flesch_reading_ease=None) for i in range(1, 6)
    ]
    shifted = _make_fingerprint(
        "p006",
        sentence_length_mean=45.0,
        function_word_rate=2.0,
        word_length_mean=9.0,  # 3rd deviating dimension (k-of-m rule needs >=3)
        flesch_reading_ease=60.0,
    )

    results = detect_outliers(baseline + [shifted])

    assert [r.paragraph_id for r in results] == ["p006"]


# ---------------------------------------------------------------------------
# Top-3 feature naming: stable, human-readable
# ---------------------------------------------------------------------------


def test_top_deviating_features_are_human_readable_not_raw_field_names():
    baseline = _uniform_baseline(5)
    shifted = _make_fingerprint(
        "p006", sentence_length_mean=45.0, function_word_rate=2.0, word_length_mean=9.0
    )

    results = detect_outliers(baseline + [shifted])

    names = results[0].top_deviating_features
    assert all(isinstance(name, str) for name in names)
    # Human-readable copy uses spaces, not raw snake_case field identifiers.
    for name in names:
        assert "_" not in name
    assert "sentence_length_mean" not in names
    assert "function_word_rate" not in names


def test_top_deviating_features_naming_is_stable_across_repeated_calls():
    baseline = _uniform_baseline(5)
    shifted = _make_fingerprint(
        "p006", sentence_length_mean=45.0, function_word_rate=2.0, word_length_mean=9.0
    )
    fingerprints = baseline + [shifted]

    first = detect_outliers(fingerprints)
    second = detect_outliers(fingerprints)

    assert first == second
    assert first and first[0].paragraph_id == "p006"  # non-trivial: a real flag


# ---------------------------------------------------------------------------
# OutlierStrategy
# ---------------------------------------------------------------------------


def test_local_outlier_factor_strategy_raises_not_implemented():
    baseline = _uniform_baseline(5)
    shifted = _make_fingerprint("p006", sentence_length_mean=45.0)

    with pytest.raises(NotImplementedError):
        detect_outliers(baseline + [shifted], strategy=OutlierStrategy.LOCAL_OUTLIER_FACTOR)


def test_detect_outliers_default_strategy_is_robust_zscore():
    baseline = _uniform_baseline(5)
    shifted = _make_fingerprint(
        "p006", sentence_length_mean=45.0, function_word_rate=2.0, word_length_mean=9.0
    )

    default_call = detect_outliers(baseline + [shifted])
    explicit_call = detect_outliers(baseline + [shifted], strategy=OutlierStrategy.ROBUST_ZSCORE)

    assert default_call == explicit_call
    assert default_call and default_call[0].paragraph_id == "p006"


# ---------------------------------------------------------------------------
# No-outlier case
# ---------------------------------------------------------------------------


def test_fully_uniform_document_has_no_outliers():
    baseline = _uniform_baseline(6)

    results = detect_outliers(baseline)

    assert results == []


def test_two_shifted_paragraphs_both_flagged():
    baseline = _uniform_baseline(5)
    shifted_1 = _make_fingerprint(
        "p006", sentence_length_mean=45.0, function_word_rate=2.0, word_length_mean=9.0
    )
    shifted_2 = _make_fingerprint(
        "p007", academic_vocab_rate=25.0, passive_voice_rate=5.0, lexical_density=0.95
    )

    results = detect_outliers(baseline + [shifted_1, shifted_2])

    assert {r.paragraph_id for r in results} == {"p006", "p007"}
    for result in results:
        assert math.isfinite(result.outlier_score)


def test_too_short_paragraph_does_not_pollute_others_baseline():
    # A too-short paragraph is excluded from the outlier pool entirely -- it must
    # not be allowed to sneak into the "others" baseline used to score paragraphs
    # that DO meet the word-count floor. Give it an extreme, clearly-invalid value
    # on a dimension the real outlier does NOT touch; if it leaked into the pool it
    # would distort the median/MAD baseline for that dimension.
    baseline = _uniform_baseline(5)
    polluter = _make_fingerprint(
        "ppolluter",
        academic_vocab_rate=500.0,
        word_count=MIN_WORDS_PER_PARAGRAPH - 1,
    )
    shifted = _make_fingerprint(
        "p006", sentence_length_mean=45.0, function_word_rate=2.0, word_length_mean=9.0
    )

    results = detect_outliers(baseline + [polluter, shifted])

    assert [r.paragraph_id for r in results] == ["p006"]
    # academic_vocab_rate must not appear -- it was untouched on the real
    # (eligible) paragraphs; only a leaked polluter value could have shifted it.
    assert "academic vocabulary rate" not in results[0].top_deviating_features


# ---------------------------------------------------------------------------
# k-of-m flag rule boundary behavior (OUTLIER_MIN_DEVIATING_FEATURES /
# OUTLIER_PER_FEATURE_Z_THRESHOLD, derived by derive_outlier_threshold.py)
# ---------------------------------------------------------------------------


def test_kofm_rule_constants_are_the_derived_values():
    # Guards the provenance chain: these are the values the derivation script chose.
    # If they change, derive_outlier_threshold.py must be re-run and this updated.
    assert OUTLIER_MIN_DEVIATING_FEATURES == 3
    assert OUTLIER_PER_FEATURE_Z_THRESHOLD == 5.25


def test_paragraph_deviating_on_exactly_two_features_is_not_flagged():
    # Exactly K-1 = 2 dimensions deviate (byte-identical baseline -> the other 14 are
    # z==0). Below the k-of-m floor, so NOT an outlier -- this is the old hair-trigger
    # behavior the derived rule deliberately suppresses.
    baseline = _uniform_baseline(5)
    two_feature_spike = _make_fingerprint(
        "p006", sentence_length_mean=45.0, function_word_rate=2.0
    )

    results = detect_outliers(baseline + [two_feature_spike])

    assert results == []


def test_paragraph_deviating_on_exactly_three_features_is_flagged():
    # Exactly K = 3 dimensions deviate -> at the floor -> flagged.
    baseline = _uniform_baseline(5)
    three_feature_shift = _make_fingerprint(
        "p006",
        sentence_length_mean=45.0,
        function_word_rate=2.0,
        word_length_mean=9.0,
    )

    results = detect_outliers(baseline + [three_feature_shift])

    assert [r.paragraph_id for r in results] == ["p006"]
    assert set(results[0].top_deviating_features) == {
        "sentence length",
        "function word rate",
        "word length",
    }


def test_single_extreme_feature_is_not_flagged():
    # One wildly-extreme dimension is NOT enough on its own (K=3): a single spiking
    # feature was the dominant human false-positive mode before the re-tune.
    baseline = _uniform_baseline(6)
    one_feature_spike = _make_fingerprint("p007", sentence_length_mean=900.0)

    results = detect_outliers(baseline + [one_feature_spike])

    assert results == []


def test_outlier_result_is_a_dataclass_with_expected_fields():
    baseline = _uniform_baseline(5)
    shifted = _make_fingerprint(
        "p006", sentence_length_mean=45.0, function_word_rate=2.0, word_length_mean=9.0
    )

    results = detect_outliers(baseline + [shifted])

    result = results[0]
    assert isinstance(result, OutlierResult)
    assert result.paragraph_id == "p006"
    assert isinstance(result.outlier_score, float)
    assert isinstance(result.top_deviating_features, list)
