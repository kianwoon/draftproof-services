"""Tests for poc/detect_v7/category_scoring.py."""
from __future__ import annotations

import pytest

from detect_v7 import category_scoring, config


def _full_signals(**overrides) -> dict:
    """A v7_signals dict with every non-comparison-gated signal populated,
    so no category is degraded unless a test explicitly knocks one out.
    Values are all 0.5 baseline unless overridden.
    """
    base = {
        "specificity_score": 0.5,
        # Detector-gated specificity split (2026-07-08): student_owned now
        # weights specificity_student_evidence and ai_generated_like weights
        # specificity_ai_evidence (see weights.json _notes.category_weights_tuning).
        "specificity_student_evidence": 0.5,
        "specificity_ai_evidence": 0.5,
        "grounding_gap": 0.5,
        "sentence_variance": 0.5,
        "generic_density": 0.5,
        "author_voice_absence": 0.5,
        "sentence_smoothness": 0.5,
        "local_style_shift": 0.5,
        "semantic_drift": 0.5,
        "paraphrase_pattern_score": 0.5,
        "meaning_preservation_score": 0.5,
        "predictable_structure": 0.5,
        "citation_gap": 0.5,
        "detector_disagreement": 0.1,
        "esl_false_positive_risk": 0.1,
        "signal_status": {},
    }
    base.update(overrides)
    return base


def test_normal_case_sums_to_one():
    signals = _full_signals()
    result = category_scoring.score_paragraph(signals, calibrated_detector_score=0.5)
    total = (
        result["student_owned"]
        + result["ai_assisted_polished"]
        + result["ai_paraphrased"]
        + result["ai_generated_like"]
    )
    assert total == pytest.approx(1.0)
    assert result["degraded"] is False
    assert result["missing_signals"] == []


def test_missing_signal_sets_degraded_and_lists_signal():
    signals = _full_signals(specificity_score=None)
    result = category_scoring.score_paragraph(signals, calibrated_detector_score=0.5)
    assert result["degraded"] is True
    assert "specificity_score" in result["missing_signals"]
    total = (
        result["student_owned"]
        + result["ai_assisted_polished"]
        + result["ai_paraphrased"]
        + result["ai_generated_like"]
    )
    assert total == pytest.approx(1.0)


def test_missing_signal_without_comparison_text_flags_but_not_crash():
    # meaning_preservation_score / paraphrase_pattern_score are unbuilt (None)
    # per signal_adapter; without comparison text they're expected-missing
    # but must still be tracked, not silently zeroed.
    signals = _full_signals(
        meaning_preservation_score=None,
        paraphrase_pattern_score=None,
        semantic_drift=None,
    )
    result = category_scoring.score_paragraph(
        signals, calibrated_detector_score=0.5, has_comparison_text=False
    )
    assert result["degraded"] is True
    assert "paraphrase_pattern_score" in result["missing_signals"]


def test_all_signals_missing_stays_normalized_and_degraded():
    # calibrated_detector_score itself can never be None (it's a required
    # positional arg, not looked up in v7_signals), so a true all-zero raw
    # total can't be constructed via v7_signals alone. This test instead
    # exercises heavy degradation (every v7_signals value missing) and
    # confirms the output still normalizes to 1.0 and reports degraded=True
    # rather than crashing — the same resilience property the zero-total
    # equal-share guard exists for.
    all_none = {name: None for name in _full_signals().keys() if name != "signal_status"}
    all_none["signal_status"] = {}
    result = category_scoring.score_paragraph(all_none, calibrated_detector_score=0.0)
    total = (
        result["student_owned"]
        + result["ai_assisted_polished"]
        + result["ai_paraphrased"]
        + result["ai_generated_like"]
    )
    assert total == pytest.approx(1.0)
    assert result["degraded"] is True


def test_flatness_mixed_signals_near_uniform_distribution():
    # Construct signals that drive the 4 categories to near-uniform shares
    # (spec's own 19/26/27/28%-ish example). calibrated_detector_score is the
    # dominant term (weight 0.25-0.35) in every category with varying
    # direction, so picking a mid-value plus mid-value signals elsewhere
    # tends to flatten the distribution since 'direct' and 'inverse' terms
    # roughly cancel across categories when all inputs sit near 0.5.
    signals = _full_signals(
        specificity_score=0.5,
        grounding_gap=0.5,
        sentence_variance=0.5,
        generic_density=0.5,
        author_voice_absence=0.5,
        sentence_smoothness=0.5,
        local_style_shift=0.5,
        semantic_drift=0.5,
        paraphrase_pattern_score=0.5,
        meaning_preservation_score=0.5,
        predictable_structure=0.5,
        citation_gap=0.5,
        detector_disagreement=0.1,
    )
    result = category_scoring.score_paragraph(
        signals, calibrated_detector_score=0.5, has_comparison_text=True
    )
    shares = [
        result["student_owned"],
        result["ai_assisted_polished"],
        result["ai_paraphrased"],
        result["ai_generated_like"],
    ]
    assert max(shares) < 0.35 + 1e-6 or result["presentation"] == "mixed_signals"
    # With all-0.5 inputs the distribution should indeed be flat enough to
    # trigger mixed_signals per the configured threshold.
    assert result["presentation"] == "mixed_signals"


def test_esl_guard_damps_ai_generated_like():
    signals = _full_signals(detector_disagreement=0.05)
    esl_cfg = config.get_esl_guard_config()
    high_esl = esl_cfg["esl_high_threshold"] + 0.05

    undamped = category_scoring.score_paragraph(
        signals, calibrated_detector_score=0.9, esl_score=None
    )
    damped = category_scoring.score_paragraph(
        signals, calibrated_detector_score=0.9, esl_score=high_esl
    )
    assert damped["ai_generated_like"] < undamped["ai_generated_like"]


def test_esl_guard_disagreement_alone_caps_confidence():
    esl_cfg = config.get_esl_guard_config()
    high_disagreement = esl_cfg["detector_disagreement_confidence_cap_threshold"] + 0.1
    signals = _full_signals(detector_disagreement=high_disagreement)
    result = category_scoring.score_paragraph(
        signals, calibrated_detector_score=0.5, esl_score=None
    )
    assert result["confidence"] == esl_cfg["confidence_cap_value"]


def test_esl_guard_cotrigger_forces_confidence_cap():
    esl_cfg = config.get_esl_guard_config()
    high_disagreement = esl_cfg["detector_disagreement_confidence_cap_threshold"] + 0.1
    high_esl = esl_cfg["esl_disagreement_cotrigger_threshold"] + 0.1
    # Use lopsided signals so the flatness gap would normally NOT force
    # confidence=low, to prove the ESL co-trigger is what forces the cap.
    signals = _full_signals(
        detector_disagreement=high_disagreement,
        specificity_score=0.95,
        grounding_gap=0.05,
        generic_density=0.05,
        author_voice_absence=0.05,
    )
    result = category_scoring.score_paragraph(
        signals, calibrated_detector_score=0.05, esl_score=high_esl
    )
    assert result["confidence"] == esl_cfg["confidence_cap_value"]


def test_midrange_direction_peaks_inside_band_lower_outside():
    band = config.get_ai_assisted_polished_band()
    mid_value = (band["low"] + band["high"]) / 2.0

    def result_for(calibrated_score: float) -> dict:
        signals = _full_signals(detector_disagreement=0.05)
        return category_scoring.score_paragraph(
            signals, calibrated_detector_score=calibrated_score
        )

    inside = result_for(mid_value)
    below = result_for(0.0)
    above = result_for(1.0)

    assert inside["ai_assisted_polished"] > below["ai_assisted_polished"]
    assert inside["ai_assisted_polished"] > above["ai_assisted_polished"]


def test_midrange_helper_plateau_and_taper_shape():
    band = config.get_ai_assisted_polished_band()
    low, high = band["low"], band["high"]
    # Inside the band: full plateau.
    assert category_scoring._midrange(low, low, high) == pytest.approx(1.0)
    assert category_scoring._midrange(high, low, high) == pytest.approx(1.0)
    assert category_scoring._midrange((low + high) / 2, low, high) == pytest.approx(1.0)
    # At the axis endpoints: fully tapered to 0.
    assert category_scoring._midrange(0.0, low, high) == pytest.approx(0.0)
    assert category_scoring._midrange(1.0, low, high) == pytest.approx(0.0)
    # Strictly between endpoint and band edge: partial value, monotonic taper.
    below_val = category_scoring._midrange(low / 2, low, high)
    above_val = category_scoring._midrange((high + 1.0) / 2, low, high)
    assert 0.0 < below_val < 1.0
    assert 0.0 < above_val < 1.0


def test_unknown_category_raises_value_error():
    with pytest.raises(ValueError):
        config.get_category_weights("not_a_real_category")


def test_by_design_absent_signals_do_not_set_degraded():
    # The unbuilt Phase-1C/2 signals (status "not_implemented") and the
    # no-comparison-text paraphrase path are roadmap facts, identical for
    # every document — they must land in missing_signals WITHOUT forcing
    # degraded=True, or degraded is a constant and the downstream
    # degraded_display / primary_category_reliable guards can't discriminate.
    signals = _full_signals(
        paraphrase_pattern_score=None,
        meaning_preservation_score=None,
        semantic_drift=None,
        signal_status={
            "paraphrase_pattern_score": "not_implemented",
            "meaning_preservation_score": "not_implemented",
            "semantic_drift": "unavailable_no_comparison_text",
        },
    )
    result = category_scoring.score_paragraph(signals, calibrated_detector_score=0.5)
    assert "paraphrase_pattern_score" in result["missing_signals"]
    assert result["degraded"] is False


def test_built_signal_unavailable_sets_degraded():
    signals = _full_signals(
        specificity_score=None,
        signal_status={"specificity_score": "unavailable"},
    )
    result = category_scoring.score_paragraph(signals, calibrated_detector_score=0.5)
    assert "specificity_score" in result["missing_signals"]
    assert result["degraded"] is True


def test_unknown_status_is_conservatively_degrading():
    signals = _full_signals(grounding_gap=None, signal_status={})
    result = category_scoring.score_paragraph(signals, calibrated_detector_score=0.5)
    assert result["degraded"] is True
