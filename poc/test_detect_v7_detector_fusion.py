"""Tests for poc/detect_v7/detector_fusion.py."""
from __future__ import annotations

import pytest

from detect_v7 import detector_fusion


def test_single_detector_quick_scan_passthrough():
    score, detail = detector_fusion.compute_calibrated_detector_score({"composite": 0.63})
    assert score == pytest.approx(0.63)
    assert detail["weights_used"] == {"composite": 1.0}
    assert detail["contributions"]["composite"] == pytest.approx(0.63)


def test_two_detector_deep_scan_weighted_sum():
    score, detail = detector_fusion.compute_calibrated_detector_score(
        {"composite": 0.40, "deberta_large": 0.80}
    )
    # weights: deberta_large 0.60, composite 0.40 (from weights.json)
    expected = 0.60 * 0.80 + 0.40 * 0.40
    assert score == pytest.approx(expected)
    assert detail["weights_used"] == {"deberta_large": 0.60, "composite": 0.40}


def test_unsupported_combo_raises_value_error():
    with pytest.raises(ValueError):
        detector_fusion.compute_calibrated_detector_score({"unknown_detector": 0.5})


def test_empty_combo_raises_value_error():
    with pytest.raises(ValueError):
        detector_fusion.compute_calibrated_detector_score({})


def test_three_detector_inert_combo_raises_value_error():
    with pytest.raises(ValueError):
        detector_fusion.compute_calibrated_detector_score(
            {"composite": 0.5, "deberta_large": 0.5, "ou_advacheck": 0.5}
        )


def test_disagreement_none_for_single_detector():
    assert detector_fusion.compute_detector_disagreement({"composite": 0.5}) is None


def test_disagreement_none_for_zero_detectors():
    assert detector_fusion.compute_detector_disagreement({}) is None


def test_disagreement_spread_for_two_detectors():
    disagreement = detector_fusion.compute_detector_disagreement(
        {"composite": 0.30, "deberta_large": 0.85}
    )
    assert disagreement == pytest.approx(0.55)


def test_disagreement_zero_when_scores_match():
    disagreement = detector_fusion.compute_detector_disagreement(
        {"composite": 0.5, "deberta_large": 0.5}
    )
    assert disagreement == pytest.approx(0.0)
