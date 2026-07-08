"""Tests for poc/detect_v7/config.py — the V7 weights.json loader.

Run:  cd poc && python -m pytest test_detect_v7_config.py -v
"""
from __future__ import annotations

import json

import pytest

from detect_v7 import config as v7config

_TOL = 1e-6


@pytest.fixture(autouse=True)
def _reset_v7_weights_env(monkeypatch):
    """Ensure DRAFTPROOF_V7_WEIGHTS_PATH never leaks between tests."""
    monkeypatch.delenv("DRAFTPROOF_V7_WEIGHTS_PATH", raising=False)
    v7config.reload_weights(force=True)
    yield
    monkeypatch.delenv("DRAFTPROOF_V7_WEIGHTS_PATH", raising=False)
    v7config.reload_weights(force=True)


def _sum_weights(pairs: list[tuple[str, float, str]]) -> float:
    return sum(entry[1] for entry in pairs)


def test_weights_json_loads_without_error():
    data = v7config.reload_weights(force=True)
    assert "fusion_weights" in data
    assert "category_weights" in data
    assert "_notes" in data


def test_student_owned_sums_to_one():
    pairs = v7config.get_category_weights("student_owned")
    assert abs(_sum_weights(pairs) - 1.0) < _TOL


def test_ai_assisted_polished_sums_to_one():
    pairs = v7config.get_category_weights("ai_assisted_polished")
    assert abs(_sum_weights(pairs) - 1.0) < _TOL


def test_ai_generated_like_sums_to_one():
    pairs = v7config.get_category_weights("ai_generated_like")
    assert abs(_sum_weights(pairs) - 1.0) < _TOL


def test_ai_paraphrased_with_comparison_sums_to_one():
    pairs = v7config.get_category_weights("ai_paraphrased", has_comparison_text=True)
    assert abs(_sum_weights(pairs) - 1.0) < _TOL


def test_ai_paraphrased_without_comparison_sums_to_one():
    pairs = v7config.get_category_weights("ai_paraphrased", has_comparison_text=False)
    assert abs(_sum_weights(pairs) - 1.0) < _TOL


def test_unknown_category_raises_value_error():
    with pytest.raises(ValueError):
        v7config.get_category_weights("not_a_real_category")


def test_fusion_quick_scan_sums_to_one():
    weights = v7config.get_fusion_weights(["composite"])
    assert abs(sum(weights.values()) - 1.0) < _TOL


def test_fusion_deep_scan_2detector_sums_to_one():
    weights = v7config.get_fusion_weights(["composite", "deberta_large"])
    assert abs(sum(weights.values()) - 1.0) < _TOL


def test_fusion_unsupported_combination_raises_value_error():
    with pytest.raises(ValueError):
        v7config.get_fusion_weights(["composite", "deberta_large", "ou_advacheck"])
    with pytest.raises(ValueError):
        v7config.get_fusion_weights([])
    with pytest.raises(ValueError):
        v7config.get_fusion_weights(["unknown_detector"])


def test_flatness_thresholds_accessor():
    thresholds = v7config.get_flatness_thresholds()
    assert "confidence_low_gap" in thresholds
    assert "mixed_signals_max_category" in thresholds


def test_esl_guard_config_accessor():
    esl = v7config.get_esl_guard_config()
    assert esl["esl_high_threshold"] == 0.70
    assert esl["confidence_cap_value"] == "low"


def test_display_consistency_guard_config_accessor():
    guard = v7config.get_display_consistency_guard_config()
    assert guard["student_owned_contradiction_tiers"] == ["concerning", "strong"]


def test_display_consistency_guard_config_missing_raises(monkeypatch, tmp_path):
    alt_weights = {
        "_notes": {"marker": "no-guard-block-fixture"},
        "fusion_weights": {
            "quick_scan": {"composite": 1.0},
            "deep_scan_2detector": {"deberta_large": 0.5, "composite": 0.5},
            "deep_scan_3detector_inert": {
                "deberta_large": 0.45,
                "composite": 0.35,
                "ou_advacheck": 0.20,
            },
        },
        "category_weights": {
            "student_owned": [{"signal": "specificity_score", "weight": 1.0}],
            "ai_assisted_polished": [
                {"signal": "calibrated_detector_score", "weight": 1.0}
            ],
            "ai_paraphrased_with_comparison": [
                {"signal": "semantic_drift", "weight": 1.0}
            ],
            "ai_paraphrased_without_comparison": [
                {"signal": "semantic_drift", "weight": 1.0}
            ],
            "ai_generated_like": [
                {"signal": "calibrated_detector_score", "weight": 1.0}
            ],
        },
        "ai_assisted_polished_band": {"low": 0.35, "high": 0.70},
        "flatness_thresholds": {
            "confidence_low_gap": 0.10,
            "mixed_signals_max_category": 0.35,
        },
        "esl_guard": {
            "esl_high_threshold": 0.70,
            "ai_generated_damping": 0.85,
            "detector_disagreement_confidence_cap_threshold": 0.25,
            "esl_disagreement_cotrigger_threshold": 0.60,
            "confidence_cap_value": "low",
        },
        "display_bands": {"strong_min": 0.5, "some_min": 0.25, "little_min": 0.10},
    }
    fixture_path = tmp_path / "no_guard_weights.json"
    fixture_path.write_text(json.dumps(alt_weights), encoding="utf-8")
    monkeypatch.setenv("DRAFTPROOF_V7_WEIGHTS_PATH", str(fixture_path))
    v7config.reload_weights(force=True)
    with pytest.raises(KeyError):
        v7config.get_display_consistency_guard_config()


def test_display_bands_accessor():
    bands = v7config.get_display_bands()
    assert bands["strong_min"] > bands["some_min"] > bands["little_min"]


def test_ai_assisted_polished_band_accessor():
    band = v7config.get_ai_assisted_polished_band()
    assert band["low"] < band["high"]


def test_paraphrase_mismatch_normalization_accessor():
    norm = v7config.get_paraphrase_mismatch_normalization()
    # Phase A study percentiles (phase_a_interaction_study.json, 2026-07-08).
    assert abs(norm["p10"] - 0.0589) < _TOL
    assert abs(norm["p90"] - 0.1623) < _TOL
    assert norm["p10"] < norm["p90"]


def test_paraphrase_mismatch_normalization_missing_raises(monkeypatch, tmp_path):
    alt_weights = {
        "_notes": {"marker": "no-norm-block-fixture"},
        "fusion_weights": {
            "quick_scan": {"composite": 1.0},
            "deep_scan_2detector": {"deberta_large": 0.5, "composite": 0.5},
            "deep_scan_3detector_inert": {
                "deberta_large": 0.45,
                "composite": 0.35,
                "ou_advacheck": 0.20,
            },
        },
        "category_weights": {
            "student_owned": [{"signal": "specificity_score", "weight": 1.0}],
            "ai_assisted_polished": [
                {"signal": "calibrated_detector_score", "weight": 1.0}
            ],
            "ai_paraphrased_with_comparison": [
                {"signal": "semantic_drift", "weight": 1.0}
            ],
            "ai_paraphrased_without_comparison": [
                {"signal": "semantic_drift", "weight": 1.0}
            ],
            "ai_generated_like": [
                {"signal": "calibrated_detector_score", "weight": 1.0}
            ],
        },
        "ai_assisted_polished_band": {"low": 0.35, "high": 0.70},
        "flatness_thresholds": {
            "confidence_low_gap": 0.10,
            "mixed_signals_max_category": 0.35,
        },
        "esl_guard": {
            "esl_high_threshold": 0.70,
            "ai_generated_damping": 0.85,
            "detector_disagreement_confidence_cap_threshold": 0.25,
            "esl_disagreement_cotrigger_threshold": 0.60,
            "confidence_cap_value": "low",
        },
        "display_bands": {"strong_min": 0.5, "some_min": 0.25, "little_min": 0.10},
    }
    fixture_path = tmp_path / "no_norm_weights.json"
    fixture_path.write_text(json.dumps(alt_weights), encoding="utf-8")
    monkeypatch.setenv("DRAFTPROOF_V7_WEIGHTS_PATH", str(fixture_path))
    v7config.reload_weights(force=True)
    with pytest.raises(KeyError):
        v7config.get_paraphrase_mismatch_normalization()


def _install_norm_fixture(tmp_path, monkeypatch, p10, p90):
    """Point config at a minimal temp weights file with the given
    paraphrase_mismatch_normalization bounds (the accessor reads only that
    block, so a minimal file suffices)."""
    fixture_path = tmp_path / "norm_bounds_weights.json"
    fixture_path.write_text(
        json.dumps({"paraphrase_mismatch_normalization": {"p10": p10, "p90": p90}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DRAFTPROOF_V7_WEIGHTS_PATH", str(fixture_path))
    v7config.reload_weights(force=True)


def test_paraphrase_mismatch_normalization_equal_bounds_raises(monkeypatch, tmp_path):
    """p10 == p90 would make signal_adapter row 9c's (p90 - p10) divisor zero —
    the accessor must fail loud (ValueError), never let a ZeroDivisionError
    reach the live scan breakdown path."""
    _install_norm_fixture(tmp_path, monkeypatch, 0.1, 0.1)
    with pytest.raises(ValueError):
        v7config.get_paraphrase_mismatch_normalization()


def test_paraphrase_mismatch_normalization_inverted_bounds_raises(monkeypatch, tmp_path):
    _install_norm_fixture(tmp_path, monkeypatch, 0.2, 0.1)
    with pytest.raises(ValueError):
        v7config.get_paraphrase_mismatch_normalization()


def test_paraphrase_mismatch_normalization_non_numeric_raises(monkeypatch, tmp_path):
    _install_norm_fixture(tmp_path, monkeypatch, "0.0589", 0.1623)
    with pytest.raises(ValueError):
        v7config.get_paraphrase_mismatch_normalization()


def test_env_var_override_loads_alternate_file(tmp_path, monkeypatch):
    alt_weights = {
        "_notes": {"marker": "alt-fixture-weights"},
        "fusion_weights": {
            "quick_scan": {"fakespot": 1.0},
            "deep_scan_2detector": {"deberta_large": 0.5, "fakespot": 0.5},
            "deep_scan_3detector_inert": {
                "deberta_large": 0.45,
                "fakespot": 0.35,
                "ou_advacheck": 0.20,
            },
        },
        "category_weights": {
            "student_owned": [{"signal": "specificity_score", "weight": 1.0}],
            "ai_assisted_polished": [
                {"signal": "calibrated_detector_score", "weight": 1.0}
            ],
            "ai_paraphrased_with_comparison": [
                {"signal": "semantic_drift", "weight": 1.0}
            ],
            "ai_paraphrased_without_comparison": [
                {"signal": "semantic_drift", "weight": 1.0}
            ],
            "ai_generated_like": [
                {"signal": "calibrated_detector_score", "weight": 1.0}
            ],
        },
        "ai_assisted_polished_band": {"low": 0.35, "high": 0.70},
        "flatness_thresholds": {
            "confidence_low_gap": 0.10,
            "mixed_signals_max_category": 0.35,
        },
        "esl_guard": {
            "esl_high_threshold": 0.70,
            "ai_generated_damping": 0.85,
            "detector_disagreement_confidence_cap_threshold": 0.25,
            "esl_disagreement_cotrigger_threshold": 0.60,
            "confidence_cap_value": "low",
        },
        "display_bands": {"strong_min": 0.5, "some_min": 0.25, "little_min": 0.10},
    }
    fixture_path = tmp_path / "alt_weights.json"
    fixture_path.write_text(json.dumps(alt_weights), encoding="utf-8")

    monkeypatch.setenv("DRAFTPROOF_V7_WEIGHTS_PATH", str(fixture_path))
    data = v7config.reload_weights(force=True)

    assert data["_notes"]["marker"] == "alt-fixture-weights"
    pairs = v7config.get_category_weights("student_owned")
    assert pairs == [("specificity_score", 1.0, "direct")]


def test_missing_file_raises_file_not_found(tmp_path, monkeypatch):
    nonexistent = tmp_path / "does_not_exist" / "weights.json"
    monkeypatch.setenv("DRAFTPROOF_V7_WEIGHTS_PATH", str(nonexistent))
    with pytest.raises(FileNotFoundError) as exc_info:
        v7config.reload_weights(force=True)
    assert str(nonexistent) in str(exc_info.value)


# ---------------------------------------------------------------------------
# display_fallback accessor (V8 three-way display fallback)
# ---------------------------------------------------------------------------


def _install_display_fallback_fixture(tmp_path, monkeypatch, display_fallback):
    """Deep-copy the real bundled weights, swap in a display_fallback block
    (or drop it entirely when display_fallback is None), write to tmp, and
    point the loader at it."""
    from pathlib import Path

    real = json.loads(
        (Path(v7config.__file__).resolve().parent / "weights.json").read_text(
            encoding="utf-8"
        )
    )
    if display_fallback is None:
        real.pop("display_fallback", None)
    else:
        real["display_fallback"] = display_fallback
    fixture_path = tmp_path / "display_fallback_weights.json"
    fixture_path.write_text(json.dumps(real), encoding="utf-8")
    monkeypatch.setenv("DRAFTPROOF_V7_WEIGHTS_PATH", str(fixture_path))
    v7config.reload_weights(force=True)


def test_display_fallback_config_accessor():
    cfg = v7config.get_display_fallback_config()
    assert cfg["mode"] == "three_way"
    assert cfg["merged_display_category"] == "ai_transformed"
    assert cfg["merged_from"] == ["ai_paraphrased", "ai_generated_like"]


def test_display_fallback_config_missing_raises(tmp_path, monkeypatch):
    _install_display_fallback_fixture(tmp_path, monkeypatch, None)
    with pytest.raises(KeyError):
        v7config.get_display_fallback_config()


def test_display_fallback_four_way_mode_is_valid(tmp_path, monkeypatch):
    _install_display_fallback_fixture(
        tmp_path,
        monkeypatch,
        {
            "mode": "four_way",
            "merged_display_category": "ai_transformed",
            "merged_from": ["ai_paraphrased", "ai_generated_like"],
        },
    )
    cfg = v7config.get_display_fallback_config()
    assert cfg["mode"] == "four_way"


def test_display_fallback_unknown_mode_raises(tmp_path, monkeypatch):
    _install_display_fallback_fixture(
        tmp_path,
        monkeypatch,
        {
            "mode": "five_way",
            "merged_display_category": "ai_transformed",
            "merged_from": ["ai_paraphrased", "ai_generated_like"],
        },
    )
    with pytest.raises(ValueError):
        v7config.get_display_fallback_config()


def test_display_fallback_unknown_merged_from_raises(tmp_path, monkeypatch):
    _install_display_fallback_fixture(
        tmp_path,
        monkeypatch,
        {
            "mode": "three_way",
            "merged_display_category": "ai_transformed",
            "merged_from": ["ai_paraphrased", "not_a_category"],
        },
    )
    with pytest.raises(ValueError):
        v7config.get_display_fallback_config()


def test_display_fallback_merged_name_collision_raises(tmp_path, monkeypatch):
    _install_display_fallback_fixture(
        tmp_path,
        monkeypatch,
        {
            "mode": "three_way",
            "merged_display_category": "student_owned",
            "merged_from": ["ai_paraphrased", "ai_generated_like"],
        },
    )
    with pytest.raises(ValueError):
        v7config.get_display_fallback_config()
