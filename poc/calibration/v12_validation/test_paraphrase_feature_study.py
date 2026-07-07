"""Smoke tests for the Task 6 feature-study math helpers (rank_auc, quantile,
pearson, cosine) on synthetic data — no ML model, no corpus. The full-corpus
run (paraphrase_feature_study.json) is the deliverable, not a test; this file
only proves the statistics are implemented correctly."""
from __future__ import annotations

import math

from calibration.v12_validation.paraphrase_feature_study import (
    cosine,
    evaluate_gate,
    pearson,
    quantile,
    rank_auc,
    stddev,
)


def test_rank_auc_perfect_separation():
    pos = [5, 6, 7]
    neg = [1, 2, 3]
    assert rank_auc(pos, neg) == 1.0


def test_rank_auc_perfect_inverse_separation():
    pos = [1, 2, 3]
    neg = [5, 6, 7]
    assert rank_auc(pos, neg) == 0.0


def test_rank_auc_no_separation_is_half():
    pos = [1, 2, 3, 4]
    neg = [1, 2, 3, 4]
    assert rank_auc(pos, neg) == 0.5


def test_rank_auc_matches_known_value():
    # pos=[2,4], neg=[1,3] -> pairs: 2>1 (1), 2>3 (0), 4>1(1), 4>3(1) => 3/4
    assert rank_auc([2, 4], [1, 3]) == 0.75


def test_rank_auc_empty_group_is_none():
    assert rank_auc([], [1, 2]) is None
    assert rank_auc([1, 2], []) is None


def test_quantile_matches_linear_interpolation():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # numpy default 'linear' method: p90 of 1..10 (0-indexed pos=0.9*9=8.1) -> 9.1
    assert math.isclose(quantile(values, 0.90), 9.1, rel_tol=1e-9)
    assert math.isclose(quantile(values, 0.10), 1.9, rel_tol=1e-9)
    assert quantile(values, 0.5) == 5.5


def test_pearson_perfect_positive_correlation():
    a = [1, 2, 3, 4, 5]
    b = [2, 4, 6, 8, 10]
    assert math.isclose(pearson(a, b), 1.0, rel_tol=1e-9)


def test_pearson_perfect_negative_correlation():
    a = [1, 2, 3, 4, 5]
    b = [10, 8, 6, 4, 2]
    assert math.isclose(pearson(a, b), -1.0, rel_tol=1e-9)


def test_pearson_zero_variance_is_none():
    assert pearson([1, 1, 1], [1, 2, 3]) is None


def test_cosine_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert math.isclose(cosine(v, v), 1.0, rel_tol=1e-9)


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_stddev_matches_population_stddev():
    assert math.isclose(stddev([2, 4, 4, 4, 5, 5, 7, 9]), 2.0, rel_tol=1e-9)


def test_evaluate_gate_passes_when_auc_high_and_esl_low():
    result = {
        "auc_one_vs_one": {
            "adjacent_cosine_mean": {"ai_paraphrased_vs_ai_generated_like": 0.85,
                                      "ai_paraphrased_vs_ai_assisted_polished": 0.60},
            "pairwise_cosine_std": {"ai_paraphrased_vs_ai_generated_like": 0.55,
                                     "ai_paraphrased_vs_ai_assisted_polished": 0.50},
        },
        "esl_subgroup_check": {
            "adjacent_cosine_mean": {"lower_vs_higher_auc": 0.50},
            "pairwise_cosine_std": {"lower_vs_higher_auc": 0.50},
        },
    }
    gate = evaluate_gate(result, auc_gate=0.70, esl_gate=0.60)
    assert gate["gate_passed"] is True
    assert gate["best_feature"] == "adjacent_cosine_mean"


def test_evaluate_gate_fails_when_best_feature_is_covert_esl_detector():
    result = {
        "auc_one_vs_one": {
            "adjacent_cosine_mean": {"ai_paraphrased_vs_ai_generated_like": 0.85,
                                      "ai_paraphrased_vs_ai_assisted_polished": 0.60},
        },
        "esl_subgroup_check": {
            "adjacent_cosine_mean": {"lower_vs_higher_auc": 0.90},
        },
    }
    gate = evaluate_gate(result, auc_gate=0.70, esl_gate=0.60)
    assert gate["gate_passed"] is False


def test_evaluate_gate_fails_when_auc_too_low():
    result = {
        "auc_one_vs_one": {
            "adjacent_cosine_mean": {"ai_paraphrased_vs_ai_generated_like": 0.55,
                                      "ai_paraphrased_vs_ai_assisted_polished": 0.50},
        },
        "esl_subgroup_check": {
            "adjacent_cosine_mean": {"lower_vs_higher_auc": 0.50},
        },
    }
    gate = evaluate_gate(result, auc_gate=0.70, esl_gate=0.60)
    assert gate["gate_passed"] is False
