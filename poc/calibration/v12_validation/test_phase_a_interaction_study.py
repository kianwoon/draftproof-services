# poc/calibration/v12_validation/test_phase_a_interaction_study.py
"""Synthetic-data tests: feature construction, None-skip accounting, gate
verdict in both directions, and reuse of the shared AUC helper."""
from calibration.v12_validation import phase_a_interaction_study as pa


def _row(label, spec, voice_abs, smooth, det, key="k"):
    return {"label": label, "doc_key": key,
            "v7_signals": {"specificity_score": spec, "specificity_student_evidence": None,
                            "author_voice_absence": voice_abs, "grounding_gap": 0.5,
                            "sentence_smoothness": smooth, "signal_status": {}},
            "calibrated_detector_score": det, "has_comparison_text": False}


def test_features_computed_and_none_skipped():
    row = _row("ai_paraphrased", spec=0.8, voice_abs=0.3, smooth=0.9, det=0.6)
    feats = pa.compute_features(row)
    assert abs(feats["specificity_x_smooth"] - 0.8 * 0.9) < 1e-9
    assert abs(feats["voice_presence_x_det"] - 0.7 * 0.6) < 1e-9
    assert feats["spec_student_ev_x_smooth"] is None  # input None -> feature None


def test_gate_pass_on_separable_synthetic():
    rows = ([_row("ai_paraphrased", 0.9, 0.2, 0.9, 0.8, key=f"p{i}") for i in range(10)]
            + [_row("ai_generated_like", 0.1, 0.8, 0.9, 0.8, key=f"g{i}") for i in range(10)]
            + [_row("student_owned", 0.9, 0.2, 0.1, 0.1, key=f"s{i}") for i in range(10)])
    result = pa.run_study(rows, prof_by_key={f"s{i}": ("higher" if i % 2 else "lower") for i in range(10)})
    assert result["gate_verdict"] == "PASS"
    assert result["winner"]["effective_auc_vs_generated"] >= 0.70


def test_gate_fails_closed_when_esl_coverage_missing():
    # Separable rows that would PASS, but prof_by_key={} means the ESL check
    # is unverifiable — the gate must fail CLOSED, not pass vacuously.
    rows = ([_row("ai_paraphrased", 0.9, 0.2, 0.9, 0.8, key=f"p{i}") for i in range(10)]
            + [_row("ai_generated_like", 0.1, 0.8, 0.9, 0.8, key=f"g{i}") for i in range(10)]
            + [_row("student_owned", 0.9, 0.2, 0.1, 0.1, key=f"s{i}") for i in range(10)])
    result = pa.run_study(rows, prof_by_key={})
    assert result["gate_verdict"] == "FAIL"
    assert result["gate_fail_reason"] == "esl_check_unverifiable_insufficient_subgroup_coverage"
    assert result["winner"] is not None  # table data kept intact for inspection
    assert result["esl_coverage"] == {"n_higher": 0, "n_lower": 0}


def test_gate_fails_closed_when_esl_coverage_one_sided():
    # Both-subgroup coverage is required — all-"higher" mapping is still
    # unverifiable in the lower-vs-higher direction.
    rows = ([_row("ai_paraphrased", 0.9, 0.2, 0.9, 0.8, key=f"p{i}") for i in range(10)]
            + [_row("ai_generated_like", 0.1, 0.8, 0.9, 0.8, key=f"g{i}") for i in range(10)]
            + [_row("student_owned", 0.9, 0.2, 0.1, 0.1, key=f"s{i}") for i in range(10)])
    result = pa.run_study(rows, prof_by_key={f"s{i}": "higher" for i in range(10)})
    assert result["gate_verdict"] == "FAIL"
    assert result["gate_fail_reason"] == "esl_check_unverifiable_insufficient_subgroup_coverage"
    assert result["esl_coverage"] == {"n_higher": 10, "n_lower": 0}


def test_gate_fail_on_inseparable_synthetic():
    rows = ([_row("ai_paraphrased", 0.5, 0.5, 0.5, 0.5, key=f"p{i}") for i in range(10)]
            + [_row("ai_generated_like", 0.5, 0.5, 0.5, 0.5, key=f"g{i}") for i in range(10)]
            + [_row("student_owned", 0.5, 0.5, 0.5, 0.5, key=f"s{i}") for i in range(10)])
    result = pa.run_study(rows, prof_by_key={f"s{i}": "higher" for i in range(10)})
    assert result["gate_verdict"] == "FAIL"
