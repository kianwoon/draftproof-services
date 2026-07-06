"""Tests for poc/detect_v7/signal_adapter.py.

Run:  cd poc && python -m pytest test_detect_v7_signal_adapter.py -v
"""
from __future__ import annotations

import pytest

from detect_v7.signal_adapter import V7_SIGNAL_NAMES, adapt_paragraph_signals
from detect.criteria import specificity as specificity_criterion
from detect.criteria import style_shift as style_shift_criterion
from detect.criteria import burstiness as burstiness_criterion
from detect.criteria import surprisal as surprisal_criterion
from detect.criteria import repetitive_structure as repetitive_structure_criterion


# A realistic, longer AI-ish paragraph so the real criteria/*.py score() calls
# (which gate on min word/sentence counts) actually produce non-placeholder values.
_AI_LIKE_TEXT = (
    "This demonstrates the importance of effective communication in the modern workplace. "
    "This highlights the significance of clear collaboration among team members. "
    "This shows the value of active listening during difficult conversations. "
    "This suggests that organizations benefit from structured feedback processes. "
    "This demonstrates the importance of trust-building activities for new teams. "
    "This highlights the significance of consistent leadership behavior over time. "
    "This shows the value of transparent decision-making in uncertain situations. "
    "In conclusion, these factors collectively contribute to a healthier workplace culture."
)

# Sentence-level metrics that mimic PredictabilityScanner output: low variance,
# low surprisal -> should read as smooth/AI-like (high value) for both criteria.
_SMOOTH_SENTENCE_METRICS = [
    {"avg_surprisal": 1.8, "sentence": s.strip()}
    for s in _AI_LIKE_TEXT.split(". ") if s.strip()
] * 1  # already >= 3 entries


def _real_criterion_scores() -> dict:
    """Build a criterion_scores dict from actual criteria/*.py score() calls."""
    return {
        "low_specificity": specificity_criterion.score(_AI_LIKE_TEXT),
        "style_shift": style_shift_criterion.score(
            _AI_LIKE_TEXT, sentence_metrics=_SMOOTH_SENTENCE_METRICS
        ),
        "low_burstiness": burstiness_criterion.score(
            _AI_LIKE_TEXT, sentence_metrics=_SMOOTH_SENTENCE_METRICS
        ),
        "low_surprisal": surprisal_criterion.score(
            _AI_LIKE_TEXT, sentence_metrics=_SMOOTH_SENTENCE_METRICS
        ),
        "repetitive_structure": repetitive_structure_criterion.score(_AI_LIKE_TEXT),
    }


def _full_raw_signals(has_comparison_text: bool = False) -> dict:
    return {
        "ai_components": {
            "generic_assertion_risk": 0.62,
            "repeated_sentence_structure_risk": 0.55,
            "detector_agreement_risk": 44.0,  # 0-100 scale, adapter should normalize
        },
        "writing_components": {
            "source_grounding_risk": 0.48,
            "lived_detail_risk": 0.71,
            "citation_weakness_risk": 0.33,
            "formulaic_conclusion_risk": 0.66,
            "signpost_paragraph_risk": 0.40,
            "draft_evolution_jump_risk": 0.28,
        },
        "transformation_features": {
            "human_anchor_score": 0.20,
        },
        "criterion_scores": _real_criterion_scores(),
        "semantic_shape": {"semantic_drift_risk": 0.31},
        "has_comparison_text": has_comparison_text,
    }


def test_all_14_signal_keys_present():
    out = adapt_paragraph_signals(_full_raw_signals())
    assert set(V7_SIGNAL_NAMES) == set(out.keys()) - {"signal_status"}
    assert len(V7_SIGNAL_NAMES) == 14
    for name in V7_SIGNAL_NAMES:
        assert name in out
        assert name in out["signal_status"]


def test_unbuilt_signals_are_none_not_implemented():
    out = adapt_paragraph_signals(_full_raw_signals())
    for name in ("paraphrase_pattern_score", "meaning_preservation_score", "esl_false_positive_risk"):
        assert out[name] is None
        assert out["signal_status"][name] == "not_implemented"


def test_semantic_drift_unavailable_without_comparison_text():
    out = adapt_paragraph_signals(_full_raw_signals(has_comparison_text=False))
    assert out["semantic_drift"] is None
    assert out["signal_status"]["semantic_drift"] == "unavailable_no_comparison_text"


def test_semantic_drift_available_with_comparison_text():
    out = adapt_paragraph_signals(_full_raw_signals(has_comparison_text=True))
    assert out["semantic_drift"] is not None
    assert out["signal_status"]["semantic_drift"] == "ok"
    assert 0.0 <= out["semantic_drift"] <= 1.0


_BUILT_OK_SIGNALS = (
    "specificity_score",
    "grounding_gap",
    "sentence_variance",
    "generic_density",
    "author_voice_absence",
    "sentence_smoothness",
    "local_style_shift",
    "predictable_structure",
    "citation_gap",
    "detector_disagreement",
)


def test_built_signals_are_floats_in_unit_range():
    out = adapt_paragraph_signals(_full_raw_signals())
    for name in _BUILT_OK_SIGNALS:
        value = out[name]
        assert isinstance(value, float), f"{name} expected float, got {type(value)}"
        assert 0.0 <= value <= 1.0, f"{name}={value} not in [0,1]"
        assert out["signal_status"][name] == "ok"


def test_detector_disagreement_normalizes_0_100_scale():
    out = adapt_paragraph_signals(_full_raw_signals())
    # input was 44.0 (0-100 scale) -> expect ~0.44
    assert abs(out["detector_disagreement"] - 0.44) < 1e-6


def test_predictable_structure_is_mean_not_max_of_four_inputs():
    raw = _full_raw_signals()
    out = adapt_paragraph_signals(raw)
    parts = [
        out["signal_status"],  # placeholder to keep structure; not used in calc
    ]
    # Recompute expected mean directly from the same four source fields used by the
    # adapter (repetitive_structure criterion value + 3 writing/ai risk fields).
    rep_val = raw["criterion_scores"]["repetitive_structure"].value
    expected_mean = (
        rep_val
        + raw["ai_components"]["repeated_sentence_structure_risk"]
        + raw["writing_components"]["formulaic_conclusion_risk"]
        + raw["writing_components"]["signpost_paragraph_risk"]
    ) / 4.0
    assert abs(out["predictable_structure"] - expected_mean) < 1e-6
    # Sanity: mean must be <= max of the same four inputs (never exceeds max).
    max_val = max(
        rep_val,
        raw["ai_components"]["repeated_sentence_structure_risk"],
        raw["writing_components"]["formulaic_conclusion_risk"],
        raw["writing_components"]["signpost_paragraph_risk"],
    )
    assert out["predictable_structure"] <= max_val + 1e-9


def test_author_voice_absence_inverts_human_anchor_score():
    raw = _full_raw_signals()
    raw["writing_components"]["lived_detail_risk"] = None  # isolate the anchor term
    del raw["writing_components"]["lived_detail_risk"]
    out = adapt_paragraph_signals(raw)
    # Only human_anchor_score (0.20) contributes -> absence = 1 - 0.20 = 0.80
    assert abs(out["author_voice_absence"] - 0.80) < 1e-6


def test_missing_inputs_yield_none_not_a_fabricated_zero():
    out = adapt_paragraph_signals({})
    for name in _BUILT_OK_SIGNALS:
        assert out[name] is None
        # BUILT signal with absent inputs this run -> "unavailable", NOT
        # "not_implemented" (which is reserved for the unbuilt Phase-1C/2
        # rows so degraded accounting can discriminate).
        assert out["signal_status"][name] == "unavailable"
    assert out["semantic_drift"] is None
    assert out["signal_status"]["semantic_drift"] == "unavailable_no_comparison_text"


def test_unbuilt_signals_stay_not_implemented():
    out = adapt_paragraph_signals({})
    for name in ("paraphrase_pattern_score", "meaning_preservation_score", "esl_false_positive_risk"):
        assert out[name] is None
        assert out["signal_status"][name] == "not_implemented"


def test_empty_raw_signals_dict_does_not_raise():
    out = adapt_paragraph_signals({})
    assert set(out.keys()) - {"signal_status"} == set(V7_SIGNAL_NAMES)
