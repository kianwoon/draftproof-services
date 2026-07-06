"""Tests for poc/detect_v7/breakdown_composer.py."""
from __future__ import annotations

from detect_v7 import breakdown_composer, config


def _aggregate(**overrides):
    base = {
        "student_owned": 0.4,
        "ai_assisted_polished": 0.3,
        "ai_paraphrased": 0.2,
        "ai_generated_like": 0.1,
        "paragraph_count": 2,
        "degraded_paragraph_count": 0,
        "primary_category": "student_owned",
    }
    base.update(overrides)
    return base


def _paragraph_result(**overrides):
    base = {
        "student_owned": 0.4,
        "ai_assisted_polished": 0.3,
        "ai_paraphrased": 0.2,
        "ai_generated_like": 0.1,
        "confidence": None,
        "presentation": None,
        "primary_category": "student_owned",
        "degraded": False,
        "missing_signals": [],
    }
    base.update(overrides)
    return base


def test_disclaimer_present_verbatim():
    result = breakdown_composer.compose_authorship_breakdown(_aggregate(), [_paragraph_result()])
    assert result["disclaimer"] == (
        "DraftProof provides authorship clarity signals and writing-risk "
        "analysis. It does not determine misconduct. Final judgement belongs "
        "to the school, teacher, or relevant academic policy."
    )


def test_schema_version_present():
    result = breakdown_composer.compose_authorship_breakdown(_aggregate(), [_paragraph_result()])
    assert result["schema_version"] == "v7_phase1a"


def test_display_mode_always_bands():
    result = breakdown_composer.compose_authorship_breakdown(_aggregate(), [_paragraph_result()])
    assert result["display_mode"] == "bands"


def test_raw_shares_included():
    agg = _aggregate()
    result = breakdown_composer.compose_authorship_breakdown(agg, [_paragraph_result()])
    assert result["document_breakdown_raw"] == {
        "student_owned": 0.4,
        "ai_assisted_polished": 0.3,
        "ai_paraphrased": 0.2,
        "ai_generated_like": 0.1,
    }


def test_band_mapping_edges():
    bands = config.get_display_bands()
    strong_min = bands["strong_min"]
    some_min = bands["some_min"]
    little_min = bands["little_min"]

    agg = _aggregate(
        student_owned=strong_min,       # exactly at strong boundary -> Strong
        ai_assisted_polished=some_min,  # exactly at some boundary -> Some
        ai_paraphrased=little_min,      # exactly at little boundary -> Little
        ai_generated_like=max(0.0, little_min - 0.01),  # below -> None
    )
    result = breakdown_composer.compose_authorship_breakdown(agg, [_paragraph_result()])
    banded = result["document_breakdown_bands"]
    assert banded["student_owned"] == "Strong"
    assert banded["ai_assisted_polished"] == "Some"
    assert banded["ai_paraphrased"] == "Little"
    assert banded["ai_generated_like"] == "None"


def test_band_mapping_just_below_strong_is_some():
    bands = config.get_display_bands()
    strong_min = bands["strong_min"]
    agg = _aggregate(student_owned=strong_min - 0.001)
    result = breakdown_composer.compose_authorship_breakdown(agg, [_paragraph_result()])
    assert result["document_breakdown_bands"]["student_owned"] == "Some"


def test_degraded_ratio_flag_below_threshold_not_degraded():
    agg = _aggregate(paragraph_count=4, degraded_paragraph_count=1)
    paragraphs = [_paragraph_result(degraded=(i == 0)) for i in range(4)]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["degraded_display"] is False


def test_degraded_ratio_flag_above_threshold_is_degraded():
    agg = _aggregate(paragraph_count=4, degraded_paragraph_count=3)
    paragraphs = [_paragraph_result(degraded=(i < 3)) for i in range(4)]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["degraded_display"] is True


def test_mixed_signals_predominant_triggers_degraded_display():
    agg = _aggregate(paragraph_count=4, degraded_paragraph_count=0)
    paragraphs = [_paragraph_result(presentation="mixed_signals") for _ in range(3)] + [
        _paragraph_result()
    ]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["degraded_display"] is True


def test_uncertainty_flag_no_comparison_text():
    agg = _aggregate()
    paragraphs = [_paragraph_result(missing_signals=["semantic_drift"])]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert "paraphrase_without_original_draft" in result["uncertainty_flags"]


def test_no_uncertainty_flag_when_comparison_available():
    agg = _aggregate()
    paragraphs = [_paragraph_result(missing_signals=[])]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["uncertainty_flags"] == []


def test_primary_category_passthrough():
    agg = _aggregate(primary_category="ai_generated_like")
    result = breakdown_composer.compose_authorship_breakdown(agg, [_paragraph_result()])
    assert result["primary_category"] == "ai_generated_like"


def test_paragraph_and_degraded_counts_passthrough():
    agg = _aggregate(paragraph_count=5, degraded_paragraph_count=2)
    paragraphs = [_paragraph_result()] * 5
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["paragraph_count"] == 5
    assert result["degraded_paragraph_count"] == 2


def test_document_confidence_low_if_any_paragraph_low():
    agg = _aggregate()
    paragraphs = [_paragraph_result(confidence=None), _paragraph_result(confidence="low")]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["confidence"] == "low"


def test_primary_category_reliable_true_when_guards_quiet():
    agg = _aggregate(paragraph_count=2, degraded_paragraph_count=0)
    paragraphs = [_paragraph_result(), _paragraph_result()]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["primary_category_reliable"] is True


def test_primary_category_reliable_false_when_confidence_low():
    agg = _aggregate(paragraph_count=2, degraded_paragraph_count=0)
    paragraphs = [_paragraph_result(confidence="low"), _paragraph_result()]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["confidence"] == "low"
    assert result["primary_category_reliable"] is False


def test_primary_category_reliable_false_when_degraded_display():
    agg = _aggregate(paragraph_count=4, degraded_paragraph_count=3)
    paragraphs = [_paragraph_result(degraded=(i < 3)) for i in range(4)]
    result = breakdown_composer.compose_authorship_breakdown(agg, paragraphs)
    assert result["degraded_display"] is True
    assert result["primary_category_reliable"] is False
