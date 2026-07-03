"""Tests for poc/detect_v7/aggregate.py — word-count-weighted document
aggregation of per-paragraph Authorship Clarity Breakdown category shares.
"""
from __future__ import annotations

import pytest

from detect_v7.aggregate import aggregate_document


def _paragraph(student_owned, ai_assisted, ai_paraphrased, ai_generated, degraded=False):
    return {
        "student_owned": student_owned,
        "ai_assisted_polished": ai_assisted,
        "ai_paraphrased": ai_paraphrased,
        "ai_generated_like": ai_generated,
        "confidence": None,
        "presentation": None,
        "primary_category": "student_owned",
        "degraded": degraded,
        "missing_signals": [],
    }


def test_single_paragraph_passthrough():
    p = _paragraph(0.7, 0.1, 0.1, 0.1)
    result = aggregate_document([p], [100])
    assert result["student_owned"] == pytest.approx(0.7)
    assert result["ai_assisted_polished"] == pytest.approx(0.1)
    assert result["ai_paraphrased"] == pytest.approx(0.1)
    assert result["ai_generated_like"] == pytest.approx(0.1)
    assert result["paragraph_count"] == 1
    assert result["degraded_paragraph_count"] == 0
    assert result["primary_category"] == "student_owned"


def test_word_count_weighting_differs_from_naive_mean():
    # Paragraph A: short (10 words), heavily AI-generated.
    # Paragraph B: long (990 words), heavily student-owned.
    p_a = _paragraph(0.0, 0.0, 0.0, 1.0)
    p_b = _paragraph(1.0, 0.0, 0.0, 0.0)

    weighted = aggregate_document([p_a, p_b], [10, 990])
    naive_mean = {
        "student_owned": (0.0 + 1.0) / 2,
        "ai_generated_like": (1.0 + 0.0) / 2,
    }

    # Weighted result should be dominated by the long paragraph (student_owned),
    # not the 50/50 naive mean.
    assert weighted["student_owned"] == pytest.approx(0.99)
    assert weighted["ai_generated_like"] == pytest.approx(0.01)
    assert weighted["student_owned"] != pytest.approx(naive_mean["student_owned"])
    assert weighted["ai_generated_like"] != pytest.approx(naive_mean["ai_generated_like"])


def test_degraded_paragraph_count():
    p1 = _paragraph(0.5, 0.2, 0.2, 0.1, degraded=True)
    p2 = _paragraph(0.4, 0.3, 0.2, 0.1, degraded=False)
    p3 = _paragraph(0.3, 0.3, 0.2, 0.2, degraded=True)
    result = aggregate_document([p1, p2, p3], [50, 50, 50])
    assert result["degraded_paragraph_count"] == 2
    assert result["paragraph_count"] == 3


def test_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        aggregate_document([], [])


def test_mismatched_lengths_raises():
    p = _paragraph(0.5, 0.2, 0.2, 0.1)
    with pytest.raises(ValueError, match="same length"):
        aggregate_document([p, p], [100])


def test_zero_total_words_raises():
    p1 = _paragraph(0.5, 0.2, 0.2, 0.1)
    p2 = _paragraph(0.4, 0.3, 0.2, 0.1)
    with pytest.raises(ValueError, match="zero"):
        aggregate_document([p1, p2], [0, 0])


def test_negative_total_words_raises():
    p = _paragraph(0.5, 0.2, 0.2, 0.1)
    with pytest.raises(ValueError, match="zero"):
        aggregate_document([p], [-5])


def test_shares_sum_to_one():
    p1 = _paragraph(0.6, 0.1, 0.1, 0.2)
    p2 = _paragraph(0.2, 0.4, 0.3, 0.1)
    result = aggregate_document([p1, p2], [37, 214])
    total = (
        result["student_owned"]
        + result["ai_assisted_polished"]
        + result["ai_paraphrased"]
        + result["ai_generated_like"]
    )
    assert total == pytest.approx(1.0)


def test_primary_category_is_argmax():
    p1 = _paragraph(0.1, 0.1, 0.1, 0.7)
    p2 = _paragraph(0.1, 0.1, 0.1, 0.7)
    result = aggregate_document([p1, p2], [50, 50])
    assert result["primary_category"] == "ai_generated_like"
