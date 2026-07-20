"""Unit tests for `_extract_genericity` in poc/detect/scoring.py.

Regression coverage for a dead-key bug: the sentence dicts built in
poc/report/builder.py use `"top10_ratio"` (float) and `"risk_label"`
(string like "medium"/"high") while `_dict_risk_to_float`-derived `"risk"`
holds a float. `_extract_genericity` was reading the non-existent
`"top10"` key and testing the float `"risk"` against string labels,
silently zeroing 60% of the formula.
"""
from poc.detect.scoring import _extract_genericity
from poc.report.models import PredictabilitySummary


def _sentence(top10_ratio: float, risk_label: str, risk_float: float) -> dict:
    """Build a sentence dict shaped like poc/report/builder.py's output
    (builder.py:494-499): 'risk' is a float, 'risk_label' is the string,
    'top10_ratio' is the ratio field — never 'top10' or 'risk' as a string.
    """
    return {
        "sentence_id": "s001",
        "sentence": "text",
        "risk_label": risk_label,
        "risk": risk_float,
        "avg_probability": 0.5,
        "avg_surprisal": 1.0,
        "top10_ratio": top10_ratio,
        "top50_ratio": top10_ratio,
        "start_char": 0,
        "end_char": 4,
        "paragraph_id": "p1",
    }


def _summary(sentences, generic_phrases=None):
    return PredictabilitySummary(
        overall_risk=0.5,
        risk_distribution={},
        sentences=sentences,
        style_shifts=[],
        generic_phrases_found=generic_phrases or [],
    )


def test_genericity_uses_top10_ratio_and_risk_label():
    sentences = [
        _sentence(top10_ratio=0.9, risk_label="high", risk_float=0.85),
        _sentence(top10_ratio=0.8, risk_label="medium", risk_float=0.55),
        _sentence(top10_ratio=0.1, risk_label="low", risk_float=0.1),
    ]
    summary = _summary(sentences, generic_phrases=["it is important to note"])

    result = _extract_genericity(summary, findings=[])

    assert result is not None
    assert result > 0.0

    # high_top10: 2/3 sentences have top10_ratio > 0.7
    high_top10 = 2 / 3
    # flagged (medium/high risk_label): 2 sentences, mean top10_ratio = 0.85
    mean_flagged_top10 = (0.9 + 0.8) / 2
    phrase_density = min(1 / 8.0, 1.0)
    expected = 0.40 * phrase_density + 0.30 * high_top10 + 0.30 * mean_flagged_top10

    assert result == expected


def test_genericity_nonzero_even_without_generic_phrases():
    """Before the fix, with no matched phrases, top10/risk terms read dead
    keys and always returned 0 regardless of sentence content. Now the
    top10_ratio/risk_label-derived terms alone must produce a non-zero score.
    """
    sentences = [
        _sentence(top10_ratio=0.95, risk_label="high", risk_float=0.9),
        _sentence(top10_ratio=0.92, risk_label="high", risk_float=0.9),
    ]
    summary = _summary(sentences, generic_phrases=[])

    result = _extract_genericity(summary, findings=[])

    assert result is not None
    assert result > 0.5  # both high_top10 and mean_flagged_top10 terms fire


def test_genericity_reading_broken_keys_would_be_zero():
    """Sanity check that pins the shape of the bug: reading the old,
    non-existent keys ('top10' as float ratio, 'risk' as a string label)
    on these realistic dicts yields 0 for both terms — confirming the
    fixed function now diverges from that broken behavior.
    """
    sentences = [
        _sentence(top10_ratio=0.95, risk_label="high", risk_float=0.9),
        _sentence(top10_ratio=0.92, risk_label="high", risk_float=0.9),
    ]

    old_high_top10 = sum(1 for s in sentences if s.get("top10", 0) > 0.7) / len(sentences)
    old_flagged = [s for s in sentences if s.get("risk", "") in ("medium", "high")]

    assert old_high_top10 == 0.0
    assert old_flagged == []
