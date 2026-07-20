#!/usr/bin/env python3
"""Regression tests for the V7-fused AI-likelihood override in transformation
classification (bug: builder passed the stale pre-fusion layer3 score, so the
transformation / turnitin-like panels disagreed with the fused badge).

Runs under pytest (``cd poc && pytest test_transformation_fusion_override.py``)
or directly (``python test_transformation_fusion_override.py``).
"""

from detect.layer3_scoring import Layer3Scorer, build_layer3_input_from_text
from detect.transformation import (
    build_transformation_features,
    classify_transformation_from_scan,
)

# allow-hardcode: inert test-fixture prose fed to the scorer, not a scoring
# oracle / matching list — no detection logic keys off these words.
_TEXT = (
    "The committee reviewed the quarterly figures and noted several trends. "
    "Revenue rose in three of four regions, while costs held steady. "
    "Management proposed a modest reinvestment in the logistics network. "
    "The board asked for a follow-up analysis before the next meeting."
)


def _scored():
    inp = build_layer3_input_from_text(
        _TEXT,
        predictability=0.46,
        topk_pattern=0.60,
        generic_phrase_density=0.10,
        citation_weakness_risk=0.30,
        source_grounding_strength=0.60,
        domain_grounding_strength=0.60,
    )
    res = Layer3Scorer().score(inp)
    return inp, res


def test_override_replaces_ai_likelihood_input():
    """A provided authoritative_ai_likelihood (0-1 fraction) overrides the
    layer3 score and flows through to the downstream calibrated risk."""
    inp, res = _scored()

    ov_low = build_transformation_features(inp, res, authoritative_ai_likelihood=0.10)
    ov_high = build_transformation_features(inp, res, authoritative_ai_likelihood=0.90)

    # The raw AI-likelihood feature is exactly the override value.
    assert ov_low.ai_likelihood == 0.10
    assert ov_high.ai_likelihood == 0.90
    # And the override actually propagates to the calibrated downstream risk.
    assert ov_high.calibrated_ai_risk > ov_low.calibrated_ai_risk

    # classify_transformation_from_scan threads the same override.
    cls_low = classify_transformation_from_scan(inp, res, authoritative_ai_likelihood=0.10)
    cls_high = classify_transformation_from_scan(inp, res, authoritative_ai_likelihood=0.90)
    assert cls_low.features["ai_likelihood"] == 0.10
    assert cls_high.features["ai_likelihood"] == 0.90


def test_none_is_byte_identical_to_legacy():
    """authoritative_ai_likelihood=None (or omitted) reproduces the exact
    pre-fix behavior."""
    inp, res = _scored()

    assert build_transformation_features(inp, res) == build_transformation_features(
        inp, res, authoritative_ai_likelihood=None
    )
    # The overridden result must differ from the legacy path (guards against a
    # no-op wiring that silently ignores the override).
    assert build_transformation_features(
        inp, res, authoritative_ai_likelihood=0.99
    ) != build_transformation_features(inp, res)

    baseline = classify_transformation_from_scan(inp, res)
    with_none = classify_transformation_from_scan(inp, res, authoritative_ai_likelihood=None)
    assert baseline.code == with_none.code
    assert baseline.features == with_none.features


if __name__ == "__main__":
    test_override_replaces_ai_likelihood_input()
    test_none_is_byte_identical_to_legacy()
    print("transformation fusion override tests passed")
