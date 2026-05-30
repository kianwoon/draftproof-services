"""External-detector likelihood estimate (the honest 'what Turnitin/GPTZero will say' number).

Additive only -- it never changes DraftProof's tier or score; it sets expectations so users are
not blindsided when a strict third-party site rates an LLM rewrite far higher than DraftProof.
"""

from __future__ import annotations

from poc.detect.layer3_scoring import estimate_external_detector_likelihood


def test_high_raw_topk_reads_high():
    est = estimate_external_detector_likelihood(
        {"topk_pattern_raw": 68, "predictability": 41, "burstiness_risk": 25}
    )
    assert est["band"] == "high"
    assert est["score"] >= 45


def test_low_signals_read_low():
    est = estimate_external_detector_likelihood(
        {"topk_pattern_raw": 20, "predictability": 15, "burstiness_risk": 10}
    )
    assert est["band"] == "low"


def test_falls_back_to_topk_pattern_when_raw_absent():
    est = estimate_external_detector_likelihood(
        {"topk_pattern": 70, "predictability": 30, "burstiness_risk": 20}
    )
    assert est["band"] == "high"  # topk 70 >= 65 gate


def test_shape_and_caveat_present_even_when_empty():
    est = estimate_external_detector_likelihood({})
    assert set(("score", "band", "note")).issubset(est)
    assert est["band"] == "low"
    assert "third-party" in est["note"]
