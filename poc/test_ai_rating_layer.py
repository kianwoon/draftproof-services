#!/usr/bin/env python3
"""Focused tests for the user-facing authorship rating layer."""

from detect.layer3_scoring import (
    Confidence,
    Layer3Input,
    Layer3Scorer,
    QualityTier,
    Tier,
    derive_authorship_rating,
)
from report.render import _authorship_rating_from_badge


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def assert_true(value, message):
    if not value:
        raise AssertionError(message)


cases = [
    (0.10, Tier.GREEN, "human_likely", "Human-Likely"),
    (0.24, Tier.GREEN, "unlikely_ai", "Unlikely AI"),
    (0.39, Tier.AMBER, "possible_ai_assisted", "Possible AI-Assisted"),
    (0.55, Tier.ORANGE, "likely_ai", "Likely AI"),
    (0.72, Tier.RED, "ai_generated_signals", "AI-Generated Signals"),
]

for score, tier, code, label in cases:
    rating = derive_authorship_rating(
        ai_score=score,
        ai_tier=tier,
        writing_quality_score=0.20,
        writing_quality_tier=QualityTier.LOW,
        confidence=Confidence.HIGH,
    )
    assert_equal(rating["code"], code, f"{score} maps to {code}")
    assert_equal(rating["label"], label, f"{score} label")
    assert_equal(rating["is_verdict"], False, "rating is not a verdict")
    assert_true("not proof" in rating["disclaimer"], "rating carries authorship disclaimer")

verified = derive_authorship_rating(
    ai_score=0.05,
    ai_tier=Tier.GREEN,
    writing_quality_score=0.10,
    writing_quality_tier=QualityTier.LOW,
    confidence=Confidence.HIGH,
    verified_ai_provenance=True,
)
assert_equal(verified["code"], "ai_generated", "verified provenance overrides score")
assert_equal(verified["label"], "AI-Generated", "verified provenance label")

low_conf = derive_authorship_rating(
    ai_score=0.41,
    ai_tier=Tier.AMBER,
    writing_quality_score=0.70,
    writing_quality_tier=QualityTier.HIGH_REVIEW,
    confidence=Confidence.LOW,
)
assert_true(low_conf["caution_notes"], "low confidence rating includes caution notes")

humanised = derive_authorship_rating(
    ai_score=0.6315,
    ai_tier=Tier.ORANGE,
    writing_quality_score=0.6761,
    writing_quality_tier=QualityTier.HIGH_REVIEW,
    confidence=Confidence.HIGH,
)
assert_equal(humanised["code"], "ai_generated_signals", "near-red humanised AI profile escalates rating")
assert_true(
    any("Escalated" in note for note in humanised["caution_notes"]),
    "near-red humanised profile includes escalation note",
)

strong_humaniser = derive_authorship_rating(
    ai_score=0.6035,
    ai_tier=Tier.ORANGE,
    writing_quality_score=0.6256,
    writing_quality_tier=QualityTier.REVIEW,
    confidence=Confidence.HIGH,
    ai_components={
        "topk_pattern": 0.8923,
        "generic_assertion_risk": 0.90,
    },
    writing_components={
        "unsupported_claim_risk": 0.90,
        "source_grounding_risk": 0.70,
        "broad_claim_risk": 0.75,
    },
)
assert_equal(
    strong_humaniser["code"],
    "ai_generated_signals",
    "component-aligned humaniser profile escalates rating even below high-review writing tier",
)

render_fallback = _authorship_rating_from_badge({
    "tier": "ORANGE",
    "ai_likelihood_score": 63.15,
    "writing_quality_score": 67.61,
    "writing_quality_tier": "HIGH_REVIEW",
})
assert_equal(
    render_fallback["label"],
    "AI-Generated Signals",
    "detect PDF renderer derives missing authorship rating from score fields",
)

render_component_fallback = _authorship_rating_from_badge({
    "tier": "ORANGE",
    "ai_likelihood_score": 60.35,
    "writing_quality_score": 62.56,
    "ai_components": {
        "topk_pattern": 89.23,
        "generic_assertion_risk": 90.0,
    },
    "writing_components": {
        "unsupported_claim_risk": 90.0,
        "source_grounding_risk": 70.0,
        "broad_claim_risk": 75.0,
    },
})
assert_equal(
    render_component_fallback["label"],
    "AI-Generated Signals",
    "detect PDF renderer uses aligned component evidence for humaniser profile",
)

scored = Layer3Scorer().score(
    Layer3Input(
        predictability=0.58,
        topk_pattern=0.70,
        generic_phrase_density=0.60,
        word_count=900,
        sentence_count=45,
        paragraph_count=8,
    )
)
assert_true(scored.authorship_rating["label"], "scorer result includes authorship rating")
assert_equal(scored.authorship_rating["score"], round(scored.ai_likelihood_score * 100, 2), "rating score mirrors AI score")

template_ai = Layer3Scorer().score(
    Layer3Input(
        predictability=0.4659,
        topk_pattern=0.8697,
        generic_phrase_density=0.0526,
        burstiness_risk=0.25,
        repeated_sentence_structure_risk=0.65,
        generic_assertion_risk=0.90,
        balanced_hedging_risk=0.15,
        broad_claim_risk=0.85,
        lived_detail_risk=0.80,
        citation_weakness_risk=0.50,
        unsupported_claim_risk=0.90,
        source_grounding_strength=0.30,
        domain_grounding_strength=0.30,
        paragraph_progression_risk=0.85,
        paragraph_uniformity_risk=0.65,
        repeated_starter_risk=0.85,
        formulaic_conclusion_risk=0.85,
        word_count=520,
        sentence_count=31,
        paragraph_count=8,
    )
)
assert_equal(template_ai.ai_cluster_name, "template_ai_style", "education AI sample triggers template AI cluster")
assert_true(template_ai.ai_likelihood_score >= 0.58, "template AI cluster floors the AI score above possible-AI band")
assert_equal(template_ai.tier, Tier.ORANGE, "template AI sample escalates to orange tier")
assert_equal(template_ai.authorship_rating["code"], "ai_generated_signals", "template AI sample rates as AI-generated signals")

print("AI rating layer tests passed")
