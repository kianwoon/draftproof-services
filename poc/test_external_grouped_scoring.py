from detect.external_grouped_scoring import (
    MODEL_VERSION,
    estimate_external_grouped_score,
    grouped_external_score_from_groups,
)


def test_grouped_formula_matches_contract_example():
    # v2 weights: 0.15*72 + 0.25*65 + 0.40*78 + 0.20*35 = 65.25
    assert grouped_external_score_from_groups(
        probability_shape_risk=72,
        detector_agreement_risk=65,
        writing_pattern_risk=78,
        grounding_gap_risk=35,
    ) == 65.25


def test_higher_topk_increases_probability_group_and_score():
    base = estimate_external_grouped_score(
        sentences=[{"avg_surprisal": 3.2, "avg_probability": 0.08, "top10_ratio": 0.30, "top50_ratio": 0.55}] * 4,
        ai_components={"topk_pattern_raw": 30, "predictability": 30, "burstiness_risk": 20},
        writing_components={"broad_claim_risk": 20, "lived_detail_risk": 40, "citation_weakness_risk": 40},
        transformation_features={"signal_agreement_score": 0.30},
    )
    higher = estimate_external_grouped_score(
        sentences=[{"avg_surprisal": 3.2, "avg_probability": 0.08, "top10_ratio": 0.80, "top50_ratio": 0.90}] * 4,
        ai_components={"topk_pattern_raw": 80, "predictability": 30, "burstiness_risk": 20},
        writing_components={"broad_claim_risk": 20, "lived_detail_risk": 40, "citation_weakness_risk": 40},
        transformation_features={"signal_agreement_score": 0.30},
    )
    assert higher["groups"]["probability_shape_risk"] > base["groups"]["probability_shape_risk"]
    assert higher["score"] > base["score"]


def test_stronger_grounding_lowers_grounding_gap_and_score():
    weak = estimate_external_grouped_score(
        ai_components={"topk_pattern_raw": 60, "predictability": 50, "burstiness_risk": 40},
        writing_components={
            "broad_claim_risk": 45,
            "lived_detail_risk": 90,
            "citation_weakness_risk": 90,
            "unsupported_claim_risk": 90,
            "source_grounding_risk": 90,
        },
        transformation_features={"signal_agreement_score": 0.45},
    )
    strong = estimate_external_grouped_score(
        ai_components={"topk_pattern_raw": 60, "predictability": 50, "burstiness_risk": 40},
        writing_components={
            "broad_claim_risk": 45,
            "lived_detail_risk": 10,
            "citation_weakness_risk": 10,
            "unsupported_claim_risk": 10,
            "source_grounding_risk": 10,
            "domain_grounding_strength": 85,
        },
        transformation_features={"signal_agreement_score": 0.45, "human_anchor_score": 0.80},
    )
    assert strong["groups"]["grounding_gap_risk"] < weak["groups"]["grounding_gap_risk"]
    assert strong["score"] < weak["score"]


def test_missing_future_detectors_are_unavailable_not_zero_or_fake():
    out = estimate_external_grouped_score(
        ai_components={"topk_pattern_raw": 50, "predictability": 40},
        transformation_features={"signal_agreement_score": 0.40},
    )
    unavailable = {s["key"]: s for s in out["signals"] if not s["available"]}
    assert unavailable["fast_detectgpt_curvature"]["note"] == "unavailable_in_v1"
    assert unavailable["fine_tuned_classifier"]["note"] == "unavailable_in_v1"
    assert out["coverage"]["detector_signals"] == {"available": 1, "total": 3}


def test_legacy_estimates_are_attached_as_audit_alternates():
    legacy_segment = {"score": 13.4, "band": "low", "model": "segment_fraction_v1"}
    legacy_likelihood = {"score": 59.8, "band": "high"}
    out = estimate_external_grouped_score(
        ai_components={"topk_pattern_raw": 79.43, "predictability": 44.49, "burstiness_risk": 25},
        writing_components={"broad_claim_risk": 20, "lived_detail_risk": 20, "citation_weakness_risk": 20},
        transformation_features={"signal_agreement_score": 0.30},
        legacy_segment_fraction=legacy_segment,
        legacy_likelihood=legacy_likelihood,
    )
    assert out["model"] == MODEL_VERSION
    assert out["alternates"]["legacy_segment_fraction"] == legacy_segment
    assert out["alternates"]["legacy_likelihood"] == legacy_likelihood


def test_topk_heavy_case_no_longer_becomes_sixty_from_likelihood_alone():
    out = estimate_external_grouped_score(
        sentences=[{"avg_surprisal": 3.2, "avg_probability": 0.08, "top10_ratio": 0.79, "top50_ratio": 0.88}] * 4,
        ai_components={
            "topk_pattern_raw": 79.43,
            "topk_calibrated_risk": 43.885,
            "predictability": 44.49,
            "burstiness_risk": 25.0,
            "generic_phrase_density": 10,
            "repeated_sentence_structure_risk": 15,
            "qualifying_text_ai_density": 20,
        },
        writing_components={
            "broad_claim_risk": 20,
            "paragraph_progression_risk": 20,
            "paragraph_uniformity_risk": 15,
            "lived_detail_risk": 20,
            "citation_weakness_risk": 20,
            "unsupported_claim_risk": 20,
            "source_grounding_risk": 20,
            "domain_grounding_strength": 80,
        },
        transformation_features={
            "signal_agreement_score": 0.30,
            "rewrite_smoothness": 0.20,
            "human_anchor_score": 0.75,
        },
        legacy_likelihood={"score": 59.8, "band": "high"},
    )
    assert out["score"] < 50
    assert out["band"] == "elevated"
    assert out["alternates"]["legacy_likelihood"]["score"] == 59.8


# ─── deep-scan flag-mass lens (2026-07-14) ────────────────────────────────────
# The external flag-risk lens gains a desklib-informed lift: when the raw
# perplexity tier is HIGH and the (post-sync) desklib tile flags >= the
# calibrated share of sentences, the external estimate is lifted to at least
# the elevated band — a WARNING about external-detector behavior, never the
# authorship tier. Operating point measured 2026-07-14 on SCoCESLE-272 +
# retune AI-91: ESL over-warn 17.3%, AI catch 63.7%, GPT-5.6 catch 81%,
# catches live scan d449aca9 (tile 43%, raw high, base estimate 14.3/low).

def _base_estimate(score=14.3):
    from detect.external_grouped_scoring import _band
    return {"score": score, "band": _band(score), "model": "external_grouped_v3",
            "confidence": "high", "note": "existing note", "groups": {}, "signals": []}


def test_flag_mass_lifts_low_estimate_to_elevated():
    from detect.external_grouped_scoring import apply_deep_scan_flag_mass, ELEVATED_BAND_MIN

    out = apply_deep_scan_flag_mass(_base_estimate(14.3), tile_pct=43, raw_tier="high")
    assert out["band"] == "elevated"
    assert out["score"] >= ELEVATED_BAND_MIN
    prov = out.get("deep_scan_flag_mass")
    assert prov and prov["applied"] is True and prov["tile_pct"] == 43
    assert prov["esl_overwarn_pct"] > 0  # measured, disclosed
    assert "over-flag" in out["note"] or "over-warn" in out["note"]


def test_flag_mass_no_trigger_below_tile_cutoff_or_without_raw_high():
    from detect.external_grouped_scoring import apply_deep_scan_flag_mass

    base = _base_estimate(14.3)
    out = apply_deep_scan_flag_mass(dict(base), tile_pct=25, raw_tier="high")
    assert out["score"] == base["score"] and out["band"] == base["band"]
    assert "deep_scan_flag_mass" not in out
    out2 = apply_deep_scan_flag_mass(dict(base), tile_pct=80, raw_tier="low")
    assert out2["score"] == base["score"] and "deep_scan_flag_mass" not in out2


def test_flag_mass_never_lowers_an_already_high_estimate():
    from detect.external_grouped_scoring import apply_deep_scan_flag_mass

    out = apply_deep_scan_flag_mass(_base_estimate(60.0), tile_pct=55, raw_tier="high")
    assert out["score"] == 60.0 and out["band"] == "high"
    assert out["deep_scan_flag_mass"]["applied"] is True  # provenance still recorded


def test_flag_mass_fails_open_on_malformed_inputs():
    from detect.external_grouped_scoring import apply_deep_scan_flag_mass

    assert apply_deep_scan_flag_mass(None, tile_pct=50, raw_tier="high") is None
    base = _base_estimate()
    assert apply_deep_scan_flag_mass(dict(base), tile_pct=None, raw_tier="high") == base
    assert apply_deep_scan_flag_mass(dict(base), tile_pct=50, raw_tier=None) == base


def test_flag_mass_calibration_artifact_is_committed_and_sane():
    import json
    from pathlib import Path
    art = json.loads((Path(__file__).parent / "calibration" / "flag_mass_lens_baseline.json").read_text())
    assert art["trigger"]["raw_tier"] == "high"
    assert 30 <= art["trigger"]["tile_pct_min"] <= 60
    assert 0 < art["measured"]["esl_overwarn_pct"] < 25
