from detect.grounding_diagnosis import (
    BUCKET_WEIGHTS,
    DRIVER_LABELS,
    MODEL_VERSION,
    _band,
    _percent,
    diagnose_grounding_gap,
)


# A fixture with hand-computed bucket scores:
#   concrete_grounding = mean(60, 40, 50, 30) = 45.0
#   authorship_trace   = mean(60, 80) = 70.0
#   llm_patterning     = mean(20,20,20,20,20,20) = 20.0
#   language_texture   = mean(50, 40) = 45.0   (abstraction omitted)
#   final = (29*45 + 13*70 + 40*20 + 18*45)/100 = 38.25
_AI = {
    "predictability": 20, "topk_pattern": 20, "repeated_sentence_structure_risk": 20,
    "balanced_hedging_risk": 20, "qualifying_text_ai_density": 50,
}
_WRITING = {
    "lived_detail_risk": 60, "broad_claim_risk": 40, "source_grounding_risk": 50,
    "citation_weakness_risk": 30, "domain_grounding_strength": 70, "source_grounding_strength": 40,
    "paragraph_uniformity_risk": 20, "repeated_starter_risk": 0,
    "signpost_paragraph_risk": 20, "formulaic_conclusion_risk": 0, "paragraph_progression_risk": 0,
}
_FEATURES = {"human_anchor_score": 0.2, "paraphrase_transformation_risk": 0.4, "rewrite_smoothness": 0.2}


def _diag(**over):
    ai = {**_AI, **over.get("ai", {})}
    writing = {**_WRITING, **over.get("writing", {})}
    features = {**_FEATURES, **over.get("features", {})}
    return diagnose_grounding_gap(ai_components=ai, writing_components=writing,
                                  transformation_features=features,
                                  scored_sentence_count=over.get("n", 30))


def test_bucket_weighted_average_math():
    d = _diag()
    b = d["buckets"]
    assert b["concrete_grounding"]["score"] == 45.0
    assert b["authorship_trace"]["score"] == 70.0
    assert b["llm_patterning"]["score"] == 20.0
    assert b["language_texture"]["score"] == 45.0
    assert d["score"] == 38.25
    # contributions sum to the final score
    assert round(sum(x["contribution"] for x in b.values()), 2) == 38.25


def test_primary_driver_is_highest_score_actionable_gap():
    # Headline driver leads with the most SEVERE actionable gap (by gap score), NOT the
    # weight*score contribution. Under the old contribution rule concrete_grounding
    # (29*45=13.05) led; now authorship_trace (raw score 70, the biggest real gap and a
    # fixable content/authorship driver) leads. worst_dimension is unchanged.
    d = _diag()
    assert d["worst_dimension"] == "authorship_trace"
    assert d["primary_driver"] == "authorship_trace"
    assert d["primary_driver_label"] == DRIVER_LABELS["authorship_trace"][0] == "authorship trace"


def test_llm_patterning_never_leads_headline_when_content_present():
    # Regression: surface AI-patterning ("vary structure and phrasing") is NOT a
    # mitigation per the project objective, so it must never LEAD the headline while a
    # content/authorship driver is active -- even when it is the single highest-scoring
    # bucket. Force every llm_patterning signal to 90 (the highest raw score) and assert
    # the headline still leads with the top actionable gap instead.
    d = _diag(
        ai={"predictability": 90, "topk_pattern": 90,
            "repeated_sentence_structure_risk": 90, "balanced_hedging_risk": 90},
        writing={"paragraph_uniformity_risk": 90, "signpost_paragraph_risk": 90},
    )
    assert d["buckets"]["llm_patterning"]["score"] == 90.0
    assert d["worst_dimension"] == "llm_patterning"      # it IS the highest raw score
    assert d["primary_driver"] != "llm_patterning"        # but it does NOT lead the headline
    assert d["primary_driver"] == "authorship_trace"      # the top actionable gap (70) leads


def test_abstraction_omitted_and_renormalized():
    d = _diag()
    assert "abstract_noun_density_unmeasured" in d["limitations"]
    tex = d["buckets"]["language_texture"]
    assert tex["total"] == 3 and tex["available"] == 2  # padding + paraphrase, abstraction missing
    abstraction = next(s for s in d["signals"] if s["key"] == "abstraction")
    assert abstraction["available"] is False


def test_driver_shown_at_some_texture_band():
    # the target-doc regime (~34) is "some_texture" and MUST still surface a driver
    d = _diag()
    assert d["band"] == "some_texture"
    assert d["primary_driver"] is not None


def test_driver_suppressed_only_when_likely_human():
    low = {k: 0 for k in _AI}
    loww = {k: (100 if k.endswith("_strength") else 0) for k in _WRITING}
    d = diagnose_grounding_gap(ai_components=low, writing_components=loww,
                               transformation_features={"human_anchor_score": 1.0},
                               scored_sentence_count=30)
    assert d["band"] == "likely_human"
    assert d["primary_driver"] is None
    assert "well-grounded" in d["primary_driver_label"]


def test_band_thresholds():
    assert _band(0) == "likely_human"
    assert _band(20) == "likely_human"
    assert _band(20.1) == "some_texture"
    assert _band(40) == "some_texture"
    assert _band(60) == "moderate"
    assert _band(80) == "strong"
    assert _band(80.1) == "very_strong"


def test_low_coverage_caveat_on_short_doc():
    d = _diag(n=3)
    assert d["low_coverage"] is True
    assert d["caveat"]  # softened note present when a driver is shown


def test_missing_components_no_crash():
    d = diagnose_grounding_gap()
    assert d["score"] == 0.0
    assert d["primary_driver"] is None
    assert d["model"] == MODEL_VERSION


def test_inputs_not_mutated():
    ai = dict(_AI); writing = dict(_WRITING); features = dict(_FEATURES)
    diagnose_grounding_gap(ai_components=ai, writing_components=writing, transformation_features=features)
    assert ai == _AI and writing == _WRITING and features == _FEATURES


def test_weights_sum_to_100():
    assert sum(BUCKET_WEIGHTS.values()) == 100


def test_percent_scale_is_explicit_at_the_1_0_edge():
    # Regression for the old value<=1.0 heuristic: a genuine 0-100 signal that happens
    # to land at exactly 1.0 (e.g. a real 1% risk from ai_components/writing_components,
    # which report/builder.py always pre-scales to 0-100) must NOT be rescaled to 100.
    assert _percent(1.0, scale="percent") == 1.0
    # A genuine 0-1 fraction signal at exactly 1.0 (e.g. transformation_features'
    # human_anchor_score = 1.0, meaning fully human) legitimately means 100%.
    assert _percent(1.0, scale="fraction") == 100.0
    # scale is mandatory -- no silent guessing.
    import pytest
    with pytest.raises(ValueError):
        _percent(1.0, scale="bogus")


def test_ai_component_at_1_0_is_not_rescaled_to_100():
    # End-to-end: a writing_components value of exactly 1.0 (a real ~1% specificity
    # gap, already on the 0-100 scale per builder.py) must stay ~1, not become 100.
    d = _diag(writing={"lived_detail_risk": 1.0})
    sig = next(s for s in d["signals"] if s["key"] == "specificity_gap")
    assert sig["value"] == 1.0


def test_driver_keys_consistent_across_tables():
    # The 4 bucket keys must be identical across the weight table and the driver
    # label/action table (the single source of truth synced to render.py + i18n).
    assert set(DRIVER_LABELS) == set(BUCKET_WEIGHTS)
    expected = {"concrete_grounding", "authorship_trace", "llm_patterning", "language_texture"}
    assert set(BUCKET_WEIGHTS) == expected
