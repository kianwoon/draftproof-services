"""Tests for the deterministic Critical Thinking Control diagnosis.

The module CONSUMES grounding_diagnosis output, so we build a real diagnosis
from the same fixture used by test_grounding_diagnosis.py and assert the
inverted control math + dimension mapping by hand.
"""
from detect.grounding_diagnosis import diagnose_grounding_gap
from detect.critical_thinking import (
    DIMENSION_LABELS,
    DIMENSION_SIGNALS,
    DIMENSION_WEIGHTS,
    FINDING_TYPE_TO_DIMENSION,
    LEAD_INELIGIBLE_DIMENSIONS,
    MODEL_VERSION,
    SIGNAL_CATEGORY_TO_DIMENSION,
    _band,
    score_critical_thinking,
    score_critical_thinking_per_paragraph,
)


def _ctc_from_signals(values, n=30):
    """Build a CTC result directly from explicit grounding signal gap values."""
    gd = {"signals": [{"key": k, "available": True, "value": v} for k, v in values.items()]}
    return score_critical_thinking(grounding_diagnosis=gd, scored_sentence_count=n)

# Same fixture as test_grounding_diagnosis.py -> the grounding signal gaps are:
#   specificity_gap=60 broad_claim_risk=40  evidence_gap=50 context_gap=30
#   author_anchor_gap=60 human_anchor_gap=80
#   predictability=20 topk_pattern=20 over_balance=20 template_flow=20 structure_reuse=20
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


def _ctc(**over):
    return score_critical_thinking(grounding_diagnosis=_diag(**over),
                                   scored_sentence_count=over.get("n", 30))


def test_dimension_control_is_inverse_of_signal_gap():
    d = _ctc()["dimensions"]
    # control = 100 - mean(gap of mapped grounding signals)
    assert d["specific_context"]["gap"] == 50.0   # mean(60, 40)
    assert d["specific_context"]["control"] == 50.0
    assert d["student_judgement"]["gap"] == 70.0  # mean(60, 80)
    assert d["student_judgement"]["control"] == 30.0
    assert d["reasoning_trail"]["control"] == 80.0   # 100 - mean(20, 20)
    assert d["ai_dependency"]["control"] == 80.0     # 100 - mean(20, 20, 20)
    assert d["evidence_grounding"]["control"] == 60.0  # 100 - mean(50, 30)


def test_weighted_control_score_math():
    c = _ctc()
    # weighted_gap = (20*50 + 20*70 + 15*20 + 10*20 + 10*40)/75 = 3300/75 = 44.0
    assert c["score"] == 56.0
    assert c["band"] == "weak_control"


def test_lead_is_weakest_available_dimension():
    c = _ctc()
    assert c["weakest_dimension"] == "student_judgement"  # control 30 = lowest
    assert c["lead_dimension"] == "student_judgement"
    assert c["lead_dimension_label"] == DIMENSION_LABELS["student_judgement"][0] == "student judgement"


def test_lead_suppressed_when_strong_control():
    # All gaps near zero, all strengths maxed -> strong control, no nagging lead.
    low = {k: 0 for k in _AI}
    loww = {k: (100 if k.endswith("_strength") else 0) for k in _WRITING}
    diag = diagnose_grounding_gap(ai_components=low, writing_components=loww,
                                  transformation_features={"human_anchor_score": 1.0},
                                  scored_sentence_count=30)
    c = score_critical_thinking(grounding_diagnosis=diag, scored_sentence_count=30)
    assert c["band"] == "strong_control"
    assert c["lead_dimension"] is None
    assert c["weakest_dimension"] is not None  # still exposed for transparency


def test_fluent_human_floor_does_not_lead_ai_dependency():
    # REGRESSION GUARD for the false-accusation bug. Documented regime: a fluent,
    # WELL-GROUNDED human essay has predictability/topk at the fluency floor (~75-78)
    # but good grounding. ai_dependency must NOT lead (it no longer reads topk), and
    # the headline must be an anchorable grounding/judgement dimension instead.
    real = {
        "predictability": 75, "topk_pattern": 78, "over_balance": 30,
        "template_flow": 30, "structure_reuse": 30,
        "specificity_gap": 30, "broad_claim_risk": 30,
        "evidence_gap": 35, "context_gap": 30,
        "author_anchor_gap": 40, "human_anchor_gap": 50,
    }
    c = _ctc_from_signals(real)
    assert c["lead_dimension"] != "ai_dependency"
    assert c["lead_dimension"] == "student_judgement"  # weakest anchorable dim
    # ai_dependency ignores predictability/topk -> control reflects over_balance only.
    assert c["dimensions"]["ai_dependency"]["control"] == 70.0


def test_ai_dependency_never_leads_even_when_weakest():
    # Force ai_dependency to be the single weakest dimension; the lead must skip it
    # (transparency bar OK, headline coaching never).
    sig = {
        "over_balance": 90,  # ai_dependency very weak
        "specificity_gap": 20, "broad_claim_risk": 20,
        "author_anchor_gap": 30, "human_anchor_gap": 30,
        "template_flow": 25, "structure_reuse": 25,
        "evidence_gap": 25, "context_gap": 25,
    }
    c = _ctc_from_signals(sig)
    assert c["weakest_dimension"] == "ai_dependency"  # true weakest, exposed
    assert c["lead_dimension"] != "ai_dependency"     # but never the headline


def test_ai_dependency_uses_only_over_balance():
    assert DIMENSION_SIGNALS["ai_dependency"] == ("over_balance",)
    assert "ai_dependency" in LEAD_INELIGIBLE_DIMENSIONS


def test_band_thresholds():
    assert _band(100) == "strong_control"
    assert _band(80) == "strong_control"
    assert _band(79.99) == "acceptable_control"
    assert _band(60) == "acceptable_control"
    assert _band(40) == "weak_control"
    assert _band(20) == "high_dependency"
    assert _band(19.99) == "very_high_dependency"


def test_insufficient_data_abstains_not_zero():
    # No grounding signals -> abstain (None), never a false "very high dependency 0".
    c = score_critical_thinking(grounding_diagnosis={"signals": []})
    assert c["score"] is None
    assert c["band"] == "insufficient_data"
    assert c["lead_dimension"] is None


def test_deterministic_repeatable():
    a = _ctc()
    b = _ctc()
    assert a == b  # byte-identical across runs (no randomness, no LLM)


def test_llm_fields_empty_in_deterministic_core():
    c = _ctc()
    assert c["llm_dimensions"] is None  # Phase 2 fills these behind the kill-switch
    assert c["highlights"] == []


def test_low_coverage_caveat_on_short_doc():
    c = _ctc(n=3)
    assert c["low_coverage"] is True
    assert c["caveat"]


def test_missing_grounding_no_crash():
    c = score_critical_thinking()
    assert c["score"] is None
    assert c["model"] == MODEL_VERSION


def test_per_paragraph_precise_finding_type_mapping():
    fbp = {
        "p001": [{"finding_type": "generic_phrase", "score": 70}],
        "p003": [{"finding_type": "uncited_claim", "score": 60}],
    }
    rows = {r["paragraph_id"]: r for r in score_critical_thinking_per_paragraph(fbp)}
    assert rows["p001"]["dimension"] == "specific_context"
    assert rows["p003"]["dimension"] == "evidence_grounding"
    assert rows["p003"]["action"] == DIMENSION_LABELS["evidence_grounding"][1]


def test_per_paragraph_authorship_risk_never_tags_student_judgement():
    # AI-likeness findings (semantic_drift etc.) are NOT evidence of missing
    # judgement -> they must produce NO per-paragraph tag (same conflation guard
    # as ai_dependency). student_judgement has no per-paragraph source.
    for ft in ("semantic_drift", "similarity_overlap", "semantic_uniformity",
               "discourse_regularity", "high_ai_generation_likelihood"):
        assert score_critical_thinking_per_paragraph({"p001": [{"finding_type": ft, "score": 90}]}) == []
    assert "student_judgement" not in FINDING_TYPE_TO_DIMENSION.values()
    assert "student_judgement" not in SIGNAL_CATEGORY_TO_DIMENSION.values()


def test_per_paragraph_surface_writing_issue_gets_no_tag():
    # PRECISION FIX: grammar/spelling/punctuation/fragment are signal_category
    # "writing_quality" but are NOT a critical-thinking gap -> no tag (must NOT be
    # mislabeled "evidence_grounding — connect this claim to a source").
    fbp = {"p001": [
        {"finding_type": "grammar_issue", "signal_category": "writing_quality", "score": 80},
        {"finding_type": "fragment_sentence", "signal_category": "writing_quality", "score": 60},
    ]}
    assert score_critical_thinking_per_paragraph(fbp) == []


def test_per_paragraph_citation_vs_grammar_split_within_writing_quality():
    # Same coarse bucket (writing_quality), different finding_type -> different fate:
    # uncited_claim tags evidence_grounding; grammar_issue contributes nothing.
    fbp = {"p001": [
        {"finding_type": "grammar_issue", "signal_category": "writing_quality", "score": 90},
        {"finding_type": "uncited_claim", "signal_category": "writing_quality", "score": 40},
    ]}
    rows = score_critical_thinking_per_paragraph(fbp)
    assert len(rows) == 1 and rows[0]["dimension"] == "evidence_grounding"


def test_per_paragraph_ai_dependency_never_leads():
    # A paragraph flagged ONLY on predictability -> ai_dependency, lead-ineligible
    # -> omitted (no false "too AI-dependent" tag).
    fbp = {"p001": [{"finding_type": "high_predictability", "score": 90}]}
    assert score_critical_thinking_per_paragraph(fbp) == []


def test_per_paragraph_signal_category_fallback_excludes_writing_quality():
    # Coarse fallback still works for the unambiguous categories...
    assert score_critical_thinking_per_paragraph(
        {"p001": [{"signal_category": "genericity", "score": 30}]}
    )[0]["dimension"] == "specific_context"
    # ...but NOT for writing_quality (ambiguous) -> no tag without a finding_type.
    assert score_critical_thinking_per_paragraph(
        {"p001": [{"signal_category": "writing_quality", "score": 80}]}
    ) == []


def test_per_paragraph_most_flagged_eligible_dimension_wins():
    fbp = {"p001": [
        {"finding_type": "generic_phrase", "score": 30},
        {"finding_type": "uncited_claim", "score": 80},
        {"finding_type": "high_predictability", "score": 100},  # ai_dependency: ignored for lead
    ]}
    rows = score_critical_thinking_per_paragraph(fbp)
    assert len(rows) == 1 and rows[0]["dimension"] == "evidence_grounding"


def test_per_paragraph_empty_and_unknown():
    assert score_critical_thinking_per_paragraph({}) == []
    assert score_critical_thinking_per_paragraph({"p001": []}) == []
    assert score_critical_thinking_per_paragraph({"p001": [{"finding_type": "mystery"}]}) == []


def test_per_paragraph_deterministic_and_sorted():
    fbp = {
        "p003": [{"finding_type": "generic_phrase", "score": 40}],
        "p001": [{"finding_type": "uncited_claim", "score": 40}],
    }
    a = score_critical_thinking_per_paragraph(fbp)
    assert a == score_critical_thinking_per_paragraph(fbp)
    assert [r["paragraph_id"] for r in a] == ["p001", "p003"]  # sorted by id


def test_per_paragraph_crosswalk_excludes_reasoning_trail():
    # reasoning_trail is document-level only; never a per-paragraph target.
    assert "reasoning_trail" not in FINDING_TYPE_TO_DIMENSION.values()
    assert "reasoning_trail" not in SIGNAL_CATEGORY_TO_DIMENSION.values()
    # writing_quality must NOT be a coarse fallback (ambiguous citation+grammar).
    assert "writing_quality" not in SIGNAL_CATEGORY_TO_DIMENSION


def test_dimension_tables_consistent():
    # Every weighted dimension must have signals + labels (single source of truth).
    assert set(DIMENSION_WEIGHTS) == set(DIMENSION_SIGNALS) == set(DIMENSION_LABELS)
    expected = {"specific_context", "student_judgement", "reasoning_trail",
                "ai_dependency", "evidence_grounding"}
    assert set(DIMENSION_WEIGHTS) == expected
