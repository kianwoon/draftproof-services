"""Tests for the deterministic, additive Submission-risk composer.

score_submission_risk is a pure function over signals computed upstream, so we
feed it minimal synthetic inputs (a fake critical_thinking_control dict, an
ai_tier, and axis_scores) and assert the layer blend + axis mapping by hand.
"""
from detect.submission_risk import (
    AXIS_LEVEL_RISK,
    LAYER_WEIGHTS,
    MODEL_VERSION,
    TIER_RISK,
    _level_from_risk,
    score_submission_risk,
)


def _ct(score=None, dims=None, llm=None):
    """Build a minimal critical_thinking_control result."""
    return {
        "score": score,
        "dimensions": dims or {},
        "llm_dimensions": llm,
    }


def _dim(control):
    return {"control": control}


# ── level banding ───────────────────────────────────────────────────────────

def test_level_bands():
    assert _level_from_risk(0) == "low"
    assert _level_from_risk(33.9) == "low"
    assert _level_from_risk(34.0) == "medium"
    assert _level_from_risk(61.9) == "medium"
    assert _level_from_risk(62.0) == "high"
    assert _level_from_risk(100) == "high"


# ── axis mapping ──────────────────────────────────────────────────────────────

def test_text_pattern_axis_from_tier_and_keeps_percentage():
    r = score_submission_risk(ai_likelihood_score=82.0, ai_tier="RED",
                              critical_thinking_control=_ct(score=50))
    tp = r["axes"]["text_pattern"]
    assert tp["display_score"] == 82.0          # percentage demoted but VISIBLE
    assert tp["risk"] == TIER_RISK["RED"]
    assert tp["level"] == "high"


def test_citation_axis_levels():
    for level, risk in AXIS_LEVEL_RISK.items():
        r = score_submission_risk(ai_tier="GREEN", critical_thinking_control=_ct(score=90),
                                  axis_scores={"citation": level})
        assert r["axes"]["citation"]["risk"] == risk


def test_ownership_is_inverse_of_control():
    r = score_submission_risk(ai_tier="GREEN", critical_thinking_control=_ct(score=70))
    assert r["axes"]["ownership"]["risk"] == 30.0  # 100 - 70
    assert r["axes"]["ownership"]["level"] == "low"


def test_ownership_unknown_when_control_abstains():
    r = score_submission_risk(ai_tier="GREEN", critical_thinking_control=_ct(score=None))
    assert r["axes"]["ownership"]["level"] == "unknown"
    assert r["axes"]["ownership"]["risk"] is None


def test_defence_readiness_from_dimensions_and_llm():
    ct = _ct(
        score=60,
        dims={
            "student_judgement": _dim(40),
            "reasoning_trail": _dim(50),
            "evidence_grounding": _dim(60),
        },
        llm={"reflection": {"score": 50}, "alternative_comparison": {"score": 50}},
    )
    r = score_submission_risk(ai_tier="AMBER", critical_thinking_control=ct)
    # mean control over [40,50,60,50,50] = 50 -> risk 50
    assert r["axes"]["defence_readiness"]["risk"] == 50.0


# ── policy / declaration is never guessed ────────────────────────────────────

def test_policy_declaration_always_unknown_and_excluded():
    # Drive every other axis HIGH; policy must still be unknown and must not
    # contribute to the overall blend.
    ct = _ct(score=5, dims={"student_judgement": _dim(5)})
    r = score_submission_risk(ai_likelihood_score=95, ai_tier="RED",
                              critical_thinking_control=ct,
                              axis_scores={"citation": "attention"})
    pol = r["axes"]["policy_declaration"]
    assert pol["level"] == "unknown"
    assert pol["risk"] is None
    assert "self-declare" in pol["note"].lower()
    # overall is high from the real axes, NOT inflated/deflated by policy
    assert r["overall"]["level"] == "high"


# ── overall blend + main reason ───────────────────────────────────────────────

def test_overall_all_high():
    ct = _ct(score=10, dims={
        "student_judgement": _dim(10), "reasoning_trail": _dim(10),
        "evidence_grounding": _dim(10),
    })
    r = score_submission_risk(ai_tier="RED", critical_thinking_control=ct,
                              axis_scores={"citation": "attention"})
    assert r["overall"]["level"] == "high"
    assert r["overall"]["risk"] > 62


def test_overall_all_low_has_no_nag_reason():
    ct = _ct(score=95, dims={
        "student_judgement": _dim(95), "reasoning_trail": _dim(95),
        "evidence_grounding": _dim(95),
    })
    r = score_submission_risk(ai_tier="GREEN", critical_thinking_control=ct,
                              axis_scores={"citation": "clear"})
    assert r["overall"]["level"] == "low"
    assert r["overall"]["main_reason"] == ""   # don't nag a clean draft


def test_ownership_dominates_main_reason():
    # ownership weight 0.50 with high risk should drive the headline reason.
    ct = _ct(score=10)  # ownership risk 90
    r = score_submission_risk(ai_tier="GREEN", critical_thinking_control=ct,
                              axis_scores={"citation": "clear"})
    assert r["overall"]["main_reason"]  # non-empty
    assert "ownership" in r["overall"]["main_reason"].lower()


def test_no_data_is_unknown_not_a_false_verdict():
    r = score_submission_risk(ai_tier=None, critical_thinking_control=_ct(score=None))
    assert r["overall"]["level"] == "unknown"
    assert r["overall"]["risk"] is None


def test_blend_renormalises_over_available_layers():
    # Only ownership available -> overall risk == ownership risk (weights renormalise).
    ct = _ct(score=40)  # ownership risk 60
    r = score_submission_risk(ai_tier=None, critical_thinking_control=ct)
    assert r["overall"]["risk"] == 60.0
    assert r["overall"]["level"] == "medium"


def test_low_coverage_flag():
    r = score_submission_risk(ai_tier="GREEN", critical_thinking_control=_ct(score=80),
                              scored_sentence_count=3)
    assert r["low_coverage"] is True


def test_floor_to_document_tier_prevents_low_on_high_report():
    # The real bug: a high-tier report with low axes read as "Low submission risk".
    # Flooring to document_tier='high' must raise it to high with a flagged-findings reason.
    ct = _ct(score=85, dims={"student_judgement": _dim(85), "reasoning_trail": _dim(85),
                             "evidence_grounding": _dim(85)})
    r = score_submission_risk(ai_tier="GREEN", critical_thinking_control=ct,
                              axis_scores={"citation": "clear"}, document_tier="high")
    assert r["overall"]["level"] == "high"
    assert r["overall"]["floored"] is True
    assert r["overall"]["main_reason_code"] == "flagged_findings"
    assert r["overall"]["risk"] >= 62


def test_floor_does_not_lower_a_higher_computed_level():
    # When the blend is already >= the tier floor, keep the computed level + reason.
    ct = _ct(score=10)  # ownership risk 90 -> high
    r = score_submission_risk(ai_tier="RED", critical_thinking_control=ct,
                              axis_scores={"citation": "attention"}, document_tier="medium")
    assert r["overall"]["level"] == "high"
    assert r["overall"]["floored"] is False
    assert r["overall"]["main_reason_code"] != "flagged_findings"


def test_clean_tier_does_not_floor():
    ct = _ct(score=95, dims={"student_judgement": _dim(95)})
    r = score_submission_risk(ai_tier="GREEN", critical_thinking_control=ct,
                              axis_scores={"citation": "clear"}, document_tier="clean")
    assert r["overall"]["level"] == "low"
    assert r["overall"]["floored"] is False


def test_floor_applies_even_with_no_axis_data():
    r = score_submission_risk(ai_tier=None, critical_thinking_control=_ct(score=None),
                              document_tier="high")
    assert r["overall"]["level"] == "high"
    assert r["overall"]["main_reason_code"] == "flagged_findings"


def test_shape_and_weights_exposed():
    r = score_submission_risk(ai_tier="AMBER", critical_thinking_control=_ct(score=50))
    assert r["model"] == MODEL_VERSION
    assert r["weights"] == LAYER_WEIGHTS
    assert set(r["axes"]) == {
        "text_pattern", "ownership", "citation", "defence_readiness", "policy_declaration",
    }
