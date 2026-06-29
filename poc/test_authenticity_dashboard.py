from detect.authenticity_dashboard import compose_authenticity_dashboard as compose


def _badge(**kw):
    base = {"ai_likelihood_score": 32.0, "tier": "AMBER", "confidence": "high"}
    base.update(kw)
    return base


def test_learning_ownership_is_ct_score():
    d = compose(ai_risk_badge=_badge(critical_thinking_control={"score": 92.0}))
    assert d["learning_ownership"]["score"] == 92.0
    assert d["learning_ownership"]["available"] is True


def test_learning_ownership_null_when_ct_abstains():
    d = compose(ai_risk_badge=_badge(critical_thinking_control={"score": None}))
    assert d["learning_ownership"]["score"] is None
    assert d["learning_ownership"]["available"] is False


def test_grounding_inverts_gap_and_guards_no_data():
    gd = {"buckets": {"concrete_grounding": {"score": 19.0, "available": 3}}}
    d = compose(ai_risk_badge=_badge(grounding_diagnosis=gd))
    assert d["grounding"]["score"] == 81.0  # 100 - 19
    # available == 0 -> null, never a false 100
    gd0 = {"buckets": {"concrete_grounding": {"score": 0.0, "available": 0}}}
    d0 = compose(ai_risk_badge=_badge(grounding_diagnosis=gd0))
    assert d0["grounding"]["score"] is None
    assert d0["grounding"]["available"] is False


def test_citation_quality_inverts_mean_of_risks():
    wc = {"citation_weakness_risk": 30.0, "source_grounding_risk": 20.0}
    d = compose(ai_risk_badge=_badge(writing_components=wc))
    assert d["citation_quality"]["score"] == 75.0  # 100 - mean(30,20)=100-25


def test_citation_quality_drops_none_component():
    wc = {"citation_weakness_risk": 40.0, "source_grounding_risk": None}
    d = compose(ai_risk_badge=_badge(writing_components=wc))
    assert d["citation_quality"]["score"] == 60.0  # 100 - 40 (None dropped)


def test_citation_quality_null_when_no_components():
    d = compose(ai_risk_badge=_badge(writing_components={}))
    assert d["citation_quality"]["score"] is None
    assert d["citation_quality"]["available"] is False


def test_ai_assistance_band_from_tier():
    for tier, band in [("GREEN", "Low"), ("AMBER", "Moderate"), ("ORANGE", "High"), ("RED", "High")]:
        d = compose(ai_risk_badge=_badge(tier=tier))
        assert d["ai_assistance"]["band"] == band


def test_ai_assistance_score_is_inverted_likelihood():
    d = compose(ai_risk_badge=_badge(ai_likelihood_score=32.0))
    assert d["ai_assistance"]["score"] == 68.0  # 100 - 32


def test_ai_assistance_ci_is_tentative_and_bounded():
    pred = {"all_sentences": [{"predictability_risk": r} for r in (0.2, 0.4, 0.6, 0.8)]}
    d = compose(ai_risk_badge=_badge(confidence="low"), predictability=pred)
    ci = d["ai_assistance"]["ci"]
    assert ci["tentative"] is True
    assert 0.0 <= ci["low"] <= ci["high"] <= 100.0


def test_overall_weakest_link_floor():
    # one failing axis must drag the headline down, not be averaged away
    badge = _badge(
        critical_thinking_control={"score": 90.0},
        grounding_diagnosis={"buckets": {"concrete_grounding": {"score": 90.0, "available": 3}}},  # grounding=10
        writing_components={"citation_weakness_risk": 5.0, "source_grounding_risk": 5.0},  # citation=95
        ai_likelihood_score=10.0,  # ai_assistance=90
    )
    d = compose(ai_risk_badge=badge)
    assert d["overall"]["score"] == 10.0     # floored to worst available dim (grounding=10)
    assert d["overall"]["band"] == "High"


def test_overall_abstains_under_two_dims():
    d = compose(ai_risk_badge={"critical_thinking_control": {"score": 80.0}})  # only 1 dim
    assert d["overall"] is None


def test_placeholders_present():
    d = compose(ai_risk_badge=_badge())
    assert d["reasoning_consistency"]["available"] is False
    assert d["revision_evidence"]["status"] == "placeholder"
