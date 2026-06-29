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
