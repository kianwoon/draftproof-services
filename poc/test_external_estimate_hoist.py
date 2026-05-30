"""The rewrite summary must carry the rewritten content's external-detector estimate, hoisted from
the rewritten scan's badge (precomputed), or recomputed from its ai_components, else None."""
from poc.rewrite_v6.production import _external_estimate_from_scan


def test_prefers_precomputed_estimate():
    scan = {"ai_risk_badge": {"external_detector_estimate": {"score": 61.2, "band": "high"}}}
    assert _external_estimate_from_scan(scan) == {"score": 61.2, "band": "high"}


def test_recomputes_from_components_when_estimate_absent():
    scan = {"ai_risk_badge": {"ai_components": {"topk_pattern": 70.0, "predictability": 40.0, "burstiness_risk": 30.0}}}
    est = _external_estimate_from_scan(scan)
    assert est is not None and "score" in est and "band" in est


def test_none_when_minimal_or_missing():
    assert _external_estimate_from_scan({"ai_risk_badge": {"ai_likelihood_score": 10.0}}) is None
    assert _external_estimate_from_scan({}) is None
    assert _external_estimate_from_scan(None) is None
