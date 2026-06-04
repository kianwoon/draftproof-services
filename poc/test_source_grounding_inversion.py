"""Regression test for the source_grounding_strength inversion bug.

`detect/criteria/source_grounding.py` returns a CONCERN (0 = well grounded, 1 = unsupported), but
`report.py` previously consumed it directly as a STRENGTH, inverting the badge: a well-grounded text
reported the WORST grounding risk (100%). This locks the corrected DIRECTION end-to-end: a text whose
factual claims carry citations must score a LOWER source_grounding_risk than the same claims uncited.

allow-hardcode: the two strings below are TEST FIXTURES (sample paragraphs), not a detect/scoring/
matching word-list. The grounding signal is computed by the content-agnostic detector under test.
"""

from rewrite_v3.pipeline import _scan_report


_CLAIMS_UNCITED = (
    "Online learning increases student engagement and improves outcomes across schools. "
    "Technology reduces barriers and raises achievement for most learners. "
    "Frequent testing causes higher long-term retention and better performance in every subject."
)

_CLAIMS_CITED = (
    "Online learning increases student engagement (Means et al., 2010). "
    "Technology reduces barriers and raises achievement for most learners (Smith & Jones, 2019). "
    "Frequent testing causes higher long-term retention (Roediger and Karpicke, 2006)."
)


def _grounding_risk(text: str):
    report = _scan_report(text)
    components = (report.get("ai_risk_badge", {}) or {}).get("writing_components", {}) or {}
    return components.get("source_grounding_risk")


def test_cited_claims_have_lower_grounding_risk_than_uncited():
    """Direction guard: citing the claims must REDUCE source_grounding_risk, not raise it.

    Under the old inverted mapping the cited text scored HIGHER risk (the bug); this asserts the fix.
    """
    uncited_risk = _grounding_risk(_CLAIMS_UNCITED)
    cited_risk = _grounding_risk(_CLAIMS_CITED)
    assert uncited_risk is not None and cited_risk is not None
    assert cited_risk < uncited_risk, (
        f"grounded text should score LOWER grounding risk than ungrounded "
        f"(cited={cited_risk}, uncited={uncited_risk})"
    )
