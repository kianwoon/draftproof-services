"""V7 deep-scan rows drive rewrite targeting.

The scan's user-facing verdict is the V7 fused score; its per-paragraph evidence lives at
``badge.tier_authority.paragraphs``. These tests pin the objective-level contract: a paragraph
flagged ONLY by the deep scan (no legacy finding, no highlight segment) must still become a
rewrite target — otherwise the rewrite cannot mitigate the exact signal the user was shown.
"""

from poc.rewrite_v6.scan import findings_for_paragraph, scan_text_with_report

TITLE = "Critical Analysis: Renewable Energy Adoption"
BODY_1 = (
    "Solar capacity in the region grew last year. Grid operators tracked the change closely. "
    "Storage remained the binding constraint."
)
BODY_2 = (
    "Wind projects faced different siting hurdles. Coastal permits took longer to clear. "
    "Local boards required extra hearings."
)
DOCUMENT = f"{TITLE}\n\n{BODY_1}\n\n{BODY_2}"


def _report(rows):
    return {
        "ai_risk_badge": {
            "ai_likelihood_score": 55.0,
            "tier_authority": {
                "proportion": 0.5,
                "reliability_floor": 0.3,
                "paragraphs": rows,
            },
        },
    }


def _deep_scan_findings_for(scan, paragraph_id):
    return [
        f for f in findings_for_paragraph(scan, paragraph_id)
        if "deep_scan_ai_flag" in f.tags
    ]


def test_deep_scan_only_paragraph_becomes_target():
    # Row index 0 = FIRST BODY paragraph (the single-line title is excluded from
    # row ordinals by the bridge, so the scan-side mapping must skip it too).
    report = _report([
        {"index": 0, "sentence_count": 3, "flagged_count": 3, "proportion": 1.0, "band": "red"},
        {"index": 1, "sentence_count": 3, "flagged_count": 0, "proportion": 0.0, "band": "insufficient"},
    ])
    scan = scan_text_with_report(DOCUMENT, report)
    paragraphs = scan.paragraphs
    assert len(paragraphs) == 3  # title + 2 body blocks
    body_1, body_2 = paragraphs[1], paragraphs[2]

    flagged = _deep_scan_findings_for(scan, body_1.id)
    assert len(flagged) == 1, [f.to_dict() for f in findings_for_paragraph(scan, body_1.id)]
    finding = flagged[0]
    assert finding.evidence["source"] == "v7_deep_scan"
    assert finding.evidence["band"] == "red"
    assert finding.severity == 100.0
    # The title block must never be a deep-scan target.
    assert not _deep_scan_findings_for(scan, paragraphs[0].id)
    # Insufficient-band rows carry no verdict weight — no target.
    assert not _deep_scan_findings_for(scan, body_2.id)


def test_deep_scan_findings_counted_in_scores():
    report = _report([
        {"index": 0, "sentence_count": 3, "flagged_count": 2, "proportion": 0.67, "band": "orange"},
        {"index": 1, "sentence_count": 3, "flagged_count": 2, "proportion": 0.67, "band": "orange"},
    ])
    scan = scan_text_with_report(DOCUMENT, report)
    assert scan.scores["deep_scan_finding_count"] == 2.0


def test_no_tier_authority_is_backward_compatible():
    scan = scan_text_with_report(DOCUMENT, {"ai_risk_badge": {"ai_likelihood_score": 55.0}})
    assert not any("deep_scan_ai_flag" in f.tags for f in scan.findings)
    assert "deep_scan_finding_count" not in scan.scores or scan.scores["deep_scan_finding_count"] == 0.0


def test_malformed_rows_fail_open():
    report = _report([
        {"index": 99, "proportion": 0.9, "band": "red"},   # out of range
        {"index": "x", "proportion": 0.9, "band": "red"},  # non-numeric
        "not-a-dict",
        {"index": 1, "proportion": None, "band": "red"},   # no proportion
    ])
    scan = scan_text_with_report(DOCUMENT, report)
    assert not any("deep_scan_ai_flag" in f.tags for f in scan.findings)
