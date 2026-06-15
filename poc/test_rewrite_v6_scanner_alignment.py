"""Scanner-fix guarantees: the V6 report-aligned scan must surface the graded detector's
predictability findings AND carry the token-level detail (spans) needed to act on them.

Regression anchor for the misalignment we found: the structural scanner flagged the LEAST
predictable sentence hardest (a comma list) and missed the MOST predictable one. With the report
blended in, the scan must cover the graded findings. (The former `DRAFTPROOF_V6_SCANNER_PREDICTABILITY`
switch was removed when scan.py's content-word detectors were dropped: grounding coverage is now
unconditional and no longer needs a flag to outrank a comma-list inflation.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poc.rewrite_v6.scan import findings_for_paragraph, scan_text_with_report

REPORT_PATH = Path(__file__).resolve().parent.parent / (
    "test_output/production_rewrite_ad62c7f1_20260529/"
    "reports__7f9eada9-e81a-4e4c-be2b-0308c7bc8b61__report.json"
)

pytestmark = pytest.mark.skipif(not REPORT_PATH.exists(), reason="saved production report not present")


def _report():
    return json.loads(REPORT_PATH.read_text())


def _document(report) -> str:
    seen: dict[str, str] = {}
    for brief in report.get("rewrite_edit_briefs", []):
        pid = brief.get("paragraph_id")
        exc = brief.get("paragraph_excerpt")
        if pid and exc and pid not in seen:
            seen[pid] = exc.strip()
    return "\n\n".join(seen[pid] for pid in sorted(seen))


def test_scan_surfaces_all_predictability_sentences_for_p005():
    report = _report()
    scan = scan_text_with_report(_document(report), report)
    p005 = findings_for_paragraph(scan, "p005")
    # The graded detector flags all six p005 sentences (f017-f022). The blended scan must cover them
    # all -- including the ones the structural scanner alone missed.
    assert len(p005) == 6, [f.sentence_id for f in p005]


def test_predictability_findings_carry_token_spans():
    report = _report()
    scan = scan_text_with_report(_document(report), report)
    p005 = findings_for_paragraph(scan, "p005")
    detailed = [f for f in p005 if (f.evidence or {}).get("predictability", {}).get("predictable_token_spans")]
    # Every flagged sentence should carry the spans the writer must break (the actionable detail).
    assert len(detailed) == len(p005), [
        (f.sentence_id, bool((f.evidence or {}).get("predictability"))) for f in p005
    ]
    # Spot-check the structure on one finding.
    detail = (detailed[0].evidence or {})["predictability"]
    assert detail["predictable_token_spans"], detail
    assert "top10_ratio" in detail


def test_predictability_enrichment_present_unconditionally():
    # The former DRAFTPROOF_V6_SCANNER_PREDICTABILITY switch is gone; grounding/predictability
    # enrichment now rides on every report-blended finding without a flag. The graded predictability
    # detail (token spans) must be attached to the flagged sentence's evidence.
    report = _report()
    scan = scan_text_with_report(_document(report), report)
    by_id = {f.sentence_id: f for f in findings_for_paragraph(scan, "p005")}
    assert "s023" in by_id, list(by_id)
    assert (by_id["s023"].evidence or {}).get("predictability")  # enrichment present unconditionally
