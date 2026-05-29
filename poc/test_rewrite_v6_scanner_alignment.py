"""Stage 1 scanner-fix guarantees: the V6 report-aligned scan must surface the graded detector's
predictability findings AND carry the token-level detail (spans) needed to act on them.

Regression anchor for the misalignment we found: the structural scanner flagged the LEAST
predictable sentence hardest (a comma list) and missed the MOST predictable one. With the report
blended in, the scan must cover the graded findings and (in predictability-primary mode) must not
let a comma list outrank the graded predictability score.
"""

from __future__ import annotations

import json
import os
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


def test_predictability_primary_demotes_structural_overweight(monkeypatch):
    report = _report()
    document = _document(report)

    monkeypatch.setenv("DRAFTPROOF_V6_SCANNER_PREDICTABILITY", "")
    structural = {f.sentence_id: f.severity for f in findings_for_paragraph(scan_text_with_report(document, report), "p005")}

    monkeypatch.setenv("DRAFTPROOF_V6_SCANNER_PREDICTABILITY", "1")
    predictability = {f.sentence_id: f.severity for f in findings_for_paragraph(scan_text_with_report(document, report), "p005")}

    # The packed-list sentence (s023) is over-weighted by the structural heuristic; under
    # predictability-primary its severity must fall to the graded score, not the comma-list inflation.
    assert structural["s023"] > predictability["s023"], (structural["s023"], predictability["s023"])
    # And no predictability-primary severity should exceed ~50 (all p005 sentences are "review" tier).
    assert max(predictability.values()) <= 60.0, predictability


def test_default_mode_is_unchanged_enrichment_only(monkeypatch):
    # Default (flag off): severities keep the legacy structural max() behaviour; enrichment is the
    # only addition. s023 keeps its structural severity (~71).
    monkeypatch.setenv("DRAFTPROOF_V6_SCANNER_PREDICTABILITY", "")
    report = _report()
    scan = scan_text_with_report(_document(report), report)
    by_id = {f.sentence_id: f for f in findings_for_paragraph(scan, "p005")}
    assert by_id["s023"].severity > 60.0
    assert (by_id["s023"].evidence or {}).get("predictability")  # enrichment present even in default mode
