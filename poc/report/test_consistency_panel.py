"""Tests for the stylometric-consistency "Writing-style outliers" display composer.

The composer re-presents Phase-1 ConsistencyDetector findings
(poc/detect/consistency.py, finding_type="stylometric_outlier", scanner=
"consistency") as an advisory, informational-only panel. It is display-only:
``scoring`` is hard-False and ConsistencyDetector.overall_risk is unconditionally
0.0, so this NEVER touches the tier or score. No findings -> None (nothing
renders; prod byte-identical when DRAFTPROOF_CONSISTENCY is off).

Mirrors poc/report/test_claim_graph_panel.py's structure (unit tests against the
pure composer) plus a report-build integration section proving the
report.report.report_to_dict wiring — the same real-pipeline approach Task 3's
poc/test_consistency_report_parity.py used (not a reimplementation).

allow-hardcode: the fixture paragraph excerpts / detail / recommendation strings
below are TEST INPUT DATA (fixture rows fed to the composer under test), mirroring
poc/report/test_claim_graph_panel.py's fixture claim text — not a matching/scoring
word-list consumed by production code.
"""
from __future__ import annotations

from report.consistency_panel import compose_consistency_display


# ── Realistic finding-row fixture (shape report.report.report_to_dict builds
#    from report.models.Finding rows filtered to scanner == "consistency" — see
#    poc/report/report.py's caller and poc/report/builder.py's paragraph_id
#    threading for scanner == "consistency"). ──
def _fixture_findings() -> list[dict]:
    return [
        {
            "paragraph_id": "p006",
            "excerpt": "The overarching paradigm shift necessitates a holistic "
                       "reevaluation of every stakeholder's core competencies.",
            "outlier_score": 0.83,
            "top_deviating_features": ["sentence length", "passive voice rate", "readability"],
            "detail": "This paragraph's writing style deviates sharply from the "
                      "rest of the document — most notably in sentence length, "
                      "passive voice rate, readability.",
            "recommendation": "Review this paragraph — its sentence structure and "
                              "word choice read differently than the rest of the "
                              "document.",
        },
        {
            "paragraph_id": "p002",
            "excerpt": "We looked at three cases from our own project files.",
            "outlier_score": 0.61,
            "top_deviating_features": ["vocabulary diversity"],
            "detail": "This paragraph's writing style deviates sharply from the "
                      "rest of the document — most notably in vocabulary diversity.",
            "recommendation": "Review this paragraph.",
        },
    ]


# ── Absent / empty -> None (prod byte-identical). ──

def test_none_when_none():
    assert compose_consistency_display(None) is None


def test_none_when_not_list():
    assert compose_consistency_display("nope") is None
    assert compose_consistency_display(123) is None
    assert compose_consistency_display({"rows": []}) is None


def test_none_when_empty_list():
    assert compose_consistency_display([]) is None


def test_none_when_all_rows_malformed():
    bad = [
        {"paragraph_id": "", "excerpt": "has excerpt but no id"},
        {"paragraph_id": "p001", "excerpt": ""},
        {"paragraph_id": "p002"},
        "not a dict",
        None,
    ]
    assert compose_consistency_display(bad) is None


# ── Populated input -> pinned contract. ──

def test_present_and_advisory():
    out = compose_consistency_display(_fixture_findings())
    assert out is not None
    assert out["present"] is True
    assert out["scoring"] is False  # ALWAYS advisory — never affects tier/score


def test_summary_count():
    out = compose_consistency_display(_fixture_findings())
    assert out["summary"]["flagged_paragraphs"] == 2


def test_row_shape():
    out = compose_consistency_display(_fixture_findings())
    row = out["rows"][0]
    assert set(row.keys()) == {
        "paragraph_id", "excerpt", "outlier_score", "features",
        "features_label", "recommendation",
    }


def test_features_are_plain_english_and_joined():
    out = compose_consistency_display(_fixture_findings())
    row = next(r for r in out["rows"] if r["paragraph_id"] == "p006")
    assert row["features"] == ["sentence length", "passive voice rate", "readability"]
    assert row["features_label"] == "sentence length, passive voice rate, readability"


def test_missing_features_falls_back_to_overall_writing_style():
    findings = [{
        "paragraph_id": "p001",
        "excerpt": "Some paragraph text.",
        "outlier_score": 0.5,
        "top_deviating_features": [],
        "recommendation": "Review it.",
    }]
    out = compose_consistency_display(findings)
    row = out["rows"][0]
    assert row["features"] == []
    assert row["features_label"] == "overall writing style"


def test_rows_sorted_by_outlier_score_descending():
    out = compose_consistency_display(_fixture_findings())
    scores = [r["outlier_score"] for r in out["rows"]]
    assert scores == sorted(scores, reverse=True)
    assert out["rows"][0]["paragraph_id"] == "p006"


def test_rows_capped():
    findings = [
        {
            "paragraph_id": f"p{i:03d}",
            "excerpt": f"Paragraph {i} text with enough content to be usable.",
            "outlier_score": float(i) / 100.0,
            "top_deviating_features": ["sentence length"],
            "recommendation": "Review it.",
        }
        for i in range(20)
    ]
    out = compose_consistency_display(findings)
    assert len(out["rows"]) <= 12


def test_excerpt_trimmed():
    findings = [{
        "paragraph_id": "p001",
        "excerpt": "X" * 500,
        "outlier_score": 0.7,
        "top_deviating_features": ["sentence length"],
        "recommendation": "Review it.",
    }]
    out = compose_consistency_display(findings)
    assert len(out["rows"][0]["excerpt"]) <= 321  # ~320 + ellipsis char


def test_outlier_score_rounded_and_coerced():
    findings = [{
        "paragraph_id": "p001",
        "excerpt": "Some paragraph text.",
        "outlier_score": "not-a-number",
        "top_deviating_features": ["sentence length"],
        "recommendation": "Review it.",
    }]
    out = compose_consistency_display(findings)
    assert out["rows"][0]["outlier_score"] is None


def test_fail_open_on_malformed():
    # Malformed rows must never raise — either None or a safe dict.
    bad = [{"paragraph_id": "p001", "excerpt": "ok", "top_deviating_features": "not-a-list",
            "outlier_score": object()}]
    out = compose_consistency_display(bad)
    assert out is None or isinstance(out, dict)


def test_never_leaks_tier_or_score_top_level_field():
    out = compose_consistency_display(_fixture_findings())
    assert "score" not in out
    assert "tier" not in out
    assert out["scoring"] is False


# ── Report-build integration: report.report.report_to_dict wiring. ──
# Uses the real production path (DetectionRunner -> ReportBuilder ->
# report_to_dict), the same faithfulness discipline as Task 3's
# poc/test_consistency_report_parity.py: DetectionRunner(detectors=[...]) bypasses
# the kill-switch gate + the heavy default ML detectors while still exercising
# the REAL builder/report_to_dict conversion, not a reimplementation.

def test_report_to_dict_omits_key_when_no_consistency_scanner_result():
    from detect.run import DetectionRunner
    from report.builder import ReportBuilder
    from report.report import report_to_dict

    text = (
        "The committee reviewed the proposal carefully before reaching a "
        "decision. Because the budget projections were incomplete, several "
        "members requested additional documentation."
    )
    # Default detector set never includes ConsistencyDetector unless
    # DRAFTPROOF_CONSISTENCY is on (poc/detect/run.py) — this proves the
    # composer hook is a true no-op (key absent, not merely None) when the
    # scanner never ran, independent of the env-var kill switch itself (already
    # covered end-to-end by poc/test_consistency_report_parity.py).
    det_report = DetectionRunner(detectors=[]).run_all(text)
    draft_report = ReportBuilder().add_detection_report(det_report).build()
    result = report_to_dict(draft_report)

    assert result.get("consistency_display") is None


def test_report_to_dict_includes_consistency_display_when_scanner_present():
    from detect.consistency import ConsistencyDetector
    from detect.run import DetectionRunner
    from detect.test_consistency import _document_with_shifted_paragraph
    from report.builder import ReportBuilder
    from report.report import report_to_dict

    # DetectionRunner(detectors=[ConsistencyDetector()]) bypasses
    # _build_detectors()/the kill switch + the heavy default ML detectors
    # (matches poc/test_consistency_report_parity.py's fast, isolated approach)
    # while still exercising the real ReportBuilder/report_to_dict conversion
    # this task adds.
    det_report = DetectionRunner(detectors=[ConsistencyDetector()]).run_all(
        _document_with_shifted_paragraph()
    )
    draft_report = ReportBuilder().add_detection_report(det_report).build()
    result = report_to_dict(draft_report)

    display = result.get("consistency_display")
    assert display is not None
    assert display["present"] is True
    assert display["scoring"] is False
    assert any(r["paragraph_id"] == "p006" for r in display["rows"])
