"""Tests for detect.consistency.ConsistencyDetector — Task 3.

Covers: (1) the brief's literal scenario — a document with a known style-shifted
paragraph produces a Finding with finding_type="stylometric_outlier",
signal_category="authorship_risk", and DetectResult.overall_risk == 0.0
unconditionally (Phase 1 = informational only, zero weight into the fused score);
(2) a document below stylometry.outliers.MIN_PARAGRAPHS produces no findings and
does not crash; (3) the kill-switch helper consistency_enabled() default-off /
falsey-value behavior; (4) precise Finding-mapping from a controlled OutlierResult
(decoupled from the real outlier statistics, which are Task 2's own test
responsibility in detect/stylometry/test_outliers.py).

allow-hardcode: the fixture paragraphs below are hand-authored TEST INPUT TEXT
(five similarly-styled prose paragraphs plus one deliberately choppy, fragment-style
paragraph) used only to exercise the detector end-to-end — not a matching/scoring
list consumed by production code.
"""
from __future__ import annotations

import pytest

from detect.base import DetectResult, Finding
from detect.consistency import (
    CONSISTENCY_KILL_SWITCH_ENV,
    ConsistencyDetector,
    consistency_enabled,
)
from detect.stylometry.outliers import MIN_PARAGRAPHS, OutlierResult

# ---------------------------------------------------------------------------
# Fixture text — five similarly-styled paragraphs + one deliberately shifted
# (short, choppy, fragment-style) paragraph. Validated empirically to flag the
# shifted paragraph (p006) via the real extract_fingerprints -> detect_outliers
# pipeline; the underlying leave-one-out robust-z statistic is noisy on a small
# real-text baseline, so this test asserts the shifted paragraph IS flagged, not
# that it is the ONLY paragraph flagged (outlier-detection precision itself is
# covered by detect/stylometry/test_outliers.py).
# ---------------------------------------------------------------------------

_BASELINE_TEMPLATE = (
    "The {a} evaluated {b} before reaching a final decision. "
    "Because the {c} were incomplete, several {d} requested additional {e}. "
    "However, the overall {f} remained sound and defensible. "
    "Therefore, the group approved a revised {g} that addressed the outstanding "
    "{h} raised during the earlier {i}, provided that the {j} would submit "
    "updated {k} within two weeks."
)

_BASELINE_TOPICS = [
    ("committee", "the proposal", "budget projections", "members", "documentation",
     "plan", "timeline", "concerns", "discussion", "finance office", "figures"),
    ("research team", "the survey results", "sample estimates", "analysts", "corrections",
     "trend", "hypothesis", "findings", "analysis", "statistics office", "numbers"),
    ("planning office", "the infrastructure request", "maintenance costs", "departments",
     "spending levels", "upgrade", "rollout", "objections", "meeting", "budget office", "totals"),
    ("editorial board", "the manuscript", "review comments", "reviewers", "revisions",
     "argument", "acceptance", "issues", "debate", "production office", "edits"),
    ("audit team", "the financial statements", "transaction records", "auditors", "reviews",
     "position", "opinion", "gaps", "inquiry", "compliance office", "entries"),
]

_SHIFTED_PARAGRAPH = " ".join([
    "Fast pace continued through the entire long day.",
    "Loud noise filled every single crowded room.",
    "Bright light hurt many tired eyes badly.",
    "Crowds pushed forward without much clear warning.",
    "Short breaks helped very little today somehow.",
    "Quick moves saved some valuable time again.",
    "Hard stops caused real serious ongoing problems.",
    "Fresh starts felt good after that moment.",
])


def _document_with_shifted_paragraph() -> str:
    baseline_paragraphs = [_BASELINE_TEMPLATE.format(
        a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, i=i, j=j, k=k,
    ) for (a, b, c, d, e, f, g, h, i, j, k) in _BASELINE_TOPICS]
    return "\n\n".join(baseline_paragraphs + [_SHIFTED_PARAGRAPH])


# ---------------------------------------------------------------------------
# End-to-end: known style-shifted paragraph
# ---------------------------------------------------------------------------


def test_style_shifted_paragraph_produces_finding_and_zero_overall_risk():
    detector = ConsistencyDetector()
    result = detector.detect(_document_with_shifted_paragraph())

    assert isinstance(result, DetectResult)
    assert result.scanner == "consistency"
    assert result.overall_risk == 0.0
    assert result.findings, "expected at least one stylometric_outlier finding"

    shifted_findings = [
        f for f in result.findings if f.location.get("paragraph_id") == "p006"
    ]
    assert shifted_findings, "the known style-shifted paragraph (p006) must be flagged"

    for f in result.findings:
        assert isinstance(f, Finding)
        assert f.finding_type == "stylometric_outlier"
        assert f.signal_category == "authorship_risk"
        assert f.risk_level == "review"
        assert f.actionability == "review_only"
        assert f.evidence.strip() != ""
        assert f.detail.strip() != ""
        assert "outlier_score" in f.metadata
        assert "top_deviating_features" in f.metadata


def test_overall_risk_is_zero_even_with_multiple_findings():
    detector = ConsistencyDetector()
    result = detector.detect(_document_with_shifted_paragraph())

    # Phase 1: informational only, regardless of how many paragraphs are flagged.
    assert result.overall_risk == 0.0


# ---------------------------------------------------------------------------
# Below MIN_PARAGRAPHS floor — no findings, no crash
# ---------------------------------------------------------------------------


def test_short_document_below_min_paragraphs_produces_no_findings_and_does_not_crash():
    assert MIN_PARAGRAPHS > 2, "test assumes a 2-paragraph document is below the floor"
    short_text = (
        "This is a short first paragraph with only a handful of words in it.\n\n"
        "This is a short second paragraph, also with just a few words inside it."
    )
    detector = ConsistencyDetector()

    result = detector.detect(short_text)

    assert result.findings == []
    assert result.overall_risk == 0.0


def test_empty_document_produces_no_findings_and_does_not_crash():
    detector = ConsistencyDetector()

    result = detector.detect("")

    assert result.findings == []
    assert result.overall_risk == 0.0


# ---------------------------------------------------------------------------
# Kill-switch helper
# ---------------------------------------------------------------------------


def test_consistency_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv(CONSISTENCY_KILL_SWITCH_ENV, raising=False)
    assert consistency_enabled() is False


@pytest.mark.parametrize("falsey_value", ["0", "false", "False", "no", "off", ""])
def test_consistency_enabled_falsey_values_are_off(monkeypatch, falsey_value):
    monkeypatch.setenv(CONSISTENCY_KILL_SWITCH_ENV, falsey_value)
    assert consistency_enabled() is False


@pytest.mark.parametrize("truthy_value", ["1", "true", "True", "yes", "on"])
def test_consistency_enabled_truthy_values_are_on(monkeypatch, truthy_value):
    monkeypatch.setenv(CONSISTENCY_KILL_SWITCH_ENV, truthy_value)
    assert consistency_enabled() is True


# ---------------------------------------------------------------------------
# Precise Finding-mapping from a controlled OutlierResult (decoupled from the
# real outlier statistics, which are Task 2's responsibility to test).
# ---------------------------------------------------------------------------


def test_finding_mapping_uses_plain_english_top_deviating_features(monkeypatch):
    controlled_outlier = OutlierResult(
        paragraph_id="p003",
        outlier_score=7.25,
        top_deviating_features=["sentence length", "passive voice rate"],
    )

    import detect.consistency as consistency_module

    monkeypatch.setattr(
        consistency_module, "extract_fingerprints", lambda content: ["fp1", "fp2"]
    )
    monkeypatch.setattr(
        consistency_module, "detect_outliers", lambda fingerprints: [controlled_outlier]
    )

    detector = ConsistencyDetector()
    result = detector.detect("Paragraph one text.\n\nParagraph two text.\n\nParagraph three text.")

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.finding_type == "stylometric_outlier"
    assert finding.signal_category == "authorship_risk"
    assert finding.location["paragraph_id"] == "p003"
    assert finding.metadata["outlier_score"] == 7.25
    assert finding.metadata["top_deviating_features"] == ["sentence length", "passive voice rate"]
    # Plain-English feature names (not raw metric identifiers) appear in the detail.
    assert "sentence length" in finding.detail
    assert "passive voice rate" in finding.detail
    assert result.overall_risk == 0.0
