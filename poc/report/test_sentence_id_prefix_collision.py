"""Regression tests for the finding -> sentence_id join in
poc/report/builder.py::ReportBuilder.add_detection (Strategy 1).

Bug fixed here: ``_build_predictability_summary`` built ``_sentence_id_map`` as a
plain ``dict`` keyed by ``sentence[:60]`` (last-write-wins). ANY two sentences
sharing a 60-char prefix — exact duplicates included — collapsed every matching
finding onto the LAST occurrence's sid, so the inline grounding/reasoning/
predictability underlines (poc/report/sentence_issue_tags.py) and per-paragraph
tallies (poc/report/report.py ``_sid_to_pid``) anchored the WRONG occurrence.

The fix keys by the same 60-char prefix but stores an in-document-order list of
``(full_sentence, sentence_id)`` and resolves a finding to its OWN occurrence:
prefer a full-text match (disambiguates *different* sentences sharing a prefix),
then consume repeated occurrences of identical text in order via a FIFO cursor.

These tests exercise the join in isolation with ``location={}`` so ONLY Strategy 1
runs (Strategies 2/3 are location-based and would otherwise mask it).

allow-hardcode: the sentence strings below are TEST INPUT DATA fed to the join
under test, not a production matching/scoring word-list.
"""
from __future__ import annotations

from types import SimpleNamespace

from report.builder import ReportBuilder


def _finding(evidence, finding_type="low_specificity", risk_level="medium"):
    # Duck-typed detect.Finding: add_detection reads these attributes plus
    # getattr fallbacks for actionability/evidence_strength/signal_category.
    return SimpleNamespace(
        finding_type=finding_type,
        risk_level=risk_level,
        evidence=evidence,
        detail="",
        recommendation="",
        location={},          # force Strategy-1-only resolution
        metadata={},
        signal_category="grounding_risk",
    )


def _pred_result(sentences, findings):
    # Duck-typed detect.DetectResult for scanner == "predictability".
    raw = {"sentences": [{"sentence_id": sid, "sentence": text}
                         for sid, text in sentences]}
    return SimpleNamespace(
        scanner="predictability",
        findings=findings,
        raw=raw,
        overall_risk=0.5,
        risk_distribution={},
    )


def _resolved(builder):
    """title/evidence -> resolved sentence_id for every converted finding."""
    return [(f.evidence, f.sentence_id) for f in builder._findings]


# 64-char shared prefix so both sentences below collide on sentence[:60].
_PREFIX = "This paragraph presents a broad and general claim about the topic"
_SENT_A = _PREFIX + " in the very first case discussed."
_SENT_B = _PREFIX + " in the entirely separate second case."
_DUP = "The results were consistent across all three experimental runs today."


def test_unique_sentences_regression_unchanged():
    """Common case: all sentences unique in their first 60 chars. Each finding
    resolves to its own sid — identical to the pre-fix behavior (no-op)."""
    sentences = [("s001", "Alpha sentence about the introduction and its scope."),
                 ("s002", "Beta sentence covering an unrelated methodological point.")]
    findings = [_finding(sentences[0][1]), _finding(sentences[1][1])]
    b = ReportBuilder()
    b.add_detection(_pred_result(sentences, findings))
    assert _resolved(b) == [(sentences[0][1], "s001"), (sentences[1][1], "s002")]


def test_prefix_collision_resolves_to_own_sentence():
    """Two DIFFERENT sentences sharing a 60-char prefix: each finding anchors to
    its OWN sentence via full-text match (was: both -> last occurrence 's002')."""
    sentences = [("s001", _SENT_A), ("s002", _SENT_B)]
    assert _SENT_A[:60] == _SENT_B[:60]        # they DO collide on the prefix
    findings = [_finding(_SENT_A), _finding(_SENT_B)]
    b = ReportBuilder()
    b.add_detection(_pred_result(sentences, findings))
    assert _resolved(b) == [(_SENT_A, "s001"), (_SENT_B, "s002")]


def test_prefix_collision_sparse_subset_only_second_flagged():
    """Only the SECOND of two prefix-colliding sentences is flagged. Full-text
    matching (not positional FIFO) anchors it to s002 — a pure in-order FIFO
    would have mis-assigned it to s001."""
    sentences = [("s001", _SENT_A), ("s002", _SENT_B)]
    findings = [_finding(_SENT_B)]
    b = ReportBuilder()
    b.add_detection(_pred_result(sentences, findings))
    assert _resolved(b) == [(_SENT_B, "s002")]


def test_exact_duplicate_findings_anchor_first_then_second():
    """Exact-duplicate sentence appearing twice, two findings about it (in doc
    order): they anchor to the FIRST then the SECOND occurrence respectively
    (was: both -> last occurrence 's003')."""
    sentences = [("s001", _DUP), ("s002", "A distinct intervening sentence here."),
                 ("s003", _DUP)]
    findings = [_finding(_DUP), _finding(_DUP)]
    b = ReportBuilder()
    b.add_detection(_pred_result(sentences, findings))
    assert _resolved(b) == [(_DUP, "s001"), (_DUP, "s003")]


def test_exact_duplicate_third_finding_falls_back_to_last():
    """More findings than occurrences: after every occurrence is consumed, the
    surplus falls back to the LAST occurrence (better than dropping it)."""
    sentences = [("s001", _DUP), ("s003", _DUP)]
    findings = [_finding(_DUP), _finding(_DUP), _finding(_DUP)]
    b = ReportBuilder()
    b.add_detection(_pred_result(sentences, findings))
    assert _resolved(b) == [(_DUP, "s001"), (_DUP, "s003"), (_DUP, "s003")]
