"""Regression test for the 2026-08-01 incident (rewrite job 5bacaeb3): _apply_residual_fix used to
diff a pass-2 candidate against the PASS-1 rewritten text, so anything pass-1 already fabricated
(invented numbers, invented first-person framing, invented added paragraphs) was silently adopted
as the baseline and never carded. Fixed by diffing against doc.initial_scan (the true original) and
unconditionally carding any paragraph with no true-original counterpart (net-new/inserted)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/poc on path

from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6 import residual_fix as rf
from poc.rewrite_v6.scan import scan_text_preserve_blocks


def _doc(initial_text: str, rewritten_text: str):
    from poc.rewrite_v6.pipeline import DocumentResult

    initial_scan = scan_text_preserve_blocks(initial_text)
    return DocumentResult(
        initial_scan=initial_scan,
        final_scan=initial_scan,
        passes=[],
        rewritten_text=rewritten_text,
        pass_trace=[],
    )


def _run(doc, *, clean_candidate_fn, monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    monkeypatch.setattr(rf, "_residual_findings", lambda scan, para: [{"tag": "generic_assertion_risk"}])
    monkeypatch.setattr(dr, "_clean_candidate", clean_candidate_fn)
    return dr._apply_residual_fix(doc, gateway=None, cancellation_check=None)


def test_fabrication_introduced_in_pass1_is_now_carded(monkeypatch):
    """(a) The fabricated specific ("30%") was already present in the pass-1 text handed to residual
    fix. The residual candidate keeps it unchanged. Diffing against the TRUE original (which has no
    such number) must still produce a fabricated-specifics card."""
    original = "Teaching hairdressing requires patience and clear demonstration of technique."
    pass1_with_fabrication = "There is roughly a 30% financial-literacy gap among trainees."

    def clean_candidate_fn(gateway, paragraph, diagnosis, findings, authorship_targets=None, lane="control"):
        return pass1_with_fabrication, []

    doc = _doc(original, pass1_with_fabrication)
    result = _run(doc, clean_candidate_fn=clean_candidate_fn, monkeypatch=monkeypatch)

    residual_rows = [r for r in result.pass_trace if r.get("selected_source") == "residual_fix"]
    assert residual_rows, "expected a residual_fix trace row"
    items = residual_rows[0].get("author_review_items", [])
    assert any("concrete detail was added" in i.get("added", "") for i in items), items


def test_added_leading_paragraph_gets_unconditional_card(monkeypatch):
    """(b) rewritten_text has one MORE paragraph than the true original (a fabricated opening
    paragraph, per the incident). The extra paragraph must get a card even if nothing about it
    trips the content-fabrication heuristics."""
    original = "Second paragraph is the real original content, unchanged and factual."
    rewritten = (
        "In my community workshops I have watched many hairdressing trainees struggle.\n\n"
        "Second paragraph is the real original content, unchanged and factual."
    )

    def clean_candidate_fn(gateway, paragraph, diagnosis, findings, authorship_targets=None, lane="control"):
        return paragraph.text, []

    doc = _doc(original, rewritten)
    result = _run(doc, clean_candidate_fn=clean_candidate_fn, monkeypatch=monkeypatch)

    added_rows = [r for r in result.pass_trace if r.get("selected_source") == "residual_checker_added_paragraph"]
    assert added_rows, f"expected an added-paragraph trace row, got: {result.pass_trace}"
    items = added_rows[0].get("author_review_items", [])
    assert any(i.get("provenance") == "added_paragraph" for i in items), items


def test_unchanged_document_gains_no_spurious_cards(monkeypatch):
    """(c) A document that pass-1 left essentially untouched (no findings at residual re-scan) must
    not gain any new review cards from this fix."""
    original = "This paragraph was left exactly as written by the author with no changes at all."

    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    monkeypatch.setattr(rf, "_residual_findings", lambda scan, para: [])

    def clean_candidate_fn(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("no findings -> writer should not be invoked")

    monkeypatch.setattr(dr, "_clean_candidate", clean_candidate_fn)

    doc = _doc(original, original)
    result = dr._apply_residual_fix(doc, gateway=None, cancellation_check=None)

    review_rows = [r for r in result.pass_trace if r.get("author_review_items")]
    assert not review_rows, review_rows
