"""Regression test for job 5bacaeb3 (2026-08-01): direct_rewrite.py never read the scan report's
`rewrite_decision`, so a `mode: targeted, full_rewrite_allowed: false` decision scoped to 22
findings in a handful of paragraphs still triggered a full-document regeneration (pass 1 rewrote
all 12 paragraphs, the residual pass rewrote all 12 again). Fixed via
poc/rewrite_v6/rewrite_targeting.py's RewriteTargeting gate, threaded through
_rewrite_document_once_inner / _rewrite_paragraphs_parallel / _apply_residual_fix."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/poc on path

from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.rewrite_targeting import resolve_rewrite_targeting
from poc.rewrite_v6.scan import scan_text_preserve_blocks


def _detect_json(targets, full_rewrite_allowed=False, mode="targeted"):
    findings = {
        "high": [
            {"finding_id": "f001", "rewrite_context": {"paragraph_id": "p001"}},
            {"finding_id": "f002", "rewrite_context": {"paragraph_id": "p002"}},
        ],
        "medium": [
            {"finding_id": "f003", "paragraph_id": "p003"},  # top-level fallback, no rewrite_context
        ],
        "low": [],
    }
    return {
        "findings": findings,
        "rewrite_decision": {
            "run_rewrite": True,
            "mode": mode,
            "targets": targets,
            "full_rewrite_allowed": full_rewrite_allowed,
        },
    }


# --- resolve_rewrite_targeting -----------------------------------------------------------------

def test_targeted_decision_resolves_target_paragraph_ids(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_HONOR_REWRITE_DECISION", raising=False)
    targeting = resolve_rewrite_targeting(_detect_json(["f001", "f003"]))
    assert targeting.active is True
    assert targeting.target_paragraph_ids == {"p001", "p003"}
    assert targeting.should_skip("p002") is True
    assert targeting.should_skip("p001") is False


def test_full_rewrite_allowed_is_a_noop_gate(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_HONOR_REWRITE_DECISION", raising=False)
    targeting = resolve_rewrite_targeting(_detect_json(["f001"], full_rewrite_allowed=True))
    assert targeting.active is False
    assert targeting.should_skip("p999") is False  # no-op gate never skips


def test_empty_targets_is_a_noop_gate(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_HONOR_REWRITE_DECISION", raising=False)
    targeting = resolve_rewrite_targeting(_detect_json([]))
    assert targeting.active is False


def test_missing_rewrite_decision_is_a_noop_gate(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_HONOR_REWRITE_DECISION", raising=False)
    targeting = resolve_rewrite_targeting({"findings": {}})
    assert targeting.active is False


def test_unresolvable_mapping_falls_back_to_full_rewrite(monkeypatch, caplog):
    """Every target id is unknown to the findings map -> can't be sure which paragraphs are meant,
    so fall back to current (rewrite-everything) behaviour rather than skip everything."""
    monkeypatch.delenv("DRAFTPROOF_V6_HONOR_REWRITE_DECISION", raising=False)
    targeting = resolve_rewrite_targeting(_detect_json(["nope_1", "nope_2"]))
    assert targeting.active is False
    assert targeting.should_skip("p001") is False


def test_kill_switch_off_is_a_noop_gate_even_with_valid_targets(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_HONOR_REWRITE_DECISION", "0")
    targeting = resolve_rewrite_targeting(_detect_json(["f001"]))
    assert targeting.active is False
    assert targeting.should_skip("p002") is False


# --- wiring into the writer loop (concurrency=1, sequential path) ------------------------------

def _doc_text(n_paragraphs=3):
    return "\n\n".join(f"Paragraph {i} needs work with some generic filler text here." for i in range(n_paragraphs))


def test_sequential_loop_skips_non_target_flagged_paragraphs(monkeypatch):
    text = _doc_text(3)
    scan = scan_text_preserve_blocks(text)
    paragraph_ids = [p.id for p in scan.paragraphs]

    monkeypatch.setattr(dr, "findings_for_paragraph", lambda scan, pid: [{"tag": "generic_assertion_risk"}])
    monkeypatch.setattr(dr, "paragraph_diagnosis", lambda pid: None)
    monkeypatch.setattr(dr, "_writer_concurrency", lambda: 1)
    calls = []

    def fake_clean_candidate(gateway, paragraph, diagnosis, findings, authorship_targets=None, lane="control", raise_rate_limit=False):
        calls.append(paragraph.id)
        return f"REWRITTEN: {paragraph.text}", []

    monkeypatch.setattr(dr, "_clean_candidate", fake_clean_candidate)

    from poc.rewrite_v6.rewrite_targeting import RewriteTargeting
    targeting = RewriteTargeting(active=True, target_paragraph_ids={paragraph_ids[1]})

    result = dr._rewrite_document_once_inner(
        scan, gateway=None, progress_callback=None, cancellation_check=None, rewrite_targeting=targeting,
    )

    assert calls == [paragraph_ids[1]]  # only the target paragraph was ever sent to the writer
    skipped_rows = [row for row in result.pass_trace if row.get("status") == "skipped_not_target"]
    assert {row["target_paragraph_id"] for row in skipped_rows} == {paragraph_ids[0], paragraph_ids[2]}
    assert "REWRITTEN:" in result.rewritten_text.split("\n\n")[1]
    assert "REWRITTEN:" not in result.rewritten_text.split("\n\n")[0]
    assert "REWRITTEN:" not in result.rewritten_text.split("\n\n")[2]


def test_sequential_loop_unaffected_when_targeting_is_none(monkeypatch):
    """(b) No targeting (legacy call sites, or a resolved no-op gate) -> byte-identical to current
    behaviour: every flagged paragraph is rewritten."""
    text = _doc_text(2)
    scan = scan_text_preserve_blocks(text)

    monkeypatch.setattr(dr, "findings_for_paragraph", lambda scan, pid: [{"tag": "generic_assertion_risk"}])
    monkeypatch.setattr(dr, "paragraph_diagnosis", lambda pid: None)
    monkeypatch.setattr(dr, "_writer_concurrency", lambda: 1)
    calls = []

    def fake_clean_candidate(gateway, paragraph, diagnosis, findings, authorship_targets=None, lane="control", raise_rate_limit=False):
        calls.append(paragraph.id)
        return f"REWRITTEN: {paragraph.text}", []

    monkeypatch.setattr(dr, "_clean_candidate", fake_clean_candidate)

    result = dr._rewrite_document_once_inner(
        scan, gateway=None, progress_callback=None, cancellation_check=None, rewrite_targeting=None,
    )
    assert len(calls) == 2  # both paragraphs rewritten, no skipped_not_target rows
    assert not any(row.get("status") == "skipped_not_target" for row in result.pass_trace)


# --- skipped_not_target rows must be user-visible (2026-08-01 follow-up #2) --------------------

def test_skipped_not_target_trace_carries_a_review_card():
    """A skipped-but-flagged paragraph must not vanish silently -- guards ANNOTATE, never SUPPRESS
    (CLAUDE.md). skipped_trace() must embed an author_review_items card the same shape
    _author_review_cards_from_pass_trace already knows how to render."""
    from poc.rewrite_v6.rewrite_targeting import RewriteTargeting

    targeting = RewriteTargeting(active=True, target_paragraph_ids={"p999"})
    row = targeting.skipped_trace(0, "p001")

    assert row["status"] == "skipped_not_target"
    items = row.get("author_review_items")
    assert isinstance(items, list) and len(items) == 1
    card = items[0]
    assert card["paragraph_id"] == "p001"
    assert card.get("why") or card.get("added")


def test_skipped_not_target_row_surfaces_as_author_review_card():
    """End-to-end: production.py's card composer must pick up the skipped row's review item."""
    from poc.rewrite_v6 import production as v6_production
    from poc.rewrite_v6.rewrite_targeting import RewriteTargeting

    targeting = RewriteTargeting(active=True, target_paragraph_ids={"p999"})
    pass_trace = [targeting.skipped_trace(0, "p001")]

    cards = v6_production._author_review_cards_from_pass_trace(pass_trace)
    assert len(cards) == 1
    assert cards[0]["where"] == "Paragraph p001"
