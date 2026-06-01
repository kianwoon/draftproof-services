"""Residual checker = rewrite pass 2. Re-scans the REWRITTEN draft and re-runs the writer on
paragraphs the fresh re-scan flags; unflagged paragraphs keep their pass-1 text (invariant:
never revert to the original submitted text)."""
import os
from poc.rewrite_v6 import direct_rewrite


def test_kill_switch_default_on_and_off(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_RESIDUAL_FIX", raising=False)
    assert direct_rewrite.residual_fix_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", off)
        assert direct_rewrite.residual_fix_enabled() is False
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    assert direct_rewrite.residual_fix_enabled() is True


import types
from poc.rewrite_v6.text import Paragraph
from poc.rewrite_v6.scan import scan_text as real_scan_text
from poc.rewrite_v6.pipeline import DocumentResult


def _doc(rewritten_text, original_text="orig A.\n\norig B."):
    """A DocumentResult as produced by pass 1. initial_scan = ORIGINAL (for before/after diff)."""
    return DocumentResult(
        initial_scan=real_scan_text(original_text),
        final_scan=real_scan_text(rewritten_text),
        passes=[],
        rewritten_text=rewritten_text,
        pass_trace=[],
    )


def _fake_scan(paragraphs, recorder=None):
    def _scan_text(text):
        if recorder is not None:
            recorder.append(text)
        return types.SimpleNamespace(paragraphs=paragraphs)
    return _scan_text


def _patch(monkeypatch, *, paragraphs, flagged_ids, candidate_by_id, recorder=None):
    monkeypatch.setattr(direct_rewrite, "scan_text", _fake_scan(paragraphs, recorder))
    monkeypatch.setattr(
        direct_rewrite, "findings_for_paragraph",
        lambda scan, pid: [types.SimpleNamespace(tags=["generic_assertion_risk"], paragraph_id=pid)]
        if pid in flagged_ids else [],
    )
    monkeypatch.setattr(
        direct_rewrite, "_clean_candidate",
        lambda gateway, paragraph, diagnosis, findings, **kw: (candidate_by_id.get(paragraph.id), []),
    )


def test_invariant_unflagged_keeps_pass1_text_and_scans_rewritten(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    rewritten = "PARA_A_REWRITTEN.\n\nPARA_B_REWRITTEN."
    paras = [Paragraph(id="p001", index=0, text="PARA_A_REWRITTEN.", sentences=[]),
             Paragraph(id="p002", index=1, text="PARA_B_REWRITTEN.", sentences=[])]
    seen = []
    _patch(monkeypatch, paragraphs=paras, flagged_ids={"p001"},
           candidate_by_id={"p001": "PARA_A_REFIXED."}, recorder=seen)
    out = direct_rewrite._apply_residual_fix(_doc(rewritten), gateway=None, cancellation_check=None)
    assert out.rewritten_text == "PARA_A_REFIXED.\n\nPARA_B_REWRITTEN."
    assert seen and seen[0] == rewritten


def test_candidate_none_falls_back_to_pass1_paragraph(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    rewritten = "PARA_A_REWRITTEN.\n\nPARA_B_REWRITTEN."
    paras = [Paragraph(id="p001", index=0, text="PARA_A_REWRITTEN.", sentences=[]),
             Paragraph(id="p002", index=1, text="PARA_B_REWRITTEN.", sentences=[])]
    _patch(monkeypatch, paragraphs=paras, flagged_ids={"p001"}, candidate_by_id={"p001": None})
    out = direct_rewrite._apply_residual_fix(_doc(rewritten), gateway=None, cancellation_check=None)
    assert out.rewritten_text == rewritten


def test_noop_records_trace_and_preserves_initial_scan(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    rewritten = "CLEAN A.\n\nCLEAN B."
    paras = [Paragraph(id="p001", index=0, text="CLEAN A.", sentences=[]),
             Paragraph(id="p002", index=1, text="CLEAN B.", sentences=[])]
    _patch(monkeypatch, paragraphs=paras, flagged_ids=set(), candidate_by_id={})
    doc = _doc(rewritten)
    out = direct_rewrite._apply_residual_fix(doc, gateway=None, cancellation_check=None)
    assert out.rewritten_text == rewritten
    assert out.initial_scan is doc.initial_scan
    assert any(e.get("selected_source") == "residual_checker" for e in out.pass_trace)


def test_kill_switch_skips_rescan(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "0")
    called = []
    monkeypatch.setattr(direct_rewrite, "scan_text", _fake_scan([], called))
    doc = _doc("X.\n\nY.")
    out = direct_rewrite._apply_residual_fix(doc, gateway=None, cancellation_check=None)
    assert out is doc
    assert called == []


def test_residual_fix_runs_before_reviewer(monkeypatch):
    """Order guard: in run_direct_rewrite_all, residual fix must execute before the reviewer."""
    order = []
    monkeypatch.setattr(direct_rewrite, "_best_of_n", lambda: 1)
    monkeypatch.setattr(direct_rewrite, "_rewrite_document_once",
                        lambda *a, **k: _doc("P1.\n\nP2."))
    monkeypatch.setattr(direct_rewrite, "_apply_residual_fix",
                        lambda doc, gateway, **k: (order.append("residual"), doc)[1])
    monkeypatch.setattr(direct_rewrite, "_apply_reviewer",
                        lambda doc, gateway, **k: (order.append("reviewer"), doc)[1])
    monkeypatch.setattr(direct_rewrite, "LLMGateway", lambda *a, **k: None)
    direct_rewrite.run_direct_rewrite_all("some original text.\n\nsecond paragraph here.")
    assert order == ["residual", "reviewer"]
