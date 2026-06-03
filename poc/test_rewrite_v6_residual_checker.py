"""Residual checker = rewrite pass 2. Re-scans the REWRITTEN draft and re-runs the writer on
paragraphs the fresh re-scan flags; unflagged paragraphs keep their pass-1 text (invariant:
never revert to the original submitted text)."""
import json
import os
import types
from poc.rewrite_v6 import direct_rewrite


def test_kill_switch_default_on_and_off(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_RESIDUAL_FIX", raising=False)
    assert direct_rewrite.residual_fix_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", off)
        assert direct_rewrite.residual_fix_enabled() is False
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    assert direct_rewrite.residual_fix_enabled() is True


from poc.rewrite_v6.text import Paragraph
from poc.rewrite_v6.scan import scan_text as real_scan_text, scan_text_preserve_blocks
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
    monkeypatch.setattr(direct_rewrite, "scan_text_preserve_blocks", _fake_scan(paragraphs, recorder))
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


class _Gateway:
    def __init__(self, rewrite):
        self.rewrite = rewrite

    def chat(self, *_args, **_kwargs):
        return types.SimpleNamespace(
            content=json.dumps({"rewrite": self.rewrite, "author_review_items": []}),
            raw_content=json.dumps({"rewrite": self.rewrite, "author_review_items": []}),
        )


def test_candidate_internal_paragraph_break_is_collapsed(monkeypatch):
    monkeypatch.setattr(direct_rewrite, "_has_broken_grammar", lambda _text: False)
    monkeypatch.setattr(direct_rewrite, "_severe_polarity_inversion", lambda _candidate, _paragraph: False)
    monkeypatch.setattr(direct_rewrite, "_severe_beat_loss", lambda _candidate, _paragraph: False)
    paragraph = Paragraph(
        id="p001",
        index=0,
        text="Students compare sources before trusting the article in class.",
        sentences=[],
    )
    candidate, _ = direct_rewrite._clean_candidate(
        _Gateway("Students compare sources before trusting the article in class.\n\nThey explain why."),
        paragraph,
        None,
        [types.SimpleNamespace(tags=["generic_assertion_risk"])],
    )
    assert candidate == "Students compare sources before trusting the article in class. They explain why."


def test_candidate_with_severe_beat_loss_is_rejected(monkeypatch):
    monkeypatch.setattr(direct_rewrite, "_has_broken_grammar", lambda _text: False)
    monkeypatch.setattr(direct_rewrite, "_severe_polarity_inversion", lambda _candidate, _paragraph: False)
    monkeypatch.setattr(direct_rewrite, "_severe_beat_loss", lambda _candidate, _paragraph: True)
    paragraph = Paragraph(
        id="p001",
        index=0,
        text="Students compare sources before trusting the article.",
        sentences=[],
    )
    candidate, _ = direct_rewrite._clean_candidate(
        _Gateway("Hiring teams prefer candidates who communicate clearly."),
        paragraph,
        None,
        [types.SimpleNamespace(tags=["generic_assertion_risk"])],
    )
    assert candidate is None


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


def test_residual_skips_when_fresh_scan_changes_paragraph_count(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    rewritten = "PASS1 A.\n\nPASS1 B."
    paras = [
        Paragraph(id="p001", index=0, text="PASS1 A.", sentences=[]),
        Paragraph(id="p002", index=1, text="PASS1 B.", sentences=[]),
        Paragraph(id="p003", index=2, text="SCANNER EXTRA.", sentences=[]),
    ]
    monkeypatch.setattr(direct_rewrite, "scan_text_preserve_blocks", _fake_scan(paras))
    out = direct_rewrite._apply_residual_fix(_doc(rewritten), gateway=None, cancellation_check=None)
    assert out.rewritten_text == rewritten
    entry = out.pass_trace[-1]
    assert entry["selected_source"] == "residual_checker"
    assert entry["status"] == "skipped_shape_mismatch"
    assert entry["expected_paragraphs"] == 2
    assert entry["actual_paragraphs"] == 3


def test_preserve_blocks_scan_does_not_create_virtual_paragraphs():
    sentence = "This sentence carries enough words to be counted as a regular sentence in the scanner output. "
    text = "".join(sentence for _ in range(12)).strip()
    assert len(real_scan_text(text).paragraphs) > 1
    assert len(scan_text_preserve_blocks(text).paragraphs) == 1


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


def test_document_selection_score_prices_raw_topk(monkeypatch):
    reports = {
        "low_ai_high_topk": {
            "ai_score": 25.0,
            "ai_risk_badge": {
                "ai_components": {
                    "topk_pattern_raw": 80.0,
                    "topk_calibrated_risk": 40.0,
                    "generic_assertion_risk": 40.0,
                }
            },
        },
        "higher_ai_low_topk": {
            "ai_score": 28.0,
            "ai_risk_badge": {
                "ai_components": {
                    "topk_pattern_raw": 55.0,
                    "topk_calibrated_risk": 20.0,
                    "generic_assertion_risk": 40.0,
                }
            },
        },
    }
    monkeypatch.setattr(direct_rewrite, "_document_scan_report", lambda text: reports[text])
    monkeypatch.setattr(direct_rewrite, "_rhythm_risk", lambda _text: 0.0)
    assert (
        direct_rewrite._document_selection_score("higher_ai_low_topk")
        < direct_rewrite._document_selection_score("low_ai_high_topk")
    )


def test_direct_prompt_can_include_scanner_derived_topk_pressure(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_WRITER_TOPK_PRESSURE", raising=False)
    monkeypatch.setattr(
        direct_rewrite,
        "_topk_pressure",
        lambda _text: {
            "purpose": "scanner-derived predictability pressure",
            "source_topk_fraction": 0.74,
            "topk_k": 10,
            "high_topk_token_ledger": [{"token": " and", "hits": 4}],
            "rules": ["Do not damage grammar."],
        },
    )
    prompt = direct_rewrite._prompt(
        "Students use online tools, and they also compare source reliability.",
        {"main_issue": "predictable"},
        ["predictable_start"],
    )
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["topk_pressure"]["source_topk_fraction"] == 0.74
    assert payload["topk_pressure"]["high_topk_token_ledger"] == [{"token": " and", "hits": 4}]


def test_direct_prompt_topk_pressure_kill_switch(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_WRITER_TOPK_PRESSURE", "0")
    monkeypatch.setattr(direct_rewrite, "_topk_pressure", lambda _text: {"source_topk_fraction": 0.74})
    prompt = direct_rewrite._prompt(
        "Students use online tools, and they also compare source reliability.",
        {"main_issue": "predictable"},
        ["predictable_start"],
    )
    payload = json.loads(prompt.split("\n", 1)[1])
    assert "topk_pressure" not in payload


def test_all_candidates_none_keeps_pass1_and_records_checked_trace(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    rewritten = "PARA_A_REWRITTEN.\n\nPARA_B_REWRITTEN."
    paras = [Paragraph(id="p001", index=0, text="PARA_A_REWRITTEN.", sentences=[]),
             Paragraph(id="p002", index=1, text="PARA_B_REWRITTEN.", sentences=[])]
    # both flagged, but the writer yields no clean candidate for either
    _patch(monkeypatch, paragraphs=paras, flagged_ids={"p001", "p002"},
           candidate_by_id={"p001": None, "p002": None})
    out = direct_rewrite._apply_residual_fix(_doc(rewritten), gateway=None, cancellation_check=None)
    assert out.rewritten_text == rewritten          # pass-1 text preserved, no regression
    entry = next(e for e in out.pass_trace if e.get("selected_source") == "residual_checker")
    assert entry["refixed"] == 0 and entry["flagged_paragraphs"] == 2
