"""Moderate surgical top-k pass after the reviewer: rewrite ONLY the highest-top-k sentences, gated to
stay fluent + faithful, stopping at the ~0.50 target. GPT-2 is mocked here (deterministic/fast); the
gate logic (lowers top-k + grammar + polarity + length) is exercised for real.

allow-hardcode: the sample sentences below are test fixtures to exercise the gate, not matching logic.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from poc.rewrite_v6 import topk_surgical as ts

HIGH = "The employers I have consulted repeatedly tell me that communication matters greatly to them."
FIX = "Employers I have spoken with stress that clear dialogue is what counts most to them."
LOW = "In my 2019 class twelve students built a working water filter together over three weeks."


def _stub(payload: str):
    class _G:
        def chat(self, prompt, **kwargs):
            return SimpleNamespace(content=payload, raw_content=payload)
    return _G()


def _mock_gpt2(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_TOPK_SURGICAL", "1")   # default is OFF; enable for these tests
    monkeypatch.setattr(ts, "_sentence_topk",
                        lambda s, k: 0.20 if "spoken with" in s else (0.60 if "consulted repeatedly" in s else 0.30))
    monkeypatch.setattr(ts, "_doc_topk",
                        lambda t, k: 0.45 if "spoken with" in t else 0.70)


def test_enabled_default_off(monkeypatch):
    # default OFF: measured no-op on fluent text (top-k is function-word-dominated)
    monkeypatch.delenv("DRAFTPROOF_V6_TOPK_SURGICAL", raising=False)
    assert ts.topk_surgical_enabled() is False
    monkeypatch.setenv("DRAFTPROOF_V6_TOPK_SURGICAL", "1")
    assert ts.topk_surgical_enabled() is True


def test_applies_gated_lowering_fix_to_high_sentence(monkeypatch):
    _mock_gpt2(monkeypatch)
    text = HIGH + " " + LOW
    payload = json.dumps({"fixes": [{"i": 0, "text": FIX}]})
    out, applied = ts.apply_surgical(text, gateway=_stub(payload))
    assert FIX in out and HIGH not in out
    assert LOW in out                              # low sentence untouched
    assert len(applied) == 1 and applied[0]["original"] == HIGH


def test_gate_rejects_non_lowering_fix(monkeypatch):
    _mock_gpt2(monkeypatch)
    # candidate that does NOT lower top-k (keeps the "consulted repeatedly" marker -> stays 0.60)
    bad = "The employers I have consulted repeatedly insist that talking clearly matters greatly to them."
    payload = json.dumps({"fixes": [{"i": 0, "text": bad}]})
    out, applied = ts.apply_surgical(HIGH + " " + LOW, gateway=_stub(payload))
    assert applied == [] and HIGH in out           # nothing applied


def test_noop_when_already_at_target(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_TOPK_SURGICAL", "1")
    monkeypatch.setattr(ts, "_doc_topk", lambda t, k: 0.40)   # already <= target
    monkeypatch.setattr(ts, "_sentence_topk", lambda s, k: 0.60)
    out, applied = ts.apply_surgical(HIGH + " " + LOW, gateway=_stub("{}"))
    assert applied == [] and out == HIGH + " " + LOW


def test_disabled_noop(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_TOPK_SURGICAL", "0")
    out, applied = ts.apply_surgical(HIGH, gateway=_stub("{}"))
    assert applied == [] and out == HIGH


def test_bad_json_fails_open(monkeypatch):
    _mock_gpt2(monkeypatch)
    out, applied = ts.apply_surgical(HIGH + " " + LOW, gateway=_stub("not json"))
    assert applied == [] and HIGH in out


# --- wiring: mutates text + re-scans + traces ---
from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text


def test_apply_topk_surgical_mutates_and_rescans(monkeypatch):
    _mock_gpt2(monkeypatch)
    txt = HIGH + " " + LOW
    doc = DocumentResult(initial_scan=scan_text(txt), final_scan=scan_text(txt),
                         rewritten_text=txt, passes=[], pass_trace=[])
    payload = json.dumps({"fixes": [{"i": 0, "text": FIX}]})
    out = dr._apply_topk_surgical(doc, _stub(payload), cancellation_check=None)
    assert FIX in out.rewritten_text                          # text mutated by the gated fix
    assert out.final_scan.source_text == out.rewritten_text   # re-scanned
    assert any(r.get("selected_source") == "topk_surgical" and r.get("applied") == 1 for r in out.pass_trace)


def test_apply_topk_surgical_disabled_noop(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_TOPK_SURGICAL", "0")
    txt = HIGH
    doc = DocumentResult(initial_scan=scan_text(txt), final_scan=scan_text(txt),
                         rewritten_text=txt, passes=[], pass_trace=[])
    assert dr._apply_topk_surgical(doc, _stub("{}"), cancellation_check=None) is doc
