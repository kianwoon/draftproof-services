"""The predictability showcase is a TEACHING layer after the QC reviewer: it flags predictable
phrasing, asks the LLM for a more distinctive alternative, and shows ONLY validated reductions. It is
annotate-only -- it must never change the shipped rewrite. GPT-2 is mocked here so the suite stays
fast and deterministic; one guarded smoke test exercises the real model if available.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from poc.rewrite_v6 import predictability_showcase as ps


def _stub(payload: str):
    class _G:
        def __init__(self):
            self.calls = []
        def chat(self, prompt, **kwargs):
            self.calls.append(prompt)
            return SimpleNamespace(content=payload, raw_content=payload)
    return _G()


def test_showcase_enabled_default_on(monkeypatch):
    # enabled in production by request; env=0 is the kill switch
    monkeypatch.delenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", raising=False)
    assert ps.showcase_enabled() is True
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "0")
    assert ps.showcase_enabled() is False


def test_generate_showcase_keeps_validated_reduction(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "1")
    # mock the GPT-2-backed helpers: the marked sentence is "predictable", its reword is not
    monkeypatch.setattr(ps, "_predictability", lambda t, k: 0.80 if "PREDICT" in t else 0.20)
    monkeypatch.setattr(ps, "_flagged_words", lambda t, k: ["important", "various"] if "PREDICT" in t else [])
    doc = "This PREDICT line is important and various. A second clean line stands apart."
    payload = json.dumps({"examples": [
        {"i": 0, "suggestion": "Last spring my twelve students argued over two conflicting sources.",
         "why": "It leaned on broad, generic words instead of a concrete moment."}
    ]})
    items = ps.generate_showcase(doc, gateway=_stub(payload))
    assert len(items) == 1
    it = items[0]
    assert it["predictability_before"] == 0.80 and it["predictability_after"] == 0.20
    assert it["reduction"] == 0.60
    assert it["flagged_words"] == ["important", "various"]
    assert "twelve students" in it["suggestion"]
    assert it["why"]


def test_generate_showcase_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "0")
    assert ps.generate_showcase("anything at all here", gateway=_stub("{}")) == []


def test_generate_showcase_bad_json_fails_open(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "1")
    monkeypatch.setattr(ps, "_predictability", lambda t, k: 0.80 if "PREDICT" in t else 0.20)
    monkeypatch.setattr(ps, "_flagged_words", lambda t, k: ["important"] if "PREDICT" in t else [])
    assert ps.generate_showcase("This PREDICT sentence is important here.", gateway=_stub("not json")) == []


def test_is_teachable_gate():
    base = dict(sentence="The system plays a crucial role in the process today.", flagged_words=[], why="x")
    good = ps.ShowcaseItem(suggestion="Our intake desk decides whether a case moves forward that day.",
                           score_before=0.80, score_after=0.50, **base)
    assert ps._is_teachable(good) is True
    no_reduction = ps.ShowcaseItem(suggestion="Our intake desk decides whether a case moves forward that day.",
                                   score_before=0.80, score_after=0.79, **base)
    assert ps._is_teachable(no_reduction) is False         # below _MIN_REDUCTION
    stub = ps.ShowcaseItem(suggestion="too short", score_before=0.80, score_after=0.10, **base)
    assert ps._is_teachable(stub) is False                 # < 3 words
    unchanged = ps.ShowcaseItem(suggestion=base["sentence"], score_before=0.80, score_after=0.10, **base)
    assert ps._is_teachable(unchanged) is False            # identical to original


# --- wiring: annotate-only after the reviewer ---
from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text


def test_apply_showcase_annotates_without_mutating(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "1")
    txt = "This PREDICT line is important and various. Another clean line follows here today."
    doc = DocumentResult(initial_scan=scan_text(txt), final_scan=scan_text(txt),
                         rewritten_text=txt, passes=[], pass_trace=[])
    monkeypatch.setattr(ps, "_predictability", lambda t, k: 0.80 if "PREDICT" in t else 0.20)
    monkeypatch.setattr(ps, "_flagged_words", lambda t, k: ["important", "various"] if "PREDICT" in t else [])
    payload = json.dumps({"examples": [
        {"i": 0, "suggestion": "Last spring my twelve students argued over two conflicting sources.", "why": "generic wording"}
    ]})
    out = dr._apply_showcase(doc, _stub(payload), cancellation_check=None)
    assert out.rewritten_text == txt                       # shipped rewrite NOT mutated
    assert out.final_scan is doc.final_scan                # NOT re-scanned
    assert out.predictability_showcase and len(out.predictability_showcase) == 1
    assert any(r.get("selected_source") == "predictability_showcase" for r in out.pass_trace)
    # serialises into the report
    assert out.to_dict()["predictability_showcase"][0]["reduction"] == 0.60


def test_apply_showcase_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "0")
    txt = "hello world this is fine"
    doc = DocumentResult(initial_scan=scan_text(txt), final_scan=scan_text(txt),
                         rewritten_text=txt, passes=[], pass_trace=[])
    out = dr._apply_showcase(doc, _stub("{}"), cancellation_check=None)
    assert out is doc


@pytest.mark.skipif(ps._ensure_gpt2()[0] is None, reason="gpt2/torch unavailable")
def test_real_gpt2_predictability_ranks_generic_higher():
    predictable = "Technology plays a crucial role in the modern world and society today."
    distinctive = "Marcus hurled his graphing calculator across room 214 at 8:47 that Tuesday."
    assert ps._predictability(predictable, 10) > ps._predictability(distinctive, 10)
