"""The grounding showcase is a TEACHING layer after the QC reviewer: it flags GENERIC sentences,
shows a grounded worked example, and surfaces ONLY genuinely-good examples (a generic sentence that
truly becomes grounded). Bad example -> no teaching. Graded on grounding (real, content-agnostic
regex check -- fast, no GPT-2), never on AI-detector scores.

allow-hardcode: the sample sentences below are test fixtures (inputs to exercise the grounding gate),
not matching logic or a phrase list.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from poc.rewrite_v6 import predictability_showcase as ps

GENERIC = "Education systems change faster than many schools can comfortably manage their plans."
GROUNDED = "Last spring my twelve ninth-graders rewrote their plans after the district's third reform."
STILL_GENERIC = "Education systems often change more quickly than schools are able to adjust to them."


def _stub(payload: str):
    class _G:
        def __init__(self): self.calls = []
        def chat(self, prompt, **kwargs):
            self.calls.append(prompt)
            return SimpleNamespace(content=payload, raw_content=payload)
    return _G()


def test_showcase_enabled_default_on(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", raising=False)
    assert ps.showcase_enabled() is True
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "0")
    assert ps.showcase_enabled() is False


def test_candidates_flag_only_generic_sentences():
    doc = GENERIC + " In my 2023 algebra class, three students used a TikTok clip to solve a problem."
    cands = ps._generic_candidates(doc, 8)
    assert GENERIC in cands                       # generic -> flagged
    assert all("2023 algebra class" not in c for c in cands)  # already grounded -> skipped


def test_is_teachable_gate():
    assert ps._is_teachable(GENERIC, GROUNDED) is True          # generic -> genuinely grounded
    assert ps._is_teachable(GENERIC, STILL_GENERIC) is False    # still generic -> not taught
    assert ps._is_teachable(GENERIC, "too short") is False      # stub
    assert ps._is_teachable(GENERIC, GENERIC) is False          # unchanged


def test_generate_showcase_keeps_grounded_example():
    payload = json.dumps({"examples": [{"i": 0, "grounded": GROUNDED, "why": "adds a season, a count, and a concrete event"}]})
    items = ps.generate_showcase(GENERIC, gateway=_stub(payload))
    assert len(items) == 1
    it = items[0]
    assert it["sentence"] == GENERIC
    assert it["suggestion"] == GROUNDED
    assert it["grounded_before"] is False and it["grounded_after"] is True
    assert it["why"]


def test_generate_showcase_drops_weak_example():
    # "if the showcase is no good, then no teaching": a still-generic suggestion is NOT taught
    payload = json.dumps({"examples": [{"i": 0, "grounded": STILL_GENERIC, "why": "reworded"}]})
    assert ps.generate_showcase(GENERIC, gateway=_stub(payload)) == []


def test_generate_showcase_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "0")
    assert ps.generate_showcase(GENERIC, gateway=_stub("{}")) == []


def test_generate_showcase_bad_json_fails_open():
    assert ps.generate_showcase(GENERIC, gateway=_stub("not json")) == []


# --- wiring: annotate-only after the reviewer ---
from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text


def test_apply_showcase_annotates_without_mutating():
    txt = GENERIC + " Schools must also weigh how each change affects their existing routines and staff."
    doc = DocumentResult(initial_scan=scan_text(txt), final_scan=scan_text(txt),
                         rewritten_text=txt, passes=[], pass_trace=[])
    payload = json.dumps({"examples": [{"i": 0, "grounded": GROUNDED, "why": "adds a concrete moment"}]})
    out = dr._apply_showcase(doc, _stub(payload), cancellation_check=None)
    assert out.rewritten_text == txt                       # shipped rewrite NOT mutated
    assert out.final_scan is doc.final_scan                # NOT re-scanned
    assert out.predictability_showcase and out.predictability_showcase[0]["suggestion"] == GROUNDED
    assert any(r.get("selected_source") == "predictability_showcase" for r in out.pass_trace)
    assert out.to_dict()["predictability_showcase"][0]["grounded_after"] is True


def test_apply_showcase_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_PREDICTABILITY_SHOWCASE", "0")
    txt = "hello world this is a short clean line"
    doc = DocumentResult(initial_scan=scan_text(txt), final_scan=scan_text(txt),
                         rewritten_text=txt, passes=[], pass_trace=[])
    assert dr._apply_showcase(doc, _stub("{}"), cancellation_check=None) is doc
