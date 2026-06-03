from __future__ import annotations

import sys
import types

# weasyprint is a PDF-only dep pulled in transitively by rewrite_v6/__init__; stub it for tests.
if "weasyprint" not in sys.modules:
    _wp = types.ModuleType("weasyprint")
    _wp.HTML = object
    _wp.CSS = object
    sys.modules["weasyprint"] = _wp

import json

from poc.rewrite_v6 import bracket_grounding as bg


class _StubGateway:
    def __init__(self, results):
        self._results = results

    def chat(self, *_args, **_kwargs):
        payload = {"results": self._results}
        return types.SimpleNamespace(content=json.dumps(payload), raw_content=json.dumps(payload))


# two un-grounded, >=6-word sentences -> both selected by _generic_candidates
TEXT = (
    "The world today rewards more than simple memorization of facts. "
    "I want my students to become thoughtful and skeptical readers over time."
)


def test_disabled_by_default_is_noop(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_BRACKET_GROUNDING", raising=False)  # code default is OFF
    out, applied = bg.apply_bracket_grounding(TEXT, gateway=_StubGateway([]))
    assert out == TEXT and applied == []


def test_improved_is_clean_text_with_green_span(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_BRACKET_GROUNDING", "1")
    gw = _StubGateway([
        {"i": 0, "improved": "Employers in my district now ask for more than memorized facts."},
        {"i": 1, "improved": ""},
    ])
    out, spans = bg.apply_bracket_grounding(TEXT, gateway=gw)
    assert "[" not in out and "]" not in out  # NO literal brackets in shipped text
    # qwen improved -> the generated sentence is present, marked 'improved' (green)
    assert "Employers in my district now ask for more than memorized facts." in out
    # qwen declined -> original kept, marked 'kept' (amber)
    assert "I want my students to become thoughtful and skeptical readers over time." in out
    assert {s["kind"] for s in spans} == {"improved", "kept"}
    # spans index the returned clean text correctly
    for s in spans:
        assert 0 <= s["start"] < s["end"] <= len(out)


def test_qwen_failure_falls_back_to_amber_kept(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_BRACKET_GROUNDING", "1")

    class _BoomGateway:
        def chat(self, *a, **k):
            raise RuntimeError("simulated qwen timeout in production")

    out, spans = bg.apply_bracket_grounding(TEXT, gateway=_BoomGateway())
    # qwen failed -> do NOT vanish; fall back to amber-kept spans on the generic sentences
    assert spans and all(s["kind"] == "kept" for s in spans)
    assert "[" not in out


def test_all_kept_when_model_declines(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_BRACKET_GROUNDING", "1")
    gw = _StubGateway([{"i": 0, "improved": ""}, {"i": 1, "improved": ""}])
    out, spans = bg.apply_bracket_grounding(TEXT, gateway=gw)
    assert "[" not in out
    assert spans and all(s["kind"] == "kept" for s in spans)
