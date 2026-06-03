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


def test_single_bracket_when_model_improves(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_BRACKET_GROUNDING", "1")
    gw = _StubGateway([
        {"i": 0, "improved": "Employers in my district now ask for more than memorized facts."},
        {"i": 1, "improved": ""},
    ])
    out, applied = bg.apply_bracket_grounding(TEXT, gateway=gw)
    assert "[Employers in my district now ask for more than memorized facts.]" in out  # single bracket
    assert "[[I want my students to become thoughtful and skeptical readers over time.]]" in out  # double bracket
    kinds = {a["bracket"] for a in applied}
    assert kinds == {"single", "double"}


def test_double_bracket_when_model_declines(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_BRACKET_GROUNDING", "1")
    gw = _StubGateway([{"i": 0, "improved": ""}, {"i": 1, "improved": ""}])
    out, applied = bg.apply_bracket_grounding(TEXT, gateway=gw)
    assert all(a["bracket"] == "double" for a in applied) and applied
    assert "[[" in out
