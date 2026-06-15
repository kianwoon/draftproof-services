"""Tests for the Phase-2 LLM-judged dimensions (fake gateway, no real LLM)."""
import json

import pytest

from detect.critical_thinking_llm import (
    ALLOWED_LABELS,
    assess_critical_thinking,
    critical_thinking_llm_enabled,
)

_REPORT = {
    "scan_intelligence": {
        "document": {
            "paragraphs": [
                {"paragraph_id": "p001", "text": "AI can improve learning by providing personalised support."},
                {"paragraph_id": "p002", "text": "There are many benefits and challenges to using AI in school."},
            ]
        }
    }
}


class _Resp:
    def __init__(self, raw):
        self.raw_content = raw
        self.content = raw


class _FakeGateway:
    model = "fake/judge"

    def __init__(self, raw):
        self._raw = raw
        self.last_prompt = None

    def chat(self, prompt, **kwargs):
        self.last_prompt = prompt
        return _Resp(self._raw)


_GOOD = json.dumps({
    "paragraphs": [
        {"paragraph_id": "p001", "alternative_comparison": 30, "reflection": 20},
        {"paragraph_id": "p002", "alternative_comparison": 45, "reflection": 150},  # over-range -> clamp 100
    ],
    "highlights": [
        {"paragraph_id": "p001", "sentence": "AI can improve learning by providing personalised support.",
         "label": "single_path_answer", "severity": "high",
         "why_it_matters": "one clean benefit, no limitation", "fix_instruction": "add a limitation or alternative"},
        {"paragraph_id": "p001", "sentence": "junk", "label": "not_a_real_label", "severity": "high"},  # bad label -> dropped
        {"paragraph_id": "p999", "sentence": "ghost", "label": "no_judgement", "severity": "low"},      # bad id -> dropped
    ],
})


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_CRITICAL_THINKING_LLM", raising=False)
    assert critical_thinking_llm_enabled() is False
    assert assess_critical_thinking(_REPORT, gateway=_FakeGateway(_GOOD)) is None


def test_enabled_parses_dimensions_and_highlights(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_LLM", "1")
    out = assess_critical_thinking(_REPORT, gateway=_FakeGateway(_GOOD))
    assert out is not None
    dims = out["llm_dimensions"]
    # alternative_comparison mean of (30, 45) = 37.5
    assert dims["alternative_comparison"]["score"] == 37.5
    # reflection mean of (20, clamp(150)->100) = 60.0
    assert dims["reflection"]["score"] == 60.0
    # only the one valid highlight survives (bad label + bad id dropped)
    assert len(out["highlights"]) == 1
    h = out["highlights"][0]
    assert h["label"] in ALLOWED_LABELS
    assert h["paragraph_id"] == "p001"
    assert h["signal_category"] == "critical_thinking"


def test_fail_open_on_bad_json(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_LLM", "1")
    assert assess_critical_thinking(_REPORT, gateway=_FakeGateway("not json at all <<<")) is None


def test_none_when_no_paragraphs(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_LLM", "1")
    assert assess_critical_thinking({"scan_intelligence": {"document": {"paragraphs": []}}},
                                    gateway=_FakeGateway(_GOOD)) is None


def test_prompt_forbids_phrase_matching(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_LLM", "1")
    gw = _FakeGateway(_GOOD)
    assess_critical_thinking(_REPORT, gateway=gw)
    # NO-HARDCODE: the judge must be told to reason about meaning, not match phrases.
    assert "do not pattern-match" in gw.last_prompt.lower()
    assert "verbatim" in gw.last_prompt.lower()


def test_empty_results_when_all_invalid(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_LLM", "1")
    bad = json.dumps({"paragraphs": [{"paragraph_id": "ghost", "alternative_comparison": 10}], "highlights": []})
    # no valid paragraph dims and no highlights -> None
    assert assess_critical_thinking(_REPORT, gateway=_FakeGateway(bad)) is None
