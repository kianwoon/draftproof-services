"""Tests for the Phase-2 LLM-judged dimensions (fake gateway, no real LLM)."""
import json

import pytest

from detect.critical_thinking_llm import (
    ALLOWED_LABELS,
    assess_critical_thinking,
    critical_thinking_llm_enabled,
    critical_thinking_questions_enabled,
    generate_reflective_questions,
    _weak_dimensions,
    _questions_prompt,
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
        self.last_kwargs = None

    def chat(self, prompt, **kwargs):
        self.last_prompt = prompt
        self.last_kwargs = kwargs
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


# ── Reflective questions ──────────────────────────────────────────────────────

_REPORT_Q = {
    "scan_intelligence": {"document": {"paragraphs": [
        {"paragraph_id": "p001", "text": "AI can improve learning by providing personalised support."},
        {"paragraph_id": "p002", "text": "There are many benefits and challenges to using AI in school."},
    ]}},
    "ai_risk_badge": {"critical_thinking_control": {"dimensions": {
        "specific_context": {"control": 17.5},
        "student_judgement": {"control": 36.0},
        "evidence_grounding": {"control": 65.0},
        "ai_dependency": {"control": 100.0},   # lead-ineligible -> excluded from steering
        "reasoning_trail": {"control": 84.0},
    }}},
}

_GOOD_Q = json.dumps({"questions": [
    {"dimension": "specific_context",
     "anchor_quote": "AI can improve learning by providing personalised support.",
     "question": "When did personalised support actually help you?"},
    {"dimension": "evidence_grounding",
     "anchor_quote": "There are many benefits and challenges",  # substring of p002 -> anchored
     "question": "Name one concrete benefit and one concrete risk."},
    {"dimension": "specific_context",
     "anchor_quote": "a fabricated sentence never written in the draft",  # NOT in corpus -> dropped
     "question": "ghost question"},
    {"dimension": "mystery_dim",
     "anchor_quote": "AI can improve learning",  # anchored; invalid dim -> normalized to ""
     "question": "What do you mean by improve?"},
]})


def test_questions_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_CRITICAL_THINKING_QUESTIONS", raising=False)
    assert critical_thinking_questions_enabled() is False
    assert generate_reflective_questions(_REPORT_Q, gateway=_FakeGateway(_GOOD_Q)) is None


def test_questions_parse_and_anchor_guard(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_QUESTIONS", "1")
    out = generate_reflective_questions(_REPORT_Q, gateway=_FakeGateway(_GOOD_Q))
    assert out is not None
    qs = out["questions"]
    # fabricated-quote question dropped; 3 anchored survive
    assert len(qs) == 3
    assert all(q["anchor_quote"] for q in qs) and all(q["question"] for q in qs)
    # invalid dimension normalized to ""
    assert qs[2]["dimension"] == ""
    # no fabricated quote survived
    assert not any("fabricated" in q["anchor_quote"] for q in qs)


def test_questions_fail_open_bad_json(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_QUESTIONS", "1")
    assert generate_reflective_questions(_REPORT_Q, gateway=_FakeGateway("not json <<<")) is None


def test_questions_none_without_weak_dimensions(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_QUESTIONS", "1")
    rep = {"scan_intelligence": _REPORT_Q["scan_intelligence"], "ai_risk_badge": {}}
    assert generate_reflective_questions(rep, gateway=_FakeGateway(_GOOD_Q)) is None


def test_questions_prompt_is_context_aware():
    weak = [("specific_context", "Specific context",
             "anchor claims to a real assignment, case, example, or observation")]
    rows = [{"paragraph_id": "p001",
             "text": "We've all seen the headlines about a 3.3% adoption rate."}]
    p = _questions_prompt(rows, weak)
    # the biasing personal-anecdote ACTION must NOT be injected as steering
    assert "anchor claims to a real assignment" not in p
    # context-awareness instruction present
    assert "HOW each claim is grounded" in p
    assert "headlines" in p.lower()
    assert "do not ask for a personal" in p.lower()
    # contract intact
    assert "specific_context" in p
    assert "anchor_quote" in p


def test_weak_dimensions_excludes_ai_dependency_and_sorts():
    weak = _weak_dimensions(_REPORT_Q)
    codes = [c for c, _l, _a in weak]
    assert "ai_dependency" not in codes            # lead-ineligible never steers
    assert codes[0] == "specific_context"          # weakest (control 17.5) first
    assert codes == ["specific_context", "student_judgement", "evidence_grounding"]


# ── Prompt-injection parity (untrusted-data clause) ────────────────────────────
# Both system prompts send the student's own document text to an LLM (json.dumps-escaped,
# no structural breakout possible). They must still tell the model to treat that text as
# data, never as instructions -- mirroring detect/defence_judge.py's _SYSTEM. We assert on
# the actual system kwarg SENT to the gateway, not on the module constant, so the test
# fails if the clause is ever dropped from the call site.

def test_dimensions_system_prompt_declares_untrusted_data(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_LLM", "1")
    gw = _FakeGateway(_GOOD)
    assess_critical_thinking(_REPORT, gateway=gw)
    system = gw.last_kwargs["system"].lower()
    assert "untrusted data" in system
    assert "never" in system and "instruction" in system


def test_questions_system_prompt_declares_untrusted_data(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_CRITICAL_THINKING_QUESTIONS", "1")
    gw = _FakeGateway(_GOOD_Q)
    generate_reflective_questions(_REPORT_Q, gateway=gw)
    system = gw.last_kwargs["system"].lower()
    assert "untrusted data" in system
    assert "never" in system and "instruction" in system
