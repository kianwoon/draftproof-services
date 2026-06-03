from __future__ import annotations

import re
from types import SimpleNamespace

from poc.predictability import highlight_spans as hs


class _FakeScanner:
    def __init__(self, top10_terms: set[str]):
        self._top10_terms = top10_terms
        self.tokenizer = self._tokenizer

    def scan_sentence(self, sentence: str):
        tokens = _tokens(sentence)
        return SimpleNamespace(
            risk_label="high",
            token_results=[
                SimpleNamespace(top_10=token in self._top10_terms)
                for token, _start, _end in tokens
            ],
        )

    def _tokenizer(self, sentence: str, **_kwargs):
        return {"offset_mapping": [(0, 0)] + [(start, end) for _token, start, end in _tokens(sentence)]}


def _tokens(sentence: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’-]*", sentence)]


def test_predictability_highlights_keep_raw_but_hide_grammar_glue_runs():
    text = "I've seen the district roll out tutorials on fractions."
    scanner = _FakeScanner({"I've", "seen", "the", "tutorials", "on", "fractions"})

    out = hs.compute_predictability_highlights(text, scanner=scanner)

    raw = [text[start:end] for start, end in out["raw_words"]]
    actionable = [text[start:end] for start, end in out["words"]]
    assert "I've seen the" in raw
    assert "tutorials on fractions" in raw
    assert "I've seen the" not in actionable
    assert "tutorials on fractions" in actionable
    assert out["sentences"] == [[0, len(text)]]
    assert out["actionable_sentences"] == [[0, len(text)]]


def test_predictability_run_actionability_is_content_agnostic_not_phrase_based():
    assert hs._is_actionable_predictable_run("I've seen the") is False
    assert hs._is_actionable_predictable_run("to the") is False
    assert hs._is_actionable_predictable_run("implemented,") is False
    assert hs._is_actionable_predictable_run("at least two new") is False
    assert hs._is_actionable_predictable_run("lectures, and") is False
    assert hs._is_actionable_predictable_run("students bombarded") is True
    assert hs._is_actionable_predictable_run("scrambling to keep pace") is True
    assert hs._is_actionable_predictable_run("tutorials on fractions") is True


def test_grounding_gate_off_by_default_keeps_actionable(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_TOPK_GROUNDING_GATE", raising=False)  # code default is OFF
    text = "In 2023 I assigned tutorials on fractions."
    scanner = _FakeScanner({"tutorials", "on", "fractions"})

    out = hs.compute_predictability_highlights(text, scanner=scanner)

    assert "tutorials on fractions" in [text[s:e] for s, e in out["words"]]


def test_grounding_gate_suppresses_runs_in_grounded_sentence(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_TOPK_GROUNDING_GATE", "1")
    monkeypatch.setattr(hs, "_structurally_concrete", lambda t: False)  # isolate: only the digit grounds it
    text = "In 2023 I assigned tutorials on fractions."
    scanner = _FakeScanner({"tutorials", "on", "fractions"})

    out = hs.compute_predictability_highlights(text, scanner=scanner)

    assert "tutorials on fractions" in [text[s:e] for s, e in out["raw_words"]]  # raw audit trail kept
    assert out["words"] == []                 # sentence grounded by "2023" -> no actionable run shown
    assert out["actionable_sentences"] == []  # and the sentence is no longer shaded


def test_grounding_gate_drops_anchor_run_when_sentence_not_grounded(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_TOPK_GROUNDING_GATE", "1")
    monkeypatch.setattr(hs, "_structurally_concrete", lambda t: False)
    text = "They kept the textbook-lecture-quiz approach unchanged."
    scanner = _FakeScanner({"textbook-lecture-quiz", "approach"})

    out = hs.compute_predictability_highlights(text, scanner=scanner)

    actionable = [text[s:e] for s, e in out["words"]]
    assert all("textbook-lecture-quiz" not in a for a in actionable)  # hyphen-compound anchor dropped


def test_actionable_spans_expand_token_fragments_to_word_boundaries():
    text = "I ask them to compare scholarly articles before accepting the claim."
    raw_words: list[list[int]] = []
    actionable_words: list[list[int]] = []
    start = text.index("olarly")
    end = text.index("articles") + len("articles")

    hs._flush_run(
        text,
        0,
        [0, 1],
        [(0, 0), (start, start + len("olarly")), (text.index("articles"), end)],
        raw_words,
        actionable_words,
    )

    assert [text[start:end] for start, end in raw_words] == ["olarly articles"]
    assert [text[start:end] for start, end in actionable_words] == ["scholarly articles"]
