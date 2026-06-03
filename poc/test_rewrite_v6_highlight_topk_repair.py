from __future__ import annotations

import json
from types import SimpleNamespace

from poc.rewrite_v6 import highlight_topk_repair as htr


def _stub(payload: dict):
    class _Gateway:
        def __init__(self):
            self.calls = []

        def chat(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
            if kwargs.get("app_label") == "HighlightTopkRepairVerifier":
                raw = json.dumps({"grammar_safe": True, "meaning_safe": True, "natural_sentence": True})
            else:
                raw = json.dumps(payload)
            return SimpleNamespace(content=raw, raw_content=raw)

    return _Gateway()


def _stub_sequence(payloads: list[dict]):
    class _Gateway:
        def __init__(self):
            self.calls = []

        def chat(self, prompt, **kwargs):
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
            payload = payloads[min(len(self.calls) - 1, len(payloads) - 1)]
            raw = json.dumps(payload)
            return SimpleNamespace(content=raw, raw_content=raw)

    return _Gateway()


def test_highlight_topk_repair_targets_exact_highlighted_sentence(monkeypatch):
    sentence = "Curriculum guides shift every five months, so my school scrambles to redesign lessons."
    low = "Students then compare two local examples."
    text = sentence + " " + low
    replacement = "Every five months the curriculum guide shifts, leaving my school to redesign lessons."
    highlights = {
        "sentences": [[0, len(sentence)]],
        "words": [[24, 41], [49, 59]],
    }
    monkeypatch.setattr(htr, "_sentence_topk", lambda value: 0.40 if value == replacement else 0.80)
    gateway = _stub({"alternatives": [replacement]})

    out, result = htr.apply_highlight_topk_repair(text, highlights, gateway=gateway)

    assert replacement in out
    assert low in out
    assert result.changed is True
    assert result.applied[0]["original"] == sentence
    assert result.applied[0]["topk_before"] == 0.80
    assert result.applied[0]["topk_after"] == 0.40


def test_highlight_topk_repair_rejects_sentence_split(monkeypatch):
    sentence = "Curriculum guides shift every five months, so my school scrambles to redesign lessons."
    split = "Curriculum guides shift every five months. My school scrambles to redesign lessons."
    highlights = {"sentences": [[0, len(sentence)]], "words": [[24, 41]]}
    monkeypatch.setattr(htr, "_sentence_topk", lambda value: 0.20 if value == split else 0.80)

    out, result = htr.apply_highlight_topk_repair(sentence, highlights, gateway=_stub({"alternatives": [split]}))

    assert out == sentence
    assert result.changed is False
    assert result.skipped[0]["reason"] == "no_safe_lower_topk_option"


def test_highlight_topk_repair_rejects_non_lowering_option(monkeypatch):
    sentence = "Students copy key points into notebooks after a forty-five-minute lecture."
    replacement = "Students copy central notes into notebooks after a forty-five-minute lecture."
    highlights = {"sentences": [[0, len(sentence)]], "words": [[0, 13]]}
    monkeypatch.setattr(htr, "_sentence_topk", lambda _value: 0.70)

    out, result = htr.apply_highlight_topk_repair(sentence, highlights, gateway=_stub({"alternatives": [replacement]}))

    assert out == sentence
    assert result.changed is False
    assert result.skipped[0]["reason"] == "no_safe_lower_topk_option"


def test_highlight_topk_repair_rejects_lower_topk_when_verifier_flags_awkward(monkeypatch):
    sentence = "When I hear debates each fall, I see the system standing at a crossroads."
    awkward = "When I hear debates each fall, the system standing at a crossroads is what I see."
    highlights = {"sentences": [[0, len(sentence)]], "words": [[0, 11]]}
    monkeypatch.setattr(htr, "_sentence_topk", lambda value: 0.30 if value == awkward else 0.70)
    gateway = _stub_sequence([
        {"alternatives": [awkward]},
        {"grammar_safe": False, "meaning_safe": True, "natural_sentence": False, "reason": "awkward clause order"},
    ])

    out, result = htr.apply_highlight_topk_repair(sentence, highlights, gateway=gateway)

    assert out == sentence
    assert result.changed is False
    assert result.skipped[0]["reason"] == "no_safe_lower_topk_option"


def test_highlight_topk_repair_targets_word_run_sentence_when_no_sentence_span(monkeypatch):
    first = "Students copy key points into notebooks after a forty-five-minute lecture."
    second = "They later compare the result with a local example."
    text = first + " " + second
    replacement = "After a forty-five-minute lecture, students copy key points into notebooks."
    start = text.index("key points")
    highlights = {"sentences": [], "words": [[start, start + len("key points")]]}
    monkeypatch.setattr(htr, "_sentence_topk", lambda value: 0.35 if value == replacement else 0.70)

    out, result = htr.apply_highlight_topk_repair(text, highlights, gateway=_stub({"alternatives": [replacement]}))

    assert replacement in out
    assert second in out
    assert result.changed is True


def test_highlight_topk_repair_skips_raw_only_false_alarm_spans(monkeypatch):
    sentence = "I've seen the district roll out new standards before spring."
    highlights = {
        "sentences": [[0, len(sentence)]],
        "actionable_sentences": [],
        "words": [],
        "actionable_words": [],
        "raw_words": [[0, len("I've seen the")]],
    }
    monkeypatch.setattr(htr, "_sentence_topk", lambda _value: 0.80)
    gateway = _stub({"alternatives": ["The district has rolled out new standards before spring, as I have seen."]})

    out, result = htr.apply_highlight_topk_repair(sentence, highlights, gateway=gateway)

    assert out == sentence
    assert result.changed is False
    assert gateway.calls == []
