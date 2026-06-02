"""Grammar-repair pass: fixes ONLY grammar (dropped articles, agreement, telegraphic) and the
content-preservation gate rejects anything that rewords content. gpt-oss is mocked; the gate (stemmed
content multiset + numbers) is exercised for real.

The sample sentences below are test fixtures for the gate, not matching logic.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from poc.rewrite_v6 import grammar_repair as gr

BROKEN = "My challenge, where students prototype water filter, has sparked real creativity."
FIXED = "My challenge, where students prototype a water filter, has sparked real creativity."
REWORDED = "My challenge, where students build a robust filtration system, has boosted innovation."


def _stub(payload: str):
    class _G:
        def chat(self, prompt, **kwargs):
            return SimpleNamespace(content=payload, raw_content=payload)
    return _G()


def test_enabled_default_on(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_GRAMMAR_REPAIR", raising=False)
    assert gr.grammar_repair_enabled() is True
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMAR_REPAIR", "0")
    assert gr.grammar_repair_enabled() is False


def test_gate_accepts_article_insertion():
    assert gr._is_grammar_only(BROKEN, FIXED) is True


def test_gate_accepts_the_and_an_insertion():
    # the dominant dropped-article fixes -- a strict >=2-letter-equality gate wrongly rejected these
    assert gr._is_grammar_only("Compare three articles on same topic today.",
                               "Compare three articles on the same topic today.") is True
    assert gr._is_grammar_only("I assign essay about climate policy each week.",
                               "I assign an essay about climate policy each week.") is True


def test_gate_rejects_short_content_word_swap():
    # a 2-3 letter content word swapped (dog->cat) is NOT an article insertion -> still rejected
    assert gr._is_grammar_only("I keep a dog at the home office daily.",
                               "I keep a cat at the home office daily.") is False


def test_gate_accepts_agreement_fix():
    assert gr._is_grammar_only("The data show clear trends.", "The data shows clear trends.") is True


def test_gate_rejects_reworded_content():
    assert gr._is_grammar_only(BROKEN, REWORDED) is False           # content changed -> reject


def test_gate_rejects_number_change():
    assert gr._is_grammar_only("I taught 12 students last term.", "I taught 15 students last term.") is False


def test_gate_rejects_short_scope_and_negation_changes():
    assert gr._is_grammar_only("The method is useful.", "The method is not useful.") is False
    assert gr._is_grammar_only("The task measures skill only.", "The task measures skill.") is False
    assert gr._is_grammar_only("The model supports both speed and quality.", "The model supports speed and quality.") is False


def test_gate_rejects_identical():
    assert gr._is_grammar_only(BROKEN, BROKEN) is False             # nothing changed


def test_gate_rejects_punctuation_only_polish():
    assert gr._is_grammar_only("Each semester my class checks drafts.",
                               "Each semester, my class checks drafts.") is False
    assert gr._is_grammar_only("The thirty‑page packet arrived today.",
                               "The thirty-page packet arrived today.") is False


def test_apply_repair_splices_only_grammar_fix():
    clean = "Students improve water quality through careful testing."
    text = BROKEN + " " + clean
    payload = json.dumps({"fixed": [{"i": 0, "text": FIXED}, {"i": 1, "text": clean}]})
    out, applied = gr.apply_repair(text, gateway=_stub(payload))
    assert "prototype a water filter" in out                        # grammar fixed
    assert clean in out                                             # clean sentence untouched
    assert len(applied) == 1 and applied[0]["revised"] == FIXED


def test_apply_repair_rejects_reword():
    payload = json.dumps({"fixed": [{"i": 0, "text": REWORDED}]})
    out, applied = gr.apply_repair(BROKEN, gateway=_stub(payload))
    assert applied == [] and out == BROKEN                          # reword rejected, original kept


def test_apply_repair_disabled_noop(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_GRAMMAR_REPAIR", "0")
    out, applied = gr.apply_repair(BROKEN, gateway=_stub("{}"))
    assert applied == [] and out == BROKEN


def test_apply_repair_bad_json_fails_open():
    out, applied = gr.apply_repair(BROKEN, gateway=_stub("not json"))
    assert applied == [] and out == BROKEN


# --- wiring: mutates + re-scans + traces ---
from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6 import llm_config
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text


def test_apply_grammar_repair_mutates_and_rescans():
    clean = "Students improve water quality through careful testing every week."
    txt = BROKEN + " " + clean
    doc = DocumentResult(initial_scan=scan_text(txt), final_scan=scan_text(txt),
                         rewritten_text=txt, passes=[], pass_trace=[])
    payload = json.dumps({"fixed": [{"i": 0, "text": FIXED}, {"i": 1, "text": clean}]})
    out = dr._apply_grammar_repair(doc, _stub(payload), cancellation_check=None)
    assert "prototype a water filter" in out.rewritten_text
    assert out.final_scan.source_text == out.rewritten_text
    assert any(r.get("selected_source") == "grammar_repair" and r.get("applied") == 1 for r in out.pass_trace)


def test_apply_grammar_repair_uses_role_gateway_when_configured(monkeypatch):
    clean = "Students improve water quality through careful testing every week."
    txt = BROKEN + " " + clean
    doc = DocumentResult(initial_scan=scan_text(txt), final_scan=scan_text(txt),
                         rewritten_text=txt, passes=[], pass_trace=[])
    fallback = _stub(json.dumps({"fixed": [{"i": 0, "text": REWORDED}]}))
    role_gateway = _stub(json.dumps({"fixed": [{"i": 0, "text": FIXED}, {"i": 1, "text": clean}]}))
    monkeypatch.setattr(llm_config, "grammar_gateway", lambda **_kwargs: role_gateway)

    out = dr._apply_grammar_repair(doc, fallback, api_key=None, base_url=None, cancellation_check=None)

    assert "prototype a water filter" in out.rewritten_text
    assert REWORDED not in out.rewritten_text
