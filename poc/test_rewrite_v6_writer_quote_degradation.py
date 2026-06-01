"""gpt-oss intermittently degrades a whole-paragraph rewrite into a run of quoted clauses
('"a," "b," "c."'). PROVEN by local stage-isolation: the stray straight-quotes appear in the WRITER
stage output (never the QC reviewer). _clean_candidate now treats that as not-clean, retries for a
real generation, and only as a last resort merges the clauses back to prose (+ a writer_quote_degraded
flag) -- never falling back to source_preserved. Legitimate inline dialogue ("got it") is untouched.

allow-hardcode: the multi-line strings below are real production-artifact samples (degraded model
output captured from R2 reports/d0a6f82b.../rewrite/rewrite.json), used as test fixtures -- not
matching logic or a phrase list.
"""
from __future__ import annotations

from types import SimpleNamespace

from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.text import Paragraph

# Real degraded writer output (P7/P8 of the production rewrite); original input had zero quotes.
PROD_P7 = (
    'Observing this lag in my classroom, I notice the curriculum often falls behind the skills '
    'employers expect, so I set aside ten minutes each week to revisit algebraic reasoning students '
    'need for work. "When I ran a unit on evaluating online sources, about half of my students cited '
    'news sites instead of random blogs they usually trust. "Following the assignment of problem '
    'sets," "I watch roughly three-quarters of them begin to question assumptions," "and their '
    'essays sharpen those arguments." "When designing projects," "I let learners pick topics that '
    'matter to them."'
)
PROD_P8 = (
    '"Entering my sophomore history class," "I notice that about three-quarters of the periods '
    'consist only of a teacher lecturing while students take notes," "even though many own tablets." '
    'Last semester I assigned a research project on the recent wildfires.'
)
LEGIT_DIALOGUE = (
    'Several students raised their hands and said they finally "got it" and could finish the next '
    'question on their own.'
)
WRAPPED_WHOLE = '"The whole paragraph was wrapped in a single pair of straight quotes by the model."'


def test_detects_quoted_clause_fragmentation():
    assert dr._is_quote_fragmented(PROD_P7) is True
    assert dr._is_quote_fragmented(PROD_P8) is True
    assert dr._is_quote_fragmented(WRAPPED_WHOLE) is True


def test_legit_inline_dialogue_not_flagged():
    assert dr._is_quote_fragmented(LEGIT_DIALOGUE) is False
    # and dequoting is a no-op safety: it is never called on non-fragmented text, but if it were,
    # the single inline pair has content between the quotes so the joint regex leaves it alone.


def test_dequote_merges_clauses_to_clean_prose():
    for sample in (PROD_P7, PROD_P8):
        out = dr._dequote_fragmented(sample)
        assert out.count('"') == 0                      # no stray straight quotes survive
        assert not dr._is_quote_fragmented(out)         # idempotent: cleaned text is not flagged
        assert "  " not in out                          # whitespace tidied
        assert " ," not in out and " ." not in out      # no space-before-punctuation artifacts
    # meaning words are preserved (we merge, not truncate)
    assert "question assumptions" in dr._dequote_fragmented(PROD_P7)
    assert "own tablets" in dr._dequote_fragmented(PROD_P8)


def test_dequote_strips_whole_wrap():
    out = dr._dequote_fragmented(WRAPPED_WHOLE)
    assert out.count('"') == 0
    assert out.startswith("The whole paragraph")


def _para():
    return Paragraph(id="p1", index=0, text="Original paragraph that needs a rewrite here.", sentences=[])


def _gw():
    return SimpleNamespace(chat=lambda *a, **k: SimpleNamespace(content="{}", raw_content="{}"))


def test_clean_candidate_retries_then_salvages_degraded(monkeypatch):
    # Every attempt returns a quote-fragmented generation -> no clean prose available -> salvage the
    # dequoted version with a writer_quote_degraded flag (NOT None / source_preserved).
    monkeypatch.setattr(dr, "_rewrite_paragraph", lambda *a, **k: (PROD_P8, []))
    monkeypatch.setattr(dr, "_is_usable", lambda c, p: bool(c and len(c.split()) >= 3))
    monkeypatch.setattr(dr, "_has_broken_grammar", lambda c: False)
    out, items = dr._clean_candidate(_gw(), _para(), None, [], attempts=2)
    assert out is not None and out.count('"') == 0
    assert any(i.get("type") == "writer_quote_degraded" for i in items)


def test_clean_candidate_prefers_clean_retry_over_salvage(monkeypatch):
    # First attempt degrades, second is clean -> the clean one wins; no degradation flag.
    seq = iter([(PROD_P8, []), ("A clean, well-formed rewrite of the paragraph in plain prose.", [])])
    monkeypatch.setattr(dr, "_rewrite_paragraph", lambda *a, **k: next(seq))
    monkeypatch.setattr(dr, "_is_usable", lambda c, p: bool(c and len(c.split()) >= 3))
    monkeypatch.setattr(dr, "_has_broken_grammar", lambda c: False)
    out, items = dr._clean_candidate(_gw(), _para(), None, [], attempts=2)
    assert out == "A clean, well-formed rewrite of the paragraph in plain prose."
    assert not any(i.get("type") == "writer_quote_degraded" for i in items)
