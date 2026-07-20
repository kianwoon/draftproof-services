"""Over-length rewrite candidates (adds grounding content beyond the ~1.75x source-word ratio) used
to be a hard reject inside `_clean_candidate` -> `_paragraph_outcome` fell back to source_preserved,
the WORST outcome per the rewrite objective (CLAUDE.md): the user sees no solution and learns
nothing. Over-length is not a meaning/grammar defect -- it's the writer doing exactly what the
objective asks (filling the grounding gap). Fix: the bound still triggers a retry on non-final
attempts, but on the FINAL attempt an otherwise-clean, non-stub over-length candidate SHIPS with a
`rewrite_over_length` review flag instead of falling back. Under-length (stub territory) keeps its
existing hard-reject behavior unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.candidate_bounds import candidate_word_bounds
from poc.rewrite_v6.text import Paragraph


def _para():
    # 10 source words -> lower=8, upper=18 (max(8, int(10*0.55))=8; max(12, int(10*1.75)+1)=18).
    text = "Original paragraph that needs a rewrite here for the test."
    assert len(text.split()) == 10
    return Paragraph(id="p1", index=0, text=text, sentences=[])


def _gw():
    return SimpleNamespace(chat=lambda *a, **k: SimpleNamespace(content="{}", raw_content="{}"))


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_bounds_sanity():
    lower, upper = candidate_word_bounds(_para())
    assert (lower, upper) == (8, 18)


def _patch_permissive(monkeypatch, *, broken_grammar=False):
    monkeypatch.setattr(dr, "_is_usable", lambda c, p: bool(c and len(c.split()) >= 3))
    monkeypatch.setattr(dr, "_has_broken_grammar", lambda c: broken_grammar)
    monkeypatch.setattr(dr, "_severe_polarity_inversion", lambda c, p: False)
    monkeypatch.setattr(dr, "_severe_beat_loss", lambda c, p: False)
    monkeypatch.setattr(dr, "_is_quote_fragmented", lambda c: False)


def test_overlength_final_attempt_ships_with_flag(monkeypatch):
    long_candidate = _words(25)  # > upper(18) on every attempt
    monkeypatch.setattr(dr, "_rewrite_paragraph", lambda *a, **k: (long_candidate, []))
    _patch_permissive(monkeypatch)
    out, items = dr._clean_candidate(_gw(), _para(), None, [], attempts=2)
    assert out == long_candidate
    assert any(i.get("type") == "rewrite_over_length" for i in items)


def test_broken_overlength_still_falls_back(monkeypatch):
    long_candidate = _words(25)
    monkeypatch.setattr(dr, "_rewrite_paragraph", lambda *a, **k: (long_candidate, []))
    _patch_permissive(monkeypatch, broken_grammar=True)  # genuinely broken on every attempt
    out, items = dr._clean_candidate(_gw(), _para(), None, [], attempts=2)
    assert out is None
    assert items == []


def test_underlength_still_hard_reject(monkeypatch):
    short_candidate = _words(5)  # usable (>=3) but < lower(8)
    monkeypatch.setattr(dr, "_rewrite_paragraph", lambda *a, **k: (short_candidate, []))
    _patch_permissive(monkeypatch)
    out, items = dr._clean_candidate(_gw(), _para(), None, [], attempts=2)
    assert out is None
    assert items == []


def test_overlength_then_fits_on_retry_has_no_flag(monkeypatch):
    long_candidate = _words(25)
    fit_candidate = _words(12)  # within [8, 18]
    seq = iter([(long_candidate, []), (fit_candidate, [])])
    monkeypatch.setattr(dr, "_rewrite_paragraph", lambda *a, **k: next(seq))
    _patch_permissive(monkeypatch)
    out, items = dr._clean_candidate(_gw(), _para(), None, [], attempts=2)
    assert out == fit_candidate
    assert not any(i.get("type") == "rewrite_over_length" for i in items)
