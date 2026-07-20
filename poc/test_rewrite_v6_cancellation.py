"""Regression coverage for a review finding: "cancellation raised inside a gateway call gets
swallowed by _clean_candidate's generic `except Exception`, degrading the paragraph to
source_preserved instead of propagating."

INVESTIGATION (see worker/app/tasks.py `RewriteCanceled`): cancellation is NOT a polled boolean --
`cancellation_check` is a callable that RAISES a dedicated exception, and that exception is defined
as `class RewriteCanceled(BaseException)` specifically *so that* the broad `except Exception:` guards
sprinkled through the rewrite pipeline (including every writer-call try/except in
poc/rewrite_v6/direct_rewrite.py: lines ~158, ~195, ~633, ~921/924/928, ~971, ~995, and the
rate-limit-retry catch in `_rewrite_paragraph_task`) cannot catch it -- `except Exception` never
matches a BaseException that isn't an Exception subclass. Verified empirically: `except Exception`
lets a `BaseException`-only subclass fall straight through, and `ThreadPoolExecutor.Future.result()`
faithfully re-raises a BaseException raised inside the worker thread's target function, so it also
survives the `_rewrite_paragraphs_parallel` pool loop's `except BaseException` re-raise.

No exception-handling change was needed in direct_rewrite.py -- these tests lock the *existing*
"BaseException bypasses except Exception" contract in place so a future refactor (e.g. someone
"simplifying" `except Exception as exc: ... else return None, []` into `except BaseException`, or
changing the cancellation signal to inherit from Exception) cannot silently reintroduce the swallow
bug the review was worried about.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.text import Paragraph


class _Canceled(BaseException):
    """Test double mirroring worker/app/tasks.py::RewriteCanceled (BaseException, not Exception)."""


def _para():
    text = "Original paragraph that needs a rewrite here for the test."
    return Paragraph(id="p1", index=0, text=text, sentences=[])


def _gw():
    return SimpleNamespace(chat=lambda *a, **k: SimpleNamespace(content="{}", raw_content="{}"))


def test_clean_candidate_does_not_swallow_cancellation(monkeypatch):
    """A BaseException-only cancellation raised from the writer call must propagate out of
    `_clean_candidate`, not be absorbed into a (None, []) source_preserved result."""
    monkeypatch.setattr(dr, "_rewrite_paragraph", lambda *a, **k: (_ for _ in ()).throw(_Canceled()))
    with pytest.raises(_Canceled):
        dr._clean_candidate(_gw(), _para(), None, [], attempts=2)


def test_rewrite_paragraph_task_propagates_cancellation_not_source_preserved(monkeypatch):
    """`_rewrite_paragraph_task`'s rate-limit retry catch (`except Exception as exc`) must not
    degrade a mid-call cancellation to the `(None, [])` -> source_preserved fallback."""
    monkeypatch.setattr(dr, "_clean_candidate", lambda *a, **k: (_ for _ in ()).throw(_Canceled()))
    with pytest.raises(_Canceled):
        dr._rewrite_paragraph_task(_gw(), _para(), None, [], {}, "control", None)


def test_residual_fix_writer_catch_does_not_swallow_cancellation(monkeypatch):
    """`_apply_residual_fix`'s per-paragraph `except Exception:` (direct_rewrite.py ~L195) must let
    a cancellation raised inside `_clean_candidate` propagate rather than degrading to pass-1 text."""
    monkeypatch.setattr(dr, "residual_fix_enabled", lambda: True)
    monkeypatch.setattr(
        dr, "scan_text_preserve_blocks", lambda text: SimpleNamespace(paragraphs=[_para()])
    )
    monkeypatch.setattr(dr, "_residual_findings", lambda scan, paragraph: ["some_finding"])
    monkeypatch.setattr(dr, "_clean_candidate", lambda *a, **k: (_ for _ in ()).throw(_Canceled()))

    doc = SimpleNamespace(
        rewritten_text="x",
        initial_scan=SimpleNamespace(paragraphs=[_para()]),
        pass_trace=[],
    )
    with pytest.raises(_Canceled):
        dr._apply_residual_fix(doc, _gw(), cancellation_check=None)


def test_parallel_pool_reraises_cancellation_from_worker_thread(monkeypatch):
    """End-to-end: cancellation raised inside the ThreadPoolExecutor worker (via `_clean_candidate`)
    must surface through `_rewrite_paragraphs_parallel`'s `except BaseException` re-raise, not be
    absorbed as a per-paragraph writer failure."""
    monkeypatch.setattr(dr, "_clean_candidate", lambda *a, **k: (_ for _ in ()).throw(_Canceled()))
    monkeypatch.setattr(dr, "findings_for_paragraph", lambda scan, pid: ["some_finding"])
    monkeypatch.setattr(dr, "paragraph_diagnosis", lambda pid: {"issue": "x"})

    scan = SimpleNamespace()
    with pytest.raises(_Canceled):
        dr._rewrite_paragraphs_parallel(
            scan, _gw(), [_para()], flagged_total=1, concurrency=2,
            progress_callback=None, cancellation_check=None,
            authorship_evidence=None, lane="control",
        )
