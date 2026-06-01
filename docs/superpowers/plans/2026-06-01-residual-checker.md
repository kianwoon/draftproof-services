# Residual Checker ("rewrite pass 2") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paragraph-level check on the rewriter's own output — re-scan the rewritten draft and re-run the writer on any paragraph that is still (or newly) flagged — slotted before the whole-doc reviewer.

**Architecture:** New `_apply_residual_fix(best_doc, …)` in `direct_rewrite.py`, called between best-of-N and `_apply_reviewer`. It scans `best_doc.rewritten_text` (NEVER the original), flags paragraphs via the FRESH re-scan `findings_for_paragraph` only (NOT `paragraph_diagnosis`, whose positional-id ContextVar still holds the original diagnosis), and re-writes flagged paragraphs with the existing `_clean_candidate` (diagnosis=None). Unflagged paragraphs keep their pass-1 text, so pass-1 gains are preserved.

**Tech Stack:** Python 3.11, pytest (`~/.pyenv/versions/3.11.0/bin/python -m pytest`), existing `poc/rewrite_v6` helpers.

---

## Background facts (verified — do not re-derive)

- `DocumentResult` (`poc/rewrite_v6/pipeline.py:87`) fields: `initial_scan, final_scan, passes, rewritten_text, pass_trace, final_text_before_quality_repair=None, quality_repair=None, naturalisation_repair=None`.
- `scan_text(text)` (`scan.py:58`) is PURE on its arg; findings derive from the text. Paragraph ids are positional `p{index+1:03d}`.
- `findings_for_paragraph(scan, pid)` (`scan.py:497`) returns `[f for f in scan.findings if f.paragraph_id == pid]`. A `Finding` has `.tags` and `.paragraph_id`.
- `_clean_candidate(gateway, paragraph, diagnosis, findings, *, attempts=2, authorship_targets=None) -> (str|None, list)` (`direct_rewrite.py:547`). Returns `None` candidate when no grammatically-clean rewrite — the no-regression fallback.
- `_trace(index, paragraph_id, source, reject_reason, review_items) -> dict` (`direct_rewrite.py:599`).
- `_review_flags(candidate, paragraph) -> list` (`direct_rewrite.py:287`).
- `paragraph_authorship_targets`, `authorship_boost_enabled` already imported in `direct_rewrite.py` and used by `_rewrite_document_once`.
- Already imported at `direct_rewrite.py:39`: `from .scan import Scan, findings_for_paragraph, scan_text`. `Paragraph` lives in `poc/rewrite_v6/text.py`.

---

### Task 1: Kill switch `residual_fix_enabled()`

**Files:**
- Modify: `poc/rewrite_v6/direct_rewrite.py` (add near the other env helpers, e.g. just above `_apply_residual_fix`)
- Test: `poc/test_rewrite_v6_residual_checker.py`

- [ ] **Step 1: Write the failing test**

```python
# poc/test_rewrite_v6_residual_checker.py
"""Residual checker = rewrite pass 2. Re-scans the REWRITTEN draft and re-runs the writer on
paragraphs the fresh re-scan flags; unflagged paragraphs keep their pass-1 text (invariant:
never revert to the original submitted text)."""
import os
from poc.rewrite_v6 import direct_rewrite


def test_kill_switch_default_on_and_off(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_RESIDUAL_FIX", raising=False)
    assert direct_rewrite.residual_fix_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", off)
        assert direct_rewrite.residual_fix_enabled() is False
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    assert direct_rewrite.residual_fix_enabled() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python -m pytest poc/test_rewrite_v6_residual_checker.py::test_kill_switch_default_on_and_off -v`
Expected: FAIL — `AttributeError: module 'poc.rewrite_v6.direct_rewrite' has no attribute 'residual_fix_enabled'`

- [ ] **Step 3: Write minimal implementation**

Add to `poc/rewrite_v6/direct_rewrite.py` (just above where `_apply_residual_fix` will go):

```python
def residual_fix_enabled() -> bool:
    """Kill switch for rewrite pass 2 (paragraph-level residual checker). Default ON; set
    DRAFTPROOF_V6_RESIDUAL_FIX=0 to disable (flow reverts to rewrite -> reviewer -> scan)."""
    return os.environ.get("DRAFTPROOF_V6_RESIDUAL_FIX", "1").strip().lower() not in {"0", "false", "no", "off"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.11.0/bin/python -m pytest poc/test_rewrite_v6_residual_checker.py::test_kill_switch_default_on_and_off -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add poc/test_rewrite_v6_residual_checker.py poc/rewrite_v6/direct_rewrite.py
git commit -m "feat(rewrite): residual_fix_enabled kill switch (rewrite pass 2)"
```

---

### Task 2: `_apply_residual_fix` — the dedicated pass-2 loop

**Files:**
- Modify: `poc/rewrite_v6/direct_rewrite.py` (add `_apply_residual_fix` directly below `residual_fix_enabled`)
- Test: `poc/test_rewrite_v6_residual_checker.py`

- [ ] **Step 1: Write the failing tests** (append to the test file)

```python
import types
from poc.rewrite_v6.text import Paragraph
from poc.rewrite_v6.scan import scan_text as real_scan_text
from poc.rewrite_v6.pipeline import DocumentResult


def _doc(rewritten_text, original_text="orig A.\n\norig B."):
    """A DocumentResult as produced by pass 1. initial_scan = ORIGINAL (for before/after diff)."""
    return DocumentResult(
        initial_scan=real_scan_text(original_text),
        final_scan=real_scan_text(rewritten_text),
        passes=[],
        rewritten_text=rewritten_text,
        pass_trace=[],
    )


def _fake_scan(paragraphs, recorder=None):
    def _scan_text(text):
        if recorder is not None:
            recorder.append(text)
        return types.SimpleNamespace(paragraphs=paragraphs)
    return _scan_text


def _patch(monkeypatch, *, paragraphs, flagged_ids, candidate_by_id, recorder=None):
    monkeypatch.setattr(direct_rewrite, "scan_text", _fake_scan(paragraphs, recorder))
    monkeypatch.setattr(
        direct_rewrite, "findings_for_paragraph",
        lambda scan, pid: [types.SimpleNamespace(tags=["generic_assertion_risk"], paragraph_id=pid)]
        if pid in flagged_ids else [],
    )
    monkeypatch.setattr(
        direct_rewrite, "_clean_candidate",
        lambda gateway, paragraph, diagnosis, findings, **kw: (candidate_by_id.get(paragraph.id), []),
    )


def test_invariant_unflagged_keeps_pass1_text_and_scans_rewritten(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    rewritten = "PARA_A_REWRITTEN.\n\nPARA_B_REWRITTEN."
    paras = [Paragraph(id="p001", index=0, text="PARA_A_REWRITTEN.", sentences=[]),
             Paragraph(id="p002", index=1, text="PARA_B_REWRITTEN.", sentences=[])]
    seen = []
    _patch(monkeypatch, paragraphs=paras, flagged_ids={"p001"},
           candidate_by_id={"p001": "PARA_A_REFIXED."}, recorder=seen)
    out = direct_rewrite._apply_residual_fix(_doc(rewritten), gateway=None, cancellation_check=None)
    # A (flagged) re-fixed; B (clean) keeps its PASS-1 text, NOT the original
    assert out.rewritten_text == "PARA_A_REFIXED.\n\nPARA_B_REWRITTEN."
    # the re-scan ran on the REWRITTEN draft, never the original
    assert seen and seen[0] == rewritten


def test_candidate_none_falls_back_to_pass1_paragraph(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    rewritten = "PARA_A_REWRITTEN.\n\nPARA_B_REWRITTEN."
    paras = [Paragraph(id="p001", index=0, text="PARA_A_REWRITTEN.", sentences=[]),
             Paragraph(id="p002", index=1, text="PARA_B_REWRITTEN.", sentences=[])]
    _patch(monkeypatch, paragraphs=paras, flagged_ids={"p001"}, candidate_by_id={"p001": None})
    out = direct_rewrite._apply_residual_fix(_doc(rewritten), gateway=None, cancellation_check=None)
    # no clean residual rewrite -> keep pass-1 text unchanged
    assert out.rewritten_text == rewritten


def test_noop_records_trace_and_preserves_initial_scan(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "1")
    rewritten = "CLEAN A.\n\nCLEAN B."
    paras = [Paragraph(id="p001", index=0, text="CLEAN A.", sentences=[]),
             Paragraph(id="p002", index=1, text="CLEAN B.", sentences=[])]
    _patch(monkeypatch, paragraphs=paras, flagged_ids=set(), candidate_by_id={})
    doc = _doc(rewritten)
    out = direct_rewrite._apply_residual_fix(doc, gateway=None, cancellation_check=None)
    assert out.rewritten_text == rewritten
    assert out.initial_scan is doc.initial_scan
    assert any(e.get("selected_source") == "residual_checker" for e in out.pass_trace)


def test_kill_switch_skips_rescan(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_RESIDUAL_FIX", "0")
    called = []
    monkeypatch.setattr(direct_rewrite, "scan_text", _fake_scan([], called))
    doc = _doc("X.\n\nY.")
    out = direct_rewrite._apply_residual_fix(doc, gateway=None, cancellation_check=None)
    assert out is doc          # unchanged
    assert called == []        # no re-scan performed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.11.0/bin/python -m pytest poc/test_rewrite_v6_residual_checker.py -v`
Expected: the 4 new tests FAIL — `AttributeError: ... has no attribute '_apply_residual_fix'`

- [ ] **Step 3: Write minimal implementation**

Add to `poc/rewrite_v6/direct_rewrite.py` directly below `residual_fix_enabled`:

```python
def _apply_residual_fix(
    doc,
    gateway,
    *,
    cancellation_check: Callable[[], None] | None,
    authorship_evidence: Any = None,
):
    """Rewrite pass 2: a paragraph-level check on the rewriter's own output.

    Re-scan the REWRITTEN draft (never the original) and re-run the writer on any paragraph the
    FRESH re-scan flags -- catching both residuals pass 1 missed and problems pass 1 introduced.
    Unflagged paragraphs keep their pass-1 text, so pass-1 gains are preserved (the load-bearing
    invariant). Flagging and rewriting drive off the fresh `findings_for_paragraph` ONLY; we pass
    diagnosis=None to `_clean_candidate` because `paragraph_diagnosis()` is a positional-id
    ContextVar still holding the ORIGINAL diagnosis (stale-leak guard, R1). On disable/any failure
    the document is returned unchanged."""
    from .pipeline import DocumentResult

    if not residual_fix_enabled():
        return doc
    try:
        residual_scan = scan_text(doc.rewritten_text)
    except Exception:
        return doc

    paragraphs = list(residual_scan.paragraphs)
    rewritten: list[str] = []
    trace = list(doc.pass_trace)
    refixed = 0
    flagged = 0
    for index, paragraph in enumerate(paragraphs):
        if cancellation_check:
            cancellation_check()
        findings = findings_for_paragraph(residual_scan, paragraph.id)
        if not findings:
            rewritten.append(paragraph.text)   # keep PASS-1 text (we scanned the rewritten draft)
            continue
        flagged += 1
        targets = (
            paragraph_authorship_targets(authorship_evidence, paragraph.text)
            if (authorship_evidence and authorship_boost_enabled())
            else {}
        )
        # diagnosis=None on purpose: fresh findings only, never stale original paragraph_diagnosis.
        candidate, review_items = _clean_candidate(
            gateway, paragraph, None, findings, authorship_targets=targets
        )
        if candidate is None:
            rewritten.append(paragraph.text)   # no clean residual fix -> keep pass-1 paragraph
        else:
            rewritten.append(candidate)
            refixed += 1
            trace.append(_trace(index, paragraph.id, "residual_fix", None,
                                review_items + _review_flags(candidate, paragraph)))

    if refixed == 0:
        # Nothing changed; record that the check ran (observability) and keep the doc as-is.
        trace.append({"selected_source": "residual_checker", "status": "checked",
                      "flagged_paragraphs": flagged, "refixed": 0})
        return DocumentResult(
            initial_scan=doc.initial_scan,
            final_scan=doc.final_scan,
            passes=doc.passes,
            rewritten_text=doc.rewritten_text,
            pass_trace=trace,
            final_text_before_quality_repair=doc.final_text_before_quality_repair,
            quality_repair=doc.quality_repair,
            naturalisation_repair=doc.naturalisation_repair,
        )

    fixed_text = "\n\n".join(rewritten)
    return DocumentResult(
        initial_scan=doc.initial_scan,            # ORIGINAL -> before/after diff intact
        final_scan=scan_text(fixed_text),         # provisional; _apply_reviewer recomputes authoritative
        passes=doc.passes,
        rewritten_text=fixed_text,
        pass_trace=trace,
        final_text_before_quality_repair=doc.final_text_before_quality_repair,
        quality_repair=doc.quality_repair,
        naturalisation_repair=doc.naturalisation_repair,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.0/bin/python -m pytest poc/test_rewrite_v6_residual_checker.py -v`
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add poc/test_rewrite_v6_residual_checker.py poc/rewrite_v6/direct_rewrite.py
git commit -m "feat(rewrite): _apply_residual_fix pass 2 (re-scan rewritten draft, fix residuals)"
```

---

### Task 3: Wire pass 2 into `run_direct_rewrite_all` (before the reviewer)

**Files:**
- Modify: `poc/rewrite_v6/direct_rewrite.py` (the `return _apply_reviewer(...)` line in `run_direct_rewrite_all`, ~line 408)
- Test: `poc/test_rewrite_v6_residual_checker.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_residual_fix_runs_before_reviewer(monkeypatch):
    """Order guard: in run_direct_rewrite_all, residual fix must execute before the reviewer."""
    order = []
    monkeypatch.setattr(direct_rewrite, "_best_of_n", lambda: 1)
    monkeypatch.setattr(direct_rewrite, "_rewrite_document_once",
                        lambda *a, **k: _doc("P1.\n\nP2."))
    monkeypatch.setattr(direct_rewrite, "_apply_residual_fix",
                        lambda doc, gateway, **k: (order.append("residual"), doc)[1])
    monkeypatch.setattr(direct_rewrite, "_apply_reviewer",
                        lambda doc, gateway, **k: (order.append("reviewer"), doc)[1])
    # Avoid real LLM/model resolution in run_direct_rewrite_all's gateway setup:
    monkeypatch.setattr(direct_rewrite, "LLMGateway", lambda *a, **k: None)
    direct_rewrite.run_direct_rewrite_all("some original text.\n\nsecond paragraph here.")
    assert order == ["residual", "reviewer"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python -m pytest poc/test_rewrite_v6_residual_checker.py::test_residual_fix_runs_before_reviewer -v`
Expected: FAIL — `order == ["reviewer"]` (residual fix not yet wired in), assertion error.

- [ ] **Step 3: Write minimal implementation**

In `run_direct_rewrite_all`, replace the final return:

```python
    # rewrite -> QC -> scan: the reviewer fixes whole-document patterns the per-paragraph writer
    # can't see, then the single authoritative final scan runs on the QC'd text.
    return _apply_reviewer(best_doc, gateway, cancellation_check=cancellation_check)
```

with:

```python
    # rewrite -> residual fix (pass 2) -> QC -> scan. Pass 2 re-scans the rewritten draft and fixes
    # paragraph-level residuals the per-paragraph writer missed or introduced; then the whole-doc
    # reviewer fixes cross-paragraph patterns; the authoritative final scan runs last, in the reviewer.
    best_doc = _apply_residual_fix(
        best_doc, gateway,
        cancellation_check=cancellation_check,
        authorship_evidence=authorship_evidence,
    )
    return _apply_reviewer(best_doc, gateway, cancellation_check=cancellation_check)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.11.0/bin/python -m pytest poc/test_rewrite_v6_residual_checker.py::test_residual_fix_runs_before_reviewer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add poc/test_rewrite_v6_residual_checker.py poc/rewrite_v6/direct_rewrite.py
git commit -m "feat(rewrite): wire residual checker before reviewer in run_direct_rewrite_all"
```

---

### Task 4: Full-suite regression + sanity

**Files:** none (verification only)

- [ ] **Step 1: Run the residual checker + adjacent rewrite tests**

Run:
```bash
~/.pyenv/versions/3.11.0/bin/python -m pytest \
  poc/test_rewrite_v6_residual_checker.py \
  poc/test_rewrite_v6_document_reviewer.py \
  poc/test_external_estimate_combine.py -q
```
Expected: all PASS (no regression in the reviewer/scan tests).

- [ ] **Step 2: Sanity-scan a real rewrite path import**

Run:
```bash
~/.pyenv/versions/3.11.0/bin/python -c "import sys; sys.path.insert(0,'poc'); from rewrite_v6.direct_rewrite import _apply_residual_fix, residual_fix_enabled; print('wired:', residual_fix_enabled())"
```
Expected: `wired: True`

- [ ] **Step 3: Commit (if any incidental fixups)**

```bash
git add -A && git commit -m "test(rewrite): residual checker full-suite green" || echo "nothing to commit"
```

---

## Self-review notes

- **Spec coverage:** flow (Task 3), invariant pass-1-preserved + scans-rewritten-not-original (Task 2 test 1), re-fix residual (Task 2), no-op + trace (Task 2 test 3), kill switch (Tasks 1–2), `initial_scan` preserved (Task 2 test 3), R1 stale-diagnosis guard (`diagnosis=None`, Task 2 impl + comment). Single pass: no loop in `_apply_residual_fix`. ✓
- **R1 resolution baked in:** flag off fresh `findings_for_paragraph`, `_clean_candidate(diagnosis=None)`, never call `paragraph_diagnosis`. ✓
- **Out of scope honored:** no scoring change, no external-estimate/evasion work, no loop. ✓
