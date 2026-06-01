# Residual Checker — "rewrite pass 2" (paragraph-level QC on the rewriter)

**Date:** 2026-06-01
**Status:** Design — awaiting user review
**Component:** `poc/rewrite_v6/direct_rewrite.py` (lean direct path, the production default)

---

## 1. Problem

The lean rewrite path has two QC tiers with different scopes, and a hole between them:

| Tier | Scope | Catches | Exists? |
|---|---|---|---|
| **Rewriter** | paragraph (local) | the *original's* flagged issues | ✅ |
| **Residual checker** | **paragraph (local)** | problems the rewriter **MISSED** + problems it **INTRODUCED** | ❌ **missing** |
| **Reviewer** (`document_reviewer`) | whole-doc (big picture) | cross-paragraph patterns (monoculture, rhythm, transitions) | ✅ |

The rewriter writes one paragraph at a time against the **original** scan's diagnosis. Nothing then
inspects what it produced at the same granularity. Two failure modes reach the final scan unfixed:

1. **Missed residual** — a paragraph the rewriter left generic/ungrounded (e.g. "Technology has
   created both opportunities and risks…" stays generic).
2. **Newly introduced** — the rewrite swapped one AI-tell for another (a fresh generic assertion, a
   new cliché).

The whole-doc reviewer runs at the wrong altitude to catch either — it only sees document-wide
*style* patterns (`detect_residual_patterns`), not per-paragraph grounding/AI findings.

## 2. Objective

Insert a paragraph-level check on the rewriter's own output, **before** the whole-doc reviewer:

```
rewrite¹ (¶) → re-scan → rewrite² (¶, on residuals) → reviewer (doc) → final scan
```

It works exactly **like the rewriter** — same machinery (`_rewrite_document_once`), pointed at the
rewritten draft's own re-scan instead of the original. This is grounding-focused quality work, NOT
score-chasing and NOT detection-evasion (see Out of Scope).

## 3. THE LOAD-BEARING INVARIANT (do not violate)

> **Pass 2 scans and rewrites the REWRITTEN draft (`best_doc.rewritten_text`) — NEVER the original
> submitted `text` / `source_scan`.**

Why it is load-bearing: `_rewrite_document_once` keeps *unflagged* paragraphs by appending
`paragraph.text` from the scan it is given.

- Scan the **rewritten draft** (correct): unflagged paragraphs keep their **pass-1 improved** text;
  only residual/new-problem paragraphs are re-written. **Pass-1 gains are preserved.** ✅
- Scan the **original** (the failure mode): unflagged paragraphs revert to **original** text →
  pass-1 work is discarded → **zero gain**, or an infinite re-rewrite of the same source. ❌

Enforced three ways:
1. `_apply_residual_fix` receives only `best_doc` and reads `best_doc.rewritten_text`.
2. The rebuilt `DocumentResult` keeps `initial_scan = best_doc.initial_scan` (original) so the
   before/after diff stays original→final; `rewritten_text` becomes pass-2 output.
3. A dedicated regression test (see Testing) fails if pass 2 ever reverts a clean paragraph to its
   original text.

**Corollary (stale-diagnosis leakage):** pass 2 must drive purely off the *fresh* re-scan of the
rewritten draft. `findings_for_paragraph` is repopulated correctly by `scan_text(rewritten)`, but
`paragraph_diagnosis()` is keyed off the **original** report contract and is NOT refreshed by a bare
re-scan. Pass 2 must therefore **not** read original `paragraph_diagnosis` for the rewritten
paragraphs — that would leak original-content guidance into the draft (a subtler form of the same
bug). Resolution verified before build (see Risk R1).

## 4. Architecture

New function `_apply_residual_fix(best_doc, gateway, *, cancellation_check, authorship_evidence)` in
`direct_rewrite.py`, slotted in `run_direct_rewrite_all`:

```
best_doc = <best-of-N winner>
best_doc = _apply_residual_fix(best_doc, gateway, …)   # NEW — pass 2
return _apply_reviewer(best_doc, gateway, …)           # unchanged — doc-level QC + final scan
```

Mechanism (single pass):
1. `residual_scan = scan_text(best_doc.rewritten_text)` — fresh per-paragraph findings on the draft.
2. If no paragraph is flagged → return `best_doc` unchanged (common for clean rewrites), but append a
   `pass_trace` entry recording the check ran with zero fixes.
3. If flagged → run **one** writer pass over the flagged paragraphs of the rewritten draft (reuse the
   `_rewrite_document_once` machinery / `_clean_candidate`), driven by the **fresh** re-scan signal.
4. Rebuild `DocumentResult`: `initial_scan = best_doc.initial_scan`, `rewritten_text` = pass-2 text,
   `passes = best_doc.passes`, `pass_trace = best_doc.pass_trace + residual_entries`,
   `final_text_before_quality_repair = best_doc.final_text_before_quality_repair`. `final_scan` is
   irrelevant here — `_apply_reviewer` computes the authoritative final scan next.

## 5. Safety

- **Single pass** — no loop, no best-of-N on pass 2 (matches the chosen behaviour). Bounds latency.
- **Fidelity by reuse** — `_clean_candidate`'s existing grammar/meaning-inversion guards mean an
  unusable second write falls back to the **pass-1** paragraph (never broken text, never the
  original). No score-based drop-one-keep-rest guard needed.
- **Kill switch** — `DRAFTPROOF_V6_RESIDUAL_FIX` (default ON; `0/false/no/off` disables), mirroring
  `DRAFTPROOF_V6_REVIEWER` and `DRAFTPROOF_V6_DIRECT_REWRITE`. On disable or any exception, return
  `best_doc` unchanged.

## 6. Observability

Pass 2 appends `pass_trace` entries (capped list of re-fixed paragraphs, or a single
"checked — none flagged" entry). This also closes the silent-failure blind spot noted previously
(a no-op QC step that left no trace).

## 7. Cost / latency

Per rewrite job (Celery worker), added cost on the best-of-N **winner only**:
- 1× `scan_text` (lightweight, heuristic — NOT the GPT-2 detector).
- ≤ 1 writer call per still-flagged paragraph (typically a handful; clean rewrites add zero calls).

The authoritative final GPT-2 scan stays where it is, inside `_apply_reviewer` — no extra heavy scan.

## 8. Risks

- **R1 (must verify first): diagnosis source on re-scan.** Confirm what `paragraph_diagnosis()`
  returns for the rewritten draft's paragraph ids. If it returns nothing, pass 2 runs findings-only
  (acceptable, still targeted by fresh findings). If it could return the *original* paragraph's
  diagnosis for a colliding id, that is stale-leakage and must be prevented (drive pass 2 off the
  fresh `residual_scan` findings only). Resolve before writing pass-2 code.
- **R2: paragraph-id stability.** `scan_text` assigns ids by structure; the rewritten draft has its
  own ids. Pass 2 operates entirely within `residual_scan`'s id space — never cross-references
  original ids — so this is contained, but the test suite must assert it.
- **R3: double-work / churn.** A paragraph re-written in pass 1 then again in pass 2 is expected and
  fine, but must not oscillate — single pass guarantees termination.

## 9. Testing (TDD)

1. **Invariant — pass-1 gains preserved:** pass 1 changes paragraph A and leaves B clean; after
   pass 2, B equals **pass-1's** text (and ≠ original), proving pass 2 targets the rewritten draft.
2. **Re-fixes a residual:** a paragraph still flagged by the re-scan gets a second write.
3. **No-op + trace:** clean rewritten draft → text unchanged, but a "checked — none flagged"
   `pass_trace` entry is present.
4. **Kill switch:** `DRAFTPROOF_V6_RESIDUAL_FIX=0` returns `best_doc` unchanged, no extra scan.
5. **Order:** residual fix runs before `_apply_reviewer` (the final scan reflects pass-2 text).
6. **`initial_scan` preserved:** rebuilt result keeps the original `initial_scan` (before/after diff
   intact).
7. **No stale-diagnosis leakage (R1):** pass 2's per-paragraph input derives from the fresh re-scan,
   not the original report contract.

## 10. Out of scope

- **Scoring changes** — explicitly set aside by the user for this work.
- **Detection-evasion / lowering the external (perplexity) estimate** — the un-movable LLM-prose
  floor; firmly rejected by `docs/draftproof_alignment_principles`. This work improves *grounding*,
  which moves DraftProof's conservative axis and content quality, not the external estimate.
- **Looping / multi-round refinement** — single pass only.
- **A dedicated grounding-fixer prompt** — rejected in favour of reusing the rewriter ("work like
  rewrite").

## 11. Config summary

| Var | Default | Effect |
|---|---|---|
| `DRAFTPROOF_V6_RESIDUAL_FIX` | `1` (on) | `0/false/no/off` disables pass 2 (flow reverts to rewrite → reviewer → scan) |
