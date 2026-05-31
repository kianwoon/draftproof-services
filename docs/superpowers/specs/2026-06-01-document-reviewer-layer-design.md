# Document Reviewer Layer — Design Spec

**Date:** 2026-06-01
**Status:** Approved for planning
**Owner:** rewrite_v6

---

## 1. Problem

The lean direct-rewrite path (`poc/rewrite_v6/direct_rewrite.py`, default-on) rewrites the
document **one paragraph per LLM call**, each call blind to the others. Its system prompt instructs:
*"Ground EVERY generic claim in the author's FIRST-PERSON lived experience... 'In my classroom, I
have watched...'"*. Every paragraph independently follows the same instruction with the same
example, producing a **monoculture**: in a real 8-paragraph output, **7/8 paragraphs opened "In
my…" and 8/8 led with "I + perception-verb"**.

This is the writer trading one AI signal (generic assertion) for another (predictable, repetitive
structure). A per-paragraph writer **cannot** fix it — no single call can see the other openers.
The rewrite is a *shown teaching solution* (the user studies the before/after diff and edits with
their own content), so a monoculture output is a poor model to learn from.

### Why not fix the writer
The writer is working well (mean final_risk ~44 vs ~52 for the legacy pipeline). Overloading its
already-large prompt with cross-paragraph variety rules risks regressing a tuned component and
re-introduces the "guidance buries the fix" failure the lean path was built to escape. We keep the
writer untouched and add a **separate, document-aware reviewer** that sees the whole doc.

---

## 2. Objective

Add a reviewer pass that reads the **full rewritten document**, deterministically detects residual
AI-shaped patterns guided by the 25-trick taxonomy, and **surgically corrects only the offending
sentences** — so the final shown content is a clean, varied teaching artifact.

Aligned with the rewrite objective (`CLAUDE.md`, `project_v6_rewrite_objective`): the reviewer
**annotates/repairs**, never re-genericizes the writer's grounded specifics, and falls back to the
writer's text whenever a correction would regress.

This is **NOT** a humanizer / detection-evasion layer (`docs/draftproof_alignment_principles.md`).
It improves *structural variety and readability* of the shown solution; it does not chase the
intrinsic perplexity/top-k floor.

---

## 3. The 25 guidelines — the reviewer's craft rubric

**The 25 guidelines are the reviewer's guiding rubric and are placed directly in the reviewer
prompt.** They are general, content-agnostic writing-craft principles (vary openings, avoid robotic
transitions, attach risk to a failure mode, etc.) — they are **good guidance, not a hardcode
violation** (confirmed by the user, 2026-06-01). The NO HARDCODE gate forbids baking
domain-specific lists/categories/answers that overfit to particular content; craft guidance that
applies to any prose is legitimate.

The rubric lives as a **single module constant** (`WRITING_CRAFT_GUIDELINES`, a compact
problem→fix list mirroring the user's 25) so it has one source of truth and is easy to maintain.

The 25 still split by **who can verify them**, which decides how the reviewer is allowed to act:

| Bucket | Guidelines | Reviewer treatment |
|---|---|---|
| **A. Machine-detectable, cross/within-doc** | #2 predictable start, #5 packed list, #6 sentence overload, #7 balance phrase, #8 robotic transition, #12 formulaic contrast, #13 rhythm sameness, #19 same subject starts, #21 dense modifiers, #24 even paragraph shape | **Deterministic detector supplies hard evidence** (e.g. "paragraphs 1,2,3,7 open 'In my'") → reviewer edits the flagged sentences, guided by the rubric |
| **B. Judgment-only** | #4 author anchor, #11 weak judgment, #17 weak evidence, #20 ownership, #15/#16 specific benefit/risk | **Review flag only** — never silent-edited (a machine can't reliably score "is this judgment strong"); the rubric still tells the user what to strengthen |
| **C. Already owned upstream** | #1/#3/#9/#10/#18/#22/#23 grounding & concreteness | Owned by the **writer** (`direct_rewrite._SYSTEM`); reviewer protects, does not re-do |
| **Wrapper** | #25 preserve original meaning | **Fidelity guard** around every correction (see §6) |

**Why detectors AND rubric.** The rubric (all 25) tells the reviewer *what good prose looks like*.
The detectors solve what the rubric alone can't: a per-paragraph writer is **blind across
paragraphs**, so the reviewer needs hard, pre-computed evidence of cross-paragraph patterns (the
observed 7/8 "In my classroom") rather than eyeballing a long doc and possibly missing or
hallucinating them. Detectors = reliable sight; rubric = craft judgment.

**Phase 1** ships the bucket-A detectors (the cross-paragraph patterns that caused the observed bug)
plus the full 25-guideline rubric in the reviewer prompt. Bucket-B items ride as advisory flags.

---

## 4. Architecture

Three isolated units. Each has one purpose, a clear interface, and is independently testable.

### 4.1 `poc/rewrite_v6/residual_patterns.py` — detectors (pure, no LLM)

Pure functions over the full rewritten text. Return **evidence**, not fixes. No LLM, no network.

```python
@dataclass
class ResidualIssue:
    rule: str                 # e.g. "opener_monoculture"  (trick id in metadata)
    trick_ids: list[int]      # e.g. [19, 2]
    evidence: str             # human-readable: "paragraphs 1,2,3,7 open 'In my'"
    target_sentences: list[str]   # exact sentences to hand the reviewer for correction

def detect_residual_patterns(text: str) -> list[ResidualIssue]: ...
```

Detectors (Phase 1):
- `opener_monoculture` (#19,#2) — normalized first-2 and first-3 word frames repeated across
  paragraphs above a share threshold → targets the repeated opening sentences. **This is the
  primary detector** (catches the observed bug).
- `repeated_subject_starts` (#19) — same first word >2× within a paragraph.
- `rhythm_sameness` (#13) — per-paragraph sentence-length coefficient-of-variation below threshold
  → targets that paragraph's sentences.
- `robotic_transitions` (#8) — sentence-initial closed-class connectives
  (Furthermore, Moreover, Additionally, In conclusion, …) following the existing
  `_POLARITY_MARKERS` / `_CITATION_MARKERS` closed-set precedent.
- `balance_phrase` (#7) — "both … and …" / "opportunities and risks"-shape abstract pairings.

Thresholds live as named module constants with a one-line rationale each (not magic numbers inline).

### 4.2 `poc/rewrite_v6/document_reviewer.py` — orchestrator + one LLM call

```python
def review_document(text: str, *, gateway, cancellation_check=None) -> ReviewResult: ...
```

Flow:
1. `issues = detect_residual_patterns(text)`.
2. If `not issues` → return `ReviewResult(text=text, corrections=[], skipped="no_issues")`
   (**no LLM call** on clean docs).
3. Build the reviewer prompt: the **`WRITING_CRAFT_GUIDELINES` rubric** (the 25) + the **full
   document** (for sight) + only the **fired issues with their evidence and target sentences**.
   Output schema asks for `{ "corrections": [ {"original": "...exact sentence...", "revised": "..."} ] }`
   — corrected sentences ONLY. Bounded output → length-proof (≈200 tokens out regardless of doc
   length).
4. Parse JSON (reuse `json_io.parse_json`). Each correction is **spliced by verbatim match** of
   `original` in the doc. Unmatched originals are skipped (safe no-op).
5. Apply the §6 fidelity guard per correction. Return the reviewed text + a `corrections` trace.

### 4.3 Wire-in — `direct_rewrite.run_direct_rewrite_all`

After best-of-N selects the winning `DocumentResult` (`direct_rewrite.py:104 return best_doc`),
if `reviewer_enabled()`:
- `reviewed = review_document(best_doc.rewritten_text, gateway=gateway, ...)`
- rebuild the `DocumentResult` with `rewritten_text=reviewed.text`,
  `final_scan=scan_text(reviewed.text)`, and append reviewer corrections into `pass_trace`.

Downstream (report assembly, before/after diff) consumes `rewritten_text`/`final_scan` unchanged —
no report or frontend changes required.

---

## 5. Data flow

```
writer (per-paragraph, unchanged)
   → best_of_N winner: rewritten_text
      → detect_residual_patterns(text)         [pure, free]
         → issues? ── no ──► ship writer text unchanged
                │ yes
                ▼
         review_document: 1 LLM call, full-doc sight, returns corrected sentences only
                ▼
         splice by verbatim match  →  §6 fidelity guard per correction (drop regressions)
                ▼
         reviewed_text → final_scan → DocumentResult → report (diff = showcase)
```

---

## 6. Fidelity guard (#25 wrapper) — per correction, granular

For **each** `{original → revised}` correction, accept it only if ALL hold (else revert that one
sentence to the writer's `original`, keep the rest):
- Re-scored document AI risk does **not increase** (`direct_rewrite._document_ai_risk`).
- No broken grammar introduced (`direct_rewrite._has_broken_grammar` on the revised sentence).
- No polarity inversion vs the writer's sentence (`selector_diagnostics._severe_polarity_inversion`).
- The revision is non-empty and not a stub.

Reviewer is instructed: **vary the opening / transition / rhythm only; never alter or remove the
grounded specifics (names, figures, scenes, first-person facts) the writer added.**

Decision (approved): **drop the single bad correction**, never the whole pass — keeps maximum
improvement.

---

## 7. Visibility (approved)

Corrections are recorded in `pass_trace` (auditable/debuggable) but **not** surfaced as a separate
UI section. The before/after diff already IS the showcase. This deliberately avoids rebuilding the
annotation-UI surface that was reverted on 2026-06-01 (commit `acb764a9`).

---

## 8. Configuration

- `DRAFTPROOF_V6_REVIEWER` — kill switch, **default ON** (matches `direct_rewrite_enabled()`
  convention: set to `0/false/no/off` to disable, no redeploy).
- Detector thresholds: named constants in `residual_patterns.py`.

---

## 9. Failure modes & safety

| Failure | Handling |
|---|---|
| LLM returns empty / invalid JSON (the revert bug) | Output is tiny (corrected sentences only) so truncation is unlikely; on parse failure → ship writer text unchanged |
| Correction regresses score/grammar/polarity | Drop that one correction (§6) |
| Reviewer `original` doesn't match doc verbatim | Skip that correction (no splice) |
| No patterns fire | No LLM call; writer text shipped as-is |
| Reviewer alters grounded specifics | Prompt forbids it; polarity/score guards catch the worst; correction dropped on regression |
| Doc extremely long (>~6000 words) | Input still <10% context; out of scope to chunk now (YAGNI) — revisit only if real docs exceed it |

---

## 10. Testing

`poc/test_rewrite_v6_document_reviewer.py`:
- **Detectors (pure, deterministic, no LLM):**
  - the real 8-paragraph "In my classroom ×7" sample → `opener_monoculture` fires with correct indices
  - a varied human sample → **no** issues (false-positive guard)
  - robotic transitions / rhythm sameness / balance phrase positive + negative cases
- **Orchestrator (stub gateway):**
  - clean doc → no LLM call, text unchanged
  - fired issue + stub correction → spliced correctly by verbatim match
  - unmatched `original` → skipped
  - regressing correction (stub raises score) → dropped, others kept
- **Wire-in:** `run_direct_rewrite_all` with reviewer on/off (kill switch) produces expected text.

Measurement: `DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py N` (N≥4) before/after —
confirm reviewer does **not** regress final_risk (it should hold or modestly improve), per
`project_v6_measurement_variance`.

---

## 11. Out of scope (YAGNI)

- Chunking/windowing for very long docs (context budget says unnecessary).
- UI teaching-note annotations (reverted today; the diff is the showcase).
- Bucket-B auto-editing (judgment rules stay advisory flags).
- Touching the writer prompt or the legacy planner/selector pipeline.

---

## 12. File summary

| File | New/Changed | Purpose | Est. lines |
|---|---|---|---|
| `poc/rewrite_v6/residual_patterns.py` | new | deterministic detectors | ~180 |
| `poc/rewrite_v6/document_reviewer.py` | new | `WRITING_CRAFT_GUIDELINES` rubric (the 25) + orchestrator + 1 LLM call + fidelity guard | ~200 |
| `poc/rewrite_v6/direct_rewrite.py` | changed | wire reviewer after best-of-N; kill switch | ~25 |
| `poc/test_rewrite_v6_document_reviewer.py` | new | detector + orchestrator + wire-in tests | ~220 |

All files stay well under the 1500-line limit.
