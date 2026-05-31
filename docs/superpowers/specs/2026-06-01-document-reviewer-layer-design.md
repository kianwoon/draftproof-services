# Document Reviewer (QC) Layer — Design Spec

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
writer untouched and add a **separate, document-aware QC reviewer** that sees the whole doc.

---

## 2. Objective

Add a **QC (quality-control) reviewer pass** that reads the **full rewritten document**, inspects it
against the 25 writing-craft guidelines, and **surgically corrects any substandard sentence** — so
the final shown content is a clean, varied, higher-quality teaching artifact.

Like a real QC inspector, the reviewer is not limited to a pre-printed defect list: it may correct
**any** sentence it judges substandard against the 25. Deterministic detectors run alongside as a
**safety net** — they *guarantee* the cross-paragraph patterns a per-paragraph writer is blind to
(the observed "In my classroom" monoculture) are always caught and handed over as **must-fix**,
never missed. Detectors = guaranteed sight on cross-doc patterns; rubric = QC craft judgment over
everything else.

This is **NOT** a humanizer / detection-evasion layer (`docs/draftproof_alignment_principles.md`).
It improves *structural variety and readability* of the shown solution; it does not chase the
intrinsic perplexity/top-k floor.

---

## 3. Pipeline order — rewrite → QC → scan (CRITICAL)

The authoritative end-to-end order (user-confirmed 2026-06-01):

```
rewrite (writer)  →  QC (reviewer)  →  final scan (once, on QC'd text)
```

- We do **NOT** scan between rewrite and QC. The single authoritative **`final_scan` runs once,
  AFTER QC**, on the QC'd text. The reported `final_risk` reflects the QC'd content.
- **Best-of-N stays inside the writer.** Best-of-N's intermediate per-attempt scoring
  (`direct_rewrite._document_ai_risk`) is the writer's *selection* mechanism only — it picks the
  best writer draft. That intermediate score is never the reported number.
- After the writer hands off its single best draft → **QC the winner → one final scan**.
- **Guard baseline:** to keep QC from ever worsening the score (there is no post-rewrite scan to
  compare against), the guard takes **one cheap internal score of the writer's pre-QC winner** as a
  *baseline only* (not reported). A QC correction is kept only if the final post-QC scan is **≤
  baseline** and grammar/polarity hold (see §6). This honors "scan after QC" for the reported number
  while preventing regressions.

---

## 4. The 25 guidelines — the QC reviewer's craft rubric

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
| **A. Machine-detectable, cross/within-doc** | #2 predictable start, #5 packed list, #6 sentence overload, #7 balance phrase, #8 robotic transition, #12 formulaic contrast, #13 rhythm sameness, #19 same subject starts, #21 dense modifiers, #24 even paragraph shape | **Deterministic detector supplies hard evidence** (e.g. "paragraphs 1,2,3,7 open 'In my'") → marked must-fix; reviewer edits the flagged sentences |
| **B. Judgment-only** | #4 author anchor, #11 weak judgment, #17 weak evidence, #20 ownership, #15/#16 specific benefit/risk | **QC may edit if clearly substandard; otherwise review flag** — a machine can't reliably score these, so they are not detector-enforced, but the rubric guides QC |
| **C. Already owned upstream** | #1/#3/#9/#10/#18/#22/#23 grounding & concreteness | Owned by the **writer** (`direct_rewrite._SYSTEM`); reviewer protects, does not re-do |
| **Wrapper** | #25 preserve original meaning | **Fidelity guard** around every correction (see §6) |

**Why detectors AND rubric.** The rubric (all 25) tells QC *what good prose looks like*. The
detectors solve what the rubric alone can't: a per-paragraph writer is **blind across paragraphs**,
so QC needs hard, pre-computed evidence of cross-paragraph patterns (the observed 7/8 "In my
classroom") rather than eyeballing a long doc and possibly missing or hallucinating them. Detectors
= guaranteed sight; rubric = QC craft judgment over everything else.

---

## 5. Architecture

Three isolated units. Each has one purpose, a clear interface, and is independently testable.

### 5.1 `poc/rewrite_v6/residual_patterns.py` — detectors (pure, no LLM)

Pure functions over the full rewritten text. Return **evidence**, not fixes. No LLM, no network.

```python
@dataclass
class ResidualIssue:
    rule: str                 # e.g. "opener_monoculture"
    trick_ids: list[int]      # e.g. [19, 2]
    evidence: str             # human-readable: "paragraphs 1,2,3,7 open 'In my'"
    target_sentences: list[str]   # exact sentences handed to QC as must-fix

def detect_residual_patterns(text: str) -> list[ResidualIssue]: ...
```

Detectors (Phase 1):
- `opener_monoculture` (#19,#2) — normalized first-2 and first-3 word frames repeated across
  paragraphs above a share threshold → targets the repeated opening sentences. **Primary detector**
  (catches the observed bug).
- `repeated_subject_starts` (#19) — same first word >2× within a paragraph.
- `rhythm_sameness` (#13) — per-paragraph sentence-length coefficient-of-variation below threshold.
- `robotic_transitions` (#8) — sentence-initial closed-class connectives (Furthermore, Moreover,
  Additionally, In conclusion, …), following the existing `_POLARITY_MARKERS` / `_CITATION_MARKERS`
  closed-set precedent.
- `balance_phrase` (#7) — "both … and …" / "opportunities and risks"-shape abstract pairings.

Thresholds are named module constants, each with a one-line rationale (not inline magic numbers).

### 5.2 `poc/rewrite_v6/document_reviewer.py` — QC orchestrator + one LLM call

```python
def review_document(text: str, *, gateway, cancellation_check=None) -> ReviewResult: ...
```

Flow:
1. `must_fix = detect_residual_patterns(text)` — the deterministic safety net.
2. `baseline = _document_ai_risk(text)` — one cheap internal score of the writer's text, for the
   §6 guard only (not the reported number).
3. Build the QC prompt: the **`WRITING_CRAFT_GUIDELINES` rubric** (the 25) + the **full document**
   (for sight) + the **`must_fix` issues with evidence** marked required. Instruction: act as QC —
   correct every `must_fix` AND any other sentence that falls short of the 25, but **change only
   what is substandard** (vary opening / transition / rhythm / wording); never alter the grounded
   specifics the writer added. Output schema:
   `{ "corrections": [ {"original": "...exact sentence...", "revised": "..."} ] }` — corrected
   sentences ONLY. Bounded output → length-proof (≈200–600 tokens regardless of doc length).
4. **QC always runs when enabled** (one LLM call per document); we do not short-circuit on empty
   `must_fix` because QC may find defects the detectors don't model. Only the kill switch skips it.
5. Parse JSON (reuse `json_io.parse_json`). Each correction is **spliced by verbatim match** of
   `original` in the doc; unmatched originals are skipped (safe no-op).
6. Apply the §6 fidelity guard per correction (using `baseline`). Record every `must_fix` not
   addressed in the trace. Return reviewed text + a `corrections` trace.

### 5.3 Wire-in — `direct_rewrite.run_direct_rewrite_all`

After best-of-N selects the winning `DocumentResult` (`direct_rewrite.py:359 return best_doc`), and
**before** the final scan is reported, if `reviewer_enabled()`:
- `reviewed = review_document(best_doc.rewritten_text, gateway=gateway, ...)`
- rebuild the `DocumentResult` with `rewritten_text=reviewed.text`, run the **single final**
  `final_scan=scan_text(reviewed.text)`, and append QC corrections into `pass_trace`.

Note: `_rewrite_document_once` currently computes `final_scan` per attempt. To honor rewrite→QC→scan,
the **authoritative** final scan is the post-QC one in the wire-in; per-attempt scans remain the
writer's internal best-of-N selection inputs only.

Downstream (report assembly, before/after diff) consumes `rewritten_text`/`final_scan` unchanged —
no report or frontend changes required.

---

## 6. Fidelity guard (#25 wrapper) — per correction, granular

For **each** `{original → revised}` correction, accept it only if ALL hold (else revert that one
sentence to the writer's `original`, keep the rest):
- Post-QC document AI risk **≤ the pre-QC `baseline`** (§3) — QC can never make the score worse.
- No broken grammar introduced (`direct_rewrite._has_broken_grammar` on the revised sentence).
- No polarity inversion vs the writer's sentence (`selector_diagnostics._severe_polarity_inversion`).
- The revision is non-empty and not a stub.

Reviewer is instructed: **vary the opening / transition / rhythm / wording only; never alter or
remove the grounded specifics (names, figures, scenes, first-person facts) the writer added.**

Decision (approved): **drop the single bad correction**, never the whole pass — keeps maximum
improvement. The score check is evaluated against `baseline` so the guard works even though the only
*reported* scan happens after QC.

---

## 7. Visibility (approved)

Corrections are recorded in `pass_trace` (auditable/debuggable) but **not** surfaced as a separate
UI section. The before/after diff already IS the showcase. This deliberately avoids rebuilding the
annotation-UI surface that was reverted on 2026-06-01 (commit `acb764a9`).

---

## 8. Configuration

- `DRAFTPROOF_V6_REVIEWER` — kill switch, **default ON** (matches `direct_rewrite_enabled()`
  convention: `0/false/no/off` disables, no redeploy).
- Detector thresholds: named constants in `residual_patterns.py`.

---

## 9. Failure modes & safety

| Failure | Handling |
|---|---|
| QC returns empty / invalid JSON (the prior revert bug) | Output is tiny (corrected sentences only) so truncation is unlikely; on parse failure → ship writer text unchanged, run final scan on it |
| Correction regresses score (> baseline) / grammar / polarity | Drop that one correction (§6) |
| QC `original` doesn't match doc verbatim | Skip that correction (no splice) |
| QC finds nothing to fix | Returns zero corrections; writer text shipped unchanged (one cheap LLM call spent), then final scan |
| QC alters grounded specifics | Prompt forbids it; polarity/score guards catch the worst; correction dropped on regression |
| Doc extremely long (>~6000 words) | Input still <10% context; chunking out of scope now (YAGNI) — revisit only if real docs exceed it |

---

## 10. Testing

`poc/test_rewrite_v6_document_reviewer.py`:
- **Detectors (pure, deterministic, no LLM):**
  - the real 8-paragraph "In my classroom ×7" sample → `opener_monoculture` fires with correct indices
  - a varied human sample → **no** issues (false-positive guard)
  - robotic transitions / rhythm sameness / balance phrase positive + negative cases
- **QC orchestrator (stub gateway):**
  - clean doc → QC runs, returns no corrections, text unchanged
  - fired issue + stub correction → spliced correctly by verbatim match
  - unmatched `original` → skipped
  - regressing correction (stub raises score above baseline) → dropped, others kept
  - order assertion: final scan is computed on the QC'd text, not the writer's text
- **Wire-in:** `run_direct_rewrite_all` with reviewer on/off (kill switch) produces expected text;
  final_scan reflects QC'd content.

Measurement: `DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py N` (N≥4) before/after —
confirm QC does **not** regress final_risk (it should hold or modestly improve), per
`project_v6_measurement_variance`.

---

## 11. Out of scope (YAGNI)

- Chunking/windowing for very long docs (context budget says unnecessary).
- UI teaching-note annotations (reverted today; the diff is the showcase).
- Running QC on every best-of-N attempt (contradicts "QC then scan once").
- Touching the writer prompt or the legacy planner/selector pipeline.

---

## 12. File summary

| File | New/Changed | Purpose | Est. lines |
|---|---|---|---|
| `poc/rewrite_v6/residual_patterns.py` | new | deterministic detectors (safety net) | ~180 |
| `poc/rewrite_v6/document_reviewer.py` | new | `WRITING_CRAFT_GUIDELINES` rubric (the 25) + QC orchestrator + 1 LLM call + fidelity guard | ~200 |
| `poc/rewrite_v6/direct_rewrite.py` | changed | wire QC after best-of-N, before the single final scan; kill switch | ~30 |
| `poc/test_rewrite_v6_document_reviewer.py` | new | detector + QC orchestrator + wire-in + order tests | ~240 |

All files stay well under the 1500-line limit.
