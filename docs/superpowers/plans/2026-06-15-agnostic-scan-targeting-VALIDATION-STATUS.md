# Agnostic Scan Targeting — Validation Status

**Branch:** `kianwoon/keen-blackburn-71fad5`
**Date:** 2026-06-15
**Companion docs:** `2026-06-15-agnostic-scan-targeting.md` (plan), `../specs/2026-06-15-agnostic-scan-targeting-design.md` (design)

This records the validation of the agnostic-scan refactor that could **not** be
completed on the dev machine because its ML stack is broken (scipy `_propack`
dlopen error — `__DATA/__thread_bss` zero-fill section, macOS 27 arm64 / pyenv
3.11.0 / scipy 1.15.2). Anything importing `transformers → sklearn → scipy`
fails at import. See memory `project_local_env_scipy_broken.md`.

## TL;DR

| Item | Status | Where it can finish |
|------|--------|---------------------|
| 1. Deterministic measurement gate (no regression) | **CI-gated** | needs working ML stack |
| 2. Reconcile legacy-planner tests in `test_rewrite_v6_report_contracts.py` | **CI-gated** (cannot enumerate locally — see below) | needs working ML stack |
| 3. Delete dead tag-keyed branches in `plan.py` | **Analysis COMPLETE** (below); edits deferred — coupled to #2 | apply + verify together in CI |
| 6. `test_rewrite_v6_production_adapter.py` / `_residual_checker.py` | **CI-gated** | needs working ML stack (imports `rewrite_v2 → rewriter → transformers`) |

**What WAS validated locally (via the `poc/conftest.py` lightweight stub):**
- `test_rewrite_v6_agnostic_scan.py` — **10 passed, 3 skipped** (skips are
  fixture-only: "saved production report not present", not env). This is the
  branch author's +121-line characterization test for *exactly* this refactor →
  **positive behavioral validation of the agnostic targeting at the scan level.**
- `test_rewrite_v6_scanner_alignment.py` — passes (within the same run).

## Regression finding — read precisely (do not over-claim)

`test_rewrite_v6_report_contracts.py` shows **31 failed / 40 passed** on this
branch. Verified this is **not a regression**:

- Reverting `scan.py` to `main` (keeping everything else) gives the **identical
  31 failed / 40 passed**.
- Running the same file on `main` (with the conftest copied in) gives the **same
  31 failing test names** (47 pass — main still has the 7 not-yet-deleted tests).
- `comm` of the failing-name sets: **31 common, 0 keen-only**. Zero new failures
  introduced by the agnostic change.

**Why they fail:** these tests route through `run_v6_rewrite` / the writer
pipeline / sklearn-importing paths and hit the broken scipy stack — either
directly (`ImportError`) or indirectly (the pipeline catches the ML import
failure and falls back to `source_preserved`, so the assertion mismatches, e.g.
`assert result.selected is None` ← got `Variant(id='source_preserved', …)`).

> **No NEW report_contracts failures — all 31 are identical on `main` and the
> branch and fire at the scipy import before `scan.py` matters, so there is no
> *detectable* regression, but these tests validate nothing either way.**
> Behavioral validation = the characterization tests (local, passing) + the
> measurement gate (CI, pending).

Blast radius of the refactor (`git diff --stat main..HEAD`): only
`poc/rewrite_v6/scan.py` (−148 lines), `poc/conftest.py` (new), the test files,
and docs. **`plan.py`, `prose_quality.py`, `integrity_guard.py`,
`selector_diagnostics.py`, `write.py`, `pipeline.py` are all unchanged from
`main`.**

## Task 2 — why it cannot be done locally

The task expected to "RUN the full file, find ALL that fail because `build_plan`
no longer receives the removed tags." This is impossible on the broken stack:
the two tests the task names —
`test_v6_unsupported_claim_gap_requires_author_proxy_grounding` and
`test_v6_context_anchor_gap_uses_same_paragraph_author_proxy_grounding` — **fail
identically on `main`, where the removed tags still exist.** They die at the
scipy import *before* reaching the removed-tag assertion. So the import masks the
real signal; "ALL that fail due to removed tags" cannot be enumerated until the
stack imports cleanly. This is genuine CI-gating.

## Task 3 — provenance analysis (COMPLETE; edits deferred)

**Question:** can the four removed tags still reach `build_plan`'s `tag_set`
after the agnostic refactor? **Answer: no** — verified statically through every
tag-entry path. (`detect/` is unchanged from `main`, so this holds on both.)

Tag entry paths into `build_plan` findings:
1. **`scan.py` `_tags`/`_risk`** (the removed content-word detectors) — these
   were the *only* historical producers of the bare tags. Now removed:
   `grep -nE 'author_anchor_gap|unsupported_claim_gap|broad_claim|semantic_bridge_gap' poc/rewrite_v6/scan.py`
   → **none**.
2. **`_report_signal_tags(signal)`** (`scan.py:320`) — normalizes a detector
   report signal's `title`/`scanner`/`category`/`key` (lowercase, non-alnum→`_`).
   The per-sentence signals come from `report.py::_segment_signal` →
   `Finding.title/.scanner/.category` + `_signal_descriptor` key. The descriptor
   key is a fixed set: `grounding_risk`, `human_anchor_score`, `source_similarity`,
   `section_style_variance`, `ai_likelihood`, `rewrite_smoothness`,
   `f.signal_category`/`category`. The detector's grounding vocab uses **`_risk`
   suffixes** (`broad_claim_risk` → tag `broad_claim_risk`, **≠** `broad_claim`;
   `unsupported_claim_risk` → `unsupported_claim_risk`, **≠** `unsupported_claim_gap`).
3. **`plan.py::_planning_tags`** — derives only `predictable_start` /
   `context_anchor_gap` (pointer-opening / quoted sentences). **Not** the four.
4. No internal producer: every `plan.py`/`planner_llm.py`/`finding_pattern.py`/
   `paragraph_architecture.py`/`writer_feedback.py` reference is a **read**
   (`in tag_set`, `& {…}`), never an add.

Per-tag verdict:

| Tag | Verdict | Evidence |
|-----|---------|----------|
| `semantic_bridge_gap` | **DEAD (airtight)** | absent from all of `detect/` + `report/`; not derived in `_planning_tags` |
| `unsupported_claim_gap` | **DEAD (airtight)** | absent from `detect/` + `report/`; detector uses `unsupported_claim_risk` only |
| `broad_claim` (bare) | **DEAD** | only appears inside compound `generic_opener_broad_claim` (→ different tag) and a human description string; never a `Finding` title/scanner/category/key |
| `author_anchor_gap` | **DEAD** | only a **bucket-level** key in `detect/grounding_diagnosis.py:133`; the grounding diagnosis is doc-level and never attached to per-sentence `highlight_segments[].signals` (which are built solely from `Finding` objects in `report.py::_document_segments`) |

**Why the edits are deferred (not done here):**
- Most `plan.py` branches **mix** dead tags with **live** ones in the same set
  literal — e.g. `plan.py:498,528,553,623,635,870,1072,1118` include the live
  `context_anchor_gap`/`predictable_start`/`paragraph_rhythm`. The correct edit
  is to **remove the dead tags from the set literals**, and delete only the
  branches / dict entries keyed **solely** on dead tags (e.g. `plan.py:532,534,
  956,961,966,976,1145,1390,1394,1398,1420,1424,1428`). Wholesale branch
  deletion would silently drop live `context_anchor_gap` handling.
- This surgery is **coupled to Task 2** (the legacy-planner tests assert this
  behavior) and exercises `build_plan` integration that is ML-gated locally.
  Applying it without being able to run a single `build_plan` test would ship
  unverified edits. Apply #2 + #3 **together** in the working-env session.

Do **not** touch the default direct-path consumers' live branches:
`rewrite_playbook.py`, `author_proxy_routes.py`, `writer_lean_prompt.py`.

## Commands to finish in a working environment (CI)

```bash
cd poc

# Task 1 — measurement gate (branch mean final_risk must be <= main; N>=4)
DRAFTPROOF_V6_DETERMINISTIC=1 python _measure_baseline.py 4   # on this branch
# then compare against main

# Task 2 — now the assertions are reachable; reconcile the genuinely-dead ones
python -m pytest test_rewrite_v6_report_contracts.py -q
#   Keep tests that survive via plan.py::_planning_tags (predictable_start /
#   context_anchor_gap pointer/quote derivation). Delete/update only those that
#   assert behavior driven by the removed tags.

# Task 3 — apply the plan.py edits above, then verify nothing in the legacy
# planner path regressed.

# Task 6 — production / residual adapters
python -m pytest test_rewrite_v6_production_adapter.py test_rewrite_v6_residual_checker.py
```

If `final_risk` regresses in Task 1: fix the agnostic logic in `scan.py`
(confirm grounding signals reach `_report_findings`; check `paragraph_diagnosis`
coverage). Do **not** add a feature flag and do **not** re-introduce a
content-word detector.

### Optional: a throwaway venv for local results
A disposable venv with a scipy build compatible with macOS 27 arm64 could run all
of the above locally, but **do not mutate the global pyenv** (no project venv
exists; it would ripple numpy/torch/transformers across every project). CI is the
sanctioned path.
