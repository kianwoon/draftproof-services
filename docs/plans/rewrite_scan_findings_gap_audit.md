# Rewrite ⇄ Scan-Findings Gap Audit & Plan

**Date:** 2026-07-16
**Author:** review pass (branch `kianwoon/rewrite-scan-findings-...`)
**Objective:** the rewrite's job is to *mitigate the AI score AND the specific findings the scan
raised*. The scan was recently enhanced with three new per-claim / per-sentence signals. This audit
establishes exactly what rewrite consumes today, proves which new signals it ignores, and lays out a
prioritized, measured plan to wire them in — without regressing the lean direct path that already won
the A/B.

---

## 1. What rewrite consumes today (verified)

Default path = `poc/rewrite_v6/direct_rewrite.py::run_direct_rewrite_all` (one writer call per flagged
paragraph). The writer prompt (`_prompt`, [direct_rewrite.py:289](../../poc/rewrite_v6/direct_rewrite.py))
is built from a **per-paragraph** diagnosis assembled by
`extract_paragraph_diagnoses` ([report_contracts.py:18](../../poc/rewrite_v6/report_contracts.py)):

| Field in writer prompt | Scan source | Granularity |
|---|---|---|
| `scanner_diagnosis` (main_issue / why_flagged / recommendation / rewrite_hint) | `paragraph_explanations.paragraphs[]` | paragraph |
| `flagged_issue_types` (= finding `.tags` → categories `generic_assertion_risk`, `citation_grounding_risk`, …) | `findings_for_paragraph()` | paragraph |
| `predictable_phrases` (exact flagged token spans) | `highlight_segments[].predictability.predictable_token_spans` | paragraph (grouped) |
| `critical_thinking_focus` / `critical_thinking_questions` | `ai_risk_badge.critical_thinking_control` (anchor-quote matched) | paragraph |

**Verdict:** the *old* signals (predictability + prose diagnosis + critical-thinking) are wired in
well. This part is healthy.

## 2. The gap — the enhanced scan's new intelligence is invisible to rewrite

All three recent scan additions are **display-only**: they never mutate tier/score AND never reach the
writer.

| New scan signal | Report location | Keyed by | Reaches writer? |
|---|---|---|---|
| **`sentence_issue_tags`** — per-sentence `ai` / `grounding` / `reasoning` tags + `fix_text` ([sentence_issue_tags.py:87-197](../../poc/report/sentence_issue_tags.py)) | top-level `result["sentence_issue_tags"]` | `sentence_id` only (no paragraph_id / text) | ❌ No |
| **`claim_graph_display.source_checks`** — per-claim entailment verdict `verified/contradicted/paywalled/unresolved` + `entailment_score` ([claim_graph_panel.py:103-108](../../poc/report/claim_graph_panel.py)) | `authorship_evidence.claim_graph_display` (gated `DRAFTPROOF_CLAIM_GRAPH`, ON in prod) | `claim_excerpt` (trunc 140); real `paragraph_id` exists upstream but composer drops it | ❌ No |
| `authorship_evidence_levels` | `authorship_evidence` | — | ❌ No (out of scope here) |

Two structural weaknesses compound this:

- **Rewrite anchors per-paragraph, not per-sentence.** Even the categories it gets are paragraph-wide.
  It cannot target "*sentence 3 is a grounding gap, sentence 5 a reasoning gap*" — exactly what
  `sentence_issue_tags` now knows.
- **Guards never verify the original finding was resolved.** The residual checker re-*scans* the
  rewritten draft with grounding heuristics but passes `diagnosis=None`
  ([direct_rewrite.py:191](../../poc/rewrite_v6/direct_rewrite.py)) — it never checks whether the
  *specific* flagged category actually dropped.

## 3. Guardrails these changes MUST respect (from CLAUDE.md / project memory)

1. **Annotate, don't suppress.** New signals become *sharper writer guidance* + review flags, never new
   rejection reasons. `source_preserved` stays the worst outcome.
2. **No fabrication.** Feeding "this claim was contradicted by its source" must instruct the writer to
   *qualify / attribute / soften* the claim (or leave an honest gap) — NOT invent a supporting
   source/stat. Mirror the existing `critical_thinking_questions` handling
   ([direct_rewrite.py:433-448](../../poc/rewrite_v6/direct_rewrite.py)).
3. **Additive & flag-safe.** `claim_graph` is env-gated and absent in a normal report — every new join
   must degrade to a no-op when the section is missing, byte-identical to today.
4. **Lean path stays lean.** Do NOT reintroduce the heavy planner's `rewrite_targets` / `flow_plans` /
   `must_keep` machinery. Add *only* the two new signals as compact prompt fields.
5. **Measure, don't eyeball.** gpt-oss writer is high-variance; single runs are noise.

## 4. Measurement gate (run before AND after every change)

```bash
cd poc && DRAFTPROOF_V6_DETERMINISTIC=1 python _measure_baseline.py 6   # N>=4; captures final_risk mean
```

Acceptance: mean `final_risk` must **not regress** vs. the pre-change baseline, and — on a fixture set
carrying grounding/citation findings — the count of *residual* grounding/reasoning findings in the
rewritten draft should drop. Record both numbers in this doc as we go.

Baseline (to fill in): `final_risk mean = ____` (n=6, pre-change). **Capture this blank baseline BEFORE
P1** — it is also the token-budget reference (see prompt-bloat guard below).

**⚠ P2 measurability gap (per Fable review):** the deterministic harness almost certainly produces **no
`claim_graph`** (it needs the Modal NLI endpoint + `DRAFTPROOF_CLAIM_GRAPH`). So the standard gate
**cannot** exercise P2. Before building P2, add a **fixture report JSON that carries a persisted
`claim_graph`** (contradicted/unresolved rows with `source.paragraph_id`) and drive `run_rewrite_pipeline_v6`
off it — otherwise P2's acceptance criterion is untestable and must not be claimed as verified.

**Prompt-bloat guard (per Fable review):** the writer payload is already ~10 instructions + CT + routes,
and best-of-N (`_best_of_n`, [direct_rewrite.py:1014](../../poc/rewrite_v6/direct_rewrite.py)) multiplies
every token. Hard caps: `flagged_sentences ≤ 3`, `unsupported_claims ≤ 2`. Measure the token delta per
paragraph against the blank baseline; if it starves the writer (`finish_reason=length` → empty →
`source_preserved`), trim before shipping.

## 5. Prioritized plan

### P1 — Sentence-issue-tag targeting (highest leverage, self-contained, no flag dependency)
- **Source (per Fable review): read `findings` directly, NOT the `sentence_issue_tags` display dict.**
  The `low_specificity` / `semantic_drift` finding rows already carry `sentence_id` + `recommendation`;
  consuming the display contract would couple rewrite to an i18n/fail-open render structure. Same
  grounding/reasoning allowlist, no display coupling.
- Build a `sentence_id → (paragraph_id, sentence_text)` map from `highlight_segments` (they carry all
  three). Normalise `sentence_id` via `str()` on both sides.
- Attach to each paragraph's diagnosis a compact `flagged_sentences: [{text, issue: grounding|reasoning,
  fix}]` (**cap 3/paragraph**; drop `ai` type — predictability already covers it). **Silently drop
  doc-level `semantic_drift` tags with no sid — never paragraph-guess an unanchored tag.**
- In `_prompt`, render `flagged_sentences` as "these specific sentences were flagged for X — fix each":
  grounding→ground it; reasoning→make the reasoning explicit. No new reject path.
- **Files:** `report_contracts.py`, `direct_rewrite.py::_prompt`. **Risk:** low. **Flag:** none needed
  (findings present in normal prod).

### P2 — Claim-graph entailment targeting (directly attacks `citation_grounding_risk`)
- **DO NOT touch `claim_graph_panel.py` (per Fable review).** `claim_graph_display` is a pinned,
  byte-for-byte render contract for two surfaces — severity-sorted, capped at 8, `claim_excerpt`
  truncated to 140. Joining rewrite off it would inherit the cap + truncation and risk the display
  contract.
- Instead, join **upstream** in `extract_paragraph_diagnoses` from the raw
  `authorship_evidence.claim_graph.claims[]`: each carries `source.paragraph_id` and the evidence
  `entailment` (schema ~`schema.py:100`). Use the **untruncated** claim text so the writer edits the
  right sentence.
- Attach `unsupported_claims: [{claim, verdict, why}]` for `contradicted/paywalled/unresolved` only
  (skip `verified`). **Cap 2/paragraph** (prompt-bloat guard).
- In `_prompt`, mirror the `critical_thinking_questions` handling **verbatim**
  ([direct_rewrite.py:433-448](../../poc/rewrite_v6/direct_rewrite.py)): allowed actions ONLY —
  qualify / attribute-as-disputed / soften / leave-an-honest-gap; explicit "do NOT invent a source and
  do NOT change the number." `_has_fabricated_specifics` already backstops any new number/citation.
  **Always emit one `author_review_item` per touched claim.**
- **Files:** `report_contracts.py`, `direct_rewrite.py::_prompt` (NOT the panel composer). **Risk:**
  low-medium (gated signal). **Flag-safe:** raw `claim_graph` absent → field empty, byte-identical.

### P3 — Finding-aware verification (closes the "did we actually mitigate it" loop)
- After the final scan, compare per-paragraph *original* finding categories vs. the fresh re-scan's
  categories; record `resolved` / `persisted` per category into `pass_trace` (telemetry first).
- When a flagged category *persists* on a paragraph the writer changed, add a review flag ("this
  grounding issue may remain — add your own specifics"). Do NOT force another rewrite loop (avoids the
  stale-diagnosis leak the residual checker deliberately guards against).
- **Keep `diagnosis=None` in the residual re-fix** ([direct_rewrite.py:191](../../poc/rewrite_v6/direct_rewrite.py)) —
  the P1/P2 signals won't reach residual fixes, and that's accepted (the stale-diagnosis leak guard the
  residual deliberately holds is more important). Verification reads the fresh re-scan vs. original
  categories; it does not re-inject the original diagnosis.
- **Files:** `direct_rewrite.py` (residual/verification stage, telemetry + review flag). **Risk:**
  medium (touches guard flow); telemetry-only first, behavior change second, each measured. **Never
  re-loop** on a persisted finding.

## 5b. Implementation status (2026-07-16)

- **P1 — DONE & unit-verified.** `_paragraph_flagged_sentences` + merge in `report_contracts.py`;
  `apply_scan_target_instructions` in `direct_prompts.py`; kill switch `DRAFTPROOF_V6_FLAGGED_SENTENCES`
  (default on, off == byte-identical pre-P1). Extracted the two system-prompt constants to a new
  `direct_prompts.py` (direct_rewrite.py was already >1500 LOC; now 1476).
- **P2 — DONE & unit-verified** (against a synthetic claim-graph). `_paragraph_unsupported_claims` reuses
  the panel composer's `_row_status`; joins upstream from `authorship_evidence.claim_graph.claims[].source.paragraph_id`;
  keeps contradicted/paywalled/unresolved (skips verified); fabrication-safe prompt. Kill switch
  `DRAFTPROOF_V6_UNSUPPORTED_CLAIMS` (default on). Inert without a claim-graph → byte-identical normal report.
- **P3 — DONE & unit-verified.** New module `finding_verification.py`: re-scans the final draft and
  reports, per document, how many flagged grounding/reasoning issues are now GONE
  (`resolved`/`persisted`/`resolution_rate`) — a **noise-robust measurer** since `final_risk` is too
  high-variance to prove a fix landed. Telemetry only (one `pass_trace` row under
  `candidate_generation_status`), never re-loops, never mutates text; persisted issues become review
  flags. Claim (P2) targets are counted but marked not-locally-verifiable (need Modal NLI). Kill switch
  `DRAFTPROOF_V6_FINDING_VERIFICATION`. Injected `_document_scan_report` to avoid a circular import.
- **Measurement note:** the harness fixture (`production_rewrite_ad62c7f1_20260529/...`) was **absent
  from the worktree**. Baked a fresh one (`_bake_fixture_report.py` → `test_output/_fixture_scan_report.json`)
  and made the path env-overridable (`DRAFTPROOF_BASELINE_REPORT`). The composite (local) scan emits only
  **1 grounding finding** (vs a paragraph-level `generic_assertion_risk` of 80) and **no claim-graph** — so
  local measurement exercises P1 lightly and P2 not at all; both are fully exercised only on the prod
  deep-scan path. Cerebras rate-limited the full best-of-N harness → A/B now toggles the P1 kill switch on
  a lean config (best-of-1, single lane, sequential) so it's the SAME code, isolating P1's prompt effect.

Baseline (P1 off) / Post (P1 on): `final_risk mean = ____ / ____` (lean n=4).

## 6. Sequencing
P1 → measure → P2 → measure → P3 → measure. Each ships behind its own small, reversible change; the
measurement gate is the accept/reject decision, not intuition. P1 and P2 are independent and could be
built in parallel; P3 depends on nothing but is riskiest, so it lands last.
