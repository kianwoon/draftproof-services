# Agnostic Scan Targeting — Design

**Date:** 2026-06-15
**Area:** `poc/rewrite_v6/scan.py` (V6 production rewrite path)
**Status:** Design — pending user review

---

## Problem

`poc/rewrite_v6/scan.py` is on the **default V6 production path**
(`DRAFTPROOF_REWRITE_V6_ENABLED=True` → `run_rewrite_pipeline_v6` → `production.py` →
`direct_rewrite.py`, `DRAFTPROOF_V6_DIRECT_REWRITE` default on). It carries ~11 hardcoded
**content-word** lists/regexes that decide a sentence's structural "shape risk", e.g.:

| Function (line) | Hardcoded content |
|---|---|
| `_predictable_start` ([570](../../../poc/rewrite_v6/scan.py)) | `("today","now","in ","this ","that ","there ","it ","they ","overall")` |
| `_broad_claim` ([603](../../../poc/rewrite_v6/scan.py)) | `("always","never","no longer","the real","the main","the most","one of the")` |
| `_transition_stack` ([608](../../../poc/rewrite_v6/scan.py)) | `("however","additionally","therefore","moreover","furthermore","in addition","as a result")` |
| `_author_anchor_gap` ([587](../../../poc/rewrite_v6/scan.py)) | `("important","challenge","concern")` + evidence markers |
| `_unsupported_claim_gap` ([621](../../../poc/rewrite_v6/scan.py)) | `("should","need to","needs to","important","serious")` |
| `_semantic_bridge_gap`, `_context_anchor_gap`, `_citation_anchor`, `_predictable_start_pressure` | various marker lists / pronoun+modal regexes |

These findings feed `select_target_paragraph` → they decide **which paragraph gets
rewritten**. Because the lists are arbitrary content-word picks, they do not generalize:
a document whose genericity shows up in vocabulary the list never anticipated is mis-targeted.
**Hardcoded content-word lists break the system's ability to cope with all kinds of content.**

`rewrite_compiler/operators.py` is *also* heavily hardcoded but is **out of scope** — it is only
reachable from the legacy V2 pipeline (`run_rewrite_pipeline`), which runs only when V6 is
disabled. It is dead in current production.

---

## What "agnostic" means here (the project's own standard)

The detector codebase already defines it, in code comments:

> "Content-agnostic … signals (**NO hardcoded domain vocabulary, per project rule**)."
> — `poc/detect/layer3_scoring.py:994`, `:1048`

So **agnostic = domain/topic-vocabulary-free**, achieved via *structural / grammatical pattern
classes* (does the sentence contain a number? a proper noun? a citation? a temporal clause? a
first-person frame?). It is **not** "list-free" — structural patterns are still enumerated, but
they generalize across every topic, which arbitrary content-word lists do not.

By this standard:
- `estimate_generic_assertion_risk` / `estimate_lived_detail_risk` (`detect/layer3_scoring.py`)
  **already meet it** (ratio of sentences lacking *any* concrete/contextual structural anchor).
- scan.py's content-word lists **do not** (`"challenge"`, `"the real"`, `"no longer"` are
  topic-flavored, not a structural class).

## Non-goal (rejected approach)

Rewriting scan.py's lists into "principled closed-class lists" (function words, connectives,
modals) was considered and **rejected**: a closed-class list is still a list, still external to
the content, still overfit to a language. It launders the hardcode rather than removing it. The
agnostic signal must be **derived from the content**, not matched against a fixed vocabulary.

---

## Key architectural facts (verified)

1. The default path already computes the agnostic signal:
   `production.py:95` → `source_scan = scan_text_with_report(original_text, detect_json)` merges
   the detector's **content-derived graded signals** (`generic_assertion_risk`,
   `citation_grounding_risk`, per-token predictability) into the scan, and passes it to
   `run_direct_rewrite_all(source_scan=...)` (`direct_rewrite.py:721`).

2. Today the hardcoded regex layer can still **out-rank** the agnostic signal in targeting:
   `_merge_findings` uses `max(structural_severity, report_severity)` when
   `_predictability_primary()` is off (the default), and `select_target_paragraph` sums severity
   per paragraph.

3. The report already separates mitigable from un-mitigable signals via an **`actionability`**
   label (`report.py:285–315`):
   - `high_predictability` / `high_topk_predictability` → `review_only`
   - `medium_predictability` alone → `review_only` ("auto-rewriting it makes things worse")
   - grounding signals (`low_specificity`, `uncited_claim`, …) → `auto_fixable` / `manual_required`
   `scan.py:_report_findings` **already captures** `actionability` into the finding entry
   (`scan.py:252`) but **ignores it** when computing severity.

4. Re-scan passes (`scan_text_preserve_blocks`, `direct_rewrite.py:138/206/...`) carry **no
   detector report** → only the regex layer. But `direct_rewrite.py:96` already reaches for the
   agnostic `estimate_generic_assertion_risk` / `estimate_lived_detail_risk` estimators for
   residual grounding detection.

---

## Design

**Principle: content-derived signals drive targeting; the hardcoded regex layer is demoted to
advisory annotation. The regex is not deleted — it rides along as shape-evidence (aligns with the
project rule "guards/signals ANNOTATE, never drive").**

Behavior change is gated behind a new default-off flag `DRAFTPROOF_V6_AGNOSTIC_TARGETING` and
validated with the deterministic harness before any default flip.

### Change 1 — targeting severity honors `actionability` (the core change)

In `scan.py`, when the agnostic-targeting mode is on, the severity used **for targeting**:
- **excludes** signals labeled `review_only` (this removes un-mitigable predictability/top-k from
  the driver's seat — using the label the report *already* computed),
- is driven by the `auto_fixable` / mitigable grounding signals.

The excluded signals remain present in the finding's `tags`/`evidence` (annotation), so nothing is
hidden from the user — they just stop **driving** which paragraph is rewritten.

This decouples the two things `_predictability_primary()` currently conflates:
- "demote the secondary layer so the content signal drives" — KEEP,
- "rank by raw predictability" — DROP (predictability is un-mitigable; see
  `project_v6_score_floor`, `project_mitigation_ceiling`).

### Change 2 — demote structural-regex findings to advisory in the merge

When the flag is on, structural-only findings (sentence not flagged by the report) are demoted to
`_ADVISORY_SEVERITY` so they annotate but do not out-rank the agnostic grounding signal. This is
the existing `_merge_findings` demotion path, generalized from `_predictability_primary()` to the
new flag and made independent of the predictability-headline behavior.

### Change 3 — re-scan passes lean on the agnostic estimators

For `scan_text_preserve_blocks` re-scans (no report), targeting should not fall back to letting the
regex drive. The residual path already calls `estimate_generic_assertion_risk` /
`estimate_lived_detail_risk` (`direct_rewrite.py:96`); ensure those agnostic estimators — not the
regex shape tags — are what select the residual re-fix target when the flag is on.

### What stays hardcoded, and why (honest scope)

- `_report_signal_severity` tier→score map (`scan.py:352`) — a numeric calibration scale, not a
  content list. Out of scope.
- The grounding estimators' structural patterns (`detect/layer3_scoring.py`) — these *are*
  enumerated regexes, but they are **domain-vocabulary-free structural classes** and already meet
  the project's agnostic standard. Not touched.
- scan.py's regex lists — **kept in the code as advisory annotation**, no longer driving. Deleting
  them is a separate, later decision once the flag has proven out.

So the precise, non-inflated claim: this makes **targeting** domain-agnostic by routing it through
structural grounding signals; it does **not** make the system "list-free."

---

## Risks & validation

- **Under-targeting risk:** demoting the regex could leave nothing driving on documents where the
  detector is quiet (few report findings). The audit memo (`project_detector_hardcode_audit`)
  warns de-overfitting a ratio-scored detect layer can also *over*-flag. **Both directions must be
  measured.**
- **Validation gate:** `DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 4` (N≥4 —
  single runs are noise per `project_v6_measurement_variance`). Compare `final_risk` mean
  flag-off vs flag-on. Ship the default flip only on no-regression (ideally improvement).
- **Kill switch:** `DRAFTPROOF_V6_AGNOSTIC_TARGETING` default off → zero production impact until
  measured and deliberately enabled, no redeploy needed to revert.

## Success criteria

1. With the flag on, paragraph targeting is decided by content-derived grounding signals;
   hardcoded content-word regex no longer changes *which* paragraph is selected.
2. Un-mitigable predictability/top-k never drives targeting (only annotates).
3. `_measure_baseline.py 4` shows `final_risk` flag-on ≤ flag-off (no regression).
4. Existing V6 tests pass; new tests cover: actionability-aware targeting, structural demotion,
   and the no-report re-scan path.

## Out of scope

- `rewrite_compiler/operators.py` (legacy, not in production).
- Deleting scan.py's regex lists (later decision).
- Adding any NLP/POS dependency (none exists in `poc/`; not introducing one).
- Touching the detector's own estimators.
