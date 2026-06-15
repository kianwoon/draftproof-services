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

**Principle: content-derived signals drive targeting; the hardcoded content-word detectors are
removed. scan.py keeps only its structural-agnostic components. Un-mitigable signals
(predictability/top-k) annotate but never drive — using the report's existing `actionability`
label. No flag: the agnostic behavior is the only behavior.**

**No feature flag.** The change is unconditional. A permanent on/off switch does not make the
function agnostic — it hides a non-agnostic one behind a toggle, and becomes another never-flipped
lever. The hardcoded content-word detectors are **removed**, not demoted behind a flag. The
deterministic harness is a **development-time validation gate**, not a runtime switch: if the
agnostic version regresses, the agnostic logic is fixed until it holds — it is never flagged off.

### First, bucket scan.py's own risk components by agnosticism

scan.py's `_risk` / `_tags` mix two kinds of component. Only the second kind is removed.

| Already agnostic (structural — **keep**) | Hardcoded content-word (non-agnostic — **remove**) |
|---|---|
| `_list_pressure` (comma/separator density) | `_predictable_start` / `_predictable_start_pressure` (opener word + pronoun/modal list) |
| `_abstract_pressure` (nominalization morphology) | `_context_anchor_gap` (connective + demonstrative list) |
| `_named_anchor_count` (proper-noun count) | `_author_anchor_gap` (`important/challenge/concern` + evidence markers) |
| `_repeated_frame_findings` (n-gram frame repetition) | `_broad_claim` (`always/never/the real/the main/...`) |
| sentence `word_count` overload | `_transition_stack` / `_semantic_bridge_gap` (connective lists) |
| | `_unsupported_claim_gap` (`should/need to/important/serious`) |
| | `_citation_anchor` reporting-verb list (keep only the structural `et al. (\d{4})` form) |

### Change 1 — targeting is driven by content-derived grounding signals

The severity used **for targeting** is computed from the detector's content-derived grounding
signals (`generic_assertion_risk`, `citation_grounding_risk`, lived-detail), which already flow in
via `scan_text_with_report`. These are domain-vocabulary-free structural estimators
(`detect/layer3_scoring.py`) that cope with any topic.

### Change 2 — targeting honors `actionability`; remove the hardcoded content-word detectors

- Targeting severity **excludes** signals the report already labels `review_only` (this keeps
  un-mitigable predictability/top-k out of the driver's seat — using the label the report already
  computed at `report.py:285–315`). They remain in `tags`/`evidence` as annotation; they just stop
  deciding which paragraph is rewritten.
- The hardcoded content-word detectors in the right column above are **deleted** from `_risk` /
  `_tags`. The remaining `_risk` keeps only the structural-agnostic components (left column), so
  scan.py's intrinsic shape score no longer matches arbitrary vocabulary.
- The conflated `_predictability_primary()` switch is removed; "content signal drives" becomes the
  unconditional behavior, and "rank by raw predictability" is dropped (predictability is
  un-mitigable; see `project_v6_score_floor`, `project_mitigation_ceiling`).

### Change 3 — re-scan passes use the agnostic estimators

`scan_text_preserve_blocks` re-scans carry no detector report. Residual targeting uses the agnostic
`estimate_generic_assertion_risk` / `estimate_lived_detail_risk` (already called at
`direct_rewrite.py:96`) plus scan.py's surviving structural-agnostic components — never the deleted
content-word regexes.

### What is intentionally not touched

- `_report_signal_severity` tier→score map (`scan.py:352`) — a numeric calibration scale, not a
  content list.
- The detector's grounding estimators (`detect/layer3_scoring.py`) — already domain-agnostic by the
  project's standard; they are the replacement, not a target.

**Precise claim:** targeting becomes domain-agnostic (driven by structural grounding signals), and
the non-agnostic content-word detectors are removed from scan.py. scan.py's surviving components
and the detector's estimators are still *enumerated structural patterns* — domain-vocabulary-free,
which is the project's agnostic bar, not "list-free."

---

## Risks & validation

- **Under-targeting risk:** removing the content-word detectors could leave nothing driving on
  documents where the detector is quiet (few report findings). The audit memo
  (`project_detector_hardcode_audit`) warns a ratio-scored detect layer can also *over*-flag.
  **Both directions must be measured and resolved in the code — not toggled off.**
- **Validation gate (development-time):** `DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 4`
  (N≥4 — single runs are noise per `project_v6_measurement_variance`). Compare `final_risk` mean of
  the current `main` against the branch. If the agnostic version regresses, fix the agnostic logic
  (e.g. tune how grounding severity maps to targeting) until `final_risk` is ≤ baseline. Do not
  merge a regression; do not add a flag to dodge it.

## Success criteria

1. Paragraph targeting is decided by content-derived grounding signals; no hardcoded content-word
   regex influences *which* paragraph is selected. No env flag exists for this behavior.
2. The content-word detectors (right column above) are removed from `scan.py`'s `_risk`/`_tags`.
3. Un-mitigable predictability/top-k never drives targeting (only annotates).
4. `_measure_baseline.py 4` shows branch `final_risk` ≤ `main` (no regression).
5. Existing V6 tests pass; new tests cover: actionability-aware targeting, content-word-detector
   removal, and the no-report re-scan path.

## Out of scope

- `rewrite_compiler/operators.py` (legacy, not in production).
- Adding any NLP/POS dependency (none exists in `poc/`; not introducing one).
- Re-tuning the detector's own estimators beyond what targeting needs.
