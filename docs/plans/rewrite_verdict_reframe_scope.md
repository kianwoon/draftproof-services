# Rewrite Verdict Reframe — Scope (design, not implementation)

**Date:** 2026-07-15  **Author:** scoping agent
**Trigger:** fine-tune v1 deep-scan detector (shipped 2026-07-14) correctly reads the rewrite
writer (Cerebras gpt-oss-120b) as AI — fused ~65/orange, 7/9 docs composite=0. The before/after
**AI-likelihood** number no longer falls on LLM-drafted text, so any verdict keyed on
`final_risk < original_risk` now fires "regressed / score_worse / Review Needed" on almost every
rewrite. Per CLAUDE.md "Objective & Rewrite Philosophy": the rewrite is a **shown solution /
reviewable draft** — success = **content gaps surfaced & filled** (grounding, specifics), NOT a
lower detector score. Guards **annotate, never suppress**. The after-score stays visible.

---

## 1. Inventory — current verdict surfaces (file:line)

| Surface | Location | What it computes/renders |
|---|---|---|
| Verdict fields (producer) | `poc/rewrite_v6/production.py:322-325` | `changed`/`cleared` → `status ∈ {ai_mitigated, partial_candidate_not_strict_safe, original_preserved}` |
| Summary payload | `production.py:333-391` | `status/outcome/public_status`, `original_risk` (378), `final_risk` (379), `detect_scores`, `strict_safe_band_achieved` (340) |
| detect_scores producer | `production.py:845-871` | original/rewritten `ai`, `grounding_quality_risk`, `human_contribution`, `findings` counts — **the signals the reframe reuses** |
| External guard status | `production.py:255` | `original_preserved_external_guard` |
| PDF label logic | `poc/report/render_rewrite.py:869-893` (`_result_label`) + `999-1069` | derives `ai_improved/worse`, `findings_improved/worse`, `score_worse`, `improved_with_review`, `mixed_result`, `scan_regressed` → label string; then overridden by `outcome` (1060-1069) |
| PDF hero panel | `poc/report/render_panels.py:728-738` (`rewrite_hero`) | `good = outcome=='ai_mitigated' or (ai_improved and not score_worse)` — **green keyed on score drop** |
| PDF outcome stamp | `render_rewrite.py:265-309` | stamp text from `final_risk` + `result_label` |
| Frontend risk fields | `draftproof-frontend/src/pages/report/reportHelpers.js:1107-1108` | `original_risk`, `rewrite_risk` (from badges/detect_scores) |
| Frontend render | `pages/Report.jsx` (`rewriteDetectorVerdict`, ~91) + i18n `src/i18n/en/rewritePage.js` | comparison section + verdict copy |
| Email lead | `worker/app/email_service.py:172-197` (`_rewrite_fused_lead_lines`) | `"AI likelihood: X% -> Y%"` + composite/deep-scan deltas |

---

## 2. Signals that ALREADY exist (no new detector work)

From `detect_scores` (production.py:853-871) + scan report:
- `original_findings` / `rewritten_findings` (full-detector counts) → **findings resolved**.
- `original_grounding_quality_risk` / `rewritten_grounding_quality_risk` → **grounding delta**.
- `original_human_contribution` / `rewritten_human_contribution` + `human_shift_score` → **authorship shift**.
- Per-category component risks (`generic_assertion_risk`, `citation_grounding_risk`) live in each
  scan's `ai_components`/findings (`layer3_scoring.py:66`, `235+`) — available on both original &
  rewritten reports; deltas computable in `_detect_scores_for_summary`.
- `bracket_grounding_spans` (production.py:396) `kind ∈ {improved, kept}` → **count of grounding
  anchors added** (green spans).

**Conclusion (precision gate):** the reframe is viable on existing data — findings-resolved,
grounding-risk delta, and grounding-anchors-added are all real, per-document, anchored signals.

---

## 3. New verdict field spec (additive to summary payload)

Produced in `production.py::run_rewrite_pipeline_v6` (and mirrored in `_detect_scores_for_summary`):

| Field | Type | Producer | Definition |
|---|---|---|---|
| `gap_resolution` | dict | production.py after detect_scores | container below |
| `gap_resolution.findings_before/after` | int | detect_scores findings | copy of counts |
| `gap_resolution.findings_resolved` | int | `max(before-after,0)` | primary success number |
| `gap_resolution.grounding_risk_before/after` | float | grounding_quality_risk | |
| `gap_resolution.grounding_risk_delta` | float | after-before (neg = better) | |
| `gap_resolution.generic_assertion_delta` | float | category component | neg = better |
| `gap_resolution.citation_grounding_delta` | float | category component | neg = better |
| `gap_resolution.anchors_added` | int | count `bracket_grounding_spans kind==improved` | concrete specifics added |
| `verdict_label` | str | new `_verdict_label()` (§4) | taxonomy below |
| `ai_likelihood_note` | str | constant | "expected to remain high on AI-drafted text — make it yours" |

**No suppression:** `original_risk`/`final_risk` remain exactly as-is (production.py:378-379).
The after-number is still shown; `ai_likelihood_note` reframes it.

---

## 4. New label taxonomy + deterministic rules

Replace the score-drop-keyed labels. New `verdict_label` computed from **gap** signals, not AI score:

| Label | Rule (deterministic) |
|---|---|
| `no_usable_rewrite` | `no_text_change` OR status `original_preserved*` (unchanged fallback) |
| `gaps_resolved` | `changed` AND `findings_resolved >= 1` AND `grounding_risk_delta <= -0.05` AND no category delta worse |
| `gaps_partially_resolved` | `changed` AND (`findings_resolved >= 1` OR `grounding_risk_delta <= -0.05` OR `anchors_added >= 1`) but not all |
| `draft_for_review` | `changed` but no measurable gap improvement (still a shown solution to edit) |

`_result_label` (render_rewrite.py:869) and `rewrite_hero.good` (render_panels.py:738) are
**rewired to `verdict_label`**, NOT to `ai_improved/score_worse`. `score_worse`/`scan_regressed`
branches (1016, 1031) are demoted to an advisory sub-line, never the headline (annotate-don't-suppress).
`good` (green hero) := `verdict_label in {gaps_resolved, gaps_partially_resolved}`.

---

## 5. Copy skeletons

**Web (Report.jsx / rewritePage.js i18n):**
- Headline per label: "Content gaps resolved" / "Gaps partially resolved" / "Draft ready to make yours" / "No usable rewrite — edit manually".
- Sub: "N findings resolved · grounding risk ↓ X% · M specifics added."
- AI-likelihood row keeps `X% -> Y%` + badge: "AI likelihood stays high because this is an AI-drafted demo. **Replace it with your own writing** — that's what lowers it."

**PDF (render_rewrite hero + stamp):** same headline; stamp shows gap metrics as the primary line,
AI-likelihood as secondary with the make-it-yours note.

**Email (`_rewrite_fused_lead_lines`):** lead with gap line first —
`"Content gaps: N findings resolved, M specifics added"` — then keep `AI likelihood: X% -> Y%`
followed by one sentence: "This stays high on AI-drafted text; edit it into your own words."

---

## 6. Effort + rollout order (kill-switched, additive-first)

1. **Data fields** (`gap_resolution`, `verdict_label`, `ai_likelihood_note`) in production.py +
   `_detect_scores_for_summary` — **~0.5 day**. Additive, no surface reads them yet. Gate behind
   `DRAFTPROOF_REWRITE_VERDICT_REFRAME` (default 0 first, flip to 1 after fields verified in R2).
2. **PDF** (`_result_label`/`rewrite_hero` rewire to `verdict_label`) — **~0.5 day**. Fallback to
   old label when flag off / field absent.
3. **Web** (Report.jsx + i18n en/zh) — **~0.5 day**. Reads `verdict_label`; old path when null.
4. **Email** (`_rewrite_fused_lead_lines`) — **~0.25 day**.

Total ~1.75 days. Each surface fails open to current behaviour when the field is absent (legacy
rewrites in R2). Verify with `DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 4`.

## 7. Risks
- **Category-delta signal reliability** — `generic_assertion_risk`/`citation_grounding_risk` deltas
  can be noisy on short paras; treat as advisory, gate `gaps_resolved` primarily on
  `findings_resolved` + `grounding_risk_delta` (the two most stable). (precision-first)
- **Findings counts not length-normalized** (production.py:867 note): a longer, better-grounded
  rewrite can show *more* absolute findings — do NOT key `regressed` on `findings_worse` alone.
- **i18n/zh** must localize new keys or /zh renders English (LOCALIZABLE trap, MEMORY).
- **Additive-composer rule:** new badge fields must reach web+PDF+email+read-time API; never
  back-to-back push (wedges worker).
