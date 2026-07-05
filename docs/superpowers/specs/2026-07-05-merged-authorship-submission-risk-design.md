# Merged Authorship Clarity + Submission Risk — Design

**Date:** 2026-07-05
**Status:** Approved for planning
**Scope:** Web report page (React) **and** downloadable PDF. Extension view out of scope for this change.

## Problem

The scan report renders two adjacent panels that overlap and, at times, compete:

- **Authorship Clarity Breakdown** ([AuthorshipClarityBreakdown.jsx](../../../draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx)) — a fused **AI-likelihood %** headline + tier chip, then 4 composition bars (student_owned → ai_generated_like), a disclaimer, and a beta feedback link.
- **Submission Risk** ([SubmissionRiskBand.jsx](../../../draftproof-frontend/src/pages/report/SubmissionRiskBand.jsx)) — an overall level + "you can defend this as your own work" line, 5 axis chips, an **AI-likelihood note** (the same number), and the DraftProof score-band table.

Four confirmed problems (owner, 2026-07-05): (1) the AI-likelihood % and the DraftProof-scale reference are duplicated across both panels; (2) two stacked boxes make the hero too tall; (3) "Student-owned · 37%" next to "Ownership risk: Low" reads as contradictory — the code even carries a "semantic bridge" paragraph fighting this; (4) the two boxes want a single polished treatment.

## Goal

Replace the two panels with **one unified card** that has a single headline number, a single scale, and presents the composition bars and risk axes as two clearly-labeled **lenses** rather than two rival verdicts. Ship identically to page and PDF.

## Design

One card, top to bottom:

1. **Header** — title ("Authorship & submission risk") + `BETA` chip + one-line subtitle.
2. **Verdict band** (the ONE headline) — big AI-likelihood % + level chip (LOW/…), the ownership lead line ("You can defend this as your own work"), and a muted evidence sub-line ("Fused from composite 14% · deep-scan 31% · flag line is 32 — not a Turnitin score."). This subsumes the authorship fused headline + the submission-risk level + ownership lead + the duplicate AI-likelihood note.
3. **Two lenses, side by side:**
   - **How the writing reads · sums to 100%** — the 4 composition bars. **Color-coded** (owner decision 2026-07-05): student_owned = teal/green (`#1D9E75`), ai_assisted_polished = neutral gray (`#888780`), ai_paraphrased + ai_generated_like = coral (`#D85A30`). Percentages always shown next to each band label (preserves owner decision 2026-07-04).
   - **Where the risk sits · independent** — the 5 axis chips (text_pattern, ownership, citation, defence_readiness, policy_declaration), each `label · level`, color by level.

   The explicit lens headers (`sums to 100%` vs `independent`) are the structural fix for problem (3): they replace the buried semantic-bridge paragraph.
4. **DraftProof scale** — the 4-row band table (0–32 Low, 32–48 Medium, 48–65 High, 65+ Critical). **Expanded by default** (owner decision 2026-07-04 preserved — it's load-bearing context for the AI-likelihood %). Behind a `<details open>` toggle so it stays collapsible.
5. **Footer** — merged single line: disclaimer + beta feedback link ("Does this match your read? Tell us").

Reference mockup was shown and approved in the brainstorming session (visualize widget `merged_authorship_submission_risk_card`).

## Data sources (unchanged — merge is presentational)

The merged component consumes the **same props already flowing today**; no backend/report-JSON changes:

- **Verdict headline:** `tierAuthority.fused_score` + `authoritativeTier` when tier_authority present (fused path); else `breakdown.deep_scan` (deep-scan path). Evidence line uses `tierAuthority.composite_score` / `tierAuthority.proportion`.
- **Ownership lead + flag line + level:** `sr.overall.level`, `sr.overall` ownership lead, `sr._fused`, `sr._flagLine`.
- **Composition bars:** `breakdown.document_breakdown_raw` (widths) + `breakdown.document_breakdown_bands` (band labels).
- **Risk axes:** `sr.axes[key].level` for the 5 `SUBMISSION_RISK_AXES` keys.
- **Scale table:** static band cutoffs already in `SubmissionRiskBand` (real committed `tier_authority` cutoffs — not invented).

## Null-safety (preserve additive/house pattern)

The card degrades gracefully — same "render nothing if data absent" contract both panels honor today:

- Both `breakdown` and `sr` absent → render nothing.
- Only `breakdown` → header + verdict (deep-scan/fused) + composition lens; omit risk-axes lens + scale.
- Only `sr` → header + verdict (level + flag line) + risk-axes lens + scale; omit composition lens.
- Both present → full card.

## Implementation approach

### Page (React)

- New component `MergedAuthorshipRisk.jsx` in `draftproof-frontend/src/pages/report/`, absorbing the render logic of both `AuthorshipClarityBreakdown.jsx` and `SubmissionRiskBand.jsx`. Reuse the existing sub-pieces (`CategoryBar`, `DeepScanHeadline`/`FusedHeadline`, axis-chip loop).
- Reuse **all existing `t()` i18n keys** — no new translation strings unless a new label (e.g. the two lens headers) is genuinely new; add those under the existing `report.authorshipBreakdown` / `report.submissionRisk` namespaces.
- Wire in [ReportHero.jsx](../../../draftproof-frontend/src/pages/report/ReportHero.jsx): today `afterTitle` (authorship) then `<SubmissionRiskBand>` render as two blocks. Replace both with the single `<MergedAuthorshipRisk>`. Update the call site in [Report.jsx](../../../draftproof-frontend/src/pages/Report.jsx) (props `breakdown`, `submissionRiskView`, `authoritativeTier`, `tierAuthority`).
- Delete `AuthorshipClarityBreakdown.jsx` + `SubmissionRiskBand.jsx` once the merged component is wired and verified (no other importers — confirm via grep first).
- New CSS for `.merged-authorship-risk` in the report stylesheet; retire the old `.authorship-breakdown-*` / `.submission-risk-*` rules that are no longer referenced.

### PDF (Python)

- Merge `render_authorship_breakdown()` + `render_scan_lead()` in [render_panels.py](../../../poc/report/render_panels.py) (555 lines — has headroom; **do not** grow [render.py](../../../poc/report/render.py), already 1930 lines) into a single `render_merged_authorship_risk(report, data)` returning one HTML block with the same top-to-bottom structure.
- PDF strings stay **hardcoded English** (existing PDF pattern — no `t()`), mirroring the frontend copy.
- Update the integration in `render.py::render_report()` — today it calls `render_scan_lead()` then `render_authorship_breakdown()` (note: PDF order is risk-first, page is authorship-first; the merged card fixes the order to one canonical sequence). Replace both calls with the single merged call.
- Merge/rename the CSS in [pdf.py](../../../poc/report/pdf.py) (`.dp-abd-*`, `.dp-hero`, `.dp-statchip`) to back the unified block; color-coded bars must match the page (`#1D9E75` / `#888780` / `#D85A30`).

## Constraints

- **Additive & null-safe** — no report-JSON schema change; older reports still render.
- **No hardcoded scoring** — the scale-band cutoffs are the existing committed values; do not invent new thresholds or recompute bands on the frontend (keep `TIER_TO_BAND` single-source pattern).
- **1500-line file limit** — new page component well under; keep `render.py` from growing (put merged PDF logic in `render_panels.py`).
- **Verify rendered artifacts** — per house rule, RENDER the PDF and visually inspect the page; grep proves emit-in-code, not render-correct.

## Out of scope

- Word / Google-Docs extension view (separate follow-up).
- Any change to the underlying scores, tiers, or report JSON.
- Changing which axes/categories exist.

## Success criteria

- Page and PDF each show **one** card with **one** AI-likelihood number and **one** scale.
- Both lenses render with their `sums to 100%` / `independent` headers.
- Older reports (missing `breakdown` or `sr`) degrade per the null-safety table with no crash.
- Hero is visibly shorter than the two-panel version.
- Color-coded bars + expanded-by-default scale match on page and PDF.
