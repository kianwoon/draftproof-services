# Landing page: honest audience labels + V7 sample-report refresh

**Scope:** Student-only. Teacher-facing landing copy was scoped out for now (shelved, not built) — revisit separately later.

## Problem

Reviewed the landing page (`draftproof-frontend/src/pages/Landing.jsx` + `i18n/{en,zh}/landing.js`) against the newly-launched V7 authorship-clarity breakdown (`poc/detect_v7/`, live in production since 2026-07-04 per `project_v7_authorship_breakdown` memory). Two gaps, both student-facing:

1. **Trust-bar mislabel.** The "Built for" bar (`Landing.jsx:158-170`) renders four `t()` keys — `students`, `researchers`, `educators`, `policyWriters` — but the *displayed copy* doesn't match the key names at all: `educators` → "Graduate students", `policyWriters` → "ESL writers". No key or copy actually says "teacher" or "educator" in a way that's true — this is leftover copy drift, not an intentional teacher claim. Precision-first: don't ship copy whose displayed meaning contradicts its own key name.
2. **Stale sample-report preview.** `SampleReportPreview` (`Landing.jsx:804+`) is the landing page's demo of what a scan report looks like. It shows only the old single-score "AI Signal" framing (`calibratedTopk`, `originalScanScore`, etc.) — no mention of the 4-category V7 authorship breakdown that now drives the *actual* tier in production (`DRAFTPROOF_V7_TIER_AUTHORITY`, live). A prospective student previewing the sample report today doesn't see DraftProof's newest, most defensible feature.

## Design

### 1. Trust-bar relabel

Rename the four i18n keys to be self-describing and drop the redundant near-duplicates ("College students" vs "University students" say almost the same thing). New copy, still 100% student-segment (no teacher claim):

| Old key | Old copy (en) | New key | New copy (en) | New copy (zh) |
|---|---|---|---|---|
| `students` | College students | `students` | Undergraduates | 本科生 |
| `researchers` | University students | `gradWriters` | Grad & thesis writers | 研究生与论文写作者 |
| `educators` | Graduate students | `eslWriters` | ESL & multilingual writers | 非母语英语写作者 |
| `policyWriters` | ESL writers | `independentWriters` | Independent researchers | 独立研究者 |

Files touched: `i18n/en/landing.js`, `i18n/zh/landing.js` (rename keys + new copy), `Landing.jsx:161-164` (update the four `t('landing.*')` calls to the new key names).

### 2. V7 sample-report tab

Add a new tab to `reportPreviewTabs` (`landing.js`) — id `authorshipBreakdown`, label "Authorship Breakdown", summary "4-way composition" — inserted as the **first** tab (real reports now lead with V7 per the `report_clarity_reframe` memory: "report V7-centered").

Render it in `SampleReportPreview` by importing and reusing the **real** `AuthorshipClarityBreakdown` component (`./report/AuthorshipClarityBreakdown`), following the exact precedent already in this file for `SubmissionRiskBand`/`PolicyRiskView` (hardcoded illustrative sample object at module scope, e.g. `SAMPLE_AUTHORSHIP_BREAKDOWN`). This is a reuse, not a re-implementation — it guarantees the landing preview can never visually drift from the real report again, and needs no new CSS (already global via `main.jsx:6` → `site-master.css`).

Illustrative sample values (clearly commented as illustrative, not a scoring oracle, matching the file's existing comment convention):
- `document_breakdown_raw`: student_owned 0.62, ai_assisted_polished 0.24, ai_paraphrased 0.09, ai_generated_like 0.05
- `document_breakdown_bands`: Strong / Some / Little / None (respectively)
- `tierAuthority`: `fused_score` 34, `composite_score` 29, `proportion` 0.18
- `authoritativeTier`: `"green"` (consistent with the sample essay's existing "Low Risk" framing elsewhere on the page)

No zh-specific work needed here beyond what `AuthorshipClarityBreakdown` already pulls from `report.authorshipBreakdown.*` i18n keys (zh already has that report page localized — confirmed the same key path is used identically on the report page, so it's proven live already, not new i18n surface).

## Out of scope (explicitly)

- Any teacher/instructor-facing section or copy — shelved per user decision.
- Per-paragraph V7 breakdown (backend limitation, doc-level only today).
- Deep-scan headline variant in the sample (only the fused/tier-authority headline is shown, since that's the state production is actually in).

## Testing / verification

- Visual check via preview server: trust-bar shows new labels (en + zh), new tab renders the real breakdown bars with correct colors/percentages, no console errors.
- Grep confirms no remaining references to the old key names (`researchers`, `educators`, `policyWriters` as landing.js audience keys) anywhere else in the frontend.
