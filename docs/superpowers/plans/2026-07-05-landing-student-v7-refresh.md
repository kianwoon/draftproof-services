# Landing Page Student-Focused V7 Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the mislabeled "Built for" trust-bar and add a real V7 authorship-clarity-breakdown tab to the landing page's sample-report preview — both student-only, no teacher-facing copy.

**Architecture:** Pure i18n-copy + one component-reuse change in the existing `Landing.jsx` marketing page. No backend, no new components — the V7 tab reuses the real `draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx` component (already imported elsewhere in this exact file for `SubmissionRiskBand`/`PolicyRiskView`), fed a hardcoded illustrative sample object, matching the established pattern in this file.

**Tech Stack:** React 18 + react-i18next, Vite. No test framework exists for this frontend (`draftproof-frontend/package.json` has no test script, zero `*.test.jsx` files) — verification is `npm run build:client` (compiles clean) plus manual visual check via the preview tool.

## Global Constraints

- Teacher-facing copy is explicitly OUT OF SCOPE — do not add any teacher/instructor/educator persona claim anywhere in this change.
- Every new/renamed i18n key must exist in BOTH `i18n/en/landing.js` and `i18n/zh/landing.js` — this codebase has a known "locale trap" failure mode (see `project_seo_landing_pages` memory) where a missing zh key silently falls back to English.
- `landing.researchers` / `landing.educators` / `landing.policyWriters` are used ONLY in `Landing.jsx` (confirmed via repo-wide grep) — safe to rename with no other call sites to update.
- Illustrative sample data must be commented as illustrative/not-a-scoring-oracle, matching the existing convention at the top of `Landing.jsx` (see `SAMPLE_POLICY_RISK` / `SAMPLE_SUBMISSION_RISK` comments).
- Single quote style, 2-space indent, trailing commas — match existing file style exactly (see any existing object literal in `landing.js`).

---

### Task 1: Trust-bar relabel (honest, non-redundant audience copy)

**Files:**
- Modify: `draftproof-frontend/src/i18n/en/landing.js:239-242`
- Modify: `draftproof-frontend/src/i18n/zh/landing.js:237-240`
- Modify: `draftproof-frontend/src/pages/Landing.jsx:161-164`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: i18n keys `landing.students`, `landing.gradWriters`, `landing.eslWriters`, `landing.independentWriters` (en+zh) — Task 2 does not depend on these.

- [ ] **Step 1: Rename and recopy the four keys in `i18n/en/landing.js`**

Replace lines 239-242 (currently):
```js
  "students": "College students",
  "researchers": "University students",
  "educators": "Graduate students",
  "policyWriters": "ESL writers",
```
with:
```js
  "students": "Undergraduates",
  "gradWriters": "Grad & thesis writers",
  "eslWriters": "ESL & multilingual writers",
  "independentWriters": "Independent researchers",
```

- [ ] **Step 2: Rename and recopy the same four keys in `i18n/zh/landing.js`**

Replace lines 237-240 (currently):
```js
  "students": "学院学生",
  "researchers": "大学学生",
  "educators": "研究生",
  "policyWriters": "非母语英语写作者",
```
with:
```js
  "students": "本科生",
  "gradWriters": "研究生与论文写作者",
  "eslWriters": "非母语英语写作者",
  "independentWriters": "独立研究者",
```

- [ ] **Step 3: Update the four `t()` calls in `Landing.jsx`**

In `Landing.jsx:161-164`, replace:
```jsx
          <strong>{t('landing.students')}</strong>
          <strong>{t('landing.researchers')}</strong>
          <strong>{t('landing.educators')}</strong>
          <strong>{t('landing.policyWriters')}</strong>
```
with:
```jsx
          <strong>{t('landing.students')}</strong>
          <strong>{t('landing.gradWriters')}</strong>
          <strong>{t('landing.eslWriters')}</strong>
          <strong>{t('landing.independentWriters')}</strong>
```

- [ ] **Step 4: Confirm no stale references remain**

Run: `grep -rn "landing.researchers\|landing.educators\|landing.policyWriters" draftproof-frontend/src`
Expected: no output (empty grep = pass).

- [ ] **Step 5: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors.

- [ ] **Step 6: Commit**

```bash
git add draftproof-frontend/src/i18n/en/landing.js draftproof-frontend/src/i18n/zh/landing.js draftproof-frontend/src/pages/Landing.jsx
git commit -m "fix(landing): relabel trust-bar audience keys to match their own copy"
```

---

### Task 2: V7 authorship-breakdown tab in the sample-report preview

**Files:**
- Modify: `draftproof-frontend/src/i18n/en/landing.js` (top-level `reportPreviewTabs` array, currently at line 405)
- Modify: `draftproof-frontend/src/i18n/zh/landing.js` (same array, currently at line 403)
- Modify: `draftproof-frontend/src/pages/Landing.jsx` (imports, sample data constants, `SampleReportPreview` default tab + render branch)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: nothing consumed elsewhere — this is the final surface.
- The reused component's contract (from reading `draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx:155`):
  `export default function AuthorshipClarityBreakdown({ t, breakdown, authoritativeTier, tierAuthority })`
  - `breakdown.document_breakdown_raw`: `{ student_owned: number, ai_assisted_polished: number, ai_paraphrased: number, ai_generated_like: number }` (fractions summing to 1)
  - `breakdown.document_breakdown_bands`: same 4 keys, values one of `"Strong" | "Some" | "Little" | "None"`
  - `tierAuthority`: `{ fused_score: number, composite_score: number, proportion: number }` (0-100 scale for scores, 0-1 fraction for proportion)
  - `authoritativeTier`: one of `"green" | "amber" | "orange" | "red"`

- [ ] **Step 1: Add the new tab entry to `i18n/en/landing.js`**

In `reportPreviewTabs` (line 405), insert as the FIRST array element, before the existing `aiSignal` entry:
```js
  "reportPreviewTabs": [
    {
      "id": "authorshipBreakdown",
      "label": "Authorship Breakdown",
      "summary": "4-way composition"
    },
    {
      "id": "aiSignal",
      "label": "AI Signal",
      "summary": "Authorship pattern"
    },
```
(leave the remaining `scoreProfile` / `actionPlan` / `findings` / `criticalThinking` entries unchanged below it)

- [ ] **Step 2: Add the matching zh tab entry to `i18n/zh/landing.js`**

In `reportPreviewTabs` (line 403), insert as the FIRST array element:
```js
  "reportPreviewTabs": [
    {
      "id": "authorshipBreakdown",
      "label": "作者身份细分",
      "summary": "四类构成"
    },
    {
      "id": "aiSignal",
      "label": "AI 信号",
      "summary": "作者身份模式"
    },
```

- [ ] **Step 3: Import the real component and add illustrative sample data in `Landing.jsx`**

Add to the imports at the top of `Landing.jsx` (after the existing `PolicyRiskView` import on line 7):
```jsx
import AuthorshipClarityBreakdown from './report/AuthorshipClarityBreakdown';
```

Add a new module-level constant, placed directly after the existing `SAMPLE_SUBMISSION_RISK` constant (currently ending at line 29), following the same illustrative-data comment convention already used there:
```jsx
// Illustrative V7 authorship-clarity breakdown for the marketing sample report.
// Mirrors the real ai_risk_badge.authorship_breakdown + tier_authority shape
// (poc/detect_v7/breakdown_composer.py, poc/report/builder.py). Values are a
// fixed illustrative example, not a scoring oracle.
const SAMPLE_AUTHORSHIP_BREAKDOWN = {
  document_breakdown_raw: {
    student_owned: 0.62,
    ai_assisted_polished: 0.24,
    ai_paraphrased: 0.09,
    ai_generated_like: 0.05,
  },
  document_breakdown_bands: {
    student_owned: 'Strong',
    ai_assisted_polished: 'Some',
    ai_paraphrased: 'Little',
    ai_generated_like: 'None',
  },
};

const SAMPLE_TIER_AUTHORITY = {
  fused_score: 34,
  composite_score: 29,
  proportion: 0.18,
};
```

- [ ] **Step 4: Render the component in the new tab and default to it**

In `SampleReportPreview` (currently starting around line 804), change the initial state from:
```jsx
  const [activeSection, setActiveSection] = useState('aiSignal');
```
to:
```jsx
  const [activeSection, setActiveSection] = useState('authorshipBreakdown');
```

Then add a new render branch immediately before the existing `{activeSection === 'aiSignal' && (` block:
```jsx
        {activeSection === 'authorshipBreakdown' && (
          <AuthorshipClarityBreakdown
            t={t}
            breakdown={SAMPLE_AUTHORSHIP_BREAKDOWN}
            authoritativeTier="green"
            tierAuthority={SAMPLE_TIER_AUTHORITY}
          />
        )}
```

- [ ] **Step 5: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors (confirms the new import path resolves and JSX is valid).

- [ ] **Step 6: Visual verification via preview tool**

Start the dev server (`mcp__Claude_Preview__preview_start` with the frontend config, or `npm run dev` if no `.claude/launch.json` entry exists yet), navigate to `/`, scroll to the sample-report preview, and confirm:
- The tab list now shows "Authorship Breakdown" as the first tab and it's active by default.
- Four bars render (Student-owned / AI-assisted / AI-paraphrased / AI-generated-like) with correct relative widths (62/24/9/5) and band labels (Strong/Some/Little/None).
- A fused headline shows "34%" with a green "Low" chip and the evidence line "Behind this score: composite detector 29%, deep-scan detector 18% (sentence-level)."
- No console errors (`mcp__Claude_Preview__preview_console_logs`, level `error`).
- Switch the locale to `/zh` and repeat — labels are in Chinese, no English fallback text visible for the new tab.

- [ ] **Step 7: Commit**

```bash
git add draftproof-frontend/src/i18n/en/landing.js draftproof-frontend/src/i18n/zh/landing.js draftproof-frontend/src/pages/Landing.jsx
git commit -m "feat(landing): show real V7 authorship-clarity breakdown in sample report"
```

---

## Self-Review Notes

- **Spec coverage:** Design doc section 1 (trust-bar) → Task 1. Section 2 (V7 sample tab) → Task 2. "Out of scope" (teacher section) → correctly omitted. ✓
- **Placeholder scan:** no TBD/TODO; all code blocks are complete and copy-pasteable. ✓
- **Type/name consistency:** `SAMPLE_AUTHORSHIP_BREAKDOWN` / `SAMPLE_TIER_AUTHORITY` names used identically in Step 3 (definition) and Step 4 (usage); prop names (`breakdown`, `authoritativeTier`, `tierAuthority`) match the real component's destructured signature read directly from `AuthorshipClarityBreakdown.jsx:155`. ✓
