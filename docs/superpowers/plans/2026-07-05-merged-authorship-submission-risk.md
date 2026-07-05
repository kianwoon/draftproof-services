# Merged Authorship + Submission Risk — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two adjacent report panels (Authorship Clarity Breakdown + Submission Risk) with one unified card — one AI-likelihood headline, one scale, two labeled lenses — on the web page and (co-located, de-duped) in the PDF, and drop the legacy Writing Score + Findings chips from the web hero.

**Architecture:** The web page gets a new `MergedAuthorshipRisk.jsx` that absorbs both old components and renders them as one card wired into `ReportHero`. The PDF keeps its richer submission section but co-locates the authorship bars + a single AI-likelihood headline at the top and de-dupes the number. Purely presentational — no report-JSON/backend/scoring change.

**Tech Stack:** React + Vite (frontend, no test harness — verify via preview), Python HTML-string builders + WeasyPrint (PDF, pytest via `poc/test_render_authorship_breakdown.py`).

**Reference:** Spec at [docs/superpowers/specs/2026-07-05-merged-authorship-submission-risk-design.md](../specs/2026-07-05-merged-authorship-submission-risk-design.md). Approved mockup: visualize widget `merged_authorship_submission_risk_card`.

## Global Constraints

- **Additive & null-safe** — no report-JSON schema change; older reports (missing `breakdown` or `sr`) must still render per the null-safety table. Render nothing when both absent.
- **No hardcoded scoring** — scale-band cutoffs are the existing committed values; do not invent thresholds or recompute bands on the frontend. Keep the `TIER_TO_BAND = { green:'low', amber:'moderate', orange:'high', red:'critical' }` single-source pattern.
- **No file over 1500 lines** — new page component well under; do NOT grow `poc/report/render.py` (already 1930 lines) — new PDF logic goes in `poc/report/render_panels.py` (555 lines).
- **Do not relabel** any remaining legacy figure (Risk Tier) as "V7".
- **Report-JSON untouched** — `writing_quality_score` and the findings array stay in the badge JSON; only the two web-hero chips are removed from render.
- **Bar colors (page + PDF must match):** `student_owned` = `#1D9E75`, `ai_assisted_polished` = `#888780`, `ai_paraphrased` = `#D85A30`, `ai_generated_like` = `#D85A30`.
- **Scale expanded by default** (`<details open>`); **percentages always shown** next to each composition band.
- **PDF strings stay hardcoded English** (no `t()`), mirroring the frontend copy. **Verify rendered artifacts** — actually render the PDF and view the page; grep proves emit-in-code, not render-correct.

## File Structure

- **Create** `draftproof-frontend/src/pages/report/MergedAuthorshipRisk.jsx` — the unified card (absorbs both old components' render logic + sub-pieces).
- **Modify** `draftproof-frontend/src/i18n/en/report.js` + `draftproof-frontend/src/i18n/zh/report.js` — add a `report.merged` block (title + two lens headers).
- **Modify** `draftproof-frontend/src/pages/report/ReportHero.jsx` — render `<MergedAuthorshipRisk>`, remove the Writing Score + Findings stat chips.
- **Modify** `draftproof-frontend/src/pages/Report.jsx` — pass merged props; stop passing `writingScore`/`issuesCount` where they only fed the removed chips.
- **Modify** `draftproof-frontend/src/styles/site-master/06-report-overview.css` — add `.merged-authorship-risk` rules; adjust `.report-hero-stats` for a single remaining stat.
- **Delete** `draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx` + `SubmissionRiskBand.jsx` once unreferenced.
- **Modify** `poc/report/render_panels.py` — add `render_merged_authorship_risk(report, data)`; refactor the two existing functions into shared pieces.
- **Modify** `poc/report/render.py` — call the merged function in `render_report()` instead of the two separate calls.
- **Modify** `poc/report/pdf.py` — add color-coded `.dp-abd-bar-fill` rules.
- **Modify** `poc/test_render_authorship_breakdown.py` — extend/add a test for the merged PDF render.

---

## Task 1: Add i18n keys for the merged card

**Files:**
- Modify: `draftproof-frontend/src/i18n/en/report.js` (insert before `"advancedSignals"` at line ~736)
- Modify: `draftproof-frontend/src/i18n/zh/report.js` (analogous location)

**Interfaces:**
- Produces: `t('report.merged.title')`, `t('report.merged.compositionLens')`, `t('report.merged.compositionLensNote')`, `t('report.merged.riskLens')`, `t('report.merged.riskLensNote')` — consumed by Task 2.

- [ ] **Step 1: Add the `merged` block to en/report.js**

Insert this key block into the `report` object in `draftproof-frontend/src/i18n/en/report.js`, immediately before the `"advancedSignals"` key (line ~736):

```javascript
  "merged": {
    "title": "Authorship & submission risk",
    "subtitle": "One read on how this document scores — the headline number, how the writing distributes, and where the risk sits.",
    "compositionLens": "How the writing reads",
    "compositionLensNote": "sums to 100%",
    "riskLens": "Where the risk sits",
    "riskLensNote": "independent"
  },
```

- [ ] **Step 2: Add the same block to zh/report.js**

Insert the analogous block into `draftproof-frontend/src/i18n/zh/report.js` at the matching location, with Chinese copy:

```javascript
  "merged": {
    "title": "作者归属与提交风险",
    "subtitle": "对本文档评分的综合解读——核心数值、写作构成分布，以及风险所在。",
    "compositionLens": "写作构成解读",
    "compositionLensNote": "合计为 100%",
    "riskLens": "风险所在",
    "riskLensNote": "各自独立"
  },
```

- [ ] **Step 3: Verify both files parse**

Run: `cd draftproof-frontend && node -e "import('./src/i18n/en/report.js').then(m=>console.log('en ok', !!m.default.merged)); import('./src/i18n/zh/report.js').then(m=>console.log('zh ok', !!m.default.merged))"`
Expected: `en ok true` and `zh ok true` (if the file isn't an ESM default export, instead run `npm run build` in a later task to catch syntax errors; a quick `node --check` on the transpiled output is not available for JSX-adjacent JS, so rely on the build).

Fallback verify (always works): `cd draftproof-frontend && npx eslint src/i18n/en/report.js src/i18n/zh/report.js` — Expected: no parse errors.

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/i18n/en/report.js draftproof-frontend/src/i18n/zh/report.js
git commit -m "i18n(report): add merged authorship+risk card keys (en, zh)"
```

---

## Task 2: Create the MergedAuthorshipRisk component

**Files:**
- Create: `draftproof-frontend/src/pages/report/MergedAuthorshipRisk.jsx`

**Interfaces:**
- Consumes props: `{ t, breakdown, sr, authoritativeTier, tierAuthority }` where:
  - `breakdown` = `badge.authorship_breakdown` (or null) — has `document_breakdown_raw`, `document_breakdown_bands`, `deep_scan`, `disclaimer`.
  - `sr` = the submission-risk view (or null) — has `overall.level`, `axes[key].level/display_score`, `_fused`, `_flagLine`.
  - `authoritativeTier` = `badge.tier || report.tier` (lowercase green/amber/orange/red).
  - `tierAuthority` = `badge.tier_authority` (or null) — has `fused_score`, `composite_score`, `proportion`.
- Produces: default export `MergedAuthorshipRisk` — consumed by Task 3.
- Reuses helper: `import { SUBMISSION_RISK_AXES } from './reportHelpers';`

- [ ] **Step 1: Write the component**

Create `draftproof-frontend/src/pages/report/MergedAuthorshipRisk.jsx` with this full content. It absorbs the render logic of `AuthorshipClarityBreakdown.jsx` (fused/deep-scan headline + bars) and `SubmissionRiskBand.jsx` (level, ownership lead, axes, note, scale), reorganized into: header → verdict band → two lenses → scale → footer. Null-safety per the spec table.

```jsx
// draftproof-frontend/src/pages/report/MergedAuthorshipRisk.jsx
// Unified V7 card: one AI-likelihood headline + two labeled lenses (composition bars
// vs risk axes) + one DraftProof scale. Replaces AuthorshipClarityBreakdown.jsx +
// SubmissionRiskBand.jsx. Additive/null-safe: renders nothing when both `breakdown`
// and `sr` are absent; degrades gracefully when only one is present.
import { SUBMISSION_RISK_AXES } from './reportHelpers';

const CATEGORY_ORDER = [
  'student_owned',
  'ai_assisted_polished',
  'ai_paraphrased',
  'ai_generated_like',
];

// Bar-fill color per category (spec Global Constraints — must match PDF pdf.py).
const CATEGORY_COLOR = {
  student_owned: '#1D9E75',
  ai_assisted_polished: '#888780',
  ai_paraphrased: '#D85A30',
  ai_generated_like: '#D85A30',
};

const KNOWN_DEEP_SCAN_BANDS = ['insufficient', 'amber', 'orange', 'red'];
const KNOWN_AUTHORITATIVE_TIERS = ['green', 'amber', 'orange', 'red'];
// Single source of truth: band label + chip color both derive from the tier the
// backend assigned. Do NOT recompute a band from the fused score against frontend
// cutoffs (duplicates backend threshold logic — a no-hardcode violation).
const TIER_TO_BAND = { green: 'low', amber: 'moderate', orange: 'high', red: 'critical' };

function CategoryBar({ t, category, raw, band }) {
  const hasRaw = typeof raw === 'number' && Number.isFinite(raw);
  const widthPct = hasRaw ? Math.max(0, Math.min(100, raw * 100)) : 0;
  const bandLabel = band
    ? t(`report.authorshipBreakdown.bands.${band}`)
    : t('report.authorshipBreakdown.bands.None');
  return (
    <div className="merged-comp-row">
      <div className="merged-comp-row-head">
        <span className="merged-comp-label">
          {t(`report.authorshipBreakdown.categories.${category}`)}
        </span>
        <span className="merged-comp-band">
          {hasRaw ? `${bandLabel} · ${Math.round(widthPct)}%` : bandLabel}
        </span>
      </div>
      <div className="merged-comp-track">
        <div
          className="merged-comp-fill"
          style={{ width: `${widthPct}%`, background: CATEGORY_COLOR[category] }}
        />
      </div>
    </div>
  );
}

// The ONE headline. Fused score (verdict) when tier_authority present, else the
// deep-scan-only estimate. The ownership lead + flag-line note ride alongside.
function VerdictBand({ t, breakdown, sr, authoritativeTier, tierAuthority }) {
  const hasAuthoritativeTier = KNOWN_AUTHORITATIVE_TIERS.includes(authoritativeTier);
  const level = sr && sr.overall ? sr.overall.level : null;

  let valueEl = null;
  let chipEl = null;
  let evidenceEl = null;

  if (tierAuthority && typeof tierAuthority.fused_score === 'number') {
    const band = hasAuthoritativeTier ? TIER_TO_BAND[authoritativeTier] : null;
    valueEl = <strong className="merged-verdict-value">{Math.round(tierAuthority.fused_score)}%</strong>;
    if (band) {
      chipEl = (
        <span className={`merged-verdict-chip is-${authoritativeTier}`}>
          {t(`report.authorshipBreakdown.fusedHeadline.bands.${band}`)}
        </span>
      );
    }
    if (typeof tierAuthority.composite_score === 'number') {
      evidenceEl = (
        <p className="merged-verdict-evidence">
          {t('report.authorshipBreakdown.fusedHeadline.evidence', {
            composite: Math.round(tierAuthority.composite_score),
            deepScan: Math.round((tierAuthority.proportion || 0) * 100),
          })}
        </p>
      );
    }
  } else if (breakdown && breakdown.deep_scan && KNOWN_DEEP_SCAN_BANDS.includes(breakdown.deep_scan.band)) {
    const ds = breakdown.deep_scan;
    const insufficient = ds.band === 'insufficient';
    const hasProp = typeof ds.proportion === 'number' && Number.isFinite(ds.proportion);
    if (insufficient || !hasProp) {
      chipEl = <span className="merged-verdict-chip is-insufficient">{t('report.authorshipBreakdown.deepScan.insufficientChip')}</span>;
    } else {
      valueEl = <strong className="merged-verdict-value">{Math.round(ds.proportion * 100)}%</strong>;
      const chipTier = hasAuthoritativeTier ? authoritativeTier : ds.band;
      chipEl = (
        <span className={`merged-verdict-chip is-${chipTier}`}>
          {hasAuthoritativeTier
            ? t('report.authorshipBreakdown.deepScan.bandDefersToTier', {
                band: t(`report.authorshipBreakdown.deepScan.bands.${ds.band}`),
                tier: t(`report.tiers.${authoritativeTier}`),
              })
            : t(`report.authorshipBreakdown.deepScan.bands.${ds.band}`)}
        </span>
      );
    }
    evidenceEl = <p className="merged-verdict-evidence">{t('report.authorshipBreakdown.deepScan.notTurnitin')}</p>;
  }

  const ownershipLead = level
    ? t(`report.submissionRisk.ownershipLead.${level}`, { defaultValue: '' })
    : '';
  const textPattern = (sr && sr.axes && sr.axes.text_pattern) || {};
  const hasScore = typeof textPattern.display_score === 'number';
  const flagNote = hasScore
    ? t(
        sr._fused && sr._flagLine != null
          ? 'report.submissionRisk.compactNoteFusedAnchored'
          : sr._fused
            ? 'report.submissionRisk.compactNoteFused'
            : 'report.submissionRisk.compactNote',
        { score: Math.round(textPattern.display_score), flagLine: sr._flagLine },
      )
    : (sr ? t('report.submissionRisk.note') : '');

  return (
    <div className={`merged-verdict${level ? ` is-${level}` : ''}`}>
      <div className="merged-verdict-headline">
        <span className="merged-verdict-kicker">{t('report.authorshipBreakdown.fusedHeadline.label')}</span>
        {valueEl}
        {chipEl}
      </div>
      {ownershipLead && <p className="merged-verdict-lead">{ownershipLead}</p>}
      {evidenceEl}
      {flagNote && <p className="merged-verdict-note">{flagNote}</p>}
    </div>
  );
}

export default function MergedAuthorshipRisk({ t, breakdown, sr, authoritativeTier, tierAuthority }) {
  const hasBreakdown = !!breakdown;
  const hasSr = !!(sr && sr.overall && sr.overall.level);
  if (!hasBreakdown && !hasSr) return null;

  const rawShares = (breakdown && breakdown.document_breakdown_raw) || {};
  const bandShares = (breakdown && breakdown.document_breakdown_bands) || {};
  const axes = (sr && sr.axes) || {};
  const textPattern = axes.text_pattern || {};
  const hasScore = typeof textPattern.display_score === 'number';

  return (
    <section className="merged-authorship-risk" aria-label={t('report.merged.title')}>
      <div className="merged-head">
        <h3>
          {t('report.merged.title')}
          <span className="merged-beta-chip">{t('report.authorshipBreakdown.betaChip')}</span>
        </h3>
        <p className="merged-subtitle">{t('report.merged.subtitle')}</p>
      </div>

      <VerdictBand
        t={t}
        breakdown={breakdown}
        sr={hasSr ? sr : null}
        authoritativeTier={authoritativeTier}
        tierAuthority={tierAuthority}
      />

      <div className="merged-lenses">
        {hasBreakdown && (
          <div className="merged-lens">
            <p className="merged-lens-head">
              {t('report.merged.compositionLens')}{' '}
              <span className="merged-lens-note">· {t('report.merged.compositionLensNote')}</span>
            </p>
            <div className="merged-comp-bars">
              {CATEGORY_ORDER.map((category) => (
                <CategoryBar key={category} t={t} category={category} raw={rawShares[category]} band={bandShares[category]} />
              ))}
            </div>
          </div>
        )}

        {hasSr && (
          <div className="merged-lens">
            <p className="merged-lens-head">
              {t('report.merged.riskLens')}{' '}
              <span className="merged-lens-note">· {t('report.merged.riskLensNote')}</span>
            </p>
            <div className="merged-risk-axes">
              {SUBMISSION_RISK_AXES.map((key) => {
                const lvl = (axes[key] || {}).level || 'unknown';
                return (
                  <div className={`merged-axis is-${lvl}`} key={key}>
                    <span>{t(`report.submissionRisk.axes.${key}`)}</span>
                    <strong>{t(`report.submissionRisk.levels.${lvl}`)}</strong>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {hasSr && hasScore && (
        <details className="merged-scale" open>
          <summary>{t('report.submissionRisk.scale.toggle')}</summary>
          <div className="merged-scale-content">
            <table className="merged-scale-table">
              <thead>
                <tr>
                  <th>{t('report.submissionRisk.scale.headers.score')}</th>
                  <th>{t('report.submissionRisk.scale.headers.reads')}</th>
                  <th>{t('report.submissionRisk.scale.headers.measured')}</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>0–32</td><td>{t('report.submissionRisk.scale.rows.low.reads')}</td><td>{t('report.submissionRisk.scale.rows.low.measured')}</td></tr>
                <tr><td>32–48</td><td>{t('report.submissionRisk.scale.rows.medium.reads')}</td><td>{t('report.submissionRisk.scale.rows.medium.measured')}</td></tr>
                <tr><td>48–65</td><td>{t('report.submissionRisk.scale.rows.high.reads')}</td><td>{t('report.submissionRisk.scale.rows.high.measured')}</td></tr>
                <tr><td>65+</td><td>{t('report.submissionRisk.scale.rows.critical.reads')}</td><td>{t('report.submissionRisk.scale.rows.critical.measured')}</td></tr>
              </tbody>
            </table>
            <p className="merged-scale-footnote">{t('report.submissionRisk.scale.notTurnitinComparable')}</p>
          </div>
        </details>
      )}

      <p className="merged-disclaimer">
        {(breakdown && breakdown.disclaimer) || t('report.authorshipBreakdown.disclaimer')}
      </p>
      <p className="merged-feedback">
        {t('report.authorshipBreakdown.feedbackPrompt')}{' '}
        <button
          type="button"
          className="merged-feedback-link"
          onClick={() => window.dispatchEvent(new Event('draftproof:open-feedback'))}
        >
          {t('report.authorshipBreakdown.feedbackAction')}
        </button>
      </p>
    </section>
  );
}
```

- [ ] **Step 2: Verify it lints/parses**

Run: `cd draftproof-frontend && npx eslint src/pages/report/MergedAuthorshipRisk.jsx`
Expected: no errors (warnings about unused vars would indicate a bug — fix them).

- [ ] **Step 3: Commit**

```bash
git add draftproof-frontend/src/pages/report/MergedAuthorshipRisk.jsx
git commit -m "feat(report): add MergedAuthorshipRisk unified card component"
```

---

## Task 3: Wire the merged card into ReportHero + remove the two legacy stat chips

**Files:**
- Modify: `draftproof-frontend/src/pages/report/ReportHero.jsx:71-116`
- Modify: `draftproof-frontend/src/pages/Report.jsx:1889-1912`

**Interfaces:**
- Consumes: `MergedAuthorshipRisk` (Task 2), the `report.merged.*` keys (Task 1).
- Produces: a hero that renders one merged card and a stat row containing only Risk Tier.

- [ ] **Step 1: Swap the import and props in ReportHero.jsx**

In `draftproof-frontend/src/pages/report/ReportHero.jsx`, replace the import (line 7):

```jsx
import MergedAuthorshipRisk from './MergedAuthorshipRisk';
```

Add a `mergedCard` prop to the destructured props (near line 30, alongside `afterTitle`):

```jsx
  mergedCard = null,
```

- [ ] **Step 2: Remove the Writing Score + Findings stat chips**

In `ReportHero.jsx`, delete the two stat blocks at lines 77-84 (Writing Score + Findings), leaving only the Risk Tier stat. The stat row becomes:

```jsx
            <div className="report-hero-stats is-single" aria-label={t('report.overview')}>
              <div className="report-hero-stat">
                <span>{t('report.summary.riskTier')}</span>
                <strong style={{ color: tier.color }}>{t(`report.tiers.${report.tier}`, { defaultValue: tier.label })}</strong>
              </div>
            </div>
```

(The `is-single` class is styled in Task 4. `writingScore` and `issuesCount` props are now unused inside ReportHero — remove them from the destructure to avoid lint warnings, but keep `issuesCount` in `Report.jsx` where the findings list body still needs it.)

- [ ] **Step 3: Replace the two rendered panels with the merged card**

In `ReportHero.jsx`, replace lines 113-115 (the `{afterTitle}` render and the `{submissionRiskView && <SubmissionRiskBand …>}` render) with:

```jsx
        {/* V7 unified card: one AI-likelihood headline + composition/risk lenses + scale. */}
        {mergedCard}
```

Also remove the now-unused `import SubmissionRiskBand from './SubmissionRiskBand';` (line 7 original) and the `submissionRiskView` / `afterTitle` props from the destructure.

- [ ] **Step 4: Pass the merged card from Report.jsx**

In `draftproof-frontend/src/pages/Report.jsx`, at the `<ReportHero … />` call (line 1889), remove the `afterTitle={…}` and `submissionRiskView={…}` props and add:

```jsx
          mergedCard={
            <MergedAuthorshipRisk
              t={t}
              breakdown={(badge && badge.authorship_breakdown) || null}
              sr={heroSubmissionRisk}
              authoritativeTier={badge.tier || report.tier}
              tierAuthority={(badge && badge.tier_authority) || null}
            />
          }
```

Add the import near the other report-hero imports at the top of `Report.jsx`:

```jsx
import MergedAuthorshipRisk from './report/MergedAuthorshipRisk';
```

Remove the now-unused `import AuthorshipClarityBreakdown …` import. Keep `issuesCount={heroFindingsCount}`/`writingScore={heroWritingScore}` OUT of the ReportHero call only if ReportHero no longer accepts them; since Step 2 removed them from ReportHero's destructure, delete both props from the call site too. (`heroFindingsCount` and `issues.length` remain used elsewhere for the findings body — do not delete those derivations.)

- [ ] **Step 5: Verify lint + build**

Run: `cd draftproof-frontend && npx eslint src/pages/report/ReportHero.jsx src/pages/Report.jsx && npm run build`
Expected: eslint clean (no unused-var errors for `writingScore`/`submissionRiskView`/`AuthorshipClarityBreakdown`/`SubmissionRiskBand`); `npm run build` exits 0.

- [ ] **Step 6: Commit**

```bash
git add draftproof-frontend/src/pages/report/ReportHero.jsx draftproof-frontend/src/pages/Report.jsx
git commit -m "feat(report): render merged card in hero; drop Writing Score + Findings chips"
```

---

## Task 4: Style the merged card + collapse the stat row

**Files:**
- Modify: `draftproof-frontend/src/styles/site-master/06-report-overview.css`

**Interfaces:**
- Consumes: class names emitted by Task 2 (`.merged-*`) and Task 3 (`.report-hero-stats.is-single`).

- [ ] **Step 1: Append the merged-card stylesheet**

Add to the end of `draftproof-frontend/src/styles/site-master/06-report-overview.css`. Use the existing report color variables where the file already defines them; the bar-fill colors are inline (Task 2), so CSS only handles layout + level tints:

```css
/* ── Merged authorship + submission-risk card ─────────────────────── */
.merged-authorship-risk {
  border: 1px solid var(--report-border, #e5e7eb);
  border-radius: 12px;
  padding: 20px 24px;
  background: var(--report-card-bg, #fff);
  margin: 16px 0 24px;
}
.merged-head h3 { display: flex; align-items: center; gap: 8px; font-size: 16px; font-weight: 600; margin: 0 0 4px; }
.merged-beta-chip { font-size: 11px; letter-spacing: .04em; padding: 2px 7px; border-radius: 20px; background: #eef2ff; color: #4338ca; text-transform: uppercase; }
.merged-subtitle { font-size: 13px; color: #6b7280; margin: 0 0 16px; line-height: 1.5; }

.merged-verdict { background: #f9fafb; border-radius: 10px; padding: 14px 16px; margin-bottom: 18px; }
.merged-verdict-headline { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.merged-verdict-kicker { font-size: 13px; color: #6b7280; }
.merged-verdict-value { font-size: 32px; font-weight: 600; line-height: 1; }
.merged-verdict-chip { font-size: 11px; font-weight: 600; padding: 2px 9px; border-radius: 20px; margin-left: auto; }
.merged-verdict-chip.is-green { background: #dcfce7; color: #15803d; }
.merged-verdict-chip.is-amber { background: #fef3c7; color: #b45309; }
.merged-verdict-chip.is-orange { background: #ffedd5; color: #c2410c; }
.merged-verdict-chip.is-red { background: #fee2e2; color: #b91c1c; }
.merged-verdict-chip.is-insufficient { background: #f3f4f6; color: #6b7280; }
.merged-verdict-lead { font-size: 14px; color: #111827; margin: 8px 0 0; }
.merged-verdict-evidence, .merged-verdict-note { font-size: 12px; color: #6b7280; margin: 6px 0 0; line-height: 1.5; }

.merged-lenses { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 18px; }
@media (max-width: 640px) { .merged-lenses { grid-template-columns: 1fr; } }
.merged-lens-head { font-size: 12px; font-weight: 600; color: #374151; margin: 0 0 10px; }
.merged-lens-note { color: #9ca3af; font-weight: 400; }

.merged-comp-bars { display: flex; flex-direction: column; gap: 9px; }
.merged-comp-row-head { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px; }
.merged-comp-band { color: #9ca3af; }
.merged-comp-track { height: 6px; background: #f3f4f6; border-radius: 3px; overflow: hidden; }
.merged-comp-fill { height: 100%; border-radius: 3px; }

.merged-risk-axes { display: flex; flex-direction: column; gap: 6px; }
.merged-axis { display: flex; justify-content: space-between; align-items: center; font-size: 12px; padding: 5px 9px; border-radius: 6px; background: #f3f4f6; color: #6b7280; }
.merged-axis.is-low { background: #dcfce7; color: #15803d; }
.merged-axis.is-medium { background: #fef3c7; color: #b45309; }
.merged-axis.is-high, .merged-axis.is-critical { background: #fee2e2; color: #b91c1c; }

.merged-scale { border-top: 1px solid #f3f4f6; padding-top: 10px; margin-bottom: 4px; }
.merged-scale > summary { font-size: 12px; color: #6b7280; cursor: pointer; }
.merged-scale-table { width: 100%; font-size: 12px; margin-top: 10px; border-collapse: collapse; }
.merged-scale-table th { color: #9ca3af; text-align: left; font-weight: 400; padding: 3px 0; }
.merged-scale-table td { padding: 3px 0; }
.merged-scale-footnote { font-size: 11px; color: #9ca3af; margin: 8px 0 0; }

.merged-disclaimer { font-size: 11px; color: #9ca3af; line-height: 1.5; margin: 12px 0 0; }
.merged-feedback { font-size: 12px; color: #6b7280; margin: 6px 0 0; }
.merged-feedback-link { background: none; border: none; color: #4338ca; cursor: pointer; padding: 0; font: inherit; text-decoration: underline; }

/* Stat row collapses to a single Risk Tier stat once Writing Score + Findings are removed. */
.report-hero-stats.is-single { grid-template-columns: 1fr; max-width: 220px; }
```

Note: if `06-report-overview.css` defines report color variables (`--report-border`, etc.), prefer those; the hex fallbacks above are safe defaults matching the existing report palette. Check the top of the file for existing variables and reuse them.

- [ ] **Step 2: Verify build picks up the CSS**

Run: `cd draftproof-frontend && npm run build`
Expected: exits 0, no CSS errors.

- [ ] **Step 3: Commit**

```bash
git add draftproof-frontend/src/styles/site-master/06-report-overview.css
git commit -m "style(report): merged card layout + single-stat hero row"
```

---

## Task 5: Delete the two superseded components

**Files:**
- Delete: `draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx`
- Delete: `draftproof-frontend/src/pages/report/SubmissionRiskBand.jsx`

- [ ] **Step 1: Confirm no remaining importers**

Run: `cd draftproof-frontend && grep -rn "AuthorshipClarityBreakdown\|SubmissionRiskBand" src/`
Expected: **no matches** (Tasks 3 removed both imports). If any match remains, fix it before deleting.

- [ ] **Step 2: Delete the files**

```bash
git rm draftproof-frontend/src/pages/report/AuthorshipClarityBreakdown.jsx draftproof-frontend/src/pages/report/SubmissionRiskBand.jsx
```

- [ ] **Step 3: Verify build still green**

Run: `cd draftproof-frontend && npm run build`
Expected: exits 0.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(report): remove superseded AuthorshipClarityBreakdown + SubmissionRiskBand"
```

---

## Task 6: PDF — co-locate authorship bars + de-dupe AI-likelihood

**Files:**
- Modify: `poc/report/render_panels.py` (add `render_merged_authorship_risk`; refactor shared pieces from `render_authorship_breakdown` + `render_scan_lead`)
- Modify: `poc/report/render.py` (`render_report()` integration — swap the two calls for one)
- Modify: `poc/report/pdf.py` (color-coded `.dp-abd-bar-fill`)
- Modify: `poc/test_render_authorship_breakdown.py` (add merged-render test)

**Interfaces:**
- Consumes: existing `render_authorship_breakdown(report_data)` (returns the bars + fused headline HTML) and `render_scan_lead(report, data)` (returns the richer submission section).
- Produces: `render_merged_authorship_risk(report, data) -> str` — the authorship bars + single AI-likelihood headline FIRST, then the richer submission sections, with the AI number surfaced once.

- [ ] **Step 1: Write the failing test**

Add to `poc/test_render_authorship_breakdown.py`:

```python
def test_merged_render_shows_ai_likelihood_once():
    from poc.report.render_panels import render_merged_authorship_risk
    # Minimal report with both an authorship breakdown (fused) and a submission-risk section.
    report = _make_report_with_breakdown_and_sr()  # helper below
    data = {"document_context": {"word_count": 400}, "overall_tier_reason": ""}
    html = render_merged_authorship_risk(report, data)
    # The composition bars are present (authorship co-located).
    assert "dp-abd-bars" in html
    # The richer submission section is preserved.
    assert "1. Submission and policy view" in html
    # The AI-likelihood % appears exactly once (de-duped): count the fused headline phrase.
    assert html.count("DraftProof AI-likelihood") == 1


def _make_report_with_breakdown_and_sr():
    from types import SimpleNamespace
    badge = {
        "tier": "green",
        "authorship_breakdown": {
            "document_breakdown_raw": {"student_owned": 0.34, "ai_assisted_polished": 0.23, "ai_paraphrased": 0.14, "ai_generated_like": 0.29},
            "document_breakdown_bands": {"student_owned": "Some", "ai_assisted_polished": "Little", "ai_paraphrased": "Little", "ai_generated_like": "Some"},
            "disclaimer": "DraftProof provides authorship clarity signals.",
        },
        "tier_authority": {"fused_score": 24, "composite_score": 14, "proportion": 0.31},
        "submission_risk": {"overall": {"level": "low"}, "axes": {"text_pattern": {"level": "low", "display_score": 24}}},
        "ai_likelihood_score": 24,
    }
    return SimpleNamespace(ai_risk_badge=badge)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd poc && python -m pytest test_render_authorship_breakdown.py::test_merged_render_shows_ai_likelihood_once -v`
Expected: FAIL — `ImportError: cannot import name 'render_merged_authorship_risk'`.

- [ ] **Step 3: Refactor `render_authorship_breakdown` into a body-only helper**

In `poc/report/render_panels.py`, extract the AI-likelihood headline emit (lines ~199-239) into a helper so both the standalone and merged paths can control whether it renders, and the bars (lines ~241-264) into a helper:

```python
def _authorship_headline_html(badge: dict, breakdown: dict) -> str:
    """The single AI-likelihood headline (fused, else deep-scan). Returns '' if neither."""
    tier_authority = badge.get("tier_authority")
    authoritative_tier = badge.get("tier")
    if isinstance(tier_authority, dict) and isinstance(tier_authority.get("fused_score"), (int, float)):
        fused_score = tier_authority["fused_score"]
        composite = tier_authority.get("composite_score")
        deep_scan_pct = round((tier_authority.get("proportion") or 0) * 100)
        band_chip = ""
        if authoritative_tier in _ACB_TIER_TO_BAND:
            band = _ACB_TIER_TO_BAND[authoritative_tier]
            band_chip = _statchip(_ACB_FUSED_BAND_LABELS.get(band, band), _level_kind(band))
        evidence = ""
        if isinstance(composite, (int, float)):
            evidence = (f"Behind this score: composite detector {round(composite)}%, "
                        f"deep-scan detector {deep_scan_pct}% (sentence-level). See the DraftProof scale above.")
        return ('<div class="dp-hero"><p class="dp-hero-read">DraftProof AI-likelihood '
                f'<b>{round(fused_score)}%</b> {band_chip}</p>'
                + (f'<p class="dp-hero-sub">{escape(evidence)}</p>' if evidence else "") + "</div>")
    deep_scan = breakdown.get("deep_scan") or {}
    band = deep_scan.get("band")
    if band and band in ("insufficient", "amber", "orange", "red"):
        proportion = deep_scan.get("proportion")
        if band == "insufficient" or not isinstance(proportion, (int, float)):
            return ('<div class="dp-hero"><p class="dp-hero-read">Deep-scan AI estimate '
                    f'{_statchip("insufficient evidence", "info")}</p>'
                    '<p class="dp-hero-sub">sentence-level signal — not a Turnitin score</p></div>')
        band_label = _ACB_DEEP_SCAN_BAND_LABELS.get(band, band)
        return ('<div class="dp-hero"><p class="dp-hero-read">Deep-scan AI estimate '
                f'<b>{round(proportion * 100)}%</b> {_statchip(band_label, _level_kind(band))}</p>'
                '<p class="dp-hero-sub">sentence-level signal — not a Turnitin score</p></div>')
    return ""


def _authorship_bars_html(breakdown: dict) -> str:
    """The 4 color-coded category bars + disclaimer. '' when no breakdown."""
    raw_shares = breakdown.get("document_breakdown_raw") or {}
    band_shares = breakdown.get("document_breakdown_bands") or {}
    rows = []
    for category in _ACB_CATEGORY_ORDER:
        raw = raw_shares.get(category)
        band = band_shares.get(category)
        has_raw = isinstance(raw, (int, float))
        width_pct = max(0.0, min(100.0, raw * 100)) if has_raw else 0.0
        band_label = _ACB_BAND_LABELS.get(band, band) if band else _ACB_BAND_LABELS["None"]
        text = f"{band_label} · {round(width_pct)}%" if has_raw else band_label
        rows.append('<div class="dp-abd-row">'
                    f'<span class="dp-abd-label">{escape(_ACB_CATEGORY_LABELS.get(category, category))}</span>'
                    f'<span class="dp-abd-bar-track"><span class="dp-abd-bar-fill dp-abd-fill--{category}" style="width:{width_pct}%"></span></span>'
                    f'<span class="dp-abd-band">{escape(text)}</span></div>')
    out = '<div class="dp-abd-bars">' + "".join(rows) + "</div>"
    disclaimer = breakdown.get("disclaimer")
    if disclaimer:
        out += f'<p class="dp-hero-sub">{escape(str(disclaimer))}</p>'
    return out
```

Then rewrite the existing `render_authorship_breakdown` body to use these two helpers (keeps its standalone behavior identical — this preserves the existing passing tests):

```python
def render_authorship_breakdown(report_data: dict) -> str:
    badge = (report_data or {}).get("ai_risk_badge") or {}
    breakdown = badge.get("authorship_breakdown")
    if not breakdown:
        return ""
    out = ['<div class="authorship-breakdown">',
           '<p class="dp-callout-title">Authorship clarity breakdown <span class="dp-statchip dp-statchip--info">Beta</span></p>',
           '<p class="dp-hero-sub">How this document\'s writing signals distribute across four '
           "authorship styles. The shares always add up to 100% — a composition of the mix, not an "
           "AI-probability. The deep-scan estimate below comes from a separate beta detector and may "
           "differ from Text-pattern risk in the summary above — different models, and both are "
           "signals rather than verdicts.</p>",
           _authorship_headline_html(badge, breakdown),
           _authorship_bars_html(breakdown),
           "</div>"]
    return "\n".join(p for p in out if p)
```

- [ ] **Step 4: Add `render_merged_authorship_risk` + de-dupe the number in the submission lead**

Add the merged function. It leads with the authorship bars + the ONE headline, then appends `render_scan_lead`'s richer output — but with the redundant AI-likelihood surfacing suppressed there. Add a keyword to `render_scan_lead` to suppress its "Text-pattern trigger" KPI + the axis-table "~X%" note (both duplicate the headline):

In `render_scan_lead(report, data)`, change the signature to `render_scan_lead(report, data, *, suppress_ai_likelihood: bool = False)`. Guard the two duplicate surfacings:
- At the KPI append (line ~365-366), wrap in `if not suppress_ai_likelihood and isinstance(dp_band.get("score"), (int, float)):`.
- At the axis-table note (lines ~441-450), wrap the `current += f" — DraftProof AI-likelihood ~{round(ai_pct)}%"` in `if not suppress_ai_likelihood:`.

Then:

```python
def render_merged_authorship_risk(report, data) -> str:
    """Unified PDF block: authorship composition bars + ONE AI-likelihood headline
    co-located at the top, then the richer submission-risk section (policy cards,
    priority fixes, axis table, verdict) with the AI number de-duped. Falls back
    gracefully: if there's no authorship breakdown, this is just render_scan_lead;
    if there's no submission-risk section, it's just the authorship panel."""
    badge = getattr(report, "ai_risk_badge", None) or {}
    breakdown = badge.get("authorship_breakdown")
    lead = render_scan_lead(report, data, suppress_ai_likelihood=bool(breakdown))
    if not breakdown:
        return lead
    header = ('<div class="authorship-breakdown">'
              '<p class="dp-callout-title">Authorship &amp; submission risk '
              '<span class="dp-statchip dp-statchip--info">Beta</span></p>'
              + _authorship_headline_html(badge, breakdown)
              + _authorship_bars_html(breakdown)
              + "</div>")
    if not lead:
        return header
    return header + "\n" + lead
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd poc && python -m pytest test_render_authorship_breakdown.py -v`
Expected: the new test PASSES and all pre-existing tests in the file still PASS (the refactor preserved `render_authorship_breakdown` output).

- [ ] **Step 6: Wire the merged call into `render.py::render_report()`**

In `poc/report/render.py::render_report()`, find the two calls (`render_scan_lead(report, data)` at ~line 1663 and `render_authorship_breakdown(...)` at ~line 1687). Replace them with a single call to the merged renderer at the `render_scan_lead` position, and delete the separate `render_authorship_breakdown` emit:

```python
    from .render_panels import render_merged_authorship_risk
    scan_lead = render_merged_authorship_risk(report, data)
    if scan_lead:
        # (existing handling that consumed render_scan_lead's return — unchanged)
```

Preserve whatever fallback the existing code used when `render_scan_lead` returned "" (the merged renderer returns the authorship panel alone in that case, or "" only when both are absent — matching the additive contract). Remove the now-orphaned standalone `render_authorship_breakdown` call block.

- [ ] **Step 7: Add color-coded bar CSS to pdf.py**

In `poc/report/pdf.py`, find the `.dp-abd-bar-fill` rule (~line 375-417) and add per-category colors:

```css
.dp-abd-fill--student_owned { background: #1D9E75; }
.dp-abd-fill--ai_assisted_polished { background: #888780; }
.dp-abd-fill--ai_paraphrased { background: #D85A30; }
.dp-abd-fill--ai_generated_like { background: #D85A30; }
```

(Keep the existing `.dp-abd-bar-fill` base rule for track/height; the `dp-abd-fill--<category>` class from Step 3 sets the color.)

- [ ] **Step 8: Run the full PDF render test suite**

Run: `cd poc && python -m pytest test_render_authorship_breakdown.py test_report_ai_likelihood_headline.py -v`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add poc/report/render_panels.py poc/report/render.py poc/report/pdf.py poc/test_render_authorship_breakdown.py
git commit -m "feat(report-pdf): co-locate authorship bars + de-dupe AI-likelihood in one block"
```

---

## Task 7: Verify rendered artifacts (page + PDF)

**Files:** none (verification only).

Per the house rule "Verify rendered artifacts": grep proves emit-in-code, not render-correct. Actually render both surfaces and look.

- [ ] **Step 1: Render the PDF from a sample report and inspect**

Run the existing report-render smoke path (use whatever fixture the repo uses — e.g. a saved badge JSON) to produce a PDF, then open it. Confirm visually:
- One "Authorship & submission risk" block with the composition bars (color-coded) at the top.
- The AI-likelihood % appears exactly once.
- Policy cards, priority fixes, axis table, and the plain-English verdict are all still present below.

Run: `cd poc && python -m pytest test_render_authorship_breakdown.py -v` (regression) and generate a sample PDF via the repo's normal render entry (document the exact command used in the commit/PR notes).

- [ ] **Step 2: Preview the web page**

Start the frontend (`cd draftproof-frontend && npm run dev`) and open a completed report. Confirm visually:
- One merged card (not two stacked panels).
- Verdict band shows one AI-likelihood % + tier chip + "You can defend this as your own work".
- Two lenses side by side with the "sums to 100%" / "independent" headers; bars color-coded.
- Scale expanded by default.
- Hero stat row shows ONLY Risk Tier (Writing Score + Findings gone).
- Older report (no `tier_authority` / no `submission_risk`) still renders without crashing (deep-scan fallback headline; missing lens omitted).

- [ ] **Step 3: Confirm no dangling references**

Run: `cd draftproof-frontend && grep -rn "AuthorshipClarityBreakdown\|SubmissionRiskBand" src/ ; grep -rn "writingScore\|issuesCount" src/pages/report/ReportHero.jsx`
Expected: no component matches; ReportHero no longer references the removed stat props.

- [ ] **Step 4: Final build + commit any fixes**

Run: `cd draftproof-frontend && npm run build`
Expected: exits 0. Commit any visual-fix tweaks discovered during Steps 1-2.

---

## Notes for the executor

- The web hero's `reportSummaryBar` (Report.jsx:1353) is dead code (defined, never rendered) — leave it alone; it's out of scope.
- Do NOT remove `writing_quality_score` or the findings array from the badge JSON — other surfaces still read them.
- If `06-report-overview.css` defines report color variables at its top, reuse them instead of the hex fallbacks in Task 4.
- The PDF's `render_scan_lead` output is partly markdown (section headers, axis table) that a downstream markdown→HTML step renders — keep the merged header as HTML (matching `render_authorship_breakdown`'s existing HTML style) and let the appended `render_scan_lead` markdown flow as before.
