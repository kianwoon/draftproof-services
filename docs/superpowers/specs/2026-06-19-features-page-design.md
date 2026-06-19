# Features Page — Design Spec
**Date:** 2026-06-19
**Status:** Approved

## Goal
Add a "Features" entry to the public marketing nav that links to a new `/features` page. The page serves as a shareable sales/pitch asset showing what DraftProof does vs. competitors.

## Route
- `/features` (en)
- `/zh/features` (zh)

## Nav placement
Add `features` to `marketingLinks` in `Header.jsx`, between `/#product` ("Why") and `/content-checker` ("Content checker"). Label: `t('nav.features')`.

## Page structure (`Features.jsx`)

### 1. Hero section
- Eyebrow: `t('featuresPage.eyebrow')` — "Why DraftProof"
- H1: `t('featuresPage.title')` — "Detect. Understand. Improve."
- Lead: `t('featuresPage.lead')` — "Every other detector tells you that you failed. DraftProof shows you how to pass — by teaching you to write better."
- Dark background using existing `app-hero app-hero-dark` + `CodeTexture` component (matches Why page pattern)

### 2. Comparison table
Section label: "How we compare"

Columns: DraftProof | GPTZero | Turnitin | Originality.ai | Winston AI

Rows (all sourced from `t('featuresPage.rows', { returnObjects: true })`):
| Feature | DP | GPTZero | Turnitin | Originality | Winston |
|---|---|---|---|---|---|
| Paragraph-level output | ✓ | ✓ | ✓ | ✓ | ✗ |
| Explains *why* content is flagged | ✓ | ✗ | ✗ | ✗ | ✗ |
| Integrated rewrite / coaching | ✓ | ✗ | ✗ | ✗ | ✗ |
| Before/after diff view | ✓ | ✗ | ✗ | ✗ | ✗ |
| Policy-aware scoring | ✓ | ✗ | ✗ | ✗ | ✗ |
| Submission risk framing | ✓ | ✗ | ✗ | ✗ | ✗ |
| Critical thinking assessment | ✓ | ✗ | ✗ | ✗ | ✗ |
| Honest about detector limits | ✓ | ✗ | ✗ | ✗ | ✗ |
| Individual access (no institution needed) | ✓ | ✓ | ✗ | ✓ | ✓ |

DraftProof column is visually accented (info background tint) to make the column stand out.

Cell values are one of three states: `yes` (✓ green), `no` (✗ muted red), `partial` (amber label). Stored as `"yes" | "no" | "partial"` in i18n data; component renders icons.

### 3. DraftProof-only feature cards
Section label: "DraftProof-only features"

4 cards in a 2-column grid (stacks to 1-col on mobile), sourced from `t('featuresPage.cards', { returnObjects: true })`:
1. **Grounded rewrite coaching** — ti-writing icon
2. **Policy risk (dual-mode)** — ti-shield-check icon
3. **Submission risk framing** — ti-clipboard-check icon
4. **Critical thinking control** — ti-brain icon

Each card: icon + title + 2-sentence description.

### 4. CTA strip (bottom)
Simple centred strip: headline + "Start your first scan" button linking to `/signin`. Reuse existing `.cta-strip` pattern from Why page.

## i18n files

### New files
- `src/i18n/en/features.js` — full EN content (eyebrow, title, lead, rows array, cards array, cta)
- `src/i18n/zh/features.js` — ZH translations of same keys

### Modified files
- `src/i18n/en/nav.js` — add `"features": "Features"`
- `src/i18n/zh/nav.js` — add `"features": "功能对比"`
- `src/i18n/en.js` — import + register `featuresPage`
- `src/i18n/zh.js` — import + register `featuresPage`

## Routing (`App.jsx`)
```jsx
import Features from './pages/Features';
// ...
<Route path="/features" element={<Features />} />
<Route path="/zh/features" element={<Features />} />
```

## SEO (`seoMetadata.js`)
Add `/features` entry:
```js
'/features': {
  titleKey: 'seo.featuresTitle',
  descriptionKey: 'seo.featuresDescription',
  canonical: '/features',
  schemaType: 'WebPage',
  freshness: { type: 'reviewed', date: '2026-06-19' },
},
```
Add `seo.featuresTitle` + `seo.featuresDescription` keys to `src/i18n/en/seo.js` and `zh/seo.js`.

## Files to create
- `src/pages/Features.jsx`
- `src/i18n/en/features.js`
- `src/i18n/zh/features.js`

## Files to modify
- `src/App.jsx` — 2 new routes + 1 import
- `src/components/Header.jsx` — 1 new entry in marketingLinks
- `src/i18n/en/nav.js` — 1 new key
- `src/i18n/zh/nav.js` — 1 new key
- `src/i18n/en.js` — 1 new import + 1 key in enTranslation
- `src/i18n/zh.js` — 1 new import + 1 key in zhTranslation
- `src/i18n/en/seo.js` — 2 new keys
- `src/i18n/zh/seo.js` — 2 new keys
- `src/seoMetadata.js` — 1 new page entry

## Out of scope
- Plagiarism comparison column (Turnitin/Originality bundle it; not a primary DraftProof axis)
- Mobile-specific table treatment beyond CSS stacking
- Animations or scroll effects
