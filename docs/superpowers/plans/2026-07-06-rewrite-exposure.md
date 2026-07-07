# Rewrite Exposure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the rewrite function real exposure via a dedicated `/rewrite` marketing page (with a 3-example before/after demo) linked from nav/footer, plus a compact before/after teaser on the landing page that links to it.

**Architecture:** One new shared presentational component (`RewriteBeforeAfter`) renders a before/after card; it is consumed by both a new public marketing page (`RewriteOverview.jsx`, following the `Why.jsx`/`Technology.jsx` pattern) and a new compact teaser section on `Landing.jsx`. Wiring (route/nav/footer/locale/SEO) mirrors the already-shipped `/technology` page exactly.

**Tech Stack:** React 18 + react-i18next + react-router-dom, Vite. No test framework exists for this frontend — verification is `npm run build:client` plus a preview-tool visual check.

## Global Constraints

- Alignment (hard, from `docs/draftproof_alignment_principles.md`): all copy stays "teaching draft / grounded writing / finish in your own words / complement Turnitin / prepare before submission." NEVER "make it undetectable / humanizer / bypass / beat the detector." The bracketed "Suggested Addition For Review — you replace this" mechanic is foregrounded to prove the rewrite is not a make-it-pass button. The CTA is honest that rewrite starts from a scan.
- No literal vendor/model names and no raw dataset numbers in new copy — capability-level only.
- Every new i18n key must exist in BOTH the `en` and `zh` file it belongs to (documented "locale trap" failure mode: a missing zh key silently falls back to English).
- zh copy uses full-width punctuation (，／：／。) consistent with the rest of the zh i18n files (lesson from commit 16425b6c).
- Reuse existing CSS classes/tokens; the ONE new CSS block permitted is the `.rewrite-ba*` before/after card (defined once in Task 1, reused on both surfaces) using only existing color tokens/values — no new design system.
- `/rewrite` (public marketing) must NOT shadow or alter the existing protected `/rewrite/:rewriteId` viewer route (`App.jsx:159`) or the existing `Rewrite.jsx` component.
- Do NOT touch `Footer.jsx:22`'s pre-existing dead `/#engine` link — it's owned by a separate in-flight task.
- Line numbers below are current at plan-writing time; if an earlier task shifts a file, find the "Currently: `<exact code>`" block by content, not line number.

---

### Task 1: Shared before/after card + `/rewrite` page content (unrouted)

**Files:**
- Create: `draftproof-frontend/src/components/RewriteBeforeAfter.jsx`
- Create: `draftproof-frontend/src/pages/RewriteOverview.jsx`
- Create: `draftproof-frontend/src/i18n/en/rewriteOverview.js`
- Create: `draftproof-frontend/src/i18n/zh/rewriteOverview.js`
- Modify: `draftproof-frontend/src/i18n/en.js` (register namespace)
- Modify: `draftproof-frontend/src/i18n/zh.js` (register namespace)
- Modify: `draftproof-frontend/src/styles/site-master/03-landing-sections.css` (append `.rewrite-ba*` block)

**Interfaces:**
- Consumes: existing cross-namespace i18n `t('rewriteFraming.title'|'.isCopy'|'.isntCopy'|'.action')` and `t('featuresPage.rewriteCards', { returnObjects: true })` (array of `{icon,title,body}`).
- Produces:
  - `export default function RewriteOverview()` — default export, no props (Task 2 imports `import RewriteOverview from './pages/RewriteOverview';`).
  - `export function RewriteBeforeAfter({ before, after, marker, beforeLabel, afterLabel })` — named export from `components/RewriteBeforeAfter.jsx` (Task 3 imports it). All props are strings; `marker` optional.

- [ ] **Step 1: Create the shared `RewriteBeforeAfter` component**

`draftproof-frontend/src/components/RewriteBeforeAfter.jsx`:
```jsx
// Presentational before/after card for the rewrite marketing surfaces (the /rewrite
// page and the landing teaser). Pure display — takes strings, renders a two-row diff
// with an optional "suggested addition you replace" marker. No i18n inside: callers
// pass already-translated strings so this stays a dumb, reusable unit.
export function RewriteBeforeAfter({ before, after, marker, beforeLabel, afterLabel }) {
  return (
    <div className="rewrite-ba">
      <div className="rewrite-ba-row rewrite-ba-before">
        <span className="rewrite-ba-label">{beforeLabel}</span>
        {before}
      </div>
      <div className="rewrite-ba-row rewrite-ba-after">
        <span className="rewrite-ba-label">{afterLabel}</span>
        {after}
        {marker && <span className="rewrite-ba-marker">{marker}</span>}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Append the `.rewrite-ba*` CSS block to `03-landing-sections.css`**

Add at the end of `draftproof-frontend/src/styles/site-master/03-landing-sections.css` (values reuse existing palette: the red/green already used by `.issue-chip-tier` at `07-report-submitted.css:1926-1928`, and tokens `--radius-md`/`--green-700`/`--muted`):
```css
.rewrite-ba {
  border: 0.5px solid rgba(13, 27, 42, .12);
  border-radius: var(--radius-md);
  overflow: hidden;
}
.rewrite-ba-row {
  padding: 0.85rem 1rem;
  font-size: 0.92rem;
  line-height: 1.55;
  color: var(--ink);
}
.rewrite-ba-row + .rewrite-ba-row {
  border-top: 0.5px solid rgba(13, 27, 42, .10);
}
.rewrite-ba-label {
  display: block;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: .06em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
}
.rewrite-ba-before { background: rgba(226, 75, 74, .05); }
.rewrite-ba-before .rewrite-ba-label { color: #b91c1c; }
.rewrite-ba-after { background: rgba(59, 168, 118, .06); }
.rewrite-ba-after .rewrite-ba-label { color: var(--green-700); }
.rewrite-ba-marker {
  display: block;
  margin-top: 0.45rem;
  font-size: 0.8rem;
  color: var(--muted);
  font-style: italic;
}
```

- [ ] **Step 3: Create `i18n/en/rewriteOverview.js`**

```js
export const rewriteOverview = {
  "eyebrow": "The DraftProof Rewrite",
  "title": "See what grounded writing looks like — then make it yours.",
  "lead": "When a scan flags a passage as thin or unsupported, DraftProof generates a worked before/after — the citation, example, or reasoning it's missing — so you can see the fix, then finish it in your own words. It's a teaching draft, not a submit-ready answer.",
  "demoHeading": "Three ways the rewrite grounds a weak passage",
  "beforeLabel": "Before",
  "afterLabel": "After",
  // allow-hardcode: static marketing sample copy (illustrative before/after built on the
  // fixed landing sample essay), never compared against user content, not a scoring oracle.
  "examples": [
    {
      "fixType": "Unsupported claim → add a source",
      "before": "Hollywood has become one of the most powerful cultural exports in history.",
      "after": "Hollywood is among the most widely exported cultural industries [add a citation you can verify — for example an export-revenue or box-office figure].",
      "marker": "Suggested Addition For Review — replace the bracketed note with a real source."
    },
    {
      "fixType": "Vague generality → add a specific example",
      "before": "American movies, music, and social media are consumed globally.",
      "after": "American cultural products reach audiences far beyond the US [add a specific example from your own reading — a film, artist, or platform, and where it's popular].",
      "marker": "Suggested Addition For Review — replace the bracketed note with your own example."
    },
    {
      "fixType": "No reasoning → add your own \"why\"",
      "before": "The United States has a strong cultural influence.",
      "after": "The United States has a strong cultural influence because [add your own reasoning — what makes it \"strong\", and how you weighed it against counter-examples].",
      "marker": "Suggested Addition For Review — replace the bracketed note with your own reasoning."
    }
  ],
  "ctaTitle": "Ready to see it on your own draft?",
  "ctaBody": "Run a scan first — it flags the passages, then the rewrite shows a worked fix for each.",
  "ctaButton": "Start a scan"
};
```

- [ ] **Step 4: Create `i18n/zh/rewriteOverview.js`**

```js
export const rewriteOverview = {
  "eyebrow": "DraftProof 改写",
  "title": "先看清有依据的写作是什么样，再把它变成你自己的。",
  "lead": "当扫描把某个段落标记为单薄或缺乏支撑时，DraftProof 会生成一份可对照的「改写前／改写后」——补上它所缺的引用、事例或推理——让你先看到修改思路，再用自己的话把它写完。这是一份用来学习的草稿，而不是可直接提交的答案。",
  "demoHeading": "改写为薄弱段落补足依据的三种方式",
  "beforeLabel": "改写前",
  "afterLabel": "改写后",
  // allow-hardcode: static marketing sample copy (illustrative before/after built on the
  // fixed landing sample essay), never compared against user content, not a scoring oracle.
  "examples": [
    {
      "fixType": "无支撑的论断 → 补一个来源",
      "before": "好莱坞已成为史上最具影响力的文化输出之一。",
      "after": "好莱坞是全球输出规模最大的文化产业之一〔补一个你能核实的引用——例如出口收入或票房数据〕。",
      "marker": "待复核的补充内容——请用一个真实来源替换括号中的提示。"
    },
    {
      "fixType": "笼统的泛论 → 补一个具体事例",
      "before": "美国的电影、音乐和社交媒体在全球范围内被广泛消费。",
      "after": "美国的文化产品触达了远超本土的受众〔补一个来自你自己阅读的具体事例——某部影片、某位艺人或某个平台，以及它在哪里流行〕。",
      "marker": "待复核的补充内容——请用你自己的事例替换括号中的提示。"
    },
    {
      "fixType": "没有推理 → 补上你自己的「为什么」",
      "before": "美国拥有强大的文化影响力。",
      "after": "美国拥有强大的文化影响力，因为〔补上你自己的推理——是什么让它「强大」，以及你如何权衡了反例〕。",
      "marker": "待复核的补充内容——请用你自己的推理替换括号中的提示。"
    }
  ],
  "ctaTitle": "想在自己的草稿上看看效果？",
  "ctaBody": "先运行一次扫描——它会标记出相关段落，然后改写会为每一处给出一份可对照的修改示范。",
  "ctaButton": "开始扫描"
};
```

- [ ] **Step 5: Register the namespace in `i18n/en.js` and `i18n/zh.js`**

In `draftproof-frontend/src/i18n/en.js`, after the existing `import { technologyPage } from './en/technologyPage.js';` (line 18), add:
```js
import { rewriteOverview } from './en/rewriteOverview.js';
```
and after the existing `"technologyPage": technologyPage,` (line 50) registration, add:
```js
  "rewriteOverview": rewriteOverview,
```
Do the exact same in `draftproof-frontend/src/i18n/zh.js` (import from `./zh/rewriteOverview.js`, register `"rewriteOverview": rewriteOverview,`) at the matching positions next to its `technologyPage` lines.

- [ ] **Step 6: Create the `RewriteOverview` page**

`draftproof-frontend/src/pages/RewriteOverview.jsx`:
```jsx
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { RewriteBeforeAfter } from '../components/RewriteBeforeAfter';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function RewriteOverview() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const examples = t('rewriteOverview.examples', { returnObjects: true });
  const beforeLabel = t('rewriteOverview.beforeLabel');
  const afterLabel = t('rewriteOverview.afterLabel');
  const valueCards = t('featuresPage.rewriteCards', { returnObjects: true });

  return (
    <main className="why-shell">
      <div className="container">
        <section className="why-hero app-hero app-hero-dark">
          <CodeTexture id="rewriteHero" />
          <div>
            <p className="eyebrow">{t('rewriteOverview.eyebrow')}</p>
            <h1>{t('rewriteOverview.title')}</h1>
            <p className="lead">{t('rewriteOverview.lead')}</p>
          </div>
        </section>

        <section className="why-section" aria-label={t('rewriteOverview.demoHeading')}>
          <h2>{t('rewriteOverview.demoHeading')}</h2>
          {(Array.isArray(examples) ? examples : []).map((ex) => (
            <div key={ex.fixType} style={{ marginBottom: '1.25rem' }}>
              <p className="why-highlight">{ex.fixType}</p>
              <RewriteBeforeAfter
                before={ex.before}
                after={ex.after}
                marker={ex.marker}
                beforeLabel={beforeLabel}
                afterLabel={afterLabel}
              />
            </div>
          ))}
        </section>

        <section className="why-section">
          <h2>{t('rewriteFraming.title')}</h2>
          <p>{t('rewriteFraming.isCopy')}</p>
          <p className="why-quote">{t('rewriteFraming.isntCopy')}</p>
          <p className="why-punch">{t('rewriteFraming.action')}</p>
        </section>

        <section className="why-section">
          <div className="why-card-grid">
            {(Array.isArray(valueCards) ? valueCards : []).map((card) => (
              <article className="why-card" key={card.title}>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="why-cta">
          <h2>{t('rewriteOverview.ctaTitle')}</h2>
          <p>{t('rewriteOverview.ctaBody')}</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('rewriteOverview.ctaButton')}</Link>
          </div>
        </section>

        <PageFreshness path="/rewrite" />
      </div>
    </main>
  );
}
```
Note: `.why-card-grid`/`.why-card` are confirmed present in `site-master/03-landing-sections.css` (the same classes the Landing page's `whyCards` grid uses). The demo/value/CTA sections reuse `why-section`/`why-hero`/`why-cta` (defined in `09-why-legal-document-viewer.css`, confirmed light-themed).

- [ ] **Step 7: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors (page not yet routed, but must compile and its i18n must load).

- [ ] **Step 8: Commit**

```bash
git add draftproof-frontend/src/components/RewriteBeforeAfter.jsx draftproof-frontend/src/pages/RewriteOverview.jsx draftproof-frontend/src/i18n/en/rewriteOverview.js draftproof-frontend/src/i18n/zh/rewriteOverview.js draftproof-frontend/src/i18n/en.js draftproof-frontend/src/i18n/zh.js draftproof-frontend/src/styles/site-master/03-landing-sections.css
git commit -m "feat(rewrite): add /rewrite page content + shared before/after card (unrouted)"
```

---

### Task 2: Route, nav, footer, SEO wiring + Features learn-more link

**Files:**
- Modify: `draftproof-frontend/src/App.jsx` (import + route pair)
- Modify: `draftproof-frontend/src/components/Header.jsx` (marketingLinks)
- Modify: `draftproof-frontend/src/components/Footer.jsx` (product-nav link)
- Modify: `draftproof-frontend/src/localeRouting.js` (LOCALIZABLE_PUBLIC_PATHS)
- Modify: `draftproof-frontend/src/seoMetadata.js` (PAGE_META entry)
- Modify: `draftproof-frontend/src/i18n/en/nav.js`, `draftproof-frontend/src/i18n/zh/nav.js` (nav.rewrite)
- Modify: `draftproof-frontend/src/i18n/en/footer.js`, `draftproof-frontend/src/i18n/zh/footer.js` (footer.rewrite)
- Modify: `draftproof-frontend/src/i18n/en/seo.js`, `draftproof-frontend/src/i18n/zh/seo.js` (rewrite SEO keys)
- Modify: `draftproof-frontend/src/pages/Features.jsx` (learn-more link in rewrite tab)
- Modify: `draftproof-frontend/src/i18n/en/features.js`, `draftproof-frontend/src/i18n/zh/features.js` (learn-more label)

**Interfaces:**
- Consumes: `RewriteOverview` default export from Task 1.
- Produces: nothing consumed by Task 3 (Task 3 links to `/rewrite` by path string, not by importing anything here).

- [ ] **Step 1: Add the import and route pair in `App.jsx`**

After the existing `import Technology from './pages/Technology';` (line 22), add:
```js
import RewriteOverview from './pages/RewriteOverview';
```
Then after the existing `<Route path="/zh/technology" element={<Technology />} />` (line 127), add:
```jsx
            <Route path="/rewrite" element={<RewriteOverview />} />
            <Route path="/zh/rewrite" element={<RewriteOverview />} />
```
(The protected `<Route path="/rewrite/:rewriteId" ...>` at line 159 is unchanged and unaffected — a bare `/rewrite` and a `/rewrite/:rewriteId` param route coexist in React Router without shadowing.)

- [ ] **Step 2: Add the nav link in `Header.jsx`**

`Header.jsx`'s `marketingLinks` currently includes (lines 17-19):
```jsx
    { to: publicPath('/#product'), label: t('nav.why') },
    { to: publicPath('/features'), label: t('nav.features') },
    { to: publicPath('/technology'), label: t('nav.technology') },
```
Add a rewrite entry right after the technology one:
```jsx
    { to: publicPath('/#product'), label: t('nav.why') },
    { to: publicPath('/features'), label: t('nav.features') },
    { to: publicPath('/technology'), label: t('nav.technology') },
    { to: publicPath('/rewrite'), label: t('nav.rewrite') },
```

- [ ] **Step 3: Add `nav.rewrite` to both nav i18n files**

In `draftproof-frontend/src/i18n/en/nav.js`, next to `"technology": "Technology",` (line 16), add:
```js
  "rewrite": "Rewrite",
```
In `draftproof-frontend/src/i18n/zh/nav.js`, next to its `"technology"` line, add:
```js
  "rewrite": "改写",
```

- [ ] **Step 4: Add the footer link in `Footer.jsx`**

In `draftproof-frontend/src/components/Footer.jsx`, the product `<nav>` list has (line 24):
```jsx
          <Link to={publicPath('/content-checker')}>{t('footer.essayChecker')}</Link>
```
Add a rewrite link immediately after it:
```jsx
          <Link to={publicPath('/content-checker')}>{t('footer.essayChecker')}</Link>
          <Link to={publicPath('/rewrite')}>{t('footer.rewrite')}</Link>
```
(Do NOT modify line 22's `/#engine` link.)

- [ ] **Step 5: Add `footer.rewrite` to both footer i18n files**

In `draftproof-frontend/src/i18n/en/footer.js`, next to `"essayChecker": "Content checker",` (line 5), add:
```js
  "rewrite": "Rewrite",
```
In `draftproof-frontend/src/i18n/zh/footer.js`, next to its `essayChecker` line, add:
```js
  "rewrite": "改写",
```

- [ ] **Step 6: Add `/rewrite` to `LOCALIZABLE_PUBLIC_PATHS`**

In `draftproof-frontend/src/localeRouting.js`, the array currently includes `'/technology'` (added for the technology page). Insert `'/rewrite'` right after `'/technology'`:
```js
export const LOCALIZABLE_PUBLIC_PATHS = ['/', '/why', '/features', '/technology', '/rewrite', '/content-checker', '/turnitin-ai-score', '/academic-integrity-ai', '/ai-declaration', '/reduce-ai-detection', '/pricing', '/faq', '/privacy', '/security', '/signin'];
```
(Match the exact current array contents when editing — copy the live line, insert `'/rewrite'` after `'/technology'`.)

- [ ] **Step 7: Add the `/rewrite` `PAGE_META` entry in `seoMetadata.js`**

Immediately after the existing `/technology` entry in `PAGE_META`, add:
```js
  '/rewrite': {
    titleKey: 'seo.rewriteTitle',
    descriptionKey: 'seo.rewriteDescription',
    socialDescriptionKey: 'seo.rewriteSocialDescription',
    canonical: '/rewrite',
    schemaType: 'WebPage',
    freshness: { type: 'reviewed', date: '2026-07-06' },
  },
```

- [ ] **Step 8: Add the SEO copy keys to `seo.js` (en + zh)**

In `draftproof-frontend/src/i18n/en/seo.js`, next to the existing `technologyTitle`/`technologyDescription`/`technologySocialDescription` keys (around line 38), add:
```js
  "rewriteTitle": "The DraftProof Rewrite — A Worked Before/After You Finish in Your Own Words | DraftProof",
  "rewriteDescription": "When a scan flags a thin or unsupported passage, DraftProof shows a worked before/after — the citation, example, or reasoning it's missing — as a teaching draft you finish in your own words. Not a make-it-pass button.",
  "rewriteSocialDescription": "See what grounded writing looks like — a worked before/after you finish in your own words. Not a bypass.",
```
In `draftproof-frontend/src/i18n/zh/seo.js`, next to its zh `technology*` keys, add:
```js
  "rewriteTitle": "DraftProof 改写——一份你用自己的话写完的「改写前／改写后」 | DraftProof",
  "rewriteDescription": "当扫描标记出单薄或缺乏支撑的段落时，DraftProof 会给出一份可对照的「改写前／改写后」——补上它所缺的引用、事例或推理——作为一份你用自己的话写完的学习草稿。它不是一个「让它通过」的按钮。",
  "rewriteSocialDescription": "先看清有依据的写作是什么样——一份你用自己的话写完的「改写前／改写后」。它不是绕过检测的工具。",
```

- [ ] **Step 9: Add a "Learn more" link in the Features rewrite tab**

In `draftproof-frontend/src/pages/Features.jsx`, the `activeCards.map(...)` render (line 97) sits inside the `<div className="feat-cards" id="feat-tab-panel">` container (opens line 92). `Link` (imported line 2) and `publicPath` (defined line 18) are already in scope — no new import needed. Immediately after the `</div>` that closes `.feat-cards`, add a link shown only for the rewrite tab:
```jsx
          {activeTab === 'rewrite' && (
            <p style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: '0.875rem' }}>
              <Link to={publicPath('/rewrite')}>{t('featuresPage.rewriteLearnMore')}</Link>
            </p>
          )}
```

- [ ] **Step 10: Add `featuresPage.rewriteLearnMore` to both features i18n files**

In `draftproof-frontend/src/i18n/en/features.js`, add near the `rewriteCards` array:
```js
  rewriteLearnMore: "See how the rewrite works →",
```
In `draftproof-frontend/src/i18n/zh/features.js`, add near its `rewriteCards`:
```js
  rewriteLearnMore: "了解改写的工作方式 →",
```

- [ ] **Step 11: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors; the prerender log shows the route count increased by exactly 2 from the last known-good run (adds `/rewrite` + `/zh/rewrite`).

- [ ] **Step 12: Visual verification via preview tool**

Start/reuse the dev server (`.claude/launch.json` config `"frontend"`). Then:
- Navigate to `/rewrite`: hero renders, the 3 before/after examples render (each with a red "Before" row, green "After" row, and italic "Suggested Addition For Review" marker), the "what it is / isn't" framing renders, 3 value cards render, CTA links to `/signin?next=/scan`.
- Navigate to `/zh/rewrite`: fully localized, no English fallback.
- Confirm the top nav shows "Rewrite" (or "改写") when logged out, and the footer has a Rewrite link.
- Confirm `/rewrite/anything` still routes to the protected viewer (redirects to signin when logged out) — i.e. the public route did not shadow it.
- Console clean (`mcp__Claude_Preview__preview_console_logs`, level `error`).

- [ ] **Step 13: Commit**

```bash
git add draftproof-frontend/src/App.jsx draftproof-frontend/src/components/Header.jsx draftproof-frontend/src/components/Footer.jsx draftproof-frontend/src/localeRouting.js draftproof-frontend/src/seoMetadata.js draftproof-frontend/src/i18n/en/nav.js draftproof-frontend/src/i18n/zh/nav.js draftproof-frontend/src/i18n/en/footer.js draftproof-frontend/src/i18n/zh/footer.js draftproof-frontend/src/i18n/en/seo.js draftproof-frontend/src/i18n/zh/seo.js draftproof-frontend/src/pages/Features.jsx draftproof-frontend/src/i18n/en/features.js draftproof-frontend/src/i18n/zh/features.js
git commit -m "feat(rewrite): route, nav, footer, SEO wiring + Features learn-more link for /rewrite"
```

---

### Task 3: Landing before/after teaser section

**Files:**
- Modify: `draftproof-frontend/src/pages/Landing.jsx` (new teaser section + import)
- Modify: `draftproof-frontend/src/i18n/en/landing.js` (teaser keys)
- Modify: `draftproof-frontend/src/i18n/zh/landing.js` (teaser keys)

**Interfaces:**
- Consumes: `RewriteBeforeAfter` from Task 1 (`import { RewriteBeforeAfter } from '../components/RewriteBeforeAfter';`).
- Produces: nothing.

- [ ] **Step 1: Add teaser i18n keys to `i18n/en/landing.js`**

Add near the other sample-report keys (top-level object):
```js
  "rewriteTeaserEyebrow": "The Rewrite",
  "rewriteTeaserTitle": "Flagged a weak claim? See the fix.",
  "rewriteTeaserBody": "When a scan flags a thin passage, the rewrite shows a worked before/after — a teaching draft you finish in your own words.",
  "rewriteTeaserBeforeLabel": "Before",
  "rewriteTeaserAfterLabel": "After",
  "rewriteTeaserBefore": "Hollywood has become one of the most powerful cultural exports in history.",
  "rewriteTeaserAfter": "Hollywood is among the most widely exported cultural industries [add a citation you can verify].",
  "rewriteTeaserMarker": "Suggested Addition For Review — you replace the bracketed note with your own source.",
  "rewriteTeaserLink": "See how the rewrite works →",
```

- [ ] **Step 2: Add the matching zh keys to `i18n/zh/landing.js`**

```js
  "rewriteTeaserEyebrow": "改写",
  "rewriteTeaserTitle": "有论断被标记了？看看该怎么修。",
  "rewriteTeaserBody": "当扫描标记出单薄的段落时，改写会给出一份可对照的「改写前／改写后」——一份你用自己的话写完的学习草稿。",
  "rewriteTeaserBeforeLabel": "改写前",
  "rewriteTeaserAfterLabel": "改写后",
  "rewriteTeaserBefore": "好莱坞已成为史上最具影响力的文化输出之一。",
  "rewriteTeaserAfter": "好莱坞是全球输出规模最大的文化产业之一〔补一个你能核实的引用〕。",
  "rewriteTeaserMarker": "待复核的补充内容——请用你自己的来源替换括号中的提示。",
  "rewriteTeaserLink": "了解改写的工作方式 →",
```

- [ ] **Step 3: Import `RewriteBeforeAfter` in `Landing.jsx`**

Near the top of `Landing.jsx` with the other imports (it already imports `AuthorshipClarityBreakdown` from `./report/AuthorshipClarityBreakdown` and other components), add:
```jsx
import { RewriteBeforeAfter } from '../components/RewriteBeforeAfter';
```

- [ ] **Step 4: Add the teaser section to `Landing.jsx`**

The landing render currently has (lines 205-211):
```jsx
      <ReportStrategyCarousel
        contentStrategies={contentStrategies}
        publicPath={publicPath}
        reportValueCards={reportValueCards}
      />

      <section id="help" className="landing-section help-section">
```
Insert a new teaser section between the `ReportStrategyCarousel` closing `/>` and the `<section id="help" ...>`:
```jsx
      <ReportStrategyCarousel
        contentStrategies={contentStrategies}
        publicPath={publicPath}
        reportValueCards={reportValueCards}
      />

      <section id="rewrite-teaser" className="landing-section">
        <div className="section-inner">
          <p className="eyebrow">{t('landing.rewriteTeaserEyebrow')}</p>
          <h2>{t('landing.rewriteTeaserTitle')}</h2>
          <p className="section-lead" style={{ color: 'var(--muted)', marginBottom: '1.25rem' }}>{t('landing.rewriteTeaserBody')}</p>
          <div style={{ maxWidth: '640px' }}>
            <RewriteBeforeAfter
              before={t('landing.rewriteTeaserBefore')}
              after={t('landing.rewriteTeaserAfter')}
              marker={t('landing.rewriteTeaserMarker')}
              beforeLabel={t('landing.rewriteTeaserBeforeLabel')}
              afterLabel={t('landing.rewriteTeaserAfterLabel')}
            />
          </div>
          <p style={{ marginTop: '1.25rem', fontSize: '0.875rem' }}>
            <Link to={publicPath('/rewrite')}>{t('landing.rewriteTeaserLink')}</Link>
          </p>
        </div>
      </section>

      <section id="help" className="landing-section help-section">
```
(`Link` and `publicPath` are already imported/defined in `Landing.jsx` — confirmed used throughout the file.)

- [ ] **Step 5: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors.

- [ ] **Step 6: Visual verification via preview tool**

Navigate to `/`, scroll to the new "The Rewrite" teaser section (right after the report/strategy carousel): confirm the before/after card renders (red Before row, green After row, italic marker) and the "See how the rewrite works →" link navigates to `/rewrite`. Repeat on `/zh` — fully localized, link goes to `/zh/rewrite`. Console clean.

- [ ] **Step 7: Commit**

```bash
git add draftproof-frontend/src/pages/Landing.jsx draftproof-frontend/src/i18n/en/landing.js draftproof-frontend/src/i18n/zh/landing.js
git commit -m "feat(landing): add rewrite before/after teaser section linking to /rewrite"
```

---

## Self-Review Notes

- **Spec coverage:** /rewrite page (hero + 3-example demo + framing + value cards + CTA) → Task 1. Route/nav/footer/locale/SEO/Features-link wiring → Task 2. Landing teaser → Task 3. Alignment guardrails → enforced in every copy block (bracketed "you replace this" marker present in all before/after examples; CTA is "Start a scan"). Shared `RewriteBeforeAfter` avoids duplicating the card. ✓
- **Placeholder scan:** no TBD/TODO; all component code, CSS, and en+zh copy are complete and copy-pasteable. The two "find/confirm exact location" notes (Features `activeCards.map` placement, `why-card-grid` class existence) are grounded verification steps, not placeholders — they name exactly what to check and the fallback. ✓
- **Type/name consistency:** `RewriteBeforeAfter` named export + its `{before, after, marker, beforeLabel, afterLabel}` prop shape defined in Task 1 Step 1, consumed identically in Task 1 Step 6 (page) and Task 3 Step 4 (landing). `RewriteOverview` default export name matches Task 2 Step 1's import. `rewriteOverview.examples` shape (`{fixType, before, after, marker}`) defined in Task 1 Steps 3-4, consumed in Step 6. `featuresPage.rewriteLearnMore` key defined in Task 2 Step 10, consumed Step 9. No drift. ✓
- **Cross-task ordering:** Task 3 imports `RewriteBeforeAfter` from Task 1, and its link target `/rewrite` is routed by Task 2 — so run order is 1 → 2 → 3. Task 3's build compiles even before Task 2 (the link is a plain path string), but the link only resolves after Task 2, so 2-before-3 is required for the Step 6 visual check to pass.
