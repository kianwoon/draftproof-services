# Features Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/features` page (EN + ZH) with a competitor comparison table and DraftProof-only feature cards, wired into the public marketing nav.

**Architecture:** New `Features.jsx` page following the `Why.jsx` shell pattern; content driven entirely from i18n so ZH works without code changes; table rows stored as structured data in translation files; new `feat-*` CSS in the existing marketing-pages stylesheet.

**Tech Stack:** React + React Router, react-i18next, Vite, existing site-master CSS

---

## File Map

| Action | File |
|--------|------|
| Create | `src/i18n/en/features.js` |
| Create | `src/i18n/zh/features.js` |
| Modify | `src/i18n/en.js` |
| Modify | `src/i18n/zh.js` |
| Modify | `src/i18n/en/nav.js` |
| Modify | `src/i18n/zh/nav.js` |
| Modify | `src/i18n/en/seo.js` |
| Modify | `src/i18n/zh/seo.js` |
| Modify | `src/seoMetadata.js` |
| Create | `src/pages/Features.jsx` |
| Modify | `src/styles/site-master/04-app-marketing-pages.css` |
| Modify | `src/App.jsx` |
| Modify | `src/components/Header.jsx` |

---

### Task 1: EN i18n content file

**Files:**
- Create: `src/i18n/en/features.js`

- [ ] **Step 1: Create the file**

```js
export const featuresPage = {
  eyebrow: "Why DraftProof",
  title: "Detect. Understand. Improve.",
  lead: "Every other detector tells you that you failed. DraftProof shows you how to pass — by teaching you to write better.",
  tableLabel: "How we compare",
  competitors: ["DraftProof", "GPTZero", "Turnitin", "Originality.ai", "Winston AI"],
  rows: [
    { label: "Paragraph-level output", values: ["yes", "yes", "yes", "yes", "no"] },
    { label: "Explains why content is flagged", values: ["yes", "no", "no", "no", "no"] },
    { label: "Integrated rewrite / coaching", values: ["yes", "no", "no", "no", "no"] },
    { label: "Before/after diff view", values: ["yes", "no", "no", "no", "no"] },
    { label: "Policy-aware scoring", values: ["yes", "no", "no", "no", "no"] },
    { label: "Submission risk framing", values: ["yes", "no", "no", "no", "no"] },
    { label: "Critical thinking assessment", values: ["yes", "no", "no", "no", "no"] },
    { label: "Honest about detector limits", values: ["yes", "no", "no", "no", "no"] },
    { label: "Individual access (no institution needed)", values: ["yes", "yes", "no", "yes", "yes"] },
  ],
  cardsLabel: "DraftProof-only features",
  cards: [
    {
      icon: "ti-writing",
      title: "Grounded rewrite coaching",
      body: "Shows a worked example of your paragraph with concrete anchors, sources, and specifics — so you see what grounded writing looks like.",
    },
    {
      icon: "ti-shield-check",
      title: "Policy risk (dual-mode)",
      body: "Two scores from one engine — AI-Allowed and AI-Restricted — matching your institution's actual policy, not a single generic verdict.",
    },
    {
      icon: "ti-clipboard-check",
      title: "Submission risk framing",
      body: "\"Can you defend this as your own work?\" — framed across three layers: text pattern, thinking ownership, and academic grounding.",
    },
    {
      icon: "ti-brain",
      title: "Critical thinking control",
      body: "Five dimensions — specific context, student judgement, reasoning trail, evidence grounding, AI dependency — tell you what to think harder about.",
    },
  ],
  ctaTitle: "See your writing through a new lens.",
  ctaBody: "Start your first scan — free with your first credits.",
  ctaButton: "Start review",
};
```

- [ ] **Step 2: Commit**

```bash
git add src/i18n/en/features.js
git commit -m "feat(i18n): add EN features page content"
```

---

### Task 2: ZH i18n content file

**Files:**
- Create: `src/i18n/zh/features.js`

- [ ] **Step 1: Create the file**

```js
export const featuresPage = {
  eyebrow: "为什么选择 DraftProof",
  title: "检测。理解。提升。",
  lead: "其他检测工具只告诉你不合格。DraftProof 告诉你如何通过 — 通过教你更好地写作。",
  tableLabel: "功能对比",
  competitors: ["DraftProof", "GPTZero", "Turnitin", "Originality.ai", "Winston AI"],
  rows: [
    { label: "段落级别输出", values: ["yes", "yes", "yes", "yes", "no"] },
    { label: "解释内容被标记的原因", values: ["yes", "no", "no", "no", "no"] },
    { label: "集成改写 / 辅导", values: ["yes", "no", "no", "no", "no"] },
    { label: "修改前后对比视图", values: ["yes", "no", "no", "no", "no"] },
    { label: "政策感知评分", values: ["yes", "no", "no", "no", "no"] },
    { label: "提交风险评估", values: ["yes", "no", "no", "no", "no"] },
    { label: "批判性思维评估", values: ["yes", "no", "no", "no", "no"] },
    { label: "如实披露检测局限性", values: ["yes", "no", "no", "no", "no"] },
    { label: "个人可直接使用（无需机构账号）", values: ["yes", "yes", "no", "yes", "yes"] },
  ],
  cardsLabel: "DraftProof 独有功能",
  cards: [
    {
      icon: "ti-writing",
      title: "基于依据的改写辅导",
      body: "展示你段落的改写示例，包含具体锚点、来源和细节 — 让你看到有依据的写作是什么样子。",
    },
    {
      icon: "ti-shield-check",
      title: "政策风险（双模式）",
      body: "同一引擎生成两个评分 — "允许使用 AI"与"限制使用 AI" — 匹配你所在机构的实际政策，而非单一通用判定。",
    },
    {
      icon: "ti-clipboard-check",
      title: "提交风险评估",
      body: ""你能证明这是你自己的作品吗？" — 从三个维度评估：文本模式、思维所有权和学术依据。",
    },
    {
      icon: "ti-brain",
      title: "批判性思维控制",
      body: "五个维度 — 具体情境、学生判断、推理轨迹、证据依据、AI 依赖度 — 告诉你需要更深入思考的方向。",
    },
  ],
  ctaTitle: "以全新视角审视你的写作。",
  ctaBody: "开始你的第一次扫描 — 首次积分免费体验。",
  ctaButton: "开始审阅",
};
```

- [ ] **Step 2: Commit**

```bash
git add src/i18n/zh/features.js
git commit -m "feat(i18n): add ZH features page content"
```

---

### Task 3: Register i18n in barrels + add nav/SEO keys

**Files:**
- Modify: `src/i18n/en.js`
- Modify: `src/i18n/zh.js`
- Modify: `src/i18n/en/nav.js`
- Modify: `src/i18n/zh/nav.js`
- Modify: `src/i18n/en/seo.js`
- Modify: `src/i18n/zh/seo.js`

- [ ] **Step 1: Add import + registration in `src/i18n/en.js`**

Add after the last existing import line (after `import { ticker } from './en/ticker.js';`):
```js
import { featuresPage } from './en/features.js';
```

Add inside `enTranslation` after the last key (after `"ticker": ticker,`):
```js
"featuresPage": featuresPage,
```

- [ ] **Step 2: Add import + registration in `src/i18n/zh.js`**

Add after `import { ticker } from './zh/ticker.js';`:
```js
import { featuresPage } from './zh/features.js';
```

Add inside `zhTranslation` after the last key:
```js
"featuresPage": featuresPage,
```

- [ ] **Step 3: Add nav key to `src/i18n/en/nav.js`**

Add inside the exported object, after `"why"`:
```js
"features": "Features",
```

- [ ] **Step 4: Add nav key to `src/i18n/zh/nav.js`**

Add after `"why"`:
```js
"features": "功能对比",
```

- [ ] **Step 5: Add SEO keys to `src/i18n/en/seo.js`**

Add at the end of the exported object, before the closing `}`:
```js
"featuresTitle": "Features | DraftProof vs GPTZero, Turnitin & More",
"featuresDescription": "See how DraftProof compares to GPTZero, Turnitin, Originality.ai, and Winston AI — the only detector that explains why content is flagged and coaches you to fix it.",
```

- [ ] **Step 6: Add SEO keys to `src/i18n/zh/seo.js`**

Add at the end of the exported object, before the closing `}`:
```js
"featuresTitle": "功能对比 | DraftProof vs GPTZero、Turnitin 等",
"featuresDescription": "了解 DraftProof 与 GPTZero、Turnitin、Originality.ai 和 Winston AI 的区别 — 唯一能解释内容被标记原因并辅导你改进的检测工具。",
```

- [ ] **Step 7: Commit**

```bash
git add src/i18n/en.js src/i18n/zh.js src/i18n/en/nav.js src/i18n/zh/nav.js src/i18n/en/seo.js src/i18n/zh/seo.js
git commit -m "feat(i18n): register featuresPage namespace + nav/seo keys (en+zh)"
```

---

### Task 4: SEO metadata entry

**Files:**
- Modify: `src/seoMetadata.js`

- [ ] **Step 1: Add the `/features` entry**

Inside `PAGE_META`, add after the `/why` entry:
```js
'/features': {
  titleKey: 'seo.featuresTitle',
  descriptionKey: 'seo.featuresDescription',
  canonical: '/features',
  schemaType: 'WebPage',
  freshness: { type: 'reviewed', date: '2026-06-19' },
},
```

- [ ] **Step 2: Commit**

```bash
git add src/seoMetadata.js
git commit -m "feat(seo): add /features page metadata"
```

---

### Task 5: CSS for comparison table and feature cards

**Files:**
- Modify: `src/styles/site-master/04-app-marketing-pages.css`

- [ ] **Step 1: Append feat-* styles at the end of the file**

```css
/* ── Features page ─────────────────────────────────────── */
.feat-shell {
  min-height: 100vh;
}

.feat-shell .container {
  max-width: 860px;
  margin: 0 auto;
  padding: 0 1.5rem 4rem;
}

.feat-section-label {
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--color-text-tertiary);
  margin: 0 0 0.75rem;
}

/* Comparison table */
.feat-table-wrap {
  overflow-x: auto;
  margin-bottom: 2.5rem;
  -webkit-overflow-scrolling: touch;
}

.feat-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8125rem;
  table-layout: fixed;
  min-width: 560px;
}

.feat-table th {
  padding: 0.5rem 0.75rem;
  font-size: 0.6875rem;
  font-weight: 600;
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border-tertiary);
  text-align: center;
  white-space: nowrap;
}

.feat-table th:first-child {
  text-align: left;
  width: 38%;
}

.feat-table th.feat-th-dp {
  background: var(--color-background-info);
  color: var(--color-text-info);
  border-radius: 6px 6px 0 0;
}

.feat-table td {
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid var(--color-border-tertiary);
  text-align: center;
  color: var(--color-text-secondary);
  vertical-align: middle;
}

.feat-table td:first-child {
  text-align: left;
  color: var(--color-text-primary);
}

.feat-table tbody tr:last-child td {
  border-bottom: none;
}

.feat-table td.feat-td-dp {
  background: var(--color-background-info);
}

.feat-yes {
  color: var(--color-text-success);
  font-size: 1rem;
  line-height: 1;
}

.feat-no {
  color: var(--color-text-danger);
  font-size: 1rem;
  line-height: 1;
}

.feat-partial {
  font-size: 0.6875rem;
  font-weight: 600;
  color: var(--color-text-warning);
}

/* Feature cards */
.feat-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin-bottom: 3rem;
}

.feat-card {
  background: var(--color-background-primary);
  border: 1px solid var(--color-border-tertiary);
  border-radius: 12px;
  padding: 1.125rem 1.25rem;
}

.feat-card-icon {
  font-size: 1.25rem;
  color: var(--color-text-info);
  margin-bottom: 0.4rem;
}

.feat-card h3 {
  font-size: 0.875rem;
  font-weight: 600;
  margin: 0 0 0.35rem;
  color: var(--color-text-primary);
}

.feat-card p {
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
  margin: 0;
  line-height: 1.55;
}

@media (max-width: 600px) {
  .feat-cards {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/styles/site-master/04-app-marketing-pages.css
git commit -m "feat(css): add feat-* styles for Features page table and cards"
```

---

### Task 6: Features.jsx page component

**Files:**
- Create: `src/pages/Features.jsx`

- [ ] **Step 1: Create the component**

```jsx
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

function CellValue({ value }) {
  if (value === 'yes') return <span className="feat-yes" aria-label="yes">✓</span>;
  if (value === 'no') return <span className="feat-no" aria-label="no">✗</span>;
  return <span className="feat-partial" aria-label="partial">partial</span>;
}

export default function Features() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);

  const competitors = t('featuresPage.competitors', { returnObjects: true });
  const rows = t('featuresPage.rows', { returnObjects: true });
  const cards = t('featuresPage.cards', { returnObjects: true });

  return (
    <main className="feat-shell">
      <div className="container">
        <section className="app-hero app-hero-dark">
          <CodeTexture id="featHero" />
          <div>
            <p className="eyebrow">{t('featuresPage.eyebrow')}</p>
            <h1>{t('featuresPage.title')}</h1>
            <p className="lead">{t('featuresPage.lead')}</p>
          </div>
        </section>

        <section style={{ marginTop: '2.5rem' }}>
          <p className="feat-section-label">{t('featuresPage.tableLabel')}</p>
          <div className="feat-table-wrap">
            <table className="feat-table">
              <thead>
                <tr>
                  <th />
                  {competitors.map((name, i) => (
                    <th key={name} className={i === 0 ? 'feat-th-dp' : undefined}>
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.label}>
                    <td>{row.label}</td>
                    {row.values.map((val, i) => (
                      <td key={i} className={i === 0 ? 'feat-td-dp' : undefined}>
                        <CellValue value={val} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <p className="feat-section-label">{t('featuresPage.cardsLabel')}</p>
          <div className="feat-cards">
            {cards.map((card) => (
              <div className="feat-card" key={card.title}>
                <div className="feat-card-icon" aria-hidden="true">
                  <i className={`ti ${card.icon}`} />
                </div>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="why-cta">
          <h2>{t('featuresPage.ctaTitle')}</h2>
          <p>{t('featuresPage.ctaBody')}</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <Link to={publicPath('/signin')} className="btn btn-primary">
              {t('featuresPage.ctaButton')}
            </Link>
          </div>
        </section>

        <PageFreshness path="/features" />
      </div>
    </main>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/pages/Features.jsx
git commit -m "feat: add Features page component with comparison table and feature cards"
```

---

### Task 7: Routes in App.jsx

**Files:**
- Modify: `src/App.jsx`

- [ ] **Step 1: Add the import**

After `import Why from './pages/Why';`, add:
```js
import Features from './pages/Features';
```

- [ ] **Step 2: Add the routes**

After `<Route path="/zh/why" element={<Why />} />`, add:
```jsx
<Route path="/features" element={<Features />} />
<Route path="/zh/features" element={<Features />} />
```

- [ ] **Step 3: Commit**

```bash
git add src/App.jsx
git commit -m "feat(routing): add /features and /zh/features routes"
```

---

### Task 8: Nav link in Header.jsx

**Files:**
- Modify: `src/components/Header.jsx`

- [ ] **Step 1: Add to marketingLinks**

In `Header.jsx`, find the `marketingLinks` array. Add a new entry after `{ to: publicPath('/why'), label: t('nav.why') }` (the first entry, "Why"):

```js
{ to: publicPath('/features'), label: t('nav.features') },
```

The full `marketingLinks` array becomes:
```js
const marketingLinks = [
  { to: publicPath('/#product'), label: t('nav.why') },
  { to: publicPath('/features'), label: t('nav.features') },
  { to: publicPath('/content-checker'), label: t('nav.essayChecker') },
  { to: publicPath('/pricing'), label: t('nav.pricing') },
  { to: publicPath('/faq'), label: t('nav.faq') },
  { to: publicPath('/#report'), label: t('nav.sampleReport') },
];
```

- [ ] **Step 2: Commit**

```bash
git add src/components/Header.jsx
git commit -m "feat(nav): add Features link to marketing nav"
```

---

### Task 9: Manual verification

- [ ] **Step 1: Start dev server**

```bash
cd draftproof-frontend
npm run dev
```

Expected: server starts at `http://localhost:3000` with no console errors.

- [ ] **Step 2: Verify nav**

Open `http://localhost:3000` (signed-out state). Confirm "Features" appears in the top nav between "Why" and "Content checker".

- [ ] **Step 3: Verify EN page**

Navigate to `http://localhost:3000/features`. Confirm:
- Dark hero with "Detect. Understand. Improve." headline
- Comparison table with DraftProof column tinted blue
- ✓ in DraftProof column for all 9 rows
- ✗ in the "Explains why content is flagged" row for all competitors
- 4 feature cards below the table
- "Start review" CTA button at bottom

- [ ] **Step 4: Verify ZH page**

Navigate to `http://localhost:3000/zh/features`. Confirm all content renders in Chinese.

- [ ] **Step 5: Verify mobile table**

Resize browser to 375px width. Confirm table scrolls horizontally and does not overflow the page.

- [ ] **Step 6: Verify mobile nav**

At 375px, open the hamburger menu. Confirm "Features" (or "功能对比" on `/zh`) appears in the mobile menu.
