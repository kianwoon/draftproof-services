# Technology Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a new public `/technology` page (+ `/zh/technology`) that showcases DraftProof's engineering rigor at a capability level (no library/model/vendor names) to build credibility with both students and educators, wired into the existing marketing-page conventions.

**Architecture:** One new page component (`Technology.jsx`) following the exact structural pattern of the existing `Why.jsx` page (hero + mapped content sections + closing CTA, reusing existing `why-*` CSS classes — no new CSS), backed by a new `technologyPage.*` i18n namespace. A second task wires it into routing, nav, SEO metadata, and locale-safety lists, plus repairs two now-fixable dead `#engine` links discovered while investigating this page's placement.

**Tech Stack:** React 18 + react-i18next + react-router-dom, Vite. No test framework exists for this frontend — verification is `npm run build:client` plus a preview-tool visual check.

## Global Constraints

- No literal library/model/vendor names anywhere in the new copy (no "Deberta," "Modal," "Cerebras," "gpt-oss," etc.) — capability-level framing only, per explicit user instruction.
- Every new/changed i18n key must exist in BOTH the `en` and `zh` variant of whichever file it lives in (locale-parity requirement; this codebase has a documented "locale trap" where a missing zh key silently falls back to English).
- `/technology` MUST be added to `LOCALIZABLE_PUBLIC_PATHS` in `draftproof-frontend/src/localeRouting.js:4`, or `/zh/technology` will silently render the English body.
- `/technology` MUST get a `PAGE_META` entry in `draftproof-frontend/src/seoMetadata.js` — `PRERENDER_PATHS` derives from `PAGE_META`'s keys automatically, so this is also what makes the SSG prerender pick it up.
- The new nav link only appears for logged-out visitors — inherited for free from `Header.jsx`'s existing `visiblePublicLinks = user ? signedInPublicLinks : marketingLinks` pattern (`Header.jsx:26`), no new conditional logic to write.
- No new CSS — reuse the existing `why-hero` / `why-section` / `why-num` / `why-highlight` / `why-punch` / `why-cta` classes already defined for the Why page (confirmed via `Why.jsx` — these are global via `main.jsx:6` → `site-master.css`).
- This is additive only — no change to `Landing.jsx`'s main content (the earlier decision to keep that page student-only stands), except the one dead-link repair named in Task 2.

---

### Task 1: Technology page content (component + i18n)

**Files:**
- Create: `draftproof-frontend/src/pages/Technology.jsx`
- Create: `draftproof-frontend/src/i18n/en/technologyPage.js`
- Create: `draftproof-frontend/src/i18n/zh/technologyPage.js`
- Modify: `draftproof-frontend/src/i18n/resources.js` (register the two new namespace files — same pattern as `whyPage`)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `export default function Technology()` (default export, no props) — Task 2 imports this as `import Technology from './pages/Technology';` and renders `<Technology />` with no props, exactly like `<Why />`/`<Features />`.

- [ ] **Step 1: Confirm how `whyPage` is registered in `resources.js`**

Run: `grep -n "whyPage" draftproof-frontend/src/i18n/resources.js`
Expected output shows two lines like:
```js
import { whyPage as whyPageEn } from './en/whyPage';
import { whyPage as whyPageZh } from './zh/whyPage';
```
plus two more lines merging `whyPageEn`/`whyPageZh` into the `en`/`zh` translation objects (e.g. `...whyPageEn,` inside the `en.translation` spread, `...whyPageZh,` inside `zh.translation`). Use the exact same 4-line pattern for `technologyPage` in Step 5 below — match whatever variable-naming and spread style you see, don't invent a different convention.

- [ ] **Step 2: Create `draftproof-frontend/src/i18n/en/technologyPage.js`**

```js
export const technologyPage = {
  "eyebrow": "The Technology Behind DraftProof",
  "title": "We didn't build another AI detector. We built the engineering discipline a new era of education actually needs.",
  "lead": "Every AI writing checker makes a promise. Very few show their work. Here's what actually runs behind every DraftProof report — the practices we test, the guardrails we enforce, and the fairness bar we refuse to drop below.",
  "pillarsLabel": "The engineering practices behind every DraftProof report",
  "pillars": [
    {
      "title": "No single black-box score decides anything.",
      "body": "A single AI-detector call is noisy and easy to game in either direction. DraftProof combines multiple independent detection signals — pattern-based analysis and a separate deep-reading model — before any tier or score is shown, so no single miscalibrated signal can swing a verdict on its own.",
      "whyItMatters": "For students: one flaky detector can't wrongly flag your work. For educators: a verdict backed by agreement across independent signals is more defensible than a single tool's number."
    },
    {
      "title": "Validated against real multilingual student writing — not just native-English samples.",
      "body": "AI-detection tools are well known to misfire more often on non-native English writers, flagging normal ESL phrasing as \"AI-generated.\" Before any detection change ships, DraftProof runs it against a dedicated corpus of real student essays across proficiency levels, and blocks the release if it would raise false-positive rates for ESL writers or widen the gap between proficiency groups.",
      "whyItMatters": "For students: your writing style shouldn't be penalized for being a second language. For educators: this is the single most common fairness complaint against AI detectors — DraftProof tests for it explicitly, every time, not just once."
    },
    {
      "title": "When a signal is missing or uncertain, the system never guesses against you.",
      "body": "Every additional detection layer is designed to fail safe: if an experimental or unavailable signal can't be computed, DraftProof falls back to its established, calibrated score rather than inventing a harsher one. Guardrails are built to flag content for review, not to silently discard a student's real work.",
      "whyItMatters": "For students: a technical hiccup can't manufacture a false accusation. For educators: the system is built to avoid false positives by design, not just by promise."
    },
    {
      "title": "Re-tested against new AI writing tools as they appear.",
      "body": "AI writing tools keep changing, and a detector tuned once against yesterday's tools quietly goes stale. DraftProof re-validates its detection signals against outputs from multiple, independently-developed AI writing systems on an ongoing basis, so accuracy doesn't quietly decay as the underlying AI landscape shifts.",
      "whyItMatters": "For students: the bar doesn't drift unfairly over time. For educators: you're not relying on a snapshot from whenever the tool launched."
    },
    {
      "title": "You see the calibration, not just a percentage.",
      "body": "Every score DraftProof shows is tied to a labeled band and an explanation of what's driving it — never a bare, unexplained number. The calibration behind each tier is documented and versioned, so a score means the same thing today as it will next term.",
      "whyItMatters": "For students: you get something to act on, not just a grade. For educators: a transparent scale is one you can actually stand behind in a conversation with a student or a parent."
    }
  ],
  "ctaTitle": "This is what \"best stack\" means to us: proof, not adjectives.",
  "ctaBody": "No tool can promise a perfect verdict on every draft. What DraftProof can promise is a system that is tested, fails toward fairness, and keeps being re-checked as AI keeps changing.",
  "ctaRun": "Try it yourself",
  "ctaHow": "See a sample report"
};
```

- [ ] **Step 3: Create `draftproof-frontend/src/i18n/zh/technologyPage.js`**

```js
export const technologyPage = {
  "eyebrow": "DraftProof 背后的技术",
  "title": "我们打造的不是另一个 AI 检测器,而是新教育时代真正需要的工程严谨性。",
  "lead": "每一个 AI 写作检测工具都会做出承诺,但很少有工具愿意展示它们的实际做法。这里展示的是每一份 DraftProof 报告背后真正运行的机制——我们测试的做法、我们执行的防护机制,以及我们绝不妥协的公平底线。",
  "pillarsLabel": "支撑每一份 DraftProof 报告的工程实践",
  "pillars": [
    {
      "title": "没有任何单一的黑箱分数能决定结果。",
      "body": "单一的 AI 检测调用充满噪声,也容易被规避。DraftProof 在给出任何等级或分数之前,会综合多个独立的检测信号——基于模式的分析和一个独立的深度阅读模型——因此不会有单一失准的信号左右最终结论。",
      "whyItMatters": "对学生而言:一个不稳定的检测器不会错误地标记你的作品。对教育者而言:由多个独立信号共同支持的结论,比单一工具给出的数字更站得住脚。"
    },
    {
      "title": "针对真实的多语言学生写作进行验证——而不仅仅是母语英语样本。",
      "body": "众所周知,AI 检测工具更容易误判非母语英语写作者,把正常的 ESL 表达误标为「AI 生成」。在任何检测逻辑变更上线之前,DraftProof 都会用一个涵盖不同水平真实学生作文的专门语料库进行测试,如果该变更会提高 ESL 写作者的误判率,就会被拦截,不予发布。",
      "whyItMatters": "对学生而言:你的写作风格不应因为是第二语言而受到惩罚。对教育者而言:这是针对 AI 检测器最常见的公平性质疑——DraftProof 每一次都会明确测试这一点,而不是只测一次。"
    },
    {
      "title": "当某个信号缺失或不确定时,系统绝不会对你做出不利的猜测。",
      "body": "每一个新增的检测层都被设计为「安全失败」:如果某个实验性或不可用的信号无法计算,DraftProof 会回退到其既有的、经过校准的分数,而不是凭空生成一个更严厉的结果。防护机制的作用是把内容标记出来供人工复核,而不是悄悄丢弃学生的真实作品。",
      "whyItMatters": "对学生而言:技术上的小故障不会被用来编造不实指控。对教育者而言:这个系统在设计上就是为了避免误判,而不仅仅是口头承诺。"
    },
    {
      "title": "随着新的 AI 写作工具出现而持续重新测试。",
      "body": "AI 写作工具在不断变化,如果检测器只针对过去的工具校准一次,就会在不知不觉中变得过时。DraftProof 会持续地针对多个独立开发的 AI 写作系统的输出,重新验证其检测信号,这样准确性就不会随着 AI 生态的变化而悄悄退化。",
      "whyItMatters": "对学生而言:评判标准不会随时间不公平地漂移。对教育者而言:你依赖的不是工具刚上线那一刻的快照。"
    },
    {
      "title": "你看到的是校准依据,而不只是一个百分比。",
      "body": "DraftProof 给出的每一个分数都对应一个明确标注的区间,并附有说明其成因的解释——而不是一个孤立、无法解释的数字。每个等级背后的校准方法都有文档记录并标注版本,因此今天的一个分数和下学期的含义是一致的。",
      "whyItMatters": "对学生而言:你得到的是可以采取行动的依据,而不只是一个成绩。对教育者而言:一个透明的评分体系,才是你能在与学生或家长对话时真正站得住脚的依据。"
    }
  ],
  "ctaTitle": "这就是我们所说的「最佳技术栈」:靠证据说话,而不是靠形容词。",
  "ctaBody": "没有任何工具能保证对每一份草稿都给出完美的判断。DraftProof 能承诺的是:一个经过测试、在不确定时始终倾向公平、并随着 AI 不断变化而持续接受复检的系统。",
  "ctaRun": "亲自试一试",
  "ctaHow": "查看示例报告"
};
```

- [ ] **Step 4: Register both files in `draftproof-frontend/src/i18n/resources.js`**

Following the exact 4-line pattern you found for `whyPage` in Step 1 (same variable-naming convention, same spread-into-`en.translation`/`zh.translation` locations), add the equivalent 4 lines for `technologyPage`. For example, if `whyPage` uses:
```js
import { whyPage as whyPageEn } from './en/whyPage';
import { whyPage as whyPageZh } from './zh/whyPage';
```
add immediately after:
```js
import { technologyPage as technologyPageEn } from './en/technologyPage';
import { technologyPage as technologyPageZh } from './zh/technologyPage';
```
and wherever `...whyPageEn,` / `...whyPageZh,` are spread into the translation objects, add `...technologyPageEn,` / `...technologyPageZh,` right next to them.

- [ ] **Step 5: Create `draftproof-frontend/src/pages/Technology.jsx`**

```jsx
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import CodeTexture from '../components/CodeTexture';
import PageFreshness from '../components/PageFreshness';
import { getLocaleFromPathname, localizePath } from '../localeRouting';

export default function Technology() {
  const { t } = useTranslation();
  const location = useLocation();
  const locale = getLocaleFromPathname(location.pathname);
  const publicPath = (path) => localizePath(path, locale);
  const pillars = t('technologyPage.pillars', { returnObjects: true });

  return (
    <main className="why-shell">
      <div className="container">
        <section className="why-hero app-hero app-hero-dark">
          <CodeTexture id="technologyHero" />
          <div>
            <p className="eyebrow">{t('technologyPage.eyebrow')}</p>
            <h1>{t('technologyPage.title')}</h1>
            <p className="lead">{t('technologyPage.lead')}</p>
          </div>
        </section>

        {pillars.map((pillar, index) => (
          <section className="why-section" key={pillar.title} aria-label={t('technologyPage.pillarsLabel')}>
            <span className="why-num">{String(index + 1).padStart(2, '0')}</span>
            <h2>{pillar.title}</h2>
            <p>{pillar.body}</p>
            <p className="why-highlight">{pillar.whyItMatters}</p>
          </section>
        ))}

        <section className="why-cta">
          <h2>{t('technologyPage.ctaTitle')}</h2>
          <p>{t('technologyPage.ctaBody')}</p>
          <div className="hero-actions" style={{ justifyContent: 'center' }}>
            <Link to="/signin?next=/scan" className="btn btn-primary">{t('technologyPage.ctaRun')}</Link>
            <Link to={publicPath('/#report')} className="btn btn-secondary">{t('technologyPage.ctaHow')}</Link>
          </div>
        </section>

        <PageFreshness path="/technology" />
      </div>
    </main>
  );
}
```

Note: the closing CTA's secondary link points to `publicPath('/#report')` (the landing page's real, existing sample-report section — see `Landing.jsx`'s `<section id="report" ...>`), NOT to `/#engine` — that anchor's section no longer exists, which is exactly the dead-link problem Task 2 also repairs elsewhere.

- [ ] **Step 6: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors. `Technology.jsx` is not yet routed, so nothing renders it, but it must still compile (no syntax errors, no missing imports) and its i18n keys must load without `resources.js` throwing.

- [ ] **Step 7: Commit**

```bash
git add draftproof-frontend/src/pages/Technology.jsx draftproof-frontend/src/i18n/en/technologyPage.js draftproof-frontend/src/i18n/zh/technologyPage.js draftproof-frontend/src/i18n/resources.js
git commit -m "feat(technology): add /technology page content (unrouted)"
```

---

### Task 2: Routing, nav, SEO wiring + dead #engine link repair

**Files:**
- Modify: `draftproof-frontend/src/App.jsx:20-21` (import), `:121-124` area (routes)
- Modify: `draftproof-frontend/src/components/Header.jsx:16-23` (marketingLinks)
- Modify: `draftproof-frontend/src/i18n/en/nav.js`, `draftproof-frontend/src/i18n/zh/nav.js` (new `technology` key)
- Modify: `draftproof-frontend/src/localeRouting.js:4` (LOCALIZABLE_PUBLIC_PATHS)
- Modify: `draftproof-frontend/src/seoMetadata.js` (new `/technology` PAGE_META entry)
- Modify: `draftproof-frontend/src/i18n/en/seo.js`, `draftproof-frontend/src/i18n/zh/seo.js` (new `technologyTitle`/`technologyDescription`/`technologySocialDescription` keys)
- Modify: `draftproof-frontend/src/pages/Landing.jsx:311` (dead `#engine` footer link repair)
- Modify: `draftproof-frontend/src/pages/Why.jsx:64` (dead `#engine` CTA link repair)

**Interfaces:**
- Consumes: `Technology` default export from Task 1 (`draftproof-frontend/src/pages/Technology.jsx`).
- Produces: nothing consumed elsewhere — this is the final surface for this feature.

- [ ] **Step 1: Add the import and route pair in `App.jsx`**

In `App.jsx`, immediately after line 21 (`import Features from './pages/Features';`), add:
```js
import Technology from './pages/Technology';
```

Then, in the `<Routes>` block, immediately after the existing:
```jsx
            <Route path="/features" element={<Features />} />
            <Route path="/zh/features" element={<Features />} />
```
add:
```jsx
            <Route path="/technology" element={<Technology />} />
            <Route path="/zh/technology" element={<Technology />} />
```

- [ ] **Step 2: Add `nav.technology` to both i18n nav files**

In `draftproof-frontend/src/i18n/en/nav.js`, next to the existing `"features": "Features",` line, add:
```js
  "technology": "Technology",
```

In `draftproof-frontend/src/i18n/zh/nav.js`, next to the existing `"features": "功能对比",` line, add:
```js
  "technology": "技术",
```

- [ ] **Step 3: Add the nav link in `Header.jsx`**

In `Header.jsx`, the `marketingLinks` array currently reads:
```jsx
  const marketingLinks = [
    { to: publicPath('/#product'), label: t('nav.why') },
    { to: publicPath('/features'), label: t('nav.features') },
    { to: publicPath('/content-checker'), label: t('nav.essayChecker') },
    { to: publicPath('/pricing'), label: t('nav.pricing') },
    { to: publicPath('/faq'), label: t('nav.faq') },
    { to: publicPath('/#report'), label: t('nav.sampleReport') },
  ];
```
Add a new entry for Technology right after the `features` entry:
```jsx
  const marketingLinks = [
    { to: publicPath('/#product'), label: t('nav.why') },
    { to: publicPath('/features'), label: t('nav.features') },
    { to: publicPath('/technology'), label: t('nav.technology') },
    { to: publicPath('/content-checker'), label: t('nav.essayChecker') },
    { to: publicPath('/pricing'), label: t('nav.pricing') },
    { to: publicPath('/faq'), label: t('nav.faq') },
    { to: publicPath('/#report'), label: t('nav.sampleReport') },
  ];
```
This array only renders when `!user` (`visiblePublicLinks = user ? signedInPublicLinks : marketingLinks`, unchanged) — no further edit needed for the logged-out-only requirement.

- [ ] **Step 4: Add `/technology` to `LOCALIZABLE_PUBLIC_PATHS`**

In `draftproof-frontend/src/localeRouting.js:4`, currently:
```js
export const LOCALIZABLE_PUBLIC_PATHS = ['/', '/why', '/features', '/content-checker', '/turnitin-ai-score', '/academic-integrity-ai', '/ai-declaration', '/reduce-ai-detection', '/pricing', '/faq', '/privacy', '/security', '/signin'];
```
Change to (inserting `/technology` right after `/features`):
```js
export const LOCALIZABLE_PUBLIC_PATHS = ['/', '/why', '/features', '/technology', '/content-checker', '/turnitin-ai-score', '/academic-integrity-ai', '/ai-declaration', '/reduce-ai-detection', '/pricing', '/faq', '/privacy', '/security', '/signin'];
```

- [ ] **Step 5: Add SEO copy keys to `seo.js` (en + zh)**

In `draftproof-frontend/src/i18n/en/seo.js`, next to the existing `featuresSocialDescription` line, add:
```js
  "technologyTitle": "The Engineering Behind DraftProof — Tested for Fairness, Built to Last | DraftProof",
  "technologyDescription": "See the engineering practices behind every DraftProof report: multiple independent detection signals, fairness testing against real ESL student writing, fail-open guardrails, and ongoing recalibration against new AI writing tools.",
  "technologySocialDescription": "Multiple independent signals, tested for ESL fairness, fails open toward the student — the engineering behind every DraftProof report.",
```

In `draftproof-frontend/src/i18n/zh/seo.js`, add the equivalent zh keys next to the zh `featuresSocialDescription` line (find its exact line via `grep -n "featuresSocialDescription" draftproof-frontend/src/i18n/zh/seo.js` first, then insert immediately after it):
```js
  "technologyTitle": "DraftProof 背后的工程实践——为公平而测试,为长期可靠而打造 | DraftProof",
  "technologyDescription": "了解每一份 DraftProof 报告背后的工程实践:多个独立检测信号、针对真实 ESL 学生写作的公平性测试、安全失败的防护机制,以及针对新出现的 AI 写作工具持续进行的重新校准。",
  "technologySocialDescription": "多个独立信号、针对 ESL 公平性测试、在不确定时始终倾向学生——这是每一份 DraftProof 报告背后的工程实践。",
```

- [ ] **Step 6: Add the `/technology` `PAGE_META` entry**

In `draftproof-frontend/src/seoMetadata.js`, immediately after the existing `/features` entry (which ends `freshness: { type: 'reviewed', date: '2026-06-19' },\n  },`), add:
```js
  '/technology': {
    titleKey: 'seo.technologyTitle',
    descriptionKey: 'seo.technologyDescription',
    socialDescriptionKey: 'seo.technologySocialDescription',
    canonical: '/technology',
    schemaType: 'AboutPage',
    freshness: { type: 'reviewed', date: '2026-07-05' },
  },
```

- [ ] **Step 7: Repair the two dead `#engine` links**

In `draftproof-frontend/src/pages/Landing.jsx:311`, change:
```jsx
            <a href="#engine">{t('footer.howItWorks')}</a>
```
to:
```jsx
            <Link to={publicPath('/technology')}>{t('footer.howItWorks')}</Link>
```
(`Link` and `publicPath` are already imported/defined in this file — confirmed at `Landing.jsx:2` and `Landing.jsx:35`.)

In `draftproof-frontend/src/pages/Why.jsx:64`, change:
```jsx
            <Link to={publicPath('/#engine')} className="btn btn-secondary">{t('whyPage.ctaHow')}</Link>
```
to:
```jsx
            <Link to={publicPath('/technology')} className="btn btn-secondary">{t('whyPage.ctaHow')}</Link>
```

- [ ] **Step 8: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors; prerender step logs 36 routes now (34 previously + `/technology` + `/zh/technology`) — confirm the "Prerendered SEO metadata for N routes" log line increased by exactly 2 from the last known-good run.

- [ ] **Step 9: Visual verification via preview tool**

Start the dev server (reuse the `.claude/launch.json` config created during the earlier landing-page work if present, otherwise start one per this repo's CLAUDE.md: `npm run dev` on port 3000). Then:
- Navigate to `/technology`: confirm hero renders, all 5 pillar sections render in order with title/body/"why it matters" line, closing CTA renders with both buttons, `PageFreshness` renders at the bottom.
- Navigate to `/zh/technology`: confirm full Chinese localization, no English fallback text anywhere.
- Confirm the top nav shows "Technology" (or "技术" on zh) when logged out.
- Sign in (or simulate a logged-in state if the preview tool supports it) and confirm the "Technology" nav link disappears, matching the existing `marketingLinks` visibility rule.
- Navigate to `/` and `/why`, click the "How It Works" footer link and the Why page's "See a sample report"-equivalent CTA button respectively, and confirm both now land on `/technology` instead of a dead `#engine` anchor.
- Check console for errors on all pages visited (`mcp__Claude_Preview__preview_console_logs`, level `error`).

- [ ] **Step 10: Commit**

```bash
git add draftproof-frontend/src/App.jsx draftproof-frontend/src/components/Header.jsx draftproof-frontend/src/i18n/en/nav.js draftproof-frontend/src/i18n/zh/nav.js draftproof-frontend/src/localeRouting.js draftproof-frontend/src/seoMetadata.js draftproof-frontend/src/i18n/en/seo.js draftproof-frontend/src/i18n/zh/seo.js draftproof-frontend/src/pages/Landing.jsx draftproof-frontend/src/pages/Why.jsx
git commit -m "feat(technology): route, nav, SEO wiring + repair dead #engine links"
```

---

## Self-Review Notes

- **Spec coverage:** Placement & wiring (routes/nav/locale/SEO) → Task 2. Page content (hero + 5 pillars + CTA) → Task 1. Out-of-scope items (no Landing.jsx content change, no new CSS, no vendor names) → respected; the one Landing.jsx edit in Task 2 is the pre-existing dead-link repair, not new landing content. ✓
- **Placeholder scan:** no TBD/TODO; all i18n copy and JSX are complete, copy-pasteable blocks. (Caught and fixed a duplicate `"title"` key in the zh `technologyPage.js` draft during self-review — corrected in place rather than left as a step for the implementer to fix.) ✓
- **Type/name consistency:** `technologyPage.pillars` (array of `{title, body, whyItMatters}`) defined in Task 1 Step 2/3 and consumed identically in Task 1 Step 6's `Technology.jsx` (`pillar.title`/`pillar.body`/`pillar.whyItMatters`) — no drift. `Technology` default export name matches Task 2 Step 1's import statement exactly. ✓
