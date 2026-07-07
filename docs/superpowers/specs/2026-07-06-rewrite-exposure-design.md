# Rewrite exposure: /rewrite marketing page + landing before/after teaser

## Problem

The rewrite function has almost no exposure (verified via a full entry-point map):
- No route/nav/footer/dashboard entry of its own. The only route (`/rewrite/:rewriteId`, `App.jsx:159`, protected) merely *views* a finished rewrite; you cannot land on it fresh.
- The only way to *start* a rewrite is a button buried in the Repair box of a completed scan report (`ReportHero.jsx:104`), shown only when the scan returned AI findings (`Report.jsx:1240`).
- To logged-out visitors, rewrite is invisible — sold only as post-scan sub-bullets, never a linkable destination.

Owner picked two exposure tracks to build now: **Track A** (a visual before/after demo on the landing page) and **Track B** (a dedicated `/rewrite` marketing page + nav link). Tracks C (in-app entry points) and D (funnel/pricing) are deferred.

## Design

**Structure:** landing **teases** → `/rewrite` page **delivers**. One rich demo lives on the page; a compact teaser on the landing page links to it. No duplicated demo.

### 1. New `/rewrite` marketing page

New component `draftproof-frontend/src/pages/RewriteOverview.jsx` (named to avoid collision with the existing protected viewer `Rewrite.jsx`). Follows the exact `Why.jsx`/`Technology.jsx` structural pattern (hero + mapped sections + closing CTA), reusing existing `why-*` / `app-hero` / `hero-actions` CSS classes — **no new visual system, no new CSS**.

**Path:** `/rewrite` (+ `/zh/rewrite`). Safe alongside the existing protected `/rewrite/:rewriteId`: bare `/rewrite` is currently unused, and React Router matches `/rewrite` (exact) vs `/rewrite/:rewriteId` (param) distinctly. Semantics: `/rewrite` = "what the rewrite is" (public), `/rewrite/<id>` = "your specific rewrite" (protected).

**Sections, top-to-bottom:**
1. **Hero** — eyebrow "The DraftProof Rewrite"; title "See what grounded writing looks like — then make it yours."; lead explaining the scan → flag → rewrite flow (honest about scan-first).
2. **3-example before/after demo** (centerpiece). Three worked examples, each a different grounding fix the real rewrite performs, each ending with the bracketed **"Suggested Addition For Review"** marker (the honest differentiator — added content is marked for the user to replace, not inserted as final):
   - Ex 1 — unsupported claim → adds `[citation you replace]`
   - Ex 2 — vague generality → adds `[specific example you replace]`
   - Ex 3 — no reasoning → adds `[your own "why" you replace]`
   Rendered with a before/after diff structure reusing the landing page's existing `hero-diff-*` visual convention (Before row tone=remove, After row tone=add) or an equivalent simple two-row card — no new CSS primitives.
3. **"What it is / what it isn't"** — reuses `rewriteFraming.js` copy verbatim ("DraftProof mitigates AI-detection risk and shows you a reviewable draft to learn from"; "It is not a 'make it pass' button. A residual estimate is expected…").
4. **3 value cards** — adapted from the existing `features.js` `rewriteCards` (Auto before/after rewrite; A solution to learn from; Before/after diff you can act on).
5. **CTA** — links to `/signin?next=/scan`, labeled honestly ("Start a scan" / "Run your first scan") because rewrite begins from a scan. Plus `PageFreshness path="/rewrite"`.

**i18n:** new namespace file `i18n/en/rewriteOverview.js` + `i18n/zh/rewriteOverview.js`, registered in `i18n/en.js`/`i18n/zh.js` (the real registration files — NOT `resources.js`, which does not carry per-namespace logic in this codebase), mirroring how `technologyPage` is registered. All demo copy is capability-level, no vendor/model names, no raw dataset numbers.

### 2. Landing before/after teaser (`Landing.jsx`)

A compact **standalone section** (not a slide in the auto-rotating carousel — standalone is more prominent, which is the point), placed immediately after the existing report/strategy carousel section (natural flow: report shows the problem → rewrite shows the fix). Contents:
- One before/after card on the **Hollywood/cultural-export sample paragraph** already threaded through the landing page (sample-report + critical-thinking tabs) — so a visitor sees one coherent essay go scan → rewrite.
- A "See how the rewrite works →" link to `publicPath('/rewrite')`.
Reuses existing landing section/card classes; no new CSS.

### 3. Wiring (identical pattern to the shipped `/technology` page)

- `App.jsx`: import `RewriteOverview`; add route pair `/rewrite` + `/zh/rewrite` (placed near the other public marketing routes, NOT inside the protected block; must not shadow `/rewrite/:rewriteId`).
- `Header.jsx` `marketingLinks`: add `{ to: publicPath('/rewrite'), label: t('nav.rewrite') }` (logged-out-only, inherited from the existing `visiblePublicLinks` pattern — no new conditional).
- `Footer.jsx`: add `<Link to={publicPath('/rewrite')}>{t('footer.rewrite')}</Link>` in the product nav list (near `content-checker`). NOTE: do not touch `Footer.jsx:22`'s pre-existing dead `/#engine` link — it's owned by a separate in-flight task (`task_e0e2c5a4`).
- `localeRouting.js`: add `/rewrite` to `LOCALIZABLE_PUBLIC_PATHS`.
- `seoMetadata.js`: add a `/rewrite` `PAGE_META` entry (title/description/social keys, `canonical: '/rewrite'`, `schemaType: 'WebPage'`, a `freshness` reviewed date) — this also auto-feeds the prerender sitemap.
- `i18n/en/nav.js` + `i18n/zh/nav.js`: new `nav.rewrite` key.
- `i18n/en/footer.js` (or wherever `footer.*` lives) + zh: new `footer.rewrite` key.
- `i18n/en/seo.js` + zh: new `rewriteTitle`/`rewriteDescription`/`rewriteSocialDescription` keys.
- `Features.jsx` rewrite tab: add a "Learn more about the rewrite →" link to `publicPath('/rewrite')` so the existing rewrite tab isn't orphaned from the new destination.

## Alignment guardrails (hard — from the alignment principles)

All copy stays: "teaching draft / grounded writing / finish in your own words / complement Turnitin / prepare before submission." **Never** "make it undetectable / humanizer / bypass / beat the detector." The bracketed "Suggested Addition For Review — you replace this" mechanic is foregrounded precisely to prove the rewrite is not a make-it-pass button. The CTA is honest that rewrite starts from a scan.

## Out of scope (explicitly)

- No change to the real rewrite pipeline, the protected `/rewrite/:rewriteId` viewer, or `Rewrite.jsx`.
- No funnel/pricing/gating changes (Track D) — scan-first dependency, findings-gate, and 5× cost are unchanged.
- No in-app nav entry for rewrite (Track C) — deferred.
- No fix to `Footer.jsx`'s dead `/#engine` link (separate task).
- No new CSS system — reuse existing classes only.

## Testing / verification

- `npm run build:client` compiles clean; prerender route count increases by exactly 2 (`/rewrite` + `/zh/rewrite`).
- Grep confirms `/rewrite` present in `LOCALIZABLE_PUBLIC_PATHS` and `PAGE_META`, and every new i18n key exists in both en and zh.
- Preview tool: `/rewrite` and `/zh/rewrite` render hero + 3-example demo + framing + value cards + CTA, fully localized on zh, no console errors; nav "Rewrite" link appears logged-out and is hidden signed-in; landing teaser section renders with a working link to `/rewrite`; the existing protected `/rewrite/:rewriteId` route still resolves (not shadowed).
