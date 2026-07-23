# DraftProof Turnitin SEO Cluster — Phase 1 Design

**Date:** 2026-07-23
**Source plan:** `draftproof_search_association_plan.md`
**Scope decision (user-approved):** Phase 1 only; I build structure/wiring, user provides final copy; 3 net-new pages + FAQPage schema support.

## Objective

Expand DraftProof's search footprint around Turnitin-intent queries, positioning it as
**"Student Authorship & Submission Readiness"** rather than a generic AI detector. Build the
on-site pages that capture pre- and post-Turnitin student searches, wired into the existing
data-driven SEO system.

## Codebase-first findings (what already exists)

- **Data-driven SEO pages**: one shared component `src/pages/SeoLandingPage.jsx` renders any
  page from an i18n namespace (`title, eyebrow, lead, heroStat, intro[], sections[], links[]`).
  `sections[]` already supports `type: 'comparison'` (table), `steps`, `templates`, default grid.
- **Structured data already shipped**: `buildSchema` (`src/seoMetadata.js:238-315`) emits
  `SoftwareApplication` + `Organization` + `WebSite` JSON-LD on every page. **Plan Phase 1 item 5
  (add SoftwareApplication/Organization) is already done** — no work needed.
- **Existing Turnitin pages** (do NOT duplicate): `/turnitin-ai-score` (turnitinScore),
  `/turnitin-vs-ai-detectors` (turnitinVsDetectors), `/turnitin-alternatives`,
  `/academic-integrity-ai`, `/ai-declaration`, `/reduce-ai-detection`.
- **Adding a page = data only**: i18n namespace pair + route pair + `LOCALIZABLE_PUBLIC_PATHS`
  entry (`src/localeRouting.js:4`) + `PAGE_META` entry (`src/seoMetadata.js:40`, auto-adds to
  `PRERENDER_PATHS` + sitemap) + register in `src/i18n/en.js` + `src/i18n/zh.js` (import the ns file
  and add its key to `enTranslation` / `zhTranslation`; `resources.js` only re-exports these
  aggregates — editing it alone does NOT register a namespace).

## Overlap decisions (avoid keyword cannibalization)

| Plan page | Decision |
|---|---|
| `/turnitin-ai-score-explained` | **DROP** — `/turnitin-ai-score` already covers it; cross-link instead. |
| `/draftproof-vs-turnitin` | **BUILD** — distinct from `/turnitin-vs-ai-detectors` (stage difference, not detector-vs-detector). |
| `/check-essay-before-turnitin` | **BUILD** — flagship pre-submission page (plan §2). No existing equivalent. |
| `/turnitin-flagged-my-essay-ai` | **BUILD** — post-panic "why flagged" page (plan §3). No existing equivalent. |

## The 3 new pages

**Build/priority order = acquisition value** (numbering below is the order to build and the expected
traffic value). #1 pre-submission is the near-perfect DraftProof entry point; #2 post-flag maps to
the authorship/grounding/defence-readiness direction; #3 branded-comparison is conversion-support
and entity definition (user usually already knows DraftProof), so it is the lowest early-traffic bet
but still worth building.

### 1. `/check-essay-before-turnitin` — namespace `checkBeforeTurnitin`
- **Intent**: `check essay before turnitin`, `check my essay before submitting`.
- **Positioning**: "Understand the risks your lecturer may see before you submit" — NOT "see what
  Turnitin will score."
- **Sections**: the 8-risk flow from plan §2 (AI-pattern, authorship clarity, AI-paraphrasing,
  consistency, citation/grounding, argument weakness, policy/declaration, defence readiness) as a
  `steps` or grid section; CTA to `/signin?next=/scan`.

### 2. `/turnitin-flagged-my-essay-ai` — namespace `turnitinFlagged`
- **Intent**: `turnitin flagged my essay as ai`, `turnitin says my essay is ai`, `turnitin false positive`.
- **Positioning**: "Find out why your writing may be questioned and strengthen evidence of
  authorship" — NOT "humanize to bypass."
- **Sections**: why human writing triggers detectors, what a % means, what to do, how to demonstrate
  authorship. **FAQPage schema** attached (see below).

### 3. `/draftproof-vs-turnitin` — namespace `draftproofVsTurnitin`
- **Intent**: strictly branded comparison — `draftproof vs turnitin`. **Do NOT target
  `turnitin alternative for students`** — that cannibalizes the existing `/turnitin-alternatives`.
- **Content**: the stage-difference comparison table from plan §5 via `section.type: 'comparison'`.
- **Guardrails**: state independence from Turnitin; do not imply affiliation; do not claim score
  reproduction/prediction.

## Cross-linking (semantic cluster, plan §4/§8)

**Outbound** — each new page's `links[]` points to the other two new pages + the most relevant
existing pages (`/turnitin-ai-score`, `/academic-integrity-ai`, `/ai-declaration`).

**Inbound (mandatory this phase — not deferred).** Add at least one inbound link from an existing
indexed page to every new page BEFORE indexing, so the cluster has authority from day one. Edit the
existing i18n namespace files' `links[]` (EN + ZH):

| From (existing) | To (new) | Anchor |
|---|---|---|
| `/turnitin-ai-score` (`turnitinScore`) | `/turnitin-flagged-my-essay-ai` | "What to do if Turnitin flags your writing as AI" |
| `/turnitin-alternatives` (`turnitinAlternatives`) | `/draftproof-vs-turnitin` | "How DraftProof differs from Turnitin" |
| `/academic-integrity-ai` (`academicIntegrity`) *or* `/ai-declaration` | `/check-essay-before-turnitin` | "Review authorship and AI-use risks before submission" |

Note: inbound links live on already-indexable pages, so they go live immediately — that is fine and
desired; the *target* new pages stay `noindex` until copy lands (a followed link to a noindex page is
normal and harmless).

## FAQPage structured data (new capability)

Add `FAQPage` support to `buildSchema` (`src/seoMetadata.js:238`):
- New `schemaType: 'FAQPage'` branch. `buildSchema(meta, url, translate)` has **no direct namespace
  access**, so the page declares a `faqKey` on its `PAGE_META` entry (e.g. `faqKey: 'turnitinFlagged.faq'`)
  and the branch resolves it via the module's existing `getResourceValue(faqKey, meta.locale)` helper
  (`seoMetadata.js:377` — its `.reduce` traverses arrays, so `<ns>.faq` returns the array).
- The resolved `faq` array is `[{question, answer}]`; emit `mainEntity` `Question`/`Answer` nodes
  **inside the existing `@graph`** alongside Organization + WebSite (same shape as the WebPage branch).
- Applied to `/turnitin-flagged-my-essay-ai` (highest relevance). Guard: if `faqKey` is absent or
  resolves empty, fall back to the generic WebPage schema (no empty FAQPage emitted).
- **Expectation note**: Google restricted FAQ rich-result *display* to authoritative gov/health
  sites (Aug 2023); the schema is still valid and machine-readable but may not render as a snippet.
- **Priority framing**: treat FAQPage as reusable *infrastructure*, NOT a ranking lever. The real
  levers, in order: search-intent-matched content ≈ internal links > titles/H1/metadata ≈
  indexability ≫ FAQPage schema. Build it once, don't over-invest.

## Copy handling & indexing guard

User provides final copy. This phase ships **clearly-marked placeholder copy** (`[DRAFT — replace]`
prefix on body text) so pages render and prerender correctly. EN and ZH namespace files both
scaffolded; ZH placeholders mirror EN structure with a `[需替换]` marker for the user's translator.

**Do not index placeholder pages, but keep link traversal.** Each new `PAGE_META` entry ships this
phase with `robots: 'noindex'` — **not** `'noindex, nofollow'`. `nofollow` would stop engines from
following the cluster's internal links, which defeats the semantic-cluster purpose; there is no
benefit to it. The user flips to indexable by deleting the `robots` line once final copy lands.

**Sitemap already does the right thing — no code change.** `renderSitemap` filters out any page
whose robots contains `noindex` (`prerender-seo.mjs:71`, `/\bnoindex\b/i`). So while placeholder:
render YES, accessible YES, `noindex` YES, **sitemap NO** — automatically. Once `noindex` is removed:
sitemap YES with hreflang/x-default alternates. `'noindex'` alone still matches the filter.

Each entry also gets its own **fresh `freshness.date` = 2026-07-23** (a new constant
`TURNITIN_CLUSTER_REVIEW_DATE`), NOT the stale `SEO_LANDING_REVIEW_DATE` (2026-06-24), because that
date drives sitemap `<lastmod>` once the page is indexable.

Each `PAGE_META` entry includes `titleKey`, `descriptionKey`, and `socialDescriptionKey` (all sibling
entries define the social key; it falls back to description only if omitted).

## Files touched

- **New**: `src/i18n/en/{checkBeforeTurnitin,turnitinFlagged,draftproofVsTurnitin}.js` (+ `zh/` twins)
- **New**: `docs/superpowers/specs/2026-07-23-turnitin-seo-cluster-design.md` (this file)
- **Edit**: `src/i18n/en.js` + `src/i18n/zh.js` (import 3 ns files each, add keys to
  `enTranslation` / `zhTranslation` — NOT `resources.js`)
- **Edit**: `src/App.jsx` (3 route pairs)
- **Edit**: `src/localeRouting.js` (3 paths in `LOCALIZABLE_PUBLIC_PATHS`)
- **Edit**: `src/seoMetadata.js` (3 `PAGE_META` entries incl. `robots: noindex` + `socialDescriptionKey`
  + `faqKey` on the panic page; new `TURNITIN_CLUSTER_REVIEW_DATE`; FAQPage branch in `buildSchema`)
- **Edit**: `src/i18n/en/seo.js` + `zh/seo.js` (title/description/socialDescription keys for 3 paths)
- **Edit (inbound links)**: existing namespace files `src/i18n/en/{turnitinScore,turnitinAlternatives,academicIntegrity}.js`
  (+ `zh/` twins) — add one `links[]` entry each per the cross-linking table.

## Measurement (Phase 1 — decides Phase 2)

Do not scale to more Turnitin articles until real Google Search Console data shows which query
families Google associates with DraftProof.

**Per-page primary intent family** (secondary families also tracked):

| Page | Primary intent |
|---|---|
| `/check-essay-before-turnitin` | check essay before Turnitin |
| `/turnitin-flagged-my-essay-ai` | Turnitin flagged essay as AI |
| `/draftproof-vs-turnitin` | DraftProof vs Turnitin |

**GSC per URL** (user-side ops, once indexable): index status, impressions, queries, avg position, CTR.

**Product funnel** (existing GA4, `GA_MEASUREMENT_ID`): SEO landing → CTA click → sign-in → scan
started → scan completed. The CTA already routes to `/signin?next=/scan` (funnel entry exists via
`SeoLandingPage`). Wiring explicit GA events on the CTA is a *recommended follow-up*, not part of this
page-build phase.

## Verification

1. `npm run build` succeeds (frontend Vite build).
2. `npm run check:i18n` passes (en/zh namespace parity gate — all 3 new namespaces must mirror).
3. `scripts/prerender-seo.mjs` prerenders the 3 new EN + 3 ZH paths successfully, but they are
   **excluded from `dist/sitemap.xml` while `noindex`** (`prerender-seo.mjs:71` filter). After final
   copy and removal of `robots: 'noindex'`, all six localized URLs appear in the sitemap with auto
   hreflang/x-default alternates (`prerender-seo.mjs:96-109`, no manual work).
4. `/zh/check-essay-before-turnitin` etc. render the ZH namespace (locale-trap guard: paths present
   in `LOCALIZABLE_PUBLIC_PATHS`).
5. FAQPage JSON-LD validates (well-formed `mainEntity` with Question/Answer) on the panic page.
6. Positioning check: no "bypass/beat Turnitin" language; independence stated on the vs page;
   category held as "Student Authorship & Submission Readiness" (not "AI detector").
7. Inbound links: each new page has ≥1 inbound link from an existing indexed page (per the
   cross-linking table), live in both EN and ZH.

## Out of scope (later phases)

- Phase 2 content hub (`/turnitin/` articles) — needs substantial long-form copy.
- Phase 3 off-site association (Reddit/YouTube/LinkedIn/AppSource) — not code.
- Phase 4 broader authorship queries.
- BreadcrumbList schema (low value at 1-level depth).
