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
  `PRERENDER_PATHS` + sitemap) + register in `src/i18n/resources`.

## Overlap decisions (avoid keyword cannibalization)

| Plan page | Decision |
|---|---|
| `/turnitin-ai-score-explained` | **DROP** — `/turnitin-ai-score` already covers it; cross-link instead. |
| `/draftproof-vs-turnitin` | **BUILD** — distinct from `/turnitin-vs-ai-detectors` (stage difference, not detector-vs-detector). |
| `/check-essay-before-turnitin` | **BUILD** — flagship pre-submission page (plan §2). No existing equivalent. |
| `/turnitin-flagged-my-essay-ai` | **BUILD** — post-panic "why flagged" page (plan §3). No existing equivalent. |

## The 3 new pages

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
- **Intent**: `draftproof vs turnitin`, `turnitin alternative for students`.
- **Content**: the stage-difference comparison table from plan §5 via `section.type: 'comparison'`.
- **Guardrails**: state independence from Turnitin; do not imply affiliation; do not claim score
  reproduction/prediction.

## Cross-linking (semantic cluster, plan §4/§8)

Each new page's `links[]` points to the other two new pages + the most relevant existing pages
(`/turnitin-ai-score`, `/academic-integrity-ai`, `/ai-declaration`). Existing pages are left as-is
this phase (optionally back-link in a follow-up).

## FAQPage structured data (new capability)

Add `FAQPage` support to `buildSchema` (`src/seoMetadata.js`):
- New `schemaType: 'FAQPage'` branch that reads an optional `faq` array from the page's i18n
  namespace (`[{question, answer}]`) and emits `mainEntity` `Question`/`Answer` nodes alongside the
  existing Organization + WebSite `@graph`.
- The i18n `faq` content is provided per-page (placeholder in this phase).
- Applied to `/turnitin-flagged-my-essay-ai` (highest rich-result value). Guard: if no `faq`
  content, fall back to the generic WebPage schema (no empty FAQPage emitted).

## Copy handling

User provides final copy. This phase ships **clearly-marked placeholder copy** (`[DRAFT — replace]`
prefix on body text) so pages render and prerender correctly. EN and ZH namespace files both
scaffolded; ZH placeholders mirror EN structure with a `[需替换]` marker for the user's translator.

## Files touched

- **New**: `src/i18n/en/{checkBeforeTurnitin,turnitinFlagged,draftproofVsTurnitin}.js` (+ `zh/` twins)
- **New**: `docs/superpowers/specs/2026-07-23-turnitin-seo-cluster-design.md` (this file)
- **Edit**: `src/i18n/resources.js` (register 3 namespaces × 2 locales)
- **Edit**: `src/App.jsx` (3 route pairs)
- **Edit**: `src/localeRouting.js` (3 paths in `LOCALIZABLE_PUBLIC_PATHS`)
- **Edit**: `src/seoMetadata.js` (3 `PAGE_META` entries + FAQPage branch in `buildSchema`)
- **Edit**: `src/i18n/en/seo.js` + `zh/seo.js` (title/description keys for 3 paths)

## Verification

1. `npm run build` succeeds (frontend Vite build).
2. `scripts/prerender-seo.mjs` output includes the 3 new EN + 3 ZH paths and they appear in
   `dist/sitemap.xml` with hreflang alternates.
3. `/zh/check-essay-before-turnitin` etc. render the ZH namespace (locale-trap guard: paths present
   in `LOCALIZABLE_PUBLIC_PATHS`).
4. FAQPage JSON-LD validates (well-formed `mainEntity` with Question/Answer) on the panic page.
5. Positioning check: no "bypass/beat Turnitin" language; independence stated on the vs page.

## Out of scope (later phases)

- Phase 2 content hub (`/turnitin/` articles) — needs substantial long-form copy.
- Phase 3 off-site association (Reddit/YouTube/LinkedIn/AppSource) — not code.
- Phase 4 broader authorship queries.
- BreadcrumbList schema (low value at 1-level depth).
