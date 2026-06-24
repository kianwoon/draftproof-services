# SEO Landing Pages — Design Spec

_2026-06-24 · approved in-session_

## Goal

Close the informational-keyword gap (see `docs/seo/keyword-map.md`) by adding 4 bilingual
SEO landing pages, reusing the existing prerender/landing-page architecture.

## Pages

| Slug | Namespace | Targets |
|---|---|---|
| `/turnitin-ai-score` | `turnitinScore` | "Turnitin AI percentage meaning", "AI score explained" |
| `/academic-integrity-ai` | `academicIntegrity` | "academic integrity AI", "responsible AI writing" (pillar) |
| `/ai-declaration` | `aiDeclaration` | "AI declaration examples" (static templates) |
| `/reduce-ai-detection` | `reduceDetection` | "how to reduce AI detection" (reframe / interception) |

## Architecture

One shared renderer `src/pages/SeoLandingPage.jsx` driven by an i18n namespace + base path,
rendering the proven EssayChecker section flow: hero → 2-col intro → N content sections
(`cards` / `steps` / `templates`) → related-links block → CTA → `PageFreshness`.

Per page, additive only:
1. `src/i18n/en/<ns>.js` + `src/i18n/zh/<ns>.js` — all copy
2. Register both in `src/i18n/en.js` / `zh.js`
3. `seo.<ns>Title/Description/SocialDescription` in `en/seo.js` + `zh/seo.js`
4. `PAGE_META['/slug']` in `seoMetadata.js` (→ auto-prerender + auto-sitemap)
5. `/slug` + `/zh/slug` routes in `App.jsx`
6. Reuse `content-checker-*` CSS + new `seo-template-*` / `seo-related-*` rules

## Content guard (binding)

Every page keeps the ugly external reality visible (false positives, "still flagged",
detector noise). DraftProof is a grounding/integrity coach, **not** a Turnitin-beater. No
humanizer/evasion promises. Declaration templates patterned on real Elsevier/Leeds formats —
generic, no fabricated specifics. See `docs/draftproof_alignment_principles.md`.

## Internal linking

`/academic-integrity-ai` (pillar) ↔ all 3 spokes; `/turnitin-ai-score` ↔ `/reduce-ai-detection`;
every page → `/content-checker` + `/why`.

## Verify

`npm run build` (runs `scripts/prerender-seo.mjs`) → 8 new URLs in `sitemap.xml`, no
"Unable to find SEO template target" error, eyeball one rendered page.

## Out of scope (YAGNI)

Generator tool for `/ai-declaration`, blog infra, exact-volume keyword tooling — tracked as
follow-ups in `docs/seo/keyword-map.md`.
