# New public page: `/technology` — engineering-credibility showcase for students & educators

## Problem

The user wants to "showcase our technology stack" to convince both students and educators that DraftProof invested serious engineering effort toward "a new education era" — explicitly **without naming libraries, models, or vendors** (capability-level framing only).

This is deliberately a *different* audience decision than the earlier 2026-07-05 landing-page work (`docs/superpowers/specs/2026-07-05-landing-student-v7-refresh-design.md`), which kept the main landing page student-only and shelved a teacher-facing section. This new page explicitly targets **both** students and educators, but as a **separate page**, not a change to the student-only landing scroll — so the two decisions don't conflict.

## Design

### Placement & wiring

A new top-level marketing page at `/technology` (+ `/zh/technology`), following the exact pattern already used by `/why` (`Why.jsx`) and `/features` (`Features.jsx`) — confirmed via codebase investigation:

- **Route pair** — `App.jsx:121-124` currently declares:
  ```jsx
  <Route path="/why" element={<Why />} />
  <Route path="/zh/why" element={<Why />} />
  <Route path="/features" element={<Features />} />
  <Route path="/zh/features" element={<Features />} />
  ```
  Add an identical pair for `Technology`.
- **New component** — `draftproof-frontend/src/pages/Technology.jsx`. Follows `Why.jsx`'s shape: hero section (`app-hero app-hero-dark` + `CodeTexture`), then a `.map()` over a `pillars` array (5 items) each rendering as its own `<section>`, then a closing CTA section. No new CSS primitives — reuses `why-section`/`why-hero`/`why-cta`-equivalent classes already defined for the Why page (confirm exact class names against `Why.jsx`/its CSS partial when implementing; if none reusable 1:1, extend the same partial rather than inventing a new one).
- **i18n namespace** — new `i18n/en/technologyPage.js` + `i18n/zh/technologyPage.js`, mirroring the `whyPage.*` convention (own file, own top-level key, not folded into `landing.js`).
- **Nav link** — `Header.jsx`'s `marketingLinks` array (lines 16-23) gets one new entry: `{ to: publicPath('/technology'), label: t('nav.technology') }`. This array is already only rendered for logged-out visitors (`visiblePublicLinks = user ? signedInPublicLinks : marketingLinks`, line 26) — no new conditional needed, the "non-signed-in only" requirement is inherited for free.
- **`nav.technology`** key added to `i18n/en/nav.js` + `i18n/zh/nav.js`.
- **Locale-safety (mandatory)** — `localeRouting.js:4`'s `LOCALIZABLE_PUBLIC_PATHS` array must include `/technology`, or `/zh/technology` silently renders the English body (documented locale-trap failure mode in this codebase).
- **SEO/prerender (mandatory)** — `seoMetadata.js`'s `PAGE_META` object must get a `/technology` entry (title/description per locale). `PRERENDER_PATHS` derives from `PAGE_META`'s keys automatically — no separate route list to edit in `prerender-seo.mjs`.

### Page content

**Hero:**
- Eyebrow: "The Technology Behind DraftProof"
- Title: "We didn't build another AI detector. We built the engineering discipline a new era of education actually needs."
- Lead: "Every AI writing checker makes a promise. Very few show their work. Here's what actually runs behind every DraftProof report — the practices we test, the guardrails we enforce, and the fairness bar we refuse to drop below."

**Five pillar sections** (each: title/claim, one short body paragraph, one "why it matters" line addressing both audiences). Every claim below is traceable to a real shipped practice — nothing invented:

1. **Multi-signal fusion** — "No single black-box score decides anything." Multiple independent detection signals (pattern-based analysis + a separate deep-reading model) are combined before any tier is shown, so one miscalibrated signal can't swing a verdict alone. *Traces to: composite detector + deep-scan detector fusion, `poc/detect_v7/detector_fusion.py` / tier-authority fused score, live in production.*
2. **Tested for ESL fairness** — "Validated against real multilingual student writing, not just native-English samples." Before any detection change ships, it's run against a dedicated corpus of real student essays across proficiency levels, and blocked from release if it raises false-positive rates for ESL writers. *Traces to: the mandatory ESL false-positive gate (`poc/calibration/fpr_subgroup_gate.py`, SCoCESLE corpus), enforced by the pre-push hook per this repo's CLAUDE.md.*
3. **Fails open toward the student** — "When a signal is missing or uncertain, the system never guesses against you." Every added detection layer fails safe to the established, calibrated score rather than inventing a harsher one when a signal is unavailable. *Traces to: fail-open behavior in `poc/detect_v7/pipeline_bridge.py` (falls back to composite score if deep-scan is absent) and the guard-tiering philosophy (fatal vs advisory blockers) elsewhere in the rewrite pipeline.*
4. **Recalibrated, not frozen** — "Re-tested against new AI writing tools as they appear." Detection signals are re-validated against outputs from multiple, independently-developed AI writing tools on an ongoing basis. *Traces to: the 2026-07-04 "Layer 2 diversity check" (`poc/calibration/v7_deberta_diversity_check.py`, tested against 3 additional AI-generator families beyond the original calibration set).*
5. **A disclosed, versioned scale** — "You see the calibration, not just a percentage." Every score ties to a labeled band with an explanation of what's driving it; the calibration is documented and versioned. *Traces to: the tier/band system (green/amber/orange/red, `weights.json` cutoffs) and the "shares = composition not AI-probability" subtitle already shown on real reports.*

**Closing CTA:** "This is what 'best stack' means to us: proof, not adjectives." No tool can promise a perfect verdict on every draft — what DraftProof can promise is a system that's tested, fails toward fairness, and keeps being re-checked as AI keeps changing. CTA button links to `/signin?next=/scan` (same conversion target as the landing page's primary CTA).

## Out of scope (explicitly)

- No change to the main landing page (`Landing.jsx`) — the earlier decision to keep that page student-only stands. This page is additive and separate.
- No literal library/model/vendor names anywhere in the copy (e.g. no "Deberta," "Modal," "Cerebras," "gpt-oss") — capability-level framing only, per the user's explicit instruction.
- No new visual design system — reuses existing Why/Features page CSS conventions.
- No backend changes — this is a static content page describing existing, already-shipped backend behavior.

## Testing / verification

- `npm run build:client` compiles clean (no test framework exists for this frontend, consistent with the prior landing-page work).
- Visual check via preview tool: `/technology` and `/zh/technology` both render all 5 pillars + hero + CTA, nav shows "Technology" only when logged out, no console errors.
- Grep confirms `/technology` present in `LOCALIZABLE_PUBLIC_PATHS` and `PAGE_META`.
- Manual click-through: nav link disappears when signed in (matches `marketingLinks` visibility rule).
