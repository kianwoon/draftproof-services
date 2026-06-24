# DraftProof SEO Keyword Map

_Last reviewed: 2026-06-24_

Prioritized keyword → page plan, grounded in (a) the existing prerender/landing-page
architecture and (b) live SERP observation. **No fabricated search volumes** — exact
volume/difficulty needs Ahrefs / SEMrush / Google Search Console. Difficulty below is
qualitative, derived from who actually ranks today.

## How a new SEO page ships (the proven mechanism)

`/content-checker` (`EssayChecker.jsx`) is the template. To add a page:

1. Add an entry to `PAGE_META` in `draftproof-frontend/src/seoMetadata.js` (canonical, schemaType, freshness).
2. Add the page component under `src/pages/` + route in `App.jsx`.
3. Add `seo.*Title` / `seo.*Description` keys to `src/i18n/en/seo.js` **and** `src/i18n/zh/seo.js`.
4. `npm run build` runs `scripts/prerender-seo.mjs` → static HTML per route + auto-updates `sitemap.xml`.

Everything is bilingual (en + `/zh`) and gets JSON-LD + hreflang for free.

## Non-negotiable content guard (alignment)

Every page keeps the **ugly external reality visible** (false positives, "still flagged",
detectors are noisy). DraftProof is a grounding/integrity coach, **not** a Turnitin-beater.
No "beat the detector" promises. See `docs/draftproof_alignment_principles.md`.

---

## Tier 1 — Build now (winnable + high intent + on-brand)

| Keyword cluster | Intent | Target page | Status | Competition (observed) |
|---|---|---|---|---|
| Turnitin AI percentage meaning · AI score explained · what does X% AI mean | Informational, high-panic → high convert | **NEW `/turnitin-ai-score`** | gap | Medium — thin essay-mill/AI-tool blogs + Turnitin docs. Beatable with honest authority. |

**Why first:** highest-intent visitor in the whole list — they have a *flagged document right now*.
The honest angle (the <20% asterisk, false positives, "not proof of misconduct") is exactly
DraftProof's voice AND what Turnitin itself says. Natural CTA: "scan your doc to see *which*
paragraphs drive the score."

## Tier 2 — Build next (on-brand, moderate difficulty)

| Keyword cluster | Intent | Target page | Status | Competition (observed) |
|---|---|---|---|---|
| AI declaration examples · how to declare AI use in an essay | Informational + tool | **NEW `/ai-declaration`** (examples + generator) | gap | High — universities own the content. Differentiate with a *template generator* they don't offer. |
| Academic integrity AI · responsible AI writing | Top-funnel, broad | **NEW `/academic-integrity-ai`** (pillar) | gap | High — universities/edu orgs. Role = pillar hub that internally links every spoke. |

## Tier 3 — Intercept the landmine (do carefully, lower priority)

| Keyword cluster | Intent | Target page | Status | Competition (observed) |
|---|---|---|---|---|
| How to reduce AI detection | High volume, **off-audience** (evasion-seekers) | **NEW `/reduce-ai-detection`** — reframe: "why that's the wrong goal" | gap | Very high — humanizer tools, QuillBot. Hard to rank. |

**Handle as interception, not capitulation.** Capture the search, redirect intent to grounding +
declaration. Never green-wash. This is the "expose the ugly side" principle as a growth tactic —
content competitors literally can't write. Lower priority because it's hard to rank and easy to get wrong.

## Already covered — optimize, don't rebuild

| Keyword | Maps to | Note |
|---|---|---|
| AI detector checker | `/` + `/content-checker` | Head term; brutal competition (Turnitin, GPTZero, Originality, ZeroGPT). Won't rank near-term — keep as brand anchor, chase long-tail instead. |

## Long-tail spokes — your real near-term traffic (lower competition)

Add as sections within the Tier-1/2 pages, or as their own thin pages later:

- `what does 30% AI on Turnitin mean` / `is 20% AI detection bad`
- `how to write an AI declaration for university`
- `Turnitin AI detection false positive`
- `is using Grammarly / ChatGPT academic misconduct`
- **Comparison (bottom-funnel, converts best):** `DraftProof vs Originality.ai`, `best AI integrity checker for students`

---

## Suggested build order

1. `/turnitin-ai-score` (Tier 1 — fastest win)
2. `/academic-integrity-ai` pillar (Tier 2 — anchors the cluster, links everything)
3. `/ai-declaration` + generator (Tier 2 — utility differentiation)
4. `/reduce-ai-detection` reframe (Tier 3 — careful interception)

## Open items (need real data, not guesses)

- Pull exact volume/difficulty from Ahrefs/SEMrush or Google Search Console before final prioritization.
- Verify Google Search Console is connected (Bing is — `public/BingSiteAuth.xml`).
- Decide canonical slug for the Turnitin page (`/turnitin-ai-score` vs `/ai-score-explained`).
