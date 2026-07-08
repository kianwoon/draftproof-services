# Turnitin vs AI Detectors — Comparison Landing Page

_2026-07-08 — targets the "ai detector turnitin" / "best ai detector" search queries (currently zero SERP presence)._

## Problem

"draftproof" ranks page 1 (unique brand string, trivial). "ai detector turnitin" has zero presence:
no existing page addresses that comparison intent — `/turnitin-ai-score` answers "what does my % mean,"
not "how does Turnitin's detector compare to others."

## Page

- Slug: `/turnitin-vs-ai-detectors` (+ `/zh/turnitin-vs-ai-detectors`)
- Rendered by existing `SeoLandingPage.jsx` (ns-driven, no new page component)
- Detectors compared: Turnitin, GPTZero, Originality.ai, DraftProof (DraftProof shown as a
  different category — prep tool, not a rival detector — per the locked "never vs Turnitin"
  positioning pillar; see `project_key_message` memory)

## Sections (in order)

1. Hero — H1 targets the query directly, CTA into `/scan`
2. Intro — 2–3 blocks, "different category" framing stated up front
3. **Comparison table** (new `section.type === 'comparison'`) — rows: how it scores, known
   false-positive/bias caveats, audience, price model. Columns: the 4 above.
4. Narrative sections (`steps`/`cards`, existing types) — how each detector's method works,
   why detector scores disagree
5. FAQ cards — "Is a Turnitin AI score proof of cheating?" etc.
6. Related links — cross-link to/from `/turnitin-ai-score`, `/academic-integrity-ai`
7. CTA — into `/scan`

## Sourced facts (verified via web search 2026-07-08, cited informally in copy)

- **Turnitin**: BERT-based classifier; publishes 98% accuracy / <1% FPR claim for scores >20%;
  publishes higher FP rate under 20% (hence its own asterisk); not sole basis for misconduct
  action per Turnitin's own FAQ. Source: guides.turnitin.com AI writing detection model + FAQ.
- **GPTZero**: perplexity + burstiness foundation, now a 7-component proprietary model; claims
  99% accuracy; independent reporting (Washington Post 2023) flagged FP concerns; a Stanford
  study found 61.3% FPR on non-native-English essays; has since added ESL debiasing work.
- **Originality.ai**: 0–100% sentence-level score, 4 detection models, trained on 160GB text;
  ~83.4% detection rate / ~5% FPR per its own published claims; 28–35% false-negative rate on
  paraphrased/humanized AI content. Pricing: ~$14.95/mo base tier (2,000 credits) as of source
  date — copy states "check current pricing," no stale numbers hardcoded into prose as fact.
- Detector-disagreement-on-same-text is well-documented industry-wide behavior, not a
  DraftProof-specific claim — stated generally, not attributed to one tool.

No claim in copy exceeds what's sourced above; anything not verifiable states the mechanism only.

## Architecture change

`SeoLandingPage.jsx`: add `section.type === 'comparison'` branch rendering
`<table className="feat-table">` (reuses `Features.jsx`'s existing CSS) from a
`{ columns: [...], rows: [{ label, values: [...] }] }` shape in the i18n content — no new CSS file.

## Wiring checklist

- `PAGE_META['/turnitin-vs-ai-detectors']` in `seoMetadata.js`
- `en/turnitinVsDetectors.js` + `zh/turnitinVsDetectors.js`, registered in `en.js`/`zh.js`
- `seo.turnitinVsDetectorsTitle/Description/SocialDescription` in `en/seo.js` + `zh/seo.js`
- Routes in `App.jsx` (en + zh)
- **`LOCALIZABLE_PUBLIC_PATHS` in `localeRouting.js`** — documented trap, must not skip
- Cross-links added from `/turnitin-ai-score` and `/academic-integrity-ai` into the new page
- `docs/seo/keyword-map.md` updated — comparison moves from "long-tail spoke" to built entry

## Verification

- `npm run build` exits 0, route count +2, no missing-template errors
- Browser-render both locales (not grep) — `document.documentElement.lang`, H1, table rows present
- Regression-check `/turnitin-ai-score`, `/academic-integrity-ai`, `/reduce-ai-detection` still
  render correctly after the `SeoLandingPage.jsx` section-type addition
