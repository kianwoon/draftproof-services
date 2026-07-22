# Site Content Review — Findings Report (2026-07-22)

**Method:** 4 parallel Sonnet reviewer agents, each covering a slice of the English site copy
(`draftproof-frontend/src/i18n/en/` + `seoMetadata.js`), read-only, findings anchored to exact
file:line. MEDIUM anchors spot-verified by the orchestrator.

**Review objectives:**
1. NOT positioned to beat/bypass the Turnitin AI detector ("complement Turnitin, never we beat them").
2. NOT helping students cheat or act against school policies (no concealment / "undetectable" framing;
   rewrite = reviewable teaching draft, not a final submission).
3. SHOULD help students strengthen their own work (grounding, specificity, real authorship,
   learning from before/after diffs).

---

## Overall verdict

**The site is strongly compliant with all three objectives. Zero HIGH findings across ~26 copy files.**
Copy repeatedly and explicitly disavows "beating," "bypassing," and "out-scoring" detectors; the
rewrite is consistently framed as a reviewable teaching draft; `legal.js` and `academicIntegrity.js`
actively prohibit evasion use. The highest-risk-sounding page (`reduceDetection.js`) is in fact the
most rigorously self-aware file on the site.

**2 MEDIUM findings** (out-of-context readability risks) and **7 LOW polish nits** remain.

> **STATUS 2026-07-23: ALL FINDINGS FIXED** (M1, M2, L1–L6, L8 — L7 needed no change) in both
> `en` and `zh` locales — 16 files, verified by full frontend build (42/42 routes prerendered).
> New copy: footer link → "The truth about AI detection" / "AI 检测的真相"; pricing FAQ →
> "Can DraftProof guarantee a Turnitin result?"; report confirm card → "Declare what you can
> honestly show"; "can't *reliably* beat" → "beating it is the wrong goal" (reduceDetection hero
> + seo description); "violation if you're caught" → "violation — detected or not"; rewriteFraming
> leads with grounding not detection-risk; essayChecker SEO adds "in your own words"; detector
> comparison title adds grounded-writing callout; dashboard re-scan step reframed from
> "check whether the risk is lower" to "confirm the flagged gaps are fixed".

---

## MEDIUM findings (recommend copy tweaks)

| # | Anchor | Passage | Objective | Issue & suggested direction |
|---|--------|---------|-----------|------------------------------|
| M1 | `i18n/en/footer.js:12` | `"reduceDetection": "Reduce AI detection"` | 1/2 | Bare footer link label reads as a score-lowering promise; the debunking frame lives only on the destination page (`seo.js`: "…and Why That's the Wrong Goal"). Reword the visible label itself, e.g. "The truth about AI detection" or "Reduce submission risk". |
| M2 | `i18n/en/pricing.js:42` | `"q": "Can DraftProof help me pass Turnitin?"` | 1/2 | FAQ *question* headline is a pass-Turnitin pitch if excerpted (SEO snippets, social cards); the answer refuses correctly. Rephrase to e.g. "Can DraftProof guarantee a Turnitin result?" |

## LOW findings (optional polish)

| # | Anchor | Passage | Objective | Note |
|---|--------|---------|-----------|------|
| L1 | `i18n/en/report.js:619` | `"confirmTitle": "Confirm yourself to lower a score"` | 2 | Centers the score-lowering mechanic; reframe around honest declaration ("Declare your honest AI use"). Borderline LOW/MEDIUM. |
| L2 | `i18n/en/reduceDetection.js:10` + `i18n/en/seo.js:54` | `"value": "You can't reliably beat it"` / "You can't reliably beat AI detectors" | 1 | "Reliably" implies partial success is possible; reject the goal outright (same hedge in both places — fix together). |
| L3 | `i18n/en/reduceDetection.js:39` | "A clean score from evasion is still a violation **if you're caught**." | 2 | Caught-conditionality implies it's only a problem when detected; drop the conditional. |
| L4 | `i18n/en/rewriteFraming.js:3` | "DraftProof mitigates AI-detection risk and shows you a reviewable draft…" | 1 | Leads with detection-risk mitigation; lead with the grounding/review framing instead. |
| L5 | `i18n/en/seo.js:9-11` | essayChecker title/description | 3 | Strong, but missing explicit "in your own words" authorship framing that `rewriteDescription` (line 42) already carries. |
| L6 | `i18n/en/turnitinVsDetectors.js:26-27` | comparison table intro | 3 | Table intro lacks the "why this helps your own authorship" callout present elsewhere on the page. |
| L7 | `i18n/en/footer.js:11` | `"aiDeclaration": "AI declaration"` | 3 | Not a violation; flagged for adjacency to M1 — confirm destination reinforces disclosure framing (per L1/report.js it does, minus the button label). |
| L8 | `i18n/en/dashboard.js:84` | "check whether the risk is lower before you submit" | 2 | Mild scan-until-clean loop framing; mitigated by line 80 ("Add your own evidence… ground each weak claim yourself"). Found in Fable verification pass. |

---

## What's already strong (keep as-is)

- **Explicit anti-evasion copy:** `turnitinScore.js:101-103` ("Why beating the detector is the wrong goal"); `turnitinAlternatives.js:76-78` ("doesn't access, bypass, or replace Turnitin… doesn't claim to out-score it"); `reduceDetection.js:16-21` (no humanizer/evasion mode, names the dishonesty); `essayChecker.js:12-13` ("Not a bypass tool"); `faqPage.js:59-60`; `landing.js:28,102` ("Never a bypass, never an accusation").
- **Policy-side enforcement:** `legal.js:23,28,42,47` ("may not… evade AI-detection"); `academicIntegrity.js:73,77` ("Disguising AI work to dodge a detector is still misconduct" / "We don't help you beat detectors").
- **Strengthen-your-work framing (objective 3):** `turnitinScore.js:71-77` ("Add the specifics only you have. Ground claims in real sources… paragraph by paragraph"); `rewriteOverview.js:4` ("teaching draft, not a submit-ready answer"); `rewritePage.js:24,39,41` ("guide and reference… verify, replace, or remove"); `report.js:896-904` (success = "content gaps filled, not a lower detector score"); whole of `whyPage.js` and `technologyPage.js`.

---

## Coverage

| Reviewer | Files | HIGH | MED | LOW |
|----------|-------|------|-----|-----|
| Turnitin positioning | turnitinScore, turnitinAlternatives, turnitinVsDetectors, seo, seoMetadata, ticker | 0 | 0 | 2 |
| Rewrite / detection-reduction | reduceDetection, rewriteFraming, rewriteOverview, rewritePage, essayChecker, aiDeclaration | 0 | 0 | 3 |
| Core marketing | landing, features, whyPage, technologyPage, faqPage, nav, footer | 0 | 1 | 1 |
| Product & legal | report, scan, academicIntegrity, legal, pricing, dashboard, common | 0 | 1 | 1 |

**Not covered:** email/PDF copy — same review could be run on it if desired.

---

## Fable quality-review verdict (2026-07-22)

**Score: 8.5/10 — PASS (bar 8/10).** All 9 finding anchors plus every "GOOD" citation verified
verbatim at the stated lines; zero wrong anchors. Full-directory adversarial sweep
(beat/bypass/evade/humaniz/undetect/pass/lower-score/guarantee, ~60 hits reviewed in context)
found **no missed MEDIUM+ violations** — only disclaimers. Additions from the pass (now folded
into the tables above): L8 `dashboard.js:84`, and `seo.js:54` grouped into L2.

Scope checks: `pages/*.jsx` components are clean (only code comments matched). **zh locale
MIRRORS both MEDIUMs verbatim** — `i18n/zh/footer.js:12` ("降低 AI 检测") and
`i18n/zh/pricing.js:42` ("能帮助我通过 Turnitin 吗？") — so any en fix for M1/M2 must
propagate to zh. Calibration judged sound; L1 (`report.js:619`) is the weakest call and
arguably MEDIUM, mitigated by the policy-floor note at `report.js:616`.
