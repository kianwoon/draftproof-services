# Landing page sample-report preview: refresh to match the real V7 report

## Problem

The landing page's `SampleReportPreview` (`Landing.jsx`, tabs driven by `landing.reportPreviewTabs`) simulates 6 tabs: `authorshipBreakdown`, `aiSignal`, `scoreProfile`, `actionPlan`, `findings`, `criticalThinking`. Verified directly against the real `Report.jsx` (2000+ lines) and its sub-components, several of these no longer match what a real scan report shows today:

- `scoreProfile` represents a feature that was **explicitly removed** from the product (`Report.jsx:1999-2005`, owner decision 2026-07-04: "the writing-signal scorecard, score-profile, authenticity dashboard... were old-methodology internals").
- `aiSignal`'s "Estimated Contribution" bars/tags dashboard no longer exists; it's been replaced by a verdict-line + grounding-diagnosis-bars card (`Report.jsx:1403-1500`).
- `actionPlan` and `findings` are structurally close to real components but use outdated visual conventions (3-tone coloring, "also-detected/main-issue/rewrite-hint" chips) that don't match the real `FixFirstChecklist`/`SignalHighlights` components.
- `criticalThinking` already matches the real `CriticalThinkingControl` — no change needed.

Two corrections made mid-investigation (recorded here for the record, since they contradicted earlier claims presented to the user in this session): `SubmissionRiskBand` is **not** removed — it still renders inside `ReportHero` (`ReportHero.jsx:115`, fed from `Report.jsx:745,1896`) — and `PolicyRiskView` was never removed either. Both stay in the `aiSignal` tab.

## Design

### Tab list change

Remove `scoreProfile` entirely from `reportPreviewTabs` (en + zh) — 6 tabs become 5. No replacement; nothing real to show in its place.

### `aiSignal` tab — keep SubmissionRiskBand/PolicyRiskView, replace the rest

Keep, unchanged: `<SubmissionRiskBand t={t} sr={SAMPLE_SUBMISSION_RISK} />` and `<PolicyRiskView t={t} pr={SAMPLE_POLICY_RISK} />` (both still real, both already accurate).

Remove entirely (all map to the deleted "Estimated Contribution" dashboard — `sample-report-pattern`, `sample-report-chart`, `sample-original-scan`, `sample-contribution`, `sample-report-notes`, and the `turnitinReference` paragraph in `Landing.jsx`'s `SampleReportPreview`): the transformation-pattern icon block, the "Low AI-writing signal" / "41% calibrated top-k" authorship badge, the original-scan score chart, the human/AI contribution bars, and their supporting tag chips.

Add, in their place, a card mirroring the real verdict + grounding-diagnosis content (`Report.jsx:1403-1465`, copy sourced from `i18n/en/report.js:97-116,619-649` — reuse these exact real strings, don't invent new phrasing):
- Verdict line: "Low AI-writing signal — low on our scale — but detectors over-flag fluent writing, so they may still flag it (a warning, not a verdict)." (real `verdictSignal.green` + `verdictFlag.low`, concatenated exactly as `Report.jsx:1428` does)
- "Main thing to fix" callout: driver label "Grounding gap", action "Add concrete anchors, named evidence, and specifics." (real `groundingDiagnosis.drivers.concrete_grounding`)
- "Risk contributors" bars (heading + "Lower is better" subheading, real `groundingDiagnosis.bucketsHeading`/`lowerIsBetter`), 4 buckets in descending order with the primary driver first: Grounding gap 58, Authorship uncertainty 34, AI-like patterning 22, Generic language texture 15 (illustrative numbers; real bucket *labels* — `buckets.*` — copied verbatim, not invented)

### `actionPlan` tab — restyle to match `FixFirstChecklist`

Replace the 3-tone (warning/quality/positive) colored item cards with the real component's plain style: a kicker+title header (reuse real copy: kicker "Repair Plan", title "What to fix first" — `i18n/en/report.js:159-161`), then numbered items each with a bold title + body line + one label chip (e.g. "High priority" / "Medium priority" / "Quick win" — illustrative, no real i18n key for this specific chip text since `FixFirstChecklist`'s `label` prop is data-driven, not a fixed vocabulary). Keep the existing 3 illustrative action items' subject matter (citation support / source grounding / generic phrasing) — only the visual chrome changes.

### `findings` tab — restructure to match `SignalHighlights`'s issue-card

Replace the single card's "also-detected/main-issue/rewrite-hint" chip layout with the real issue-card shape (`SignalHighlights.jsx:169-266`, copy from `i18n/en/report.js:501-533`):
- Position chip: "2/5" (real `positionShort` pattern `{{current}}/{{total}}`)
- Tier chip: "HIGH" (real `severities.high`)
- Signal-label chip: keep existing "AI Likelihood" framing
- Count chip: "3 flagged sentences in paragraph" (real `paragraphSignals_other` pattern)
- Reader summary: keep existing description text
- Critical-thinking dimension callout (new): "Evidence grounding — Connect each claim to a source, example, or data point." (real `criticalThinking.dimensions.evidence_grounding`)
- Flagged-sentences list (new, replaces the old rewrite-hint): 1-2 items, each a score % + the quoted sentence + a suggestion — reuse the existing sample paragraph's content (the Hollywood/cultural-export example already used elsewhere on this page) reshaped into this list format instead of the old single rewrite-hint paragraph

### `criticalThinking` tab

No change — already matches `CriticalThinkingControl`'s real quote+question structure.

### i18n cleanup

Removing the `aiSignal` tab's old content orphans a number of `landing.js` keys entirely (`transformationPattern`, `humanUncertain`, `lowConfidence`, `notVerdict`, `aiSignal`, `lowAiSignal`, `calibratedTopk`, `originalScan`, `originalScanScore`, `calibratedAiRisk`, `humanAnchorDiscount`, `calibrationConfidence`, `reportingSuppression`, `estimatedContribution`, `contributionBody`, `humanContribution`, `aiTransformation`, `sampleReportNotes`, `turnitinReference`, and the entire `scoreProfile`-tab-only keys: `scoreProfile`, `whyScoreMoved`, `scoreProfileBody`, `aiStyleSignal`, `sourceGroundingSignal`, `humanAnchorSignal`, `sampleScoreSignals`). These must be deleted from both `i18n/en/landing.js` and `i18n/zh/landing.js`, not just left unreferenced — this whole feature exists because stale, unreferenced content misleads. `authorshipRating`/`good`/`calibratedRisk` also look orphaned already (pre-existing, unrelated to this change) — confirm at plan-writing time whether they're truly dead and include their removal only if so.

## Out of scope (explicitly)

- No change to `Landing.jsx`'s other sections (hero, trust-bar, use-case carousel, technology page, etc.).
- No change to the real `Report.jsx`/its components — this is a landing-page content-accuracy fix only.
- No re-architecture of the tabbed preview UI shell into a non-tabbed single scroll (considered and rejected during brainstorming).
- `authorshipBreakdown` and `criticalThinking` tabs are not touched beyond the tab-list reordering side-effect of removing `scoreProfile`.

## Testing / verification

- `npm run build:client` compiles clean.
- Grep confirms all listed orphaned keys are removed from both `en/landing.js` and `zh/landing.js`, and that `reportPreviewTabs` has exactly 5 entries in both locales.
- Visual check via preview tool: `/` (landing) sample-report preview shows 5 tabs, `aiSignal` tab shows the new verdict/diagnosis card (not the old contribution dashboard), `actionPlan` and `findings` tabs show the restyled content, no console errors, zh fully localized.
