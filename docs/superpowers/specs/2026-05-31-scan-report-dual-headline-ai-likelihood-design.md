# Scan Report Dual-Headline AI-Likelihood — Design

**Date:** 2026-05-31
**Status:** Approved (brainstorming) — ready for implementation planning
**Scope:** Scan-report presentation only (frontend page **and** PDF/markdown). No scoring change, no rewrite-page change, no A/B.

## Problem

The scan report under-communicates where a document actually stands, especially its
real-world Turnitin risk:

1. **Three competing DraftProof numbers** for the same doc — badge `ai_likelihood_score`
   (e.g. 42%), footprint `calibrated_ai_risk` (27%), and the displayed headline (21%).
   The multiplicity confuses rather than informs.
2. **The Turnitin/external estimate is buried.** `ai_risk_badge.external_detector_estimate`
   (e.g. 59.8%, band `high`) is DraftProof's own estimate of what strict third-party
   detectors (Turnitin/GPTZero) would say — the number the user actually faces at
   submission. On the page it renders as a small secondary stat; **in the PDF it is
   absent entirely** (page and PDF are already out of sync on this).

Net: a doc whose honest external risk is ~60% can read as "21% · Human anchoring
dominates," leaving the user unaware of their submission risk.

## Goals

- Lead the scan report with a **dual headline**: one canonical DraftProof number paired
  with the Turnitin/external estimate, clearly labeled, with a one-line "why they differ."
- Collapse / demote the redundant derivation numbers so they remain available but stop
  competing for the headline.
- **Page and PDF must show the same content** (same numbers, labels, ordering, copy).

## Non-Goals (explicitly out of scope)

- No change to any score, calibration, suppression, or tier math.
- No change to the rewrite page (its framing banner was handled separately).
- No backfill of old reports (consistent with the 2026-05-31 no-backfill decision).

## Shared Data Contract (single source of truth for both surfaces)

Both surfaces read from the scan report's `ai_risk_badge`:

| Field | Source | content11 example |
|---|---|---|
| DraftProof score | `badge.ai_likelihood_score` (0–100) | 42.0 |
| DraftProof band | `badge.tier` ∈ {GREEN, AMBER, ORANGE, RED} | AMBER |
| External score | `badge.external_detector_estimate.score` (0–100) | 59.8 |
| External band | `badge.external_detector_estimate.band` ∈ {low, elevated, high} | high |
| External note | `badge.external_detector_estimate.note` | (canonical Turnitin note) |
| Pattern label | `badge.transformation_classification.label` + `.confidence` | AI-stitched / patchwork, high |
| Rating stamp | `badge.authorship_rating_label` (+ "Not A Verdict") | Possible AI-Assisted |

Frontend access: `report.ai_risk_badge.*` (the API hoists `ai_risk_badge` to top-level on
the scan report; confirmed in `Report.jsx`). Renderer access: `badge = report.ai_risk_badge`.

### Band → label / color mapping (MUST match on both surfaces)

DraftProof (`badge.tier`):

| tier | label suffix | color |
|---|---|---|
| GREEN | conservative | `#16a34a` |
| AMBER | conservative | `#d97706` |
| ORANGE | conservative | `#ea580c` |
| RED | conservative | `#dc2626` |

External (`external_detector_estimate.band`):

| band | label | color |
|---|---|---|
| low | unlikely to be flagged | `#16a34a` |
| elevated | possibly flagged | `#d97706` |
| high | likely to be flagged | `#dc2626` |

The DraftProof number is always labeled "conservative"; the external number labeled per
the table. Round both to whole percent; external is shown with a `~` prefix (it is a
directional estimate).

### Canonical copy (duplicated verbatim: JS i18n `en` + Python strings)

- **Block title:** "AI Likelihood"
- **DraftProof caption:** "DraftProof (conservative)"
- **External caption:** "Turnitin / external"
- **Why-they-differ:** "DraftProof is false-positive-averse, so it avoids wrongly accusing
  human writers. Strict detectors (Turnitin, GPTZero) weight raw token predictability far
  more aggressively. For a genuine pass, finish the draft in your own words."
- **Collapse/section heading:** "How DraftProof calibrates this"
- **External-unavailable fallback:** "External estimate unavailable — re-scan to populate."

(Chinese `zh` translations added for the page i18n; the PDF/markdown is English, matching
the existing report renderer.)

## Surface A — Frontend page (`Report.jsx`, `reportHelpers.js`, `resources.js`, CSS)

1. New lead block at the top of the AI-risk section:
   ```
   AI LIKELIHOOD
     DraftProof (conservative)        Turnitin / external
        42%  AMBER                       ~60%  likely to be flagged
   Pattern: AI-stitched / patchwork (high confidence)  ·  Possible AI-Assisted
   <why-they-differ line>
   ```
2. The derivation gauges (calibrated 27%, displayed 21%, Human Contribution 72% /
   AI Transformation 28%, Human-Anchor / Confidence / Suppression) move into a
   `<details>` element titled "How DraftProof calibrates this", collapsed by default.
3. Remove the now-redundant standalone external-estimate stat (it is absorbed into the
   dual headline).
4. Pure band→label/color mapping extracted into a small testable helper in
   `reportHelpers.js` (e.g. `aiLikelihoodBands(badge)` → `{draftproof:{score,label,color},
   external:{score,label,color}|null}`).

## Surface B — PDF / markdown (`poc/report/render.py:render_markdown`)

The PDF is `markdown → HTML → PDF` via `poc/report/pdf.py` (WeasyPrint), driven entirely
by `render_markdown` — so updating the markdown syncs the PDF; `pdf.py` is untouched.

1. Add a dual-headline section at the top of the AI section:
   ```
   ## AI Likelihood
   - **DraftProof (conservative): 42% — AMBER**
   - **Turnitin / external: ~60% — likely to be flagged**

   Pattern: AI-stitched / patchwork (high confidence) · Possible AI-Assisted

   <why-they-differ line>
   ```
2. Demote the calibrated/contribution/suppression numbers to a secondary subsection
   **"How DraftProof calibrates this"** (lower in the doc, smaller heading) — the static
   equivalent of the page's collapse.
3. This also closes the existing gap (the external estimate is currently absent from the PDF).
4. Pure band→label helper in Python (e.g. `_ai_likelihood_bands(badge)`) mirroring the JS
   helper, unit-tested.

## Fallbacks

- `external_detector_estimate` missing/None (old pre-deploy reports): render the DraftProof
  number alone plus the muted "External estimate unavailable — re-scan to populate." line.
  No crash, graceful degradation.
- `ai_likelihood_score` missing: the block is skipped entirely (no headline) — same as today.

## Testing

**Python (`poc/`):**
- `_ai_likelihood_bands` helper unit tests: each DraftProof tier + each external band →
  correct label/color; None external → fallback; None score → skip.
- `render_markdown` test: for a fixture report with both fields, assert (a) both numbers
  appear in the output, (b) the "AI Likelihood" dual headline appears **before** the "How
  DraftProof calibrates this" subsection, (c) external-missing fixture shows the fallback
  line and does not crash.

**Frontend:**
- `aiLikelihoodBands` helper unit test (mirror of the Python cases).
- Render check that the block shows both numbers and the `<details>` collapses the gauges.
- `npm run build` clean.

## Sync guarantee

The two surfaces share the data contract + band mapping + canonical copy documented above
(this spec is the single source of truth). The copy is necessarily duplicated across JS and
Python; the band-mapping helper on each side is unit-tested against the same table, so a
drift in either surface fails a test. A `render_markdown` assertion that both numbers appear
guards the PDF against silently dropping the external estimate again.
