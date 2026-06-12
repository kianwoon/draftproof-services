# Signal Highlights — Issue-First Redesign

**Date:** 2026-06-13
**Area:** `draftproof-frontend` report page → "Signal highlights" section
**Status:** Approved design, ready for implementation plan

## Problem

The "Signal highlights" section renders the **entire submitted document inline** as
its primary surface — every paragraph, always, with `max-height: none`
(`07-report-submitted.css`). At ~518 words the left column is already ~1,580px
(taller than the viewport). At **2,000 words it is ~6,000px+**, and the reader must
scroll the whole essay to reach the ~N flagged paragraphs, which are visually equal
to the clean ones.

Three structural weaknesses make long documents worse:

1. **No way to hide clean content** — there is no "show only flagged" / collapse
   affordance. Most paragraphs in a long essay are scroll-noise.
2. **Hand-rolled sticky panel** — the right "repair note" panel follows the selected
   paragraph via a manual JS transform (`panelOffset` translateY, clamped, recomputed
   on selection: `Report.jsx` ~773-785). On a 6,000px column this drifts and fights
   the browser.
3. **Mobile break** — the two-column grid stacks (document, then a detached panel);
   the translateY sticky does not work stacked, so the actionable detail ends up far
   from its paragraph.

The actionable unit is the **flagged paragraph**, but the current design optimizes
for *reading the whole essay*.

## Goal

Make the section **issue-first**: lead with a compact, scannable set of the flagged
paragraphs, each expandable to its repair note. Demote the full essay to a collapsed
"Read full document" view in the same section. A 2,000-word document then acts on
like a 200-word one, because only the flagged cards render up front.

## Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Approach | Issue-first redesign |
| Full essay placement | Collapsed in the same section, behind a "Read full document" tab |
| Card interaction | **B — expandable issue cards** (single column, tap to expand inline) |
| Default sort | **Document order** (severity still shown per card; heatmap is the severity map) |
| Accordion | One card open at a time; first card open by default |
| Detail disclosure | "Main issue" + "How to improve" always visible; rest behind "＋ More detail" |

## Design

### Layout (top → bottom)

1. **Header** — kicker "Submitted Content", title "Signal highlights", highlighted
   count, "Manual Rewrite / Correction" button. *Unchanged.*
2. **Finding-density heatmap + legend chips.** *Kept.* Remains the document-order
   severity map; clicking a segment selects/opens that paragraph's card.
3. **Tabs:** `Issues · N` | `Read full document`. Default = **Issues**.
4. **Issues tab** — single column of expandable cards, in **document order**:
   - **Collapsed card** shows: position (`1 / 8`), severity chip (color from
     `TIER_CONFIG`), primary signal chip, signal-strength `%`, findings count, and a
     1–2 line snippet of the paragraph with the flagged phrase emphasized.
   - **Expanded card** (one open at a time; first open by default) shows:
     - Always: **Main issue to fix**, **How to improve this paragraph**.
     - **＋ More detail** expander reveals: *What the reader may notice*, *Also
       detected*, *Rewrite hint*, and the signal-strength gauge.
     - Footer: **Edit this paragraph** (opens the existing editor sheet) and
       **Prev / Next** controls that collapse the current card and open the
       adjacent one (sequential editing pass). Prev/Next auto-scroll the opened
       card into view.
5. **Read full document tab** — the current annotated essay (inline paragraph
   highlights). Clicking a highlighted paragraph **switches to the Issues tab and
   opens that paragraph's card**, so there is a single detail surface (no duplicate
   side panel here).

### Behavior & data reuse

- Reuses existing data and helpers wholesale — this is a **presentation** change, not
  a recomputation:
  - `selectedParagraph` + its guidance (`why_flagged`, recommendation, rewrite hint),
  - `signalLabel` / `signalDescription`, severity tiers (`TIER_CONFIG`),
  - the existing editor sheet (`openSubmittedEditorForParagraph`),
  - `ParagraphSeverityBar` (heatmap), the signal legend.
- **Selected-paragraph state** drives which card is open. `highlightedParagraphs`
  (`Report.jsx:1039`) already exists and is the card list source (document order).
- **Removed:** the two-column `submitted-content-grid`, the `submitted-signal-panel`,
  and the `panelOffset` / translateY logic (`Report.jsx` ~773-785, ~317-319). Detail
  now lives inline in the open card, so no hand-rolled sticky is needed.

### Edge cases

- **Clean document (0 flagged):** Issues tab shows a positive empty state
  ("No major issues detected"), with a pointer to "Read full document".
- **One flagged paragraph:** Prev/Next disabled (matches current behavior).
- **Reduced motion:** card expand/collapse and auto-scroll respect
  `prefers-reduced-motion`.

### Mobile

Single column already. Tabs stack; cards expand inline; the full-document tab's
tap-to-open routes to the Issues tab. This removes the current stacked-panel break.

## Scope & files

**New**
- `src/pages/report/SignalHighlights.jsx` — the issue-first section: tabs, the issue
  card (collapsed + expanded), accordion logic, "More detail" disclosure, empty
  state. Extracting it keeps the change out of the 130KB `Report.jsx` and respects
  the 1,500-line-per-file rule.

**Modified**
- `src/pages/Report.jsx` — render `<SignalHighlights … />` in place of the current
  inline `submitted-content-review` block (~2281–2600+); pass the existing props
  (`submittedContent`, `selectedParagraph`, selection handlers, editor entry,
  `paragraphSeverityBar`, `highlightedParagraphs`, guidance); remove the now-unused
  `panelOffset` state and effect.
- `src/styles/site-master/07-report-submitted.css` — add tab + issue-card styles;
  retire the `submitted-content-grid` / `submitted-signal-panel` rules (or scope any
  still used by the full-document tab).
- `src/i18n/en/report.js` + `src/i18n/zh/report.js` — add keys: tab labels
  (`Issues`, `Read full document`), `moreDetail`, issues empty-state title/body.
  Reuse existing `submitted.*` keys for the repair-note content.

**Reused (unchanged)**
- `ParagraphSeverityBar`, the editor sheet components, `reportHelpers` signal/tier
  helpers.

## Non-goals (YAGNI)

- No change to detection/scoring or to the data the report produces.
- No severity-sort or group-by-signal modes (document order only; can be added later).
- No virtualization (collapsed cards are cheap; revisit only if a real doc shows lag).
- No change to the rewrite/edit pipeline — only the entry point (the card's Edit
  button) is rewired to the existing sheet.

## Verification

1. `npm run build` passes.
2. Visual check: live-inject the new markup/CSS on a real report + dev build with the
   sample report; confirm collapsed→expand, Prev/Next auto-advance, tab switching,
   and full-doc tap-to-open.
3. Mobile viewport (≤390px): single column, cards expand inline, no detached panel.
4. Reduced-motion: no animation on expand/scroll.
