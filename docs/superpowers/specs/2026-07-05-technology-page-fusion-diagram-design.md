# /technology page: Signal Fusion diagram inside Pillar 1

## Problem

The shipped `/technology` page (`docs/superpowers/specs/2026-07-05-technology-page-design.md`) is text-only. The user wants a visual diagram to make Pillar 1 ("No single black-box score decides anything.") visually self-evident instead of asking the reader to picture the fusion process from prose alone.

Explored via the visual-companion brainstorming tool (3 rough concepts: linear pipeline, signal fusion, continuous loop) — user picked **Signal Fusion**, refined over 3 iterations to the version below.

## Design

### What the diagram shows

Two independent signal nodes converging into one fused-score node, feeding a single labeled-band output — visually proving Pillar 1's actual claim ("no single signal decides alone"):

```
[Pattern-based analysis]  ─┐
                            ├─→ (Fused score) ─→ [● Low — labeled band]
[A separate deep-reading   ─┘
 model]
```

Below the diagram, three small pill-shaped tags reinforce Pillar 2's existing rigor claims — **capability-level only, no vendor/model names, no raw dataset counts** (a raw number like "270+ essays" was explicitly rejected during brainstorming: it risks reading as "small" to a reader who doesn't know what's sufficient for a fairness gate, even though the real number is meaningful):
- "Tested against real student writing"
- "Spans ESL & multilingual proficiency levels"
- "Re-validated across multiple AI-writing systems"

These three tags reuse language already established in Pillar 2/4's shipped prose — no new claims invented for the diagram.

### Placement

Inside Pillar 1's `<section>` in `Technology.jsx`, in this exact order (top to bottom): pillar number → title → body paragraph → **diagram + proof-chips** (new) → `whyItMatters` caption (existing, stays last). Only Pillar 1 gets the diagram — the other 4 pillars are unchanged.

### Visual style

- Fixed on the "Low" band (green dot + "Low" + "labeled band" caption) — not cycling through Low/Moderate/High/Critical. Matches the page's generally reassuring tone.
- **Correction from the initial brainstorm mockup:** the mockup used a dark card as a rough placeholder, but `Technology.jsx`'s Pillar 1 renders inside `.why-section` (`site-master/09-why-legal-document-viewer.css:80`), which is **light-themed** (`var(--ink)` dark text, light surface) — only the page's top hero (`app-hero-dark`) is dark. Confirmed with the user: the diagram card must be **light**, matching Pillar 1's actual section — not dark. Node borders in an indigo/accent tone; the fused-score node likewise; the output band chip reuses the exact same light-mode green-chip styling already shipped for real report tier bands (`.authorship-breakdown-fused-band-chip.is-green` in `site-master/08-rewrite-and-report-details.css:724-727`: `border-color: rgba(22, 163, 74, .35); background: rgba(22, 163, 74, .12); color: #15803d;`) — reuse those exact values/pattern rather than inventing new ones, for visual consistency with the real product.
- Colors MUST use this project's existing CSS custom properties / already-shipped color values (see above) rather than the brainstorm mockup's placeholder dark-theme hex values (`#1f2430`, `#6a6ad0`, `#d0a06a`, `#6ad08a`, `#16161f`, etc.) — none of those survive into the implementation.
- Implemented as **inline SVG** (not a static image asset) — scalable, no asset file to maintain, matches the mockup's construction.

### Component structure

A small, isolated `SignalFusionDiagram` component (own file or a named function within `Technology.jsx` — decide based on file-size at planning time) takes the diagram's text content as props/data, rendered only for Pillar 1:

```jsx
{pillar.diagram && <SignalFusionDiagram data={pillar.diagram} />}
```

### i18n (mandatory — this project has a documented locale-trap failure mode)

Every text label inside the diagram must be a translatable i18n string, not hardcoded English baked into JSX/SVG — matching how every other string on this page already works. Add a new `diagram` sub-object to Pillar 1 only, in both `i18n/en/technologyPage.js` and `i18n/zh/technologyPage.js`:

```js
"diagram": {
  "signal1": "Pattern-based analysis",
  "signal2": "A separate deep-reading model",
  "fusedLabel": "Fused score",
  "bandLabel": "Low",
  "bandCaption": "labeled band",
  "chips": [
    "Tested against real student writing",
    "Spans ESL & multilingual proficiency levels",
    "Re-validated across multiple AI-writing systems"
  ]
}
```

(Exact zh translations to be authored at plan-writing time, following the same full-width-punctuation convention already established and corrected in the prior `/technology` page work — see `feedback`/lessons from commit `16425b6c`.)

## Out of scope (explicitly)

- No vendor/model names anywhere in the diagram (no "GPT-5-mini," "Gemini," etc.) — this reaffirms, not revisits, the page's original constraint. Reasoning surfaced during brainstorming: naming specific evaluation targets invites gaming, and the list goes stale as new models ship.
- No raw dataset size numbers (e.g. essay counts) — replaced with qualitative, already-shipped phrasing.
- No diagrams added to Pillars 2-5 — this is scoped to Pillar 1 only. If the other pillars want diagrams later, that's a separate brainstorm/spec.
- No cycling/animation through band states — fixed on "Low".
- No change to any other page content, routing, or the two already-repaired dead `#engine` links.

## Testing / verification

- `npm run build:client` compiles clean.
- Visual check via preview tool: `/technology` and `/zh/technology` both show the diagram inside Pillar 1 only, correct colors matching the site theme (not the mockup's placeholder hex), fixed "Low" band, all text localized on zh with no English fallback, no console errors.
- Grep confirms every new i18n key exists in both `en/technologyPage.js` and `zh/technologyPage.js`.
