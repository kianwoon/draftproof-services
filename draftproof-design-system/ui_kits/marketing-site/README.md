# Marketing site — UI kit

A high-fidelity recreation of the **DraftProof marketing landing** (`draftproof.app`),
lifted from the production React page (`draftproof-frontend/src/pages/Landing.jsx`) and the
real stylesheet (`site-master.css`, copied in verbatim).

## What it is
The current live site is the React SPA — a **dark navy "engine room" hero** with the drifting
**code-texture** motif, sitting above calm warm-paper content sections. (An older light static
landing with the "Before you submit, prove your work is grounded." headline still lives in the
repo at `draftproof_landing_site/`; the React design shown here is what's deployed.)

## Run it
Open `index.html`. No build step — React + Babel load from CDN and the `.jsx` files mount in order.

## Files
| File | Role |
|---|---|
| `index.html` | Assembles the full landing; small presentation overrides at the top (see below). |
| `site-master.css` | The production stylesheet, verbatim. Source of truth for every class. |
| `CodeTexture.jsx` | The signature drifting monospace motif (two rows, opposite directions, masked fade). |
| `SiteHeader.jsx` | Fixed dark header + brand mark + nav + green CTA. |
| `Hero.jsx` | Navy hero with the headline (green highlight span) + floating "Quick Summary" review panel. |
| `SampleReportSection.jsx` | The interactive 3-tab report preview (AI Signal / Score Profile / Action Plan), auto-rotating. |
| `LandingSections.jsx` | Trust bar, "human-written" section, why cards, four-checks engine, beliefs, CTA, footer. |
| `assets/` | Brand mark SVG. |

## Components covered
SiteHeader · Hero + review panel + metric bars · TrustBar · HumanWrittenSection (signal list +
guardrails) · SampleReport (tabbed, interactive) · WhySection (numbered cards) · EngineSection
(four checks on a connector line) · BeliefsSection (×/✓ rows) · LandingCTA · LandingFooter.

## Faithfulness notes
- Every class name + structure mirrors production; copy is verbatim from the live i18n (`en`).
- **Presentation overrides** (top of `index.html`, clearly commented, *not* in the source CSS):
  (1) `html/#root height:auto` so the static page scrolls (production is a height:100% SPA);
  (2) the landing hero collapses to one column at ≤1080px instead of ≤920px, so narrow preview
  panes show the real stacked layout rather than the cramped in-between band; (3) the headline
  highlight is scoped to a class so direct-edit text wrappers don't tint the whole h1 green.
- Best viewed ≥1080px wide (the registered card renders at 1280). The code texture is intentionally
  prominent on dark panels — that's the production aesthetic, not a bug.
