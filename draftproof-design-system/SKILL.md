---
name: draftproof-design
description: Use this skill to generate well-branded interfaces and assets for DraftProof, either for production or throwaway prototypes/mocks/etc. Contains essential design guidelines, colors, type, fonts, assets, and UI kit components for prototyping.
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files.
If creating visual artifacts (slides, mocks, throwaway prototypes, etc), copy assets out and create static HTML files for the user to view. If working on production code, you can copy assets and read the rules here to become an expert in designing with this brand.
If the user invokes this skill without any other guidance, ask them what they want to build or design, ask some questions, and act as an expert designer who outputs HTML artifacts _or_ production code, depending on the need.

## Quick orientation

DraftProof is a **pre-submission writing-integrity review** for students (and educators / researchers).
It finds citation gaps, weak source grounding, similarity risk, and AI-like writing patterns, then
turns them into a revision plan. Its stance is calm and anti-hype: **signals, not verdicts** — never an
AI-detector verdict, never a "bypass," never a promise of zero findings.

**The fastest path to on-brand output:**
1. Read `README.md` (full brand: voice, color, type, motion, iconography, the code-texture motif).
2. Use the tokens in `colors_and_type.css` — or just copy the real stylesheet
   `ui_kits/web-app/site-master.css` (it is the production CSS, verbatim) and reuse its class names.
3. Pull components/screens from `ui_kits/web-app/` (signed-in product) and `ui_kits/marketing-site/`
   (landing). Copy the `.jsx` files and `CodeTexture.jsx`; they emit the real class names.
4. For decks, start from `slides/index.html`.
5. Copy brand assets from `assets/` (shield mark, favicons, OG image). Never redraw the logo.

## Non-negotiables
- **Navy `#0D1B2A` ink + teal-green `#3BA876` action** on **warm paper `#F4F2EE`**. Green is the only
  "good / action" voice. Status tiers: green = low, gold `#C9973A` = medium, red `#E24B4A` = high.
- **Type: Inter** for UI/headings (sentence case, weight 600), **Georgia** for the user's prose /
  report body / rewrite diff. The drifting monospace **code texture** is the signature motif on every
  dark navy panel — reuse `CodeTexture.jsx`, don't reinvent it.
- **Voice:** calm, precise, sentence case, addresses the reader as "you." Contrastive pairs
  ("Not an AI detector. … A review you can act on."). Mandatory footer disclaimer — see README §4.
- **No emoji** except the audience markers noted in the README. **No purple gradients, no
  glassmorphism, no rounded-left-border-accent cards.** Radii 8/14/16, navy-tinted soft shadows,
  one easing `cubic-bezier(.22,1,.36,1)`, gentle hover lift.
