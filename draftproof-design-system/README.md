# DraftProof — Design System

> Pre-submission **writing integrity reviews** for education and research.
> *Not an AI detector. Not a plagiarism verdict. A review you can act on.*

This repository is the brand + UI source of truth for **DraftProof** (draftproof.app).
Everything here is lifted from the real product code — not invented — so that any
agent can produce on-brand interfaces, marketing, and decks.

---

## 1. What DraftProof is

DraftProof helps students, educators, and researchers review a draft **before they
submit it**. A user pastes or uploads their writing; DraftProof scans it and returns a
**tier classification** (`clean` / `acceptable` / `concerning` / `strong`) plus detailed,
explained findings across four lenses:

1. **Citation gaps** — claims that lack a supporting source.
2. **Source grounding / integrity** — whether sources actually back the claims.
3. **Predictability / AI-risk signals** — patterns that *look* machine-generated, surfaced
   **without making an authorship accusation**.
4. **Similarity risk** — quoted terms, templates, and common academic phrasing in context.

The findings are sorted into **action categories** — *citation repair*, *rewrite priority*,
*review-only signals*, and *no action* — and an optional **rewrite** pass demonstrates a
reviewable "solution" draft (a highlighted before/after diff) that the user edits with their
own content. The product's stance is explicit: it **does not chase zero findings** and **does
not blindly "humanize"** text — it fixes the right thing for the kind of writing you actually have.

**Positioning line:** *"Most writing tools judge the final text. DraftProof reviews the
evidence behind it."*

### Pricing (from the live product)
- **Scan:** $0.90 per 1,000 words (short scans are free).
- **Rewrite:** $4.50 (starts from a scan).
- Token-based credits; Google / Microsoft OAuth sign-in.

---

## 2. Products / surfaces

| Surface | What it is | Source of truth |
|---|---|---|
| **Web app** | React + Vite SPA — the signed-in product (dashboard, scan, report, rewrite, pricing, sign-in). This is what `draftproof.app` serves. | `draftproof-frontend/` |
| **Marketing site** | Public landing — hero, engine beliefs, feature grid, content-aware strategies, sample report, audience, CTA. | The React `Landing` page + the standalone `draftproof_landing_site/` |

Both surfaces share **one CSS system** (`site-master.css`) and the same tokens. The web app
adds a warm-paper background and dark navy hero panels; the marketing landing leans into the
dark navy "engine" aesthetic with the animated code texture.

---

## 3. Sources (for whoever maintains this)

> You may not have access to these — they are recorded so the lineage is clear.

- **GitHub repo:** `kianwoon/draftproof-services` (private), branch `main`.
  - `draftproof-frontend/` — React + Vite SPA (the production UI). Canonical tokens live in
    `draftproof-frontend/src/styles/site-master.css` (`:root`).
  - `draftproof_landing_site/` — a standalone static landing (`index.html` + `site-master.css`).
    Note: this standalone uses an **older** palette (blue `#235caa`, Georgia display headings).
    **The React frontend tokens win** wherever the two disagree.
  - `draftproof-api/`, `worker/`, `poc/` — FastAPI backend, Celery worker, detection/rewrite
    pipeline. Not design-relevant; kept for product context. See repo `CLAUDE.md`.
- **Live site:** https://draftproof.app/  (theme-color `#0D1B2A`; OG image at `/og-image.png`).
- **Imported into this project:** the frontend `src/` (pages, components, full CSS), the
  standalone landing, and brand assets (logos, favicons, OG image). See the index at the bottom.

### Substitutions / flags
- **Fonts: no files shipped.** The product loads **Inter** + **Noto Sans SC** from Google Fonts
  and uses **Georgia** (a system serif) for reading content + **ui-monospace** (system) for the
  code texture. `colors_and_type.css` `@import`s Inter/Noto from Google Fonts. If you need offline
  font files, ask the brand owner — none are in the repo.
- **Palette decision:** the project brief leaned "warm editorial / amber." The real product is
  **navy + teal-green** with a warm-paper surface. I followed the codebase (source of truth). The
  warm-paper background (`#F4F2EE`) keeps an editorial warmth; flag if you'd prefer the amber route.

---

## 4. Content fundamentals — how DraftProof writes

**Voice: calm, precise, anti-hype, reassuring.** The copy's whole job is to *de-escalate* anxiety
("is my essay going to get flagged?") into *clear, bounded action*. It never accuses, never
promises a magic score, never says "AI-free."

- **Person:** addresses the reader as **you** ("Before *you* submit, prove *your* work is grounded").
  Refers to the product as **DraftProof** in the third person, rarely "we."
- **Casing:** **Sentence case** for everything — headings, buttons, nav. No Title Case, no ALL-CAPS
  except the small uppercase **eyebrow** kickers (e.g. `WRITING INTEGRITY REVIEWS FOR EDUCATION AND RESEARCH`).
- **Sentence shape:** short, declarative, often **contrastive pairs** that set DraftProof apart:
  - *"Not an AI detector. Not a plagiarism verdict. A writing integrity review you can act on."*
  - *"Submit stronger work. Not just cleaner text."*
  - *"DraftProof does not chase zero findings."*
- **Verbs:** action-first and concrete — *review, prove, fix, repair, ground, flag, separate, surface.*
- **Hedging is deliberate, not weak:** "signals," "likelihood," "estimated," "risk" — DraftProof
  reports *signals*, it does **not** render verdicts. This is a legal + ethical stance and must be
  preserved. Mandatory disclaimer (footer): *"DraftProof provides writing integrity signals and
  review guidance. It does not determine misconduct, plagiarism, or AI authorship."*
- **No emoji in product chrome or headlines.** Emoji appear **only** as small audience markers on the
  marketing "who it's for" cards (🎓 students, 🧑‍🏫 educators, 👥 writing centres, 🔬 researchers).
  Do not introduce emoji elsewhere.
- **Numbers are specific and modest:** `16.8%`, `30.0%`, `4,982 words`, `$0.90`. Percentages get a
  decimal; risk is always paired with a qualitative tier word (Low / Medium / Strong).
- **Vibe:** a trustworthy academic-integrity advisor — the writing-centre tutor who explains *why*,
  not the proctor who catches you.

**Microcopy examples (verbatim from product).** Two surfaces ship slightly different copy —
the **live React app** (current) leans Turnitin-forward; the **older standalone landing** uses
the broader "writing integrity" framing. Both are on-voice; match whichever surface you're building.

*Live React app (current — `draftproof.app`):*
- Eyebrow: `Turnitin-safe essay review for students`
- H1: `Submit your essay with stronger evidence and lower Turnitin risk.` (the second clause is green)
- Buttons: `Review my essay` · `View sample report` · `Start review` · `Start scan`
- Trust note: `Not a Turnitin bypass · Not a misconduct verdict · A pre-submission essay review you can act on.`

*Older standalone landing (`draftproof_landing_site/` — alternate voice):*
- Eyebrow: `Writing integrity reviews for education and research`
- H1: `Before you submit, prove your work is grounded.`
- Trust note: `Not an AI detector. Not a plagiarism verdict. A writing integrity review you can act on.`
- Belief cards: `Similarity is not always plagiarism` · `AI scores are not proof` · `Citation gaps matter`
- Positioning line: `Most writing tools judge the final text. DraftProof reviews the evidence behind it.`

---

## 5. Visual foundations

**Overall feeling:** quiet, technical, trustworthy. A dark **navy** "engine room" for moments of
authority (hero, sign-in, scan, footer) sitting on a warm **paper** workspace (`#F4F2EE`) for the
calm, readable product surfaces. Teal-**green** is the single voice of action and "good."

### Color
- **Ink / dark:** navy `#0D1B2A` (`--navy-950`), with `#1A2E42`, `#243B53` for layered dark panels.
- **Primary / action / positive:** green `#3BA876` (`--green`), hover/deep `#0F6E56` (`--green-dark`).
- **Mint** `#E1F5EE` — soft green tint for pills, callouts, tinted surfaces.
- **Surfaces:** card white `#fff`; **warm paper** `#F4F2EE` is the app background (NOT pure white).
- **Muted text:** `#7A8694`. **Hairlines:** `rgba(13,27,42,.10)` — thin, low-contrast.
- **Status tiers:** green = low/clean · gold `#C9973A` = medium/concerning · red `#E24B4A` = high/strong.
- **Logo greens are brighter** than the UI green: edge `#22C55E`, check `#34D399`, fill `#0B2A25`.
  Use these *only* inside the mark; UI accents use `--green`.

### Type
- **Two voices.** UI chrome = **Inter** (weight 600 headings, 500 nav/labels, 400 body). The user's
  **writing / report body / rewrite diff = Georgia serif** — a deliberate editorial register that
  signals "this is your prose, being read." Don't set product chrome in serif, and don't set essay
  content in Inter.
- Headings are **sentence case, weight 600, letter-spacing 0** — restrained, not loud. Clamped sizes
  (`h1: 2.25→3.6rem`, `h2: 1.9→2.5rem`). Body line-height a generous `1.7`; reading content `1.82`.
- **Eyebrow** kicker: 12px, uppercase, `+.8px` tracking, muted — sits above most section headings.

### Backgrounds & the signature motif
- **Code texture** is THE brand motif: a faint, slowly drifting field of monospace strings made of
  binary + domain verbs (`source.check()  citation.match  grounded=true  claim.verify`) behind every
  dark navy panel. Two rows drift in opposite directions (`codeDriftLeft 28s`, `codeDriftRight 34s`,
  linear/infinite); row A is white at ~30% opacity, row B is green at ~52%. Masked with a horizontal
  fade so it never competes with text. See `assets/code-texture.css` + the UI kit's `CodeTexture.jsx`.
- Light surfaces get **very subtle radial tints** of green/navy, never loud gradients. No purple,
  no rainbow gradients, no glassmorphism-for-its-own-sake.
- Dark heroes are flat navy with the code texture; a soft top-vignette (`::before`) adds depth.

### Depth, shape, motion
- **Radii:** `8px` (controls/inputs), `14px` (cards/panels), `16px` (large cards, hero panels). Pills
  are `20px`. Restrained — nothing fully rounded except pills and avatars.
- **Shadows:** soft and navy-tinted, never black. `--shadow-sm: 0 8px 22px rgba(13,27,42,.07)`,
  `--shadow-md: 0 18px 46px rgba(13,27,42,.12)`. Cards often use a **hairline border** *instead of*
  or *with* a faint shadow. On dark panels, elevation = `rgba(255,255,255,.06)` fill + `.12` border.
- **Borders:** hairline, `.5px` where the renderer supports it, low-opacity ink.
- **Motion:** one easing everywhere — `cubic-bezier(.22,1,.36,1)` (`--ease`), durations 150–200ms.
  Hovers are gentle: buttons **lift** `translateY(-1px)` + pick up `--shadow-sm`; cards lift `-2px`.
  No bounce, no spring, no big scale. `prefers-reduced-motion` disables the texture drift + snapping.
- **Hover states:** primary button darkens green `#3BA876 → #0F6E56` + lifts; secondary fills a
  faint `rgba(13,27,42,.04)` + darker hairline; nav links go from `.86 → 1` opacity.
- **Press / focus:** inputs focus to a **green** border (`--green`); no heavy glow.
- **Transparency & blur:** the fixed header is `rgba(244,242,238,.92)` + `backdrop-filter: blur(18px)`.
  Dark glass panels use `rgba(255,255,255,.06)` fills. Blur is reserved for the header and overlays.

### Layout
- Max content width **1200px**, centered; fixed header **76px**; generous section padding
  (`72px` vertical). 12-ish column intuition via CSS grid with `gap`. Marketing uses scroll-snap
  sections; the app is normal scroll.

---

## 6. Iconography

DraftProof's icon language is **deliberately minimal and home-grown** — there is **no icon-font
dependency** (no Font Awesome / Material Icons) and **no third-party SVG icon library** in the repo.

- **Inline stroke SVGs**, hand-authored, `viewBox 0 0 24 24` (or `16`/`32`), `fill="none"`,
  `stroke="currentColor"`, `stroke-width` ≈ **1.4–1.8**, round caps/joins. They inherit text color,
  so they pick up navy / green / muted from context. Examples in the repo: report/document glyph,
  history clock, scan square-plus, nav chevrons. Match this style for any new icon.
- **The shield-check mark** is the one true brand icon (`assets/logo-mark.svg` / `favicon.svg`):
  a shield outline with an interior checkmark. Two treatments —
  - **Navy outline** on light surfaces (header brand, `assets/logo-shield-navy.png`).
  - **Green-filled** as the app icon / on dark (`assets/apple-touch-icon.png`, `assets/favicon.png`,
    OG image): dark `#0B2A25` shield, `#22C55E` edge, `#34D399` check.
- **Geometric Unicode glyphs** stand in for decorative section/feature icons on the *standalone*
  landing (`▣ ✓ ◇ ★ ▧ ▥ ⬡ ◎ ✎ ◌ ⇧ ⌕ ☷ ▽ ☑`). These are a lightweight, font-driven choice — fine for
  marketing accents, but prefer real stroke SVGs in the app.
- **Brand-colored provider logos** (Google 4-color, Microsoft 4-square) appear only on the sign-in
  buttons, drawn as inline SVG with official colors.
- **Emoji:** audience markers only (see Content Fundamentals). Never as UI affordances.

If you need an icon that doesn't exist, **draw a stroke SVG in the house style** (1.6px, round,
currentColor) rather than importing a library. If a library is unavoidable, the closest match is
**Lucide** (same 24px / ~1.75 stroke / round-join language) — flag the substitution.

---

## 7. Index — what's in this folder

| Path | What |
|---|---|
| `README.md` | This file. |
| `SKILL.md` | Agent-Skill manifest (works in Claude Code too). |
| `colors_and_type.css` | All design tokens as CSS vars + semantic element styles. **Start here.** |
| `assets/` | Logos, favicons, OG image, and `code-texture.css` (the motif). |
| `assets/logo-mark.svg` / `logo-shield.svg` | Canonical shield-check mark (green). |
| `assets/logo-shield-navy.png` | Navy outline shield (light-surface lockup). |
| `assets/favicon.png` · `apple-touch-icon.png` · `og-image.png` | App icons / social. |
| `assets/code-texture.css` | Standalone CSS for the drifting code-texture motif. |
| `preview/` | Small Design-System cards (type, color, spacing, components, brand). |
| `ui_kits/web-app/` | Web-app UI kit — interactive recreation (dashboard, scan, report, sign-in, pricing). |
| `ui_kits/marketing-site/` | Marketing-site UI kit — landing recreation. |
| `slides/` | On-brand slide template (title, statement, comparison, report, closing). |
| `draftproof-frontend/`, `draftproof_landing_site/` | Imported source (reference only). |
| `reference/` | Screenshots captured during build. |

**Reading order for a new agent:** `colors_and_type.css` → §4 Content Fundamentals →
§5 Visual Foundations → the relevant `ui_kits/<surface>/index.html`.
