# Fine-Tune v1 Landing Announcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one new static, i18n'd line to the landing-page hero announcing the fine-tune v1 detector, with copy verified against the live production calibration (not the superseded pre-promotion numbers).

**Architecture:** Pure frontend, static content. A new i18n key (`landing.detectorUpdateNote`) in both locale files, rendered as one new line in `Landing.jsx`'s existing hero-copy block, styled via one new CSS rule that reuses the existing `.trust-note` layout and `.hero-free-credit-spark` icon. No new component, no backend/API/scoring changes.

**Tech Stack:** React 18 + Vite + react-i18next (existing app, no new dependencies). Verification via the repo's existing `check:i18n` script and SSR prerender build (no frontend test framework exists in this repo — none is being introduced for one marketing line; see Task 2 rationale).

## Global Constraints

- **No specific model names in the copy** (owner decision, spec §1.2): use "frontier AI writing," never "GPT-5.5," "GPT-5.6," "Gemini," or "Qwen."
- **Exact approved copy** (spec §2.1, EN): "Detector freshly fine-tuned on frontier AI writing and thousands of real essays — ESL false positives held under 1%, even as detection accuracy rose."
- **No absolute "0%" claims** — the verified live number is 0.74% (2/272 SCoCESLE), i.e. "under 1%," never "0%."
- **Tag all new static copy** with an `// allow-hardcode: ...` comment per the repo's existing convention (this is marketing copy, not detection logic — see `src/i18n/en/landing.js` for the existing pattern).
- **Both locales required**: `en/landing.js` and `zh/landing.js` — never ship an EN-only key on this page (repo-wide i18n completeness expectation).
- **No new component, no new npm dependency** (spec §3 non-goals — Option 2's reusable announcement system was explicitly declined for this instance).
- Full spec: [docs/superpowers/specs/2026-07-14-finetune-v1-landing-announcement-design.md](../specs/2026-07-14-finetune-v1-landing-announcement-design.md)

---

## Task 1: i18n copy (EN + ZH) with a real regression check

**Files:**
- Modify: `draftproof-frontend/scripts/check-i18n-resources.mjs`
- Modify: `draftproof-frontend/src/i18n/en/landing.js:28` (insert after the existing `trustNote` line)
- Modify: `draftproof-frontend/src/i18n/zh/landing.js:27` (insert after the existing `trustNote` line)

**Interfaces:**
- Produces: i18n key `landing.detectorUpdateNote`, readable via `t('landing.detectorUpdateNote')` in any component under the `I18nextProvider` (used by Task 2).

This repo has no frontend test framework (`find src -iname "*.test.jsx"` returns nothing, no Jest/Vitest in `package.json`). It does have a real, existing regression check for landing/legal copy: `draftproof-frontend/scripts/check-i18n-resources.mjs`, run via `npm run check:i18n`, which asserts specific translated strings match an expected pattern. That script is this task's test.

- [ ] **Step 1: Add the (currently failing) check entries**

Open `draftproof-frontend/scripts/check-i18n-resources.mjs`. Add two entries to the `checks` array (after the existing `security page` entries, before the closing `];`):

```javascript
  {
    label: 'English detector update note',
    value: resources.en.translation.landing.detectorUpdateNote,
    matches: /^Detector freshly fine-tuned on frontier AI writing/,
  },
  {
    label: 'Chinese detector update note',
    value: resources.zh.translation.landing.detectorUpdateNote,
    matches: /^检测器刚完成微调/,
  },
```

- [ ] **Step 2: Run the check to verify it fails**

Run: `cd draftproof-frontend && npm run check:i18n`
Expected: throws `Error: English detector update note has unexpected translation: undefined` (the key doesn't exist yet).

- [ ] **Step 3: Add the English key**

In `draftproof-frontend/src/i18n/en/landing.js`, find line 28 (`"trustNote": "Not a Turnitin replacement · ...",`). Insert immediately after it:

```javascript
  // allow-hardcode: static landing-page marketing copy announcing the fine-tune v1
  // detector (2026-07-14, poc/detect_v7/weights.json re-weight commits
  // 647fac65/ea5847bf — ESL FPR 0.74%/272, NOT 0%). Not a scoring oracle. If the
  // live ESL FPR or operating point changes again, re-verify against weights.json
  // before editing this string. TODO: revisit/remove this framing after 2026-08-15.
  "detectorUpdateNote": "Detector freshly fine-tuned on frontier AI writing and thousands of real essays — ESL false positives held under 1%, even as detection accuracy rose.",
```

- [ ] **Step 4: Add the Chinese key**

In `draftproof-frontend/src/i18n/zh/landing.js`, find line 27 (`"trustNote": "不是 Turnitin 的替代品 · ...",`). Insert immediately after it:

```javascript
  // allow-hardcode: 见 en/landing.js 同名条目注释 — 静态营销文案，非评分依据。
  "detectorUpdateNote": "检测器刚完成微调，训练数据涵盖前沿 AI 生成文本与数千篇真实文章 —— 检测准确率提升的同时，英语非母语写作者的误判率控制在 1% 以下。",
```

- [ ] **Step 5: Run the check to verify it passes**

Run: `cd draftproof-frontend && npm run check:i18n`
Expected: exits with no output and status 0 (no `Error` thrown).

- [ ] **Step 6: Flag the Chinese copy for native review**

This translation is Claude-drafted, not native-reviewed (spec §2.3 open item). Leave it shipped behind the check above (which only verifies the opening clause, not full fluency) and tell the owner explicitly in the handoff summary that this line specifically wants a native pass — do not silently assume it's polished.

- [ ] **Step 7: Commit**

```bash
cd draftproof-frontend && git add scripts/check-i18n-resources.mjs src/i18n/en/landing.js src/i18n/zh/landing.js
git commit -m "feat(landing): add fine-tune v1 announcement copy (en+zh)"
```

---

## Task 2: Hero placement + styling

**Files:**
- Modify: `draftproof-frontend/src/pages/Landing.jsx:186-189` (insert new block after the existing `.trust-note` div)
- Modify: `draftproof-frontend/src/styles/site-master/02-landing-hero.css` (insert new rule after the `.hero-free-credit-link:hover` rule, ~line 379)

**Interfaces:**
- Consumes: `t('landing.detectorUpdateNote')` (Task 1) and the existing `.hero-free-credit-spark` CSS class (already defined at `02-landing-hero.css:361`, unchanged).
- Produces: a rendered `.detector-update-note` line in the hero, visible to Task 3's build/prerender check.

No component test framework exists for this page (see Task 1). The verification here is the same one this repo actually uses for landing-page changes: render it and look, plus a prerendered-HTML grep (Task 3) — this repo has a prior shipped bug where landing copy rendered fine in dev but was absent from the prerendered HTML actually served to crawlers/first paint (`project_seo_body_not_prerendered`, fixed 2026-07-01), so "looks right in `npm run dev`" is not sufficient proof on its own.

- [ ] **Step 1: Add the JSX block**

In `draftproof-frontend/src/pages/Landing.jsx`, current lines 185-189 read:

```jsx
            <div className="trust-note">
              <span className="mini-shield" aria-hidden="true" />
              <span>{t('landing.trustNote')}</span>
            </div>
          </div>
```

Replace with:

```jsx
            <div className="trust-note">
              <span className="mini-shield" aria-hidden="true" />
              <span>{t('landing.trustNote')}</span>
            </div>

            <div className="trust-note detector-update-note">
              <span className="hero-free-credit-spark" aria-hidden="true" />
              <span>{t('landing.detectorUpdateNote')}</span>
            </div>
          </div>
```

(The outer `.hero-copy` closing `</div>` on the last line is unchanged — only the new block is inserted before it.)

- [ ] **Step 2: Add the CSS rule**

In `draftproof-frontend/src/styles/site-master/02-landing-hero.css`, after the `.hero-free-credit-link:hover` rule (~line 379):

```css
.hero-free-credit-link:hover {
  color: #fff;
}

.landing-hero .detector-update-note {
  color: rgba(255, 255, 255, .78);
  margin-top: 8px;
}
```

(`.detector-update-note` only overrides color/spacing — it inherits flex layout, gap, icon sizing, and font-size from the existing `.trust-note` / `.hero-free-credit-spark` rules already in this file. Brighter than `.trust-note`'s `rgba(255,255,255,.44)` since this is a positive highlight, not a muted disclaimer.)

- [ ] **Step 3: Visual check — desktop**

Run: `cd draftproof-frontend && npm run dev`
Open `http://localhost:3000/` in a browser. Confirm:
- The new line appears directly below the "Not a Turnitin replacement..." line, with a small green dot icon (not the shield square).
- Text is legible against the hero background video/texture.
- No overlap with `HeroReviewPanel` on the right.

- [ ] **Step 4: Visual check — mobile + Chinese locale**

In the same dev server, resize to a 375px-wide viewport and check `http://localhost:3000/zh` (or use the in-app language switcher). Confirm:
- No text overflow/clipping on narrow width (hero already has mobile-specific CSS from commit `f82f2fe1` — confirm this new line respects it, doesn't need its own media query).
- The Chinese line renders (visually confirms Task 1's zh key resolved correctly, not just the automated check).

- [ ] **Step 5: Commit**

```bash
cd draftproof-frontend && git add src/pages/Landing.jsx src/styles/site-master/02-landing-hero.css
git commit -m "feat(landing): render fine-tune v1 announcement in hero"
```

---

## Task 3: Prerender verification

**Files:**
- None modified — this task only builds and inspects output, per spec §4 step 3.

**Interfaces:**
- Consumes: the committed state of Task 1 + Task 2.
- Produces: a pass/fail confirmation that the SSR prerender pipeline actually emits this copy (the repo's own precedent, `project_seo_body_not_prerendered`, is that dev-server-correct is not the same as prerendered-correct).

- [ ] **Step 1: Run the full production build**

Run: `cd draftproof-frontend && npm run build`
Expected: completes with no errors (this runs `vite build`, the SSR build, `prerender-seo.mjs`, and `prerender-render.mjs` in sequence — per `package.json`'s `build` script).

- [ ] **Step 2: Grep the prerendered EN output**

Run: `grep -r "Detector freshly fine-tuned" dist/ | head -5`
Expected: at least one match inside the prerendered landing-page HTML file under `dist/` (the exact path depends on `prerender-seo.mjs`'s output routing — if the first grep finds nothing, run `grep -rl "hero-free-credit-spark" dist/` to locate the correct prerendered landing HTML file, then inspect it directly).

- [ ] **Step 3: Grep the prerendered ZH output**

Run: `grep -r "检测器刚完成微调" dist/ | head -5`
Expected: at least one match inside the prerendered `/zh` landing-page HTML.

- [ ] **Step 4: If either grep finds nothing**

Do not treat this as done. This exact failure mode (renders in dev, absent from prerendered HTML) shipped before in this repo. Read `scripts/prerender-render.mjs` and `scripts/prerender-seo.mjs` to find why the hero isn't reaching the prerendered output, fix it, and re-run Steps 1-3 before proceeding — this is a blocking check, not an optional nice-to-have.

- [ ] **Step 5: Report result to the owner**

No commit in this task (nothing changed, only verified). Summarize pass/fail for both locales in the handoff message, and explicitly flag the ZH copy for the owner's own native-fluency pass (Task 1, Step 6).

---

## Self-Review Notes

- **Spec coverage:** §2.1 copy → Task 1. §2.2 placement → Task 2. §2.3 i18n (both locales + allow-hardcode) → Task 1. §2.4 styling → Task 2. §2.5 staleness (date anchor + TODO) → Task 1 Step 3's comment includes the TODO; the visible "(July 2026)" date-anchor from spec §2.5 was evaluated and dropped from the copy itself to keep the approved sentence (spec §2.1) verbatim — the TODO comment is the enforcement mechanism instead. **Flagging this as a deliberate deviation from spec §2.5's literal wording, not an oversight:** the spec offered a visible date-anchor as one option ("e.g. trailing '(July 2026)'"), and this plan chooses the TODO-comment-only half of that section rather than both, to avoid re-opening the exact copy the owner already approved. Surfacing this now rather than silently picking one. §2.6 rollback coupling → not a code task (no feature flag exists to couple it to); recorded as a manual runbook note, already captured in spec §2.6 itself. §4 validation → Task 3. §5 rollout → no separate task needed (plain `git push` after all three tasks land, same as any other frontend change).
- **Placeholder scan:** no TBD/TODO-as-plan-gap found; the one literal `TODO:` string is intentional shipped code (a dated maintenance comment), not a plan placeholder.
- **Type consistency:** single new identifier (`landing.detectorUpdateNote`) used identically across Tasks 1-3; no signature drift possible (no functions introduced).
- **Resolved (pre-flight, owner):** whether to also add a visible date-anchor in the copy itself (spec §2.5's other half) — owner chose to ship the exact approved sentence with no visible date anchor, relying on the TODO-comment only. No plan change needed; Task 1 already matches this.
