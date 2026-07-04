# Technology Page Fusion Diagram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an inline-SVG "Signal Fusion" diagram inside Pillar 1 of the already-shipped `/technology` page, visually proving the pillar's claim ("no single signal decides alone") instead of leaving it to prose alone.

**Architecture:** One new small component, `SignalFusionDiagram`, defined inside the existing `draftproof-frontend/src/pages/Technology.jsx` (the file is only 48 lines — no need for a separate file, matching how `Landing.jsx` keeps multiple small helper components in one file). It renders only for Pillar 1 (index 0), reads its text content from new i18n keys, and reuses this project's real design tokens and an existing report chip's exact color values — no new colors invented.

**Tech Stack:** React 18 + react-i18next, inline SVG, existing `site-master.css` custom properties. No test framework exists for this frontend — verification is `npm run build:client` plus a preview-tool visual check.

## Global Constraints

- No literal library/model/vendor names anywhere in the diagram (no "GPT-5-mini," "Gemini," etc.) and no raw dataset-size numbers (e.g. essay counts) — capability-level language only, reusing phrasing already shipped in Pillar 2/4's prose. This was explicitly re-confirmed during this feature's brainstorm after the raw-count idea was proposed and rejected (risk: a small-sounding number undermines credibility even though the real number is meaningful).
- Every i18n key must exist in BOTH `i18n/en/technologyPage.js` and `i18n/zh/technologyPage.js`.
- The diagram card is **light-themed** (matches `.why-section`'s real light background — confirmed by reading `site-master/09-why-legal-document-viewer.css:80-86`), NOT dark. Only the page's top hero (`app-hero-dark`) is dark.
- Colors MUST come from this project's real design tokens/values — no new hex colors invented:
  - Input signal box borders: `var(--navy-900)` (`#1A2E42`)
  - Fused-score circle border: `var(--gold)` (`#C9973A`)
  - Output "Low" band chip: copy the exact values already shipped for `.authorship-breakdown-fused-band-chip.is-green` (`site-master/08-rewrite-and-report-details.css:724-727`): `border-color: rgba(22, 163, 74, .35); background: rgba(22, 163, 74, .12); color: #15803d;`
  - Connector lines: `rgba(13, 27, 42, .25)`
  - Primary text: `var(--ink)`; secondary/caption text: `var(--muted)`
  - Card border: `rgba(13, 27, 42, .10)` (matches `.why-section`'s own hairline border convention — no filled card background, per this site's existing minimal-card aesthetic confirmed in `.anchor-card`)
- The diagram renders ONLY for Pillar 1 (index 0) — Pillars 2-5 are visually unchanged.
- Fixed on the "Low" band — no cycling/animation through Low/Moderate/High/Critical.
- Implemented as inline SVG (not a static image asset).

---

### Task 1: Diagram content + component + wiring

**Files:**
- Modify: `draftproof-frontend/src/i18n/en/technologyPage.js` (add `diagram` key to Pillar 1 only)
- Modify: `draftproof-frontend/src/i18n/zh/technologyPage.js` (same, zh)
- Modify: `draftproof-frontend/src/pages/Technology.jsx` (new `SignalFusionDiagram` component + render wiring)
- Modify: `draftproof-frontend/src/styles/site-master/09-why-legal-document-viewer.css` (new, small CSS block for the diagram's non-SVG elements — the proof-chips row and card wrapper; the file already owns `.why-section`/`.why-*` styling, so new `.fusion-diagram-*` classes belong here, not a new CSS file)

**Interfaces:**
- Consumes: nothing from other tasks (this plan has only one task — see Task Right-Sizing note below).
- Produces: nothing consumed elsewhere — this is the full feature.

**Note on task sizing:** this feature is small and tightly coupled (i18n content, the component that renders it, and its CSS all only make sense together — a reviewer can't meaningfully approve the CSS while rejecting the i18n content it styles), so it is ONE task rather than split further, per the plan's Task Right-Sizing rule.

- [ ] **Step 1: Add the `diagram` key to Pillar 1 in `i18n/en/technologyPage.js`**

Find the first object in the `pillars` array (the one whose `"title"` is `"No single black-box score decides anything."`). Add a `"diagram"` key to that object, after its existing `"whyItMatters"` key:

```js
    {
      "title": "No single black-box score decides anything.",
      "body": "A single AI-detector call is noisy and easy to game in either direction. DraftProof combines multiple independent detection signals — pattern-based analysis and a separate deep-reading model — before any tier or score is shown, so no single miscalibrated signal can swing a verdict on its own.",
      "whyItMatters": "For students: one flaky detector can't wrongly flag your work. For educators: a verdict backed by agreement across independent signals is more defensible than a single tool's number.",
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
    },
```

Do not add a `"diagram"` key to any of the other 4 pillar objects.

- [ ] **Step 2: Add the matching `diagram` key to Pillar 1 in `i18n/zh/technologyPage.js`**

Find the first object in the zh `pillars` array (the one whose `"title"` is `"没有任何单一的黑箱分数能决定结果。"`). Add the equivalent `"diagram"` key, using full-width punctuation consistent with the rest of this file (per the lesson from this page's earlier punctuation fixup, commit `16425b6c`):

```js
      "diagram": {
        "signal1": "基于模式的分析",
        "signal2": "一个独立的深度阅读模型",
        "fusedLabel": "综合分数",
        "bandLabel": "低",
        "bandCaption": "标注区间",
        "chips": [
          "针对真实学生写作进行测试",
          "涵盖 ESL 与多语言写作水平",
          "针对多个 AI 写作系统持续重新验证"
        ]
      }
```

- [ ] **Step 3: Add the diagram CSS block to `draftproof-frontend/src/styles/site-master/09-why-legal-document-viewer.css`**

Append this block at the end of the file (after the existing `.why-*` rules — this file already owns the Why/Technology page's section styling):

```css
.fusion-diagram {
  margin: 1.25rem 0;
  padding: 1rem 1.25rem 1.25rem;
  border: 0.5px solid rgba(13, 27, 42, .10);
  border-radius: var(--radius-md);
}

.fusion-diagram-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.fusion-diagram-chip {
  border: 0.5px solid rgba(13, 27, 42, .15);
  border-radius: 999px;
  padding: 0.3rem 0.75rem;
  font-size: 0.78rem;
  color: var(--muted);
  background: transparent;
}
```

- [ ] **Step 4: Add the `SignalFusionDiagram` component to `Technology.jsx`**

In `draftproof-frontend/src/pages/Technology.jsx`, add this function after the default-exported `Technology` function (i.e. at the end of the file, after the closing `}` on line 48):

```jsx
function SignalFusionDiagram({ data }) {
  if (!data) return null;
  const chips = Array.isArray(data.chips) ? data.chips : [];

  return (
    <div className="fusion-diagram">
      <svg viewBox="0 0 460 170" width="100%" height="190" role="img" aria-label={`${data.signal1} + ${data.signal2} → ${data.fusedLabel} → ${data.bandLabel}`}>
        <rect x="10" y="14" width="150" height="48" rx="8" fill="none" stroke="var(--navy-900)" strokeWidth="1.5" />
        <text x="85" y="34" fontSize="12" fill="var(--ink)" textAnchor="middle" fontWeight="600">{data.signal1}</text>

        <rect x="10" y="104" width="150" height="48" rx="8" fill="none" stroke="var(--navy-900)" strokeWidth="1.5" />
        <text x="85" y="124" fontSize="12" fill="var(--ink)" textAnchor="middle" fontWeight="600">{data.signal2}</text>

        <line x1="160" y1="38" x2="220" y2="85" stroke="rgba(13, 27, 42, .25)" strokeWidth="1.5" />
        <line x1="160" y1="128" x2="220" y2="85" stroke="rgba(13, 27, 42, .25)" strokeWidth="1.5" />

        <circle cx="260" cy="85" r="46" fill="none" stroke="var(--gold)" strokeWidth="2" />
        <text x="260" y="82" fontSize="12" fill="var(--ink)" textAnchor="middle" fontWeight="600">{data.fusedLabel}</text>

        <line x1="306" y1="85" x2="352" y2="85" stroke="rgba(13, 27, 42, .25)" strokeWidth="1.5" markerEnd="url(#fusionArrow)" />

        <rect x="352" y="60" width="98" height="50" rx="8" fill="rgba(22, 163, 74, .12)" stroke="rgba(22, 163, 74, .35)" strokeWidth="1.5" />
        <circle cx="374" cy="85" r="6" fill="#15803d" />
        <text x="410" y="81" fontSize="12" fill="#15803d" textAnchor="middle" fontWeight="600">{data.bandLabel}</text>
        <text x="410" y="95" fontSize="9" fill="var(--muted)" textAnchor="middle">{data.bandCaption}</text>

        <defs>
          <marker id="fusionArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="rgba(13, 27, 42, .45)" />
          </marker>
        </defs>
      </svg>

      <div className="fusion-diagram-chips">
        {chips.map((chip) => (
          <span className="fusion-diagram-chip" key={chip}>{chip}</span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Render the diagram inside Pillar 1 only**

In `Technology.jsx`'s `pillars.map(...)` block (currently at line 26-33), change:

```jsx
        {pillars.map((pillar, index) => (
          <section className="why-section" key={pillar.title} aria-label={t('technologyPage.pillarsLabel')}>
            <span className="why-num">{String(index + 1).padStart(2, '0')}</span>
            <h2>{pillar.title}</h2>
            <p>{pillar.body}</p>
            <p className="why-highlight">{pillar.whyItMatters}</p>
          </section>
        ))}
```

to:

```jsx
        {pillars.map((pillar, index) => (
          <section className="why-section" key={pillar.title} aria-label={t('technologyPage.pillarsLabel')}>
            <span className="why-num">{String(index + 1).padStart(2, '0')}</span>
            <h2>{pillar.title}</h2>
            <p>{pillar.body}</p>
            {pillar.diagram && <SignalFusionDiagram data={pillar.diagram} />}
            <p className="why-highlight">{pillar.whyItMatters}</p>
          </section>
        ))}
```

This renders `SignalFusionDiagram` only when `pillar.diagram` is present (Pillar 1 only, since only that pillar object has the key added in Steps 1-2), in the exact order the spec requires: title → body → diagram → `whyItMatters` caption last.

- [ ] **Step 6: Build check**

Run: `cd draftproof-frontend && npm run build:client`
Expected: build completes with no errors.

- [ ] **Step 7: Visual verification via preview tool**

Start the dev server (reuse `.claude/launch.json` from prior work in this worktree if present). Then:
- Navigate to `/technology`: confirm the diagram renders inside Pillar 1 only (Pillars 2-5 unchanged), light card with a thin border (not dark), two signal boxes with navy borders, a gold-bordered "Fused score" circle, connector lines, and a green "Low" output chip matching the real report's tier-chip color.
- Confirm the three proof-chips render below the diagram, in a row, with the exact capability-level text (no vendor names, no numbers).
- Navigate to `/zh/technology`: confirm the diagram and chips are fully localized, no English fallback text.
- Check console for errors (`mcp__Claude_Preview__preview_console_logs`, level `error`).

- [ ] **Step 8: Commit**

```bash
git add draftproof-frontend/src/i18n/en/technologyPage.js draftproof-frontend/src/i18n/zh/technologyPage.js draftproof-frontend/src/pages/Technology.jsx draftproof-frontend/src/styles/site-master/09-why-legal-document-viewer.css
git commit -m "feat(technology): add signal-fusion diagram to Pillar 1"
```

---

## Self-Review Notes

- **Spec coverage:** Diagram content/placement/order → Step 5. Light-theme correction + real color tokens → Global Constraints + Step 4/6. i18n en/zh parity → Steps 1-2. Inline SVG, no vendor names/numbers → Step 4/1-2. Pillar-1-only scoping → Step 5's conditional render. ✓
- **Placeholder scan:** no TBD/TODO; all code blocks are complete, copy-pasteable. ✓
- **Type/name consistency:** `SignalFusionDiagram` component name and its `data` prop shape (`signal1`/`signal2`/`fusedLabel`/`bandLabel`/`bandCaption`/`chips`) defined identically in Step 1/2 (i18n content) and Step 4 (component reading `data.signal1` etc.) and Step 5 (passing `pillar.diagram` as `data`) — no drift. ✓
