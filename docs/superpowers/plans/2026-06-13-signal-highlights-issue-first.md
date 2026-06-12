# Signal Highlights Issue-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-essay inline "Signal highlights" view with an issue-first, tabbed view (expandable per-paragraph cards + collapsed "Read full document") that scales to 2,000-word documents.

**Architecture:** Extract the section from the 130 KB `Report.jsx` into a focused child component `report/SignalHighlights.jsx`. The parent keeps selection state and the data model; the child owns tabs, issue cards, and the detail presentation. Removes the hand-rolled `panelOffset` sticky.

**Tech Stack:** React (function components + hooks), Vite, react-i18next, plain CSS in `styles/site-master/07-report-submitted.css`. **No frontend unit-test runner exists** (scripts: `dev`, `build`, `check:i18n`, `preview`; zero test deps). The verification gate per task is therefore: `npm run build` passes, `npm run check:i18n` passes (when i18n changes), and a **visual check** via the running app / live DOM inject. Treat those as the "tests."

---

## Verification harness (read once)

- **Build:** `cd draftproof-frontend && npm run build` → must end with `✓ built` and `Prerendered SEO metadata for N routes.`
- **i18n completeness:** `cd draftproof-frontend && npm run check:i18n` → must pass (run after ANY i18n edit; en/zh keys must match).
- **Visual (desktop):** the report page is auth-gated. Use the already-open Chrome session on a real report (`/report/<id>`) and live-inject markup/CSS to preview, OR run `npm run dev` and view the sample report. Confirm the behavior named in the task.
- **Visual (mobile):** resize the page to 390×844 and confirm single-column + inline expansion.
- **Reduced motion:** toggle OS reduce-motion (or emulate) and confirm no expand/scroll animation.

There is no `git commit` blocking gate, but **commit after each task** (frequent commits).

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/pages/report/SignalHighlights.jsx` | Tabs, issue-card list, expandable detail, full-document tab, empty state. Self-contained presentation of the submitted-content model. | **Create** |
| `src/pages/Report.jsx` | Owns selection state + data model; renders `<SignalHighlights/>`; drops `panelOffset`. | **Modify** |
| `src/styles/site-master/07-report-submitted.css` | Tab + issue-card styles; retire grid/side-panel rules. | **Modify** |
| `src/i18n/en/report.js`, `src/i18n/zh/report.js` | New keys: tabs, more-detail, issues empty state. | **Modify** |

### Component interface (locked here)

`SignalHighlights` is a presentational component. Props (all already exist in `Report.jsx`'s render scope):

```jsx
<SignalHighlights
  submittedContent={submittedContent}              // { paragraphs[], legend[], highlightedCount }
  selectedParagraph={selectedParagraph}            // currently-selected paragraph object | null
  selectedParagraphId={selectedParagraphId}        // string | null
  highlightedParagraphs={highlightedParagraphs}    // paragraphs with signals, document order
  selectedHighlightPosition={selectedHighlightPosition} // 1-based index | null
  paragraphSeverityBar={paragraphSeverityBar}      // heatmap data | null
  selectedReaderSummary={selectedReaderSummary}    // string
  selectedMainIssue={selectedMainIssue}            // string  (guidance.main_issue)
  selectedWhyFlagged={selectedWhyFlagged}          // string[] (already sliced)
  selectedRecommendation={selectedRecommendation}  // string  (derived "how to improve")
  selectedRewriteHint={selectedRewriteHint}        // string  (guidance.rewrite_hint)
  showSubmittedEditEntry={showSubmittedEditEntry}  // boolean
  onSelectParagraph={lockAndScrollParagraph}       // (id) => void  (select + open card)
  onPreviewParagraph={previewParagraph}            // (id) => void  (hover/focus preview)
  onAdjacent={selectAdjacentHighlightedParagraph}  // (direction: -1|1) => void
  onEditParagraph={openSubmittedEditorForParagraph}// (paragraph?) => void
  onCopyGuidance={copySelectedParagraphGuidance}   // () => Promise<void>
  renderSignalGauge={renderSubmittedSignalGauge}   // () => JSX  (gauge for selected paragraph)
/>
```

> Note: the detail values (`selectedReaderSummary`, `selectedMainIssue`, `selectedWhyFlagged`, `selectedRecommendation`, `selectedRewriteHint`) and the gauge/copy callbacks are computed in `Report.jsx` (`:1049–1063`, `:1120`, `:1196`) and passed as props in this first pass — faithful, low-risk, no logic moves. "Also detected" is read directly from `selectedParagraph.signals` in the child. A later cleanup may move these derivations into the child; out of scope here.

---

## Task 1: Scaffold the component and move the full-document view behind a tab

**Files:**
- Create: `src/pages/report/SignalHighlights.jsx`
- Modify: `src/pages/Report.jsx` (replace inline `submitted-content-review` block ~2281–2470 with `<SignalHighlights .../>`)

- [ ] **Step 1: Create the component shell with tab state + full-document tab**

Create `src/pages/report/SignalHighlights.jsx`. Start with two tabs; the **Issues** tab is a placeholder for now, the **Read full document** tab contains the EXISTING document markup moved verbatim from `Report.jsx` (the `submittedContent.paragraphs.map(...)` block currently at `Report.jsx:2329-2366`), minus the `<aside className="submitted-signal-panel">`.

```jsx
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { signalClassName, signalLabel, signalDescription } from './reportHelpers';
import ParagraphSeverityBar from '../../components/ParagraphSeverityBar';
import EditPencilIcon from './EditPencilIcon';

export default function SignalHighlights({
  submittedContent, selectedParagraph, selectedParagraphId, highlightedParagraphs,
  selectedHighlightPosition, paragraphSeverityBar, selectedReaderSummary,
  selectedMainIssue, selectedWhyFlagged, selectedRecommendation, selectedRewriteHint,
  showSubmittedEditEntry, onSelectParagraph, onPreviewParagraph, onAdjacent,
  onEditParagraph, onCopyGuidance, renderSignalGauge,
}) {
  const { t } = useTranslation();
  const [tab, setTab] = useState('issues'); // 'issues' | 'document'

  if (!submittedContent?.paragraphs?.length) return null;

  const FullDocument = (
    <div className="submitted-document" aria-label={t('report.submitted.documentText')}>
      {submittedContent.paragraphs.map((paragraph) => {
        const signal = paragraph.primarySignal;
        const isSelected = selectedParagraphId === paragraph.id;
        if (!signal) {
          return (
            <p key={paragraph.id}>
              <button type="button" data-paragraph-id={paragraph.id}
                className={`submitted-clean-paragraph${isSelected ? ' is-selected' : ''}`}
                onMouseEnter={() => onPreviewParagraph(paragraph.id)}
                onFocus={() => onPreviewParagraph(paragraph.id)}
                onClick={() => { onSelectParagraph(paragraph.id); setTab('issues'); }}>
                {paragraph.text}
              </button>
            </p>
          );
        }
        return (
          <p key={paragraph.id}>
            <button type="button" data-paragraph-id={paragraph.id}
              className={`submitted-highlight submitted-paragraph-highlight signal-style-${signalClassName(signal.key)}${isSelected ? ' is-selected' : ''}`}
              style={{ '--signal-color': signal.color }}
              title={signalDescription(signal.key, signal.description, t)}
              onMouseEnter={() => onPreviewParagraph(paragraph.id)}
              onFocus={() => onPreviewParagraph(paragraph.id)}
              onClick={() => { onSelectParagraph(paragraph.id); setTab('issues'); }}>
              {paragraph.text}
            </button>
          </p>
        );
      })}
    </div>
  );

  return (
    <section className="submitted-content-review" aria-label={t('report.submitted.sectionLabel')}>
      <div className="submitted-content-head">
        <div>
          <span className="submitted-content-kicker">{t('report.submitted.kicker')}</span>
          <h2>{t('report.submitted.title')}</h2>
        </div>
        <div className="submitted-content-actions">
          <div className="submitted-content-count">
            <strong>{submittedContent.highlightedCount}</strong>
            <span>{t('report.submitted.highlightedSections')}</span>
          </div>
          {showSubmittedEditEntry && (
            <button type="button" className="btn btn-secondary submitted-edit-button"
              onClick={() => onEditParagraph()}>
              <EditPencilIcon />{t('report.submitted.editor.editDraft')}
            </button>
          )}
        </div>
      </div>

      {paragraphSeverityBar?.length > 0 && (
        <ParagraphSeverityBar bar={paragraphSeverityBar} selectedId={selectedParagraph?.id} onSelect={onSelectParagraph} />
      )}

      {submittedContent.legend?.length > 0 && (
        <div className="submitted-signal-legend" aria-label={t('report.submitted.legend')}>
          {submittedContent.legend.slice(0, 6).map((signal) => (
            <span key={signal.key}
              className={`submitted-signal-chip signal-style-${signalClassName(signal.key)}`}
              style={{ '--signal-color': signal.color }}>
              <i aria-hidden="true" />{signalLabel(signal.key, signal.label, t)}<strong>{signal.count}</strong>
            </span>
          ))}
        </div>
      )}

      <div className="submitted-tabs" role="tablist" aria-label={t('report.submitted.title')}>
        <button type="button" role="tab" aria-selected={tab === 'issues'}
          className={`submitted-tab${tab === 'issues' ? ' is-active' : ''}`}
          onClick={() => setTab('issues')}>
          {t('report.submitted.tabIssues', { count: highlightedParagraphs.length })}
        </button>
        <button type="button" role="tab" aria-selected={tab === 'document'}
          className={`submitted-tab${tab === 'document' ? ' is-active' : ''}`}
          onClick={() => setTab('document')}>
          {t('report.submitted.tabDocument')}
        </button>
      </div>

      {tab === 'document' && FullDocument}
      {tab === 'issues' && <div className="submitted-issues-placeholder">issues go here</div>}
    </section>
  );
}
```

- [ ] **Step 2: Wire it into Report.jsx**

In `Report.jsx`, replace the entire inline `{submittedContent.paragraphs.length > 0 && ( <section className="submitted-content-review">…</section> )}` block (currently ~2281–2470, ending after the `</aside></div></section>`) with:

```jsx
{submittedContent.paragraphs.length > 0 && (
  <SignalHighlights
    submittedContent={submittedContent}
    selectedParagraph={selectedParagraph}
    selectedParagraphId={selectedParagraphId}
    highlightedParagraphs={highlightedParagraphs}
    selectedHighlightPosition={selectedHighlightPosition}
    paragraphSeverityBar={paragraphSeverityBar}
    selectedReaderSummary={selectedReaderSummary}
    selectedMainIssue={selectedMainIssue}
    selectedWhyFlagged={selectedWhyFlagged}
    selectedRecommendation={selectedRecommendation}
    selectedRewriteHint={selectedRewriteHint}
    showSubmittedEditEntry={showSubmittedEditEntry}
    onSelectParagraph={lockAndScrollParagraph}
    onPreviewParagraph={previewParagraph}
    onAdjacent={selectAdjacentHighlightedParagraph}
    onEditParagraph={openSubmittedEditorForParagraph}
    onCopyGuidance={copySelectedParagraphGuidance}
    renderSignalGauge={renderSubmittedSignalGauge}
  />
)}
```

Add the import near the other `report/` imports at the top of `Report.jsx`:
```jsx
import SignalHighlights from './report/SignalHighlights';
```

> Leave the `submittedDocumentRef`, `submittedPanelRef`, `panelOffset` state and its effect in place FOR NOW — Task 6 removes them, so the build stays green between tasks.

- [ ] **Step 3: Build**

Run: `cd draftproof-frontend && npm run build`
Expected: `✓ built`. No "SignalHighlights is not defined" / unused-import errors that fail the build.

- [ ] **Step 4: Visual check — full-document tab parity**

On a real report (Chrome) or `npm run dev` sample report: the section now shows two tabs; **Read full document** renders the same annotated essay as before. The Issues tab shows the placeholder. Tabs switch.

- [ ] **Step 5: Commit**

```bash
git add draftproof-frontend/src/pages/report/SignalHighlights.jsx draftproof-frontend/src/pages/Report.jsx
git commit -m "Signal highlights: extract section into SignalHighlights, add tabs + full-document tab"
```

---

## Task 2: Issue card list (collapsed) in the Issues tab

**Files:**
- Modify: `src/pages/report/SignalHighlights.jsx`

- [ ] **Step 1: Render collapsed issue cards from `highlightedParagraphs`**

Replace the `submitted-issues-placeholder` div with the issue list. Add, above the `return`, a local open-state and a helper:

```jsx
const [openId, setOpenId] = useState(highlightedParagraphs[0]?.id ?? null);
const toggleCard = (id) => setOpenId((cur) => (cur === id ? null : id));
```

Issues tab body:

```jsx
{tab === 'issues' && (
  highlightedParagraphs.length === 0 ? (
    <div className="submitted-issues-empty">
      <h3>{t('report.submitted.issuesEmptyTitle')}</h3>
      <p>{t('report.submitted.issuesEmptyBody')}</p>
    </div>
  ) : (
    <div className="submitted-issues">
      {highlightedParagraphs.map((paragraph, index) => {
        const signal = paragraph.primarySignal;
        const isOpen = openId === paragraph.id;
        const tier = signal?.tier;
        const snippet = (paragraph.text || '').slice(0, 160);
        return (
          <article key={paragraph.id}
            className={`issue-card signal-style-${signalClassName(signal?.key)}${isOpen ? ' is-open' : ''}`}
            style={{ '--signal-color': signal?.color }}>
            <button type="button" className="issue-card-head"
              aria-expanded={isOpen}
              onClick={() => { toggleCard(paragraph.id); onSelectParagraph(paragraph.id); }}
              onMouseEnter={() => onPreviewParagraph(paragraph.id)}>
              <span className="issue-card-num">{t('report.submitted.position', { current: index + 1, total: highlightedParagraphs.length })}</span>
              <span className="issue-card-main">
                <span className="issue-card-chips">
                  {tier && <em className={`issue-chip issue-chip-tier is-${tier}`}>{t(`report.severities.${tier}`, { defaultValue: tier })}</em>}
                  {signal && <em className="issue-chip">{signalLabel(signal.key, signal.label, t)}</em>}
                  <em className="issue-chip">{t('report.submitted.paragraphSignals', { count: paragraph.signalCount || paragraph.signals.length })}</em>
                </span>
                <span className="issue-card-snippet">{snippet}{paragraph.text.length > 160 ? '…' : ''}</span>
              </span>
              <span className="issue-card-caret" aria-hidden="true">{isOpen ? '▾' : '▸'}</span>
            </button>
            {isOpen && <div className="issue-card-body">{/* Task 3 */}</div>}
          </article>
        );
      })}
    </div>
  )
)}
```

- [ ] **Step 2: Build**

Run: `cd draftproof-frontend && npm run build` → Expected: `✓ built`.

- [ ] **Step 3: Visual check**

Issues tab lists one card per flagged paragraph in document order; each shows position, severity chip, signal chip, findings count, and a snippet. First card is "open" (empty body for now). Unstyled is fine — CSS is Task 5.

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/pages/report/SignalHighlights.jsx
git commit -m "Signal highlights: collapsed issue cards in document order"
```

---

## Task 3: Expanded card detail with progressive disclosure

**Files:**
- Modify: `src/pages/report/SignalHighlights.jsx`

- [ ] **Step 1: Add per-card "more detail" state and render the body**

Add near the other state: `const [showMore, setShowMore] = useState(false);` and reset it when the open card changes — replace `toggleCard` with:

```jsx
const toggleCard = (id) => setOpenId((cur) => { setShowMore(false); return cur === id ? null : id; });
```

The open card body renders detail for the **selected** paragraph (the parent sets `selectedParagraph` via `onSelectParagraph`, which fires on head click). Guard that the body only renders rich detail when `selectedParagraph?.id === paragraph.id`:

```jsx
{isOpen && (
  <div className="issue-card-body">
    {selectedParagraph?.id === paragraph.id ? (
      <>
        <p className="issue-card-summary">{selectedReaderSummary}</p>
        {selectedMainIssue && (
          <div className="issue-action">
            <span className="issue-action-label">{t('report.submitted.mainIssue')}</span>
            <p>{selectedMainIssue}</p>
          </div>
        )}
        {selectedRecommendation && (
          <div className="issue-action">
            <span className="issue-action-label">{t('report.submitted.recommendation')}</span>
            <p>{selectedRecommendation}</p>
          </div>
        )}

        <button type="button" className="issue-more-toggle" aria-expanded={showMore}
          onClick={() => setShowMore((v) => !v)}>
          {showMore ? t('report.submitted.lessDetail') : t('report.submitted.moreDetail')}
        </button>

        {showMore && (
          <div className="issue-more">
            {selectedWhyFlagged.length > 0 && (
              <div className="issue-action">
                <span className="issue-action-label">{t('report.submitted.whyFlagged')}</span>
                <ul>{selectedWhyFlagged.map((reason) => <li key={reason}>{reason}</li>)}</ul>
              </div>
            )}
            {selectedRewriteHint && (
              <div className="issue-action">
                <span className="issue-action-label">{t('report.submitted.rewriteHint')}</span>
                <p>{selectedRewriteHint}</p>
              </div>
            )}
            {selectedParagraph.signals.length > 1 && (
              <div className="issue-action">
                <span className="issue-action-label">{t('report.submitted.alsoDetected')}</span>
                <p>{selectedParagraph.signals.slice(1, 4).map((s) => signalLabel(s.key, s.label, t)).join(' · ')}</p>
              </div>
            )}
            {renderSignalGauge()}
          </div>
        )}
      </>
    ) : (
      <p className="issue-card-summary">{t('report.submitted.noSignal')}</p>
    )}
  </div>
)}
```

> Field sources (verified against `Report.jsx:1049–1063` + `:2402–2437`): `selectedMainIssue` = `guidance.main_issue`; `selectedWhyFlagged` = sliced `guidance.why_flagged`; `selectedRecommendation` = derived "how to improve"; `selectedRewriteHint` = `guidance.rewrite_hint`; "Also detected" = `selectedParagraph.signals.slice(1, 4)`. All passed as props or read from `selectedParagraph` — no invented keys.

- [ ] **Step 2: Build**

Run: `cd draftproof-frontend && npm run build` → Expected: `✓ built`.

- [ ] **Step 3: Visual check**

Open card shows summary + Main issue + How-to-improve. "＋ More detail" reveals why-flagged / rewrite-hint / gauge. Collapsing and reopening another card resets More detail to closed.

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/pages/report/SignalHighlights.jsx
git commit -m "Signal highlights: expandable card detail with progressive disclosure"
```

---

## Task 4: Card footer — Edit + Prev/Next auto-advance

**Files:**
- Modify: `src/pages/report/SignalHighlights.jsx`

- [ ] **Step 1: Add the footer inside the open card body (after the More-detail block, still inside the `selectedParagraph?.id === paragraph.id` branch)**

```jsx
<div className="issue-card-foot">
  <div className="issue-card-foot-left">
    {showSubmittedEditEntry && (
      <button type="button" className="btn btn-primary btn-small" onClick={() => onEditParagraph(paragraph)}>
        <EditPencilIcon />{t('report.submitted.editParagraph')}
      </button>
    )}
    <button type="button" className="btn btn-ghost btn-small" onClick={onCopyGuidance}>
      {t('report.submitted.copyGuidance')}
    </button>
  </div>
  <div className="issue-card-nav">
    <button type="button" className="btn btn-secondary btn-small"
      disabled={highlightedParagraphs.length < 2}
      onClick={() => { onAdjacent(-1); }}>
      {t('report.submitted.previousIssue')}
    </button>
    <button type="button" className="btn btn-secondary btn-small"
      disabled={highlightedParagraphs.length < 2}
      onClick={() => { onAdjacent(1); }}>
      {t('report.submitted.nextIssue')}
    </button>
  </div>
</div>
```

- [ ] **Step 2: Sync the open card with the selected paragraph (so Prev/Next opens the next card)**

`onAdjacent` changes `selectedParagraphId` in the parent → `selectedParagraph` prop changes. Open the card that matches the new selection and scroll it into view. Add this effect:

```jsx
import { useEffect, useRef } from 'react';
// ...
const issuesRef = useRef(null);
useEffect(() => {
  if (tab !== 'issues' || !selectedParagraph?.id) return;
  setOpenId(selectedParagraph.id);
  const el = issuesRef.current?.querySelector(`[data-issue-id="${selectedParagraph.id}"]`);
  if (el) {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    el.scrollIntoView({ block: 'nearest', behavior: prefersReduced ? 'auto' : 'smooth' });
  }
}, [selectedParagraph?.id, tab]);
```

Add `ref={issuesRef}` to the `.submitted-issues` div and `data-issue-id={paragraph.id}` to each `<article>`.

- [ ] **Step 3: Build**

Run: `cd draftproof-frontend && npm run build` → Expected: `✓ built`.

- [ ] **Step 4: Visual check**

Open a card → click **Next** → current collapses, next card opens and scrolls into view. **Edit this paragraph** opens the existing editor sheet for that paragraph. Prev/Next disabled when only one issue.

- [ ] **Step 5: Commit**

```bash
git add draftproof-frontend/src/pages/report/SignalHighlights.jsx
git commit -m "Signal highlights: card footer with Edit + Prev/Next auto-advance"
```

---

## Task 5: CSS — tabs, issue cards, retire grid/panel

**Files:**
- Modify: `src/styles/site-master/07-report-submitted.css`

- [ ] **Step 1: Add tab + issue-card styles**

Append to `07-report-submitted.css`:

```css
/* ── Signal highlights: tabs ─────────────────────────────────────── */
.submitted-tabs { display: flex; gap: .25rem; padding: 0 1.15rem; border-bottom: 1px solid #e2e8f0; }
.submitted-tab {
  appearance: none; border: 1px solid transparent; border-bottom: none; background: none;
  padding: .65rem 1rem; font: inherit; font-weight: 700; font-size: .9rem; color: #64748b;
  border-radius: 8px 8px 0 0; cursor: pointer;
}
.submitted-tab.is-active { color: var(--navy-950); background: #fff; border-color: #e2e8f0; margin-bottom: -1px; }
.submitted-tab:focus-visible { outline: 2px solid var(--green-500, #3ba876); outline-offset: 2px; }

/* ── Issue cards ─────────────────────────────────────────────────── */
.submitted-issues { padding: 1rem 1.15rem; display: flex; flex-direction: column; gap: .55rem; }
.issue-card { border: 1px solid #e2e8f0; border-left: 4px solid var(--signal-color, #0f766e); border-radius: 8px; background: #fff; }
.issue-card.is-open { box-shadow: 0 8px 22px rgba(13, 27, 42, .10); border-color: #cbd5e1; }
.issue-card-head {
  display: flex; align-items: flex-start; gap: .7rem; width: 100%;
  padding: .75rem .85rem; background: none; border: none; font: inherit; text-align: left; cursor: pointer;
}
.issue-card-head:focus-visible { outline: 2px solid var(--green-500, #3ba876); outline-offset: 2px; }
.issue-card-num { min-width: 56px; font-size: .72rem; font-weight: 800; color: #94a3b8; }
.issue-card-main { flex: 1; min-width: 0; }
.issue-card-chips { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .3rem; }
.issue-chip { font-style: normal; font-size: .68rem; font-weight: 800; padding: .12rem .5rem; border-radius: 999px; background: #f1f5f9; color: #334155; }
.issue-chip-tier.is-high, .issue-chip-tier.is-critical { background: #fef2f2; color: #b91c1c; }
.issue-chip-tier.is-moderate, .issue-chip-tier.is-medium { background: #fffbeb; color: #92400e; }
.issue-chip-tier.is-low { background: #f0fdf4; color: #15803d; }
.issue-card-snippet { display: block; font-size: .9rem; line-height: 1.5; color: #475569; }
.issue-card-caret { color: #94a3b8; font-size: 1rem; }
.issue-card-body { padding: 0 .85rem .85rem 4rem; }
.issue-card-summary { font-size: .9rem; color: #475569; line-height: 1.6; margin: 0 0 .6rem; }
.issue-action { border: 1px solid #e2e8f0; border-left: 3px solid var(--green-700, #0f766e); border-radius: 6px; padding: .55rem .7rem; margin-bottom: .5rem; background: #f8fafc; }
.issue-action-label { display: block; font-size: .66rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--green-700, #0f766e); margin-bottom: .25rem; }
.issue-action p, .issue-action ul { margin: 0; font-size: .88rem; color: #334155; line-height: 1.55; }
.issue-action ul { padding-left: 1.1rem; }
.issue-more-toggle { appearance: none; border: none; background: none; font: inherit; font-weight: 700; font-size: .82rem; color: var(--green-700, #0f766e); padding: .35rem 0; cursor: pointer; }
.issue-card-foot { display: flex; justify-content: space-between; gap: .5rem; margin-top: .7rem; flex-wrap: wrap; }
.issue-card-nav { display: flex; gap: .4rem; }
.submitted-issues-empty { padding: 2rem 1.15rem; text-align: center; }
.submitted-issues-empty h3 { color: var(--navy-950); }

@media (max-width: 760px) {
  .issue-card-body { padding-left: .85rem; }
  .issue-card-num { min-width: 44px; }
}
```

- [ ] **Step 2: Retire the old grid/side-panel rules**

In `07-report-submitted.css`, the `.submitted-content-grid` two-column layout and `.submitted-signal-panel` are no longer rendered. Leave `.submitted-document`, `.submitted-highlight`, `.submitted-clean-paragraph`, `.submitted-signal-legend`, `.submitted-content-head/-actions/-count`, `.submitted-edit-button` intact (still used). Change `.submitted-content-grid` (if it set `display:grid`) so the document tab spans full width — simplest: locate the `.submitted-content-grid { … }` rule and confirm the document now lives directly in the section (no grid wrapper); if the wrapper class is gone from the JSX, the rule is dead and can be deleted. Do not delete `.submitted-signal-panel` rules if any sub-rule is shared; grep first: `grep -rn "submitted-signal-panel" src/` — if only CSS references remain, delete the block.

- [ ] **Step 3: Build + visual**

Run: `cd draftproof-frontend && npm run build` → `✓ built`. Visual: cards match the approved mockup (severity-colored left rail, chips, snippet, expanded actions, footer). Tabs styled.

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/styles/site-master/07-report-submitted.css
git commit -m "Signal highlights: tab + issue-card styles, retire grid/side-panel"
```

---

## Task 6: Remove the dead panelOffset sticky from Report.jsx

**Files:**
- Modify: `src/pages/Report.jsx`

- [ ] **Step 1: Delete the now-unused sticky machinery**

Remove: the `panelOffset` state (`const [panelOffset, setPanelOffset] = useState(0);` ~`Report.jsx:319`), the `submittedPanelRef` (`:318`), and the effect that computes `setPanelOffset(...)` (~`:773-785`). Keep `submittedDocumentRef` only if still referenced (grep: `grep -n submittedDocumentRef src/pages/Report.jsx`); if unreferenced after the extraction, remove it too.

- [ ] **Step 2: Build**

Run: `cd draftproof-frontend && npm run build` → Expected: `✓ built`, no "panelOffset is not defined" / unused-var failures.

- [ ] **Step 3: Visual regression check**

Full report renders; Signal highlights behaves as in Task 4; nothing else on the report page broke (the panel was only used by this section).

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/pages/Report.jsx
git commit -m "Signal highlights: remove dead panelOffset sticky logic"
```

---

## Task 7: i18n keys (en + zh)

**Files:**
- Modify: `src/i18n/en/report.js`, `src/i18n/zh/report.js`

- [ ] **Step 1: Add keys under the existing `submitted` block (en)**

In `src/i18n/en/report.js`, inside the `"submitted": { … }` object, add:

```js
"tabIssues": "Issues · {{count}}",
"tabDocument": "Read full document",
"moreDetail": "＋ More detail",
"lessDetail": "− Less detail",
"issuesEmptyTitle": "No major issues detected",
"issuesEmptyBody": "Nothing is driving the report score. Open \"Read full document\" to review the full text.",
```

- [ ] **Step 2: Add the same keys (zh)**

In `src/i18n/zh/report.js`, inside `"submitted"`:

```js
"tabIssues": "问题 · {{count}}",
"tabDocument": "查看完整文档",
"moreDetail": "＋ 更多细节",
"lessDetail": "− 收起细节",
"issuesEmptyTitle": "未检测到主要问题",
"issuesEmptyBody": "没有信号在拉高报告分数。可打开"查看完整文档"通读全文。",
```

- [ ] **Step 3: i18n completeness check**

Run: `cd draftproof-frontend && npm run check:i18n`
Expected: PASS (en and zh have matching keys). Fix any reported missing key.

- [ ] **Step 4: Build**

Run: `cd draftproof-frontend && npm run build` → Expected: `✓ built`. No raw key strings (e.g. `report.submitted.tabIssues`) visible in the UI.

- [ ] **Step 5: Commit**

```bash
git add draftproof-frontend/src/i18n/en/report.js draftproof-frontend/src/i18n/zh/report.js
git commit -m "Signal highlights: i18n keys for tabs, more-detail, empty state"
```

---

## Task 8: Final verification (desktop, mobile, reduced-motion)

**Files:** none (verification only)

- [ ] **Step 1: Build + i18n**

Run: `cd draftproof-frontend && npm run build && npm run check:i18n`
Expected: both pass.

- [ ] **Step 2: Desktop walkthrough**

On a real report: Issues tab is default; cards in document order; first open; expand/collapse; More detail; Prev/Next auto-advance + scroll; Edit opens sheet; heatmap click selects+opens a card; legend present. Read-full-document tab shows the essay; clicking a highlight switches to Issues and opens that card.

- [ ] **Step 3: Mobile (390×844)**

Single column; tabs usable; cards expand inline; no detached panel; no horizontal overflow.

- [ ] **Step 4: Reduced motion**

With reduce-motion on: Prev/Next jumps (no smooth scroll); expand/collapse has no animation.

- [ ] **Step 5: Long-content sanity**

If a ≥1,500-word report is available, confirm the Issues tab stays short (one card per flagged paragraph) and the 6,000px essay only appears under Read-full-document. Otherwise note this as manually unverifiable and rely on the structural guarantee (clean paragraphs never render in the Issues tab).

- [ ] **Step 6: Final commit (if any verification fixups)**

```bash
git add -A && git commit -m "Signal highlights: verification fixups"
```

---

## Self-review notes (author)

- **Spec coverage:** tabs (T1), issue cards/document-order (T2), expand + progressive disclosure (T3), Edit + Prev/Next auto-advance (T4), CSS + retire grid/panel (T5), remove panelOffset (T6), full-doc tap→Issues (T1 onClick `setTab('issues')`), empty state (T2/T7), heatmap+legend kept (T1), mobile + reduced-motion (T5/T4/T8). All spec sections mapped.
- **Detail field names verified** against `Report.jsx:1049–1063` and the original panel render (`:2402–2437`): `main_issue`, sliced `why_flagged`, derived `selectedRecommendation`, `rewrite_hint`, and `selectedParagraph.signals.slice(1,4)` for "Also detected". Passed as props (`selectedMainIssue`/`selectedWhyFlagged`/`selectedRecommendation`/`selectedRewriteHint`) + `onCopyGuidance`. No invented keys remain.
- **Type/name consistency:** prop list in the interface section == destructuring in Task 1 == props passed in Task 1 Step 2 == usage in Tasks 3–4 (`selectedMainIssue`, `selectedWhyFlagged`, `selectedRecommendation`, `selectedRewriteHint`, `onCopyGuidance`, `renderSignalGauge`, `onAdjacent`, `onEditParagraph`, `onSelectParagraph`, `onPreviewParagraph`).
- **No FE unit tests by design** — this repo has none; gate is build + check:i18n + visual, stated up top.
