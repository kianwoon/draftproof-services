# Scan-Report Dual-Headline AI-Likelihood — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lead the scan report (page AND PDF) with DraftProof (conservative) vs Turnitin/external AI-likelihood side by side, and demote the redundant calibrated/contribution/suppression derivations.

**Architecture:** Presentation-only. Two surfaces share one data contract (`ai_risk_badge.ai_likelihood_score` + `badge.external_detector_estimate`) and one band→label/color table. Python `render_markdown` drives the WeasyPrint PDF (no separate PDF layout). Frontend `Report.jsx` renders the page. No scoring/calibration/tier change; no A/B; no rewrite-page change.

**Tech Stack:** Python (poc/report), pytest; React/Vite (draftproof-frontend), i18next. No JS unit-test runner exists, so JS helper correctness is build-verified + reviewed against the spec table; the **Python helper test is the canonical band-mapping guard** (both sides mirror the same table).

**Spec:** `docs/superpowers/specs/2026-05-31-scan-report-dual-headline-ai-likelihood-design.md`

**Verify env (Python):** `cd <repo>; PYTHONPATH="$PWD:$PWD/poc" ~/.pyenv/versions/3.11.0/bin/python3 -m pytest <test> -q`
**Verify env (frontend):** `cd draftproof-frontend; export PATH="/opt/homebrew/bin:$PATH"; npm run build`

**Constraint:** NEVER `git add -A` (repo has ~220 untracked files). Stage exact paths only.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `poc/report/render.py` | Python band helper + markdown dual-headline + demote derivations | Modify |
| `poc/test_report_ai_likelihood_headline.py` | Python helper + render_markdown tests | Create |
| `draftproof-frontend/src/pages/report/reportHelpers.js` | JS band helper (mirror) | Modify |
| `draftproof-frontend/src/i18n/resources.js` | `report.aiLikelihood.*` copy (en + zh) | Modify |
| `draftproof-frontend/src/styles/site-master.css` | dual-headline block + `<details>` styles | Modify |
| `draftproof-frontend/src/pages/Report.jsx` | render dual headline, collapse derivations, drop standalone external stat | Modify |

---

## Task 1: Python band helper `_ai_likelihood_bands`

**Files:**
- Modify: `poc/report/render.py` (add helper near the other `_transformation_*` helpers, e.g. after `_transformation_contribution_summary`)
- Test: `poc/test_report_ai_likelihood_headline.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `poc/test_report_ai_likelihood_headline.py`:

```python
"""Dual-headline AI-likelihood rendering (page + PDF share this band table)."""
from report.render import _ai_likelihood_bands


def test_draftproof_tier_maps_to_color():
    out = _ai_likelihood_bands({"ai_likelihood_score": 42.0, "tier": "AMBER"})
    assert out["draftproof"] == {"score": 42, "tier": "AMBER", "color": "#d97706"}


def test_all_draftproof_tiers():
    colors = {"GREEN": "#16a34a", "AMBER": "#d97706", "ORANGE": "#ea580c", "RED": "#dc2626"}
    for tier, color in colors.items():
        out = _ai_likelihood_bands({"ai_likelihood_score": 50, "tier": tier})
        assert out["draftproof"]["color"] == color


def test_external_bands_map_to_label_and_color():
    cases = {
        "low": ("unlikely to be flagged", "#16a34a"),
        "elevated": ("possibly flagged", "#d97706"),
        "high": ("likely to be flagged", "#dc2626"),
    }
    for band, (label, color) in cases.items():
        out = _ai_likelihood_bands({
            "ai_likelihood_score": 42, "tier": "AMBER",
            "external_detector_estimate": {"score": 59.8, "band": band},
        })
        assert out["external"]["label"] == label
        assert out["external"]["color"] == color
        assert out["external"]["score"] == 60  # rounded


def test_missing_external_returns_none():
    out = _ai_likelihood_bands({"ai_likelihood_score": 42, "tier": "AMBER"})
    assert out["external"] is None
    assert out["draftproof"]["score"] == 42


def test_missing_score_returns_none_draftproof():
    assert _ai_likelihood_bands({})["draftproof"] is None
    assert _ai_likelihood_bands(None)["draftproof"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/poc" ~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_report_ai_likelihood_headline.py -q`
Expected: FAIL — `ImportError: cannot import name '_ai_likelihood_bands'`

- [ ] **Step 3: Implement the helper in `poc/report/render.py`**

Add near the top-level module constants and helpers (after `_transformation_contribution_summary`):

```python
_DRAFTPROOF_TIER_COLORS = {
    "GREEN": "#16a34a", "AMBER": "#d97706", "ORANGE": "#ea580c", "RED": "#dc2626",
}
_EXTERNAL_BAND_LABELS = {
    "low": ("unlikely to be flagged", "#16a34a"),
    "elevated": ("possibly flagged", "#d97706"),
    "high": ("likely to be flagged", "#dc2626"),
}


def _ai_likelihood_bands(badge: dict | None) -> dict:
    """Shared band->label/color mapping for the dual headline. Mirrors the JS
    ``aiLikelihoodBands`` in reportHelpers.js (kept in sync via the spec table)."""
    badge = badge or {}
    score = badge.get("ai_likelihood_score")
    tier = str(badge.get("tier") or "").upper()
    draftproof = None
    if isinstance(score, (int, float)):
        draftproof = {
            "score": round(score),
            "tier": tier or "AMBER",
            "color": _DRAFTPROOF_TIER_COLORS.get(tier, "#d97706"),
        }
    ext = badge.get("external_detector_estimate") or {}
    ext_score = ext.get("score")
    external = None
    if isinstance(ext_score, (int, float)):
        band = str(ext.get("band") or "").lower()
        label, color = _EXTERNAL_BAND_LABELS.get(band, ("estimated", "#475569"))
        external = {"score": round(ext_score), "band": band, "label": label, "color": color}
    return {"draftproof": draftproof, "external": external}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/poc" ~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_report_ai_likelihood_headline.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/report/render.py poc/test_report_ai_likelihood_headline.py
git commit -m "feat(report): _ai_likelihood_bands helper for dual-headline rendering"
```

---

## Task 2: Markdown dual-headline + demote derivations (drives the PDF)

**Files:**
- Modify: `poc/report/render.py` — add `_render_ai_likelihood_headline(badge)`; call it in `render_markdown` at the top of the AI/transformation section; relabel the existing calibrated/contribution block under a "How DraftProof calibrates this" subheading.
- Test: `poc/test_report_ai_likelihood_headline.py` (add cases)

- [ ] **Step 1: Write the failing test (add to the existing file)**

```python
from report.render import render_markdown
from report.report import DraftReport, PredictabilitySummary, Tier


def _report_with_badge(ext=True):
    badge = {
        "ai_likelihood_score": 42.0, "tier": "AMBER",
        "authorship_rating_label": "Possible AI-Assisted",
        "transformation_classification": {"label": "AI-stitched / patchwork", "confidence": "high"},
    }
    if ext:
        badge["external_detector_estimate"] = {"score": 59.8, "band": "high", "note": "x"}
    return DraftReport(
        overall_tier=Tier.MEDIUM, finding_count=0, findings_by_tier={},
        original_text="Some essay text about education and technology today.",
        predictability=PredictabilitySummary(
            overall_risk=0.42, risk_distribution={}, sentences=[], style_shifts=[], generic_phrases_found=[]),
        ai_risk_badge=badge,
    )


def test_markdown_shows_both_numbers_and_ordering():
    md = render_markdown(_report_with_badge(ext=True))
    assert "AI Likelihood" in md
    assert "42%" in md and "~60%" in md
    assert "likely to be flagged" in md
    # dual headline must precede the calibration breakdown
    assert md.index("AI Likelihood") < md.index("How DraftProof calibrates this")


def test_markdown_external_unavailable_fallback():
    md = render_markdown(_report_with_badge(ext=False))
    assert "External estimate unavailable" in md
    assert "42%" in md  # DraftProof number still shown, no crash
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/poc" ~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_report_ai_likelihood_headline.py -q`
Expected: FAIL — markdown lacks "AI Likelihood" / "How DraftProof calibrates this".

- [ ] **Step 3: Implement `_render_ai_likelihood_headline` in `render.py`**

```python
_AI_LIKELIHOOD_WHY = (
    "DraftProof is false-positive-averse, so it avoids wrongly accusing human writers. "
    "Strict detectors (Turnitin, GPTZero) weight raw token predictability far more "
    "aggressively. For a genuine pass, finish the draft in your own words."
)


def _render_ai_likelihood_headline(badge: dict | None) -> str:
    bands = _ai_likelihood_bands(badge)
    dp = bands["draftproof"]
    if not dp:
        return ""
    badge = badge or {}
    out = ["## AI Likelihood", ""]
    out.append(f"- **DraftProof (conservative): {dp['score']}% — {dp['tier']}**")
    ext = bands["external"]
    if ext:
        out.append(f"- **Turnitin / external: ~{ext['score']}% — {ext['label']}**")
    else:
        out.append("- _External estimate unavailable — re-scan to populate._")
    out.append("")
    tc = badge.get("transformation_classification") or {}
    meta = []
    if tc.get("label"):
        conf = tc.get("confidence")
        meta.append(f"Pattern: {tc['label']}" + (f" ({conf} confidence)" if conf else ""))
    if badge.get("authorship_rating_label"):
        meta.append(str(badge["authorship_rating_label"]))
    if meta:
        out.append(" · ".join(meta))
        out.append("")
    out.append(_AI_LIKELIHOOD_WHY)
    out.append("")
    return "\n".join(out)
```

- [ ] **Step 4: Wire into `render_markdown` (`render.py:1508`)**

In `render_markdown`, locate where the AI/transformation section begins (search for the transformation contribution emission — the block that prints `calibrated_ai_risk` / the "Turnitin reference:" footnote at module line ~154, or the contribution summary). Insert the headline at the START of that section:

```python
        # Dual-headline AI likelihood (DraftProof vs external) — leads the AI section.
        headline = _render_ai_likelihood_headline(report.ai_risk_badge)
        if headline:
            sections.append(headline)   # use the local accumulator name used in render_markdown
```

Then, immediately before the existing calibrated/contribution/suppression lines, emit the demotion subheading so the derivations read as secondary:

```python
        sections.append("### How DraftProof calibrates this")
```

(Place the `###` heading before the existing contribution/`calibrated_ai_risk` block; do not delete that block — only relabel/demote it. Match the local variable name `render_markdown` already uses to accumulate output.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH="$PWD:$PWD/poc" ~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_report_ai_likelihood_headline.py -q`
Expected: PASS (7 tests)

- [ ] **Step 6: Smoke-render a real PDF to confirm no WeasyPrint breakage**

Run:
```bash
PYTHONPATH=. DRAFTPROOF_V6_DETERMINISTIC=1 ~/.pyenv/versions/3.11.0/bin/python3 -c "
from pathlib import Path; from poc.detect_pipeline import run_detect
import json
det = run_detect(Path('test_content11.txt').read_text(), output_dir='test_output/_dualhead')
md = Path(det['json_path']).parent
print('OK', det.get('tier'))
"
```
Expected: prints `OK medium` and writes `report.pdf`/`report.md` under the output dir without exception. Open `report.md` and confirm the "AI Likelihood" block leads and shows `42%` + `~60%`.

- [ ] **Step 7: Commit**

```bash
git add poc/report/render.py poc/test_report_ai_likelihood_headline.py
git commit -m "feat(report): dual-headline AI-likelihood in markdown/PDF; demote derivations"
```

---

## Task 3: Frontend band helper `aiLikelihoodBands`

**Files:**
- Modify: `draftproof-frontend/src/pages/report/reportHelpers.js` (add + export the helper; mirror of the Python one but returns band KEY — the page resolves the label via i18n)

- [ ] **Step 1: Add the helper to `reportHelpers.js`**

```javascript
const DRAFTPROOF_TIER_COLORS = { GREEN: '#16a34a', AMBER: '#d97706', ORANGE: '#ea580c', RED: '#dc2626' };
const EXTERNAL_BAND_COLORS = { low: '#16a34a', elevated: '#d97706', high: '#dc2626' };

// Mirror of report.render._ai_likelihood_bands (same band table; see spec). Returns the
// external band KEY; the label is resolved via i18n (report.aiLikelihood.externalBand.<band>).
function aiLikelihoodBands(badge) {
  const b = badge || {};
  const score = b.ai_likelihood_score;
  const tier = String(b.tier || '').toUpperCase();
  const draftproof = typeof score === 'number'
    ? { score: Math.round(score), tier: tier || 'AMBER', color: DRAFTPROOF_TIER_COLORS[tier] || '#d97706' }
    : null;
  const ext = b.external_detector_estimate || {};
  const band = String(ext.band || '').toLowerCase();
  const external = typeof ext.score === 'number'
    ? { score: Math.round(ext.score), band, color: EXTERNAL_BAND_COLORS[band] || '#475569', note: ext.note || '' }
    : null;
  return { draftproof, external };
}
```

Add `aiLikelihoodBands` to the file's export list (the bottom `export { ... }` block that already exports `buildSubmittedContentModel`).

- [ ] **Step 2: Build to verify no syntax error**

Run: `cd draftproof-frontend && export PATH="/opt/homebrew/bin:$PATH" && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Verify the mapping against the spec table (one-off node check)**

Run (from `draftproof-frontend`):
```bash
node --input-type=module -e "
const DRAFTPROOF_TIER_COLORS={GREEN:'#16a34a',AMBER:'#d97706',ORANGE:'#ea580c',RED:'#dc2626'};
const EXTERNAL_BAND_COLORS={low:'#16a34a',elevated:'#d97706',high:'#dc2626'};
function f(b){b=b||{};const s=b.ai_likelihood_score,t=String(b.tier||'').toUpperCase();
const dp=typeof s==='number'?{score:Math.round(s),tier:t||'AMBER',color:DRAFTPROOF_TIER_COLORS[t]||'#d97706'}:null;
const e=b.external_detector_estimate||{},bd=String(e.band||'').toLowerCase();
const ex=typeof e.score==='number'?{score:Math.round(e.score),band:bd,color:EXTERNAL_BAND_COLORS[bd]||'#475569'}:null;
return {draftproof:dp,external:ex};}
const o=f({ai_likelihood_score:42,tier:'AMBER',external_detector_estimate:{score:59.8,band:'high'}});
console.assert(o.draftproof.color==='#d97706'&&o.draftproof.score===42,'dp');
console.assert(o.external.color==='#dc2626'&&o.external.score===60&&o.external.band==='high','ext');
console.log('band mapping OK');
"
```
Expected: prints `band mapping OK` (the inlined copy must match the table — confirms the logic before wiring into JSX).

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/pages/report/reportHelpers.js
git commit -m "feat(frontend): aiLikelihoodBands helper (mirror of render._ai_likelihood_bands)"
```

---

## Task 4: i18n copy (`report.aiLikelihood.*`, en + zh)

**Files:**
- Modify: `draftproof-frontend/src/i18n/resources.js`

- [ ] **Step 1: Add the `aiLikelihood` block under `en.translation.report`**

```javascript
        aiLikelihood: {
          title: 'AI Likelihood',
          draftproof: 'DraftProof (conservative)',
          external: 'Turnitin / external',
          externalBand: {
            low: 'unlikely to be flagged',
            elevated: 'possibly flagged',
            high: 'likely to be flagged',
          },
          externalUnavailable: 'External estimate unavailable — re-scan to populate.',
          whyDiffer: 'DraftProof is false-positive-averse, so it avoids wrongly accusing human writers. Strict detectors (Turnitin, GPTZero) weight raw token predictability far more aggressively. For a genuine pass, finish the draft in your own words.',
          calibrateHeading: 'How DraftProof calibrates this',
        },
```

- [ ] **Step 2: Add the matching block under `zh.translation.report`**

```javascript
        aiLikelihood: {
          title: 'AI 可能性',
          draftproof: 'DraftProof（保守估计）',
          external: 'Turnitin / 外部检测器',
          externalBand: {
            low: '不太可能被标记',
            elevated: '可能被标记',
            high: '很可能被标记',
          },
          externalUnavailable: '暂无外部估计——重新扫描以生成。',
          whyDiffer: 'DraftProof 倾向于避免误判，因此不会轻易将人类写作判定为 AI。严格的检测器（Turnitin、GPTZero）对原始词元可预测性的权重要高得多。若要真正通过检测，请用自己的话完成稿件。',
          calibrateHeading: 'DraftProof 如何校准此结果',
        },
```

- [ ] **Step 3: Build to verify resources parse**

Run: `cd draftproof-frontend && export PATH="/opt/homebrew/bin:$PATH" && npm run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/i18n/resources.js
git commit -m "feat(i18n): report.aiLikelihood dual-headline copy (en + zh)"
```

---

## Task 5: CSS for the dual-headline block + collapse

**Files:**
- Modify: `draftproof-frontend/src/styles/site-master.css`

- [ ] **Step 1: Append styles (near the existing `.transformation-detail` rules)**

```css
.ai-likelihood-block {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 16px;
  background: #fbfdff;
}
.ai-likelihood-pair {
  display: flex;
  gap: 32px;
  flex-wrap: wrap;
}
.ai-likelihood-metric { min-width: 160px; }
.ai-likelihood-metric .ai-likelihood-caption {
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: #64748b;
}
.ai-likelihood-metric .ai-likelihood-score {
  font-size: 2rem;
  font-weight: 700;
  line-height: 1.1;
}
.ai-likelihood-metric .ai-likelihood-band { font-size: 0.85rem; color: #475569; }
.ai-likelihood-meta { margin-top: 10px; font-size: 0.9rem; color: #334155; }
.ai-likelihood-why { margin-top: 8px; font-size: 0.85rem; color: #64748b; }
.ai-likelihood-unavailable { font-size: 0.85rem; color: #94a3b8; font-style: italic; }
details.ai-likelihood-calibration { margin-top: 14px; }
details.ai-likelihood-calibration > summary {
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  color: #475569;
}
details.ai-likelihood-calibration[open] > summary { margin-bottom: 10px; }
```

- [ ] **Step 2: Build to verify CSS parses**

Run: `cd draftproof-frontend && export PATH="/opt/homebrew/bin:$PATH" && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add draftproof-frontend/src/styles/site-master.css
git commit -m "style(frontend): dual-headline AI-likelihood block + calibration collapse"
```

---

## Task 6: Render the dual headline in `Report.jsx`, collapse derivations

**Files:**
- Modify: `draftproof-frontend/src/pages/Report.jsx`
  - Import `aiLikelihoodBands` (add to the existing import from `./report/reportHelpers`, alongside `buildSubmittedContentModel`).
  - Remove the standalone external-estimate stat at **1346-1359** (absorbed into the new block).
  - In `renderTransformationDetails` (1378+): render the dual-headline block at the top; wrap the contribution/adjustment rows (the `transformation-ratio-summary` block, 1401-1424+) in `<details className="ai-likelihood-calibration">` with a `<summary>` of `t('report.aiLikelihood.calibrateHeading')`.

- [ ] **Step 1: Add the import**

Find the existing import (around line 55):
```javascript
import { buildSubmittedContentModel } from './report/reportHelpers';
```
Change to include the new helper (match the actual existing import shape — it may be a multi-name import):
```javascript
import { buildSubmittedContentModel, aiLikelihoodBands } from './report/reportHelpers';
```

- [ ] **Step 2: Build the dual-headline JSX as a local render fn inside the component**

Add near the other render helpers (e.g. just above `renderTransformationDetails` at 1378). `t` is in scope inside the component:

```jsx
  const renderAiLikelihoodHeadline = (variantBadge) => {
    const bands = aiLikelihoodBands(variantBadge);
    if (!bands.draftproof) return null;
    const dp = bands.draftproof;
    const ext = bands.external;
    const tc = variantBadge?.transformation_classification || {};
    const ratingLabel = variantBadge?.authorship_rating_label;
    return (
      <div className="ai-likelihood-block">
        <div className="ai-likelihood-caption">{t('report.aiLikelihood.title')}</div>
        <div className="ai-likelihood-pair">
          <div className="ai-likelihood-metric">
            <div className="ai-likelihood-caption">{t('report.aiLikelihood.draftproof')}</div>
            <div className="ai-likelihood-score" style={{ color: dp.color }}>{dp.score}%</div>
            <div className="ai-likelihood-band">{dp.tier}</div>
          </div>
          <div className="ai-likelihood-metric">
            <div className="ai-likelihood-caption">{t('report.aiLikelihood.external')}</div>
            {ext ? (
              <>
                <div className="ai-likelihood-score" style={{ color: ext.color }}>~{ext.score}%</div>
                <div className="ai-likelihood-band">{t(`report.aiLikelihood.externalBand.${ext.band}`, { defaultValue: '' })}</div>
              </>
            ) : (
              <div className="ai-likelihood-unavailable">{t('report.aiLikelihood.externalUnavailable')}</div>
            )}
          </div>
        </div>
        <div className="ai-likelihood-meta">
          {tc.label ? `${tc.label}${tc.confidence ? ` (${tc.confidence})` : ''}` : null}
          {ratingLabel ? `  ·  ${ratingLabel}` : null}
        </div>
        <div className="ai-likelihood-why">{t('report.aiLikelihood.whyDiffer')}</div>
      </div>
    );
  };
```

- [ ] **Step 3: Render the headline + collapse the derivations in `renderTransformationDetails`**

Inside `renderTransformationDetails`, immediately after the opening `<div className="transformation-detail ...">` and before `<div className="transformation-detail-head">`, insert:
```jsx
        {variant === 'original' && renderAiLikelihoodHeadline(originalComparisonBadge)}
```
(For the rewritten variant pass `rewrittenBadge`; if the rewritten path is not in scope for this change, gate on `variant === 'original'` as shown so only the original scan gets the new block.)

Wrap the existing `{summary && ( <div className="transformation-ratio-summary"> ... </div> )}` block (1401-1424+) in a `<details>`:
```jsx
        {summary && (
          <details className="ai-likelihood-calibration">
            <summary>{t('report.aiLikelihood.calibrateHeading')}</summary>
            <div className="transformation-ratio-summary">
              {/* ...existing contents unchanged... */}
            </div>
          </details>
        )}
```

- [ ] **Step 4: Remove the now-redundant standalone external stat (1346-1359)**

Delete the `{externalDetectorEstimate?.score != null && ( <div className="report-stat" ...>...</div> )}` block (the external estimate is now in the dual headline). Leave the surrounding `rawAuthorshipSignal` and `writingScore` stats intact.

- [ ] **Step 5: Build**

Run: `cd draftproof-frontend && export PATH="/opt/homebrew/bin:$PATH" && npm run build`
Expected: build succeeds, no unused-var error for `externalDetectorEstimate` (if it becomes unused after Step 4, remove its declaration at line 844 too).

- [ ] **Step 6: Visual verification**

Use the project `run`/`verify` flow or `npm run dev` and open a completed scan report. Confirm: the AI-likelihood block leads with DraftProof % + Turnitin ~% (two colored numbers), the why-line shows, and "How DraftProof calibrates this" is a collapsed expander containing the old gauges. Confirm an old report (no external estimate) shows the DraftProof number + the "unavailable" line without error.

- [ ] **Step 7: Commit**

```bash
git add draftproof-frontend/src/pages/Report.jsx
git commit -m "feat(frontend): lead scan report with dual-headline AI-likelihood; collapse derivations"
```

---

## Task 7: Final sync check + push

- [ ] **Step 1: Confirm page and PDF agree**

Re-render the PDF (Task 2 Step 6) and compare `report.md`'s "AI Likelihood" block against the page: same DraftProof %, same external ~%, same band label wording, same "why" copy, derivations demoted on both. Note any drift and fix.

- [ ] **Step 2: Run the Python suite for the touched area**

Run: `PYTHONPATH="$PWD:$PWD/poc" ~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_report_ai_likelihood_headline.py poc/test_document_structure.py -q`
Expected: all pass.

- [ ] **Step 3: Push (follow git push protocol)**

```bash
git fetch origin
git rev-list --count HEAD..origin/main   # expect 0
git push origin main
```

---

## Notes for the implementer

- **Do not** change `ai_likelihood_score`, `external_detector_estimate`, calibration, suppression, or tier math. This is presentation only.
- Use the RAW `tc.label` for the pattern on the page (NOT `transformationLabel(...)`), because the PDF (render.py) emits the raw label — using the raw label on both surfaces guarantees page⇄PDF sync.
- Keep the band table identical to the spec on both sides. The Python test (Task 1) is the canonical guard; if you change a color/label, change it in the spec, render.py, reportHelpers.js, and i18n together.
- If `externalDetectorEstimate` (Report.jsx:844) becomes unused after Task 6 Step 4, remove the declaration to keep the build clean.
