# Up-front Expectation Framing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A persistent framing banner atop the Rewrite page that states what DraftProof does/doesn't do and shows the rewritten content's honest external-detector estimate, so a residual flag reads as the expected gate, not a failure.

**Architecture:** Hoist the rewritten content's external-detector estimate (already computed inside `rewritten_scan_report.ai_risk_badge` by the full detect builder) to a top-level `summary["external_detector_estimate"]`; declare that field in the `RewriteReportOut` Pydantic model so it isn't stripped; render a framing `<section>` on `Rewrite.jsx` that always shows the copy and conditionally shows the estimate.

**Tech Stack:** Python (`poc/`), pytest, FastAPI/Pydantic (`draftproof-api/`), React + Vite + i18next (`draftproof-frontend/`).

**Spec:** `docs/superpowers/specs/2026-05-31-upfront-expectation-framing-design.md`

**Environment:** Python via `~/.pyenv/versions/3.11.0/bin/python3`. Frontend: `export PATH="/opt/homebrew/bin:$PATH"`. NEVER `git add -A` (≈220 untracked files); stage by explicit path.

**Note on a spec simplification:** the spec listed adding `external_detector_estimate` to `SCAN_BADGE_KEYS` (to survive detect-scan compaction). Deeper reading shows the SERVED rewrite report is the production `summary` dict (written to `json_path`), which uses the **full, uncompacted** `rewritten_scan_report` at production time — so compaction never touches the served estimate. `SCAN_BADGE_KEYS` is therefore **omitted (YAGNI)**.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `poc/rewrite_v6/production.py` | `_external_estimate_from_scan` helper + hoist into `summary` | Modify (import ~23; helper near other helpers; hoist ~195) |
| `poc/test_external_estimate_hoist.py` | Unit tests for the helper | Create |
| `draftproof-api/app/models/__init__.py` | Declare `external_detector_estimate` on `RewriteReportOut` | Modify (~line 99) |
| `draftproof-frontend/src/i18n/resources.js` | `rewriteFraming` keys (en + zh) | Modify |
| `draftproof-frontend/src/pages/Rewrite.jsx` | Framing banner render | Modify |

---

## Task 1: Backend — hoist the estimate + declare the field

**Files:**
- Modify: `poc/rewrite_v6/production.py`
- Modify: `draftproof-api/app/models/__init__.py`
- Test: `poc/test_external_estimate_hoist.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# poc/test_external_estimate_hoist.py
"""The rewrite summary must carry the rewritten content's external-detector estimate, hoisted from
the rewritten scan's badge (precomputed), or recomputed from its ai_components, else None."""
from poc.rewrite_v6.production import _external_estimate_from_scan


def test_prefers_precomputed_estimate():
    scan = {"ai_risk_badge": {"external_detector_estimate": {"score": 61.2, "band": "high"}}}
    assert _external_estimate_from_scan(scan) == {"score": 61.2, "band": "high"}


def test_recomputes_from_components_when_estimate_absent():
    scan = {"ai_risk_badge": {"ai_components": {"topk_pattern": 70.0, "predictability": 40.0, "burstiness_risk": 30.0}}}
    est = _external_estimate_from_scan(scan)
    assert est is not None and "score" in est and "band" in est


def test_none_when_minimal_or_missing():
    assert _external_estimate_from_scan({"ai_risk_badge": {"ai_likelihood_score": 10.0}}) is None
    assert _external_estimate_from_scan({}) is None
    assert _external_estimate_from_scan(None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_external_estimate_hoist.py -v`
Expected: FAIL — `ImportError: cannot import name '_external_estimate_from_scan'`

- [ ] **Step 3: Add the import in `poc/rewrite_v6/production.py`**

Find the existing import added by the authorship work (it imports from `report.authorship_evidence`). Near it / the other `report.*` imports, add a try/except import:
```python
try:
    from detect.layer3_scoring import estimate_external_detector_likelihood
except ImportError:
    from poc.detect.layer3_scoring import estimate_external_detector_likelihood
```

- [ ] **Step 4: Add the helper**

Add this function near the other module-level helpers in `production.py` (e.g. just above or below `_scan_report_shape` / `_scan_report_for_summary`):
```python
def _external_estimate_from_scan(scan_report: dict | None) -> dict | None:
    """The rewritten content's external-detector estimate: prefer the precomputed badge value, else
    recompute from the badge's ai_components, else None (graceful for minimal/missing scans)."""
    badge = scan_report.get("ai_risk_badge") if isinstance(scan_report, dict) else None
    badge = badge if isinstance(badge, dict) else {}
    estimate = badge.get("external_detector_estimate")
    if estimate:
        return estimate
    components = badge.get("ai_components")
    if isinstance(components, dict) and components:
        return estimate_external_detector_likelihood(components)
    return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_external_estimate_hoist.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Hoist into the summary**

In `run_rewrite_pipeline_v6`, find these two lines (~194-195, added by the authorship work):
```python
    authorship_evidence["preserved_ideas"] = preserved_idea_spans(original_text, final_text)
    summary["authorship_evidence"] = authorship_evidence
```
Immediately AFTER them, add:
```python
    summary["external_detector_estimate"] = _external_estimate_from_scan(rewritten_scan_report)
```
(`rewritten_scan_report` is the local built at ~line 136 and already stored as `summary["detect_scan_rewritten"]`.)

- [ ] **Step 7: Declare the field on `RewriteReportOut`**

In `draftproof-api/app/models/__init__.py`, find `class RewriteReportOut(BaseModel):` and its fields (it ends with `authorship_evidence: Optional[Any] = None`). Add one line after that:
```python
    external_detector_estimate: Optional[Any] = None
```

- [ ] **Step 8: Verify wiring + parse**

Run:
```bash
~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_external_estimate_hoist.py -v
~/.pyenv/versions/3.11.0/bin/python3 -c "import ast; ast.parse(open('poc/rewrite_v6/production.py').read()); ast.parse(open('draftproof-api/app/models/__init__.py').read()); print('parse OK')"
~/.pyenv/versions/3.11.0/bin/python3 -c "
src=open('poc/rewrite_v6/production.py').read()
assert 'summary[\"external_detector_estimate\"] = _external_estimate_from_scan(rewritten_scan_report)' in src
m=open('draftproof-api/app/models/__init__.py').read()
assert 'external_detector_estimate: Optional[Any] = None' in m and 'class RewriteReportOut' in m
print('wired OK')
"
```
Expected: 3 passed / `parse OK` / `wired OK`

- [ ] **Step 9: Commit**

```bash
git add poc/rewrite_v6/production.py poc/test_external_estimate_hoist.py draftproof-api/app/models/__init__.py
git commit -m "feat(rewrite): hoist rewritten external-detector estimate into the rewrite report (declared on RewriteReportOut)"
```

---

## Task 2: Frontend i18n keys (en + zh)

**Files:**
- Modify: `draftproof-frontend/src/i18n/resources.js`

- [ ] **Step 1: Add the `rewriteFraming` namespace under the `en` block**

Find the `en` translation object (sibling of the existing `rewritePage` / `authorshipEvidence` namespaces) and add, matching the existing indentation:
```javascript
      rewriteFraming: {
        title: 'Before you review this rewrite',
        isCopy: 'DraftProof mitigates AI-detection risk and shows you a reviewable draft to learn from — then edit it with your own specifics.',
        isntCopy: 'It is not a “make it pass” button. A residual estimate is expected: AI detectors score token predictability, which even strong, human-grounded writing can trigger.',
        estimateLabel: 'Honest external-detector estimate for this rewrite',
        estimateContext: 'Treat external detectors as probabilistic signals, not verdicts.',
        action: 'Review the before/after, then replace the highlighted additions with your own real content.',
      },
```

- [ ] **Step 2: Add the same namespace, translated, under the `zh` block**

```javascript
      rewriteFraming: {
        title: '在查看这份改写之前',
        isCopy: 'DraftProof 用于降低 AI 检测风险，并提供一份可供学习的改写草稿 —— 之后请用你自己的具体内容来修改。',
        isntCopy: '它不是「一键通过」的按钮。残留的检测估计是预期之内的：AI 检测器衡量的是用词的可预测性，即使是扎实、有真实依据的人类写作也可能被触发。',
        estimateLabel: '本次改写的诚实第三方检测器估计',
        estimateContext: '请将第三方检测器视为概率性信号，而非定论。',
        action: '对照前后差异，然后用你自己的真实内容替换高亮的补充部分。',
      },
```

- [ ] **Step 3: Verify the bundle parses**

```bash
export PATH="/opt/homebrew/bin:$PATH"
cd draftproof-frontend && node --check src/i18n/resources.js && echo "syntax OK"
```
Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/i18n/resources.js
git commit -m "feat(i18n): rewriteFraming expectation copy (en + zh)"
```

---

## Task 3: Frontend — the framing banner (`Rewrite.jsx`)

**Files:**
- Modify: `draftproof-frontend/src/pages/Rewrite.jsx`

- [ ] **Step 1: Derive the estimate near the other derived values**

Near the existing `const authorshipEvidence = ...` line (~241), add:
```jsx
  const externalEstimate = report?.external_detector_estimate || null;
```

- [ ] **Step 2: Add the banner at the top of the result body**

Find the `{error && <ErrorReload message={error} />}` line (~306) and the `{requiresManualReview && report?.final_text && (` block that follows (~308). Insert the framing banner BETWEEN them (so it renders once the report is loaded, above the manual-review alert and the diff):
```jsx
        {report?.final_text && (
          <section className="rewrite-review-section" aria-label={t('rewriteFraming.title')}>
            <div className="rewrite-review-heading">
              <div>
                <h3>{t('rewriteFraming.title')}</h3>
              </div>
            </div>
            <p className="rewrite-review-copy">{t('rewriteFraming.isCopy')}</p>
            <p className="rewrite-review-copy">{t('rewriteFraming.isntCopy')}</p>
            {externalEstimate?.score != null && (
              <p className="rewrite-review-copy">
                <strong>{t('rewriteFraming.estimateLabel')}: </strong>
                <strong
                  style={{ color: externalEstimate.band === 'high' ? '#dc2626' : externalEstimate.band === 'elevated' ? '#d97706' : '#16a34a' }}
                >
                  {`${Math.round(externalEstimate.score)}%`}
                </strong>
                {' — '}
                {t('rewriteFraming.estimateContext')}
              </p>
            )}
            <p className="rewrite-review-copy">{t('rewriteFraming.action')}</p>
          </section>
        )}
```
(Reuses existing `rewrite-review-section` / `rewrite-review-heading` / `rewrite-review-copy` classes — the same neutral block the authorship panel uses. `Math.round` avoids any dependency on a formatter not imported here.)

- [ ] **Step 3: Verify the build**

```bash
export PATH="/opt/homebrew/bin:$PATH"
cd draftproof-frontend && npm run build 2>&1 | tail -8
```
Expected: clean build (vite "built in" line, no error).

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/pages/Rewrite.jsx
git commit -m "feat(frontend): up-front expectation framing banner on the rewrite page"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** Component 1 (copy) → Task 2. Component 2 (wire estimate: hoist + declare) → Task 1 (helper + hoist + `RewriteReportOut` field). Component 3 (banner, always-copy + conditional estimate) → Task 3. Honesty guardrails (contextualized, graceful copy-only) → Task 3 conditional `externalEstimate?.score != null` + Task 2 copy. Verification §1 → Task 1 tests; §2 serialization-declare → Task 1 Step 7-8; §3 frontend build → Task 3 Step 3. Spec's `SCAN_BADGE_KEYS` step intentionally dropped (documented in the header note — served report uses the uncompacted scan).
- **Placeholder scan:** none — every code step has complete code; commands have expected output.
- **Type consistency:** `external_detector_estimate` is the key name across the hoist (Task 1 Step 6), the model field (Step 7), and the frontend read `report?.external_detector_estimate` (Task 3 Step 1). The estimate object shape `{score, band}` matches `estimate_external_detector_likelihood`'s return and is read identically (`.score`, `.band`) in Task 3, mirroring the existing `Report.jsx` render. i18n keys (`rewriteFraming.title/isCopy/isntCopy/estimateLabel/estimateContext/action`) are defined in Task 2 and consumed verbatim in Task 3.
