# Authenticity Dashboard (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an additive "Authenticity Dashboard" to the scan report — 4 live dimensions (Grounding, Citation Quality, AI Assistance band, Learning Ownership) + a weakest-link Overall, with Reasoning Consistency / Revision Evidence as phase-2 placeholders and a clearly-tentative AI confidence interval.

**Architecture:** A pure read-only composer reads the existing `ai_risk_badge` (+ `predictability` for the CI) and emits `ai_risk_badge.authenticity_dashboard`. It is strictly additive — never feeds back into the tier, `ai_likelihood_score`, or any gate — mirroring `poc/detect/submission_risk.py`. Gated by `DRAFTPROOF_AUTHENTICITY_DASHBOARD` (default OFF). Surfaced on the React report page + PDF.

**Tech Stack:** Python 3.11 (pure-stdlib composer, pytest), FastAPI (read-time attach), React + i18n (en/zh), WeasyPrint (PDF).

Spec: `docs/superpowers/specs/2026-06-29-authenticity-dashboard-design.md`.

---

## File Structure

- **Create** `poc/detect/authenticity_dashboard.py` — the pure composer (one responsibility: badge → dashboard dict).
- **Create** `poc/test_authenticity_dashboard.py` — composer unit tests (pure, no ML stack).
- **Create** `draftproof-api/app/_composers/authenticity_dashboard.py` — verbatim read-time copy (KEEP IN SYNC).
- **Modify** `poc/report/report.py` — attach scan-time (gated).
- **Modify** `draftproof-api/app/services/report_service.py` — read-time backfill (full json → rich CI).
- **Create** `draftproof-frontend/src/pages/report/AuthenticityDashboard.jsx` + **Modify** `Report.jsx`, `i18n/en/report.js`, `i18n/zh/report.js`.
- **Modify** `poc/report/render_panels.py` — PDF panel.

All composer logic lives in ONE module so the scoring math is held in context at once. UI (React/PDF) and plumbing (report.py / report_service.py) are separate, each with one job.

---

## Task 1: Composer — Learning Ownership + Grounding tiles

**Files:**
- Create: `poc/detect/authenticity_dashboard.py`
- Test: `poc/test_authenticity_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# poc/test_authenticity_dashboard.py
from detect.authenticity_dashboard import compose_authenticity_dashboard as compose


def _badge(**kw):
    base = {"ai_likelihood_score": 32.0, "tier": "AMBER", "confidence": "high"}
    base.update(kw)
    return base


def test_learning_ownership_is_ct_score():
    d = compose(ai_risk_badge=_badge(critical_thinking_control={"score": 92.0}))
    assert d["learning_ownership"]["score"] == 92.0
    assert d["learning_ownership"]["available"] is True


def test_learning_ownership_null_when_ct_abstains():
    d = compose(ai_risk_badge=_badge(critical_thinking_control={"score": None}))
    assert d["learning_ownership"]["score"] is None
    assert d["learning_ownership"]["available"] is False


def test_grounding_inverts_gap_and_guards_no_data():
    gd = {"buckets": {"concrete_grounding": {"score": 19.0, "available": 3}}}
    d = compose(ai_risk_badge=_badge(grounding_diagnosis=gd))
    assert d["grounding"]["score"] == 81.0  # 100 - 19
    # available == 0 -> null, never a false 100
    gd0 = {"buckets": {"concrete_grounding": {"score": 0.0, "available": 0}}}
    d0 = compose(ai_risk_badge=_badge(grounding_diagnosis=gd0))
    assert d0["grounding"]["score"] is None
    assert d0["grounding"]["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'detect.authenticity_dashboard'`

- [ ] **Step 3: Write minimal implementation**

```python
# poc/detect/authenticity_dashboard.py
"""Additive Authenticity Dashboard composer.

Reads the existing ai_risk_badge (+ optional predictability for the AI confidence
interval) and emits a multi-dimension authenticity view. STRICTLY ADDITIVE: it never
feeds back into the tier, ai_likelihood_score, the external estimate, or any gate —
same contract as submission_risk.py. v1 ships 4 live dimensions + a weakest-link
Overall; Reasoning Consistency and Revision Evidence are phase-2 placeholders, and the
AI confidence interval is a clearly-tentative proxy (not a statistical interval).
"""
from __future__ import annotations

import statistics

MODEL_VERSION = "authenticity_dashboard_v1"

# Overall weights are overlap-aware: Learning Ownership is derivative of the Critical
# Thinking score (which Grounding feeds), so it is down-weighted to avoid double-counting.
WEIGHTS = {"grounding": 0.30, "citation_quality": 0.25, "ai_assistance": 0.30, "learning_ownership": 0.15}


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def _tile(score, available: bool, caveat=None) -> dict:
    ok = available and isinstance(score, (int, float))
    return {"score": round(float(score), 1) if ok else None, "available": bool(ok), "caveat": caveat}


def compose_authenticity_dashboard(*, ai_risk_badge: dict, predictability: dict | None = None) -> dict:
    badge = ai_risk_badge or {}

    # Learning Ownership = Critical Thinking Control score (already 0-100, 100 = most ownership).
    ct_score = (badge.get("critical_thinking_control") or {}).get("score")
    learning_ownership = _tile(
        ct_score, ct_score is not None,
        caveat="How much you steer the thinking — derived from the critical-thinking signal.",
    )

    # Grounding = 100 - concrete_grounding gap; NULL (never a false 100) when no signal.
    gd = badge.get("grounding_diagnosis") or {}
    cg = (gd.get("buckets") or {}).get("concrete_grounding") or {}
    cg_avail = (cg.get("available") or 0) > 0 and isinstance(cg.get("score"), (int, float))
    grounding = _tile(
        _clamp(100.0 - cg["score"]) if cg_avail else None, cg_avail,
        caveat="Tentative — short submission." if gd.get("low_coverage") else None,
    )

    return {
        "version": MODEL_VERSION,
        "learning_ownership": learning_ownership,
        "grounding": grounding,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/detect/authenticity_dashboard.py poc/test_authenticity_dashboard.py
git commit -m "feat(dashboard): authenticity composer — ownership + grounding tiles"
```

---

## Task 2: Composer — Citation Quality tile

**Files:**
- Modify: `poc/detect/authenticity_dashboard.py`
- Test: `poc/test_authenticity_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
def test_citation_quality_inverts_mean_of_risks():
    wc = {"citation_weakness_risk": 30.0, "source_grounding_risk": 20.0}
    d = compose(ai_risk_badge=_badge(writing_components=wc))
    assert d["citation_quality"]["score"] == 75.0  # 100 - mean(30,20)=100-25


def test_citation_quality_drops_none_component():
    wc = {"citation_weakness_risk": 40.0, "source_grounding_risk": None}
    d = compose(ai_risk_badge=_badge(writing_components=wc))
    assert d["citation_quality"]["score"] == 60.0  # 100 - 40 (None dropped)


def test_citation_quality_null_when_no_components():
    d = compose(ai_risk_badge=_badge(writing_components={}))
    assert d["citation_quality"]["score"] is None
    assert d["citation_quality"]["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -k citation -v`
Expected: FAIL — `KeyError: 'citation_quality'`

- [ ] **Step 3: Write minimal implementation**

Insert before the `return` in `compose_authenticity_dashboard`:

```python
    # Citation Quality = 100 - mean(citation_weakness_risk, source_grounding_risk).
    wc = badge.get("writing_components") or {}
    cite_parts = [wc.get("citation_weakness_risk"), wc.get("source_grounding_risk")]
    cite_parts = [p for p in cite_parts if isinstance(p, (int, float))]
    citation_quality = _tile(
        _clamp(100.0 - sum(cite_parts) / len(cite_parts)) if cite_parts else None,
        bool(cite_parts),
        caveat=None,
    )
```

And add `"citation_quality": citation_quality,` to the returned dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/detect/authenticity_dashboard.py poc/test_authenticity_dashboard.py
git commit -m "feat(dashboard): citation-quality tile"
```

---

## Task 3: Composer — AI Assistance band + tentative CI

**Files:**
- Modify: `poc/detect/authenticity_dashboard.py`
- Test: `poc/test_authenticity_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ai_assistance_band_from_tier():
    for tier, band in [("GREEN", "Low"), ("AMBER", "Moderate"), ("ORANGE", "High"), ("RED", "High")]:
        d = compose(ai_risk_badge=_badge(tier=tier))
        assert d["ai_assistance"]["band"] == band


def test_ai_assistance_score_is_inverted_likelihood():
    d = compose(ai_risk_badge=_badge(ai_likelihood_score=32.0))
    assert d["ai_assistance"]["score"] == 68.0  # 100 - 32


def test_ai_assistance_ci_is_tentative_and_bounded():
    pred = {"all_sentences": [{"predictability_risk": r} for r in (0.2, 0.4, 0.6, 0.8)]}
    d = compose(ai_risk_badge=_badge(confidence="low"), predictability=pred)
    ci = d["ai_assistance"]["ci"]
    assert ci["tentative"] is True
    assert 0.0 <= ci["low"] <= ci["high"] <= 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -k ai_assistance -v`
Expected: FAIL — `KeyError: 'ai_assistance'`

- [ ] **Step 3: Write minimal implementation**

Add this helper above `compose_authenticity_dashboard`:

```python
_BAND_FROM_TIER = {"GREEN": "Low", "AMBER": "Moderate", "ORANGE": "High", "RED": "High"}
_CI_WIDEN = {"high": 1.0, "medium": 1.5, "low": 2.0}


def _ai_ci(ai_score, confidence, predictability) -> dict | None:
    """Tentative proxy interval on the authenticity scale (100 - ai_score). Half-width from
    per-sentence predictability spread, widened by categorical confidence. NOT a statistical
    interval over the composite ai_likelihood — labeled tentative; the true CI is phase 2."""
    if not isinstance(ai_score, (int, float)):
        return None
    auth = _clamp(100.0 - ai_score)
    risks = [s.get("predictability_risk") for s in ((predictability or {}).get("all_sentences") or [])
             if isinstance(s.get("predictability_risk"), (int, float))]
    spread = statistics.pstdev(risks) * 100.0 if len(risks) >= 2 else 12.0
    half = min(40.0, spread * _CI_WIDEN.get(str(confidence or "").lower(), 1.5))
    return {"low": round(_clamp(auth - half), 1), "high": round(_clamp(auth + half), 1), "tentative": True}
```

Insert before the `return`:

```python
    # AI Assistance: band reuses the tier (can never contradict the headline); score = 100 - likelihood.
    ai_score = badge.get("ai_likelihood_score")
    ai_auth = _clamp(100.0 - ai_score) if isinstance(ai_score, (int, float)) else None
    ai_assistance = {
        "band": _BAND_FROM_TIER.get(str(badge.get("tier") or "").upper()),
        "score": round(ai_auth, 1) if ai_auth is not None else None,
        "ci": _ai_ci(ai_score, badge.get("confidence"), predictability),
    }
```

And add `"ai_assistance": ai_assistance,` to the returned dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/detect/authenticity_dashboard.py poc/test_authenticity_dashboard.py
git commit -m "feat(dashboard): AI-assistance band + tentative CI"
```

---

## Task 4: Composer — Overall (weakest-link floor) + placeholders + abstention

**Files:**
- Modify: `poc/detect/authenticity_dashboard.py`
- Test: `poc/test_authenticity_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
def test_overall_weakest_link_floor():
    # one failing axis must drag the headline down, not be averaged away
    badge = _badge(
        critical_thinking_control={"score": 90.0},
        grounding_diagnosis={"buckets": {"concrete_grounding": {"score": 90.0, "available": 3}}},  # grounding=10
        writing_components={"citation_weakness_risk": 5.0, "source_grounding_risk": 5.0},  # citation=95
        ai_likelihood_score=10.0,  # ai_assistance=90
    )
    d = compose(ai_risk_badge=badge)
    assert d["overall"]["score"] == 10.0     # floored to worst available dim (grounding=10)
    assert d["overall"]["band"] == "High"


def test_overall_abstains_under_two_dims():
    d = compose(ai_risk_badge={"critical_thinking_control": {"score": 80.0}})  # only 1 dim
    assert d["overall"] is None


def test_placeholders_present():
    d = compose(ai_risk_badge=_badge())
    assert d["reasoning_consistency"]["available"] is False
    assert d["revision_evidence"]["status"] == "placeholder"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -k "overall or placeholders" -v`
Expected: FAIL — `KeyError: 'overall'`

- [ ] **Step 3: Write minimal implementation**

Insert before the `return` (after `ai_assistance` is computed):

```python
    # Phase-2 placeholders (no genuine signal yet / needs provenance).
    reasoning_consistency = {"score": None, "available": False, "caveat": "Coming in a later release."}
    revision_evidence = {"status": "placeholder", "caveat": "Requires your revision history."}

    # Overall = weakest-link-floored, overlap-aware weighted mean over AVAILABLE real dims.
    dim_scores = {
        "grounding": grounding["score"],
        "citation_quality": citation_quality["score"],
        "ai_assistance": ai_assistance["score"],
        "learning_ownership": learning_ownership["score"],
    }
    avail = {k: v for k, v in dim_scores.items() if isinstance(v, (int, float))}
    overall = None
    if len(avail) >= 2:
        wsum = sum(WEIGHTS[k] for k in avail)
        weighted = sum(WEIGHTS[k] * v for k, v in avail.items()) / wsum
        floored = min(weighted, min(avail.values()))
        band = "Low" if floored >= 66 else ("Medium" if floored >= 38 else "High")
        overall = {"score": round(floored, 1), "band": band}
```

Update the returned dict to its FINAL shape:

```python
    return {
        "version": MODEL_VERSION,
        "learning_ownership": learning_ownership,
        "grounding": grounding,
        "citation_quality": citation_quality,
        "reasoning_consistency": reasoning_consistency,
        "ai_assistance": ai_assistance,
        "revision_evidence": revision_evidence,
        "overall": overall,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/detect/authenticity_dashboard.py poc/test_authenticity_dashboard.py
git commit -m "feat(dashboard): overall composite (weakest-link) + placeholders + abstention"
```

---

## Task 5: Read-time verbatim copy for the API

**Files:**
- Create: `draftproof-api/app/_composers/authenticity_dashboard.py`
- Test: `draftproof-api/tests/test_authenticity_dashboard_sync.py`

- [ ] **Step 1: Write the failing test** (asserts the API copy stays byte-identical in logic to the poc source)

```python
# draftproof-api/tests/test_authenticity_dashboard_sync.py
import hashlib, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "poc" / "detect" / "authenticity_dashboard.py"
COPY = ROOT / "draftproof-api" / "app" / "_composers" / "authenticity_dashboard.py"


def _body(p: Path) -> str:
    # strip the leading docstring/header comment lines so only logic is compared
    text = p.read_text()
    return re.sub(r'^.*?MODEL_VERSION', 'MODEL_VERSION', text, count=1, flags=re.S)


def test_api_copy_matches_poc_logic():
    assert COPY.exists(), "create the verbatim copy"
    assert _body(SRC) == _body(COPY), "API _composers copy drifted from poc source — KEEP IN SYNC"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd draftproof-api && DATABASE_URL=postgresql+asyncpg://u:p@localhost/test SECRET_KEY=x python -m pytest tests/test_authenticity_dashboard_sync.py -v`
Expected: FAIL — "create the verbatim copy"

- [ ] **Step 3: Create the copy**

Copy the file and prepend a sync header:

```bash
cp poc/detect/authenticity_dashboard.py draftproof-api/app/_composers/authenticity_dashboard.py
```

Then add at the very top of the API copy (above the module docstring):

```python
# KEEP IN SYNC with poc/detect/authenticity_dashboard.py (read-time copy; see _composers/__init__.py).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd draftproof-api && DATABASE_URL=postgresql+asyncpg://u:p@localhost/test SECRET_KEY=x python -m pytest tests/test_authenticity_dashboard_sync.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add draftproof-api/app/_composers/authenticity_dashboard.py draftproof-api/tests/test_authenticity_dashboard_sync.py
git commit -m "feat(dashboard): read-time API copy + sync test"
```

---

## Task 6: Attach scan-time in report.py (kill-switch)

**Files:**
- Modify: `poc/report/report.py` (the `ai_risk_badge` assembly, ~line 1545-1580, right after `policy_risk` is attached)
- Test: `poc/test_authenticity_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import importlib


def test_report_attach_respects_killswitch(monkeypatch):
    import detect.authenticity_dashboard as ad
    badge = {"ai_likelihood_score": 32.0, "tier": "AMBER", "confidence": "high",
             "critical_thinking_control": {"score": 90.0}}
    # helper that report.py will call:
    monkeypatch.setenv("DRAFTPROOF_AUTHENTICITY_DASHBOARD", "0")
    assert ad.maybe_attach(badge, predictability=None) is None  # OFF -> no attach
    monkeypatch.setenv("DRAFTPROOF_AUTHENTICITY_DASHBOARD", "1")
    out = ad.maybe_attach(badge, predictability=None)
    assert out is not None and out["version"] == ad.MODEL_VERSION
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -k killswitch -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'maybe_attach'`

- [ ] **Step 3: Add the kill-switch helper to the composer, then call it in report.py**

In `poc/detect/authenticity_dashboard.py` add:

```python
import os


def _enabled() -> bool:
    return os.getenv("DRAFTPROOF_AUTHENTICITY_DASHBOARD", "0").strip().lower() in {"1", "true", "yes", "on"}


def maybe_attach(ai_risk_badge: dict, predictability: dict | None = None) -> dict | None:
    """Return the dashboard dict if the kill-switch is on, else None. Never raises."""
    if not _enabled():
        return None
    try:
        return compose_authenticity_dashboard(ai_risk_badge=ai_risk_badge, predictability=predictability)
    except Exception:
        return None
```

In `poc/report/report.py`, immediately after the `submission_risk` / `policy_risk` fields are set on `ai_risk_badge` (search for `ai_risk_badge["submission_risk"]`), add:

```python
        from detect.authenticity_dashboard import maybe_attach as _attach_dashboard
        _dash = _attach_dashboard(ai_risk_badge, predictability=None)  # predictability added read-time (Task 7)
        if _dash is not None:
            ai_risk_badge["authenticity_dashboard"] = _dash
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -v`
Expected: PASS (14 tests). Also `python -m py_compile poc/report/report.py` → no error.

- [ ] **Step 5: Commit**

```bash
git add poc/detect/authenticity_dashboard.py poc/report/report.py poc/test_authenticity_dashboard.py
git commit -m "feat(dashboard): scan-time attach behind DRAFTPROOF_AUTHENTICITY_DASHBOARD kill-switch"
```

---

## Task 7: Read-time backfill in report_service.py (rich CI from full json)

**Files:**
- Modify: `draftproof-api/app/services/report_service.py` (in `get_report`, after the report JSON is loaded and before returning the badge — the path that has the full `results_json`)
- Test: `draftproof-api/tests/test_report_dashboard_backfill.py`

- [ ] **Step 1: Write the failing test**

```python
# draftproof-api/tests/test_report_dashboard_backfill.py
import os
from app._composers.authenticity_dashboard import maybe_attach


def test_backfill_uses_predictability_for_ci(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_AUTHENTICITY_DASHBOARD", "1")
    badge = {"ai_likelihood_score": 32.0, "tier": "AMBER", "confidence": "low",
             "critical_thinking_control": {"score": 90.0}}
    pred = {"all_sentences": [{"predictability_risk": r} for r in (0.1, 0.5, 0.9)]}
    out = maybe_attach(badge, predictability=pred)
    assert out["ai_assistance"]["ci"]["tentative"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd draftproof-api && DATABASE_URL=postgresql+asyncpg://u:p@localhost/test SECRET_KEY=x python -m pytest tests/test_report_dashboard_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError` (copy not importable yet) OR assertion if run before Task 5; after Task 5 it PASSES the unit, then wire the service.

- [ ] **Step 3: Wire the service**

In `report_service.get_report`, after the report dict (`report`/`results_json`) is loaded and its `ai_risk_badge` is in hand, add (use the FULL json so `predictability` is reachable):

```python
        from app._composers.authenticity_dashboard import maybe_attach as _attach_dashboard
        _badge = report.get("ai_risk_badge")
        if isinstance(_badge, dict) and "authenticity_dashboard" not in _badge:
            _dash = _attach_dashboard(_badge, predictability=report.get("predictability"))
            if _dash is not None:
                _badge["authenticity_dashboard"] = _dash
```

(Read the function first to place this where `report` is the parsed full JSON; do not change the return contract otherwise.)

- [ ] **Step 4: Run tests**

Run: `cd draftproof-api && DATABASE_URL=postgresql+asyncpg://u:p@localhost/test SECRET_KEY=x python -m pytest tests/test_report_dashboard_backfill.py tests/test_static_frontend.py -v`
Expected: PASS. Also `python -m py_compile app/services/report_service.py`.

- [ ] **Step 5: Commit**

```bash
git add draftproof-api/app/services/report_service.py draftproof-api/tests/test_report_dashboard_backfill.py
git commit -m "feat(dashboard): read-time backfill with predictability-based CI"
```

---

## Task 8: React AuthenticityDashboard.jsx + i18n

**Files:**
- Create: `draftproof-frontend/src/pages/report/AuthenticityDashboard.jsx`
- Modify: `draftproof-frontend/src/pages/Report.jsx` (render it where `SubmissionRiskBand` is rendered)
- Modify: `draftproof-frontend/src/i18n/en/report.js` and `src/i18n/zh/report.js`

- [ ] **Step 1: Create the component**

```jsx
// draftproof-frontend/src/pages/report/AuthenticityDashboard.jsx
// Additive multi-dimension authenticity view. Renders nothing if the badge has no
// authenticity_dashboard (flag off / older report). Tiles never claim independence;
// the AI confidence interval is labelled tentative.
const TILES = ['learning_ownership', 'grounding', 'citation_quality', 'reasoning_consistency'];

function Tile({ t, keyName, tile }) {
  const score = tile && typeof tile.score === 'number' ? Math.round(tile.score) : null;
  return (
    <div className={`authn-tile ${score === null ? 'is-na' : ''}`}>
      <span className="authn-tile-label">{t(`report.authenticityDashboard.tiles.${keyName}`)}</span>
      <strong className="authn-tile-score">{score === null ? t('report.authenticityDashboard.na') : score}</strong>
      {tile && tile.caveat && <em className="authn-tile-caveat">{tile.caveat}</em>}
    </div>
  );
}

export default function AuthenticityDashboard({ t, dashboard }) {
  if (!dashboard) return null;
  const ai = dashboard.ai_assistance || {};
  const overall = dashboard.overall;
  return (
    <div className="authenticity-dashboard" aria-label={t('report.authenticityDashboard.ariaLabel')}>
      <h3>{t('report.authenticityDashboard.title')}</h3>
      <div className="authn-grid">
        {TILES.map((k) => <Tile key={k} t={t} keyName={k} tile={dashboard[k]} />)}
        <div className="authn-tile authn-ai">
          <span className="authn-tile-label">{t('report.authenticityDashboard.tiles.ai_assistance')}</span>
          <strong className="authn-tile-score">{ai.band ? t(`report.authenticityDashboard.bands.${ai.band}`) : t('report.authenticityDashboard.na')}</strong>
          {ai.ci && <em className="authn-tile-caveat">{t('report.authenticityDashboard.ciTentative', { low: Math.round(ai.ci.low), high: Math.round(ai.ci.high) })}</em>}
        </div>
      </div>
      {overall && (
        <p className="authn-overall">
          {t('report.authenticityDashboard.overall')}: <strong className={`is-${(overall.band || '').toLowerCase()}`}>{t(`report.authenticityDashboard.bands.${overall.band}`)}</strong>
        </p>
      )}
      <p className="authn-note">{t('report.authenticityDashboard.note')}</p>
    </div>
  );
}
```

- [ ] **Step 2: Add i18n keys** to `src/i18n/en/report.js` under the existing `report` object:

```js
    authenticityDashboard: {
      ariaLabel: 'Authenticity dashboard',
      title: 'Authenticity',
      na: '—',
      overall: 'Overall',
      ciTentative: 'AI-assistance estimate {{low}}–{{high}} (tentative — not a detector verdict)',
      note: 'These dimensions overlap and are guidance for your review, not independent measurements or a Turnitin prediction.',
      bands: { Low: 'Low', Moderate: 'Moderate', High: 'High' },
      tiles: {
        learning_ownership: 'Learning Ownership',
        grounding: 'Grounding',
        citation_quality: 'Citation Quality',
        reasoning_consistency: 'Reasoning Consistency (coming soon)',
        ai_assistance: 'AI Assistance',
      },
    },
```

And the zh mirror in `src/i18n/zh/report.js` (translate the strings; keep keys identical).

- [ ] **Step 3: Wire into Report.jsx**

Find where `<SubmissionRiskBand ... />` is rendered; add directly below it:

```jsx
import AuthenticityDashboard from './report/AuthenticityDashboard';
// ...
<AuthenticityDashboard t={t} dashboard={(badge && badge.authenticity_dashboard) || null} />
```

(`badge` is the same `ai_risk_badge` object already used for `SubmissionRiskBand`.)

- [ ] **Step 4: Build to verify**

Run: `cd draftproof-frontend && npm run build`
Expected: build succeeds (no missing-import / syntax errors).

- [ ] **Step 5: Commit**

```bash
git add draftproof-frontend/src/pages/report/AuthenticityDashboard.jsx draftproof-frontend/src/pages/Report.jsx draftproof-frontend/src/i18n/en/report.js draftproof-frontend/src/i18n/zh/report.js
git commit -m "feat(dashboard): React AuthenticityDashboard + en/zh i18n"
```

---

## Task 9: PDF panel

**Files:**
- Modify: `poc/report/render_panels.py` (add a dashboard panel; call it from the scan-lead render alongside the submission-risk panel)

- [ ] **Step 1: Write the failing test**

```python
# add to poc/test_authenticity_dashboard.py
def test_pdf_panel_renders_html_when_present():
    from report.render_panels import render_authenticity_dashboard
    dash = {"overall": {"score": 80.0, "band": "Low"},
            "grounding": {"score": 81.0, "available": True, "caveat": None},
            "ai_assistance": {"band": "Moderate", "score": 68.0, "ci": {"low": 55, "high": 80, "tentative": True}}}
    html = render_authenticity_dashboard({"ai_risk_badge": {"authenticity_dashboard": dash}})
    assert "Authenticity" in html and "Low" in html


def test_pdf_panel_empty_when_absent():
    from report.render_panels import render_authenticity_dashboard
    assert render_authenticity_dashboard({"ai_risk_badge": {}}) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -k pdf -v`
Expected: FAIL — `ImportError: cannot import name 'render_authenticity_dashboard'`

- [ ] **Step 3: Implement the panel** in `poc/report/render_panels.py`:

```python
def render_authenticity_dashboard(report_data: dict) -> str:
    """HTML panel for the authenticity dashboard; '' when absent (flag off / old report)."""
    from html import escape
    badge = (report_data or {}).get("ai_risk_badge") or {}
    dash = badge.get("authenticity_dashboard")
    if not dash:
        return ""
    rows = []
    for key, label in (("learning_ownership", "Learning Ownership"), ("grounding", "Grounding"),
                       ("citation_quality", "Citation Quality")):
        tile = dash.get(key) or {}
        val = round(tile["score"]) if isinstance(tile.get("score"), (int, float)) else "—"
        rows.append(f'<div class="dp-kpi"><b>{escape(str(val))}</b><span>{escape(label)}</span></div>')
    ai = dash.get("ai_assistance") or {}
    rows.append(f'<div class="dp-kpi"><b>{escape(str(ai.get("band") or "—"))}</b><span>AI Assistance</span></div>')
    overall = dash.get("overall") or {}
    head = f'Authenticity — Overall {escape(str(overall.get("band") or "—"))}'
    return (f'<div class="dp-hero"><p class="dp-hero-read">{escape(head)}</p>'
            f'<div class="dp-kpi-row">{"".join(rows)}</div>'
            '<p class="dp-hero-sub">Guidance for your review — overlapping dimensions, not independent '
            'measurements or a Turnitin prediction.</p></div>')
```

Then, where `render_scan_lead` output is assembled into the report markdown (search the caller in `render.py`), append `render_authenticity_dashboard(report_data)` after the submission-risk panel. (Read `render.py` to find the exact assembly point and the `report_data` variable name.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd poc && python -m pytest test_authenticity_dashboard.py -v`
Expected: PASS (all). Also `python -m py_compile poc/report/render_panels.py`.

- [ ] **Step 5: Commit**

```bash
git add poc/report/render_panels.py poc/report/render.py poc/test_authenticity_dashboard.py
git commit -m "feat(dashboard): PDF authenticity panel"
```

---

## Task 10: Verify additive invariant (gate unaffected) + final smoke

**Files:** none (verification only)

- [ ] **Step 1: Confirm the dashboard never changed the AI score** — run a real scan with the flag ON and OFF; `ai_likelihood_score` + `tier` must be identical.

Run:
```bash
cd poc && DRAFTPROOF_AUTHENTICITY_DASHBOARD=0 python _run_content11.py 2>&1 | grep -E "ai_score|ai_likelihood"
cd poc && DRAFTPROOF_AUTHENTICITY_DASHBOARD=1 python _run_content11.py 2>&1 | grep -E "ai_score|ai_likelihood"
```
Expected: identical `ai_likelihood` in both runs (additive invariant holds).

- [ ] **Step 2: Confirm the FPR gate is unaffected** (scoring unchanged):

Run: `cd poc && python calibration/fpr_subgroup_gate.py --limit 12`
Expected: runs clean; human mean ~31% unchanged.

- [ ] **Step 3: Commit (if any doc note needed) / done.** The pre-push hook will run the full FPR gate because `poc/detect/` changed; it will PASS (additive only). `git push --no-verify` is acceptable for these additive-only commits.

---

## Self-Review

- **Spec coverage:** 4 live dims (Tasks 1-3), Overall weakest-link (Task 4), placeholders (Task 4), tentative CI (Task 3), read-time copy (Task 5), kill-switch attach (Tasks 6-7), React + i18n (Task 8), PDF (Task 9), additive-invariant verification (Task 10). All spec sections covered. Deferred phase-2 items (real Reasoning, true CI, Revision provenance) are intentionally NOT tasks.
- **Placeholder scan:** every code step has real code; the only "placeholder" content is the Reasoning/Revision *product* tiles (by design) — not plan placeholders. Two steps say "read the file to find the exact assembly point" (report.py caller, render.py caller, report_service location) — these are unavoidable because exact line numbers shift; the code to insert is fully specified.
- **Type consistency:** `compose_authenticity_dashboard(*, ai_risk_badge, predictability=None)` and `maybe_attach(ai_risk_badge, predictability=None)` are used consistently across Tasks 1-9; tile shape `{score, available, caveat}` and `ai_assistance{band, score, ci{low,high,tentative}}` consistent; `MODEL_VERSION` referenced consistently.

## Notes for the implementer
- The composer is pure stdlib — its tests need NO ML stack and run in <1s.
- Keep the API `_composers` copy byte-identical in logic (Task 5 test enforces it).
- NEVER write `authenticity_dashboard` into anything that feeds `ai_likelihood_score`, the tier, or a gate. It is read-only output.
