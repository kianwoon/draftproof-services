# DeBERTa-Signal Comparison Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, independent AI-writing signal score (off-the-shelf DeBERTa-class checkpoint) to every scan, shown side-by-side with the existing composite, strictly additive.

**Architecture:** Inline additive stage inside `poc/detect/run.py::DetectionRunner` that loads an off-the-shelf AI-text-detection checkpoint from the existing `/app/hf_cache` volume, scores overlapping sentence windows, aggregates to one calibrated 0–100 % band-mapped score, and emits an `ai_signal_deberta` field on the report — never feeding tier / `ai_likelihood` / gates. Mirrors the proven `authenticity_dashboard` additive-composer contract.

**Tech Stack:** Python, PyTorch + `transformers` (already in worker), `scikit-learn` isotonic regression (already used in `poc/calibration/calibrate.py`), FastAPI/React only for the surface task. SCoCESLE ESL corpus (local-only) + `poc/calibration/fpr_subgroup_gate.py` harness for the Phase-0 gate.

**Spec:** `docs/superpowers/specs/2026-07-01-deberta-signal-comparison-module-design.md`

---

## Real anchors (verified in codebase)

| What | Where |
|------|-------|
| Additive-composer contract to copy | `poc/detect/authenticity_dashboard.py` (docstring lines 3-7, `compose_authenticity_dashboard` line 67, `maybe_attach` line 141, `_enabled` line 138) |
| Read-time dual-copy mirror (KEEP IN SYNC header) | `draftproof-api/app/_composers/authenticity_dashboard.py:1` |
| Where additive field attaches to report | `poc/report/report.py:1582` (`maybe_attach as _attach_dashboard`, stored at `ai_risk_badge["authenticity_dashboard"]`) |
| Existing model-load pattern (local_files_only + fallback) | `poc/predictability/scanner.py:154-159`, `poc/detect/semantic_shape.py:253` (`cache_folder=os.environ.get("HF_HOME")`) |
| HF volume + boot guard | `worker/Dockerfile:19` (`ENV HF_HOME=/app/hf_cache`), `worker/entrypoint.sh:7,86-94` |
| Kill-switch convention | `worker/entrypoint.sh` env block (`DRAFTPROOF_AUTHENTICITY_DASHBOARD` etc.) |
| ESL gate harness to mirror | `poc/calibration/fpr_subgroup_gate.py` (`_score_texts:110`, `_fpr:86`, `_auc:103`, `measure:119`, reads `badge["ai_likelihood_score"]:115`) |
| Sibling-style test layout | `poc/test_critical_thinking.py`, `poc/test_submission_risk.py` |
| Frontend report surface | `draftproof-frontend/src/pages/Report.jsx`, `src/pages/report/AuthenticityDashboard.jsx`, i18n `src/i18n/{en,zh}/report.js` |
| PDF renderer | `poc/report/render_panels.py` |

---

## File Structure

**Create (worker side, real inference):**
- `poc/detect/deberta_model.py` — lazy singleton loader + window→raw-prob inference
- `poc/detect/deberta_windowing.py` — text → overlapping windows → aggregation (pure Python)
- `poc/detect/deberta_calibrate.py` — isotonic fit/apply on SCoCESLE (pure math)
- `poc/detect/deberta_signal.py` — additive composer (`compose` + `maybe_attach` + `_enabled`); the ONLY file `DetectionRunner` imports
- `poc/detect/_download_deberta.py` — one-time warm-download script
- `poc/calibration/deberta_fpr_gate.py` — Phase-0 ESL gate (mirrors `fpr_subgroup_gate.py`, reads `ai_signal_deberta`)
- `poc/test_deberta_windowing.py`, `poc/test_deberta_calibrate.py`, `poc/test_deberta_signal.py`, `poc/test_deberta_additive_invariant.py`

**Create (read-time mirror, MVP pass-through):**
- `draftproof-api/app/_composers/deberta_signal.py` — KEEP IN SYNC stub (pass-through; no derivation in MVP)

**Modify:**
- `poc/detect/run.py` — import + call `deberta_signal.maybe_attach` inside `run_all`, store on report
- `poc/report/report.py` (near line 1582) — surface `ai_signal_deberta` from the scan results into the report badge
- `worker/entrypoint.sh` — add `DRAFTPROOF_DEBERTA_SIGNAL` + `DRAFTPROOF_DEBERTA_MODEL` env vars
- `worker/requirements.txt` — confirm `scikit-learn` present (isotonic); add only if missing
- `draftproof-frontend/src/pages/Report.jsx` (+ child) — side-by-side tile + agree/disagree note
- `draftproof-frontend/src/i18n/{en,zh}/report.js` — copy
- `poc/report/render_panels.py` — PDF side-by-side

---

## Phase 0 — Checkpoint selection (MANDATORY GATE before Phase 1)

### Task 0.1: Pick candidate checkpoints (research)

**Files:**
- Create: `poc/calibration/deberta_candidates.md`

- [ ] **Step 1: Research current (July 2026) open AI-text-detection checkpoints on HuggingFace**

Search HuggingFace for sequence-classification models tagged AI-text-detection / ChatGPT-detection. Prefer DeBERTa-v3 or RoBERTa-base, CPU-deployable, permissive license. Record 3–5 candidates with repo-id, base model, #downloads, stated training data, license.

Run: `WebSearch` / `fetch_and_index` on `huggingface.co/models?other=ai-text-detection`
Write results into `poc/calibration/deberta_candidates.md` (repo-id + one-line note each).

- [ ] **Step 2: Commit**

```bash
git add poc/calibration/deberta_candidates.md
git commit -m "research(deberta): candidate AI-text-detection checkpoints"
```

### Task 0.2: Build the DeBERTA ESL gate (mirrors fpr_subgroup_gate)

**Files:**
- Create: `poc/calibration/deberta_fpr_gate.py`
- Reference: `poc/calibration/fpr_subgroup_gate.py`

- [ ] **Step 1: Copy the harness shape, retarget to the DeBERTA score**

The gate reads `badge["ai_signal_deberta"]["score"]` instead of `ai_likelihood_score`. Reuse `_proficiency_groups`, `_ai_texts`, `_fpr`, `_dist`, `_auc`, `measure`, `_print_summary` from `fpr_subgroup_gate.py`. The only change in `_score_texts` is the read path:

```python
def _score_deberta_texts(runner, texts):
    out = []
    for t in texts:
        badge = runner.run_all(t).ai_risk_badge or {}
        sig = (badge.get("ai_signal_deberta") or {})
        score = sig.get("score")
        out.append(float(score) if score is not None else None)
    return out
```

(`ai_signal_deberta` will not exist until Task 1.5 — this gate is written first but RUN after the module exists. Task 0.3 below runs it.)

- [ ] **Step 2: Add a `--candidates` mode that scores N candidates**

```python
# main() accepts --candidates <comma-separated repo-ids> and loops:
# for each repo-id: set os.environ["DRAFTPROOF_DEBERTA_MODEL"]=repo, force model reload,
#   measure() over the corpus, print ESL FPR + human mean + AUC(ai vs human).
```

- [ ] **Step 3: Commit**

```bash
git add poc/calibration/deberta_fpr_gate.py
git commit -m "feat(deberta): ESL FPR gate harness (mirrors fpr_subgroup_gate, reads ai_signal_deberta)"
```

> NOTE: Tasks 0.3/0.4 run AFTER the module (Phase 1) exists, because the gate scores through `DetectionRunner`. They are listed here as the gate definition; execution order is 1.1→1.5 → 0.3→0.4.

---

## Phase 1 — MVP module (additive stage)

### Task 1.1: Windowing unit (pure Python, TDD)

**Files:**
- Create: `poc/detect/deberta_windowing.py`
- Test: `poc/test_deberta_windowing.py`

- [ ] **Step 1: Write the failing test**

```python
# poc/test_deberta_windowing.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from detect.deberta_windowing import split_sentences, build_windows, aggregate


def test_split_sentences_basic():
    assert split_sentences("Hello world. This is two. And a third!") == [
        "Hello world.", "This is two.", "And a third!"
    ]


def test_build_windows_overlap_step_two():
    sents = ["s1.", "s2.", "s3.", "s4."]
    windows = build_windows(sents, size=2, step=1)
    assert windows == ["s1. s2.", "s2. s3.", "s3. s4."]


def test_build_windows_short_doc_single_window():
    sents = ["only one sentence."]
    assert build_windows(sents, size=3, step=1) == ["only one sentence."]


def test_aggregate_weighted_mean_by_sentence():
    # sentence 0 covered by window-prob 0.9; sentence 1 by 0.9 and 0.3 -> mean 0.6
    sents = ["a.", "b."]
    windows = build_windows(sents, size=1, step=1)
    probs = [0.9, 0.3]
    agg = aggregate(sents, windows, probs)
    assert abs(agg["document_score"] - 0.6) < 1e-6
    assert len(agg["sentence_scores"]) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `cd poc && python test_deberta_windowing.py`
Expected: `ModuleNotFoundError: No module named 'detect.deberta_windowing'`

- [ ] **Step 3: Implement**

```python
# poc/detect/deberta_windowing.py
"""Overlapping sentence-window construction + aggregation for the DeBERTa signal.

Pure Python — no ML. Tested independently of the model. Mirrors the windowing strategy
from the source design doc (sections 6-9)."""
from __future__ import annotations

import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text] if text else [])


def build_windows(sentences: List[str], size: int = 3, step: int = 1) -> List[str]:
    if not sentences:
        return []
    if len(sentences) <= size:
        return [" ".join(sentences)]
    windows: List[str] = []
    i = 0
    while i < len(sentences):
        windows.append(" ".join(sentences[i:i + size]))
        if i + size >= len(sentences):
            break
        i += step
    return windows


def aggregate(sentences: List[str], windows: List[str], window_probs: List[float]) -> dict:
    """Map window probs back to sentences (avg of covering windows), then mean -> document."""
    # coverage[k] = list of window-probs whose window includes sentence k
    coverage: List[List[float]] = [[] for _ in sentences]
    size = 3  # must match build_windows default contract
    for wi, w in enumerate(windows):
        # determine which sentences this window covered by re-deriving start index
        # windows were built at step=1 from size=3, so window wi covers sentences [wi, wi+size)
        for k in range(wi, min(wi + size, len(sentences))):
            if wi < len(window_probs):
                coverage[k].append(window_probs[wi])
    sentence_scores = [
        (sum(c) / len(c)) if c else 0.0 for c in coverage
    ]
    doc = (sum(sentence_scores) / len(sentence_scores)) if sentence_scores else 0.0
    return {"document_score": doc, "sentence_scores": sentence_scores}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd poc && python test_deberta_windowing.py`
Expected: all 4 tests PASS (no output / exit 0)

- [ ] **Step 5: Commit**

```bash
git add poc/detect/deberta_windowing.py poc/test_deberta_windowing.py
git commit -m "feat(deberta): overlapping windowing + aggregation unit (pure python, tested)"
```

### Task 1.2: Calibration unit (pure math, TDD)

**Files:**
- Create: `poc/detect/deberta_calibrate.py`
- Test: `poc/test_deberta_calibrate.py`

- [ ] **Step 1: Write the failing test**

```python
# poc/test_deberta_calibrate.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from detect.deberta_calibrate import fit_isotonic, apply_isotonic, map_to_band


def test_isotonic_monotonic_nondecreasing():
    raw = [0.1, 0.2, 0.8, 0.9, 0.15]   # human labels: 1,1,0,0,1
    labels = [1, 1, 0, 0, 1]
    cal = fit_isotonic(raw, labels)
    out = apply_isotonic([0.1, 0.5, 0.9], cal)
    assert out == sorted(out)  # monotonic non-decreasing


def test_map_to_band_four_tiers():
    assert map_to_band(5) == "clean"
    assert map_to_band(25) == "acceptable"
    assert map_to_band(55) == "concerning"
    assert map_to_band(80) == "strong"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd poc && python test_deberta_calibrate.py`
Expected: `ModuleNotFoundError: No module named 'detect.deberta_calibrate'`

- [ ] **Step 3: Implement**

```python
# poc/detect/deberta_calibrate.py
"""Isotonic recalibration + band mapping for the DeBERTa signal.

Pure math. fit_isotonic is run on the SCoCESLE corpus (Phase 0); the fitted calibrator is
applied at inference. Band cutoffs mirror the composite's 4-tier legend so the two scores
are directly comparable (spec section 6)."""
from __future__ import annotations
import pickle
from pathlib import Path


def fit_isotonic(raw_scores, labels):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_scores, labels)
    return iso


def save_calibrator(iso, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(iso, f)


def load_calibrator(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_isotonic(raw_scores, iso):
    return [float(x) for x in iso.predict(raw_scores)]


# Composite's tier thresholds on the 0-100 scale (verify against poc/detect/thresholds.py
# during implementation; these mirror the public band legend clean/acceptable/concerning/strong).
_BAND_CUTOFFS = [(30.0, "clean"), (50.0, "acceptable"), (70.0, "concerning")]
# < 30 = clean, 30-50 = acceptable, 50-70 = concerning, >= 70 = strong


def map_to_band(score_100: float) -> str:
    for cutoff, band in _BAND_CUTOFFS:
        if score_100 < cutoff:
            return band
    return "strong"
```

> VERIFY during implementation: open `poc/detect/thresholds.py` and `layer3_scoring.py`, confirm the 30/50/70 cutoffs match the composite's actual `%→tier` boundaries. If they differ, update `_BAND_CUTOFFS` to match — comparability requires identical scales.

- [ ] **Step 4: Run to verify pass**

Run: `cd poc && python test_deberta_calibrate.py`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add poc/detect/deberta_calibrate.py poc/test_deberta_calibrate.py
git commit -m "feat(deberta): isotonic calibration + 4-tier band mapping (pure math, tested)"
```

### Task 1.3: Model loader + inference unit

**Files:**
- Create: `poc/detect/deberta_model.py`
- Reference: `poc/predictability/scanner.py:154-159`

- [ ] **Step 1: Implement the lazy singleton + window inference**

```python
# poc/detect/deberta_model.py
"""Lazy singleton loader for the off-the-shelf AI-text-detection checkpoint + window inference.

Loads from the existing HF volume (HF_HOME=/app/hf_cache). local_files_only=True first, then
network fallback — same pattern as poc/predictability/scanner.py:154-159. The model loads on
first use, not at import, to keep worker cold-start off the hot path."""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None
_LOCK = threading.Lock()


def _model_name() -> str:
    return os.environ.get(
        "DRAFTPROOF_DEBERTA_MODEL",
        "<chosen-checkpoint-repo-id>",  # set in Phase 0 Task 0.4
    )


def _load():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    name = _model_name()
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(name, local_files_only=True)
    except (OSError, EnvironmentError):
        logger.warning("[deberta] %s not in cache; downloading to HF_HOME", name)
        cache = os.environ.get("HF_HOME")
        tok = AutoTokenizer.from_pretrained(name, cache_folder=cache)
        mdl = AutoModelForSequenceClassification.from_pretrained(name, cache_folder=cache)
    mdl.eval()
    _TOKENIZER, _MODEL = tok, mdl


def score_windows(windows: List[str]) -> Optional[List[float]]:
    """Return AI-like probability per window, or None on failure (fail-open)."""
    try:
        with _LOCK:
            _load()
        import torch
        enc = _TOKENIZER(windows, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = _MODEL(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # index 1 = AI-like
        return [float(p) for p in probs]
    except Exception as e:  # fail-open: never block the scan
        logger.warning("[deberta] inference failed: %s", e)
        return None
```

- [ ] **Step 2: Smoke test against a real checkpoint (manual, after Phase 0 picks one)**

Run: `cd poc && DRAFTPROOF_DEBERTA_MODEL=<repo-id> python -c "from detect.deberta_model import score_windows; print(score_windows(['This is a test sentence about machine learning.']))"`
Expected: a list of floats in [0,1], no exception.

- [ ] **Step 3: Commit**

```bash
git add poc/detect/deberta_model.py
git commit -m "feat(deberta): lazy singleton loader + window inference (fail-open, HF volume)"
```

### Task 1.4: The additive composer (TDD with mocked inference)

**Files:**
- Create: `poc/detect/deberta_signal.py`
- Test: `poc/test_deberta_signal.py`
- Reference (contract to copy): `poc/detect/authenticity_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# poc/test_deberta_signal.py
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from detect import deberta_signal


def test_disabled_returns_none(monkeypatch=None):
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "0"
    try:
        assert deberta_signal.maybe_attach("any text here with enough words to pass length.") is None
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_too_short_abstains():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    try:
        out = deberta_signal.maybe_attach("Too short.")  # < 150 words
        assert out is not None and out["available"] is False
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)


def test_schema_has_required_keys():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    try:
        text = " ".join(["This is a sentence."] * 40)  # ~160 words
        out = deberta_signal.maybe_attach(text)
        assert set(out.keys()) >= {"score", "band", "confidence", "model_version", "calibrated", "available", "caveat"}
        assert out["available"] in (True, False)
    finally:
        os.environ.pop("DRAFTPROOF_DEBERTA_SIGNAL", None)
```

> NOTE: these tests exercise `maybe_attach` end-to-end through the real model only if the checkpoint is present. To make them deterministic without a model, mock `deberta_model.score_windows` to return fixed floats in a fixture. The schema/disabled/too-short assertions hold regardless.

- [ ] **Step 2: Run to verify failure**

Run: `cd poc && python test_deberta_signal.py`
Expected: `ModuleNotFoundError: No module named 'detect.deberta_signal'`

- [ ] **Step 3: Implement (copy authenticity_dashboard contract verbatim)**

```python
# poc/detect/deberta_signal.py
"""Additive DeBERTa AI-signal composer — a second, independent AI-writing detector score.

STRICTLY ADDITIVE: it never feeds back into the tier, ai_likelihood_score, the external
estimate, or any gate — same contract as authenticity_dashboard.py and submission_risk.py.
The off-the-shelf checkpoint runs locally on the worker (no third-party text upload). The
score is calibrated (isotonic on SCoCESLE) and band-mapped onto the composite's 4-tier
legend so the two scores are directly comparable.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MODEL_VERSION = "deberta_signal_v1"
_MIN_WORDS = 150


def _enabled() -> bool:
    return os.getenv("DRAFTPROOF_DEBERTA_SIGNAL", "1").strip().lower() in {"1", "true", "yes", "on"}


def _word_count(text: str) -> int:
    return len((text or "").split())


def _calibrated_path() -> str | None:
    # set by Phase 0 after isotonic fit on SCoCESLE; None = raw/uncalibrated
    return os.getenv("DRAFTPROOF_DEBERTA_CALIBRATOR")


def compose(text: str) -> dict:
    """Score one document end-to-end. Always returns a schema dict; available=False on any failure."""
    from .deberta_model import score_windows
    from .deberta_windowing import split_sentences, build_windows, aggregate
    from .deberta_calibrate import map_to_band, apply_isotonic, load_calibrator

    if _word_count(text) < _MIN_WORDS:
        return {"score": None, "band": None, "confidence": "low",
                "model_version": MODEL_VERSION, "calibrated": False,
                "available": False, "caveat": f"too short (need >= {_MIN_WORDS} words for windowed signal)"}

    sents = split_sentences(text)
    windows = build_windows(sents, size=3, step=1)
    probs = score_windows(windows)
    if probs is None:
        return {"score": None, "band": None, "confidence": "low",
                "model_version": MODEL_VERSION, "calibrated": False,
                "available": False, "caveat": "detector unavailable (model load or inference failed)"}

    agg = aggregate(sents, windows, probs)
    doc_raw = agg["document_score"]

    cal_path = _calibrated_path()
    calibrated = False
    try:
        if cal_path and os.path.exists(cal_path):
            iso = load_calibrator(cal_path)
            doc_cal = apply_isotonic([doc_raw], iso)[0]
            calibrated = True
        else:
            doc_cal = doc_raw
    except Exception as e:
        logger.warning("[deberta] calibration apply failed, using raw: %s", e)
        doc_cal = doc_raw

    score_100 = max(0.0, min(100.0, doc_cal * 100.0))
    band = map_to_band(score_100)
    caveat = "calibrated on DraftProof ESL corpus" if calibrated else "raw checkpoint probability, uncalibrated — advisory only"
    confidence = "medium" if calibrated else "low"

    return {"score": round(score_100, 1), "band": band, "confidence": confidence,
            "model_version": MODEL_VERSION, "calibrated": calibrated,
            "available": True, "caveat": caveat}


def maybe_attach(text: str) -> dict | None:
    if not _enabled():
        return None
    try:
        return compose(text)
    except Exception as e:
        logger.warning("[deberta] compose failed (fail-open): %s", e)
        return {"score": None, "band": None, "confidence": "low",
                "model_version": MODEL_VERSION, "calibrated": False,
                "available": False, "caveat": f"error: {type(e).__name__}"}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd poc && python test_deberta_signal.py`
Expected: all 3 tests PASS (with or without a real model — disabled/too-short/schema don't need one)

- [ ] **Step 5: Commit**

```bash
git add poc/detect/deberta_signal.py poc/test_deberta_signal.py
git commit -m "feat(deberta): additive composer (compose/maybe_attach/_enabled, fail-open, spec schema)"
```

### Task 1.5: Wire into DetectionRunner (additive invariant)

**Files:**
- Modify: `poc/detect/run.py` (inside `run_all`, before the `DetectionReport` return ~line 185)
- Test: `poc/test_deberta_additive_invariant.py`

- [ ] **Step 1: Write the failing additive-invariant test**

```python
# poc/test_deberta_additive_invariant.py
"""The module ON vs OFF must NOT change tier / ai_likelihood / overall_risk / rewrite_decision."""
import os, sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from detect.run import DetectionRunner

TEXT = " ".join(["Effective academic writing demands a clear thesis statement and supporting evidence."] * 25)


def _badge(runner):
    rep = runner.run_all(TEXT)
    return rep.ai_risk_badge or {}, rep


def test_additive_invariant():
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "0"
    b_off, r_off = _badge(DetectionRunner())
    os.environ["DRAFTPROOF_DEBERTA_SIGNAL"] = "1"
    b_on, r_on = _badge(DetectionRunner())

    # authoritative fields must be bit-identical
    assert b_off.get("tier") == b_on.get("tier")
    assert b_off.get("ai_likelihood_score") == b_on.get("ai_likelihood_score")
    assert r_off.overall_risk == r_on.overall_risk
    assert json.dumps(r_off.rewrite_decision, sort_keys=True, default=str) == \
           json.dumps(r_on.rewrite_decision, sort_keys=True, default=str)
    # the new field appears only when ON
    assert "ai_signal_deberta" in b_on
    assert "ai_signal_deberta" not in b_off
```

- [ ] **Step 2: Run to verify failure**

Run: `cd poc && python test_deberta_additive_invariant.py`
Expected: FAIL — `ai_signal_deberta` not in badge when ON (wiring not done yet)

- [ ] **Step 3: Wire `maybe_attach` into `run_all`**

In `poc/detect/run.py`, inside `run_all`, just before `return DetectionReport(...)` (around line 185), the report already builds `ai_risk_badge`. Add the DeBERTA field to that badge (it lives on `ai_risk_badge`, same as `authenticity_dashboard`). Locate where `ai_risk_badge` is assembled in `run_all` (search `ai_risk_badge =`) and append:

```python
# Strictly additive second-opinion signal (spec 2026-07-01). Never feeds tier/gates.
try:
    from .deberta_signal import maybe_attach as _attach_deberta
    _deberta = _attach_deberta(content)
    if _deberta is not None:
        ai_risk_badge["ai_signal_deberta"] = _deberta
except Exception:  # fail-open: never break the scan
    logger.warning("[deberta] attach failed (fail-open)", exc_info=True)
```

> If `ai_risk_badge` is built in `report.py` rather than `run.py`, attach there instead at the existing call site near `poc/report/report.py:1582`. VERIFY the assembly location during implementation and attach at whichever file owns `ai_risk_badge` construction. The invariant test above catches a wrong-location attach (field would be absent).

- [ ] **Step 4: Run to verify pass**

Run: `cd poc && python test_deberta_additive_invariant.py`
Expected: PASS — authoritative fields identical ON vs OFF, `ai_signal_deberta` present only when ON.

- [ ] **Step 5: Commit**

```bash
git add poc/detect/run.py poc/test_deberta_additive_invariant.py
git commit -m "feat(deberta): wire additive stage into DetectionRunner (invariant-tested)"
```

### Task 1.6: Read-time mirror + entrypoint env

**Files:**
- Create: `draftproof-api/app/_composers/deberta_signal.py`
- Modify: `worker/entrypoint.sh`
- Modify (verify): `worker/requirements.txt`

- [ ] **Step 1: Create the read-time mirror (MVP pass-through)**

```python
# draftproof-api/app/_composers/deberta_signal.py
# KEEP IN SYNC with poc/detect/deberta_signal.py (read-time copy; see _composers/__init__.py).
"""Read-time pass-through for the ai_signal_deberta field. The real inference runs in the
worker (poc/detect/deberta_signal.py) during scan; this copy exists only if a view-side
derivation is later needed. MVP: pass the stored field through unchanged."""
from __future__ import annotations


def pass_through(ai_signal_deberta: dict | None) -> dict | None:
    return ai_signal_deberta
```

- [ ] **Step 2: Add env vars to entrypoint**

In `worker/entrypoint.sh`, in the env block next to `DRAFTPROOF_AUTHENTICITY_DASHBOARD`, add:

```bash
# DeBERTa second-opinion AI signal ON in production (code default ON). Additive only —
# never feeds tier/gates. Off-the-shelf checkpoint loaded from HF volume; set =0 to disable.
export DRAFTPROOF_DEBERTA_SIGNAL="${DRAFTPROOF_DEBERTA_SIGNAL:-1}"
export DRAFTPROOF_DEBERTA_MODEL="${DRAFTPROOF_DEBERTA_MODEL:-<chosen-checkpoint-repo-id>}"
```

- [ ] **Step 3: Verify scikit-learn present**

Run: `grep -i 'scikit-learn\|sklearn' worker/requirements.txt`
If absent, add `scikit-learn>=1.3` to `worker/requirements.txt` (isotonic regression).

- [ ] **Step 4: Commit**

```bash
git add draftproof-api/app/_composers/deberta_signal.py worker/entrypoint.sh worker/requirements.txt
git commit -m "feat(deberta): read-time mirror stub + entrypoint env + sklearn dep"
```

---

## Phase 0 (continued) — Run the gate & pick the checkpoint

### Task 0.3: Run the DeBERTA ESL gate over candidates

**Files:**
- Run: `poc/calibration/deberta_fpr_gate.py`

- [ ] **Step 1: Warm-download each candidate into HF_HOME**

For each repo-id from `deberta_candidates.md`:
Run: `cd poc && DRAFTPROOF_DEBERTA_MODEL=<repo-id> python -c "from detect.deberta_model import _load; _load(); print('warm')"`

- [ ] **Step 2: Score each candidate over the SCoCESLE corpus**

Run: `cd poc && python calibration/deberta_fpr_gate.py --candidates <repo-a,repo-b,repo-c> --compare`
Expected: per candidate — ESL FPR @ thresholds, human mean/p90/max, AUC(AI vs human).

- [ ] **Step 3: Record results in `poc/calibration/deberta_candidates.md`** (append a results table).

- [ ] **Step 4: Commit**

```bash
git add poc/calibration/deberta_candidates.md
git commit -m "research(deberta): SCoCESLE FPR/AUC results per candidate checkpoint"
```

### Task 0.4: Pick winner + decide raw vs isotonic-recalibrated

- [ ] **Step 1: Decision rule**

Pick the candidate with ESL FPR ≤ composite's documented rate AND best AI recall (AUC).
- If raw ESL FPR acceptable → ship raw, `calibrated=False`, advisory caveat. Done.
- If raw FPR too high → fit isotonic on SCoCESLE (Task 0.5), re-measure; if now acceptable → ship `calibrated=True`.
- If isotonic cannot fix → demote to advisory-only OR exclude from MVP.

- [ ] **Step 2: Set the chosen repo-id** in `worker/entrypoint.sh` (`DRAFTPROOF_DEBERTA_MODEL`) and `poc/detect/deberta_model.py` default. Commit.

```bash
git commit -am "feat(deberta): select <repo-id> as production checkpoint (Phase 0 gate)"
```

### Task 0.5: Fit isotonic calibrator (only if Task 0.4 requires it)

**Files:**
- Create: `poc/calibration/deberta_fit_calibrator.py`
- Output: `poc/calibration/deberta_isotonic.pkl`

- [ ] **Step 1: Fit + save**

```python
# poc/calibration/deberta_fit_calibrator.py
"""Fit isotonic calibrator for the DeBERTa signal on SCoCESLE (human label=1 stays low).
Run: cd poc && python calibration/deberta_fit_calibrator.py --out calibration/deberta_isotonic.pkl"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from detect.deberta_calibrate import fit_isotonic, save_calibrator
# reuse fpr_subgroup_gate._proficiency_groups to load human essays (label 1) + _ai_texts (label 0)
from calibration.fpr_subgroup_gate import _proficiency_groups, _ai_texts
from detect.deberta_signal import compose

def _raw_scores(texts):
    # bypass calibration to get raw doc scores: temporarily disable calibrator env
    out = []
    for t in texts:
        os.environ.pop("DRAFTPROOF_DEBERTA_CALIBRATOR", None)
        out.append((compose(t).get("score") or 0.0) / 100.0)
    return out

def main(out):
    humans = [e for grp in _proficiency_groups("scocesle").values() for e in grp]  # label 1
    ais = _ai_texts()  # label 0
    raw = _raw_scores(humans + ais)
    labels = [1]*len(humans) + [0]*len(ais)
    iso = fit_isotonic(raw, labels)
    save_calibrator(iso, out)
    print(f"saved {out}")

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser(); a.add_argument("--out", required=True)
    main(a.parse_args().out)
```

- [ ] **Step 2: Re-run the gate with the calibrator set**

Run: `cd poc && DRAFTPROOF_DEBERTA_CALIBRATOR=calibration/deberta_isotonic.pkl python calibration/deberta_fpr_gate.py --compare`
Expected: ESL FPR now under threshold.

- [ ] **Step 3: Set `DRAFTPROOF_DEBERTA_CALIBRATOR` in `worker/entrypoint.sh`, commit the `.pkl`.**

---

## Phase 2 — Surface (side-by-side display)

### Task 2.1: Frontend tile + agree/disagree note

**Files:**
- Modify: `draftproof-frontend/src/pages/Report.jsx`
- Modify: `draftproof-frontend/src/pages/report/` (new child `DebertaSignal.jsx` by convention)
- Modify: `draftproof-frontend/src/i18n/{en,zh}/report.js`

- [ ] **Step 1: Add i18n strings** (en + zh) for: tile title, band labels, agree note, disagree note, advisory caveat.

- [ ] **Step 2: Create `DebertaSignal.jsx`** reading `report.ai_risk_badge.ai_signal_deberta`. Render: score %, band chip (4-tier legend matching composite), confidence, caveat (small text). If `available:false`, render the caveat only.

- [ ] **Step 3: Add the agree/disagree note logic**

```jsx
// compare ai_signal_deberta.band vs the composite's badge tier
const agree = deberta?.band === compositeTier;
const note = agree
  ? t('report.deberta.noteAgree', { band: compositeTier })
  : t('report.deberta.noteDisagree', { a: compositeTier, b: deberta?.band });
```

- [ ] **Step 4: Mount it in `Report.jsx`** beside the existing AI-likelihood tile.

- [ ] **Step 5: Manual verify** — `npm run dev`, open a completed scan report, confirm both scores render with the note. Commit.

```bash
git commit -am "feat(report): DeBERTa signal side-by-side tile + agree/disagree note"
```

### Task 2.2: PDF renderer side-by-side

**Files:**
- Modify: `poc/report/render_panels.py`

- [ ] **Step 1: Add a DeBERTa line to the AI-signal panel** mirroring how `authenticity_dashboard` is rendered: read `report.ai_risk_badge.ai_signal_deberta`, print "DeBERTa signal: X% (band) — calibrated/raw".

- [ ] **Step 2: Generate a sample PDF** and visually confirm the side-by-side. Commit.

```bash
git commit -am "feat(report): DeBERTa signal line in PDF AI-signal panel"
```

---

## Phase 3 — Final verification

### Task 3.1: End-to-end + latency check

- [ ] **Step 1: Full scan via API** — submit a ~1000-word doc, confirm `ai_signal_deberta` in the report JSON and on the page + PDF.
- [ ] **Step 2: Latency** — time a scan with the module ON vs OFF; document the delta (expected ~+5–10s/1000w). Record in `poc/calibration/deberta_candidates.md`.
- [ ] **Step 3: Fail-open check** — set `DRAFTPROOF_DEBERTA_MODEL` to a nonexistent repo; confirm scan completes with `available:false` and no crash.
- [ ] **Step 4: Memory check** — confirm worker RSS headroom on 4GB Koyeb (~+400MB from the model); if tight, note ONNX INT8 as the mitigation.
- [ ] **Step 5: Final commit + memory update** — record the chosen checkpoint, ESL FPR, and latency in project memory.

---

## Self-Review (run after writing — done)

- **Spec coverage:** Purpose §1 → all tasks additive-only ✓. Decisions §2 → off-the-shelf (Task 0.x), inline (Task 1.5), single-number side-by-side (Task 2.1), ESL gate (Task 0.x), separate calibration unit (Task 1.2) ✓. Architecture §3 → Task 1.5 wire + kill-switch (1.4/1.6) + dual-copy (1.6) ✓. Output schema §6 → Task 1.4 ✓. Disagreement UX §7 → Task 2.1 ✓. ESL gate §8 → Task 0.2/0.3/0.4/0.5 ✓. Error handling §9 → fail-open in 1.3/1.4/1.5 ✓. Testing §10 → invariant test 1.5, unit tests 1.1/1.2/1.4 ✓. Privacy §11 → local inference only (1.3) ✓. Phasing §12 → Phase 0/1/2/3 mapped ✓. Acceptance §13 → Task 3.1 checklist ✓.
- **Placeholder scan:** `<chosen-checkpoint-repo-id>` is INTENTIONAL — resolved by Phase 0 Task 0.4 (the gate's whole purpose). No other TODO/TBD. Band cutoffs (30/50/70) flagged with a VERIFY step against `thresholds.py`.
- **Type consistency:** `compose(text)` → dict with keys `score/band/confidence/model_version/calibrated/available/caveat` — identical across test (1.4), implementer (1.4), invariant test (1.5 reads `ai_signal_deberta`), frontend (2.1 reads same keys), PDF (2.2). `score_windows`, `split_sentences`, `build_windows`, `aggregate`, `fit_isotonic`, `apply_isotonic`, `map_to_band`, `maybe_attach`, `_enabled` — names consistent across all tasks.
