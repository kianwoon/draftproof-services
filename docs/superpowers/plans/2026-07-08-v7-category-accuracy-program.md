# V7 Category-Breakdown Accuracy Program — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the V7 authorship-category breakdown from macro primary accuracy 34% (middle classes at 0%, ESL humans 55% falsely AI-primary on quick-scan) to a measured, promoted improvement — via fused-path measurement, offline weight tuning, and (gated) a new MiniLM paraphrase signal.

**Architecture:** Three sequential phases + one conditional. Phase 1 measures the §12 corpus on the production (deep-scan-fused) path with a cache-backed Modal wrapper (~$0.60 once, free re-runs). Phase 2 captures per-doc signal vectors once, then tunes `category_weights` + `ai_assisted_polished_band` offline by calling the REAL `category_scoring.score_paragraph` under `DRAFTPROOF_V7_WEIGHTS_PATH` candidates (offline == end-to-end by construction). Phase 3 (only if tuned `ai_paraphrased` accuracy < 0.30) builds `paraphrase_pattern_score` in `poc/detect_v7/` (never touching `poc/detect/`, so the ESL FPR pre-push gate is not triggered). Phase 4 (ESL estimator) is a decision point, not a build task.

**Tech Stack:** Python 3, existing `poc/detect_v7` + `poc/calibration` modules, `sentence-transformers` (already a dependency), Modal deep-scan endpoint (creds in `.env` as `DRAFTPROOF_MODAL_ENDPOINT_URL/TOKEN`).

## Global Constraints

- **NO HARDCODED scoring numbers in Python** — every weight/threshold/quantile lives in `weights.json` (or a candidate weights file). `poc/detect_v7/config.py` docstring is the enforcement reference.
- **Never touch `poc/detect/`** in this program (avoids the ESL FPR pre-push gate; Phase 3 lives in `poc/detect_v7/`).
- **Paid Modal calls run ONLY in the main orchestrator session**, never inside a subagent. Approved budget: ~$0.60 for the 198-doc corpus, cache-backed (append-only JSONL at `poc/calibration/retune/cache/deepscan_scores.jsonl`).
- **Committable artifacts are numbers-only** — no corpus text, no SCoCESLE filenames (license). Corpus/capture files stay gitignored.
- **Candidates → staging first** (`poc/calibration/v12_validation/candidates/`), promotion to `poc/detect_v7/weights.json` is its own reviewed task with `_notes` provenance.
- All work on a feature branch; run `cd poc && python -m pytest detect_v7_tests/ -x -q` (adjust to the actual V7 test dir found in repo — `git grep -l "detect_v7" poc/*test*` locates it) before each commit that touches `poc/detect_v7/`.
- Determinism: every stochastic step takes an explicit `--seed` (default 42).
- File cap: no file exceeds 1500 lines.

---

## Phase 1 — Fused-path §12 measurement (paid, one-time)

### Task 1: Cache-backed Modal wrapper + `--fused` mode in measure.py

**Files:**
- Modify: `poc/calibration/v12_validation/measure.py`
- Test: `poc/calibration/v12_validation/test_measure_fused.py` (new)

**Interfaces:**
- Consumes: `calibration.retune.deepscan_cache` (`load_cache`, `append`, `content_key`, `checkpoint_tag`, `DEFAULT_CACHE`), `detect_v7.modal_client.call_deep_scan(sentences) -> {"available": bool, "calibrated": bool, "chunk_scores": list[float]}`.
- Produces: `measure(limit_per_class, fused: bool) -> dict` (adds `fused` kwarg); CLI flag `--fused`; output file `category_agreement_fused_baseline.json` when fused. Also `install_cached_deep_scan(cache_path: Path) -> None` — later phases reuse this exact function.

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/v12_validation/test_measure_fused.py
"""Cache-wrapper behavior: hit serves from cache without calling Modal;
miss calls through and appends ONLY calibrated responses."""
import json
from pathlib import Path

from calibration.v12_validation import measure as m
from calibration.retune import deepscan_cache


def _fake_modal(calls: list, response: dict):
    def fake(sentences):
        calls.append(list(sentences))
        return response
    return fake


def test_cache_hit_skips_modal(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    sentences = ["First sentence.", "Second sentence."]
    key = deepscan_cache.content_key("\n".join(sentences), deepscan_cache.checkpoint_tag())
    deepscan_cache.append(cache_path, key, [0.1, 0.9995])

    calls: list = []
    import detect_v7.modal_client as mc
    monkeypatch.setattr(mc, "call_deep_scan", _fake_modal(calls, {"available": False}))
    m.install_cached_deep_scan(cache_path)
    try:
        resp = mc.call_deep_scan(sentences)
    finally:
        m.uninstall_cached_deep_scan()
    assert resp == {"available": True, "calibrated": True, "chunk_scores": [0.1, 0.9995]}
    assert calls == []  # Modal never hit


def test_cache_miss_calls_through_and_appends_calibrated(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    sentences = ["Only sentence here."]
    calls: list = []
    real = {"available": True, "calibrated": True, "chunk_scores": [0.5]}
    import detect_v7.modal_client as mc
    monkeypatch.setattr(mc, "call_deep_scan", _fake_modal(calls, real))
    m.install_cached_deep_scan(cache_path)
    try:
        resp = mc.call_deep_scan(sentences)
    finally:
        m.uninstall_cached_deep_scan()
    assert resp == real and len(calls) == 1
    key = deepscan_cache.content_key("\n".join(sentences), deepscan_cache.checkpoint_tag())
    assert deepscan_cache.load_cache(cache_path)[key] == [0.5]


def test_uncalibrated_response_not_cached(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    calls: list = []
    real = {"available": True, "calibrated": False, "chunk_scores": [0.5]}
    import detect_v7.modal_client as mc
    monkeypatch.setattr(mc, "call_deep_scan", _fake_modal(calls, real))
    m.install_cached_deep_scan(cache_path)
    try:
        resp = mc.call_deep_scan(["A sentence."])
    finally:
        m.uninstall_cached_deep_scan()
    assert resp == real
    assert deepscan_cache.load_cache(cache_path) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest calibration/v12_validation/test_measure_fused.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'install_cached_deep_scan'`

- [ ] **Step 3: Implement the wrapper + `--fused` in measure.py**

Add to `measure.py` (module level, after imports — note the deferred `detect_v7` import mirrors the file's existing heavy-import-after-env pattern):

```python
_wrapper_state: dict = {}


def install_cached_deep_scan(cache_path) -> None:
    """Wrap detect_v7.modal_client.call_deep_scan with a JSONL cache.

    Cache key = content_key(joined sentences, checkpoint). Only calibrated
    responses are cached (an uncalibrated response means checkpoint drift —
    caching it would silently pin stale scores)."""
    import detect_v7.modal_client as mc
    from calibration.retune import deepscan_cache as dc

    cache = dc.load_cache(cache_path)
    real = mc.call_deep_scan

    def cached(sentences):
        key = dc.content_key("\n".join(sentences), dc.checkpoint_tag())
        if key in cache:
            return {"available": True, "calibrated": True, "chunk_scores": cache[key]}
        resp = real(sentences)
        if (isinstance(resp, dict) and resp.get("available") is True
                and resp.get("calibrated") is True
                and isinstance(resp.get("chunk_scores"), list)):
            dc.append(cache_path, key, resp["chunk_scores"])
            cache[key] = resp["chunk_scores"]
        return resp

    _wrapper_state["real"] = real
    mc.call_deep_scan = cached


def uninstall_cached_deep_scan() -> None:
    import detect_v7.modal_client as mc
    if "real" in _wrapper_state:
        mc.call_deep_scan = _wrapper_state.pop("real")
```

Then modify `measure(limit_per_class)` → `measure(limit_per_class, fused: bool = False)`:
- When `fused`: `os.environ["DRAFTPROOF_V7_DEEP_SCAN"] = "1"` (instead of the current `pop`), call `install_cached_deep_scan(DEFAULT_CACHE)` from `calibration.retune.deepscan_cache`, and set `result["spec"]` / `caveats[1]` to say `deep-scan-fused path (cached Modal)`.
- When not fused: current behavior byte-identical.
- CLI: `ap.add_argument("--fused", action="store_true")`; when set, default `--out` becomes `HERE / "category_agreement_fused_baseline.json"`.
- `main()` must `uninstall_cached_deep_scan()` in a `finally:` block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd poc && python -m pytest calibration/v12_validation/test_measure_fused.py -v`
Expected: 3 PASS

- [ ] **Step 5: Verify quick-scan path unchanged (regression)**

Run: `cd poc && python -m calibration.v12_validation.measure --limit 2 --out /tmp/quickscan_smoke.json && python -c "import json; d=json.load(open('/tmp/quickscan_smoke.json')); print(d['spec'])"`
Expected: runs clean, spec line still says `quick-scan path`.

- [ ] **Step 6: Commit**

```bash
git add poc/calibration/v12_validation/measure.py poc/calibration/v12_validation/test_measure_fused.py
git commit -m "feat(v12): --fused measurement mode with cache-backed Modal deep-scan wrapper"
```

### Task 2: Run the paid fused measurement (MAIN SESSION ONLY)

**Files:**
- Create: `poc/calibration/v12_validation/category_agreement_fused_baseline.json` (committed, numbers-only)

- [ ] **Step 1: Smoke run (≤8 paid docs, ~$0.03)**

```bash
cd poc && set -a && source ../.env && set +a
python -m calibration.v12_validation.measure --fused --limit 2 --out /tmp/fused_smoke.json
```
Expected: completes; `/tmp/fused_smoke.json` has non-null per_class numbers. If Modal errors, STOP and fix before spending more.

- [ ] **Step 2: Full run (198 docs, ~$0.60, cache-backed)**

```bash
cd poc && python -m calibration.v12_validation.measure --fused
```
Expected: writes `category_agreement_fused_baseline.json`. Record `macro_primary_accuracy` and `student_owned.false_ai_primary_rate` — these are Phase 2's constraint constants and Phase 4's decision input.

- [ ] **Step 3: Re-run to prove cache works ($0)**

Re-run the same command; expected: completes in a fraction of the time, no new Modal spend (`modal billing report --for today` if in doubt), identical output metrics.

- [ ] **Step 4: Commit the baseline**

```bash
git add poc/calibration/v12_validation/category_agreement_fused_baseline.json
git commit -m "data(v12): fused-path category-agreement baseline (deep-scan cached)"
```

**DECISION CHECKPOINT (owner-visible):** report quick-scan vs fused per-class table. If fused `student_owned.false_ai_primary_rate` ≤ 0.20, Phase 4 (ESL estimator) is SKIPPED. Middle classes are expected to remain ~0% (weights problem) → proceed to Phase 2 regardless.

---

## Phase 2 — Offline weight tuning (free after Phase 1)

### Task 3: Signal-vector capture script

**Files:**
- Create: `poc/calibration/v12_validation/capture_signals.py`
- Create: `poc/calibration/v12_validation/test_capture_signals.py`
- Output (gitignored): `poc/calibration/v12_validation/captures/signals_fused.jsonl` — add `captures/` to the existing gitignore entry for this dir.

**Interfaces:**
- Consumes: `measure.py`'s `_resolve_text`, `install_cached_deep_scan`; `calibration.measure_end_to_end.scan_text`; `detect_v7.pipeline_bridge` helpers `_build_raw_signals`, `_extract_calibrated_score`, `get_deep_scan_proportion`; `detect_v7.signal_adapter.adapt_paragraph_signals`.
- Produces: one JSONL row per doc: `{"label": str, "doc_key": sha256-of-text, "v7_signals": {...}, "calibrated_detector_score": float, "has_comparison_text": false}`. Downstream (Task 4) replays these through `category_scoring.score_paragraph`.

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/v12_validation/test_capture_signals.py
"""capture_one must produce exactly what score_paragraph consumes, and its
offline prediction must MATCH the end-to-end pipeline's primary_category
for the same text (offline == e2e by construction is the whole point)."""
import os

from calibration.v12_validation.capture_signals import capture_one
from detect_v7 import category_scoring

_TEXT = (
    "The industrial revolution transformed European society in profound ways. "
    "Factories replaced workshops, and cities grew rapidly as workers migrated. "
    "My grandmother's village in Guangdong still had a hand loom in 1980, which "
    "is why I find the timeline of mechanization so uneven across regions."
) * 3


def test_capture_matches_end_to_end(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN", "1")
    monkeypatch.delenv("DRAFTPROOF_V7_DEEP_SCAN", raising=False)  # quick-scan: no spend
    from calibration.measure_end_to_end import scan_text
    from detect.run import DetectionRunner

    runner = DetectionRunner()
    row, rep = capture_one(runner, _TEXT, label="student_owned")

    assert set(row) >= {"label", "doc_key", "v7_signals", "calibrated_detector_score"}
    offline = category_scoring.score_paragraph(
        row["v7_signals"], row["calibrated_detector_score"],
        has_comparison_text=False, esl_score=None,
    )
    e2e = (rep.get("ai_risk_badge") or {}).get("authorship_breakdown") or {}
    assert offline["primary_category"] == e2e.get("primary_category")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest calibration/v12_validation/test_capture_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: ... capture_signals`

- [ ] **Step 3: Implement capture_signals.py**

```python
# poc/calibration/v12_validation/capture_signals.py
"""Capture per-doc V7 signal vectors (fused path, cached Modal) so weight
tuning replays category_scoring.score_paragraph offline — no re-scans.

Numbers-only output (signals + label + text hash); corpus text never leaves
the gitignored captures/ dir. Usage:
    python -m calibration.v12_validation.capture_signals            # fused
    python -m calibration.v12_validation.capture_signals --quick    # no Modal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "captures" / "signals_fused.jsonl"


def capture_one(runner, text: str, label: str) -> tuple[dict, dict]:
    """Run the real scan once; return (capture_row, full_report)."""
    from calibration.measure_end_to_end import scan_text
    from detect_v7 import detector_fusion, pipeline_bridge, signal_adapter

    rep = scan_text(runner, text)
    badge = dict(rep.get("ai_risk_badge") or {})
    det = {**badge, "document_text": text}
    deep = pipeline_bridge.get_deep_scan_proportion(det)  # None on quick-scan
    if deep is not None:
        det["_precomputed_deep_scan"] = deep

    raw = pipeline_bridge._build_raw_signals(det)
    v7_signals = signal_adapter.adapt_paragraph_signals(raw)
    composite = pipeline_bridge._extract_calibrated_score(det)
    detector_scores = {"composite": composite}
    if deep is not None and not deep["below_floor"] and not deep["uncalibrated"]:
        detector_scores["deberta_large"] = deep["proportion"]
    calibrated = detector_fusion.compute_calibrated_detector_score(detector_scores)
    row = {
        "label": label,
        "doc_key": hashlib.sha256(text.encode()).hexdigest()[:16],
        "v7_signals": v7_signals,
        "calibrated_detector_score": calibrated,
        "has_comparison_text": False,
    }
    return row, rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="quick-scan path, no Modal")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    os.environ["DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"] = "1"
    from calibration.v12_validation import measure as m
    if args.quick:
        os.environ.pop("DRAFTPROOF_V7_DEEP_SCAN", None)
    else:
        os.environ["DRAFTPROOF_V7_DEEP_SCAN"] = "1"
        from calibration.retune.deepscan_cache import DEFAULT_CACHE
        m.install_cached_deep_scan(DEFAULT_CACHE)
    try:
        from calibration.retune.intake import DEFAULT_SCOCESLE
        from detect.run import DetectionRunner
        manifest = json.loads((m.CORPUS_DIR / "manifest.json").read_text())
        by_class: dict[str, list] = defaultdict(list)
        for r in manifest["rows"]:
            by_class[r["label"]].append(r)
        runner = DetectionRunner()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with args.out.open("w", encoding="utf-8") as f:
            for label in m._CLASSES:
                rows = by_class.get(label, [])
                if args.limit:
                    rows = rows[: args.limit]
                for r in rows:
                    text = m._resolve_text(r, DEFAULT_SCOCESLE)
                    if not text:
                        continue
                    row, _ = capture_one(runner, text, label)
                    f.write(json.dumps(row) + "\n")
                    n += 1
                    print(".", end="", flush=True)
        print(f"\ncaptured {n} docs -> {args.out}")
    finally:
        if not args.quick:
            m.uninstall_cached_deep_scan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

NOTE for implementer: `capture_one` calls two `pipeline_bridge` underscore helpers. Verify their exact names/signatures against `poc/detect_v7/pipeline_bridge.py` (`_build_raw_signals` at ~L510, `_extract_calibrated_score` at ~L458, `get_deep_scan_proportion` at ~L200) and `detector_fusion.compute_calibrated_detector_score` before running; if `run_v7_breakdown` composes them differently (e.g. normalization step between), mirror ITS exact call order — the test in Step 1 is the contract that catches any divergence.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd poc && python -m pytest calibration/v12_validation/test_capture_signals.py -v`
Expected: PASS. If `primary_category` mismatches, read `run_v7_breakdown` and fix `capture_one`'s call order — do NOT loosen the test.

- [ ] **Step 5: Full capture run (cache-warm from Task 2, $0)**

```bash
cd poc && set -a && source ../.env && set +a
python -m calibration.v12_validation.capture_signals
```
Expected: `captured 198 docs` (minus any skipped). Confirm `captures/` is gitignored: `git status --short` shows no captures file.

- [ ] **Step 6: Commit**

```bash
git add poc/calibration/v12_validation/capture_signals.py poc/calibration/v12_validation/test_capture_signals.py poc/calibration/v12_validation/.gitignore
git commit -m "feat(v12): per-doc signal capture for offline category-weight tuning"
```

### Task 4: The tuner

**Files:**
- Create: `poc/calibration/v12_validation/tune_weights.py`
- Create: `poc/calibration/v12_validation/test_tune_weights.py`
- Output: `poc/calibration/v12_validation/candidates/weights_candidate_<seed>.json` (committed) + `candidates/trials_<seed>.jsonl` (gitignored)

**Interfaces:**
- Consumes: `captures/signals_fused.jsonl` rows; `detect_v7.config.reload_weights`; `detect_v7.category_scoring.score_paragraph`; the bundled `poc/detect_v7/weights.json` as the search's starting point.
- Produces: candidate weights file — a FULL weights.json copy with only `category_weights` and `ai_assisted_polished_band` mutated, plus a `_tuning_provenance` key (seed, trial count, tune/holdout metrics, baseline constants). Task 5 consumes this file.

**Search design (all constants CLI-flagged, defaults shown):**
- Stratified 70/30 tune/holdout split by label, `--seed 42`.
- `--trials 2000` random candidates: for each category, sample weight vector from Dirichlet(α = 8 × current_weights) — concentrated near current, occasional exploration; renormalize to sum 1.0; floor each weight at 0.02. `ai_assisted_polished_band` lo/hi sampled uniform within ±0.15 of current, keeping lo < hi.
- Candidate evaluation: point `DRAFTPROOF_V7_WEIGHTS_PATH` at a temp candidate file, `config.reload_weights(force=True)`, replay every tune-split row through `score_paragraph`, compute macro primary accuracy + `student_owned` false-AI rate. (Real scorer, real config path — zero reimplementation of scoring math.)
- Objective: maximize tune-split macro accuracy **subject to** tune-split `student_owned` false-AI rate ≤ the fused baseline value from Task 2. Tie-break: higher `student_owned` accuracy.
- Top-5 candidates → coordinate refinement (±0.05 per weight, renormalize, keep improvements).
- Winner must ALSO satisfy on holdout: macro ≥ (tune macro − 10 pts) AND false-AI constraint. Otherwise report "no candidate generalizes" honestly and stop — do not ship an overfit vector.
- Restore original `DRAFTPROOF_V7_WEIGHTS_PATH` state in `finally:`.

- [ ] **Step 1: Write the failing tests**

```python
# poc/calibration/v12_validation/test_tune_weights.py
import json

from calibration.v12_validation import tune_weights as tw


def _rows():
    # Two synthetic classes whose signals are trivially separable on
    # generic_density so ANY sane search finds a discriminating vector.
    base = {k: 0.5 for k in (
        "specificity_score", "grounding_gap", "sentence_variance",
        "generic_density", "author_voice_absence", "sentence_smoothness",
        "local_style_shift", "semantic_drift", "predictable_structure",
        "paraphrase_pattern_score", "meaning_preservation_score",
    )}
    rows = []
    for i in range(10):
        s = dict(base, generic_density=0.05, specificity_score=0.9)
        rows.append({"label": "student_owned", "doc_key": f"h{i}",
                     "v7_signals": {**s, "signal_status": {}},
                     "calibrated_detector_score": 0.1, "has_comparison_text": False})
        a = dict(base, generic_density=0.95, specificity_score=0.1)
        rows.append({"label": "ai_generated_like", "doc_key": f"a{i}",
                     "v7_signals": {**a, "signal_status": {}},
                     "calibrated_detector_score": 0.9, "has_comparison_text": False})
    return rows


def test_split_is_stratified_and_deterministic():
    rows = _rows()
    t1, h1 = tw.split_rows(rows, seed=42)
    t2, h2 = tw.split_rows(rows, seed=42)
    assert [r["doc_key"] for r in t1] == [r["doc_key"] for r in t2]
    labels_t = {r["label"] for r in t1}
    assert labels_t == {"student_owned", "ai_generated_like"}


def test_evaluate_uses_real_scorer(tmp_path):
    rows = _rows()
    cand = tw.load_base_weights()
    path = tmp_path / "cand.json"
    path.write_text(json.dumps(cand))
    metrics = tw.evaluate_candidate(path, rows)
    assert set(metrics) >= {"macro_primary_accuracy", "student_owned_false_ai_rate", "per_class"}
    assert 0.0 <= metrics["macro_primary_accuracy"] <= 1.0


def test_candidate_weights_normalized():
    import random
    rng = random.Random(42)
    cand = tw.sample_candidate(tw.load_base_weights(), rng)
    for cat, entries in cand["category_weights"].items():
        total = sum(e["weight"] for e in entries)
        assert abs(total - 1.0) < 1e-6, cat
        assert all(e["weight"] >= 0.02 for e in entries), cat
    band = cand["ai_assisted_polished_band"]
    lo, hi = tw.band_bounds(band)
    assert lo < hi
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd poc && python -m pytest calibration/v12_validation/test_tune_weights.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tune_weights.py**

Public functions the tests pin down (implementer fills bodies to the search design above; ~250 lines):

```python
# poc/calibration/v12_validation/tune_weights.py — skeleton of the pinned API
def load_base_weights() -> dict: ...        # deep-copy of bundled weights.json
def split_rows(rows, seed) -> tuple[list, list]: ...  # stratified 70/30
def band_bounds(band: dict) -> tuple[float, float]: ...  # read the band's lo/hi keys AS NAMED in weights.json (inspect first)
def sample_candidate(base: dict, rng) -> dict: ...  # Dirichlet-perturbed, normalized, floored
def evaluate_candidate(candidate_path, rows) -> dict:
    # os.environ["DRAFTPROOF_V7_WEIGHTS_PATH"] = str(candidate_path)
    # config.reload_weights(force=True)
    # for each row: category_scoring.score_paragraph(row["v7_signals"],
    #     row["calibrated_detector_score"], has_comparison_text=False, esl_score=None)
    # → confusion → macro/per-class/false-AI metrics; restore env in finally
def main() -> int: ...  # CLI: --trials --seed --captures --baseline-json --out-dir
```

Constraint constants (baseline false-AI rate) are READ from `category_agreement_fused_baseline.json` via `--baseline-json`, never hardcoded.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd poc && python -m pytest calibration/v12_validation/test_tune_weights.py -v`
Expected: 3 PASS

- [ ] **Step 5: Full tuning run ($0, pure CPU)**

```bash
cd poc && python -m calibration.v12_validation.tune_weights \
  --captures calibration/v12_validation/captures/signals_fused.jsonl \
  --baseline-json calibration/v12_validation/category_agreement_fused_baseline.json \
  --trials 2000 --seed 42
```
Expected: prints tune + holdout tables for the winner; writes `candidates/weights_candidate_42.json`. If "no candidate generalizes" — STOP, report to owner, Phase 3 gate auto-fires.

- [ ] **Step 6: Commit**

```bash
git add poc/calibration/v12_validation/tune_weights.py poc/calibration/v12_validation/test_tune_weights.py poc/calibration/v12_validation/candidates/weights_candidate_42.json
git commit -m "feat(v12): offline category-weight tuner + first tuned candidate"
```

### Task 5: End-to-end validation + promotion of the winning candidate

**Files:**
- Modify: `poc/detect_v7/weights.json` (category_weights, ai_assisted_polished_band, `_notes` provenance)

- [ ] **Step 1: End-to-end fused re-measure under the candidate ($0, cache-warm)**

```bash
cd poc && set -a && source ../.env && set +a
DRAFTPROOF_V7_WEIGHTS_PATH=calibration/v12_validation/candidates/weights_candidate_42.json \
  python -m calibration.v12_validation.measure --fused --out /tmp/fused_candidate.json
```
Expected: per-class metrics within ±2 pts of the tuner's full-set numbers (offline == e2e check at scale). A larger gap = a capture bug; go back to Task 3, do not promote.

- [ ] **Step 2: Tier-invariance proof**

Category weights are display-layer only — the badge tier comes from the fused-score tier authority, NOT from category shares. Prove it: run one scan with and without `DRAFTPROOF_V7_WEIGHTS_PATH` and diff everything except `authorship_breakdown`:

```bash
cd poc && python - <<'EOF'
import json, os
os.environ["DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN"] = "1"
from calibration.measure_end_to_end import scan_text
from detect.run import DetectionRunner
TEXT = open("calibration/authorship_cases/" + sorted(os.listdir("calibration/authorship_cases"))[0]).read()
TEXT = json.loads(TEXT)["text"]
runner = DetectionRunner()
def strip(rep):
    b = dict(rep.get("ai_risk_badge") or {}); b.pop("authorship_breakdown", None); return b
base = strip(scan_text(runner, TEXT))
os.environ["DRAFTPROOF_V7_WEIGHTS_PATH"] = "calibration/v12_validation/candidates/weights_candidate_42.json"
import detect_v7.config as c; c.reload_weights(force=True)
cand = strip(scan_text(runner, TEXT))
assert json.dumps(base, sort_keys=True) == json.dumps(cand, sort_keys=True), "TIER DRIFT — do not promote"
print("tier-invariant: OK")
EOF
```
Expected: `tier-invariant: OK`. Any drift means candidate touched keys beyond category display — reject.

- [ ] **Step 3: Promote into weights.json**

Copy the candidate's `category_weights` + `ai_assisted_polished_band` into `poc/detect_v7/weights.json`. Add to `_notes`: seed, trial count, tune/holdout/e2e metrics, baseline it beat, date, and the candidate file path (the `_notes` block is the provenance ledger — same convention as the existing `deep_scan_calibration` note).

- [ ] **Step 4: Full V7 test suite**

Run: `cd poc && python -m pytest $(git grep -l "detect_v7" -- "poc/**/test_*.py" | sed 's|^poc/||' | tr '\n' ' ') -q`
Expected: all pass. Any test pinning old weight VALUES (not structure) gets its expectation updated with a comment pointing at the candidate provenance.

- [ ] **Step 5: Commit**

```bash
git add poc/detect_v7/weights.json
git commit -m "feat(v7): promote tuned category weights (seed 42) — macro <X>% -> <Y>%, false-AI <A>% -> <B>%"
```
(Fill X/Y/A/B from the measured numbers — the commit message carries the evidence.)

**PHASE 3 GATE (owner-visible):** if promoted `ai_paraphrased` primary accuracy ≥ 0.30 on the fused e2e re-measure → Phase 3 is SKIPPED, program ends here with a summary. Below 0.30 → proceed.

---

## Phase 3 — `paraphrase_pattern_score` (GATED on Phase 2 result)

### Task 6: Feature study — measure before building

**Files:**
- Create: `poc/calibration/v12_validation/paraphrase_feature_study.py`
- Output: `poc/calibration/v12_validation/paraphrase_feature_study.json` (committed, numbers-only)

**Interfaces:**
- Consumes: §12 corpus via `measure._resolve_text`; `sentence_transformers.SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")` (~90 MB, downloads on first run — spec §5 names this exact checkpoint for Phase 1C).
- Produces: per-feature AUC table (ai_paraphrased vs each other class) + p10/p90 normalization quantiles for the winning feature — Task 7 copies these numbers into `weights.json`.

- [ ] **Step 1: Implement the study script** (measurement code, no production surface — test is the committed output itself)

Candidate features, each computed per document from MiniLM sentence embeddings:
1. `adjacent_cosine_mean` — mean cosine similarity of consecutive sentence embeddings (paraphrased-AI text keeps the source's unnaturally even semantic gait).
2. `pairwise_cosine_std` — std of all-pairs cosine (low = semantic monotone).
3. `embedding_norm_cv` — coefficient of variation of embedding L2 norms.
4. `frame_reuse` — the existing detect-side `structural_reuse` criterion value, read from each capture row's raw signals if present (spec §5 marks it "partial" coverage — measure whether MiniLM features beat it, complement it, or duplicate it).

Script loads the corpus (same `_resolve_text` path as measure.py), computes all four per doc, reports one-vs-rest AUC per feature per class pair, Pearson correlation matrix between features, and p10/p90 per winning feature. Output JSON committable (no text).

- [ ] **Step 2: Run it**

```bash
cd poc && python -m calibration.v12_validation.paraphrase_feature_study
```

**HARD GATE:** the best feature must reach AUC ≥ 0.70 for `ai_paraphrased` vs `ai_generated_like` (the confusion that matters — 34/39 collapse) AND AUC ≤ 0.60 for `student_owned` vs rest **in the AI-flagging direction** (i.e. it must NOT be a covert ESL detector — check the higher/lower proficiency subgroups separately using the manifest's SCoCESLE rows). If no feature passes, STOP and report honestly: "signal not viable from these features" — per precision-first, do not ship a blurry signal.

- [ ] **Step 3: Commit the study**

```bash
git add poc/calibration/v12_validation/paraphrase_feature_study.py poc/calibration/v12_validation/paraphrase_feature_study.json
git commit -m "data(v12): paraphrase-signal feature study — AUC + normalization quantiles"
```

### Task 7: Build + wire the signal (only if Task 6 gate passed)

**Files:**
- Create: `poc/detect_v7/paraphrase_signal.py`
- Modify: `poc/detect_v7/weights.json` (new `paraphrase_signal` block: model name, winning feature name, p10/p90 quantiles — all data, no Python literals)
- Modify: `poc/detect_v7/signal_adapter.py` rows 12 (~L289-294): replace `None`/`_STATUS_NOT_IMPLEMENTED` with the computed value / `_STATUS_OK`, falling back to `unavailable` when the embedder or text is absent (fail-open, matching every other signal's contract)
- Modify: `poc/detect_v7/pipeline_bridge.py` `_build_raw_signals` (~L510): attach `paraphrase_pattern_score` computed from `document_text` (document-level, mirroring how deep-scan is document-level)
- Test: `poc/detect_v7/` test dir — new `test_paraphrase_signal.py`

**Interfaces:**
- Produces: `paraphrase_signal.compute(document_text: str) -> float | None` — 0-1 (quantile-normalized winning feature, clamped), `None` on any failure (no model, empty text, <3 sentences). Deterministic for fixed text.

- [ ] **Step 1: Write failing tests** — (a) returns None on empty/short text; (b) returns value in [0,1] on a 5-sentence text; (c) `adapt_paragraph_signals` reports status `ok` when value present, `unavailable` when None — and NOT `not_implemented` anymore; (d) `score_paragraph` degraded accounting treats it as a built signal. Concrete test code written against the exact status constants in `signal_adapter.py` (`_STATUS_OK` / `_STATUS_UNAVAILABLE`).
- [ ] **Step 2: Verify tests fail**
- [ ] **Step 3: Implement `paraphrase_signal.py`** — lazy-load the MiniLM model (module-level cache, same pattern as `poc/detect/semantic_shape.py:249`), compute the winning feature, normalize via weights.json quantiles: `score = clamp01((raw - p10) / (p90 - p10))`.
- [ ] **Step 4: Wire signal_adapter + pipeline_bridge; verify tests pass**
- [ ] **Step 5: Re-tune with the signal live** — rerun Task 3 capture (cache-warm, $0) → Task 4 tuner (`--seed 43`) → Task 5 validation/promotion. The paraphrase categories' 0.20 starter weight on this signal now has a real input; the tuner decides its true weight.
- [ ] **Step 6: ESL safety re-check** — the fused re-measure's `student_owned.false_ai_primary_rate` must not exceed the Phase-2-promoted value. If it does, the signal is ESL-biased in composition even if its solo AUC looked clean — revert the weight to 0 in weights.json (keep the signal computed + surfaced as data) and report.
- [ ] **Step 7: Commit + full V7 suite**

```bash
git add poc/detect_v7/paraphrase_signal.py poc/detect_v7/signal_adapter.py poc/detect_v7/pipeline_bridge.py poc/detect_v7/weights.json <test files>
git commit -m "feat(v7): paraphrase_pattern_score — MiniLM structural-reuse signal (Phase 1C)"
```

---

## Phase 4 — ESL estimator: DECISION ONLY, no build task in this plan

After Phase 1 (and again after Phase 2 promotion): if fused `student_owned.false_ai_primary_rate` ≤ 0.20, record the decision "ESL estimator not needed — fused path + tuned weights suffice" in the plan's outcome summary and leave the `esl_guard_unavailable` flag honest as-is. If > 0.20, scope a separate plan (it is a new estimator with its own corpus validation — out of scope here by design).

---

## Agent Orchestration Map

Executed via **superpowers:subagent-driven-development** — fresh subagent per task, orchestrator reviews between tasks.

| Task | Runner | Why | Review |
|---|---|---|---|
| 1 (wrapper + --fused) | Sonnet subagent | standard implementation, pinned tests | code-reviewer agent |
| 2 (paid run) | **MAIN SESSION** | Modal spend — never delegated | orchestrator verifies billing + output |
| 3 (capture) | Sonnet subagent | standard, but the e2e-match test is the safety net | code-reviewer agent |
| 4 (tuner) | Opus subagent | search design + overfitting guards = the hardest code in the program | code-reviewer + orchestrator reads trials log |
| 5 (validation/promotion) | MAIN SESSION | promotion is a review gate by definition | orchestrator |
| 6 (feature study) | Sonnet subagent (no Modal, local MiniLM) | measurement script | orchestrator reads the AUC table |
| 7 (signal build) | Opus subagent | cross-cutting: 3 detect_v7 files + weights schema | code-reviewer + orchestrator |

Every subagent prompt MUST include the Tool Routing preamble from `~/.claude/rules/agents.md` and the constraint block from this plan's Global Constraints. Subagent completion claims are verified independently (ls + test run) before marking a task done — this codebase has a documented history of subagents reporting completion before files existed.
