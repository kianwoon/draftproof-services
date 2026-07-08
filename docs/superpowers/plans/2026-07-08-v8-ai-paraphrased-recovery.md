# V8 ai_paraphrased Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift `ai_paraphrased` from 0% to ≥40% primary accuracy with single-document evidence, holding all of today's floors — or record a second honest rejection and ship the display fallback.

**Architecture:** Phase A studies interaction features (content-humanness × surface-AIness) offline over fresh 16-signal captures, behind the same hard gate that killed MiniLM (effective AUC ≥0.70 vs generated + ESL-direction check). Only a gate PASS unlocks production wiring (signal_adapter derived signals → tuner with a new `--para-floor` constraint → promotion gates). Phase B (lexical/structural raggedness) runs only if A fails; Phase C stops for an owner decision. Spec: `docs/superpowers/specs/2026-07-08-v8-ai-paraphrased-recovery-design.md`.

**Tech Stack:** Python 3 from `poc/`, existing `poc/calibration/v12_validation/` harness (captures, tuner, measure, leakage guard), `poc/detect_v7/` signal stack. All runs expected $0 (deep-scan cache warm).

## Global Constraints

- Acceptance (e2e fused §12, promoted weights): `ai_paraphrased ≥ 0.40`, `student_owned false-AI ≤ 0.10`, `ai_generated_like ≥ 0.70`, `ai_assisted_polished ≥ 0.65`, macro ≥ ~0.65 (no regression beyond noise vs 65.5%).
- Study gate per phase: best feature **effective AUC ≥ 0.70** (direction-aware, `max(auc, 1−auc)` — mirror `paraphrase_feature_study.py`'s fixed methodology) for para vs `ai_generated_like`; para vs `ai_assisted_polished` AUC reported; **ESL-direction subgroup check mandatory** (higher vs lower proficiency student_owned, in the winner's flagging direction, fail if ≥0.60). Gate thresholds are CLI flags with these defaults. No threshold torturing — a FAIL is a valid outcome.
- **MiniLM/embedding features BANNED** (gate-rejected 2026-07-08). Phase B is lexical/structural only.
- NO HARDCODED scoring numbers in production Python — weights/thresholds/quantiles in weights.json or CLI flags.
- Never touch `poc/detect/` (Phase C would; it STOPS for owner decision first). `poc/report/`/frontend untouched (no UI work in V8).
- Committed JSONs numbers-only — `test_no_text_leakage.py` must stay green (long prose only under `_notes`/`_tuning_provenance`/`provenance` keys).
- Paid/state-changing runs (captures, measures, tuning, promotion) execute in the MAIN SESSION, not subagents.
- Determinism: every stochastic step takes `--seed`; same seed + inputs → identical output.
- Import-order gotcha: any CLI importing `detect.*` must `import calibration.measure_end_to_end` first (sys.path shim).
- No file exceeds 1500 lines.

---

### Task 1 (MAIN SESSION): Fresh 16-signal capture

**Files:** none created in repo (gitignored captures output)

- [ ] **Step 1: Run the capture (cache-warm, ~$0)**

```bash
cd poc && set -a && source ../.env && set +a
python -m calibration.v12_validation.capture_signals --out calibration/v12_validation/captures/signals_fused_v3.jsonl
```
Expected: `captured 198 docs`. Verify 16 signals present:

```bash
python3 -c "
import json
r = json.loads(open('calibration/v12_validation/captures/signals_fused_v3.jsonl').readline())
s = r['v7_signals']
assert 'specificity_student_evidence' in s and 'specificity_ai_evidence' in s, 'split signals missing'
print('signals:', len([k for k in s if k != 'signal_status']))"
```
Expected: `signals: 16`.

### Task 2: Phase A interaction study script

**Files:**
- Create: `poc/calibration/v12_validation/phase_a_interaction_study.py`
- Test: `poc/calibration/v12_validation/test_phase_a_interaction_study.py`
- Output (Task 3 commits it): `poc/calibration/v12_validation/phase_a_interaction_study.json`

**Interfaces:**
- Consumes: capture rows (`{"label", "doc_key", "v7_signals", "calibrated_detector_score", ...}`); `paraphrase_feature_study.py`'s existing helpers — IMPORT `rank_auc` (Mann-Whitney with tie-averaging) and the direction-aware gate evaluation if it is a reusable function; if the gate logic is inline there, extract it into a shared helper IN `paraphrase_feature_study.py` (e.g. `evaluate_gate(per_feature_values, labels, auc_gate, esl_gate) -> dict`) and import — never duplicate the AUC/direction math.
- Produces: `run_study(rows, prof_by_key, auc_gate=0.70, esl_gate=0.60) -> dict` (the study result), CLI `python -m calibration.v12_validation.phase_a_interaction_study --captures <path> [--auc-gate 0.70] [--esl-gate 0.60] [--out <path>]`. Result dict carries `gate_verdict` ("PASS"/"FAIL"), `winner` {name, direction, effective_auc_vs_generated, auc_vs_polished, esl_directional_auc, p10, p90}, full per-feature AUC table, feature correlation matrix, skip counts.

**Candidate features** (computed per capture row; row skipped-and-counted when any input is None):

```python
CONTENT_PROXIES = {
    "specificity": lambda s, det: s.get("specificity_score"),
    "spec_student_ev": lambda s, det: s.get("specificity_student_evidence"),
    "voice_presence": lambda s, det: None if s.get("author_voice_absence") is None else 1.0 - s["author_voice_absence"],
    "grounded": lambda s, det: None if s.get("grounding_gap") is None else 1.0 - s["grounding_gap"],
}
SURFACE_PROXIES = {
    "smooth": lambda s, det: s.get("sentence_smoothness"),
    "det": lambda s, det: det,
}
# 4 x 2 = 8 interaction features, named f"{content}_x_{surface}", value = content * surface (clamped [0,1])
```

**ESL-direction check:** map `doc_key` → proficiency group by resolving corpus student files and hashing (`sha256(text)[:16]`, exactly like the false-AI diagnosis did): reuse `measure._resolve_text` + `DEFAULT_SCOCESLE` glob dirs (`*higher proficiency*` / `*lower proficiency*`). The winner's directional lower-vs-higher AUC ≥ `esl_gate` in the flagging direction = FAIL.

- [ ] **Step 1: Write the failing tests**

```python
# poc/calibration/v12_validation/test_phase_a_interaction_study.py
"""Synthetic-data tests: feature construction, None-skip accounting, gate
verdict in both directions, and reuse of the shared AUC helper."""
from calibration.v12_validation import phase_a_interaction_study as pa


def _row(label, spec, voice_abs, smooth, det, key="k"):
    return {"label": label, "doc_key": key,
            "v7_signals": {"specificity_score": spec, "specificity_student_evidence": None,
                            "author_voice_absence": voice_abs, "grounding_gap": 0.5,
                            "sentence_smoothness": smooth, "signal_status": {}},
            "calibrated_detector_score": det, "has_comparison_text": False}


def test_features_computed_and_none_skipped():
    row = _row("ai_paraphrased", spec=0.8, voice_abs=0.3, smooth=0.9, det=0.6)
    feats = pa.compute_features(row)
    assert abs(feats["specificity_x_smooth"] - 0.8 * 0.9) < 1e-9
    assert abs(feats["voice_presence_x_det"] - 0.7 * 0.6) < 1e-9
    assert feats["spec_student_ev_x_smooth"] is None  # input None -> feature None


def test_gate_pass_on_separable_synthetic():
    rows = ([_row("ai_paraphrased", 0.9, 0.2, 0.9, 0.8, key=f"p{i}") for i in range(10)]
            + [_row("ai_generated_like", 0.1, 0.8, 0.9, 0.8, key=f"g{i}") for i in range(10)]
            + [_row("student_owned", 0.9, 0.2, 0.1, 0.1, key=f"s{i}") for i in range(10)])
    result = pa.run_study(rows, prof_by_key={f"s{i}": ("higher" if i % 2 else "lower") for i in range(10)})
    assert result["gate_verdict"] == "PASS"
    assert result["winner"]["effective_auc_vs_generated"] >= 0.70


def test_gate_fail_on_inseparable_synthetic():
    rows = ([_row("ai_paraphrased", 0.5, 0.5, 0.5, 0.5, key=f"p{i}") for i in range(10)]
            + [_row("ai_generated_like", 0.5, 0.5, 0.5, 0.5, key=f"g{i}") for i in range(10)]
            + [_row("student_owned", 0.5, 0.5, 0.5, 0.5, key=f"s{i}") for i in range(10)])
    result = pa.run_study(rows, prof_by_key={f"s{i}": "higher" for i in range(10)})
    assert result["gate_verdict"] == "FAIL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd poc && python -m pytest calibration/v12_validation/test_phase_a_interaction_study.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement** — `compute_features(row) -> dict[str, float|None]`, `run_study(...)` (per-feature one-vs-rest and para-vs-generated / para-vs-polished AUCs via the SHARED helper, direction-aware winner by effect size, ESL directional check on the winner, p10/p90 of the winner over all rows), CLI main() that loads captures, builds `prof_by_key` from the corpus (import-order gotcha applies), writes numbers-only JSON. If you extracted a shared gate helper from `paraphrase_feature_study.py`, its existing tests must still pass.

- [ ] **Step 4: Run tests to verify they pass** — the new file AND `test_paraphrase_feature_study.py` (shared-helper regression) AND `test_no_text_leakage.py`.

- [ ] **Step 5: Commit**

```bash
git add poc/calibration/v12_validation/phase_a_interaction_study.py poc/calibration/v12_validation/test_phase_a_interaction_study.py <paraphrase_feature_study.py if refactored>
git commit -m "feat(v12): Phase A interaction feature study — content-humanness x surface-AIness, shared gate methodology"
```

### Task 3 (MAIN SESSION): Run Phase A study — GATE DECISION

- [ ] **Step 1: Run**

```bash
cd poc && python -m calibration.v12_validation.phase_a_interaction_study --captures calibration/v12_validation/captures/signals_fused_v3.jsonl
```

- [ ] **Step 2: Commit the study JSON** (numbers-only; leakage test green) with message `data(v12): Phase A interaction study — <PASS/FAIL>, winner <name> effective AUC <x>`.

- [ ] **Step 3: GATE.** PASS → Tasks 4–6. FAIL → skip to Task 7 (Phase B). Report the AUC table + verdict to the owner either way before proceeding.

### Task 4 (gated on A-PASS): Tuner `--para-floor` constraint

**Files:**
- Modify: `poc/calibration/v12_validation/tune_weights.py` (`satisfies_constraints`, `search`, `main`)
- Test: extend `poc/calibration/v12_validation/test_tune_weights.py`

**Interfaces:**
- Consumes: existing `satisfies_constraints(metrics, baseline, generated_floor) -> bool` and `search(..., generated_floor, ...)`.
- Produces: `satisfies_constraints(metrics, baseline, generated_floor, para_floor=0.0) -> bool` (backward-compatible default 0.0 = no constraint); `search(..., para_floor, ...)`; CLI `--para-floor` (default 0.0; the V8 run passes 0.40). Constraint: `metrics["per_class"]["ai_paraphrased"]["primary_accuracy"] >= para_floor`, enforced wherever the two existing constraints are (search filter, refinement re-check, holdout gate).

- [ ] **Step 1: Write the failing test**

```python
def test_para_floor_constraint():
    metrics = {"macro_primary_accuracy": 0.7, "student_owned_false_ai_rate": 0.05,
               "per_class": {"ai_generated_like": {"primary_accuracy": 0.8},
                              "ai_paraphrased": {"primary_accuracy": 0.30},
                              "student_owned": {"primary_accuracy": 0.9},
                              "ai_assisted_polished": {"primary_accuracy": 0.7}}}
    baseline = {"student_owned_false_ai_rate": 0.10, "ai_generated_like_accuracy": 0.78}
    from calibration.v12_validation.tune_weights import satisfies_constraints
    assert satisfies_constraints(metrics, baseline, generated_floor=0.08)                    # default: no para floor
    assert not satisfies_constraints(metrics, baseline, generated_floor=0.08, para_floor=0.40)
    metrics["per_class"]["ai_paraphrased"]["primary_accuracy"] = 0.45
    assert satisfies_constraints(metrics, baseline, generated_floor=0.08, para_floor=0.40)
```

- [ ] **Step 2: Verify it fails** (TypeError: unexpected keyword) → **Step 3: implement** (thread `para_floor` through search filter at ~L572-576, refinement re-check ~L526, holdout gate ~L600, CLI) → **Step 4: full tuner test file green** → **Step 5: Commit** `feat(v12): tuner --para-floor third hard constraint (tune + refinement + holdout)`.

### Task 5 (gated on A-PASS): Production wiring of the winning interaction signal

**Files:**
- Modify: `poc/detect_v7/signal_adapter.py` (new derived row; signals 16→17)
- Modify: `poc/detect_v7/weights.json` (`ai_paraphrased_without_comparison` gains the new signal at starter weight 0.20 renormalized; `_notes` provenance)
- Test: extend `poc/test_detect_v7_signal_adapter.py`

**Interfaces:**
- Consumes: the Phase A winner's definition from `phase_a_interaction_study.json` (feature = product of two already-adapted values; e.g. winner `voice_presence_x_det` ⇒ `paraphrase_mismatch = (1 − author_voice_absence) × calibrated_detector_score`). The exact inputs come from the committed study JSON — the implementer reads it and transcribes the winner's formula.
- Produces: `v7_signals["paraphrase_mismatch"]` (name fixed regardless of which product won; the study JSON's `_notes` records the winning composition), status ok/unavailable, clamp01, derived INSIDE `adapt_paragraph_signals` from already-adapted values + the threaded `raw_signals["calibrated_detector_score"]` (both available since the specificity split — no new plumbing).

- [ ] **Step 1: Failing tests** — value = product of the winner's inputs on a full row; `None` when either input missing (status `unavailable`); signal-count assertion 16→17; partition/clamp sanity. Concrete test code written against the winner named in the study JSON (the implementer transcribes actual numbers).
- [ ] **Step 2: verify fail → Step 3: implement adapter row (numbered-comment style, cite the study JSON as provenance) → Step 4: adapter + category_scoring + pipeline_bridge test files green.**
- [ ] **Step 5: weights.json** — insert `{"signal": "paraphrase_mismatch", "weight": 0.20, "direction": "direct"}` into `ai_paraphrased_without_comparison`, renormalize that list to sum 1.0 (scale existing entries by 0.8), `_notes` update. The tuner owns the final weight — this is only a sane starting point.
- [ ] **Step 6: Commit** `feat(v7): paraphrase_mismatch signal — Phase A winner wired (study-gated), starter weight for tuning`.

### Task 6 (MAIN SESSION, gated on A-PASS): Re-capture, re-tune, validate, promote

- [ ] **Step 1: capture v4** (same command as Task 1, `--out .../signals_fused_v4.jsonl`); verify 17 signals, `paraphrase_mismatch` status ok.
- [ ] **Step 2: tune** `python -m calibration.v12_validation.tune_weights --captures .../signals_fused_v4.jsonl --baseline-json /tmp/v8_anchors.json --para-floor 0.40 --trials 2000 --seed 46` where `/tmp/v8_anchors.json` = current fused baseline with anchors false-AI 0.10 / gen-like 0.78 (write it the way the frontier probe did, `_notes`-labeled). NO_CANDIDATE_GENERALIZES → STOP, report to owner (fallback decision, Task 8).
- [ ] **Step 3: e2e validate** — `measure --fused` under the candidate (`DRAFTPROOF_V7_WEIGHTS_PATH`), compare offline == e2e (±2 pts per class), check ALL acceptance criteria from Global Constraints.
- [ ] **Step 4: tier-invariance** — breakdown flag on/off badge byte-compare (the one-doc script used at every prior promotion).
- [ ] **Step 5: promote** — candidate `category_weights` + band into `poc/detect_v7/weights.json` with `_notes` provenance; full V7 + v12 suites; leakage guard; commit `feat(v7): promote V8 weights — ai_paraphrased 0%-><X>% (seed 46), all floors held`.

### Task 7 (only if Phase A FAILED): Phase B raggedness study

**Files:**
- Create: `poc/calibration/v12_validation/phase_b_raggedness_study.py` + test + committed JSON

Same shape as Task 2 (shared gate helper, same CLI flags, same ESL check via corpus text resolution). Features (computed from corpus TEXT via `measure._resolve_text`, lexical/structural ONLY — no embeddings, no model downloads):
1. `paragraph_len_cv` — coefficient of variation of paragraph word counts (paragraphs split on blank lines; docs with <3 paragraphs skipped-and-counted).
2. `sentence_len_cv_by_para` — mean within-paragraph sentence-length CV (sentences via `detect.deberta_windowing.split_sentences`).
3. `connective_irregularity` — variance of per-paragraph connective density (connective list read from the study's own committed config block, flagged `_notes`-style as study-only, NOT a production detect list — the no-hardcode rule binds production code; a measurement script's candidate lexicon is data recorded in the output).
4. Each of 1–3 crossed with the capture rows' `sentence_smoothness` and `calibrated_detector_score` (join by doc_key hash).

Gate identical to Phase A. PASS → return to Tasks 4–6 with the Phase B winner (adapter derivation will need the raw text-level feature threaded through `pipeline_bridge._build_raw_signals` — a heavier wiring task; the orchestrator re-scopes Task 5 accordingly with Opus). FAIL → Task 8.

### Task 8 (only if A AND B FAILED): STOP — owner decision

Present both study JSONs. Options for owner: (a) authorize Phase C scoping (per-token predictability split; touches `poc/detect/`, ESL FPR gate, separate plan), or (b) ship the fallback: `breakdown_composer` display merge — paraphrased share folded into an "AI-transformed" display band with an uncertainty note (annotate-never-suppress; concrete design written at decision time WITH the owner, since its copy/UX needs their sign-off). Do not build either without the decision.

---

## Orchestration Map

| Task | Runner | Review |
|---|---|---|
| 1, 3, 6 (runs, gates, promotion) | MAIN SESSION | orchestrator |
| 2 (Phase A study script) | Sonnet | Sonnet reviewer |
| 4 (tuner para-floor) | Sonnet | Sonnet reviewer |
| 5 (production wiring) | Opus | Opus reviewer |
| 7 (Phase B script, if reached) | Sonnet | Sonnet reviewer |
| 8 (stop point) | owner + orchestrator | — |

Every subagent prompt carries the Tool Routing preamble (rules/agents.md) + this plan's Global Constraints. Subagent completion claims are independently verified (files exist, tests run) before marking done.
