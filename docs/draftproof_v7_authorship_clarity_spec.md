# DraftProof V7 Technical Design Spec: Authorship Clarity Breakdown

**Version:** 3.0 (supersedes v2 draft `draftproof_authorship_clarity_breakdown_tech_spec_v2.md`)
**Stack name:** **V7** — production detection/report is currently the v6-era stack; V7 is the next additive layer.
**Status:** Design — approved direction: Modal serverless GPU, pay-per-request only.

---

## 0. Decision Log (changes from the v2 draft)

| # | v2 draft said | V7 spec says | Why |
|---|---|---|---|
| D1 | "Modal or RunPod" | **Modal only** (`modal.com`) | Owner decision. Credentials already in `.env` (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`). `min_containers=0` → cost only on request. |
| D2 | Unversioned "Phase 1" | Everything ships under the **V7** namespace + kill switch `DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN` (default OFF until validated) | Matches house pattern (additive composer + env kill switch); production stays v6 until flipped. |
| D3 | "App server" runs quick-scan inference | **Celery worker** runs Quick Scan; API server unchanged | Production detection already runs in the worker (Koyeb, HF volume). The v2 draft's "app server" wording doesn't match the deployed topology. |
| D4 | Hardcoded fusion/category weights in formulas | **All weights and thresholds live in a config file** (`poc/detect_v7/weights.json`), loaded at runtime, tuned only via the validation harness | NO-HARDCODE gate. The v2 numbers become the *initial contents of the config*, never literals in code. |
| D5 | Show four percentages from day one | **Bands + primary category first; exact percentages only after the §12 validation set exists** | Precision-first: the v2 draft's own §25 example normalizes to 19/26/27/28% — near-uniform, false precision. Percentages without 4-class labels are unverifiable. |
| D6 | 13 new signals implied | **10 of 13 map to existing code** (see §5 mapping table); only 3 are new work | Verified against `poc/detect/` — avoids rebuilding the shipped signal engine. |
| D7 | Isotonic calibration recommended generically | Calibration follows the **deberta_signal v2 lesson**: threshold-proportion over windows, never mean-then-isotonic | v1 isotonic degenerated in production (scores collapsed to ~0). Documented in `poc/detect/deberta_signal.py` header. |
| D8 | ESL guard as a runtime multiplier | ESL guard = runtime confidence cap **plus** the binding **SCoCESLE FPR gate** on every `poc/detect/` change | The gate (`poc/calibration/fpr_subgroup_gate.py --compare`) is already enforced by the pre-push hook; V7 does not bypass it. |
| D9 | "AI-paraphrased" always shown | Without comparison text (draft/source), AI-paraphrased is **folded into a combined "AI-rewritten-like" presentation** with an uncertainty flag, not a separate % | The signal is weakly identifiable without an original; showing a standalone number would overclaim. |
| D10 | DeBERTa-large at 45–60% fusion weight | Accepted, **recorded as an explicit reversal** of the earlier "comparison-not-replacement" stance: in Deep Scan, DeBERTa-large becomes the primary detector | Owner-commissioned spec; decision is now on the record. |

Product principle, categories (§3 of the v2 draft), risk-control wording (§27), and policy modes (§19) are **unchanged** — they align with the shipped positioning ("authorship signals, not a misconduct verdict") and are incorporated by reference.

---

## 1. Objective (unchanged)

Per-paragraph, then document-level, distribution across four authorship categories:

1. **Student-owned**
2. **AI-assisted / polished**
3. **AI-paraphrased** (comparison-text dependent — see D9)
4. **AI-generated-like**

Output is evidence + review guidance. Final judgement belongs to the school/teacher/policy.

---

## 2. V7 Code Layout

```text
poc/detect_v7/                      # NEW — additive, imports FROM poc/detect, never modifies it
├── __init__.py
├── config.py                       # loads weights.json + env overrides; NO literals in scoring code
├── weights.json                    # all fusion + category weights and thresholds (v2 defaults as data)
├── signal_adapter.py               # maps existing poc/detect signals → V7 signal names (§5 table)
├── detector_fusion.py              # calibrated_detector_score from available detectors
├── category_scoring.py             # 4-category raw scores + normalization + confidence
├── aggregate.py                    # paragraph → document weighted aggregation
├── breakdown_composer.py           # report-facing composer (additive; house composer pattern)
└── modal_client.py                 # HTTPS client for the Modal deep-scan endpoint + fallback logic

modal_endpoints/                    # NEW top-level dir (deployed via `modal deploy`, not Koyeb)
└── deberta_large_detector.py       # Modal App: DeBERTa-v3-large chunk inference only

worker/app/…                        # scan task gains an optional V7 step (kill-switched)
```

Kill switches (house pattern, all default OFF until validation passes):

```text
DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN=0   # master switch — compose + surface the breakdown
DRAFTPROOF_V7_DEEP_SCAN=0              # call the Modal endpoint (Deep Scan)
```

Rules:
- `poc/detect_v7` is **additive**. It never mutates `ai_likelihood`, tier, or badge (same invariant as the authenticity dashboard and submission_risk composers).
- Any change that touches `poc/detect/` itself (e.g. exposing a criterion sub-score) runs the **ESL FPR gate** before push.

---

## 3. Deployment Architecture

```text
Student essay
   ↓
draftproof-api (Koyeb)            — unchanged: job record, Celery enqueue, SSE relay
   ↓
Celery worker (Koyeb, HF volume) — Quick Scan: existing DetectionRunner signals
   │                                + V7 signal_adapter + fusion + category scoring
   │
   ├── DRAFTPROOF_V7_DEEP_SCAN=1 ──→ Modal serverless endpoint (HTTPS)
   │                                  DeBERTa-v3-large chunk scores
   │                                  min_containers=0 → $0 when idle
   │
   ↓
Fusion + category scoring (worker)
   ↓
report JSON (+ authorship_breakdown block) → R2 → all surfaces
```

### 3.1 Worker responsibilities (Quick Scan — no new infra)

- Everything the scan does today (DetectionRunner, fakespot RoBERTa via `deberta_signal.py`, criteria engine).
- V7 additions: signal adaptation, detector fusion, category scoring, aggregation, composer.
- Fallback owner: if Modal is cold-timeout/unavailable → complete as Quick Scan, set `deep_scan: "unavailable"`, cap confidence at `medium`, avoid strong AI-generated-like claims (v2 §27.5 preserved).

### 3.2 Modal endpoint responsibilities (Deep Scan only)

The Modal app does **only**:
- Load DeBERTa-v3-large detector checkpoint (baked into the image at build time — never runtime-download; that is the 60–180 s cold-start failure mode the v2 draft warned about).
- 512-token chunk inference (overlap 64–96 tokens), batch 4–8.
- Return per-chunk raw scores + aggregation metadata. **No** report logic, no fakespot, no MiniLM, no LanguageTool.

Modal app shape (verify API details against current Modal docs at implementation time):

```python
app = modal.App("draftproof-deberta-deep-scan")

@app.cls(
    gpu="L4",                      # 24 GB — v2 §7.3 recommended tier
    min_containers=0,              # scale to zero — pay per request only
    scaledown_window=120,          # keep warm ~2 min after a request for burst reuse
    image=image_with_baked_model,  # model weights in the image layer
    secrets=[…],                   # endpoint auth token
)
class DebertaDetector:
    @modal.enter()
    def load(self): ...            # load once per container
    @modal.fastapi_endpoint(method="POST")
    def score(self, req): ...      # chunks in → scores out
```

- **Auth:** worker → Modal calls carry a shared bearer token (new env var `DRAFTPROOF_MODAL_ENDPOINT_TOKEN` + endpoint URL `DRAFTPROOF_MODAL_ENDPOINT_URL`). `MODAL_TOKEN_ID/SECRET` in `.env` are for *deploying*, not for runtime calls.
- **Timeout budget:** worker waits ≤ 60 s (cold start 15–30 s + inference 1–5 s); on timeout → Quick Scan fallback. Celery task budget (600 s) is unaffected.
- **Cost model:** zero idle cost; per-scan GPU cost ≈ seconds of L4 time. No always-on GPU anywhere.

### 3.3 UX for cold starts

Scan is already async (Celery → Redis stream → SSE), so:
- Publish Quick Scan progress immediately.
- Deep Scan result merges when the Modal call returns; if the report was already finalized on timeout, the deep result is dropped (no re-open) — keep it simple in Phase 1B.

---

## 4. Detector Fusion

Detectors available per scan (graceful degradation, never break on a missing one):

| Detector | Where | Status |
|---|---|---|
| `fakespot-ai/roberta-base-ai-text-detection-v1` | worker (shipped) | Production baseline — `poc/detect/deberta_signal.py` v2 (threshold-proportion) |
| DeBERTa-v3-large detector (checkpoint **TBD — Task 1B.0**) | Modal | Deep Scan challenger |
| `OU-Advacheck/deberta-v3-base-daigenc-mgt1a` | worker (optional) | **On hold** — first verify it was not one of the two candidates rejected in the June 2026 evaluation (one inverted, one GPT-2-era) |

Fusion (weights from `weights.json`, shown here as initial data values):

```text
deep scan (both):   0.60 × deberta_large + 0.40 × fakespot
quick scan (one):   1.00 × fakespot
```

Calibration requirements (blocking, in order):
1. **Task 1B.0:** pin the exact DeBERTa-large checkpoint (evaluate top Kaggle-DAIGT-era candidates on SCoCESLE + AI set exactly as Phase 0 did for fakespot).
2. Calibrate it on SCoCESLE **before** any fusion weight > 0 — raw fakespot ESL FPR was 20.5% pre-calibration; assume the same class of problem.
3. Use the **v2 lesson**: threshold-proportion aggregation over chunks; if a mapping to probability is needed, calibrate per-chunk *before* aggregation, never isotonic-after-mean.
4. `detector_disagreement` = |calibrated scores| spread → confidence reducer (already exists as `detector_agreement_risk` in `external_grouped_scoring.py`; reuse it).
5. Separate calibration curves for ESL and short paragraphs remain the target end-state (v2 §12), gated on validation data.

Interpretation invariant (v2 §12, kept verbatim): the score means *detector-like signal strength*, not probability of cheating. Raw model scores are never shown to users.

---

## 5. Signal Plan — Mapping Table (verified against code)

The v2 draft's 13 signals, mapped to what `poc/detect/` actually produces:

| V7 signal (spec name) | Existing source | File | Work |
|---|---|---|---|
| `grounding_gap` | `grounding_gap_risk` / `source_grounding` criterion + grounding diagnosis | `poc/detect/grounding_diagnosis.py`, `poc/detect/criteria/source_grounding.py` | **Adapter only** |
| `generic_density` | `generic_assertion_risk` / `generic_phrase_density` | `poc/detect/criteria/generic_phrases.py` | **Adapter only** |
| `predictable_structure` | `repetitive_structure` + `repeated_sentence_structure_risk`, `formulaic_conclusion_risk`, `signpost_paragraph_risk` | `poc/detect/criteria/repetitive_structure.py`, `external_grouped_scoring.py` | **Adapter (blend)** |
| `sentence_smoothness` | inverse `burstiness` + `surprisal` (low burstiness + low surprisal = smooth) | `poc/detect/criteria/burstiness.py`, `surprisal.py` | **Adapter (derived)** |
| `author_voice_absence` | `lived_detail_risk`, `human_anchor_score`, authorship windows, critical-thinking voice dims | `external_grouped_scoring.py`, `authorship_windows.py`, `critical_thinking.py` | **Adapter (blend)** — needs a definition pass |
| `citation_gap` | `citation_weakness_risk` / `citation_grounding_gap` (polish_vs_grounding) | `poc/detect/citation.py`, `criteria/polish_vs_grounding.py` | **Adapter only** |
| `semantic_drift` | semantic shape + `draft_evolution` criterion (comparison-text dependent) | `poc/detect/semantic_shape.py`, `criteria/draft_evolution.py` | **Adapter**; full drift needs comparison text |
| `sentence_variance` | `burstiness` (this *is* sentence-length variance) | `criteria/burstiness.py` | **Adapter only** |
| `specificity_score` | specificity criterion (exact match) | `criteria/specificity.py` | **None** |
| `local_style_shift` | style-shift criterion (exact match) | `criteria/style_shift.py` | **None** |
| `detector_disagreement` | `detector_agreement_risk` | `external_grouped_scoring.py` | **Adapter only** |
| `paraphrase_pattern_score` | *(partial: `structural_reuse`)* | — | **NEW** — Phase 1C, MiniLM-based |
| `meaning_preservation_score` | exists only rewrite-side (`rewrite_pipeline.py`) | — | **NEW** for detect; requires comparison text |
| `esl_false_positive_risk` | handled today via corpus calibration + FPR gate, not per-paragraph | — | **NEW** — per-paragraph estimator; until built, apply document-level ESL confidence cap |

Net: **the "19-signal engine" is real and covers 11 of 14 rows via `signal_adapter.py`.** New model work is exactly three signals, all deferrable past Phase 1A.

New Phase-1A dependencies on the worker (memory check required — 4 GB target):
- `sentence-transformers/paraphrase-MiniLM-L6-v2` (~90 MB) — Phase 1C, not 1A.
- spaCy small + textstat — only if an adapter blend actually needs them; do not add speculatively.
- LanguageTool: **deferred** (v2 §5.2 kept).

---

## 6. Category Scoring

Four raw scores per paragraph, computed in `category_scoring.py` from adapted signals + `calibrated_detector_score`, with **all weights read from `weights.json`** (initial values = v2 draft §14/§24; they are starting data, tuned only through the validation harness):

- `student_owned_raw`, `ai_assisted_polished_raw`, `ai_paraphrased_raw`, `ai_generated_like_raw` — formulas as in v2 §14.1–14.4, including `midrange()` for the polished band.
- Without comparison text: paraphrase fallback formula + `uncertainty_flags: ["paraphrase_without_original_draft"]`, and paraphrased is **presented merged** into "AI-rewritten-like" (D9).

Normalization: sum-to-1 as in v2 §15, **plus a flatness check**:

```text
if (top_category - second_category) < 0.10:
    confidence = "low"; presentation = "mixed signals"   # v2 §16 rule, enforced in code
if max_category < 0.35:
    presentation = "mixed signals"                        # near-uniform distribution → never show 4 numbers
```

Confidence tiers, primary driver, recommended action: per v2 §16 unchanged.

ESL guard (v2 §18) — kept, with values in config not code:

```text
esl_high_threshold, ai_generated_damping, disagreement_threshold → weights.json
effect: damp ai_generated_like_raw, cap confidence, add review note
```

### 6.1 Display policy (D5 — the biggest change from v2)

Phase 1 (no 4-class validation set yet):

```text
Authorship Clarity Breakdown

Student-owned:            Strong   ████████░░
AI-assisted / polished:   Some     ███░░░░░░░
AI-rewritten-like:        Some     ███░░░░░░░   (paraphrased + generated merged when low confidence)
```

- Bands: `Strong / Some / Little / None` mapped from the normalized shares — band edges in `weights.json`.
- Primary category + confidence + primary driver + recommended action per paragraph (v2 §16 JSON schema kept, adding `"display_band"`).
- Exact percentages (v2 §17/§20 displays) unlock **only after** the §12 validation set shows calibration error within target. The API JSON may carry the raw shares from day one (internal + future-proofing); the *user-facing* surfaces show bands.

Document aggregation: word-count-weighted mean of paragraph shares (v2 §17 unchanged).

---

## 7. Scan Modes

| | Quick Scan | Deep Scan |
|---|---|---|
| Runs on | worker only | worker + Modal |
| Detectors | fakespot (calibrated) | fakespot + DeBERTa-large fused |
| Cost | today's cost | + seconds of L4 |
| Latency add | ~0 | warm 1–5 s / cold 15–30 s (async-merged) |
| Trigger | every scan | flag-gated; later: tier-triggered or user-selected |
| Fallback | — | Quick Scan result + `deep_scan: unavailable` + confidence cap |

---

## 8. Surfaces & Rollout (learned from policy_risk)

The breakdown is a new additive badge-class field; per the additive-composers rule it must reach **all** surfaces or explicitly declare fallback:

| Surface | Phase | Notes |
|---|---|---|
| Scan report JSON (`report.py` builder) | 1A | `authorship_breakdown` block via composer |
| Report web page (React, en+zh) | 1A | new panel; reconcile with badge-tier display (breakdown NEVER overrides badge tier) |
| Scan PDF (`render_panels.py`) | 1A | same banded panel |
| Read-time API (`draftproof-api/app/_composers/`) | 1A | pure-stdlib composer copy + sync test (house pattern) |
| Email summary | 1B | one-line breakdown |
| Rewrite re-scan comparison | 1C | before/after breakdown |

Rollout: ship dark (flag off) → internal validation on SCoCESLE + AI corpus → flip `DRAFTPROOF_V7_AUTHORSHIP_BREAKDOWN=1` → Deep Scan flag separately.

---

## 9. Validation & Metrics

Binding gates (in force today, V7 inherits them):
- `poc/calibration/fpr_subgroup_gate.py --compare` on any `poc/detect/` change (pre-push hook).
- Deterministic rewrite harness untouched (V7 is scan-side).

New V7 validation (v2 §21–22 kept, prioritized):
1. **ESL false positive rate** (SCoCESLE) — hard gate.
2. Category agreement vs reviewer labels — required before percentage display (D5).
3. Paragraph-level localization F1.
4. Stability under light edits.
5. Deep-scan latency + Modal failure rate (new operational metrics; log per-scan `gpu_endpoint_status`).

Dataset build (v2 §21 list kept: human / ESL / AI / AI-polished / AI-paraphrased / lightly-edited / citation-heavy / weak-genuine / DraftProof-rewritten / mixed). MVP 500 docs with paragraph labels where possible; SCoCESLE + existing AI sets seed categories 1–3 immediately.

MLflow: adopt for V7 calibration runs only (checkpoint choice, weight tuning, gate metrics). Not wired into production serving.

---

## 10. Implementation Plan

### Phase 1A — Quick Scan breakdown (worker-only, no new infra)
1. `poc/detect_v7/` skeleton: config loader + `weights.json` (v2 defaults as data).
2. `signal_adapter.py` — the §5 table in code, unit-tested against real DetectionRunner output.
3. `detector_fusion.py` (fakespot-only path) + `category_scoring.py` + flatness/ESL/confidence rules.
4. `breakdown_composer.py` + report JSON block + web/PDF/read-time surfaces (banded display).
5. Sync tests (API composer copy), kill switch, ESL gate run.

### Phase 1B — Deep Scan on Modal
0. **Task 1B.0 (blocking):** pin DeBERTa-large checkpoint; SCoCESLE + AI-set evaluation; calibrate (v2-lesson compliant); record in MLflow.
1. `modal_endpoints/deberta_large_detector.py` — image with baked weights, L4, `min_containers=0`, bearer auth.
2. `modal_client.py` in worker — timeout, fallback, `gpu_endpoint_status` in report.
3. Fusion weights → deep-scan values; re-run ESL gate on fused path.
4. Env: `DRAFTPROOF_MODAL_ENDPOINT_URL`, `DRAFTPROOF_MODAL_ENDPOINT_TOKEN` (worker); deploy via `modal deploy` using `.env` tokens.

### Phase 1C — Paraphrase signal
1. MiniLM on worker (memory-check first) → `paraphrase_pattern_score`.
2. Comparison-text ingestion (optional draft upload) → `meaning_preservation_score`, `semantic_drift` full version.
3. Un-merge "AI-paraphrased" from "AI-rewritten-like" when comparison text exists.

### Phase 2 — per-paragraph `esl_false_positive_risk` estimator + dedicated category classifiers.
### Phase 3 — learned fusion (replace formula weights with a constrained learned model; explainability preserved).

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Near-uniform category outputs (correlated features) | Flatness check → "mixed signals"; bands not %; Phase 2 classifiers are the real separator |
| DeBERTa-large ESL FPR like raw fakespot (20.5%) | Task 1B.0 calibration is blocking; ESL gate on fused path |
| Isotonic degeneration repeat | D7: threshold-proportion aggregation, calibrate pre-aggregation |
| Modal cold start hurts UX | Async merge; 60 s worker timeout; Quick Scan always completes |
| Worker RAM (4 GB) with MiniLM/spaCy | Defer to 1C; measure before adding; LanguageTool stays out |
| Surface drift (policy_risk lesson) | §8 surface table is part of "done" for each phase |
| Weights ossify as pseudo-hardcode | weights.json + MLflow-tracked tuning; no literals in code (D4) |

### Known limitations (measured, 2026-07-08 category-accuracy program)

- **`ai_paraphrased` remains 0% primary accuracy.** The MiniLM cosine feature family (semantic-monotony: adjacent/pairwise cosine, embedding-norm CV) was measured and **rejected by the hard gate** (best effective AUC 0.669 < 0.70 vs `ai_generated_like`; committed evidence `poc/calibration/v12_validation/paraphrase_feature_study.json`) — do not retry it or variants. Recovery needs a genuinely new feature family, likely paired-draft / rewrite-comparison based (source-vs-final rewrite-distance features; the `ai_paraphrased_with_comparison` weight variant and `meaning_preservation_score` slot already anticipate a comparison text — the missing piece is the earlier-draft workflow plus detect-side features). Tracked as the V8 recovery effort.
- Post-split reference state (198-doc fused §12 corpus, seed-45 weights): macro 65.5%, `student_owned` 97.5% / false-AI 2.5%, `ai_assisted_polished` 87.2%, `ai_generated_like` 77.5%.

---

## 12. Summary

V7 = the existing v6-era signal engine (11/14 signals already built), re-composed into a four-category authorship clarity breakdown, with:
- config-driven weights (no hardcode),
- fakespot-calibrated Quick Scan on the worker,
- Modal serverless (scale-to-zero, L4) DeBERTa-large Deep Scan,
- banded user-facing display until the validation set earns percentages,
- everything additive and kill-switched, gated by the SCoCESLE ESL FPR gate.

> DraftProof shows authorship signals, not a misconduct verdict.
