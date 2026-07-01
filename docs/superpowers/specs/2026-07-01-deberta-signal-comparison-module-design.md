# DeBERTa-Signal Comparison Module — Design Spec

**Date:** 2026-07-01
**Status:** Draft (pending implementation plan)
**Source spec:** `Downloads/draftproof_ai_writing_signal_classifier_technical_design.md`

---

## 1. Purpose

Add a **second, independent AI-writing signal score** to every scan, produced by a different
detector method than the existing calibrated composite. The user sees both scores side-by-side
so they can **compare two detector perspectives**.

- **NOT** a replacement for the existing `ai_likelihood_score` / tier.
- **NOT** "this document is X% AI-generated." Wording: *"X% of the document shows AI-like writing
  signal under the DeBERTa detector."*
- The existing composite detector remains the authoritative score for tiering, gating, and the
  report headline. This module is strictly advisory/additive.

This reopens the previously-settled "build a fine-tuned classifier = NO" decision
(`project_detector_upgrade_settled.md`) under a **narrower scope**: we are *not* building a
detector to replace the composite — we are adding a deliberately-different second opinion for
comparison. The two scores are intentionally orthogonal (composite = perplexity/topk/grounding
stylistic + structural; DeBERTa = stylistic token-pattern classifier).

---

## 2. Decisions (from brainstorm)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model path | **Off-the-shelf pretrained checkpoint first** | Ships a real comparison in days; fine-tuning our own is multi-month data+training. Own fine-tune remains a future option. |
| Checkpoint selection | **Benchmark candidates against SCoCESLE, then pick** | Choice driven by our own ESL data, not reputation. |
| Run timing | **Inline in every scan** | Score always present, both ready together. Latency roughly doubles (~4s → ~8–14s) — accepted. |
| Output richness | **Single number + band, side-by-side** | Minimal MVP. No sentence localization yet. |
| Disagreement UX | **Both scores + short agree/disagree note** | Honest framing; no averaging/smoothing. |
| ESL safety | **Gate on SCoCESLE corpus before wiring in** | Highest-stakes documented risk. Recalibrate (isotonic) or demote to advisory if raw FPR fails. |
| Architecture | **Inline additive stage inside `DetectionRunner`** | Matches existing composer pattern; zero new infra. |
| Calibration unit | **Separate unit** (not folded into composer) | Makes the ESL gate re-runnable independently. |
| Model provisioning | **Download to `/app/hf_cache`** (existing HF volume) | Reuses worker pattern; no new download mechanism. |

---

## 3. Architecture & Integration

New inline stage inside `poc/detect/run.py::DetectionRunner.run_all()`, slotted after
overall_risk/layer3 computation and before the `DetectionReport` return.

```
DetectionRunner.run_all()
  ├─ (existing) build + run detectors (predictability, topk, grounding, ai_generation, …)
  ├─ (existing) overall_risk, layer3 → ai_likelihood_score, tier
  ├─ (existing) postprocess, top_findings, rewrite_decision
  ├─ NEW: deberta_signal.compose()  →  ai_signal_deberta field on report
  └─ (existing) return DetectionReport(...)
```

**Safety contract** (copied verbatim from `poc/detect/authenticity_dashboard.py:3-7`):

> STRICTLY ADDITIVE: it never feeds back into the tier, ai_likelihood_score, the external
> estimate, or any gate — same contract as submission_risk.py.

Nothing downstream (rewrite decision, tier, gate, external estimate) reads
`ai_signal_deberta`. It rides along on the report JSON only.

### Kill switch

`DRAFTPROOF_DEBERTA_SIGNAL` (default `"1"` ON), mirroring `DRAFTPROOF_AUTHENTICITY_DASHBOARD`.
When off, `maybe_attach()` returns `None`; the stage is a no-op; the primary scan path is
byte-identical. Declared in `worker/entrypoint.sh` next to the existing kill-switches:

```bash
export DRAFTPROOF_DEBERTA_SIGNAL="${DRAFTPROOF_DEBERTA_SIGNAL:-1}"
export DRAFTPROOF_DEBERTA_MODEL="${DRAFTPROOF_DEBERTA_MODEL:-<chosen-checkpoint-repo-id>}"
```

### Dual-copy (mandatory)

Per `project_additive_composers_all_surfaces`: a new additive field must reach the worker
(real inference) AND the read-time API surface. MVP scope likely passes the stored score
through with no derivation, but the mirror file must exist:

- `poc/detect/deberta_signal.py` — worker side (real inference)
- `draftproof-api/app/_composers/deberta_signal.py` — pure-stdlib mirror (read-time pass-through / enrichment)

---

## 4. Components (5 units, independently testable)

| # | Unit | Responsibility | Dependencies |
|---|------|----------------|--------------|
| 1 | `poc/detect/_download_deberta.py` | One-time warm-download of chosen checkpoint into `$HF_HOME/hub`. Run once against the volume (or self-downloads on first use via fallback). | `transformers`, env |
| 2 | `poc/detect/deberta_model.py` | Lazy singleton model loader + inference: text window → raw prob. Wraps the `local_files_only=True` + network fallback pattern from `poc/predictability/scanner.py:154-159`. | HF volume |
| 3 | `poc/detect/deberta_windowing.py` | Text → overlapping sentence windows (doc §6–7) → window probs → sentence + document aggregation (doc §8–9). Pure-Python aggregation over unit-2 probs. | unit 2 |
| 4 | `poc/detect/deberta_calibrate.py` | Isotonic recalibration fitted on SCoCESLE if raw ESL FPR fails the gate. Saves/loads a calibrator; reports `calibrated: bool`. Pure math on a prob array. | SCoCESLE corpus |
| 5 | `poc/detect/deberta_signal.py` *(composer)* | The additive stage. `compose()` + `maybe_attach()` + `_enabled()`, kill-switch default ON. Emits `ai_signal_deberta`. The only unit `DetectionRunner` imports. | units 2–4 |

**Why split:** each unit is independently testable. Windowing (3) and calibration (4) are
pure Python/math — verifiable without running the 13-min corpus gate. Only the composer (5)
ties them together. Separating calibration makes the ESL gate re-runnable without touching
model code.

---

## 5. Model Provisioning

Reuses the existing worker HF-volume infrastructure. **No new download mechanism.**

### Resolution (three layers, already in production)

1. `worker/Dockerfile:19` — `ENV HF_HOME=/app/hf_cache` (the attached volume `eeb30e19`).
2. `worker/entrypoint.sh:7` — `CACHE_DIR="${HF_HOME}/hub"`; **boot-fails** if `/app/hf_cache`
   is not a real mounted volume (device check, lines 86–94). A misconfigured deploy crashes
   loudly rather than silently re-downloading.
3. Each loader passes `cache_folder=os.environ.get("HF_HOME")` and uses `local_files_only=True`
   with a network fallback — once the checkpoint is on the volume, no runtime network is needed.

### Loader code shape (unit 2)

```python
# poc/detect/deberta_model.py
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)
except (OSError, EnvironmentError):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_folder=os.environ.get("HF_HOME"))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_folder=os.environ.get("HF_HOME"))
```

### Cold-start mitigation

- **Lazy load** (singleton in unit 2): the ~400MB model loads on first scan, not at worker import.
- **Pre-download** (unit 1): run once against the volume so the first real scan after deploy is
  warm. Only an un-warmed fresh volume pays the cold-download cost (doc §3: 10–40s).

---

## 6. Output Schema

New additive field on the report:

```json
{
  "ai_signal_deberta": {
    "score": 34.0,            // 0–100, AI-like writing signal %
    "band": "green"|"amber"|"orange"|"red",  // SAME traffic-light legend the composite renders (frontend reportHelpers.js + i18n tiers: Low/Moderate/High/Critical Risk), so the two scores are directly comparable
    "confidence": "low"|"medium"|"high",
    "model_version": "deberta_signal_v1 (<checkpoint-repo>)",
    "calibrated": true,       // true if isotonic fitted on SCoCESLE; false = raw, demoted advisory
    "available": true,        // false if disabled / too-short / load-failed
    "caveat": "..."           // "calibrated on DraftProof ESL corpus" / "raw, uncalibrated" etc.
  }
}
```

`score` is the **calibrated**, band-mapped percentage — NOT the raw model probability. This
prevents the "61% reads as accusation" misread the source doc warns against (§1, §24-mistake-10).
Raw checkpoint probability is internal only.

**Band comparability:** the two scores must be on the **same %→band scale** for the side-by-side
to be meaningful. The composite's displayed legend is the traffic-light **green/amber/orange/red**
(Low/Moderate/High/Critical Risk), with `ai_score` cutoffs **0.32 / 0.48 / 0.65** (×100 = 32/48/65),
anchored in `layer3_scoring._derive_ai_tier` and `draftproof-frontend/.../reportHelpers.js`. The
DeBERTa module uses these **exact same cutoffs and keys** (`poc/detect/deberta_calibrate.map_to_band`)
so its band is directly comparable with zero new frontend band i18n. If a checkpoint cannot be
mapped onto this shared scale without distorting its ESL FPR, that is a Phase-0 failure for that
candidate.

---

## 7. Disagreement UX (report surface)

Two scores shown side-by-side with a short interpretive note. **No averaging, no smoothing.**

- DraftProof composite: *X% AI-like* (band)
- DeBERTa signal: *Y% AI-like* (band)
- Note (auto-selected by band relationship):
  - **Agree** (same band): *"Both detection methods place this in the {band} band."*
  - **Disagree** (different band): *"Two detection methods disagree ({band_a} vs {band_b}).
    This is common when methods use different signals. Review the flagged passages yourself."*

The disagreement note is written so that disagreement on genuinely-human ESL text does **not**
read as an accusation — consistent with `feedback_expose_ugly_side` (never false-reassure, but
never false-accuse).

**Frontend scope:** MVP surfaces the field on the report page (`draftproof-frontend`, report
component) + report PDF (`render_panels.py`). Exact placement deferred to implementation plan.

---

## 8. ESL/L2 False-Positive Gate (PRE-IMPLEMENTATION, mandatory)

Before the module touches real student text:

1. **Benchmark candidate checkpoints** (chosen via research, July 2026) through the existing
   `poc/calibration/measure_detector_signals.py` harness against the 272-essay SCoCESLE corpus.
2. Measure **ESL FPR** (fraction of genuine ESL essays scored above the high band) + **AI recall**
   on the matched AI set.
3. Pick the checkpoint with acceptable ESL FPR and best AI recall.
4. If the winner's raw ESL FPR is unacceptable:
   - **Recalibrate** via isotonic regression fitted on SCoCESLE (unit 4) → re-measure → ship with
     `calibrated: true`.
   - If isotonic cannot bring ESL FPR under threshold → **demote to advisory-only** with a visible
     "raw, uncalibrated" caveat, OR exclude from MVP.

**Acceptance:** ESL FPR must be no worse than the existing composite's documented behavior
(`project_esl_fpr_corpus_and_specificity.md`: own `ai_likelihood` mean 31%, 97% <50%).

This gate is the single most important acceptance criterion — it is the #1 documented failure mode.

---

## 9. Error Handling

The stage **fail-opens** and never blocks the primary scan:

| Failure | Behavior |
|---------|----------|
| Kill-switch off | `maybe_attach()` → `None`. Field absent. |
| Model load fails / OOM | `available: false`, primary scan completes normally. |
| Text < 150 words (doc §5) | `available: false` with caveat "too short for windowed signal". Field present but marked unavailable (not a missing field). |
| Inference exception | Caught, logged (safe fields only — §11), `available: false`. |
| SCoCESLE gate not yet run | `calibrated: false`, advisory caveat. Never ship raw-as-authoritative. |

The composite detector's tier/`ai_likelihood_score` is **never** affected by any failure here.

---

## 10. Testing

| Layer | Approach |
|-------|----------|
| Unit 3 (windowing) | Pure-Python tests: given known sentence splits, assert window overlap + aggregation math. No ML. |
| Unit 4 (calibration) | Pure-math tests: given a prob array + labels, assert isotonic fit is monotonic + idempotent. No ML. |
| Unit 5 (composer) | Mock unit 2/3/4: assert `compose()` emits correct schema; `maybe_attach()` honors kill-switch; disabled = `None`. |
| Unit 2 (model) | Lazy-load test with a tiny stub checkpoint or mocked logits; assert singleton + fallback path. |
| Integration | `DetectionRunner` end-to-end with kill-switch ON: assert field present, tier/ai_likelihood unchanged (additive invariant). |
| ESL gate (§8) | The acceptance gate itself — corpus run, measured FPR, recorded. |

**Additive invariant test (must pass):** with the module ON vs OFF, `overall_risk`, `tier`,
`ai_likelihood_score`, `rewrite_decision`, and the external estimate must be **bit-identical**.
This is the enforceable proof of the strict-additive contract.

---

## 11. Privacy & Logging

Per source doc §20: **never log full essay text.** Safe logs only:

```
request_id, timestamp, word_count, sentence_count, window_count,
deberta_score, deberta_band, calibrated, model_version, latency, error_type
```

No submitted text to third parties; the off-the-shelf checkpoint runs **locally** on the worker
(no external API calls with essay text). This preserves the privacy posture that matters for
student/institutional trust.

---

## 12. Phasing

| Phase | Scope | Exit |
|-------|-------|------|
| **0 — Checkpoint selection** | Research candidates (July 2026); benchmark on SCoCESLE; pick winner; decide raw vs isotonic-recalibrated. | Checkpoint repo-id chosen; ESL FPR recorded. |
| **1 — MVP module** | Units 1–5; kill-switch; inline stage; field on report; additive-invariant test passing. | Field present, additive proven, scan still completes. |
| **2 — Surface** | Report page (frontend) + PDF side-by-side display; agree/disagree note. | User sees both scores. |
| **3 — Future (out of scope)** | Sentence-level localization (map DeBERTa windows onto `highlight_segments`); our own fine-tune; multi-class (plain/paraphrased/hybrid, doc §2). | — |

Phase 0 is mandatory and gates Phase 1.

---

## 13. Acceptance Criteria

MVP (Phase 1+2) is accepted when:

- [ ] Checkpoint chosen via SCoCESLE benchmark; ESL FPR recorded and ≤ composite's documented rate.
- [ ] `ai_signal_deberta` field present on scan report when kill-switch ON.
- [ ] Additive invariant holds: tier/`ai_likelihood`/external/rewrite-decision unchanged ON vs OFF.
- [ ] Kill-switch OFF → field absent, scan path byte-identical.
- [ ] Model loads from `/app/hf_cache` (no runtime network on warm volume).
- [ ] Fail-open verified: model-load failure → `available:false`, scan completes.
- [ ] < 150 words → `available:false` with caveat.
- [ ] Report page + PDF show both scores side-by-side with agree/disagree note.
- [ ] No full essay text in logs.
- [ ] Scan latency documented (~8–14s/1000w accepted).

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Off-the-shelf checkpoint false-accuses ESL | §8 SCoCESLE gate before wiring; isotonic recalibration or advisory-only demotion. |
| Scan latency doubles for all users | Accepted (decision §2). Kill-switch is escape hatch; lazy load keeps cold-start off hot path. |
| Worker OOM (~400MB resident on 4GB) | Verify headroom pre-Phase-1; lazy singleton; ONNX INT8 future option (doc §3). |
| Two scores misread as accusation | Calibrated band-mapped score (not raw prob); disagreement note worded not-to-accuse (§7). |
| Checkpoint goes stale (new model families) | Off-the-shelf burden — someone else re-trains; revisit periodically. Documented limitation. |
| Cold-start stalls first scan after deploy | Pre-download script (unit 1) warms the volume. |

---

## 15. Open Questions (for implementation plan)

- Exact checkpoint repo-id (resolved in Phase 0 benchmark).
- Whether report-page side-by-side is a new tile or folds into the existing AI-likelihood tile (frontend decision, deferred).
- Whether isotonic calibrator is shipped as a bundled artifact or fitted at deploy time.
