# V8: ai_paraphrased Category Recovery — Design

**Date:** 2026-07-08 · **Owner-approved** (this session) · **Status:** design approved, plan pending

## Problem

`ai_paraphrased` primary accuracy is 0% in the V7 authorship breakdown (post specificity-split state: macro 65.5%, student_owned 97.5% / false-AI 2.5%, polished 87.2%, gen-like 77.5% — `poc/calibration/v12_validation/` fused baselines). The class collapses into `ai_generated_like` (17/39) and `ai_assisted_polished` (17/39). The MiniLM cosine feature family (semantic monotony) was measured and **gate-rejected** (best effective AUC 0.669 < 0.70; `paraphrase_feature_study.json`) — it and its variants are banned.

## Hard constraint (owner)

Users upload **one copy** of the content. No comparison-document workflow. Single-document evidence only.

## Theory of the signature

Corpus construction defines the class: an LLM fully rewriting a *human* (ESL) essay. Single-doc signature is a **mismatch**: human-inherited content (specific, idiosyncratic, unevenly structured) wearing an AI surface (smooth, detector-high).

- `ai_generated_like` = AI surface + generic content
- `ai_assisted_polished` = human surface (mostly) + human content
- `ai_paraphrased` = **AI surface + human content** ← the mismatch to measure
- `student_owned` = human surface + human content, detector-low

Precedent: the detector-gated specificity split (2026-07-08) proved interaction features can carve classes that raw signals and weight-tuning cannot.

## Acceptance criteria (owner-set)

E2e fused §12 measurement under promoted weights:
- `ai_paraphrased` primary accuracy **≥ 40%**
- `student_owned` false-AI **≤ 10%** (today 2.5% — must not materially regress)
- `ai_generated_like` **≥ 70%**, `ai_assisted_polished` **≥ 65%**
- macro **≥ ~65%** (no regression beyond noise vs today's 65.5%)

**Fallback if no family passes its study gate** (same no-threshold-torturing discipline): `breakdown_composer` merges paraphrased into an honest "AI-transformed" display band (annotation, weights untouched), and this spec gains a second measured-rejection record.

## Phased feature studies — gate BEFORE any production wiring

Study gate (each phase): best feature **effective AUC ≥ 0.70** for `ai_paraphrased` vs `ai_generated_like` (direction-aware, per the fixed feature-study methodology); AUC vs `ai_assisted_polished` reported (both collapse routes matter); **ESL-direction check** on higher/lower-proficiency student_owned subgroups — mandatory and stricter here because the corpus paraphrases were generated FROM ESL essays (a "human content" proxy multiplied by detector score must not become an ESL detector).

### Phase A — interaction/mismatch family (first; free, offline)

Fresh capture (cache-warm Modal, all 16 signals) → offline study over products of:
- content-humanness proxies: `specificity_score`, `specificity_student_evidence`, `1 − author_voice_absence`, `1 − grounding_gap`
- surface-AIness proxies: `sentence_smoothness`, `calibrated_detector_score`

Candidate features are pairwise products (and the top candidates' complements), computed from capture rows — no new detection code, no corpus re-scan.

### Phase B — structural raggedness (only if A fails its gate)

Standalone study script over corpus text (like the MiniLM study but **lexical/structural only — NO embedding features**, respecting the ban's spirit): paragraph-length CV, discourse-marker irregularity, connective-pattern raggedness — each alone and crossed with smoothness/detector. Same gate.

### Phase C — function/content-word perplexity asymmetry (only if B fails)

Requires per-token predictability outputs from `poc/detect/` → ESL FPR pre-push gate territory. **Scoped as its own owner decision point before any work starts.** Not designed further here.

## Wiring & tuning (whichever family passes)

Exactly the specificity-split pattern:
1. Derived signal(s) in `poc/detect_v7/signal_adapter.py` (16 → N; status ok/unavailable; clamp01), inputs threaded via `raw_signals` (detector score already threaded).
2. `ai_paraphrased_without_comparison` formula updated in `weights.json` (the `with_comparison` variant stays for a future comparison workflow).
3. Tuner (`tune_weights.py`) extended with a **third hard constraint**: `--para-floor` (default 0.40, CLI flag, baseline-anchored like the others; enforced on tune AND holdout).
4. Re-capture → re-tune → holdout generalization → tier-invariance byte-check → `test_no_text_leakage` → e2e fused measurement vs ALL acceptance criteria → promote with `_notes` provenance.
5. UI: no changes needed (category already rendered; tier-consistency guard and framing labels remain in force).

## Validation infrastructure (exists, reuse)

`poc/calibration/v12_validation/`: corpus (198 rows, rebuild via `build.py` if lost — it is worktree-fragile), `capture_signals.py` (offline==e2e contract test), `tune_weights.py`, `measure.py --fused` (cache-backed), `test_no_text_leakage.py`, `baseline_lock.json`. Deep-scan cache warm (`calibration/retune/cache/`) — all V8 runs expected $0.

## Orchestration map

| Work | Runner |
|---|---|
| Phase A capture + study script | Sonnet subagent (study runs: main session) |
| Tuner `--para-floor` extension | Sonnet subagent |
| Phase B study script (if reached) | Sonnet subagent |
| Production wiring (signal_adapter/pipeline_bridge/weights) | Opus subagent |
| Reviews | Sonnet for scripts, Opus for detect_v7 changes |
| Measurements, gates, promotion | Main session (orchestrator) |

## Risks

| Risk | Mitigation |
|---|---|
| Interaction features act as covert ESL detectors | Mandatory ESL-direction subgroup check in every study gate; false-AI ≤10% hard constraint in tuner |
| Para floor makes tuner infeasible (NO_CANDIDATE) | Report honestly; drop to fallback decision with owner, never relax other floors silently |
| Corpus is a single generator-rotation construction | Same caveat as all §12 work (construction labels); reviewer labels remain the open D5 step |
| Overfitting 39 paraphrased docs | 70/30 stratified holdout enforced on ALL constraints incl. para floor |
