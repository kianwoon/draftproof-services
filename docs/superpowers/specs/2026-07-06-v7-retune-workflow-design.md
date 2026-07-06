# V7 Re-Tune Workflow — Design Spec

**Date:** 2026-07-06
**Status:** Design (awaiting user review before implementation plan)
**Owner:** wiserly@gmail.com

---

## 1. Problem

V7 (the authorship-clarity + fused deep-scan detection stack) was calibrated during the
2026-07-03/04 launch using **one-off scripts**. As new essays are collated and new AI
generators ship (`gpt-5.6`, `gpt-6`, …), V7 must be **re-tuned** so it keeps separating
content-lacking/AI writing from genuine human writing **without regressing false-accusation
of real ESL writers**.

Today that is not repeatable. The concrete pain, verified on disk:

- **Model IDs are hardcoded across three scripts** — `build_ai_corpus.py:50-52`
  (`gpt-4o-mini`, `claude-haiku-4.5`, `gemini-2.5-flash`), `build_ai_gptoss.py` (gpt-oss),
  `v7_deberta_diversity_check.py` (the gpt-5-mini / qwen run). Adding `gpt-6` means editing
  3 files and re-running 3 scripts by hand.
- **Corpus drift is silent.** `gpt-5-mini` was run in the diversity check but its essays were
  **never persisted** to `authorship_cases/` — only summary numbers survive in
  `v7_deberta_diversity_v2.json`. A re-tune today would silently exclude it. What is measured
  ≠ what the operator believes is measured.
- **Ambiguous data in the corpus dir.** `authorship_cases/` holds 65 AI essays (5 families)
  **and** 15 Gutenberg human-literary passages. The FPR gate filters `authorship == "ai"`
  (`fpr_subgroup_gate.py:81`), so the Gutenberg passages are neither counted as AI nor as
  ESL-human — they are dead/unlabeled weight sitting in an otherwise-AI directory.

This spec turns the one-off scripts into a **versioned, gated, manual-on-demand re-tune
workflow** covering the full lifecycle: **corpus intake → re-calibration → (escalation-only)
GPU checkpoint fine-tune**, with a single fairness gate as the acceptance oracle for both
tuning paths.

## 2. Goals / Non-Goals

**Goals**
- One command to add a new AI-generator family and re-run the full re-tune cycle.
- A versioned corpus manifest so "what is in the corpus" is explicit, diffable, and never drifts silently.
- Re-calibration (no GPU) as the default path; GPU checkpoint fine-tune as a documented escalation.
- **The ESL FPR subgroup gate is the single ship/no-ship oracle for BOTH paths.**
- Reuse maximum existing infra; no data migration.

**Non-Goals**
- Full automated MLOps (DVC, model registry, auto-retrain triggers) — trigger is manual/on-demand (user decision), and the corpus is local-only/non-redistributable so it cannot live in CI.
- Real user-submission harvesting/labeling — excluded (consent + label-noise). Labels come from generation-provenance (AI) and corpus-of-record (human) only.
- Changing V7's runtime scoring behavior. This workflow produces **candidate** artifacts; promotion to production is a separate, gated, human-approved step.

## 3. Current-State Inventory (verified 2026-07-06)

| Asset | Location | Role |
|---|---|---|
| SCoCESLE, 272 ESL human essays (local-only, no redistribution) | `~/Downloads/Small Corpus of Colombian English as a Second Language Essays (SCoCESLE)/` | ESL fairness set (higher/lower proficiency subdirs) |
| 65 AI essays, 5 families | `poc/calibration/authorship_cases/*.json` (`authorship=="ai"`) | AI-positive set |
| 15 Gutenberg human passages | `poc/calibration/authorship_cases/*.json` (non-`ai`) | Currently unused by the gate |
| Committed baselines | `fpr_subgroup_baseline.json`, `v7_deberta_academic_baseline.json`, `v7_fused_gate_result.json` | Regression references |
| Generators (to be unified) | `build_ai_corpus.py`, `build_ai_gptoss.py`, `v7_deberta_diversity_check.py` | AI-set generation |
| The gate (oracle) | `fpr_subgroup_gate.py` — `--compare` exits 1 on FPR rise >3pts / AUC drop ≥0.05 / parity widen >4pts | Acceptance oracle |
| Fused gate | `v7_fused_gate_run.py` → `v7_fused_gate_result.json` | Fusion validation |
| Live checkpoint (Modal) | app `draftproof-v7-deberta-deep-scan`, `desklib/ai-text-detector-academic-v1.01` | Deep-scan serving |

`deberta_fit_calibrator.py` is **excluded** — superseded/dead (its own docstring: "SUPERSEDED
IN PRODUCTION 2026-07 … source of the 0%-bug"). Not called anywhere in the retune chain.

## 4. Architecture — "One gate, two paths"

```
                    ┌─────────────────────────────────────────┐
  Phase 1           │  Phase 2 (default)     Phase 3 (escalate)│
  Corpus Intake ──▶ │  Re-calibrate (no GPU) ──▶ GPU fine-tune │
  & Versioning      │  (thresholds/fusion/       (new DeBERTa   │
                    │   isotonic)                 checkpoint)    │
                    └──────────────┬──────────────────┬─────────┘
                                   ▼                  ▼
                        ┌───────────────────────────────────────┐
                        │  ACCEPTANCE ORACLE (same for both)     │
                        │  fpr_subgroup_gate.py --compare        │
                        │  + fused gate  → PASS/FAIL + deltas     │
                        └──────────────────┬────────────────────┘
                                           ▼
                        RETUNE_LOG.md entry + candidate artifacts
                        (human approves promotion to production)
```

**Design invariant:** no candidate — re-calibrated thresholds *or* a new checkpoint — is
promotable unless it passes the identical ESL FPR subgroup gate vs the committed baseline.
This is what prevents an eager re-tune from trading ESL fairness for AI-recall.

New code lives under `poc/calibration/retune/`. Nothing outside that dir + the manifest is
modified by a re-tune run until the explicit promotion step.

## 5. Phase 1 — Corpus Intake & Versioning

### 5.1 `models.json` (single source of truth for generators)
Replaces the hardcoded lists in the three scripts. Example:
```json
{
  "version": "2026-07-06",
  "generators": [
    {"id": "openai/gpt-4o-mini",        "provider": "openrouter", "family": "gpt-4",   "n_per_topic": 1},
    {"id": "google/gemini-2.5-flash",   "provider": "openrouter", "family": "gemini",  "n_per_topic": 1},
    {"id": "anthropic/claude-haiku-4.5","provider": "openrouter", "family": "claude",  "n_per_topic": 1},
    {"id": "qwen/qwen-2.5-7b-instruct", "provider": "openrouter", "family": "qwen",    "n_per_topic": 1},
    {"id": "openai/gpt-5-mini",         "provider": "openrouter", "family": "gpt-5",   "n_per_topic": 1}
  ]
}
```
Adding `gpt-6` = one row. Provider fan-out reuses the working OpenRouter path (Cerebras
403s on urllib default UA — known gotcha, carried over).

### 5.2 `corpus/manifest.json` (versioned corpus of record)
One row per essay: `{id, source_path, label: human|ai, family, model_id, license: redistributable|local_only, split: cal|test|holdout, sha256, added_utc}`. Built by scanning:
- SCoCESLE proficiency subdirs → `label=human, license=local_only`
- `authorship_cases/*.json` → `label` from the `authorship` field (fixes the Gutenberg ambiguity by recording it explicitly, not by guessing)

### 5.3 Intake CLI — `retune/intake.py`
- `--generate` reads `models.json`, calls the unified generator, **persists every essay to disk** (closes the gpt-5-mini silent-drift gap), refreshes the manifest.
- `--rebuild-manifest` re-scans existing dirs without generating.
- **Leakage guard:** the same `sha256` may not appear in both `cal` and `test` splits; run aborts on collision.
- **License guard:** rows marked `local_only` are refused from any artifact that would be committed to the repo (SCoCESLE text never enters git — carries the existing rule forward mechanically).

**Provenance model (recommended, per user's "recommend it" answer):**
- AI label = generation provenance (we generated it from a named model). Authoritative.
- Human label = corpus-of-record (SCoCESLE consented-human).
- Third-party labeled datasets (RAID/HC3/etc.), if later added: land in the **`holdout` split only** — used for a secondary domain-shift AUC read, **never** in the ESL FPR gate (their non-ESL distribution would corrupt the fairness measurement, which is the load-bearing metric).

## 6. Phase 2 — Re-calibration (no GPU), gated

The default re-tune path. Reuses existing tuning scripts, orchestrated in order:

1. **Deep-scan sweep** — re-sweep `sent_threshold` / `doc_floor` against the new AI families (`v7_deberta_academic_calibrate.py` path) → candidate `academic.json`.
2. **Fused gate** — `v7_fused_gate_run.py` → candidate `fused.json` (TPR/AUC/parity of `0.4×composite + 0.6×proportion`).
3. **Tier-cutoff check** — confirm the `32/48/65` fused cutoffs in `weights.json tier_authority` still hold on the new distribution; propose a re-sweep only if TPR/FPR at the operating point moved materially.

The isotonic re-fit step (`deberta_fit_calibrator.py`) is **excluded** — dead/superseded, see
§3. `poc.calibration.retune.run_cycle --generate --paid` runs steps 1–2 for you, writing both
candidates to `--staging` (never a committed path).

**Acceptance:** run `fpr_subgroup_gate.py --compare` against the committed
`fpr_subgroup_baseline.json`. Candidate artifacts are written to a staging dir and are
promotable **only on PASS**. On FAIL, the run stops and the `RETUNE_LOG.md` records which
tolerance broke (FPR rise / AUC drop / parity widen).

## 7. Phase 3 — GPU Checkpoint Fine-Tune (Modal), escalation-only

**Trigger:** Phase 2 exhausts headroom — e.g. a new model (`gpt-6`) drives fused TPR below
the operating bar and no threshold/fusion re-sweep recovers it while the FPR gate still
passes. Re-calibration is *always tried first* (July precedent: re-fit alone recovered TPR
10%→85% with no training).

**Steps:**
1. Train a new DeBERTa checkpoint on the `cal`/train split on **Modal** (same platform as the
   live deep-scan app `draftproof-v7-deberta-deep-scan`; secrets already provisioned).
   Output = a versioned checkpoint artifact `v7-deberta-YYYY-MM-DD`.
2. Score the `test` split + SCoCESLE through the **candidate** checkpoint.
3. **Same oracle:** `fpr_subgroup_gate.py --compare` + fused gate must PASS before the
   checkpoint may replace the live one.
4. Promotion = deploy candidate to a Modal *staging* endpoint, re-verify, then flip the live
   app's checkpoint. Kill-switch/fail-open to the current checkpoint preserved.

Phase 3 is specced but **not built in the first implementation** unless the user asks — it is
the documented escalation contract, not day-one work.

## 8. The Orchestrator & Runbook (deliverable "the workflow")

- **`retune/run_cycle.py`** — thin orchestrator that runs Phase 1 → Phase 2, prints the gate
  verdict + deltas, and appends a `RETUNE_LOG.md` entry. Optionally exposed as a Claude Code
  `Workflow()` script for the AI-set fan-out, but all heavy compute (the ~13-min gate, any
  GPU) stays in the Phase CLIs, not inside subagents.
- **`docs/runbooks/v7-retune.md`** — the human runbook: "new model dropped → add row to
  `models.json` → `intake --generate` → `run_cycle` → read gate verdict → promote or stop."
- **`RETUNE_LOG.md`** — append-only decision log: corpus version, models added, gate deltas,
  ship/no-ship, who approved.

## 9. Versioning

Corpus `vN` (manifest), calibration `vN` (baseline JSONs + isotonic pkl), checkpoint `vN`
(Modal artifact). Each re-tune stamps all three into one `RETUNE_LOG.md` row so any
production V7 behavior is traceable to an exact corpus+calibration+checkpoint triple.

## 10. Testing

- **Manifest unit tests:** label assignment, license guard refuses `local_only` into committed
  artifacts, leakage guard catches a duplicated sha across splits.
- **Intake integration (mocked LLM):** `--generate` with a stub provider persists files +
  updates manifest; re-run is idempotent.
- **Gate-as-oracle test:** a deliberately-worse candidate (inflated ESL FPR) must FAIL
  `--compare` (exit 1); an unchanged candidate must PASS.
- **No-op invariant:** running the cycle with zero new data reproduces the committed baseline
  numbers (determinism check).

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Re-tune trades ESL fairness for AI-recall | Single FPR gate oracle; no promotion on regression |
| Silent corpus drift (the gpt-5-mini gap) | Manifest is the source of truth; intake persists every essay |
| SCoCESLE text leaking into git | License guard refuses `local_only` from committed artifacts |
| Small AI-set → noisy AUC (July: AUC swung 0.97↔0.75 with the AI set) | Manifest tracks per-family counts; runbook warns when a family is under-sampled |
| GPU fine-tune overfits to a single new model | Test split held out; gate uses the diverse multi-family set |
| Provider API drift (Cerebras UA 403, id 404s) | Unified generator carries the known gotchas; `models.json` ids validated against live `/models` |

## 12. Decisions (resolved 2026-07-06)

1. **First implementation scope: Phase 1 + Phase 2 + orchestrator/runbook.** Phase 3 (Modal
   GPU fine-tune) is **specced-only** — built later, only if re-calibration fails the bar.
2. **Orchestrator form: plain CLI** (`run_cycle.py`) to start. No Claude Code `Workflow()`
   wrapper day one; may add later if the cadence warrants it.
