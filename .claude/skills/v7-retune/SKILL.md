---
name: v7-retune
description: >-
  Run the V7 detector fine-tune / re-tune cycle in DraftProof. Use this whenever the user wants to
  re-tune, fine-tune, re-calibrate, or refresh V7 because NEW ESSAYS were collated or a NEW MODEL
  shipped (gpt-6, gpt-7, a new frontier generator, etc.) — even if they just say "fine-tune V7",
  "re-run the V7 calibration", "we have new essays, update the detector", "add gpt-6 and re-tune",
  "recalibrate the ESL gate", or "check if V7 still holds on the new data". This skill knows the exact
  command, the free-vs-paid (Modal $) distinction, where to put new essays/models, the deep-scan cache,
  how to read the verdict, and how to promote candidates. Prefer this skill over improvising the
  calibration scripts by hand — running them out of order or against the wrong paths wastes Modal money
  or produces wrong verdicts.
---

# V7 Re-Tune (fine-tune the detector on new essays / new models)

## What this does and why it exists

V7 is DraftProof's authorship/AI detector. It was **created** by a calibration sequence (generate an AI
corpus from diverse models → sweep the deep-scan thresholds → fused ESL gate → ESL false-positive check).
This skill re-runs **that same sequence** on an expanded corpus so V7 keeps separating AI/content-lacking
writing from genuine human writing **without newly false-accusing real ESL writers** — the load-bearing
fairness constraint the whole product depends on.

Two triggers, **one command**:
1. **New essays** collated → re-tune.
2. **New model** ships (gpt-6, gpt-7…) → re-tune.

It produces **candidate** artifacts in a staging dir. It never changes production automatically —
promotion is a separate human step.

## Golden path (do this)

Always run from the **repo root** (there is no `poc/__init__.py`; `cd poc` breaks `-m` module resolution).
Use the project's Python: `~/.pyenv/versions/3.11.0/bin/python3` (has the ML stack).

```bash
# 1. FREE — costs nothing. Generates+persists new AI essays, rebuilds the corpus manifest,
#    and runs the local ESL fairness gate. Run this FIRST to see whether a re-tune is even needed.
~/.pyenv/versions/3.11.0/bin/python3 -m poc.calibration.retune.run_cycle --generate

# 2. PAID — real Modal $ (~cents). Adds the actual re-calibration: deep-scan threshold sweep +
#    fused ESL gate. Do this when you actually want to re-tune (new model, or the free gate shows drift).
~/.pyenv/versions/3.11.0/bin/python3 -m poc.calibration.retune.run_cycle --generate --paid

# Cheap smoke of the paid path (small subset, tiny cost) before the full run:
~/.pyenv/versions/3.11.0/bin/python3 -m poc.calibration.retune.run_cycle --generate --paid --limit 12
```

`--generate` reads `poc/calibration/retune/models.json`, generates + **persists** essays to
`poc/calibration/authorship_cases/`, and rebuilds `poc/calibration/retune/corpus/manifest.json`.
Omit `--generate` to re-run the gates on the existing corpus without adding essays.

## Adding new inputs

**New model (e.g. gpt-6):** add ONE row to `poc/calibration/retune/models.json` (do NOT hardcode model
IDs anywhere else — that file is the single source of truth), then run the golden path with `--generate`.
Validate the model id against OpenRouter's live `/models` list first; guessed ids 404.
```json
{"id": "openai/gpt-6", "provider": "openrouter", "family": "gpt-6", "n_per_topic": 1}
```

**New human ESL essays:** drop `.txt` files into the SCoCESLE proficiency subfolders
(`~/Downloads/Small Corpus of Colombian English as a Second Language Essays (SCoCESLE)/SCoCESLE txt higher proficiency `
or `… lower proficiency`). The gate auto-discovers them — no config change. The plain `SCoCESLE txt`
folder is deliberately ignored (avoids double-counting). Point elsewhere with `--scocesle PATH` only if the
corpus lives somewhere else; that folder must keep the `*proficiency*` subdir structure.

## Reading the result

The run prints `=== V7 RE-TUNE: <verdict> ===` and appends a row to
`poc/calibration/retune/RETUNE_LOG.md` (columns: version, n_rows, families, gate, auc_line, calibration).

- **PASS** — the expanded corpus does NOT regress ESL false-accusation (FPR rise ≤3pts, AUC drop ≤0.05,
  parity widen ≤4pts vs the committed baseline). Candidate artifacts are safe to consider for promotion.
- **FAIL** — a tolerance broke. **Do NOT promote.** Inspect which one in the gate stdout.
- **NO-CORPUS** — SCoCESLE dir not found; pass `--scocesle PATH`.
- The `calibration` column (paid runs) carries the fused-gate verdict (`gate_pass` true/false); `skipped` on free runs.

## Cost safety (respect this — Modal calls are real money)

- Default (no `--paid`) is **free**: intake + local FPR gate only. Recommend running it first.
- `--paid` is required for ANY Modal call. Use `--limit N` for a cheap smoke before a full paid run.
- The **deep-scan cache** (`poc/calibration/retune/cache/deepscan_scores.jsonl`) skips re-paying Modal for
  any essay (human or AI) scored before. It caches raw per-sentence scores keyed by content-hash+checkpoint
  and derives the proportion locally, so it stays valid across threshold changes. You don't manage it — it
  just makes repeat paid runs cheap. **One caveat:** if a NEW deep-scan checkpoint is ever deployed (a
  future GPU fine-tune), bump `DRAFTPROOF_MODAL_CHECKPOINT` in the root `.env` (or clear the cache dir),
  else the cache serves stale scores under the new checkpoint = wrong verdicts.

## Safety invariants (never violate)

- **Candidates only.** A run writes to a per-run staging dir `poc/calibration/retune/staging/run-<ts>/` and
  the persistent cache. It never overwrites committed baselines
  (`v7_deberta_academic_baseline.json`, `v7_fused_gate_result.json`, `fpr_subgroup_baseline.json`) or
  `poc/detect_v7/weights.json`. Promotion is a deliberate, separate human step.
- **SCoCESLE is local-only** — never commit its essay text or let it into any committed artifact.
- **`deberta_fit_calibrator.py` is DEAD/superseded** (was the source of a production 0%-bug). Never call it.
- Re-calibration is always tried first; a GPU checkpoint fine-tune (Phase 3) is an escalation, not built yet.

## Promotion (only on PASS, deliberate)

Candidates live in staging. To make an operating point authoritative, a human re-baselines intentionally:
```bash
cd poc && ~/.pyenv/versions/3.11.0/bin/python3 calibration/fpr_subgroup_gate.py --out calibration/fpr_subgroup_baseline.json
```
Commit the **numbers-only** baseline (never the SCoCESLE text). Then update `poc/detect_v7/weights.json`
cutoffs from the reviewed candidate if the sweep moved them. Confirm with the user before promoting.

## Environment notes

- `.env` (with `OPENROUTER_API_KEY`, `DRAFTPROOF_MODAL_ENDPOINT_URL/TOKEN`) lives at the **main repo root**,
  not inside a git worktree — the scripts walk up parent dirs to find it.
- The paid steps need the live Modal deep-scan app running.

## Deeper reference (read if you need internals or hit an error)

- `docs/runbooks/v7-retune.md` — the operator runbook (this skill is its executable summary).
- `docs/superpowers/specs/2026-07-06-v7-retune-workflow-design.md` — full design, the "one gate, two paths"
  invariant, and the Phase-3 GPU-fine-tune escalation contract.
- `poc/calibration/retune/` — the code: `run_cycle.py` (orchestrator), `recalibrate.py` (paid chain),
  `gate.py` (FPR oracle), `deepscan_cache.py`, `intake.py`, `manifest.py`, `models.json`.
