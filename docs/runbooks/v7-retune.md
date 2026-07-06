# Runbook — V7 Re-Tune (manual, on-demand)

**When:** a new AI model ships (gpt-5.6, gpt-6…) or enough new essays accumulate.

## Add a new model
1. Add one row to `poc/calibration/retune/models.json` (`id`, `provider`, `family`, `n_per_topic`).
   Validate the id against OpenRouter's live `/models` list first (guessed ids 404).

## Run the cycle
2. From the repo root (NOT `cd poc`), run:
   ```bash
   ~/.pyenv/versions/3.11.0/bin/python3 -m poc.calibration.retune.run_cycle --generate
   ```
   - **Important:** The `-m poc.calibration.retune.*` commands MUST be run from the repo/worktree root. There is no `poc/__init__.py`, so `cd poc` breaks module resolution.
   - Generates + PERSISTS AI essays, rebuilds the manifest, runs the FPR gate.
   - `.env` must have `OPENROUTER_API_KEY` (lives in the MAIN repo root, not the worktree).

3. Read the verdict line + the appended row in `poc/calibration/retune/RETUNE_LOG.md`:
   - **PASS** — the new corpus does not regress ESL false-accusation (FPR rise ≤3pts, AUC drop ≤0.05, parity widen ≤4pts vs the committed baseline). Candidate artifacts are safe to promote.
   - **FAIL** — a tolerance broke. Do NOT promote. Inspect the gate stdout for which one.
   - **NO-CORPUS** — SCoCESLE dir not found; set `--scocesle PATH`.

## Re-calibrate (Phase 2 detail) — costs real money (Modal)
4. If discrimination dropped (new model evades), run the SAME command with `--paid`. This
   re-tunes on the FULL chain (FREE FPR gate, then the two Modal-cost scripts) and hands
   back CANDIDATE artifacts gated on ESL fairness — nothing is written to committed paths:
   ```bash
   ~/.pyenv/versions/3.11.0/bin/python3 -m poc.calibration.retune.run_cycle --generate --paid
   ```
   - `--paid` is required to spend anything — without it, run_cycle stays free (intake + FPR
     gate only, current default behavior).
   - `--staging DIR` (default `poc/calibration/retune/staging/`, gitignored) is the BASE dir.
     The orchestrator now isolates each `--paid` run automatically: it creates a per-run
     subdirectory `run-<sanitized-timestamp>` (derived from the cycle's UTC `now_iso`, with
     `:`, `+`, `.` replaced by `-`) under `--staging`, and that subdirectory — not the base
     dir — is where `academic.json`, `fused.json`, and `fused_progress.jsonl` actually land.
     This means two back-to-back `--paid` runs never share a `fused_progress.jsonl`, so a
     changed `sent_threshold` or fusion weights can no longer silently mix old- and new-weight
     scores. To resume a crashed run, point the fused-gate script's `--progress`/`--out` at
     that specific `poc/calibration/retune/staging/run-<timestamp>/` subdirectory rather than
     the base `--staging` dir.
   - `--weights PATH` scores a candidate `weights.json` instead of production's.
   - `--limit N` passes through as `--limit-per-group N` for a cheap smoke run.
   - Read the `calibration` column appended to `RETUNE_LOG.md` for the fused-gate verdict.
   - `deberta_fit_calibrator.py` is EXCLUDED from this chain — it is dead/superseded (its own
     docstring: "SUPERSEDED IN PRODUCTION 2026-07 … source of the 0%-bug"). Do not call it.
   - **Deep-scan cache** — essays scored once by the paid Modal endpoint are cached by
     content-hash (sha256 of checkpoint + text) in the PERSISTENT, cross-run
     `poc/calibration/retune/cache/deepscan_scores.jsonl` (gitignored; no essay text stored,
     only the hash key and raw per-sentence scores) and are never re-paid in a later run. The
     cache stores RAW scores and derives the proportion locally at whatever `sent_threshold` is
     configured, so it is threshold-INDEPENDENT — changing `sent_threshold` or fusion weights
     does NOT invalidate it, unlike the per-run `fused_progress.jsonl`. Only a checkpoint change
     (`DRAFTPROOF_MODAL_CHECKPOINT`, e.g. the Phase 3 fine-tune) invalidates a row, since the
     checkpoint is part of the cache key. Override the default with `--cache PATH` on
     `run_cycle.py` or `v7_fused_gate_run.py` directly if you need an isolated cache for
     experimentation.

## Promote (only on PASS)
5. Re-baseline intentionally. From within `poc/`, run:
   ```bash
   cd poc
   python calibration/fpr_subgroup_gate.py --out calibration/fpr_subgroup_baseline.json
   ```
   Commit the numbers-only baseline (never the SCoCESLE text — the manifest's `local_only` license guard blocks that by construction).

## Phase 3 (GPU checkpoint fine-tune) — NOT built yet
Escalation only, when re-calibration can't recover discrimination. Specced in `docs/superpowers/specs/2026-07-06-v7-retune-workflow-design.md` §7 (Modal, same gate oracle).
