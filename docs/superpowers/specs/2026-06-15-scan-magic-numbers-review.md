# Scan Scoring — Magic-Number Review

## ⏭️ REVISIT LATER (marked 2026-06-15)
Status: **reviewed; one change shipped, the rest deferred by decision.**
- **SHIPPED (provisional):** `EXTERNAL_GROUNDING_DISCOUNT_ALPHA = 0.30` (external_grouped_v3, commit `7f096483`). Fit on 32 docs — **a hypothesis, not validated.** Revisit: re-fit/confirm α once the corpus grows; or set `0.0` to disable.
- **DEFERRED (do NOT touch without the prerequisite below):** the hard gates (tier `0.65/0.48/0.32`, cluster floors `0.58/0.60/0.42`, sample caps `0.24/0.32`, external `HIGH_CAP 49.9`) and the composite weights (`AI_WEIGHTS`, `QUALITY_WEIGHTS`, `GROUP_WEIGHTS`, `BUCKET_WEIGHTS`, `ALL_CRITERIA`) + ~25 band step-functions.
- **PREREQUISITE / TRIGGER to revisit:** a **larger, representative labeled corpus** (esp. genuine *student* writing — not just classic essayists vs 2 LLMs). 32 docs cannot responsibly fit ~55 parameters. When that exists → calibrate the hard gates first (maximize honest human-vs-AI separation, NOT Turnitin agreement), then centralize survivors into one versioned `scoring_config`.
- Validation tooling already built: `poc/calibration/measure_detector_signals.py` + `measure_end_to_end.py`.

---


**Date:** 2026-06-15 · **Scope:** the hardcoded numeric thresholds/weights/band-cutoffs that DERIVE the scan's shipped scores + signals. Companion to the content-word de-hardcode (`2026-06-15-scan-hardcode-audit.md`).

## Framing (this is NOT like the content-word lists)
You cannot *delete* these — a scoring function needs parameters. The problem is twofold:
1. **Unjustified:** ~55 of them are **hand-set guesses** with no calibration basis; only ~3 have any data fit (and those are N=1 / single-reference).
2. **Scattered:** spread across ~15 files with no single source of truth; many duplicate the same cutoff.

The fix is **calibrate + centralize**, not remove. The labeled corpus (`poc/calibration/authorship_cases/`) + the harnesses I built (`measure_detector_signals.py`, `measure_end_to_end.py`) are exactly the tools to fit these instead of guessing.

## How the 4 shipped scores are derived (where magic enters)

- **`ai_likelihood`** = weighted avg of 8 mechanical signals (`AI_WEIGHTS`, layer3_scoring.py:1142 — `qualifying_density 0.24 + predictability 0.18 + generic_assertion 0.14 + topk 0.14` ≈ **70%** from 4) → then **topk dampening** (×0.25–1.0 gates) → **cluster boosts + floors** (force score ≥0.58/0.60/0.42) → **sample caps** (≤0.24/0.32 on short text) → **human-provenance ×0.80**. Tier = **0.65→RED / 0.48→ORANGE / 0.32→AMBER** (layer3:1379).
- **`writing_quality`** = weighted avg of 10 grounding/structure signals (`QUALITY_WEIGHTS`, layer3:1298 — `lived_detail 0.18, broad_claim 0.15, citation 0.15…`) − grounding credit (cap 0.15).
- **`external_proxy` (v2)** = 4-group weighted avg (`GROUP_WEIGHTS`: **writing 0.40**, detector 0.25, grounding 0.20, prob 0.15) → 3 hard **caps** (`HIGH_CAP_SCORE 49.9` when `STRONG_GROUNDING_MIN 65` + weak detector coverage, or low coverage).
- **`grounding_diagnosis`** = 4-bucket weighted mean (`BUCKET_WEIGHTS`: **llm_patterning 40**, concrete 29, language 18, authorship 13) → bands 20/40/60/80.

## Classification (reconciled across the 3 audits)

**Genuinely (weakly) CALIBRATED — ~3:**
- `topk_calibration.py:69` piecewise curve — fit to reproduce a GPT-2 reference (raw 67→cal 24.8). Documented.
- `external_grouped_scoring.py` v2 weight shifts (`prob 0.35→0.15`, `overpolish 4.0→1.0`, `para_route/packed_list ↑`) — based on a corpus discrimination pass, but **N=1 Turnitin sample, not third-party validated**.
- `turnitin_like.py` component weights — documented "Turnitin-like" formula, single source, but not validated.

**GUESSED hand-set — ~55** (the review target). Highest-leverage groups:

| group | file:line | what it derives | hard gate? |
|---|---|---|---|
| **Tier thresholds 0.65/0.48/0.32** | layer3:1379 | the RED/ORANGE/AMBER/GREEN the **user sees** | **YES — most consequential** |
| `AI_WEIGHTS` (8 weights) | layer3:1142 | the entire `ai_likelihood` composite | soft but dominant |
| **Cluster floors** 0.58/0.60/0.42 + boosts +0.08–0.12 | layer3:1173–1238 | force a min `ai_likelihood` for "AI-style" clusters | **YES (floor)** |
| **Sample caps** ≤0.24/0.32 + provenance ×0.80 | layer3:1239 | hard ceiling on short/human-provenance text | **YES (ceiling)** |
| `QUALITY_WEIGHTS` (10 weights) | layer3:1298 | `writing_quality` composite | soft |
| `GROUP_WEIGHTS` + `HIGH_CAP_SCORE 49.9` + `STRONG_GROUNDING_MIN 65` | external_grouped_scoring.py:26,71–76 | `external_proxy` + its caps | **YES (cap)** |
| `BUCKET_WEIGHTS` (40/29/18/13) + bands 20/40/60/80 | grounding_diagnosis.py:23,41 | grounding diagnosis + "what to fix" driver | soft |
| `ALL_CRITERIA` weights (12) | criteria/__init__.py:33 | predictability-criteria composite | soft |
| `DEFAULT_WEIGHTS` + concern tiers 0.15/0.25/0.40/0.65 | scoring.py:40,128 | authorship_concern | **YES (tier)** |
| **~25 BAND step-functions** (`ratio≥X→risk Y`) | every estimator (layer3 + criteria) | each per-signal risk value | soft, pervasive |
| Per-criterion scaling consts (0.3, 0.35, 2.0/3.0/2.5, 0.55/0.30…) | criteria/{surprisal,topk_ratio,style_shift,draft_evolution,polish,structural_reuse}.py | each criterion's score | soft |
| Transformation rule thresholds (0.20/0.72/0.55, 0.45/0.60/0.30…) | transformation.py:209–433 | transformation classification label | label |
| Authorship-window labels 0.72/0.42/0.22 | authorship_windows.py:17 | per-paragraph AI labels | label |

## Priorities (if acting)
1. **The hard gates that flip the user-facing verdict** — tier thresholds `0.65/0.48/0.32`, cluster floors `0.58/0.60/0.42`, sample caps `0.24/0.32`, `external HIGH_CAP 49.9`. These decide RED-vs-GREEN; a guessed cutoff here is the highest risk. **Calibrate these first** against the corpus (pick thresholds that maximize AI-vs-human separation + minimize false-positives).
2. **The composite weights** — `AI_WEIGHTS`, `QUALITY_WEIGHTS`, `GROUP_WEIGHTS`, `BUCKET_WEIGHTS`, `ALL_CRITERIA`. Fit by logistic-regression / weight-search on the labeled corpus instead of hand-set.
3. **The ~25 BAND step-functions** — lowest individual leverage, highest count. Either leave (they're monotone, low-risk) or replace step-bands with a smooth monotone transform.
4. **Centralize:** lift the surviving constants into one versioned `scoring_config` module so they're auditable + tunable in one place (no scattered duplicates).

## Honest caveats
- The corpus is **32 docs (classic essayists vs gptoss/qwen)** — enough to sanity-check separation, **not** enough to robustly fit ~55 parameters (overfitting risk). Real calibration needs a larger labeled set (esp. genuine student writing, not just classic essayists).
- Calibrating toward the corpus's Turnitin labels re-raises the settled "don't chase external detectors" question — calibrate toward **honest AI-vs-human separation**, not Turnitin agreement.
- Validation is gated by the same harnesses; every change must hold/improve corpus AUC.
