# Authenticity Dashboard — Design Spec (v1)

Date: 2026-06-29
Status: design for review (no code yet)

## Goal

Replace the single, easily-misread "AI %" headline with a multi-dimensional **Authenticity
Dashboard** that leads with *ownership and grounding* and demotes AI-likelihood to **one
banded dimension with a confidence interval**. This dissolves the relief/panic problem and
aligns the report with the product's grounding-coach positioning — without pretending to
detect AI better than we can (the AUC ceiling is ~0.75 and unchanged by this work).

Target layout (from the mockup):

```
Learning Ownership   92        Reasoning Consistency  88
Grounding            81        Citation Quality       75
Revision Evidence    95        AI Assistance   Moderate (± CI)
                       Overall Risk: Low
```

## Scope decision (approved)

**v1 = honest reuse now; enhance in phase 2.** Ship the dimensions that are genuinely real
today, label the proxies honestly, defer what needs new signal. No false precision.

## Verified mapping (anchored to code; from the 6-agent mapping pass)

| Tile | v1 source | Transform (100 = most authentic) | Status |
|---|---|---|---|
| **Grounding** | `ai_risk_badge.grounding_diagnosis.buckets.concrete_grounding` | `100 − score`; **null when `available==0`**; carry `low_coverage` caveat | ✅ real |
| **Citation Quality** | `ai_risk_badge.writing_components.{citation_weakness_risk, source_grounding_risk}` | `100 − mean(...)`, drop a `None` component; **low-confidence when no bibliography** | ✅ real (caveated) |
| **AI Assistance — band** | `ai_likelihood_score` + `_derive_ai_tier` cuts | GREEN→Low / AMBER→Moderate / ORANGE+RED→High; can never contradict the headline tier | ✅ real |
| **Learning Ownership** | `ai_risk_badge.critical_thinking_control.score` | use directly (already 0-100); **null when CT abstains** | ✅ real, but derivative (see Overlap) |
| **AI Assistance — CI** | `predictability.all_sentences[].predictability_risk` spread | proxy interval, widened by categorical `confidence`; **labeled "tentative"** | 🟡 proxy |
| **Reasoning Consistency** | — | **deferred to a phase-2 placeholder** (see Decision R) | 🔴 no real signal |
| **Revision Evidence** | — | permanent placeholder ("requires revision history") | 🔴 needs provenance |
| **Overall Risk** | the tiles above | new composite (see below) | 🟡 must build |

## Two honesty constraints baked into the design

1. **Dimensions are NOT independent.** Chain: `grounding_diagnosis → critical_thinking →
   ownership`. Learning Ownership *is* the CT score; Grounding feeds CT. So the Overall
   composite must **not** double-count: it weights the more-independent axes (Grounding,
   Citation, AI-Assistance) and **down-weights Learning Ownership** (derivative of Grounding/CT).
   UI copy must not present the tiles as independent measurements.
2. **No false precision.** The CI ships labeled *"tentative / directional — not a statistical
   interval."* Reasoning Consistency is deferred rather than shown as a misleading proxy.

## Overall Risk composite

`Overall = min( weighted_mean_over_available(dims), worst_available_dim )` — a weakest-link
floor (mirrors `submission_risk.py:289-294`) so one failing axis can't be averaged into a
"Low risk" headline. Bands on the 100=authentic axis: **≥66 Low · 38–66 Medium · <38 High.**
Default weights (overlap-aware, **flagged for review**): Grounding 0.30, Citation 0.25,
AI-Assistance 0.30, Learning Ownership 0.15. Abstains if fewer than 2 dimensions have data.

## Architecture

A new **additive read-time composer**, mirroring `submission_risk.py` / the `app/_composers/`
read-time pattern. **Additive invariant: it never feeds back into the tier, `ai_likelihood_score`,
external estimate, or any gate** — it only adds `ai_risk_badge['authenticity_dashboard']`.

**Input:** the **full `report.json`** (not badge-only) — the CI source (`predictability.all_sentences`)
and any coherence signal are sibling top-level keys, unreachable from the badge alone.

**Output:** `ai_risk_badge.authenticity_dashboard = { learning_ownership, grounding,
citation_quality, reasoning_consistency, ai_assistance:{band, score, ci:{low, high, tentative:true}},
overall:{score, band}, revision_evidence:{status:"placeholder"} }` — each tile
`{score: 0-100|null, available: bool, caveat}`.

**Files (create):** `poc/detect/authenticity_dashboard.py` (pure function, `submission_risk.py`
shape: `MODEL_VERSION`, tunable `WEIGHTS`, renormalize-over-available) + `draftproof-api/app/_composers/authenticity_dashboard.py` (verbatim "KEEP IN SYNC" copy).

**Files (touch):** `poc/report/report.py` (~1553, attach scan-time, gated) · `draftproof-api/app/services/report_service.py` (~221, read-time backfill with full json — the only path with the CI source) · `draftproof-api/app/policy_enrich.py` + `routes/ext.py:147` (badge-only backfill → degraded categorical-confidence CI, documented) · `draftproof-frontend/src/pages/report/AuthenticityDashboard.jsx` (mirror `SubmissionRiskBand.jsx`) wired into `Report.jsx` + i18n `report.authenticityDashboard.*` (en + zh) · `poc/report/render_panels.py` + `render.py` (PDF panel).

**Kill-switch:** `DRAFTPROOF_AUTHENTICITY_DASHBOARD`, **default OFF**, checked at every attach site.

## Data flow
detect run builds `ai_risk_badge` + `predictability` + `scan_intelligence` (siblings) → R2 →
`report_service.get_report` loads full `results_json` → composer reads badge + `predictability.all_sentences`
→ emits dashboard, attaches additively → React/PDF render. Scan-time attach gets the rich proxy CI;
read-time backfill of old reports via `report_service` also gets it; rewrite/ext badge-only paths get the degraded CI.

## No-data / abstention UX
When CT abstains (<5 scored sentences) Learning Ownership → "not enough text to assess".
Grounding null when `available==0`. Overall computes over survivors; abstains if <2 remain.
Tiles never silently show 100 on no-data.

## Validation
- **Additive ⇒ the SCoCESLE FPR gate is unaffected** (no change to `ai_likelihood`/tier). Confirm the gate still passes; note the pre-push hook will run it because the composer lives under `poc/detect/` (it will pass — provably additive; `--no-verify` acceptable for the additive-only commit).
- Unit tests for the composer: each transform + polarity, the `available==0` / `None` guards, the weakest-link floor, abstention, and band cuts.

## Phase 2 (out of scope for v1)
- **Real Reasoning Consistency**: per-paragraph claim→conclusion entailment / sign-flip check, OR enable + variance-validate the existing LLM dims (`critical_thinking_llm.py`, default OFF).
- **True CI**: per-paragraph composite via `Layer3Scorer.calculate_ai_likelihood` (pure-Python, no LLM) for a real dispersion of the band metric.
  - **VERIFIED DEFERRED (2026-06-29, adversarially traced end-to-end — NOT cheaply/additively shippable as written):**
    1. `calculate_ai_likelihood` runs **once, document-level** (`layer3_scoring.py:1431` ← `report.py:1471`). `build_layer3_input_from_text` takes `predictability`/`topk_pattern` as **parameters** (it does not compute them — it only runs cheap text-statistic `estimate_*` calls). So a per-paragraph call either **omits the dominant signals** (predictability/topk — the ~71 AI floor), or must **recompute the GPT-2 model per paragraph** (~N× the ~4s scan, and statistically noisy on short paragraphs).
    2. `report.py:466` stamps the **same doc-level** `ai_generation` `result.likelihood_score` onto every finding; `report.py:916`'s `max(...)` is therefore **not discarding a distribution — there is no per-paragraph distribution to recover.**
    3. The dominant signals are **already composited per-sentence** in `predictability.all_sentences[]` (`predictability_risk` = `0.45·top10 + 0.25·top50 + 0.20·(1/(1+surprisal)) + 0.10·generic`, per `report.py:4628`), at **finer-than-paragraph granularity, for free** — and the shipped `_ai_ci` already dispersions that composite.
    - **Conclusion:** a literal per-paragraph composite is **inferior** to the shipped proxy (weaker if cheap, expensive+noisy if faithful); no richer per-unit signal exists without recompute. The only honest "True CI" is a perf+noise project (per-unit GPT-2 recompute) that the detector's ~0.75 AUC ceiling does not justify. **Keep the tentative proxy; do not relabel it "True CI."**
    - Open micro-fix (separate concern, not Phase 2 #2): the `<2-sentence` fallback invents a `_CI_DEFAULT_SPREAD = 12.0` magic-constant half-spread (false precision on short docs) → should **abstain** (`ci: null`, "not enough text to estimate a range") to match the dashboard's abstention UX and drop a hardcoded number.
- **Revision Evidence**: provenance capture (draft snapshots / paste / time) in the add-ins.

## Decisions for your review (defaults chosen; change any)
- **R — Reasoning Consistency**: *deferred to a phase-2 placeholder* (vs. showing `reasoning_trail.control` as a clearly-labeled "structural, not logical" proxy). Chosen defer to avoid both false precision and worsening the overlap illusion.
- **Overall weights** 0.30/0.25/0.30/0.15 — placeholder; tune.
- **Band scales**: AI-Assistance band is on the AI-likelihood axis (cuts 32/48); Overall band is on the composite-authenticity axis (66/38). Intentionally different metrics — UI copy must label them so the two scales don't read as contradictory.
- **CI**: ship the tentative proxy now vs. hold until the phase-2 true CI. Default: ship, clearly labeled.
