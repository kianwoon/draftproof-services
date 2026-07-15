# Phase-1 M4 — Claim-Graph §B Calibration Report (MEASURED, NOT PROMOTED)

**Status:** Measurement report. **Date:** 2026-07-15.
**Governing specs:** `docs/plans/phase1_claim_graph_execution_plan.md` (M4 row, §4 calibration plans),
`docs/plans/credible_authorship_assessment_v2.md` §B (five-part protocol) + §A headline invariant.
**Nature:** Every Phase-1 signal stays **EXPERIMENTAL** (`scoring_enabled:false`, `fusion_weight=0`).
This report produces **RECOMMENDED-NOT-PROMOTED** threshold observations; promotion is an owner
decision in M5. **No weights are invented here** (§B4) — every number below is measured from the
committed harness over the committed/proxy-labeled corpus.

## Harness + corpus

- **Harness:** `poc/claim_graph/eval/run_eval.py` — per doc: canonical
  `structured_sentence_segments` → M2 extraction (real Cerebras `gpt-oss-120b` gateway, in-process
  cache) → validated graph → M3 signals → row. Resume-safe JSONL (`eval_rows.jsonl`, gitignored).
- **Manifest:** `poc/claim_graph/eval/corpus_manifest.json` (built by `build_corpus_manifest.py`).

| Group | n | Label | Source |
|---|---|---|---|
| **H** | 25 | human-student, presumed-experiential | `realbenpope/PERSUADE_manageable`, first 25 essays ≥120 words (CSV order); texts local-only, referenced by `essay_id`+`sha256` |
| **A** | 20 | AI-generic | first 20 `ai_gpt_5_6_user_NN` cases in `poc/calibration/authorship_cases` |
| **G** | 15 | **GAMING** — fabricated specifics (§I probe) | freshly generated via gpt-oss, committed under `poc/claim_graph/eval/gaming_cases/` (synthetic adversarial fixtures: invented statistics, fake named studies, synthetic "I initially thought…" revision narratives) |

**Total 60 docs** (= `MAX_DOCS`). Unit contract (`interrog_ab.py`, matched live via
`interrog_current_recompute` == the signal's own composite in every row): interrogatability composite
is **0–1, higher = more interrogatable**; substitutability is **0–1, higher = weaker grounding**.

## Cost / latency actuals

- **Eval gateway calls:** 128 total, **mean 2.13/doc, max 3/doc, 0 docs over the 5-call soft cap**
  (well under the ≤5/doc and 350-global budget). Gaming generation: 30 calls (15 refused under the
  first framing, 15 succeeded after a fiction-framing rewrite). Substitutability embeds are local
  MiniLM (near-free).
- **Wall:** ~456 s for the 60-doc eval (~7.6 min); ~2× shorter with the in-process cache warm.
- **Est. cost:** Cerebras `gpt-oss-120b` at fractions of a cent per short-essay call → total run
  **< ~$0.20**. (No Modal / paid deep-scan touched.)

---

## Headline answers (a)–(f)

### (a) Does interrogatability separate H (human) from A (AI-generic)? — WEAKLY, NO USABLE MARGIN
`interrogatability_current` mean composite: **H 0.515 vs A 0.464** (Δ ≈ 0.05). Band distribution is
nearly identical — both pile into *moderate*:

| Group | mean | low | moderate | high |
|---|---|---|---|---|
| H | 0.515 | 0.00 | 0.92 | 0.08 |
| A | 0.464 | 0.05 | 0.95 | 0.00 |

The composite does **not** give a threshold that separates human from AI-generic. `mean_unverified_
specific_rate` is higher for H (0.134) than A (0.078) — genuine student writing carries *more*
unverified specifics (lived detail) than generic AI text, exactly why the §A invariant forbids
crediting them.

### (b) Does the GAMING group G score HIGH interrogatability under the CURRENT formula? — YES (feared failure CONFIRMED)
**G mean composite 0.725; 86.7% of G docs band `high`** (13/15) vs H's 8% and A's 0%. The current
`specificity_presence` component counts specifics regardless of `verification_status`, so an essay
**stuffed with fabricated statistics and fake studies scores the HIGHEST interrogatability of any
group** — a direct violation of the §A headline invariant. This is the failure the M4 probe existed
to catch, and it reproduces cleanly.

### (c) A/B — verification-weighted specificity (offline, same graphs): does it fix G without tanking H? — NO, IT DOES NOT FIX G
Recompute (offline, no new LLM calls) with `specificity_presence` restricted to
verified/corroborated/**internally_supported** claims:

| | G high-rate | H false-thin rate (banded low) |
|---|---|---|
| current | 0.867 | **0.00** |
| verified-weighted | **0.867** | **0.00** |

- **G high-rate is UNCHANGED (0.867 → 0.867);** G mean barely moves (0.725 → 0.723).
- **Root cause:** in Phase 1 `internally_supported` means *internally coherent, NOT factually true*
  (§2). A fabricated-but-coherent gaming essay's claims are promoted to `internally_supported` by the
  validator, so verification-weighting **still credits the fabricated specifics.** The A/B as
  specified is therefore **ineffective** — the restriction has to be to an *external* verification
  path, which **no Phase-1 engine can supply** (entailment/context grounding are Phase 2).
- **Fairness-gate analog (mislabeling genuine work as thin):** H false-thin rate is **0.00 for both
  variants** — no genuine student essay is banded `low`. So the variant does not harm H; it simply
  fails to fix G.

**Conclusion (c):** verification-weighting does not rescue interrogatability's composite in Phase 1.
The only §A-safe treatment is to **not credit the specificity component at all** and keep
interrogatability as a pure **audit-question surface** (`unverified_specific_rate` + emitted QUESTIONs).

### (d) Substitutability separation + agreement with `generic_assertion_risk` — DOES NOT SEPARATE; MODERATE AGREEMENT
Mean substitutability: **H 0.985, A 0.967, G 0.912** — all band `high` under the provisional 0.90
cut (H/A 100% high, G 60% high / 40% moderate). It does **not** separate genuine from generic:
- **G is the LOWEST** (least substitutable) because injected rare/specific nouns lower cosine — the
  §I gaming counter behaving exactly as documented (specific-injection lowers substitutability without
  real grounding), confirming it is **necessary-not-sufficient, never an independent credit**.
- **Agreement with `generic_assertion_risk`:** Pearson **r = 0.552** (moderate, same direction).
  Notably the existing lexical `generic_assertion_risk` separates the groups *better* than
  substitutability here (H 0.70, A 0.89, G 0.357), i.e. substitutability is **not adding a
  discriminating axis** at current cuts — it augments, it does not replace (as the plan states).

### (e) Origin-map sanity per group — PLAUSIBLE BUT WEAK DISCRIMINATOR
`unsupported_assertion_rate` is low and flat (H 0.059, A 0.036, G 0.041) — does not separate.
Primary-origin distribution is directionally sane but LLM-label-noisy:

| Group | primary_origin counts |
|---|---|
| H | original_analysis 13, external_source 5, personal_observation 4, common_knowledge 3 |
| A | original_analysis 15, interpretation 2, common_knowledge 1, external_source 1 |
| G | external_source 7, original_analysis 4, personal_observation 4 |

Sane signals: H carries the most `personal_observation`; **G leans `external_source`** (its fabricated
citations), which is exactly why origin alone must stay descriptive audit metadata, never a credit
(§I) — "external_source" here points at *fake* sources.

### (f) Measured threshold recommendations — RECOMMENDED, NOT PROMOTED (owner decides in M5)

1. **interrogatability composite — DO NOT SCORE, DO NOT BAND AS A CREDIT.** It ranks the fabricated
   gaming set highest (b) and the verification-weighted fix fails in Phase 1 (c). **Recommendation:**
   keep interrogatability as an **audit-only** signal — surface `unverified_specific_rate` and the
   emitted QUESTION nodes; drop the composite band from any promotion path until a Phase-2 external
   verification path exists. Stays EXPERIMENTAL.
2. **`specificity_presence` component — flag the M3 agenda item RESOLVED as "cannot credit in Phase
   1".** Neither the current nor the internally_supported-weighted variant is §A-safe; both credit
   fabricated specifics. No threshold recommended.
3. **substitutability — recalibrate bands UPWARD if ever banded** (current 0.80/0.90 collapse
   everything to `high`; observed range 0.91–0.985). A provisional re-cut near low<0.94 / high>0.975
   would at least spread the mass, **but it still does not separate G from H** → keep EXPERIMENTAL,
   necessary-not-sufficient, corroborate-only.
4. **origin_map — no threshold; keep descriptive.** `unsupported_assertion_rate` does not separate;
   `personal_observation`/`external_source` stay NEUTRAL audit labels (§I).
5. **Fairness gate (false-thin) status:** H false-thin = 0.00 on both interrogatability variants — the
   mislabel-genuine-work harm is not observed, but only because the bands are lenient; this is **not**
   evidence the signal is ready, given (a)/(b)/(c).

**Overall:** No Phase-1 claim-graph signal passes the §B bar to leave EXPERIMENTAL. The strongest,
most actionable finding is the **§I gaming failure (b)** and the demonstration that the obvious fix
**(c) does not work in Phase 1** — both are load-bearing inputs to the M5 promotion decision (the
honest answer is "do not promote; interrogatability is audit-only until Phase-2 entailment exists").

---

## Reproduce

```bash
# 1. build manifest (H local-only, A committed) + regenerate gaming set (needs gateway keys)
python -m poc.claim_graph.eval.build_corpus_manifest
python -m poc.claim_graph.eval.generate_gaming_cases      # writes eval/gaming_cases/G_NN.json
python -m poc.claim_graph.eval.build_corpus_manifest      # re-pick up G
# 2. run the eval (resume-safe; ~7.6 min, ~128 gateway calls)
python -m poc.claim_graph.eval.run_eval
# 3. aggregate the (a)-(f) numbers
python -m poc.claim_graph.eval.analyze
```

Deterministic utilities are unit-tested in `poc/claim_graph/eval/test_eval.py` (manifest determinism,
row/analysis schema, and the offline A/B's unit-parity with `signals.compute_interrogatability`). The
eval run itself is a measurement, not a test (plan §7).
