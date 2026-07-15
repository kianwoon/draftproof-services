# Phase-2 N6 — Entailment §B Calibration Report (MEASURED, RECOMMENDED-NOT-PROMOTED)

**Status:** Measurement report. **Date:** 2026-07-15.
**Governing specs:** `docs/plans/phase2_entailment_grounding_scope.md` (§3 verdict asymmetry,
§6 validation, [G1]–[G3]); `docs/plans/phase1_m4_calibration_report.md` (the M4 baseline this
must beat); `docs/plans/credible_authorship_assessment_v2.md` §A/§B.
**Nature:** Every Phase-2 signal stays **EXPERIMENTAL** (`scoring_enabled:false`, `fusion_weight=0`).
This report produces **RECOMMENDED-NOT-PROMOTED** threshold observations; promotion is an owner
decision at M5. **No committed thresholds/weights were changed** (§B4).

The eval ran **LIVE**: entailment ON (`DRAFTPROOF_ENTAILMENT=1`), the NLI cross-encoder hitting the
deployed Modal endpoint (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`), gpt-oss-120b
(Cerebras) for extraction. Source resolution replays a **frozen Crossref snapshot** (N2
`SourceSnapshot`, `poc/claim_graph/eval/real_citation_snapshot.json`) so the run is reproducible
without re-hitting Crossref (risk R4).

## Harness + corpora

- **Harness:** `poc/claim_graph/eval/run_eval.py` (parameterised: `--manifest/--out/--groups`),
  per doc: canonical segments → M2 extraction → N1 citation-link → N2 resolve (snapshot) → N3
  entailment (live NLI) → N4 verified-only credit → row. Resume-safe JSONL. Analysis:
  `poc/claim_graph/eval/analyze_n6.py`.
- **Gaming corpus (G):** the 15 committed M4 `gaming_cases/` (fabricated statistics + fake named
  studies; **no real DOIs**). Rows: `eval_rows_n6_gaming.jsonl`.
- **Real-citation slice (R):** 15 NEW committed fixtures (`real_citation_cases/`), built from **real
  Crossref metadata** — 8 **entailed** + 7 **misattributed**. Rows: `eval_rows_n6_real.jsonl`.

### Anti-fabrication method for the real-citation slice (§6, the critical guarantee)

Zero invented DOIs, papers, findings, or numbers. `build_real_citation_cases.py`:
1. Live Crossref search filtered `has-abstract:true,type:journal-article` across 8 distinct topics →
   real DOIs whose retrieved abstract is the ground truth.
2. The **entailed** claim is a finding sentence **lifted VERBATIM** from *that same paper's own
   retrieved abstract* (`extract_finding_sentence`; cleaning byte-identical to `sources._clean_text`,
   so the claim is provably a substring of the resolver's `source_text`). Cited with that paper's real
   DOI → resolve → same abstract → entail → `verified`. Honest because the claim **is** what the source
   says. Verified end-to-end: all 8 entailed claims are verbatim substrings of their cited paper's
   snapshotted abstract; all 7 misattributed are NOT.
3. The **misattributed** claim is a verbatim finding from paper X paired with a **different** real
   paper Y's DOI (a different topic) → source Y won't entail X's claim → `unverified` (the §I
   citation-stuffing / wrong-source counter).
4. Any candidate whose abstract yielded no clean finding sentence was **dropped, never padded**.

**Worked examples.**
- *Entailed* `RC_00`: claim *"The daily target of 10,000 steps per day was achieved at a higher
  proportion than reported in international studies and the general New Zealand."* cited with
  `10.1177/2048004017749015` (the paper it was lifted from) → resolve → entail **0.986** → **verified**.
- *Misattributed* `RC_13`: claim about *elderly urban residents' mental health / dementia costs* (lifted
  from an urban-green-space paper `10.3390/ijerph16050789`) cited with `10.1177/19433654251377005`
  (**"Effects of Wildfire Smoke Inhalation on Respiratory Health"**) → resolve wildfire abstract →
  entailment **0.036** → **unverified**. The wrong source earns no credit.

---

## Headline answers (a)–(f)

### (a) Gaming interrogatability — the M4 failure is FIXED. Δmean −0.27, "high" band eliminated.

| Variant (group G, n=15) | mean composite | high-rate |
|---|---|---|
| **M4 committed baseline** (all-specifics credit, entailment OFF) | **0.725** | **86.7%** |
| N6 same-graph M4 formula (`interrog_current_recompute`, all-specifics) | 0.668 | 66.7% |
| **N6 pipeline — VERIFIED-ONLY credit, entailment ON** | **0.459** | **0.0%** |

Band distribution under the N6 pipeline: **1 low / 14 moderate / 0 high**. **Verified claims across
the entire gaming corpus: 0** (fake named studies carry no resolvable DOI → `unresolved` → NEUTRAL →
no specificity credit). The verified-only credit gate (N4 / M4 rec (f)1) **drops the gaming mean by
0.266 vs the committed baseline** and **eliminates the §A-violating "high" band entirely (86.7% →
0%)**. The same-graph comparison (0.668 → 0.459, high 66.7% → 0%) isolates the credit-rule change from
gpt-oss extraction variance and agrees. **This is the headline: entailment + verified-only credit
neutralises the M4 gaming score that verification-weighting alone could not (M4 report (c)).**

### (b) Real-citation separation — verified fires on grounded work, never on wrong-source.

- **Entailed verified-fire: 8/8 (100%).** Every genuinely-grounded claim → `verified`.
- **Misattributed verified-fire: 0/7 (0%).** No wrong-source claim earns credit.

The separation is categorical: entailed entailment scores **[0.975 … 0.995]**, misattributed **[0.003
… 0.112]** — a gap of ~0.86 around the 0.65 verified threshold.

### (c) Fairness / §B harm — grounded work is NOT mislabeled unverified.

- **False-unverified rate on entailed docs: 0/8 (0.0%).** No genuinely-cited claim was mislabeled
  `unverified`. This is the Phase-2 analog of the ESL-FPR fairness gate and it holds on this slice.
- (Two misattributed fixtures, `RC_08`/`RC_09`, extracted 0 claims — their lifted sentences were
  method/aim statements gpt-oss did not treat as CLAIMs — so the negative side is **5/5 tested docs
  correctly not-verified**, plus 2 vacuously-safe untested docs.)

### (d) Contradicted precision — RARE and never on genuine work, but one thin near-miss.

- **`contradicted` fired 0 times** across all 30 docs (0 entailed, 0 misattributed). It is precise and
  rare, as §3 requires.
- The gray-zone-falls-to-unverified rule held: misattributed `RC_11` produced contradiction **0.7135**
  (below the 0.80 `contradicted_threshold`) → correctly **`unverified`**, not `contradicted`.
- **Near-miss to flag (§B2):** entailed `RC_07` produced a passage with contradiction **0.9916**
  *alongside* its entailing passage **0.9938**; it landed `verified` **only because entailment edged
  contradiction by 0.0022** (the `classify` `e>c` dominance rule). The claim was a *conditional* finding
  (*"…when team members are highly experienced, productivity increases by ~12.2%"*), and passage-level
  max-contradiction on conditional claims is a genuine residual risk: a slightly different phrasing
  could have flipped a genuine claim to `contradicted` — the exact §B2 harm. See (e) recommendation.

### (e) Observed score distributions → threshold validation (RECOMMENDED, not promoted)

| | entailed | misattributed |
|---|---|---|
| entailment score range | 0.975 – 0.995 | 0.003 – 0.112 |
| contradiction score range | ~0.001 – **0.9916** (1 outlier) | ~0.002 – 0.7135 |

- **`verified_threshold = 0.65` — VALIDATED (keep).** The entailed/misattributed entailment
  distributions are cleanly separated with a ~0.86 margin; any cut in [0.15, 0.97] separates them. 0.65
  is safely conservative. No change recommended.
- **`contradicted_threshold = 0.80` — VALIDATED for the observed misattributed case (keep).** The only
  misattributed contradiction (0.7135) sits below it → correctly `unverified`. The asymmetry
  (0.80 > 0.65) held.
- **RECOMMENDED (not promoted): add a verified-dominance MARGIN.** Require `entailment − contradiction
  ≥ δ` (e.g. δ≈0.1) for `verified`, so a genuine claim with a simultaneously-high contradiction passage
  (`RC_07`) resolves to `unverified` (abstain) rather than being decided by a 0.0022 coin-flip — and can
  never flip to `contradicted`. This is a `classify`-logic recommendation for owner review at M5; **not
  applied here** (no committed-logic/threshold change this phase).

### (f) Cost + latency actuals (well under every cap)

| Component | This report's runs | Cap | Notes |
|---|---|---|---|
| NLI endpoint calls | ~35 total (final real run: 13; gaming: 0) | ≤ 200 | 1 POST/doc batches ≤12 passages; gaming has no resolved sources |
| gpt-oss extraction calls | ~70 total (final real: 16; gaming: 30) | ≤ 400 | short essays, ≤2–3 calls/doc |
| Crossref calls | ~48 total (8 search + 8 resolve per build) | ≤ 60 | cached/snapshotted; do NOT rebuild without budget |
| Wall (final runs) | real 78.4 s (15 docs) · gaming 100.2 s (15 docs) | — | snapshot resolution offline; NLI warm |
| Est. $ | **< $0.05** | — | Cerebras fractions-of-a-cent/call; NLI = own Modal endpoint; Crossref free |

---

## Promotion recommendation (input to the M5 owner decision) — signals STAY EXPERIMENTAL

**Recommendation: ADVISORY-eligible for the GAMING-DEFENCE use; stay EXPERIMENTAL for external
verified-credit — pending owner sign-off at M5.**

What the evidence supports:
1. The **load-bearing gaming fix works** (a): verified-only credit + entailment eliminates the
   §A-violating "high" band on fabricated specifics (86.7% → 0%), which M4 proved verification-weighting
   alone could not do. This is the strongest, most decision-relevant result.
2. The **positive side works and is fair** (b/c): entailed 8/8 `verified`, misattributed 0/7, zero
   grounded-work mislabeling, zero spurious `contradicted`. Clean, large-margin separation.

Why it must stay EXPERIMENTAL (not auto-promote):
1. **Small, single-slice n** (15 gaming + 15 real; 8/7 split). Not the multi-corpus, double-gated bar.
2. **Positive control is near-verbatim.** Entailed claims are sentences lifted from the source, so NLI
   fires ~0.98; real student *paraphrase* stress is under-tested. The margin is huge, but the corpus is
   easy on the positive side by construction.
3. **The `RC_07` conditional-claim near-miss** (d/e) is an unmitigated §B2 risk until the
   verified-dominance-margin recommendation is adopted.
4. **Coverage is intentionally low** — only DOIs-with-abstracts resolve; books/paywalled/in-text stay
   `unverified` (NEUTRAL, never penalised, §3). The value is the gaming *defence*, not universal
   verification (§8 risk 1). Cross-signal corroboration rule stands: entailment credit is
   necessary-not-sufficient (§6/§I).

**Net:** recommend the owner treat entailment as **ready for ADVISORY consideration as the gaming
defence**, conditioned on (i) scaling the real-citation corpus with paraphrased positives, and (ii)
adopting the verified-dominance margin. Until owner sign-off at M5, all Phase-2 signals remain
EXPERIMENTAL (`scoring_enabled:false`, `fusion_weight=0`).

---

## Reproduce

```bash
# 0. endpoint + keys in env (repo-root .env): DRAFTPROOF_ENTAILMENT_ENDPOINT_URL/_TOKEN,
#    CEREBRAS_API_KEY. Then:
export DRAFTPROOF_ENTAILMENT=1
# 1. (one-shot, network) rebuild the real-citation slice + snapshot from live Crossref:
python -m claim_graph.eval.build_real_citation_cases
# 2. real-citation slice — offline resolution (snapshot) + LIVE NLI:
export DRAFTPROOF_CLAIM_GRAPH_SOURCE_SNAPSHOT="$PWD/claim_graph/eval/real_citation_snapshot.json"
python -m claim_graph.eval.run_eval --manifest "$PWD/claim_graph/eval/real_citation_manifest.json" \
    --out "$PWD/claim_graph/eval/eval_rows_n6_real.jsonl" --groups R --max-docs 15
# 3. gaming corpus (no real DOIs; no snapshot needed) — entailment ON:
unset DRAFTPROOF_CLAIM_GRAPH_SOURCE_SNAPSHOT
python -m claim_graph.eval.run_eval --manifest "$PWD/claim_graph/eval/corpus_manifest.json" \
    --out "$PWD/claim_graph/eval/eval_rows_n6_gaming.jsonl" --groups G --max-docs 60
# 4. aggregate (a)-(f):
python -m claim_graph.eval.analyze_n6
```

Deterministic utilities (finding extraction, doc assembly, cross-pairing invariant, analysis schema)
are unit-tested in `poc/claim_graph/eval/test_real_citation.py`. The eval runs and the live Crossref
build are measurements, not tests (plan §7).
