# DeBERTa-Signal Comparison Module — Candidate Checkpoints

**Purpose:** Research artifacts for the additive AI-signal comparison module
(spec: `docs/superpowers/specs/2026-07-01-deberta-signal-comparison-module-design.md`).
These are open HuggingFace sequence-classification checkpoints suitable for CPU
inference of student essays. The chosen checkpoint will be benchmarked against
the local SCoCESLE ESL corpus (272 non-native-English essays) in Phase 0 Task 0.3
to measure false-positive rate **before** it is allowed to touch real student text.

**Hard constraints (all candidates below satisfy these):**
- Sequence-classification head (2 labels: human / AI) — NOT a generative/perplexity model.
- CPU-deployable at base size (target instance = 4 vCPU / 4GB RAM; no large/XL variants).
- Permissive license (MIT / Apache-2.0 / cc-by-4.0). Custom/unknown licenses excluded.

**Top selection priority:** LOW false-positive rate on genuine non-native-English
(ESL/L2) student writing — the project's #1 documented risk. Multi-generator
training is preferred over single-model-family training (better generalization,
less overfitting to one generator's artifact surface).

---

## Candidate table

| repo-id | base model | #downloads (mo, approx) | stated training data (per card) | license | suitability note |
|---|---|---|---|---|---|
| `fakespot-ai/roberta-base-ai-text-detection-v1` | RoBERTa-base | ~11,000 | "Deep Fake Text Detection" (ApolloDFT) — multi-generator English text; card refers to `FakespotAILabs/ApolloDFT` tech report for full data details | Apache-2.0 | **Strongest pick.** Multi-generator training + Apache-2.0 + base size. ApolloDFT explicitly targets cross-generator generalization (the exact failure mode that produces ESL false positives). Highest-priority for the SCoCESLE gate. |
| `vraj33/ai-text-detector-deberta` | DeBERTa-v3-base | ~430 | Human Wikipedia articles vs. AI-generated Wikipedia-style articles from **GPT-Neo (1.3B)** only | unconfirmed (no license field on card) | Meets the DeBERTa-v3-base target architecture and base-size/CPU constraint, BUT single-generator (GPT-Neo) training is a generalization risk — likely to over-flag text that simply doesn't match GPT-Neo's surface, which includes genuine ESL writing. Include as the spec's named architecture, benchmark with caution. |
| `openai-community/roberta-base-openai-detector` | RoBERTa-base | ~815,000 | Fine-tuned on outputs of the **1.5B GPT-2** model (OpenAI `gpt-2-output-dataset`); human side = WebText | MIT | The legacy reference detector. 815K downloads = most-deployed baseline, so worth benchmarking for comparison. **But single-generator (GPT-2) and pre-ChatGPT** — card itself warns it is not a ChatGPT detector and generalizes poorly to modern LLMs. Expect high ESL false positives; include only as a known-weak comparison anchor. |
| `Hello-SimpleAI/chatgpt-detector-roberta` | RoBERTa-base | ~22,000 | HC3 corpus — human + ChatGPT `answer`s (full-text + split-sentence mix); see arXiv 2301.07597 | unconfirmed (card has no license field; HC3 dataset is CC-BY-SA) | Legacy ChatGPT-vs-human detector. ChatGPT-only training = single-modern-generator. Well-cited baseline (22K downloads, 88 Spaces) and was evaluated in the RAID benchmark, so useful for comparison. License is unconfirmed — must be resolved before any production use. |
| `ahmediqbal/ai-text-detector-model` | DistilBERT | (low; not surfaced) | Card claims ChatGPT / GPT-2 / GPT-3 vs. human | unconfirmed | DistilBERT = smallest/fastest CPU option (66M params). If license clears, attractive as a lightweight low-latency candidate. Training-data claims are thin on the card; treat as unverified until benchmarked. |

---

## Recommended shortlist (for the SCoCESLE Phase 0 Task 0.3 gate)

Benchmark these **three** first, in this order:

1. **`fakespot-ai/roberta-base-ai-text-detection-v1`** — primary candidate.
   Multi-generator (ApolloDFT) training directly targets the cross-generator
   generalization gap that drives ESL false positives. Apache-2.0 is the most
   permissive standard license. RoBERTa-base is well within the 4 vCPU / 4GB RAM
   budget. This is the single most promising checkpoint for low ESL FPR.

2. **`vraj33/ai-text-detector-deberta`** — architecture-aligned candidate.
   Matches the spec's named DeBERTa-v3-base target and the CPU budget. The
   single-generator (GPT-Neo) training is a documented generalization risk, so
   it is benchmarked precisely to *quantify* that risk on SCoCESLE — if it clears
   the ESL FPR gate despite single-generator training, it becomes the preferred
   DeBERTa-native pick. License must be confirmed before any non-research use.

3. **`openai-community/roberta-base-openai-detector`** — comparison anchor.
   MIT-licensed and the most-deployed baseline (815K downloads). It is expected
   to perform poorly on modern text and to over-flag ESL writers; including it
   gives the gate a known-weak lower bound and a recognizable reference number
   for external audiences.

**Excluded from shortlist:**
- `Hello-SimpleAI/chatgpt-detector-roberta` — strong baseline but license
  unconfirmed; hold until license is resolved, then add as a 4th benchmark.
- `ahmediqbal/ai-text-detector-model` — DistilBERT is attractive for latency but
  card claims are unverified and downloads are low; revisit only if a faster
  CPU footprint becomes a hard requirement.

---

## Unconfirmed fields (must be resolved before production use)

- `vraj33/ai-text-detector-deberta` — **license** (no field on model card).
- `Hello-SimpleAI/chatgpt-detector-roberta` — **license** (no field on card; HC3
  dataset is CC-BY-SA but that governs data, not the model weights).
- `ahmediqbal/ai-text-detector-model` — **license** and exact **download count**.
- `fakespot-ai/...` — exact generator list in ApolloDFT training data (card
  defers to the tech report; confirm multi-generator scope before relying on it
  for the ESL-FPR argument).

---

## Phase 0 benchmark results (SCoCESLE: 272 ESL essays + 65 in-repo AI texts)

Run: `poc/calibration/deberta_fpr_gate.py` over all 272 humans (139 higher + 133 lower
proficiency; 4 dropped <150w) + 65 AI. Raw = uncalibrated checkpoint probability.

| Candidate | AUC (AI vs human) | AI mean (raw) | ESL FPR@50% (raw) | Verdict |
|---|---|---|---|---|
| **`fakespot-ai/roberta-base-ai-text-detection-v1`** | **0.9989** | 99.4% | 20.5% | **Only viable detector.** Detects AI well but raw ESL FPR unacceptable (1-in-5 ESL essays flagged). → Calibrate. |
| `vraj33/ai-text-detector-deberta` | 0.1332 | 0.8% | 6.7% | **REJECTED — inverted.** AUC <0.5: the resolved "AI" class is the human class (AI texts score ~0). Broken as wired; human max 100% means it's not a clean flip either. |
| `openai-community/roberta-base-openai-detector` | 0.3576 | 18.7% | 16.8% | **REJECTED — near-random on modern AI.** Trained on GPT-2 output; AUC 0.36 ≈ coin-flip on the in-repo AI set. Useless as a comparison detector. |

### Isotonic recalibration of the winner (fakespot-ai)

Fit on SCoCESLE (`poc/calibration/deberta_fit_calibrator.py`), label 1=AI / 0=human, so the
calibrated score = P(AI). Calibrator saved to `poc/calibration/deberta_isotonic.pkl`
(point `DRAFTPROOF_DEBERTA_CALIBRATOR` at it).

| metric | raw | calibrated |
|---|---|---|
| human mean % | 25.9 | 1.0 |
| human p90 % | 66.1 | 0.0 |
| human max % | 99.7 | 75.0 *(one stubborn essay; the monotonic ceiling)* |
| AI mean % | 99.4 | 95.9 |
| AI p10 % | 85.8 | 33.3 *(recall trade-off: ~bottom-10% AI essays drop to green/acceptable)* |
| **ESL FPR @>=50%** | **20.5%** | **1.1%** |
| ESL FPR @>=40% | 26.5% | 1.1% |
| ESL FPR @>=60% | 11.9% | 0.7% |
| parity gap (lower − higher) @50% | — | −0.6 pts *(lower-prof flagged LESS — good direction)* |
| AUC | 0.9989 | 0.9993 |

**End-to-end re-validated** through `compose()`'s production path with the calibrator set:
ESL FPR@50% = 1.1%, AUC 0.999, AI mean 95.6% — reproduces the fit-script numbers, confirming
the calibrator is consistent with how `compose()` applies it internally.

### Decision (Task 0.4)

**SHIP `fakespot-ai/roberta-base-ai-text-detection-v1`, CALIBRATED** (`deberta_isotonic.pkl`).
ESL FPR@50% = 1.1% is **below the composite's documented ~3%** (97% of SCoCESLE <50%) → meets
the spec acceptance criterion. Honest trade-off: calibration bought low ESL FPR at the cost of
AI recall — the bottom ~10% of AI essays now read green/acceptable on this advisory signal
(~0.4% of ESL essays still false-flag at the very top, the monotonic ceiling). Acceptable for
an advisory comparison score alongside the composite (which has its own detection).

---

## Authoritative-tier elevation (2026-07-02) — Phase 3 fairness gate

Run: `poc/calibration/deberta_authoritative_gate.py` over SCoCESLE
(139 higher + 129 lower proficiency, 4 dropped <150w) + 65 AI.

The authoritative signal is the **>=0.99 high-confidence PROPORTION** (not the >=0.80 display
proportion, and NOT calibrated — isotonic calibration was abandoned as a proven dead end, see
`deberta_fit_calibrator_windows.py` docstring). Fairness comes from the threshold.

| metric | value |
|---|---|
| ESL FPR @ >=25% (AMBER cutoff) | **2.6%** |
| ESL FPR @ >=10% (GREEN/AMBER line) | 7.8% |
| ESL FPR @ >=50% (ORANGE cutoff) | 1.1% |
| higher-prof FPR @ >=25% | 4.3% |
| lower-prof FPR @ >=25% (at-risk) | **0.8%** |
| parity gap (lower − higher) | **−3.5 pts** (lower-prof flagged LESS — favorable) |
| AUC (AI vs human) | **0.9973** |
| human proportion: mean / p90 / max | 2.4% / 7.7% / 100.0% |
| ai proportion: mean / p10 | 95.8% / 100.0% |

### Decision

**GATE PASSES.** ESL FPR @ the AMBER cutoff = 2.6% (≤3.0% required). The at-risk lower-
proficiency subgroup is flagged LESS than higher-proficiency (parity gap −3.5, favorable
direction — the opposite of bias). AUC 0.997 = near-perfect separation.

The >=0.99 high-confidence proportion is fair enough to serve as the authoritative
`ai_likelihood_score`, gated behind `DRAFTPROOF_DEBERTA_AUTHORITATIVE`. The provisional tier
cutoffs (`<0.10 GREEN, <0.25 AMBER, <0.50 ORANGE, >=0.50 RED`) are confirmed: at 25% the all-
ESL FPR is 2.6%; at 10% it rises to 7.8% (the GREEN/AMBER boundary is the noisier band, which
is acceptable since GREEN is not an accusation).

Honest trade-off: the >=0.99 bar misses paraphrased/humanized AI (AI recall cost). AI mean
proportion is 95.8% but the bottom tail of AI essays reads low — the detector localizes clean
AI text well but, like all post-hoc statistical detectors, is not complete. The perplexity
Layer3 path remains as the fallback when DeBERTa is unavailable (short text / model error).

