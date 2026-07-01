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
