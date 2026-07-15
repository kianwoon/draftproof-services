# Phase 1 — Claim-Graph Execution Plan

**Status:** Execution plan (proposal). **Date:** 2026-07-15.
**Governing spec:** `docs/plans/credible_authorship_assessment_v2.md` (v2.1) — §H Phase 1 row.
**Scope (strict):** hierarchical claim extraction + linking + interrogatability + substitutability +
origin map. Kill-switch `DRAFTPROOF_CLAIM_GRAPH`. **Every Phase-1 signal ships EXPERIMENTAL**
(internal only, `fusion_weight=0`, no user-facing impact) per §B lifecycle. **Effort ~2–3 wk.**
Entailment (§4.1), context grounding (§4.3), provenance (§E2), and data-derived fusion (§11) are
Phase 2+ and OUT OF SCOPE here.

**Non-negotiable invariants carried from the spec** (encoded throughout this plan):
- **Headline invariant:** never credit an unverified specific — verify it, or turn it into a QUESTION.
- **§2/§10 span invariant:** no CLAIM node without a deterministic span
  `{paragraph_id, sentence_id, char_start, char_end}` anchored to canonical scan segments.
- **Node types CLAIM / INFERENCE / QUESTION never mixed** (§2).
- **§3-below:** internal consistency is a RELIABILITY MODIFIER, never independent strong evidence.
- **§D:** LLM proposes nodes/edges; the application deterministically validates and owns graph truth.
- **§I:** no single evidence pattern establishes authorship — each signal ships with a gaming counter.

---

## 0. Grounding — real seams this plan builds on (grepped, not guessed)

| Concern | Real seam |
|---|---|
| Canonical sentence/paragraph ids + spans | `poc/detect/document_structure.py::structured_sentence_segments` → `StructuredSentence{sentence_id "sNNN", paragraph_id "pNNN", start_char, end_char, sentence}`. Claims MUST anchor to these ids/spans. **Note:** the dataclass field names are `start_char`/`end_char`; the spec's node contract uses `char_start`/`char_end` — the validator maps between them (§3). |
| LLM plumbing | `poc/llm/gateway.py::LLMGateway.chat(...)` + `LLMConfig` (supports `response_format`; `model_supports_structured_outputs`/`model_supports_json_object_response_format`). Provider/model resolution: `poc/rewrite_v6/llm_config.py` (Cerebras `gpt-oss` cheap default). **Reuse the gateway directly; do NOT import from `rewrite_v6` at module level** (circular-import lesson — copy the small resolver values via env/config, not the module). |
| Embeddings (substitutability) | Worker preloads MiniLM via `worker/app/model_preload.py::_preload_semantic_model`; live embedder is `poc/detect/semantic_shape.py` (`SentenceTransformer(all-MiniLM-L6-v2)`, `_cosine`). Reuse this embedder/util — do not load a second model. |
| Async/paid execution | `poc/detect_v7/modal_client.py::call_deep_scan` + deep-scan windowing/cache. Tier-1 mirrors this paid, async, cached invocation model. |
| Report attach seam | `poc/report/report.py` (`build_authorship_evidence` seam ~L3121, `headline_confidence` ~L3429). Phase-0 Evidence Levels already attach here (git log `a1c9fc9a`, `f4073aca`). Phase-1 graph attaches under the same `authorship_evidence` object. |
| Compaction survival | `worker/app/rewrite_scan_compaction.py::SCAN_BADGE_KEYS` keep-list. |
| Test conventions | `poc/report/test_authorship_evidence_levels.py` (advisory/display-only + fresh-interpreter import test). |
| **Existing signals NOT to duplicate** | `generic_assertion_risk` + `citation_grounding_risk` (`layer3_scoring.py`), `grounding_diagnosis.py`, `citation.py`, `authorship_evidence.py`, `critical_thinking.py`. Phase-1 signals reconcile with these, never re-implement them. |

---

## 1. Data model — `authorship_evidence.claim_graph`

Graph persists in the report JSON first (§E3), Postgres later. Attached under the existing
`authorship_evidence` object so it rides the R2 report with no schema change.

```json
"authorship_evidence": {
  "claim_graph": {
    "schema_version": "cg-1",
    "kill_switch": "DRAFTPROOF_CLAIM_GRAPH",
    "status": "experimental",
    "truncated": false,
    "claims":   [ /* CLAIM/INFERENCE/QUESTION nodes */ ],
    "evidence": [ /* evidence nodes */ ],
    "edges":    [ /* typed edges */ ],
    "signals":  [ /* lifecycle-tagged signal objects */ ]
  }
}
```

**Claim node** (spans are mandatory for `CLAIM`; `INFERENCE`/`QUESTION` are system-generated and
carry `source: null`):

```json
{ "id": "c_001", "node_type": "CLAIM|INFERENCE|QUESTION",
  "text": "The intervention reduced processing time by 35%.",
  "claim_type": "factual|analytical|personal_observation|interpretation|conclusion",
  "source": { "paragraph_id": "p004", "sentence_id": "s012", "char_start": 432, "char_end": 487 },
  "verification_status": "verified|corroborated|internally_supported|unverified|unverifiable|contradicted",
  "verification_paths": [],
  "origins": ["external_source","personal_observation","original_analysis"],
  "primary_origin": "original_analysis",
  "evidence_level": 0, "assessment_confidence": "low|moderate|high", "limitations": [] }
```

- **Phase-1 verification reality:** with no entailment/context engine (Phase 2), a `CLAIM` resolves
  only to `internally_supported` (consistent sub-graph), `unverified`, `unverifiable`, or
  `contradicted` (via `inconsistent_with`). **`verified`/`corroborated` are unreachable in Phase 1**
  by construction — the validator rejects any node the LLM marks `verified` (headline invariant:
  no Phase-1 path can verify a specific, so none may be credited as verified).

**Evidence node:** `{ "id":"e_001", "kind":"internal_reference|causal_link|...", "claim_ids":[...] }`.
**Edge:** `{ "id":"x_001", "type":"supports|revises|qualified_by|...", "src":"c_001", "dst":"c_007" }`
— full normative edge set from §4 (`supports, derived_from, corroborated_by, qualified_by, revises,
contradicts, inconsistent_with, depends_on, explains, causes, observed_in, supported_by`).

**Size budget + compaction (this WILL hit R2/report caps — bounded by construction):**
- `MAX_CLAIMS = 120`, `MAX_EDGES = 300` (env-tunable `DRAFTPROOF_CLAIM_GRAPH_MAX_*`).
- **Eviction rule** when over cap: keep nodes by priority — (1) any node on a `contradicted`/
  `inconsistent_with`/`revises` edge (fabrication + epistemic signal, load-bearing), (2) `CLAIM`
  nodes with the most edges, (3) then by document order. Dropped-node ids are recorded in
  `truncated_dropped_ids[]` and `truncated=true`. Signals are computed on the FULL graph *before*
  eviction, so truncation never changes a signal value (only the persisted node list).
- **Compaction keep-list:** ~~add `authorship_evidence` … retained through `SCAN_BADGE_KEYS`~~
  **AMENDED (M1, 2026-07-15).** `SCAN_BADGE_KEYS` filters `ai_risk_badge`, but the graph attaches at
  the **top-level** `authorship_evidence.claim_graph` (not on the badge — Phase-0's
  `authorship_evidence_levels` is the badge key, a different object). That top-level object is already
  DROPPED by `compact_rewrite_scan_summary`'s top-level keep-list. **Decision: keep the claim-graph
  OUT of compacted rewrite scan copies** — a populated graph is large (see measured sizes) and every
  rewrite stores a before + after scan, so retaining it would ×2 the bloat on each rewrite artifact.
  The primary R2 scan report keeps the full graph. `test_rewrite_scan_compaction` asserts the
  exclusion; the exclusion is documented in `worker/app/rewrite_scan_compaction.py`.

**Measured sizes (M1, 2026-07-15 — `json.dumps(...).encode()` on the `cg-1` schema):**
- Empty container: **200 bytes** (negligible; this is all M1 ever attaches).
- Synthetic worst case at caps (120 claims w/ realistic text+spans, 300 edges): **79,333 bytes
  (~77.5 KB)**.
- Reference bound: `worker/app/rewrite_debug.py::MAX_REWRITE_JSON_BYTES` default **1,500,000**
  (min 250,000). Worst case ≈ **5.2%** of that bound — the graph fits the primary report comfortably,
  but ~77 KB ×2 per rewrite is why it is excluded from the compacted rewrite copies above.

---

## 2. Module layout — new package `poc/claim_graph/`

```
poc/claim_graph/
  __init__.py            # kill-switch predicate claim_graph_enabled(); no heavy imports at module load
  schema.py              # dataclasses + node-type/edge-type enums + JSON (de)serialise
  extractor.py           # Stage-1 paragraph-batch extraction (gateway.chat, structured output)
  reconciler.py          # Stage-2 reconciliation — OWNS cross-batch edge discovery (§D [owner])
  validators.py          # deterministic gate: ids, spans, edge legality, dupes, cycles (owns graph truth)
  build.py               # orchestrator: segments -> batches -> extract -> reconcile -> validate -> signals
  signals/
    interrogatability.py  # §5 EXPERIMENTAL
    substitutability.py   # §6 EXPERIMENTAL (MiniLM)
    origin_map.py         # §7 EXPERIMENTAL multi-label
  test_*.py
```

- **Kill-switch predicate** (`__init__.py`): `DRAFTPROOF_CLAIM_GRAPH` default OFF for Phase 1
  (opt-in, paid-tier), mirroring the `authorship_evidence.py` env pattern
  (`os.environ.get(...).strip().lower() not in {"0","false","no","off"}`).
- **NO `rewrite_v6` imports at module level.** The extractor needs writer model/base_url/api_key;
  resolve them from env directly (same env vars `rewrite_v6/llm_config.py` reads), or lazy-import
  inside a function if unavoidable. Circular-import lesson from CLAUDE.md.
- **Report-attach seam:** `build.py::build_claim_graph(text, segments, tier1_client)` is called from
  the existing `build_authorship_evidence(...)` site in `report.py`, guarded by the kill-switch, and
  returns `{}` (empty-capable) when disabled — Phase 0 output is untouched.

---

## 3. Extraction prompt + output contract + deterministic validators

**Stage-1 per-batch LLM output contract** (JSON via `response_format`; degrade to JSON-object mode
when `model_supports_structured_outputs` is false):

```json
{ "claims": [ { "sentence_id": "s012", "quote": "reduced processing time by 35%",
                "node_type": "CLAIM", "claim_type": "factual",
                "origins": ["external_source"], "primary_origin": "external_source" } ],
  "edges":  [ { "type": "supports", "src_quote": "...", "dst_quote": "..." } ] }
```

The LLM references sentences by `sentence_id` and quotes the exact span text; it does NOT emit char
offsets or node ids (the application assigns both).

**Deterministic validator rules (`validators.py` — must be pure + fully deterministic, §7):**
1. **Span-match enforcement (headline-invariant guard).** For each proposed `CLAIM`, locate `quote`
   inside the declared `sentence_id`'s source text; set `char_start/char_end` from the match offset
   (relative to `start_char`). **Reject the node if the quote is not found** at/within that sentence
   span — this rejects hallucinated claims the LLM invented (§I / risk R1). `INFERENCE`/`QUESTION`
   carry `source:null` and skip this check.
2. **Id assignment + dedup.** Assign `c_NNN`; collapse claims whose span ranges overlap >90% into one.
3. **Verification-status clamp.** Reject/rewrite any `verified`/`corroborated` (unreachable in
   Phase 1, §1). Default `unverified`; promote to `internally_supported` only if the node sits in a
   consistent sub-graph; set `contradicted` on both endpoints of a surviving `inconsistent_with` edge.
4. **Edge legality matrix.** Reject edges whose `type` ∉ §4 set, or whose endpoints don't resolve to
   existing node ids after dedup. `revises`/`qualified_by` between two claims SUPPRESS a would-be
   `inconsistent_with`/`contradicted` between the same pair (the required §A↔§9 epistemic-development
   protection — signposted revision/scoping is not fabrication).
5. **Cycle detection.** `depends_on` must be a DAG; reject/break edges that introduce a cycle
   (deterministic: drop the edge that closes the cycle in document order).
6. **Node-type purity.** A node is exactly one of CLAIM/INFERENCE/QUESTION; never re-typed post-hoc.

**Stage-2 reconciliation** explicitly OWNS cross-batch edge discovery (a `revises` between p2 and p7
that no single batch could see) — §D [owner refinement]. Reconciler proposes cross-batch edges; the
same `validators.py` rules 4–5 gate them. **Stage-3 targeted retries (0–5)** re-prompt only batches
the validator flagged as ambiguous (e.g. unresolved quote, dangling edge endpoint).

---

## 4. Signals spec (all EXPERIMENTAL — `fusion_weight=0`, `scoring_enabled:false`)

Each signal object carries the §B metadata contract:
`{ "signal","status":"experimental","scoring_enabled":false,"calibration_version":null,"fairness_gate_passed":false, "value":..., "coverage":{...}, "limitations":[...] }`.

**4a. Interrogatability (§5).** Per §A, high interrogatability of an *unverified* specific is a
**teacher-probe question count / audit surface, NOT an ownership credit**. Formula components (from
§5/§A): count of specifics (numbers, named entities, cited-but-unresolved figures) with
`verification_status ∈ {unverified, unverifiable}`; for each, emit a `QUESTION` node ("Which survey?
What sample? Cited where?"). Value = `unverified_specific_rate` = unverified specifics / total
specifics.
- **§I gaming counter:** a writer could strip all specifics to score "0 unverified". Counter — pair
  with substitutability (a generic doc scores high there), so stripping specifics moves the audit to
  a different axis; never credit low interrogatability alone.
- **§B calibration plan (Phase 4):** label a mini-corpus for genuinely-grounded vs thin; fairness
  gate analog = false-THIN rate by ESL/proficiency subgroup must not rise (mirror
  `fpr_subgroup_gate.py`).
- **Reconciles with, does not duplicate:** `citation.py` (in-text counts) + `grounding_diagnosis.py`
  buckets — interrogatability consumes their outputs, it does not re-parse citations.

**4b. Substitutability (§6) — MiniLM noun-swap test.** Concrete algorithm:
1. For each `CLAIM` sentence, produce a topic-neutralised variant by masking its named entities /
   domain nouns (reuse the entity spans already extracted for interrogatability) with generic
   placeholders.
2. Embed original + neutralised variant with the shared `semantic_shape` MiniLM embedder; compute
   `_cosine`.
3. **High cosine (≈unchanged meaning after swap) ⇒ high substitutability ⇒ weak grounding** — the
   sentence "still reads" with the nouns swapped. Value = mean cosine over claims (0–1, higher=worse).
- **§I gaming counter:** inject rare/specific nouns to lower cosine without real grounding — counter:
  substitutability is necessary-not-sufficient (§I), never independently a credit; corroborate with
  interrogatability's verification status.
- **§B calibration plan (Phase 4):** threshold calibrated on labeled generic-vs-grounded corpus;
  same false-thin fairness gate.
- **Reconciles with:** `generic_assertion_risk` (`layer3_scoring.py`) — this is the explicit
  embedding-based noun-swap test that signal only approximates lexically; it AUGMENTS, and the eval
  harness (§6) records their correlation so we don't ship a redundant axis.

**4c. Origin map (§7) — multi-label.** Per-claim `origins[]` + `primary_origin` over the taxonomy
`{external_source, personal_observation, original_analysis, common_knowledge, assignment_prompt,
interpretation}`. LLM proposes labels; validator ensures `primary_origin ∈ origins`. Value = origin
distribution over the graph.
- **§I gaming counter:** claim "personal_observation" on everything (unfalsifiable) — counter: origin
  is descriptive audit metadata, never a standalone credit; `personal_observation` maps to
  `unverifiable` (NEUTRAL, §2 fairness note), so it cannot lift a score.
- **§B calibration plan (Phase 4):** inter-annotator agreement on labels; no fairness gate needed
  until it scores (stays EXPERIMENTAL).
- **Reconciles with:** `authorship_evidence.py` (authorship_trace) + `critical_thinking.py` — origin
  map is the per-claim typed layer those document-level signals lack; it does not replace them.

---

## 5. Cost / latency budget

Per-scan Tier-1 (§D hierarchical): **~5–10 extraction + 1–3 reconciliation + 0–5 retry = ~6–18
`gateway.chat` calls** on Cerebras `gpt-oss` (cheap default). Substitutability adds MiniLM embeds
(local, already-preloaded model — near-free). Origin labels ride the extraction calls (no extra
round-trips).

- **Execution model:** mirror paid deep-scan (`modal_client.call_deep_scan`) — Tier-1 runs **async,
  opt-in/premium**, behind `DRAFTPROOF_CLAIM_GRAPH`, windowed by the same per-tier cap discipline as
  deep-scan. Never on the free quick-pass path.
- **Caching by text hash:** key the whole graph by `sha256(normalized_submitted_text) + schema_version
  + model_id`; reuse the deep-scan cache plumbing. A rescan of unchanged text is a cache hit (also
  makes the §6 eval harness cheap/deterministic to re-run).

---

## 6. Staged milestones (each independently committable + kill-switched)

| M | Deliverable | LLM? | Ships as |
|---|---|---|---|
| **M1** | `schema.py` + `validators.py` + `build.py` (empty-graph-capable) + report-attach seam + compaction keep-list entry. No extraction. Kill-switch wired; disabled → `{}`. | No | plumbing |
| **M2** | `extractor.py` + `reconciler.py` behind `DRAFTPROOF_CLAIM_GRAPH`. Graph populated in report JSON, **no UI**. | Yes | EXPERIMENTAL |
| **M3** | `signals/` (interrogatability, substitutability, origin) computed on the graph, attached with §B metadata. | Yes | EXPERIMENTAL |
| **M4** | Internal eval harness + labeled mini-corpus for §B calibration (deterministic-measurement convention). | — | eval only |
| **M5** | ADVISORY-promotion proposal (annotation-only, `fusion_weight=0`): owner decision + the §B CI guard test. | — | proposal |

No milestone flips a signal to SCORING; that is Phase 3+ and double-gated (owner sign-off + CI guard
asserting `scoring_enabled ⇒ fairness_gate_passed && calibration_version`).

### M5 decision record (2026-07-15 — owner: "Proceed" on the keep-EXPERIMENTAL recommendation)

**Decision: NO signal is promoted to ADVISORY. All three remain EXPERIMENTAL, audit-only,
`DRAFTPROOF_CLAIM_GRAPH` default OFF.** Basis: M4 calibration report
(`docs/plans/phase1_m4_calibration_report.md`, commit 12c61ee7):

- Interrogatability fails the §I adversarial test — the fabricated-specifics gaming set scored
  *highest* (mean 0.725, 86.7% "high") vs human 0.515 / AI-generic 0.464. Advisory exposure would
  reward fabrication, inverting the headline invariant.
- The parked specificity-weighting question is settled by data: verification-weighting does NOT fix
  the gaming failure (86.7% → 86.7%) because Phase-1 `internally_supported` measures coherence, not
  truth. **No formula change; the fix is Phase-2 entailment.**
- Fairness analog clean: 0.00 false-thin rate on genuine human essays, both variants.
- Substitutability does not separate groups; `generic_assertion_risk` separates better (r=0.552).
- Origin map: sane audit metadata only.

The §B CI guard test ships with this record (`poc/claim_graph/test_signals.py::test_scoring_promotion_ci_guard`).
Re-run the M4 harness (`poc/claim_graph/eval/run_eval.py`) after Phase-2 entailment before any
promotion proposal is re-raised. **Phase 1 is hereby CLOSED (M1–M5 complete).**

---

## 7. Test plan

- **Validator unit tests (deterministic):** span-match rejection (hallucinated quote → node dropped);
  edge-legality matrix (illegal type rejected); cycle detection on `depends_on`; `revises`/
  `qualified_by` suppresses `inconsistent_with`; verification-status clamp rejects `verified`;
  node-type purity.
- **Extraction contract tests with a MOCKED gateway** — feed a canned structured-output JSON, assert
  the built graph matches; the LLM is never called in CI.
- **Kill-switch parity:** `DRAFTPROOF_CLAIM_GRAPH` off → report byte-identical to pre-Phase-1
  (Phase-0 Evidence Levels untouched).
- **Compaction survival:** `authorship_evidence.claim_graph` present after
  `rewrite_scan_compaction` runs (nested-retain test).
- **Fresh-interpreter import test** (mirror `test_authorship_evidence_levels.py`): importing
  `poc.claim_graph` in a clean subprocess raises no error and pulls in no `rewrite_v6`.
- **Determinism note:** LLM extraction is non-deterministic — tests mock it; the VALIDATORS and
  signal math are pure/deterministic and tested directly; the M4 eval harness uses the
  deterministic-measurement convention (`DRAFTPROOF_V6_DETERMINISTIC=1`-style, N≥4, cache-backed),
  never single runs.

---

## 8. Risks

| # | Risk | Countermeasure |
|---|---|---|
| **R1** | **LLM hallucinates claims** not in the source (the 2026-07-14 fabricated-survey failure mode). | Span-match rejection in `validators.py` (§3 rule 1): a claim whose quote is not found at its declared span is dropped. The LLM does not own graph truth. |
| **R2** | **Cost creep** — extraction calls per scan. | ~6–18 calls capped, async/premium-only, text-hash cached, windowed like deep-scan (§5). |
| **R3** | **Graph size** blows the R2/report cap. | Bounded graph: `MAX_CLAIMS/MAX_EDGES`, priority eviction, signals computed pre-eviction (§1). |
| **R4** | **Batching drops cross-paragraph edges** (§8/§9/§A topology). | Stage-2 reconciliation explicitly owns cross-batch edge discovery (§3). |
| **R5** | **§I gaming** — specific-stripping, fake personal-observation labels, synthetic revision narratives, citation stuffing. | Per-signal gaming counters (§4); core rule — no single signal is a credit; all Phase-1 signals EXPERIMENTAL (`fusion_weight=0`), so nothing is yet an optimization target with score impact. |
| **R6** | Epistemic development mis-flagged as fabrication. | `revises`/`qualified_by` edges suppress the fabrication tripwire (§3 rule 4). |
| **R7** | Field-name drift (`start_char`/`end_char` vs `char_start`/`char_end`). | Validator is the single mapping point (§0/§3); tested. |
```
