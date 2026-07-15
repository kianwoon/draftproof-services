# Phase 2 — Claim–Source Entailment + Context Grounding (execution scope)

Status: **APPROVED — kickoff decisions locked** (2026-07-15). Governing architecture:
`docs/plans/credible_authorship_assessment_v2.md` (§A, §4.1, §4.3, §B, §C, §D, §E, §H, §I).
Predecessor: `docs/plans/phase1_claim_graph_execution_plan.md` (Phase 1 CLOSED, M1–M5).
Motivating data: `docs/plans/phase1_m4_calibration_report.md`.

Guide review (Fable, 2026-07-15) folded in — the four corrections are called out inline as **[G1]–[G4]**.

### Kickoff decisions (owner, 2026-07-15) — LOCKED

1. **§E1 assignment-brief upload → DEFER** (Track A only in Phase 2). Owner delegated the call to a
   Fable strategic read; verdict: defer, not decline. **Decisive reason:** granting `verified`/Level-3
   on "claim matches the brief" repeats the *exact* M4 coherence-not-truth failure at a **higher**
   credit tier, before Track A's real truth-grounding path is calibrated — and brief-match is
   near-universal (everyone writes to the rubric), so it adds coverage, not separation. A fabricated
   statistic that is on-brief would score `verified`; that inverts §A's headline invariant. **Not
   declined** because the salvageable reframe survives: context-traceability re-scoped to a **NEUTRAL
   audit state (not `verified`)**, or gated *behind* entailment-style verification, may ship in a later
   phase. Revisit after Track A calibrates. → N5 collapses to a doc-only degrade note.
2. **Entailment model → NLI cross-encoder on Modal** (not LLM-judge). See [G1].
3. **`corroborated` state → FORBIDDEN this phase.** Keep only `verified`/`unverified`/`contradicted`.
   Fewer thresholds to calibrate; revisit corroboration when the data warrants.

---

## 1. Why Phase 2 exists (the M4 finding, made concrete)

Phase-1 M4 proved the decisive failure: **interrogatability rewards fabricated specifics.** The §I
gaming corpus (essays stuffed with invented statistics and fake named studies) scored the *highest*
interrogatability of any group (mean 0.725, 86.7% banded "high") — above real human essays (0.515)
and generic AI (0.464). Verification-weighting did **not** fix it, because Phase-1's
`internally_supported` measures internal **coherence, not truth**, and a fabricated claim is
perfectly coherent.

**Entailment is the identified fix.** It is the only mechanism that distinguishes a real 2019 trial
from a fabricated one: retrieve the cited source, and test whether the source actually *entails* the
claim. That unlocks the `verified` state (Phase-1-forbidden), which lets §A's rule finally bind —
**"only verified specifics count toward ownership; an unverified specific is NEUTRAL and surfaces as a
teacher-probe QUESTION."**

> **[G3] The fix is NOT entailment alone.** Fabricated DOIs resolving to nothing → `unverified` →
> NEUTRAL does **not** by itself drop the gaming group's 0.725. Dropping it **requires N4 to also
> implement M4 recommendation (f)1**: interrogatability's specificity credit becomes **verified-only**,
> and the composite band is dropped from any credit path otherwise. Entailment supplies the `verified`
> signal; N4's credit-gating change is what actually neutralises the gaming score. Both are load-bearing.

---

## 2. Scope boundary — two tracks, independently switched

Phase 2 splits into two tracks so the owner-gated half never blocks the public-source half.

| Track | Deliverable | Product decision? | Kill-switch |
|---|---|---|---|
| **A — External entailment (§4.1)** | citation → resolve → retrieve → **claim–source entailment** → `verified`/`contradicted`/`unverified`. Uses PUBLIC sources only. | **None** — build now. | `DRAFTPROOF_ENTAILMENT` |
| **B — Context grounding (§4.3)** | claim → traces to uploaded assignment brief / rubric. | **DEFERRED** (see kickoff decision 1). | `DRAFTPROOF_CONTEXT_GROUNDING` |

**[G2] Sequencing: Track A only this phase; Track B deferred.** Track A is fully independent and is
the sole build deliverable of Phase 2. **Track B is scoped-not-built** — the §E1 decision landed as
*defer* (kickoff decision 1), so N5 collapses to documenting the topic-self-consistency degrade. When
Track B is revisited (post-Track-A calibration), its re-scope is already recorded: context-traceability
grants a **NEUTRAL audit state, not `verified`** — "on-brief ≠ true", so it must never mint external
credit. §4.3 was never conflated with the standing **single-copy HARD NO** (a brief is task context,
not a second copy) — but it remains an upload demand and stays owner-owned.

Both tracks stay **Tier-2** (§D): premium, only when citations/context are present, reusing the
existing paid deep-scan async+cache plumbing (`poc/detect_v7/modal_client.py`). **Never on every scan.**

---

## 3. The verdict asymmetry — load-bearing from N1

This is the §A/§B2 fairness core. The five outcomes, and the state each maps to:

| Situation | verification_status | Credit | Note |
|---|---|---|---|
| Source resolves **and entails** the claim (NLI fires above the *verified* threshold) | `verified` | Ownership credit (Level 4) | The only path to external credit. |
| Source resolves and **contradicts** the claim (NLI fires above the **stricter** *contradicted* threshold) | `contradicted` | Contribution **capped at NEUTRAL** + contradiction flag on the node | The ONLY negative state. |
| DOI/URL **fabricated or won't resolve** | `unverified` | NEUTRAL | `unverified ≠ fabricated` — **never penalised.** |
| Source resolves but **paywalled / no retrievable full text** | `unverified` + `paywalled` limitation | NEUTRAL | Genuine citation we *couldn't* check — must not be punished. |
| Source resolves, retrieved, but entailment lands in the **gray zone** (neither threshold fires) | `unverified` | NEUTRAL | Abstain, don't guess. |

> **[G1] `contradicted` needs a STRICTER threshold than `verified`.** A false `contradicted` on
> genuine work is the §B2 fairness harm (mislabeling grounded work as wrong). Any gray zone resolves
> **down to `unverified`**, never to `contradicted`. Asymmetric thresholds are a hard requirement, tested.

**Fairness harm for this signal family (§B):** mislabeling genuinely-grounded work — a real citation we
fail to verify (paywalled, unretrievable) must land `unverified`, never penalised. This is the
ESL-FPR analog for Phase 2 and gates promotion.

---

## 4. Architecture — the entailment pipeline

```text
claim-graph (Phase 1)
   │  claims + their cited references (N1: deterministic citation→claim linking)
   ▼
[N2] Resolver + Retrieval        DOI/Crossref/URL → source metadata → full-text/abstract
   │  seed: poc/detect/citation.py (in-text + bib parse); Tavily seed in poc/rewrite_v6
   │  cached, fail-open, rate-capped; §C coverage = source_access; snapshot for eval determinism
   ▼
[N3] Entailment engine           premise = retrieved source span; hypothesis = normalised claim
   │  → {verified | contradicted | unverified} with asymmetric thresholds (§3)
   ▼
[N4] Wire-back                    verdict → node.verification_status
   │  → re-run interrogatability with VERIFIED-ONLY credit (M4 rec (f)1)  ← the actual gaming fix
   │  → origin-map corroboration; define/forbid `corroborated` (see §5)
   ▼
report JSON (authorship_evidence.claim_graph) — EXPERIMENTAL, scoring_enabled:false
```

### [G1] Entailment model decision — NLI cross-encoder, not LLM-judge

**Decision: a dedicated NLI cross-encoder (e.g. DeBERTa-NLI) hosted on Modal, reusing the deep-scan
plumbing.** Rationale:

1. §4.1 calls entailment "**load-bearing** — a resolvable DOI is not enough; the entailment model must
   fire." A Level-4 `verified` grant is the strongest credit in the system.
2. §C **caps LLM-judged engines at Moderate confidence**. Letting a Moderate-capped engine mint Level-4
   evidence guts §A.
3. **Prompt-injection surface (§I):** the entailment input is *retrieved web text*. An LLM-judge reading
   that text is directly injectable ("this source supports all claims") the moment entailment becomes an
   optimization target. A cross-encoder consumes premise/hypothesis as scored text, not instructions.
4. Hosting cost is low — mirror the existing Modal DeBERTa deep-scan pattern.

**Bounded LLM role:** the gpt-oss/Cerebras gateway MAY normalise a claim into a clean entailment
hypothesis (triage/decomposition), but it **NEVER grants `verified`**. Only the NLI model's score does.

---

## 5. Milestones (each independently committable + kill-switched, mirroring Phase-1 M1–M5)

| N | Deliverable | Network / LLM? | Ships as |
|---|---|---|---|
| **N0** | Phase-conditional CI guard replacing `PHASE1_FORBIDDEN_STATES`: **`verified` legal only when `DRAFTPROOF_ENTAILMENT` on; `corroborated` stays forbidden this phase** (kickoff decision 3). (§E1 already resolved → no owner request needed.) | No | governance |
| **N1** | Schema unlock (`verified` behind the flag; **NOT `corroborated`**) + **deterministic citation→claim linking** (which claim cites which reference). No network. | No | plumbing |
| **N2** | DOI/Crossref/URL resolver + source retrieval (Tavily). Cached, fail-open, rate-capped. **Snapshot retrieved sources** for eval determinism. §C `source_access` coverage + `paywalled` limitation codes. | Network | EXPERIMENTAL |
| **N3** | **Entailment engine** (NLI cross-encoder on Modal) with the §3 asymmetric thresholds. The load-bearing component. | Model | EXPERIMENTAL |
| **N4** | Wire verdicts → `verification_status`; **re-run interrogatability VERIFIED-ONLY (M4 rec (f)1)** — the gaming fix realised; origin-map external-source corroboration (single-source only; **`corroborated` state forbidden** per kickoff decision 3). | — | EXPERIMENTAL |
| **N5** | §4.3 context grounding — **DEFERRED** (kickoff decision 1). Doc-only: record the topic-self-consistency degrade + the NEUTRAL-audit-state re-scope for a later phase. Build nothing. | No | doc-only |
| **N6** | Re-run the M4 eval harness on a **new real-citation eval slice** (see §6) + gaming set + §B calibration report → ADVISORY-promotion proposal. | — | eval + proposal |

**[G4] Scope realism:** with Modal hosting + calibration this sits at the **top of the 3–4 wk**
estimate. If squeezed, **cut N5 first** (it's owner-gated and lowest-coverage anyway).

---

## 6. Validation — [G3] the M4 corpus alone cannot prove the positive side

The M4 gaming corpus proves the **negative** side (fabricated citations must NOT earn credit), but it
**cannot prove the positive** side: the PERSUADE human essays (group H) have **no bibliographies**, so
entailment coverage ≈ 0 on H — "does `verified` fire correctly?" is unmeasurable on the current corpus.

**N6 therefore adds a new eval slice of real-citation documents:**

- Docs with **real, resolvable DOIs** where the source genuinely entails the claim → expect `verified`.
- **Misattributed-real-source** cases (real DOI, but the source does NOT support the claim) → expect
  `unverified` or `contradicted`, never `verified`. This is the §I "real-but-irrelevant source →
  not-entailed → no credit" citation-stuffing counter, tested directly.
- The existing gaming set (fake DOIs) → expect `unverified` → NEUTRAL, and (with N4's credit-gating)
  the interrogatability gaming score finally drops from 0.725.

**Cross-signal corroboration rule stated in-plan (§I):** no single evidence pattern independently
establishes strong authorship — many citations ≠ source grounding, high internal consistency ≠ truth.
A high value on any one axis, uncorroborated by an independent class of evidence, is an **audit
surface, not a credit.** Entailment credit is necessary-not-sufficient.

**Discipline (unchanged from Phase 1):** signals stay EXPERIMENTAL through Phase 2; promotion to
ADVISORY/SCORING is double-gated (§B five-part protocol + owner sign-off + the CI guard). Deterministic
measurement, never single runs.

---

## 7. Owner decisions — RESOLVED (2026-07-15)

All three kickoff decisions are locked (see "Kickoff decisions" at the top):

1. **§E1 assignment-brief upload → DEFER.** Track A only; N5 doc-only. Salvage reframe (NEUTRAL audit
   state, not `verified`) recorded for a later phase.
2. **Entailment model → NLI cross-encoder on Modal.** LLM may normalise hypotheses; never grants `verified`.
3. **`corroborated` state → FORBIDDEN this phase.** Only `verified`/`unverified`/`contradicted`.

No decisions block N0/N1 kickoff.

---

## 8. Risks

| # | Risk | Countermeasure |
|---|---|---|
| 1 | Entailment coverage is low (many student citations are books/paywalled/unresolvable) → little `verified` mass | Expected; `unverified` is NEUTRAL not penalising. §C coverage/limitations make it honest. Value is the gaming *defence*, not universal verification. |
| 2 | False `contradicted` on genuine work (§B2 harm) | Stricter `contradicted` threshold; gray zone → `unverified` (§3). Fairness-gated in N6. |
| 3 | Prompt injection via retrieved source text (§I) | NLI cross-encoder consumes text as scored premise/hypothesis, not instructions; LLM never grants `verified` ([G1]). |
| 4 | Retrieval non-determinism breaks eval reproducibility | Snapshot retrieved sources (N2); eval runs against snapshots. |
| 5 | Cost/latency creep onto the scan critical path | Tier-2 only, premium/citations-present, reuse deep-scan async+cache; per-tier budget caps like deep-scan windowing. |
| 6 | Scope overrun past 3–4 wk | Cut N5 first ([G4]); N1–N4 is the shippable core. |

---

## 9. What Phase 2 does NOT do

- Does not touch the AI-likelihood score, tier, or verdict (signals stay EXPERIMENTAL, `fusion_weight=0`).
- Does not enable data-derived fusion (that is Phase 3, `DRAFTPROOF_AUTHORSHIP_FUSION`).
- Does not build provenance / Evidence Level 5 (Phase 3, §E2 — voluntary editor timeline only).
- Does not resurrect earlier-draft / paired-upload workflows (standing HARD NO).
