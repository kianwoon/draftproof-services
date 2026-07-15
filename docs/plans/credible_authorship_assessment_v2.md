# Credible Authorship Assessment — Technical Architecture (v2)

**Status:** Governing plan. **Date:** 2026-07-15. **Supersedes:** the owner v1 doc
(`credible-authorship-assessment-technical-architecture.md`), which remains the SPINE — this v2
amends and *hardens* it, it does not replace it.
**Voice/constraints:** CLAUDE.md "Objective & Rewrite Philosophy" (annotate-don't-suppress;
evidence over verdicts); `docs/plans/rewrite_verdict_reframe_scope.md` (verdict reframe already
scoped/shipping).

## Core Principle (unchanged from v1)

From submitted content alone a system **cannot prove authorship**. It can only build a
**probabilistic authorship assessment** by extracting and fusing independent classes of evidence:
text characteristics, grounding, source, reasoning, process/provenance, and AI-pattern signals.

> **Do not place the AI detector at the centre of the system.** It is one sensor among several.

**What v2 adds over v1.** v1 is directionally right but has three exploitable gaps that this
revision closes as first-class sections:
1. **Hallucinated grounding** (§A) — v1 rewards specificity/interrogatability, but an LLM
   fabricates specifics fluently. Crediting unverified specifics *trains fabrication*. Live proof:
   DraftProof's own rewriter emitted *"a recent industry survey cited by trade magazines reports
   about one-third…"* (2026-07-14 E2E) — invented. Specificity must be **verified or neutral**.
2. **No calibration discipline** (§B) — the project has scar tissue: a subagent fabricated category
   weights (July); THREE eval↔prod unit-mismatch incidents the week of 2026-07-14. Every new signal
   needs a labeled corpus, a fairness gate, and unit-parity verification before it can score.
3. **No uncertainty contract** (§C) — v1 mentions confidence once; v2 makes banded confidence +
   coverage window + limitation codes a **cross-cutting contract on every engine**.

Sections 1–16 below preserve v1's structure, diagrams, and ideas; the lettered sections (A–H) are
the hardening layer and are normative.

---

# 1–3. Architecture, Decomposition, Claim–Evidence Graph (preserved)

The v1 target architecture stands (SUBMITTED CONTENT → Normalisation → Text Forensics / Grounding /
Source engines → **Claim/Evidence Graph** → Authorship / Integrity / AI-pattern evidence → Fusion →
Assessment). The AI detector never generates the verdict.

Document decomposition (v1 §2) into paragraphs/sentences/claims/citations/quotations/entities/
numbers/dates/personal- and assignment-references is the foundation. Per claim, extract: claim type
(factual/analytical/personal-observation/interpretation/conclusion), specificity, evidence
requirement, citation support, source relationship, contextual uniqueness, verifiability.

The **Claim–Evidence Graph** (v1 §3) is the spine. The metric is **claim-to-evidence connectivity**,
not reference count. v2 adds one edge class that v1 omitted and that §A depends on:

```text
[Claim]
   ├──supported_by──> [Citation]
   ├──derived_from──> [Dataset]
   ├──observed_in───> [Experience]
   ├──contradicts───> [Other Claim]      # v1
   └──inconsistent_with──> [Other Claim]  # NEW (v2 §A): fabrication tripwire
```

Outputs (grounded / unsupported-factual / generic / **unverifiable-specific**) are per-document and
must be **anchored to the exact claim node** (precision-first) — never "the document is generic".

---

# A. Hallucinated-Grounding Countermeasures (NEW — the v1 blind spot)

**Rule: specificity and interrogatability CREDIT authorship only when the specific is verifiable or
verified. An unverified specific is NEUTRAL, and surfaces as a teacher-probe QUESTION, not as
evidence.** This inverts v1 §4.4/§5, which implicitly credited "grounded specificity" by its
presence.

A specific earns credit on any one of three verification paths:

| Path | Mechanism | Evidence Level reached |
|---|---|---|
| **Source-entailed** | The §4.1 citation→source→**entailment** path confirms the source supports the specific. Entailment is **load-bearing here** — a resolvable DOI is not enough; the claim–source entailment model must fire. | 4 |
| **Context-traceable** | The specific traces to uploaded assignment/rubric context (§4.3). Requires the §E1 product decision. | 3 |
| **Internally consistent** | The specific participates in a consistent sub-graph (referenced again, causally consumed, not contradicted) across the claim graph. | 3 (internal) |

A specific that satisfies none of these scores **NEUTRAL** (never negative — absence of proof is not
proof of fabrication) and is emitted as an **interrogatability question** ("Which survey? What
sample? Cited where?"). Interrogatability (§5) is thus reframed: high interrogatability of an
*unverified* claim is a **question count**, i.e. an audit surface for a teacher — NOT an ownership
credit. Only verified specifics raise the ownership indicator.

**Fabrication tripwire.** The new `inconsistent_with` edges are the detector. When two claims assert
incompatible specifics (a figure stated two ways; an experience that contradicts a later dependency),
raise a **contradiction flag** on both nodes and cap their contribution to grounding at NEUTRAL.
Contradiction is treated per v1 §9 (epistemic development can *legitimately* revise a position) —
so the tripwire fires only on **unsignposted** inconsistency (no "however/I revised" reasoning edge
linking them), distinguishing genuine revision from fabrication.

**Motivation, made concrete.** An architecture that rewards unverified specificity does not just
mis-score submissions — via the shared claim graph (§F) it would train the *rewriter* to add
plausible-but-fake specifics to lift a score. The 2026-07-14 fabricated-survey output is exactly
that failure mode. Verification-gating specificity is therefore a safety requirement, not a nicety.

---

# 4. Grounding Assessment (preserved; §A gates the credit)

- **4.1 External** — citation parser → DOI/Crossref/URL resolver → source retrieval →
  **claim–source entailment**. v2 elevates entailment from "possible flow" to the required credit
  gate for external specifics (§A). Existing seed: `poc/detect/citation.py` (in-text citation
  count/gating, bib-decoupled).
- **4.2 Internal** — measurement→evidence, survey→sample/method, experiment→method/result,
  comparison→basis. Existing seed: `generic_assertion_risk`/`citation_grounding_risk` in
  `poc/detect/layer3_scoring.py`; grounding buckets in `poc/detect/grounding_diagnosis.py`.
- **4.3 Context** — reverse-RAG against assignment brief / rubric / lecture / materials. **Requires
  §E1.** Without an uploaded brief this degrades to *topic self-consistency only* — state that
  honestly per phase (§H), never present topic-consistency as context grounding.
- **4.4 Experiential** — extract "I observed / I expected / first attempt failed"; assess specific,
  internally consistent, connected, consequential. **Subject to §A: specificity ≠ credit until
  verified.**

---

# 5–9. Interrogatability, Substitutability, Origin Map, Reasoning, Epistemic (preserved)

- **§5 Interrogatability** — reframed by §A into a *teacher-probe question surface* for unverified
  specifics; an ownership indicator only for verified ones. Not an authorship score.
- **§6 Generic Substitutability** — "swap the nouns, does it still read?" Maps to existing
  `generic_assertion_risk` (`layer3_scoring.py`). Evidence of weak grounding, **not** proof of AI.
- **§7 Information-Origin Map** — per-claim origin taxonomy (common-knowledge / external /
  assignment / observation / original-analysis / dataset / unsupported / unknown). A Tier-1 signal
  (§D): needs LLM claim extraction.
- **§8 Reasoning Continuity** & **§9 Epistemic Development** — reasoning-dependency graph; revision
  is a *strength*, not a defect. **Both are LLM-judged → strictest confidence bands (§C)**: project
  memory records that LLM rating *over-judges fluent human text*, so these signals must never drive
  a headline and must carry Low/Moderate bands until calibrated (§B).

---

# 10. Evidence Levels 0–5 (preserved — now the Tier-0 output)

```text
0 No evidence   1 Statistical indication (DeBERTa/V7/generator classifiers)
2 Content weakness   3 Internal evidence   4 External corroboration   5 Process provenance
```
Report as `Authorship evidence level: 3/5` — more defensible than `Human: 92%`. **Phase 0 (§H)
produces this from signals that already exist**, no new detectors.

---

# B. Calibration Protocol (NEW — normative for every signal)

No signal SCORES (contributes to fusion or a user number) until it ships with **all five**:

1. **Labeled corpus + numbers-only committed baseline** — a versioned dataset and a baseline JSON
   holding only numbers, in the culture of `poc/detect_v7/weights.json` provenance notes and
   `poc/calibration/fpr_subgroup_baseline.json`.
2. **FPR-analog fairness gate.** For AI detection the harm is false-positive on ESL/human text
   (`poc/calibration/fpr_subgroup_gate.py`). For **grounding signals the harm is mislabeling
   genuinely-grounded work as thin** (the ESL-equivalent) — the gate must FAIL on a rise in that
   false-thin rate, by proficiency/subgroup, vs baseline.
3. **Explicit scoring-unit statement + eval↔prod unit-parity check.** State the unit (0–1 risk?
   0–100 mark? higher=better or worse?) in the module docstring AND assert the eval harness and the
   prod path use the identical unit. *Three unit-mismatch incidents in one week (2026-07-14) make
   this mandatory, not advisory.*
4. **No invented weights.** The v1 §11 fusion percentages (Grounding 30 / Context 20 / …) are
   marked **ILLUSTRATIVE, pending data-derived fusion** (see §11). A subagent fabricating category
   weights already happened once; weights come from labeled data or they do not ship.
5. **Coverage/limitations metadata** (see §C) travels with every emitted signal.

Reference gate command culture (CLAUDE.md): `python poc/calibration/fpr_subgroup_gate.py --compare`,
auto-enforced by the `pre-push` hook when `poc/detect/` is touched.

---

# C. Confidence & Coverage Metadata (NEW — cross-cutting contract on EVERY engine)

Every engine emits, alongside its value:

- **`detection_confidence`** — banded **High / Moderate / Low** with **enumerated reasons**.
  Explicitly **NOT a second percentage** (mirrors `poc/report/headline_confidence.py`, which bands
  the AI badge with reason codes and never adds a number). LLM-judged engines (§8/§9) are **capped
  at Moderate** and default Low until calibrated — the over-judgment memory demands it.
- **`coverage_window`** — e.g. `"known generators through <month year>"`, **derived from
  checkpoint/corpus provenance** (V7 checkpoint label / calibration corpus date), not hardcoded.
- **`limitations`** — enumerated codes (e.g. `no_assignment_context`, `unverified_specifics_present`,
  `short_paragraphs_low_confidence`, `outside_training_coverage`).

This is additive and must reach ALL surfaces (web + PDF + email + read-time API) per the
additive-composer rule (MEMORY). It never changes the score — annotate, don't suppress.

---

# D. Cost Tiers (NEW — maps engines to a call/latency budget)

| Tier | What runs | Cost / latency | When |
|---|---|---|---|
| **Tier-0 quick pass** | Existing deterministic signals (`grounding_diagnosis`, `generic_assertion_risk`, `citation` counts, `critical_thinking`, `submission_risk`, V7) recast into **Evidence Levels 0–2 + §C metadata**. | Near-free, no extra LLM. +ms. | **Every scan.** |
| **Tier-1 claim-graph audit** | LLM claim extraction + internal-consistency (§A) + interrogatability + substitutability + **information-origin map** (§7). | ~10–50 LLM calls, async. | Premium / opt-in. Mirror the existing **paid deep-scan** pattern (Modal endpoint). |
| **Tier-2 external corroboration** | Source retrieval + **claim–source entailment** (§4.1) via Crossref/DOI + Tavily. | Network + retrieval latency, rate-limited. | Premium, when citations present. |

Tier-1/2 reuse the deep-scan async+cache plumbing already in prod (`poc/detect_v7/modal_client.py`,
deep-scan heatmap). Budgets are per-tier caps, enforced like the existing deep-scan windowing.

---

# E. Product-Decision Registry (NEW — owner-owned, NOT assumed)

These are **deliberate product decisions requiring explicit owner sign-off**, not engineering
defaults. Recorded here so no phase silently assumes them.

1. **Assignment-brief / rubric upload** (enables §4.3 context grounding). **Distinct from** the
   standing **single-copy HARD NO** on earlier-draft / paired-document uploads (owner reiterated ×3;
   do NOT resurrect). A brief/rubric is task context, not a second copy of the student's work —
   but it is still an upload demand and needs sign-off. **Without it, §4.3 degrades to
   topic-self-consistency** — each phase (§H) must state which it gets.
2. **Provenance channel (Evidence Level 5).** Only via a **voluntary in-editor / extension draft
   timeline offered as a service to the student** — never an upload demand, never retroactive.
   Requires explicit owner sign-off before any build.
3. **Claim-graph persistence.** **Report-JSON first** (rides existing R2 report, no schema change),
   Postgres tables later (`documents/claims/evidence/claim_evidence_edges/assessment_signals` per
   v1 §13) — aligned with the **3-day retention** policy.

---

# F. Rewrite Synergy (NEW — closes the evidentiary loop)

The claim–evidence graph **IS the rewrite targeting map**. Ungrounded / unverified-specific /
generic claim nodes are precisely the fix-first targets the rewriter should surface and fill —
"make it yours". This connects directly to the **already-shipping verdict reframe**
(`docs/plans/rewrite_verdict_reframe_scope.md`): the reframe's `gap_resolution` fields
(`findings_resolved`, `grounding_risk_delta`, `anchors_added`) are the *downstream measurement* of
graph-node repair. Constraint from §A: the rewriter must fill gaps with **flagged, reviewable**
specifics the student replaces — it must NOT fabricate verified-looking specifics (that is what
produced the 2026-07-14 survey). The graph tells the rewriter *where* the gap is; §A keeps it honest
about *what kind* of fill is legitimate.

---

# G. Exists-vs-New Inventory (grepped, not guessed)

| v1 section | Existing DraftProof component | Gap to close |
|---|---|---|
| §2 Decomposition | `poc/detect/layer3_scoring.py` (sentence split, claim-ish units) | No typed **claim** objects / graph nodes — new (Tier-1). |
| §3 Claim–Evidence Graph | — | **Entirely new.** No graph today. `inconsistent_with` edge new (§A). |
| §4.1 External grounding | `poc/detect/citation.py` (in-text count, bib-decoupled gating) | No DOI/Crossref resolve, **no entailment model** — new (Tier-2). |
| §4.2 Internal grounding | `poc/detect/grounding_diagnosis.py`; `generic_assertion_risk`/`citation_grounding_risk` in `layer3_scoring.py`, `mitigation.py`, `transformation.py` | Rule-level exists; claim-typed expectation checks new. |
| §4.3 Context grounding | — | New + **§E1 product decision** (brief upload). |
| §4.4 / §5 Experiential & Interrogatability | `grounding_diagnosis.py` (authorship_trace bucket), `poc/report/authorship_evidence.py` (honest inventory, no verdict) | Present-as-credit today; **§A verification gate new**. |
| §6 Substitutability | `generic_assertion_risk` (`layer3_scoring.py`) | Signal exists; explicit noun-swap test new (Tier-1). |
| §7 Origin map | — | New (Tier-1 LLM). |
| §8/§9 Reasoning & Epistemic | `poc/detect/critical_thinking.py` (deterministic control dims) | Reasoning-**graph** + LLM revision detection new; strict bands (§C). |
| §10 Evidence Levels | Signals exist scattered | New **presentation** recast (Phase 0). |
| §11 Fusion | `poc/detect_v7/detector_fusion.py`, `weights.json`, tier authority | AI-side fusion exists; **authorship-evidence fusion new, weights ILLUSTRATIVE** (§B4). |
| §14 Report | `poc/report/render*.py`, `headline_confidence.py`, `submission_risk.py` | Confidence banding exists (badge); **evidence-level + coverage block new (§C)**. |
| ESL coexistence | `fpr_subgroup_gate.py`, always-fused tier authority | Principle live; extend fairness gate to grounding signals (§B2). |

---

# 11. Evidence Fusion (preserved — weights now ILLUSTRATIVE)

The v1 conceptual weighting (Grounding 30 / Context 20 / Reasoning 15 / Source 15 / Provenance 15 /
AI-pattern 5) is **marked ILLUSTRATIVE pending data-derived fusion (§B4)**. Do not ship these as
live weights. Start with **calibrated rule-based** fusion once per-signal calibration (§B) exists;
move to **Bayesian** fusion only after every input signal is individually calibrated. Avoid a single
magic number — prefer the banded "AUTHORSHIP CLARITY" panel (v1 §14).

---

# 14. Report Design (preserved + §C block)

```text
AUTHORSHIP CLARITY
────────────────────────────────
Content ownership evidence   Strong        Grounding             Strong
Source traceability          Moderate      Assignment specificity Strong
Reasoning development        Moderate      Process provenance    Not available
AI-pattern signal            Elevated      AI-pattern confidence  Low–Moderate

Detection confidence  Moderate — reasons: no assignment context; 2 unverified specifics
Coverage              Known generators through Jul 2026
Limitations           no_assignment_context; unverified_specifics_present
```

**An elevated AI signal must coexist with strong authorship evidence** (ESL / polished-human /
AI-mediated-language / heavily-edited / new-model-coverage protection). This v1 principle is
strengthened by §C: the coverage window makes "outside training coverage" an explicit, honest state
rather than a false-positive.

---

# 15–16. Evidence Hierarchy & Strategic Direction (preserved)

Hierarchy (bottom→top): TEXT PATTERNS → REASONING DEPTH → GROUNDING → VERIFIABILITY → PROVENANCE.
AI detection is the **bottom** layer, never the verdict. The three sustainable questions:

> What does the text statistically resemble? · How well is the work grounded in evidence, context,
> and particular reality? · What evidence exists that the author owns the intellectual process?

Positioning: **evidentiary, not accusatory** — help students *build* credible authorship. Long-term
frame: **Authorship Evidence Graph**, not "AI detector".

---

# H. Phased Roadmap (NEW — effort, kill-switches, §E dependencies)

Each phase is additive and kill-switched (project convention); no phase enables an §E decision
without owner sign-off.

| Phase | Deliverable | New signals? | §4.3 context | Effort | Kill-switch |
|---|---|---|---|---|---|
| **0** | Evidence-Level report (0–2) + §C confidence/coverage block, **recast from existing signals**. Display-layer only. | No | topic-self-consistency only | ~days | `DRAFTPROOF_AUTHORSHIP_EVIDENCE_LEVELS` |
| **1** | Claim extraction + linking + interrogatability + substitutability + origin map. **§B calibration required** before scoring. | Yes (Tier-1) | still self-consistency | ~2–3 wk | `DRAFTPROOF_CLAIM_GRAPH` |
| **2** | Claim–source **entailment** (§4.1) + **context grounding** (§4.3, needs §E1) + origin-map corroboration. | Yes (Tier-2) | full, if §E1 approved | ~3–4 wk | `DRAFTPROOF_ENTAILMENT` / `DRAFTPROOF_CONTEXT_GROUNDING` |
| **3** | Provenance (§E2, Level 5) + **data-derived fusion** (Bayesian only after per-signal calibration). | Fusion | — | ~ongoing | `DRAFTPROOF_AUTHORSHIP_FUSION` |

**Gate between every phase:** the §B five-part protocol must pass (labeled corpus, fairness gate,
unit-parity, no-invented-weights, coverage metadata) before a signal graduates from advisory
(annotate) to scoring (fusion input). Measure with the deterministic harness, never single runs.

---

# Final Principle (v1, preserved)

> Credible authorship assessment is **evidence-based, multi-layered, calibrated, and explicit about
> uncertainty.** Never confuse *AI-like style* with *lack of human authorship*. **And (v2): never
> credit an unverified specific — verify it, or turn it into a question.**
