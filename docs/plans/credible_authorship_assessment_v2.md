# Credible Authorship Assessment — Technical Architecture (v2.1)

> **Headline invariant (applies to the entire architecture):**
> **"Never credit an unverified specific — verify it, or turn it into a question."**

**Status:** Governing plan. **Date:** 2026-07-15. **Revision:** v2.1 — incorporates the accepted
2026-07-15 external review (`docs/plans/reviews/credible_authorship_v2_review_comments.md`); all 15
review points adopted, plus three owner-agreed refinements (flagged **[owner]** inline).
**Supersedes:** the owner v1 doc (`credible-authorship-assessment-technical-architecture.md`), which
remains the SPINE — this doc amends and *hardens* it, it does not replace it.
**Voice/constraints:** CLAUDE.md "Objective & Rewrite Philosophy" (annotate-don't-suppress; evidence
over verdicts); `docs/plans/rewrite_verdict_reframe_scope.md` (verdict reframe already scoped/shipping).

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

Sections 1–16 preserve v1's structure, diagrams, and ideas; the lettered sections (A–I) are the
hardening layer and are normative.

---

# 1–3. Architecture, Decomposition, Claim–Evidence Graph (preserved)

The v1 target architecture stands (SUBMITTED CONTENT → Normalisation → Text Forensics / Grounding /
Source engines → **Claim/Evidence Graph** → Authorship / Integrity / AI-pattern evidence → Fusion →
Assessment). The AI detector never generates the verdict.

Document decomposition (v1 §2) into paragraphs/sentences/claims/citations/quotations/entities/
numbers/dates/personal- and assignment-references is the foundation. Per claim, extract: claim type
(factual/analytical/personal-observation/interpretation/conclusion), specificity, evidence
requirement, citation support, source relationship, contextual uniqueness, verifiability.

## §2 data-model invariants (NEW — review §10)

- **No claim node without a deterministic text span.** Every CLAIM node MUST carry
  `{paragraph_id, sentence_id, char_start, char_end}` pointing back to exact student source text.
  Without this, an LLM-generated interpretation could be mistaken for a student-authored claim.
- **Three separate node types — never mixed:**
  - `CLAIM` — student text; must map to a source span.
  - `INFERENCE` — system-generated interpretation of the text.
  - `QUESTION` — system-generated teacher probe.

## §2 claim verification status (NEW — review §2)

Each claim node carries a **first-class `verification_status`** and its `verification_paths[]`:

```json
{
  "claim_id": "c_001",
  "text": "The intervention reduced processing time by 35%.",
  "source": {"paragraph_id": "p4", "sentence_id": "s12", "char_start": 432, "char_end": 487},
  "verification_status": "verified | corroborated | internally_supported | unverified | unverifiable | contradicted",
  "verification_paths": [],
  "evidence_level": 0,
  "assessment_confidence": "low | moderate | high",
  "limitations": []
}
```

| Status | Meaning |
|---|---|
| **verified** | Direct external or contextual evidence supports the claim (claim→source→entailment confirmed). |
| **corroborated** | Multiple independent pieces support it, but no authoritative verification exists. |
| **internally_supported** | Consistently used / causally connected elsewhere. **Does NOT mean factually true — only internally coherent.** |
| **unverified** | Potentially verifiable, but the required evidence is unavailable. |
| **unverifiable** | Cannot realistically be checked from available evidence (e.g. *"I felt nervous before the presentation."*). |
| **contradicted** | Available evidence conflicts with the claim. |

**Fairness note — `unverified` ≠ `unverifiable` ≠ `contradicted`.** These are three different
states and MUST NOT collapse into one. The genuine-but-uncheckable case (*"During our Tuesday
practical session…"* — the system has no class record) is **`unverifiable`, which stays NEUTRAL**,
never penalised. Only `contradicted` is negative. `unverified ≠ fabricated`.

## §3 the Claim–Evidence Graph (v1 spine)

The **Claim–Evidence Graph** (v1 §3) is the spine. The metric is **claim-to-evidence connectivity**,
not reference count. The full edge set (§4) is normative. Outputs (grounded / unsupported-factual /
generic / **unverifiable-specific**) are per-document and must be **anchored to the exact claim
node** (precision-first) — never "the document is generic".

---

# A. Hallucinated-Grounding Countermeasures (NEW — the v1 blind spot)

**Rule: specificity and interrogatability CREDIT authorship only when the specific is verifiable or
verified. An unverified specific is NEUTRAL, and surfaces as a teacher-probe QUESTION, not as
evidence.** This inverts v1 §4.4/§5, which implicitly credited "grounded specificity" by its presence.

A specific earns credit on any one of three verification paths (evidence levels per §3-below):

| Path | Mechanism | Verification status | Evidence contribution |
|---|---|---|---|
| **Source-entailed** | §4.1 citation→source→**entailment** confirms the source supports the specific. Entailment is **load-bearing** — a resolvable DOI is not enough; the entailment model must fire. | verified | Level 4 (external) |
| **Context-traceable** | Traces to uploaded assignment/rubric context (§4.3). Requires the §E1 product decision. | verified | Level 3 (context) |
| **Internally consistent** | Participates in a consistent sub-graph (referenced again, causally consumed, not contradicted). | internally_supported | **Supporting property only — see §3-below; cannot independently reach Level 3.** |

A specific that satisfies none scores **NEUTRAL** (never negative — absence of proof is not proof of
fabrication) and is emitted as an **interrogatability question** ("Which survey? What sample? Cited
where?"). Interrogatability (§5) is thus reframed: high interrogatability of an *unverified* claim is
a **question count** / audit surface for a teacher — NOT an ownership credit. Only verified specifics
raise the ownership indicator.

**Fabrication tripwire.** The `inconsistent_with` edges (§4) are the detector. When two claims assert
incompatible specifics (a figure stated two ways; an experience that contradicts a later dependency),
raise a **contradiction flag** on both nodes and cap their grounding contribution at NEUTRAL.
Contradiction is treated per v1 §9 (epistemic development can *legitimately* revise a position) — so
the tripwire fires only on **unsignposted** inconsistency, distinguished by the `revises` /
`qualified_by` edges (§4). This §A↔§4↔§9 interaction is a REQUIRED fix, not optional.

**Motivation, made concrete.** An architecture that rewards unverified specificity does not just
mis-score submissions — via the shared claim graph (§F) it would train the *rewriter* to add
plausible-but-fake specifics to lift a score. The 2026-07-14 fabricated-survey output is exactly that
failure mode. Verification-gating specificity is a safety requirement, not a nicety.

## §3-below. Internal consistency is a RELIABILITY MODIFIER (NEW — review §3)

Internal consistency must **never independently establish strong evidence.** A fabricated story can
be perfectly internally consistent; an LLM can emit `Claim A → B → C → Conclusion` with zero
contradictions. Therefore **evidence strength** and **consistency quality** are separated:

```text
External verified      Evidence Level 4
Context traceable      Evidence Level 3
Process provenance     Evidence Level 5
Internal consistency   supporting property only — cannot independently establish Level 3
```

Consistency instead makes a claim *eligible for internal-support credit* and **modifies the
reliability** of other evidence. Illustrative implementation (values **ILLUSTRATIVE, pending
calibration — §B applies**; must not ship uncalibrated):

```text
Internal consistency multiplier (ILLUSTRATIVE):
0.0 – contradictory
0.5 – incomplete
1.0 – consistent
```

---

# 4. Claim–Evidence Edge Set (preserved + extended — review §4)

The metric is claim-to-evidence connectivity. Full normative edge set:

```text
supports          derived_from      corroborated_by   qualified_by
revises           contradicts       inconsistent_with depends_on
explains          causes            observed_in       supported_by
```

Rationale for the review-added edges:

- **`revises`** — `Claim B --revises--> Claim A` ("I initially expected X" → "results instead showed
  Y"). Prevents legitimate **epistemic development** (§9) from tripping the §A fabrication tripwire.
- **`qualified_by`** — "AI improves productivity" *qualified_by* "only for routine tasks". Prevents
  differently-**scoped** statements from being misclassified as `inconsistent_with`.
  Together, `revises` + `qualified_by` are the **REQUIRED interaction fix between §A and §9**: without
  them, genuine revision and scoping trip the fabrication tripwire.
- **`depends_on`** — `Conclusion D → Finding C → Observation B`; represents reasoning topology (§8),
  not just evidence linkage.
- **`corroborated_by`**, **`explains`**, **`causes`** — richer semantic linkage for the `corroborated`
  and `internally_supported` verification states.
- **`inconsistent_with`** (v2 §A) — the fabrication tripwire; **`contradicts`** (v1) — asserted
  conflict.

Every edge is proposed by the LLM and **deterministically validated** by the application (§D): legal
edge type, both endpoints exist, no illegal cycles.

---

# 4.x Grounding Assessment (preserved; §A gates the credit)

- **4.1 External** — citation parser → DOI/Crossref/URL resolver → source retrieval →
  **claim–source entailment**. v2 elevates entailment from "possible flow" to the required credit
  gate for external specifics (§A). Seed: `poc/detect/citation.py` (in-text count/gating, bib-decoupled).
- **4.2 Internal** — measurement→evidence, survey→sample/method, experiment→method/result,
  comparison→basis. Seed: `generic_assertion_risk`/`citation_grounding_risk` in
  `poc/detect/layer3_scoring.py`; grounding buckets in `poc/detect/grounding_diagnosis.py`.
- **4.3 Context** — reverse-RAG against assignment brief / rubric / lecture / materials. **Requires
  §E1.** Without an uploaded brief this degrades to *topic self-consistency only* — state that
  honestly per phase (§H), never present topic-consistency as context grounding.
- **4.4 Experiential** — extract "I observed / I expected / first attempt failed"; assess specific,
  internally consistent, connected, consequential. **Subject to §A: specificity ≠ credit until verified.**

---

# 5–9. Interrogatability, Substitutability, Origin Map, Reasoning, Epistemic (preserved)

- **§5 Interrogatability** — reframed by §A into a *teacher-probe question surface* for unverified
  specifics; an ownership indicator only for verified ones. Not an authorship score.
- **§6 Generic Substitutability** — "swap the nouns, does it still read?" Maps to existing
  `generic_assertion_risk` (`layer3_scoring.py`). Evidence of weak grounding, **not** proof of AI.
- **§7 Information-Origin Map** — per-claim origin taxonomy. **Multi-label (review §11):** a claim may
  carry several origins with one primary; probabilities deferred until calibration justifies them.

  ```json
  { "origins": ["external_source", "personal_observation", "original_analysis"],
    "primary_origin": "original_analysis" }
  ```
  (e.g. "Sweller's cognitive load theory explains why the students in my observation struggled…" =
  external theory + personal observation + original interpretation.) A Tier-1 signal (§D).
- **§8 Reasoning Continuity** & **§9 Epistemic Development** — reasoning-dependency graph (`depends_on`
  edges); revision (`revises` edges) is a *strength*, not a defect. **Both are LLM-judged → strictest
  confidence bands (§C)**: project memory records LLM rating *over-judges fluent human text*, so these
  signals must never drive a headline and carry Low/Moderate bands until calibrated (§B).

---

# 10. Evidence Levels 0–5 (preserved — the Tier-0 output)

```text
0 No evidence   1 Statistical indication (DeBERTa/V7/generator classifiers)
2 Content weakness   3 Internal/context evidence   4 External corroboration   5 Process provenance
```
Report as `Authorship evidence level: 3/5` — more defensible than `Human: 92%`. **Phase 0 (§H)
produces this from signals that already exist**, no new detectors. Note the §3-below correction:
**internal consistency alone cannot reach Level 3** — context-traceable evidence can.

---

# B. Calibration Protocol (NEW — normative for every signal)

No signal SCORES (contributes to fusion or a user number) until it ships with **all five**:

1. **Labeled corpus + numbers-only committed baseline** — versioned dataset + baseline JSON of only
   numbers, in the culture of `poc/detect_v7/weights.json` provenance notes and
   `poc/calibration/fpr_subgroup_baseline.json`.
2. **FPR-analog fairness gate.** For AI detection the harm is false-positive on ESL/human text
   (`poc/calibration/fpr_subgroup_gate.py`). For **grounding signals the harm is mislabeling
   genuinely-grounded work as thin** — the gate must FAIL on a rise in that false-thin rate, by
   proficiency/subgroup, vs baseline.
3. **Explicit scoring-unit statement + eval↔prod unit-parity check.** State the unit (0–1 risk? 0–100
   mark? higher=better or worse?) in the module docstring AND assert the eval harness and prod path
   use the identical unit. *Three unit-mismatch incidents in one week (2026-07-14) make this mandatory.*
4. **No invented weights.** The v1 §11 fusion percentages are **ILLUSTRATIVE, pending data-derived
   fusion** (§11). A subagent fabricating category weights already happened; weights come from labeled
   data or they do not ship.
5. **Coverage/limitations metadata** (see §C) travels with every emitted signal.

## Signal lifecycle (NEW — review §5)

**A signal is not automatically a score.** Every signal moves through a governed lifecycle:

```text
EXPERIMENTAL  →  ADVISORY  →  SCORING
```

| Stage | User impact | Fusion |
|---|---|---|
| **EXPERIMENTAL** | Internal only, no user-facing impact. | none |
| **ADVISORY** | May appear in reports as **annotation** ("2 claims require verification"). | `fusion_weight = 0` |
| **SCORING** | Only after the §B protocol passes. | `fusion_weight > 0` |

Machine-readable metadata on every signal:

```json
{ "signal": "unverified_specific_rate", "status": "advisory",
  "scoring_enabled": false, "calibration_version": "2026-07-15-a", "fairness_gate_passed": false }
```

**[owner] refinement — the ADVISORY → SCORING promotion is double-gated.** It requires BOTH (a)
explicit owner sign-off AND (b) a **mechanical CI guard**: a test asserting
`scoring_enabled ⇒ (fairness_gate_passed && calibration_version present)`. Norms alone failed this
project before (the fabricated-weights incident); schema rules do not enforce themselves — the CI
guard does. Reference gate culture (CLAUDE.md): `python poc/calibration/fpr_subgroup_gate.py
--compare`, auto-enforced by the `pre-push` hook when `poc/detect/` is touched.

---

# C. Confidence & Coverage Metadata (NEW — cross-cutting contract on EVERY engine)

Every engine emits, alongside its value:

- **`assessment_confidence`** (NEW name — review §6) — banded **High / Moderate / Low** with
  **enumerated reasons**. Explicitly **NOT a second percentage** (mirrors
  `poc/report/headline_confidence.py`). Renamed from `detection_confidence` because **not every engine
  is a detector** (citation-entailment, reasoning-continuity, etc.); the generic contract is
  `assessment_confidence`, and the report composer aggregates it into *overall evidence confidence*.
  The AI detector keeps `detection_confidence` as its **specific instance** of this generic field.
  LLM-judged engines (§8/§9) are **capped at Moderate** and default Low until calibrated.
- **`coverage`** (NOW a TYPED object — review §7) — coverage is engine-specific, not a single temporal
  field. Each engine declares its own dimension:

  ```json
  { "coverage": { "type": "generator_window",     "value": "through_2026_07" } }
  { "coverage": { "type": "source_access",        "value": ["doi", "crossref", "public_web"] } }
  { "coverage": { "type": "context_availability", "value": "assignment_brief_absent" } }
  ```
  `generator_window` derives from checkpoint/corpus provenance (V7 checkpoint label / corpus date),
  not hardcoded.
- **`limitations`** — enumerated codes (`no_assignment_context`, `unverified_specifics_present`,
  `short_paragraphs_low_confidence`, `outside_training_coverage`).

Additive; must reach ALL surfaces (web + PDF + email + read-time API) per the additive-composer rule
(MEMORY). It never changes the score — annotate, don't suppress.

---

# D. Cost Tiers + Tier-1 Extraction (NEW — review §8/§9)

| Tier | What runs | Cost / latency | When |
|---|---|---|---|
| **Tier-0 quick pass** | Existing deterministic signals (`grounding_diagnosis`, `generic_assertion_risk`, `citation` counts, `critical_thinking`, `submission_risk`, V7) recast into **Evidence Levels 0–2 + §C metadata**. | Near-free, no extra LLM. +ms. | **Every scan.** |
| **Tier-1 claim-graph audit** | Hierarchical claim extraction + internal-consistency (§A) + interrogatability + substitutability + **origin map** (§7). | See below. | Premium / opt-in. Mirror the existing **paid deep-scan** (Modal). |
| **Tier-2 external corroboration** | Source retrieval + **claim–source entailment** (§4.1) via Crossref/DOI + Tavily. | Network + retrieval latency, rate-limited. | Premium, when citations present. |

## Tier-1 hierarchical extraction (NEW — review §9)

The naive "~10–50 independent LLM calls" design is expensive, slow, and inconsistent. Use a
**hierarchical, batched** design instead:

```text
Stage 1  Paragraph-batch extraction   3–5 paragraphs per call   → ~5–10 calls
Stage 2  Graph reconciliation         1–3 calls
Stage 3  Targeted ambiguity retries   0–5 calls
```

**The LLM proposes** nodes / edges / claim types. **The application deterministically validates**:
IDs, references, sentence spans, edge legality, duplicates, cycles. **The LLM does not own graph
truth.**

**[owner] refinement — reconciliation OWNS cross-batch edge discovery.** Batched paragraph extraction
structurally *misses cross-paragraph claims* (a claim in p2 that `revises` one in p7). Stage 2 is not
merely dedup — it **explicitly owns discovery of edges that span batches**. Without this, the batching
optimisation silently drops the very reasoning/revision topology §8/§9/§A depend on.

Tier-1/2 reuse the deep-scan async+cache plumbing already in prod (`poc/detect_v7/modal_client.py`,
deep-scan heatmap). Budgets are per-tier caps, enforced like existing deep-scan windowing.

---

# E. Product-Decision Registry (NEW — owner-owned, NOT assumed)

Deliberate product decisions requiring explicit owner sign-off, not engineering defaults:

1. **Assignment-brief / rubric upload** (enables §4.3 context grounding). **Distinct from** the
   standing **single-copy HARD NO** on earlier-draft / paired-document uploads (owner reiterated ×3;
   do NOT resurrect). A brief/rubric is task context, not a second copy of the student's work — but it
   is still an upload demand and needs sign-off. **Without it, §4.3 degrades to topic-self-consistency**
   — each phase (§H) must state which it gets.
2. **Provenance channel (Evidence Level 5).** Only via a **voluntary in-editor / extension draft
   timeline offered as a service to the student** — never an upload demand, never retroactive.
   Requires explicit owner sign-off before any build.
3. **Claim-graph persistence.** **Report-JSON first** (rides existing R2 report, no schema change),
   Postgres tables later (`documents/claims/evidence/claim_evidence_edges/assessment_signals` per v1
   §13) — aligned with **3-day retention**. JSON → normalised Postgres, **never the reverse**.

---

# F. Rewrite Synergy — graph and rewrite plan stay SEPARATE (NEW — review §12)

The claim–evidence graph **IS the rewrite targeting source** — but the **assessment graph and the
rewrite plan are separate objects, and the rewriter never mutates the graph.** Required flow:

```text
Claim–Evidence Graph → Gap Detection → Rewrite Targets → Rewrite → RESCAN → after-graph → graph delta
```

The rewriter produces a candidate revision; the document is **rescanned** to produce a fresh
after-graph. It does NOT edit evidence state directly. The **graph delta** is the measurable outcome:

```text
unverified_specifics: 4 → 1     generic_claims: 8 → 3     grounded_claims: 7 → 12
```

This graph delta is the **v2 of the shipped `gap_resolution` metric** in the already-shipping verdict
reframe (`docs/plans/rewrite_verdict_reframe_scope.md`: `findings_resolved`, `grounding_risk_delta`,
`anchors_added`). Constraint from §A: the rewriter must fill gaps with **flagged, reviewable**
specifics the student replaces — it must NOT fabricate verified-looking specifics (that is what
produced the 2026-07-14 survey).

---

# G. Exists-vs-New Inventory (grepped, not guessed)

| v1 section | Existing DraftProof component | Gap to close |
|---|---|---|
| §2 Decomposition | `poc/detect/layer3_scoring.py` (sentence split, claim-ish units) | No typed CLAIM/INFERENCE/QUESTION nodes, no text-span invariant — new (Tier-1). |
| §3 Claim–Evidence Graph | — | **Entirely new.** Extended edge set + `verification_status` new (§2/§4). |
| §4.1 External grounding | `poc/detect/citation.py` (in-text count, bib-decoupled) | No DOI/Crossref resolve, **no entailment model** — new (Tier-2). |
| §4.2 Internal grounding | `grounding_diagnosis.py`; `generic_assertion_risk`/`citation_grounding_risk` | Rule-level exists; claim-typed expectation checks new. |
| §4.3 Context grounding | — | New + **§E1 product decision** (brief upload). |
| §4.4 / §5 Experiential & Interrogatability | `grounding_diagnosis.py` (authorship_trace), `poc/report/authorship_evidence.py` | Present-as-credit today; **§A verification gate new**. |
| §6 Substitutability | `generic_assertion_risk` | Signal exists; explicit noun-swap test new (Tier-1). |
| §7 Origin map | — | New (Tier-1 LLM); **multi-label** design new. |
| §8/§9 Reasoning & Epistemic | `poc/detect/critical_thinking.py` | Reasoning-**graph** (`depends_on`/`revises`) + LLM revision detection new; strict bands (§C). |
| §10 Evidence Levels | Signals exist scattered | New **presentation** recast (Phase 0). |
| §11 Fusion | `poc/detect_v7/detector_fusion.py`, `weights.json`, tier authority | AI-side fusion exists; **authorship-evidence fusion new, weights ILLUSTRATIVE** (§B4). |
| §14 Report | `poc/report/render*.py`, `headline_confidence.py`, `submission_risk.py` | Banding exists (badge); **three-axis + coverage block new (§C, final model)**. |
| ESL coexistence | `fpr_subgroup_gate.py`, always-fused tier authority | Principle live; extend fairness gate to grounding signals (§B2). |

---

# 11. Evidence Fusion (preserved — weights now ILLUSTRATIVE)

The v1 conceptual weighting (Grounding 30 / Context 20 / Reasoning 15 / Source 15 / Provenance 15 /
AI-pattern 5) is **marked ILLUSTRATIVE pending data-derived fusion (§B4)**. Do not ship these as live
weights. Start with **calibrated rule-based** fusion once per-signal calibration (§B) exists; move to
**Bayesian** fusion only after every input signal is individually calibrated. Prefer the banded
three-axis panel (final model) over a single magic number.

---

# 14. Report Design (preserved + §C block)

```text
AUTHORSHIP CLARITY
────────────────────────────────
Content ownership evidence   Strong        Grounding             Strong
Source traceability          Moderate      Assignment specificity Strong
Reasoning development        Moderate      Process provenance    Not available
AI-pattern signal            Elevated      AI-pattern confidence  Low–Moderate

Assessment confidence  Moderate — reasons: no assignment context; 2 unverified specifics
Coverage               generator_window: through Jul 2026; context: assignment_brief_absent
Limitations            no_assignment_context; unverified_specifics_present
```

**An elevated AI signal must coexist with strong authorship evidence** (ESL / polished-human /
AI-mediated-language / heavily-edited / new-model-coverage protection). Strengthened by §C: the
coverage window makes "outside training coverage" an explicit, honest state rather than a false-positive.

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

Each phase is additive and kill-switched; no phase enables an §E decision without owner sign-off.

| Phase | Deliverable | New signals? | §4.3 context | Effort | Kill-switch |
|---|---|---|---|---|---|
| **0** | Evidence-Level report (0–2) + §C confidence/coverage block, **recast from existing signals**. Display-layer only. | No | topic-self-consistency only | ~days | `DRAFTPROOF_AUTHORSHIP_EVIDENCE_LEVELS` |
| **1** | Hierarchical claim extraction + linking + interrogatability + substitutability + origin map. **§B calibration required** before scoring. | Yes (Tier-1) | still self-consistency | ~2–3 wk | `DRAFTPROOF_CLAIM_GRAPH` |
| **2** | Claim–source **entailment** (§4.1) + **context grounding** (§4.3, needs §E1) + origin-map corroboration. | Yes (Tier-2) | full, if §E1 approved | ~3–4 wk | `DRAFTPROOF_ENTAILMENT` / `DRAFTPROOF_CONTEXT_GROUNDING` |
| **3** | Provenance (§E2, Level 5) + **data-derived fusion** (Bayesian only after per-signal calibration). | Fusion | — | ~ongoing | `DRAFTPROOF_AUTHORSHIP_FUSION` |

**Gate between every phase:** the §B five-part protocol + signal-lifecycle promotion (EXPERIMENTAL →
ADVISORY → SCORING, double-gated) must pass before a signal graduates to scoring. Measure with the
deterministic harness, never single runs.

---

# I. Adversarial & Gaming Resistance (NEW — review §14)

Every load-bearing signal becomes an **optimization target the moment it ships.** Once users or
rewriters know the system rewards verified grounding, claim connectivity, epistemic development, or
reasoning continuity, expect: fake citations, fabricated-but-internally-consistent experiences,
synthetic revision narratives, generated "I initially thought…" structures, citation stuffing, and
claim-graph gaming.

**Core rule:**

> **No single evidence pattern independently establishes strong authorship.**

The `≠` list (each pattern is necessary-not-sufficient):

```text
Many citations           ≠  source grounding
Many personal experiences ≠  ownership
Many revisions           ≠  process provenance
High internal consistency ≠  truth
Low AI score             ≠  human authorship
```

Design for **cross-signal corroboration**, never signal maximisation. A high value on any one axis,
uncorroborated by an independent class of evidence, is treated as an audit surface — not a credit.

---

# Final Evidence Model — three axes, not one score (NEW — review §15)

The output models **three distinct concepts**, never a single collapsed number:

```text
AUTHORSHIP ASSESSMENT
├── Evidence Strength     How much supporting evidence exists?
├── Evidence Reliability  How trustworthy is that evidence?  (§3-below consistency modifier feeds here)
└── Evidence Coverage     How much of the document/process can we actually assess?  (§C coverage feeds here)
```

Worked interpretation:

```text
Evidence strength   Strong        →  "Strong evidence + Low coverage" means:
Evidence reliability Moderate          what we CAN see is strong, but we cannot assess the
Evidence coverage   Partial/Low        full process — NOT the same as strong-and-complete.
```

"Strong + low coverage" and "moderate + high coverage" mean genuinely different things; a single
`Authorship score: 82` erases that distinction. This three-axis frame becomes more important as the
architecture matures.

---

# Final Principle (v1, preserved)

> Credible authorship assessment is **evidence-based, multi-layered, calibrated, and explicit about
> uncertainty.** Never confuse *AI-like style* with *lack of human authorship*. **And (v2.1): never
> credit an unverified specific — verify it, or turn it into a question.**
