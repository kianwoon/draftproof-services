# Review Comments on Credible Authorship Assessment — Technical Architecture v2

**Date:** 2026-07-15  
**Purpose:** Companion review note for `credible_authorship_assessment_v2.md`

---

# Overall Assessment

The v2 revision is materially stronger than the original architecture.

The biggest improvement is that it closes the most dangerous loophole in the first version:

> **Specificity is not evidence unless it is verified, traceable, or internally substantiated.**

Without this rule, the system could eventually reward exactly the behaviour DraftProof should prevent: adding plausible-looking details to make text appear more human.

The revised `verified or neutral` rule is the correct default.

The architecture now has five important properties:

```text
1. Detection is subordinate to evidence
2. Specificity is verification-gated
3. New signals cannot silently become scores
4. Uncertainty is part of the API contract
5. The roadmap starts by reusing existing signals before adding expensive intelligence
```

This is no longer just a better AI classifier.

It is becoming an **evidence system with explicit epistemic limits**.

---

# 1. §A Is the Most Important Improvement

The original architecture had a genuine exploitable weakness:

```text
specificity
    ↓
interrogatability
    ↓
authorship evidence
```

The revised architecture correctly changes this to:

```text
specificity
    │
    ▼
Can it be verified?
    │
 ┌──┴───────────────┐
 │                  │
YES                NO
 │                  │
 ▼                  ▼
Evidence          Neutral
                   +
              Teacher probe
```

The core invariant should remain:

> **An unverified specific is neutral, not positive and not negative.**

This avoids two opposite failures.

## Failure 1: Rewarding Fabrication

Example:

```text
"A 2025 industry survey found that 37%..."
```

Without evidence:

```text
specificity ↑
human-likeness ↑
score ↑
```

This is unsafe.

## Failure 2: Punishing Genuine but Unverifiable Experience

Example:

```text
"During our Tuesday practical session..."
```

The system may not have access to the class record.

Therefore:

```text
unverified ≠ fabricated
```

The neutral state is essential.

---

# 2. Make Verification Status a First-Class Claim State

Every claim node should explicitly carry a verification state.

Example:

```json
{
  "claim_id": "c_001",
  "verification_status": "verified | corroborated | internally_supported | unverified | contradicted | unverifiable",
  "verification_paths": [],
  "evidence_level": 0,
  "confidence": "low | moderate | high",
  "limitations": []
}
```

Recommended states:

## Verified

Direct external or contextual evidence supports the claim.

```text
claim
  ↓
source
  ↓
entailment confirmed
```

## Corroborated

Multiple independent pieces of evidence support the claim, but no authoritative verification exists.

## Internally Supported

The claim is consistently used and causally connected elsewhere in the document.

Important:

> **Internally supported does not mean factually true.**

It only means internally coherent.

## Unverified

Potentially verifiable, but the required evidence is unavailable.

## Unverifiable

The claim cannot realistically be checked from available evidence.

Example:

```text
"I felt nervous before the presentation."
```

## Contradicted

Available evidence conflicts with the claim.

This gives the Claim–Evidence Graph much more semantic precision.

---

# 3. Internal Consistency Should Not Independently Establish Strong Evidence

One architectural point should be tightened.

The current design allows:

```text
Internally consistent → Evidence Level 3 (internal)
```

This is understandable, but risky.

A fabricated story can be perfectly internally consistent.

An LLM can generate:

```text
Claim A
    ↓
Claim B
    ↓
Claim C
    ↓
Conclusion
```

with zero contradictions.

Therefore, separate:

```text
EVIDENCE STRENGTH
```

from:

```text
CONSISTENCY QUALITY
```

Recommended interpretation:

```text
External verified          Level 4
Context traceable          Level 3
Process provenance         Level 5

Internally consistent      supporting property only
```

Instead of:

```text
internally consistent = evidence level 3
```

Use:

```text
internally consistent
        ↓
eligible for internal-support credit
        ↓
cannot independently establish Level 3
```

A possible implementation is:

```text
Internal consistency multiplier:
0.0 – contradictory
0.5 – incomplete
1.0 – consistent
```

The exact values must not ship without calibration.

The important principle is that consistency should modify the reliability of other evidence rather than independently create strong authorship evidence.

---

# 4. Extend Claim–Evidence Edge Types

The current edge set is useful:

```text
supported_by
derived_from
observed_in
contradicts
inconsistent_with
```

Recommended additions:

```text
supports
derived_from
corroborated_by
qualified_by
revises
contradicts
inconsistent_with
depends_on
explains
causes
```

## `revises`

Example:

```text
Claim A:
"I initially expected X."

Claim B:
"The results instead showed Y."

Claim B --revises--> Claim A
```

This prevents legitimate epistemic development from triggering a fabrication tripwire.

## `qualified_by`

Example:

```text
"AI improves productivity."

qualified_by

"This effect was observed only for routine tasks."
```

Without this relationship, differently scoped statements may be misclassified as inconsistent.

## `depends_on`

Useful for reasoning continuity.

```text
Conclusion D
   depends_on
Finding C
   depends_on
Observation B
```

This allows the system to represent reasoning topology rather than only evidence linkage.

---

# 5. The Calibration Protocol Is a Major Strength

The five requirements are strong:

```text
labeled corpus
fairness gate
unit parity
no invented weights
coverage metadata
```

The key governance principle should be made explicit:

> **A signal is not automatically a score.**

Recommended lifecycle:

```text
EXPERIMENTAL
    ↓
ADVISORY
    ↓
SCORING
```

## Experimental

Internal only.

```text
No user-facing impact
No fusion contribution
```

## Advisory

May appear in reports as annotation.

Example:

```text
"2 claims require verification"
```

But:

```text
fusion_weight = 0
```

## Scoring

Only after the calibration protocol passes.

```text
fusion_weight > 0
```

Example metadata:

```json
{
  "signal": "unverified_specific_rate",
  "status": "advisory",
  "scoring_enabled": false,
  "calibration_version": "2026-07-15-a",
  "fairness_gate_passed": false
}
```

This creates a hard governance model and reduces the risk of future agents or developers accidentally promoting an unvalidated signal into scoring.

---

# 6. Use `assessment_confidence`, Not `detection_confidence`, as the Generic Contract

Every engine emitting:

```text
detection_confidence
coverage_window
limitations
```

is directionally correct.

However, not every engine is a detector.

A better generic field is:

```text
assessment_confidence
```

Example:

```json
{
  "engine": "citation_entailment",
  "value": "supported",
  "assessment_confidence": "high"
}
```

Versus:

```json
{
  "engine": "reasoning_continuity",
  "value": "moderate",
  "assessment_confidence": "low"
}
```

This allows the report composer to produce:

```text
overall evidence confidence
```

without pretending every subsystem performs detection.

---

# 7. Coverage Metadata Should Be Engine-Specific

For V7:

```text
known generators through July 2026
```

makes sense.

For citation entailment, the same temporal coverage model may not.

The universal contract should be:

```text
coverage metadata
```

but each engine should define its own coverage dimension.

Examples:

```json
{
  "coverage": {
    "type": "generator_window",
    "value": "through_2026_07"
  }
}
```

```json
{
  "coverage": {
    "type": "source_access",
    "value": [
      "doi",
      "crossref",
      "public_web"
    ]
  }
}
```

```json
{
  "coverage": {
    "type": "context_availability",
    "value": "assignment_brief_absent"
  }
}
```

This will scale better than forcing all engines into a single temporal coverage field.

---

# 8. The Cost-Tier Design Is Realistic

The tiers are well chosen.

```text
Tier 0
existing deterministic signals
every scan
near free
```

Phase 0 should start here.

The current system can already expose:

```text
Evidence Level
Grounding weaknesses
Generic assertion risk
Citation risk
Critical thinking indicators
V7 pattern signal
Confidence metadata
```

This allows the product narrative to shift before the expensive Claim–Evidence Graph architecture is complete.

That is strategically important.

---

# 9. Reduce Tier-1 LLM Call Count with Hierarchical Extraction

A design using:

```text
~10–50 independent LLM calls
```

could become expensive, slow, and inconsistent.

Prefer hierarchical extraction:

```text
Document
   ↓
Paragraph-level structured extraction
   ↓
Claim objects
   ↓
Document-level graph reconciliation
```

Example architecture:

```text
Stage 1
Paragraph claim extraction
10–20 parallel calls

Stage 2
Graph reconciliation
1–3 calls

Stage 3
Targeted ambiguity resolution
0–5 calls
```

A stronger optimisation is to batch short paragraphs:

```text
3–5 paragraphs per extraction call
```

Resulting in approximately:

```text
5–10 extraction calls
1 reconciliation call
targeted retries only
```

The LLM should propose:

```text
nodes
edges
claim types
```

The application should deterministically validate:

```text
IDs
references
sentence spans
edge legality
duplicates
cycles
```

The LLM should not own graph truth.

---

# 10. Claim Extraction Needs Deterministic Text-Span Anchoring

Every claim should point back to exact source text.

Example:

```json
{
  "claim_id": "c17",
  "text": "The intervention reduced processing time by 35%.",
  "source": {
    "paragraph_id": "p4",
    "sentence_id": "s12",
    "char_start": 432,
    "char_end": 487
  }
}
```

Core invariant:

> **No claim node without a deterministic text span.**

Otherwise an LLM-generated interpretation could be mistaken for a student-authored claim.

Derived analytical objects should use separate node types.

```text
CLAIM
must map to source text

INFERENCE
system-generated interpretation

QUESTION
system-generated teacher probe
```

These must never be mixed.

---

# 11. The Information-Origin Map Should Be Multi-Label

A claim may contain multiple origins.

Example:

> "Sweller's cognitive load theory explains why the students in my observation struggled when three instructions were presented simultaneously."

This includes:

```text
external theory
personal observation
original interpretation
```

Therefore:

```text
origin = one category
```

may be too restrictive.

A simple multi-label design is sufficient initially:

```json
{
  "origins": [
    "external_source",
    "personal_observation",
    "original_analysis"
  ],
  "primary_origin": "original_analysis"
}
```

Probabilities can be added later if calibration justifies them.

---

# 12. Keep the Assessment Graph and Rewrite Plan Logically Separate

The Claim–Evidence Graph can be the rewrite targeting source.

However, the architecture should preserve:

```text
Assessment Graph
```

and:

```text
Rewrite Plan
```

as separate objects.

Recommended flow:

```text
Claim–Evidence Graph
        ↓
Gap Detection
        ↓
Rewrite Target Generation
        ↓
Rewrite Plan
```

Do not allow:

```text
rewriter edits graph
```

Instead:

```text
rewriter produces candidate revision
        ↓
document rescanned
        ↓
new graph generated
```

This creates a clean before/after loop:

```text
BEFORE GRAPH
     ↓
rewrite
     ↓
AFTER GRAPH
     ↓
graph delta
```

Possible measurable deltas:

```text
unverified_specifics: 4 → 1
generic_claims:       8 → 3
grounded_claims:      7 → 12
```

This is cleaner and safer than allowing the rewrite subsystem to mutate evidence state directly.

---

# 13. The Product-Decision Registry Is Excellent

The distinction between:

```text
assignment brief upload
```

and:

```text
student earlier-draft upload
```

should remain explicit.

The recommendation to use:

```text
Report JSON first
Postgres persistence later
```

is also strong.

For Phase 1, the graph can live inside the report object:

```json
{
  "authorship_evidence": {
    "version": "1.0",
    "claims": [],
    "evidence": [],
    "edges": [],
    "signals": [],
    "assessment": {}
  }
}
```

After the schema stabilises:

```text
JSON
 ↓
normalised Postgres
```

rather than the reverse.

---

# 14. Add a New Hardening Section: Adversarial and Gaming Resistance

This is the main remaining architectural gap.

Once users or rewriters know the system rewards:

```text
verified grounding
claim connectivity
epistemic development
reasoning continuity
```

those signals will become optimisation targets.

Potential attacks include:

```text
fake citations
fabricated but internally consistent experiences
synthetic revision narratives
generated "I initially thought..." structures
citation stuffing
claim graph gaming
```

Recommended new section:

# I. Adversarial and Gaming Resistance

Core rule:

> **No single evidence pattern should independently establish strong authorship.**

Examples:

```text
Many citations
≠
source grounding

Many personal experiences
≠
ownership

Many revisions
≠
process provenance

High internal consistency
≠
truth

Low AI score
≠
human authorship
```

Use:

```text
cross-signal corroboration
```

rather than signal maximisation.

---

# 15. Recommended Final Evidence Model

Model three different concepts rather than one:

```text
AUTHORSHIP ASSESSMENT
│
├── Evidence Strength
│     How much supporting evidence exists?
│
├── Evidence Reliability
│     How trustworthy is that evidence?
│
└── Evidence Coverage
      How much of the document/process can we actually assess?
```

Example:

```text
Evidence strength      Strong
Evidence reliability   Moderate
Evidence coverage      Partial
```

This is more informative than:

```text
Authorship score: 82
```

For example:

```text
Strong evidence
+
Low coverage
```

means:

> What we can see is strong, but we cannot assess the full process.

Whereas:

```text
Moderate evidence
+
High coverage
```

means something different.

This distinction will become increasingly important as the architecture matures.

---

# Final Recommendations

The v2 architecture is ready to govern implementation.

Three changes are recommended before treating it as fully hardened:

1. **Do not allow internal consistency alone to establish Evidence Level 3.** Treat it as supporting evidence or a reliability modifier.
2. **Make claim verification states and deterministic text-span anchoring explicit in the data model.**
3. **Add an adversarial/gaming-resistance section**, because every successful authorship signal will eventually become a target for optimisation.

The overall direction is strong:

```text
AI detection
    ↓
one weak sensor

Claim–Evidence Graph
    ↓
structured assessment

Verification + context + provenance
    ↓
stronger evidence

Uncertainty + calibration
    ↓
credible output
```

The most important rule remains:

> **Never credit an unverified specific — verify it, or turn it into a question.**

That one rule prevents the architecture from accidentally turning DraftProof into a system that teaches students or rewriters how to fabricate human-looking evidence.
