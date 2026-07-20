# Policy-Risk External Review — Advisor Response + Architecture

**Date:** 2026-07-20 · **Advisor:** Fable (consult session, branch `kianwoon/authorship-risk-architecture-801b31`)
**Subject:** External evaluation proposing to replace `poc/detect/policy_risk.py` with
`policy_risk = evidence_risk x policy_sensitivity_multiplier` + a 4-layer rearchitecture.
**Rev 2 (same day):** reviewer follow-up accepted the multiplier rejection and Phases 0–2; its
refinements (floor semantics wording, `editing_only` note, naming note, new Phase 4) are folded
in below — see "Reviewer follow-up" at the end of Part 1.
**Rev 3 (same day) — REVIEWER CLOSE-OUT, DEBATE CLOSED:** the reviewer's explicit FINAL comment
approves the plan ("architecturally sound enough to stop debating the scoring model and move into
implementation"). **Scoring architecture is FROZEN for V1** — `policy_risk_v1` is not to be
changed unless implementation testing exposes an actual bug; the multiplier rejection is not to be
revisited before Phase 4. Phase 0 is settled as written. Status: **READY FOR IMPLEMENTATION,
starting Phase 0/1.** The final comment's five refinements (all presentation/scoping, zero scoring
changes) are folded in below, marked *(Rev 3)*: Phase 1 assignment-level framing; Phase 2
`editing_only` explanatory copy; Phase 3 three-input principle; Phase 4 lens-vs-educator
disagreement metric; JSON-shape migration option under the naming note.

**Verdict: diagnosis correct (and already documented in-code); prescription rejected for the
scoring layer; the valuable parts are PRODUCT FEATURES with unbuilt prerequisites, phased below.**

---

## Part 1 — Assessment of the critique

### Ground truth established by code sweep (decisive)

- **No policy-context input exists anywhere in the product.** `ScanJob`
  (`draftproof-api/app/models/db.py`) has no policy/institution/assignment column; `ScanRequest`
  is document_id+text only; no migration, no frontend selector. Confirmed independently — and it
  matches the 2026-07-18 8-dim gap assessment ("policy-RAG = zero implementation").
- **No declaration capture.** `DeclarationGenerator.jsx` generates copy-paste template text
  client-side; nothing reaches the backend. `submission_risk.py` hardcodes declaration =
  "unknown — self-declare".
- **`PolicyRiskView.jsx` renders both lenses side-by-side** as explicitly hypothetical readings
  with confirm-yourself checkboxes. The module `_NOTE` says exactly this: "estimate how the
  draft may read under different school policies."

Therefore the reviewer's **Layer 3 (Policy Context) has zero data to condition on today**. The
4-layer rearchitecture smuggles a major unbuilt product feature (policy capture + declaration
capture) into what reads as a scoring-formula critique. Layers 1–2 already exist (category
breakdown; the grounding/critical-thinking/submission composers).

### Section-by-section

- **§1 "design smell".** Same diagnosis this session already made and the code already documents
  (`_floor_restricted_to_allowed` docstring cites the exact live inversion, 36.2 vs 33.39). Agreed
  the inversion is organic. But "two different assessments" is *intentional*: the lenses answer
  different questions (module docstring), and each produces its **own `main_issue`** — Allowed can
  say "grounding" while Restricted says "surface AI patterning". That is a diagnostic feature.

- **§2/§6 multiplicative fix — REJECTED for scoring.**
  1. `allowed_policy_multiplier` / `restricted_policy_multiplier` have **no derivation source**.
     There is no labeled policy-conditional-outcome data in the repo and none collectible; the
     multipliers would be hardcoded constants in costume — the exact fabricated-number problem
     flagged earlier today, relocated from weights (which ARE an owner-documented, test-matched
     spec: `policy_risk.py:30-52`, `test_spec_worked_example_renormalised`) to numbers with no
     spec at all. Collides with the repo's no-hardcode gate.
  2. A single `evidence_risk` scaled two ways makes both scores the **same shape** — per-lens
     `main_issue` collapses to one argmax. Real information loss.
  3. The one genuine architectural win the reviewer wants — ordering by construction — **already
     exists**: the shipped floor is exactly `restricted = max(allowed_reading, restricted_reading)`.
     Semantics (wording revised per the 2026-07-20 follow-up): the floored Restricted score is the
     **worst applicable reading across the two diagnostic lenses** — NOT literally "risk under a
     restrictive policy". The earlier framing ("a stricter policy inherits every permissive-policy
     concern, plus more") is a defensible product-semantic choice, not a proven scoring invariant:
     a floored-high case driven by permissive-lens signals (e.g. very weak grounding with low
     surface-AI + low voice gap) is mathematically constructible, and whether educators read that
     as restricted-policy risk must be **empirically validated later (Phase 4)**, not asserted.
     What holds by construction is only the display *ordering* (Restricted >= Allowed). When the
     floor fires, nothing is hidden: `pre_floor_score` / `floored_to_ai_allowed` preserve the
     organic disagreement and `main_issue` still names the organic driver (the reviewer's §6
     "hides disagreement" claim is factually wrong against the shipped code — the follow-up's
     floored-high scenario is consistent with, and fully annotated by, these fields). The observed
     live divergence was 2.8 points; large divergences are constructible in principle but their
     real-world frequency is a Phase-4 question, not a settled fact.

- **§3 "permitted use should lower Allowed risk" — misreads the module's claim.** `ai_allowed`
  is "does this look like *acceptable* AI-assisted work?" — grounding/judgment/specificity ARE
  the definition of acceptability under permissive policies (schools that allow AI still require
  the student to control and ground the work). The reviewer's scenario (explicit permission +
  correct declaration) requires policy + declaration data the product does not capture. The
  distinction the reviewer wants (Authorship Risk != Policy Risk) becomes valid **only after**
  Phase 1 below exists; then it is handled by *presentation selection*, not by re-scoring.

- **§4 conditional-risk framing — ALREADY IMPLEMENTED.** "Current risk: Moderate / if declared:
  Low" is precisely `confirm_delta` + `confirm_level`, rendered as confirm-yourself checkboxes in
  `PolicyRiskView.jsx`. Nothing to build; at most a copy tweak.

- **§5 `declaration_consistency` (3-state) — the best idea in the document**, and genuinely
  differentiating ("is the declared AI use consistent with the apparent involvement?"). But it is
  a product feature: needs a student-facing declaration input, API field, DB column, and a
  consistency comparison. Not buildable from data captured today. Compatible with the owner's
  single-copy constraint (a self-declaration is not an earlier-draft upload) but is new product
  surface → owner sign-off required.

### Bottom line

Keep `policy_risk_v1` scoring exactly as shipped (spec'd, tested, honest, traced). Do **not**
adopt the multiplicative model. Build the missing *policy-context data layer* the critique
presupposes; once policy is known, the "policy layer modifies the consequence of the evidence"
principle is satisfied by **choosing which existing lens to headline**, with zero new numbers.

### Reviewer follow-up (2026-07-20) — refinements accepted

The reviewer's revision review agrees with the multiplier rejection and approves Phases 0–2.
Four refinements accepted and folded into this document:

1. **Floor semantics softened** (see §2/§6 item 3 above and Phase 0): the floor stays for V1,
   but is documented as "worst applicable reading across the two diagnostic lenses" — a product-
   semantic choice pending empirical validation (Phase 4), never claimed as mathematically or
   conceptually proven.
2. **Internal naming (non-urgent, no code change now):** `ai_allowed`/`ai_restricted` can be
   misread as *known* policy conditions when they are only hypothetical lenses. Future candidate
   rename: `permissive_policy_reading`/`restrictive_policy_reading`. Note before acting: these
   keys are a cross-surface report-JSON contract (composers, frontend, PDF, tests), so a rename
   is a deliberate migration, not a drive-by — revisit at earliest alongside Phase 2 work.
   *(Rev 3) Future JSON-shape migration option — flag only, do NOT build now:* once real
   `ai_policy` data exists, `ai_policy=prohibited` sitting next to `ai_allowed.score` /
   `ai_restricted.score` becomes conceptually confusing. When Phase 2 (policy-conditional
   headline selection) actually starts, weigh migrating the report-JSON shape to something like
   `policy_context: prohibited` + `policy_readings: {permissive: {score}, restrictive: {score}}`
   + `headline_reading: restrictive`. Same cross-surface contract cost applies; it is a design
   option to evaluate at Phase 2's natural moment, not a commitment.
3. **`editing_only` lens question** — noted inside Phase 2.
4. **Phase 4 added** — empirical validation once real usage data exists.

---

## Part 2 — GUIDE plan (phased, scoped to buildable)

**Phase 0 — document the max() semantics (small, optional, do now)**
- **P0 · Sonnet standard:** Update `_floor_restricted_to_allowed` docstring + module `_NOTE` (and
  the frontend i18n note if it paraphrases it) to state the floor's semantics as: the displayed
  Restricted score is the **worst applicable reading across the two diagnostic lenses** — not
  literally "risk under a restrictive policy". Only the ordering (Restricted >= Allowed) holds by
  construction; the interpretation is a documented product-semantic choice to be empirically
  validated later (Phase 4), not treated as proven. No scoring change; tests untouched. 1–2 files.

**Phase 1 — policy-context capture (prerequisite for everything the reviewer wants)**
- **P1a · Haiku recon:** Map the scan submission flow end-to-end (frontend form → `POST
  /api/scans` → `ScanJob` → worker task kwargs → `DetectionRunner`/report composer inputs),
  listing every seam an optional new field must cross. Read-only.
- **P1b · Sonnet standard:** Add optional `ai_policy` enum on scans
  (`prohibited | editing_only | allowed_with_declaration | collaboration_allowed | unknown`,
  default `unknown`): Alembic migration (nullable), API request schema, worker passthrough,
  echoed into report JSON. Additive; absent field = byte-identical reports.
  *(Rev 3) Assignment-level, not institution-level:* the enum captures the AI policy of THIS
  ASSIGNMENT, not institution-wide policy — one university can have different AI rules across
  modules/assignments. Treat it conceptually as `assignment_ai_policy` even though the DB field
  name stays `ai_policy`; all UI copy and docs must frame it as "this assignment permits AI
  under these conditions", never "institution X allows AI".
- **P1c · Sonnet standard:** Frontend: optional policy dropdown at scan submission (default
  "I don't know"), i18n en/zh (LOCALIZABLE_PUBLIC_PATHS trap does not apply — authed page — but
  check). Kill-switch env for surfacing.

**Phase 2 — policy-conditional presentation (the real "Layer 4", zero new constants)**
- **P2 · Sonnet standard, Opus review:** When `ai_policy` is known, HEADLINE the matching lens
  (`prohibited`/`editing_only` → lead `ai_restricted`; `allowed_*` → lead `ai_allowed`;
  `unknown` → current side-by-side). Pure selection over the two existing scores — no
  recomputation, no multipliers. Must hit ALL surfaces per the additive-composers rule
  (web + PDF + email + read-time API) and mirror in `draftproof-api/app/_composers/policy_risk.py`.
  Render-verify the PDF visually (verify-rendered-artifacts agreement).
  *`editing_only` caveat (reviewer follow-up, accepted):* mapping `editing_only` → restrictive
  lens is the initial grouping only; it is a flagged **candidate for a future dedicated
  interpretation** ("AI-assisted/polished potentially acceptable, AI-transformed problematic" is
  meaningfully different from full prohibition). No third scoring lens now — no derivation source
  for its weights; revisit in Phase 4 with real usage data.
  *(Rev 3) `editing_only` explanatory copy — explicit Phase 2 DELIVERABLE, not just a scoring
  caveat:* when `editing_only` is selected, the UI must ship policy-specific explanatory text
  alongside the (unchanged) Restricted-lens score — e.g. "Editing-only policy: light AI-assisted
  polishing may be acceptable, but substantial AI transformation may create policy risk." Score
  identical to the restrictive lens; only the explanation is policy-specific. Purpose: prevent
  users reading DraftProof as equating grammar correction with prohibited AI generation. Applies
  to all surfaces Phase 2 touches (web + PDF + email + read-time API), i18n en/zh.

**Phase 3 — declaration_consistency (differentiator; OWNER DECISION GATE first)**
- **P3a · Owner gate (no code):** Approve new surface: student-facing "declare your AI use"
  input on scan submission. Confirm single-copy compatibility (it is — self-declaration, not a
  paired draft).
- **P3b · Opus hard:** Design + implement 3-state `declaration_consistency`
  (`declared_consistent | declared_inconsistent | undeclared`): compare declared scope against
  observed transformation level (category breakdown + `llm_patterning` bucket). Rubric must be
  derived/calibratable — no hardcoded thresholds; ship as **advisory annotation only**
  (annotate-don't-suppress), never a scoring input until an eval slice exists.
  *(Rev 3) ARCHITECTURAL PRINCIPLE — three inputs, never a single-signal shortcut:*
  `declaration_consistency` != AI-detection confidence. The rubric MUST compare ALL THREE of
  (1) student-declared use, (2) observed transformation character, (3) assignment policy
  (`ai_policy` from Phase 1) — it must NEVER collapse to "high AI score → inconsistent".
  Canonical examples: declared "I used AI to rewrite several paragraphs" + high observed AI
  transformation → CAN be `declared_consistent` despite substantial AI involvement; declared
  "I only used spellcheck" + extensive observed transformation → `declared_inconsistent`.
  This is written down precisely because it is the kind of nuance implementation shortcuts
  erase — any P3b implementation that derives consistency from the AI score alone is wrong
  by definition and must be rejected at review.
- **P3c · Sonnet standard:** When a declaration exists, replace the confirm-checkbox with
  declaration status in `PolicyRiskView` + PDF; keep the `confirm_delta` path for `undeclared`.

**Phase 4 — empirical validation (added per reviewer follow-up; GATED ON REAL USAGE DATA, no
code now)**
- **P4 · analysis, model TBD when data exists:** Once Phases 1–3 have accumulated real policy /
  declaration / outcome data, evaluate: (1) does the max() floor improve or worsen calibration —
  how often do large floored divergences actually occur, and do educators read the floored score
  as intended? (2) does `editing_only` need its own dedicated lens? (3) should policy context
  affect scoring, or remain presentation-only? (4) does `declaration_consistency` predict
  educator concern? (5) *(Rev 3)* **lens-vs-educator disagreement rate:** how often does the
  headlined policy lens disagree with educator/reviewer judgment — e.g. (policy=`prohibited`,
  Restricted headline=High, educator concern=Low) or (policy=`allowed_with_declaration`,
  Allowed headline=Low, educator concern=High)? These disagreement cases reveal more about
  calibration than raw score distributions alone. Only at this point may weights be reconsidered
  or a policy-conditioned model introduced. Nothing in Phases 0–3 pre-commits any answer.

**Fable ② review gates:** after Phase 2 and after Phase 3b.
**Explicitly rejected:** multiplicative evidence×policy scoring (no derivation source; loses
per-lens diagnostics); removing the floor for V1 (the ordering Restricted >= Allowed holds by
construction; its *interpretation* is the documented product-semantic choice above, subject to
Phase-4 validation); building Layer-3 scoring before Layer-3 data exists.
