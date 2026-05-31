# DraftProof — Alignment Principles (Service ⇄ Objective)

**Status:** Source of truth for how every DraftProof surface (scan report page, scan PDF,
rewrite report page, rewrite PDF, completion emails, copy) must express the product
objective. Written 2026-05-31 after a measurement-driven review that exposed a drift
between the service's signals and its charter.

---

## 1. The Objective (one north star)

**DraftProof mitigates AI-writing risk by coaching the user to GROUND their content.**

- AI risk comes primarily from **content lacking** — generic, unanchored, ungrounded
  claims — **not** from word choice or token "perplexity."
- **The rewrite feature is a SHOWCASE: it demonstrates to the user *how* to improve their
  content to be AI-risk-free.** It is a teaching draft / reviewable example — **not** a
  final submission, and **not** machine output the user should hand in as-is.
- **The user owns the final words.** They review the before/after, learn what grounding
  looks like, and re-express it in their own real content.

DraftProof is an **integrity & grounding-coaching tool.** It is **not** a "lower your
Turnitin score" service, a "humanizer," or a detection-evasion tool.

---

## 2. The Core Truth (data-backed, settles the drift)

Two different things can be "optimized," and they point in **opposite directions**:

| | What it measures | Can the user improve it? | Role |
|---|---|---|---|
| **DraftProof score** (grounding) | content lacking — generic-assertion, specificity, grounding | **Yes** — by adding real specifics/anchors | **NORTH STAR — the optimizable signal** |
| **External / "Turnitin" estimate** (perplexity) | raw token **fluency** | **No** — flags any fluent prose, human or AI | **Risk warning — NOT a target** |

**Evidence (measured this session, local GPT-2 detector):**
- Genuinely human, specific, idiosyncratic writing scored **56% external** with **raw top-k 82** — *higher* than the AI original's 79. The external estimate measures fluency, so it **over-flags human writing.**
- No rewrite — LM or genuine human — reaches the <20% "pass" range. The only way to lower a fluency score is to write *less fluently*, which contradicts the goal.
- DraftProof's own score, by contrast, dropped **42 → 26%** on genuinely-grounded content — because it discounts raw top-k (deliberate false-positive protection) and rewards grounding. That is the objective working.

**Conclusion:** Optimizing the external/perplexity number is both **impossible** and **off-objective**. Optimizing grounding is **possible** and **is** the objective.

---

## 3. Signal Principles (how to frame each number, everywhere)

1. **DraftProof score = the metric to act on.** Lead with it. Frame it as "what improving
   your grounding will move." This is where the rewrite showcase and user effort pay off.
2. **External / Turnitin estimate = a risk-awareness warning, never a target.**
   - DO say: "Strict third-party detectors weight raw fluency and over-flag — *including
     human writing*. This is a heads-up, not a score to beat. Your real safeguard is
     owning and grounding your content."
   - DO NOT say / imply: "target < 20%", "not met", "drive this down", or anything that
     frames it as a goal the rewrite (or the user) should chase. **(Walk back the
     "target < 20% — not met" indicator added earlier — it points the wrong way.)**
3. **Rewrite rating / score-drop = a teaching signal, not a verdict of success.** The
   rewrite *showcases* grounding; a lower DraftProof score means "this is what grounded
   content looks like," not "you may now submit this."
4. **Calibrated/derived sub-numbers stay in their labeled breakdown** ("How DraftProof
   calibrates this"), never competing with the headline. (One canonical AI number per
   surface — already enforced by removing the 0.5 display multiplier.)

---

## 4. Surface Principles (consistency rules)

- **One objective, one story, every surface.** Page, both PDFs, and emails must express
  the same framing: grounding is the goal; the rewrite shows *how*; the user finishes.
- **No surface re-derives a user-facing number.** All read the badge's canonical values
  (`ai_likelihood_score`, `external_detector_estimate`) — no per-surface multipliers,
  tables, or relabels. (Lesson paid for this session: 4 copies of a 0.5 multiplier.)
- **Every AI-style flag must point to an action the user can take** (add a date, a source,
  a specific example, a personal observation) — because grounding is the lever.

---

## 5. Anti-Goals (what DraftProof must NOT become)

- ❌ A Turnitin/AI-detector score-beater. (The number flags fluency; chasing it is futile
  and would mean degrading the writing.)
- ❌ A "humanizer" / paraphrase-churn / decode-trick / perplexity-gaming tool. (Games
  DraftProof's own score, not real detectors; off-mission; enables dishonesty.)
- ❌ A tool that implies its machine rewrite is submission-ready or detector-safe.
- ❌ Fabrication as the end state. The rewrite may show *illustrative* anchors, but the
  user must replace them with their own real specifics.

---

## 6. Implications for current code (alignment backlog)

1. **Reframe the external estimate** (page `report.aiLikelihood.*`, render.py, render_rewrite.py,
   email_service.py): replace "Turnitin pass target < 20% — not met" with the
   risk-awareness wording in §3.2. Keep the number; change its meaning.
2. **Make grounding the lead optimizable signal** in the report narrative; frame the
   rewrite as a "grounding map / showcase."
3. **Author-proxy = elicit + integrate the user's real specifics** (use the existing
   `user_input_needed`) — scaffold the user's own grounding rather than implying
   auto-fix. Do not expand it to chase the external number.
4. Keep the single-canonical-number + no-re-derivation invariants (guard tests already
   added: `_display_ai_score`/`aiLikelihoodBands` not halved; compaction keep-list).
