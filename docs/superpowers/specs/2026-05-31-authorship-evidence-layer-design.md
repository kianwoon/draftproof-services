# Authorship Evidence Layer — Design

- **Date:** 2026-05-31
- **Status:** Approved (locked)
- **Component:** `poc` (detect + rewrite) · `draftproof-frontend`
- **Relation:** Opportunity #3 of the mitigation/trust thread. Built on the project objective: *mitigate the AI flag honestly; never an evasion/humanizer arms race.*

## Problem

A user submits their own work and DraftProof flags it as AI-risk. The dominant signals (`generic_assertion`, `citation_grounding`) are real, and the perplexity/topk floor is intrinsic and un-mitigable. The user's reasonable reaction: *"This is my own work — why is it flagged, and can I trust the rewrite?"* Today we show the flag and the rewrite, but never the **other side of the ledger**: the human-authorship evidence already present in their draft, and which gaps they can honestly close.

The raw materials already exist in the scan report (`authorship_concern.signals`, `false_positives`) and the rewrite (the diff). None of it is surfaced or reused.

## Objective

Surface an **honest authorship-evidence inventory** — descriptive, no verdict, no score — on both the scan and rewrite pages, and reuse the same data to **steer the rewrite** (protect the user's authentic voice; target grounding at measured gaps). The win is *"the work is more defensibly yours,"* never *"the flag is cleared."*

## Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Placement | **Both** scan (the wound) + rewrite (the reassurance) |
| 2 | Claim | **Evidence inventory** — descriptive, no verdict/score/tier surfaced |
| 3 | Honesty rule | **Present markers + thin signals as action items** (doubles as the mitigation guide) |
| 4 | Build approach | **A — presentation over already-measured signals** (one backend builder; honest by construction, DRY, testable) |
| 5 | Writer boost depth | **Protect + target** — protect authentic spans, aim grounding additions at thin signals |

### Two corrections forced by the content11 PoC (2026-05-31)

- **C1 — Signal direction is risk-oriented (`low = human-positive`).** Confirmed against `derive_concern_tier` (higher `source_grounding`/`draft_evolution`/`structural_reuse` ⇒ *more* concern). The earlier draft note had it inverted. The builder must classify a signal as a *present marker* when its **risk value is low**, and a *thin signal → action* when **risk is high**.
- **C2 — Exclude the perplexity trio.** `predictability`, `surprisal_risk`, `topk_pattern_risk` are intrinsic LLM token-floor stats, not human-authorship evidence and not honestly mitigable. They are excluded from the inventory so the layer never implies authorship from un-ownable statistics.

## Architecture

One pure builder → three consumers (scan report, rewrite report, writer brief). The builder's only inputs are signals the detector already measured, so it **cannot fabricate** authorship evidence; the worst it can do is omit, which the "thin signals as actions" rule prevents.

### Component 1 — `poc/report/authorship_evidence.py` (new, pure function)

`build_authorship_evidence(signals, false_positives, confidence_label, *, sentence_map=None) -> dict`

Output block:

```json
{
  "schema_version": "authorship_evidence.v1",
  "present_markers":        [{"signal": "...", "label": "...", "risk": 0.11}],
  "mixed_markers":          [{"signal": "...", "label": "...", "risk": 0.42}],
  "thin_signals":           [{"signal": "...", "action": "...", "risk": 0.69}],
  "human_recognized_spans": [{"sentence_id": "...", "text": "...", "reason": "..."}],
  "preserved_ideas":        [{"text": "..."}],
  "confidence":             "high|medium|low",
  "summary_line":           "<one descriptive sentence, no verdict>"
}
```

Classification (risk-oriented, per C1): `risk ≤ 0.35` → **present marker**; `risk ≥ 0.55` → **thin signal → action**; in between → **mixed** (shown softly, never a headline claim). `confidence` is carried from `authorship_concern.confidence`; **low confidence softens copy** (short docs) and is shown, never hidden.

**Signal table (included; risk-oriented, low = human-present):**

| Signal | Present-marker label | Thin → action |
|--------|----------------------|---------------|
| `genericity` | Original, non-generic phrasing | Replace boilerplate phrasing with your own wording |
| `specificity` | Concrete, specific detail | Add concrete examples, names, numbers, or scenarios you know |
| `source_grounding` | Claims tied to sources | Tie your key claims to a source or citation |
| `citation_integrity` *(if present)* | Citations support the claims | Add or verify citations for your claims |
| `draft_evolution` | Signs of genuine revision | Keep/show your drafting history where possible |
| `structural_reuse` | Original structure | Reshape any reused/templated structure into your own |
| `burstiness` | Natural sentence-length variance | Let sentence lengths vary as your natural voice would |

**Excluded (per C2):** `predictability`, `surprisal_risk`, `topk_pattern_risk`.

Copy is **content-agnostic** — no hardcoded domain allow-lists (project rule). `human_recognized_spans` is sourced from `false_positives[].sentence/reason` and may be **empty** (graceful — content11 has none).

### Component 2 — Report wiring (`poc/report/report.py`)

Attach `report.authorship_evidence` where `concern`/`false_positives` are already computed (~lines 1319, 4107). Scan report carries the block. Zero new detection.

### Component 3 — Rewrite preserved-ideas (`poc/rewrite_v6/direct_rewrite.py` / `worker/app/tasks.py`)

Diff original vs final text (both already available) → `preserved_ideas` = verbatim **equal** spans (the user's own surviving words). Merge into the same block on the rewrite report.

### Component 4 — Writer boost: protect + target (`direct_rewrite.py`)

Flag-gated `DRAFTPROOF_AUTHORSHIP_BOOST` (default **on**; kill switch `=0`). Extend the `_prompt` JSON payload, threaded through the existing per-paragraph `diagnosis`, with two fields scoped to the paragraph:

- `protected_spans` — present-marker / `false_positive` / high-specificity sentences intersecting this paragraph → *"keep these verbatim; they are the author's voice."*
- `grounding_targets` — thin-signal action directives → *"add concrete anchors here."*

Annotates/steers only. Protected spans are **guidance, not a hard guard** — never force-keep broken grammar, never block a rewrite. Falls back cleanly when the block is empty.

### Component 5 — Frontend (both pages, en + zh)

- **Scan (`Report.jsx`)** — "Authorship evidence" panel near the AI flag: present markers (with example sentences), thin-signal action items, human-recognized sentences. Low-confidence copy softened.
- **Rewrite (`Rewrite.jsx`)** — "What's yours, preserved" section: `preserved_ideas` + the inventory framed as *"your voice, kept and protected."*

New i18n keys in `src/i18n/resources.js` for both `en` and `zh` (project uses `fallbackLng: 'en'`).

## Honesty guardrails (carried from project rules)

- **No verdict/score/tier** surfaced. Never "this clears you / passes Turnitin." Only "more defensibly *yours*."
- Builder **re-presents measured signals** — cannot fabricate; thin signals shown as actions, never hidden.
- **Content-agnostic** — no hardcoded domain allow-lists; reuse the agnostic detectors.
- Writer boost **annotates/steers, never suppresses**.
- **Honest expectation stated in copy:** strengthening thin signals raises the *movable* `generic_assertion`/specificity and authentic-voice retention; it does **not** move the headline AI flag (topk floor).

## Verification

1. **Unit tests** — `poc/test_authorship_evidence.py`: only `risk ≤ 0.35` → markers; `risk ≥ 0.55` → actions; direction correctness per signal (C1); perplexity trio excluded (C2); `preserved_ideas` = exact equal spans; empty `false_positives` degrades gracefully; cross-domain agnostic (no domain vocab dependence).
2. **Builder corpus A/B** — run across `test_content*` and eyeball the present/thin split per doc (must not be all-flattering or all-empty).
3. **Writer boost** — deterministic harness `DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 4` (N≥4): confirm **no `final_risk` regression**, protected spans actually preserved verbatim, `generic_assertion` holds/improves. Honest expectation: headline flag unchanged.
4. **Frontend** — render both pages against a saved report (content11) incl. the empty-`false_positives` case.

## Out of scope (YAGNI / explicitly rejected)

- **Optimize-for-authorship-score** selection objective (rejected — proxy-gaming risk, score-chasing).
- **New authorship-analysis module** with its own scoring (rejected — second source of truth that can contradict the badge).
- **Pasteable defense statement** (deferred — possible future opt-in; not in this build).
