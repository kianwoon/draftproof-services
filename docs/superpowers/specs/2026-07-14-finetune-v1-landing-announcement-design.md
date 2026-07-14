# Design: Fine-Tune v1 Landing Page Announcement

**Date:** 2026-07-14
**Status:** Draft — Fable 5 review REVISE incorporated (see §6); pending owner re-approval of corrected copy
**Scope:** One new static marketing line in the hero section of the public landing page. No backend, scoring, or detection changes.

---

## 1. Context (evidence-grounded)

The last 6 pushes to `main` (all 2026-07-14) shipped the "fine-tune v1" detector: a desklib-architecture DeBERTa-v3-large re-trained from the prod academic checkpoint (`e40f7ece`), followed by calibration/rollout work (`1baeaa1b`, `c1337d25`, `647fac65`, `ea5847bf`) and a same-day prod-incident fix (`18b0a7c5`, a units mismatch in the doc-level deep-scan call, patched ~7 minutes after detection, 74 tests green).

**Confirmed live in production:** Koyeb worker env `DRAFTPROOF_MODAL_CHECKPOINT=draftproof/finetune-v1-gpt55` (verified via `koyeb deployment get`), deployed same-day.

**Verified results — SUPERSEDED, see §1.1.** The `e40f7ece` commit reported 0% FPR across 648 held-out humans, but that was the pre-promotion eval on a simplified sentence splitter, self-flagged in the same commit as "NOT promoted: pending exact-harness re-verify." Do not quote it.

### 1.1 The actual live operating point (verified 2026-07-14, second review pass)

Two later commits changed the operating point AND re-measured FPR on the promoted checkpoint (`draftproof/finetune-v1-gpt55`):
- `647fac65` (10:44am): re-weighted tier authority 0.50/0.50 → 0.35/0.65 (composite/deep_scan), `sent_threshold` 0.999 → 0.99. Evidence at the live amber cutoff: **ESL FPR 0.74% (2/272, both higher-proficiency; lower-prof 0%)**, RAID human FPR 0%, RAID TPR rises (none 79.5→90.0, paraphrase 55.5→74.0), GPT-5.6 100% under live composite=0 sensitivity.
- `ea5847bf` (11:00am): calibrated-mean representation swap, re-confirms **ESL FPR@32 0.74%** (down from an interim 1.10% during tuning), RAID human FPR 0%, GPT-5.6 100% at composite=0.

**Corrected claim for the hero:** ESL false-positive rate is **under 1% (0.74%, 2/272 SCoCESLE essays)**, not 0%. GPT-5.6 zero-shot transfer (100%) still holds at the live operating point across both re-weight commits, but per the owner's model-naming decision (§1.2) this isn't named directly either. The "648 eval humans" and "0.95→0.21" figures are pre-promotion-methodology numbers and are dropped from any public claim — only the twice-confirmed, live-operating-point numbers (272-essay ESL check, RAID human 0%) are used.

### 1.2 Correction to the initial ask (model naming)

The user's framing named training on "GPT5.5, gemini, qwen... and thousands of essays." I downloaded and inspected the actual training file (`train_windows.jsonl`, off the `draftproof-finetune-v1` Modal volume) — it contains only bare `text`/`label` fields, and the commit's own final corpus tally (GPT-5.5 + gpt-5.1-family + RAID + PERSUADE, summing to ~5,818 of the stated 5,828 docs) never accounts for the "frontier_mix_distill" (Gemini/Grok/Qwen/Claude) dataset the corpus-builder script (`build_v1_corpus.py`) reaches for — that stream most likely silently returned ~0 rows (bare `try/except` around the HF `load_dataset` call). Separately, "Qwen" elsewhere in this repo is the grammar-repair model in the rewrite pipeline — unrelated to detector training. **Decision (owner, this session): do not name any specific model.** Initial wording used "frontier AI models" generically; the Fable 5 review (§6) further refined this to "frontier AI writing" (final copy in §2.1) since "models" plural overstates corpus diversity even without naming names.

## 2. Design

### 2.1 Copy

New i18n key `landing.detectorUpdateNote`, English:

> "Detector freshly fine-tuned on frontier AI writing and thousands of real essays — ESL false positives held under 1%, even as detection accuracy rose."

Changes from the pre-review draft, both required by the Fable 5 review (§6):
- **"frontier AI models" → "frontier AI writing"** — mass noun, not plural models. Rigorously only GPT-5.5 (2,153 real docs) + same-lab gpt-5.1-family (168 docs) are confirmed training contributors; RAID is a pre-existing 2024-era multi-source benchmark, not new frontier training data. "Models" (plural) overstates how many distinct frontier sources are actually in the corpus.
- **"0% ... including ESL students" → "under 1%"** — the live, twice-confirmed operating-point number is 0.74% (2/272 SCoCESLE), not 0%. Dropped the "648 human writers" figure (pre-promotion methodology, superseded) and the specific "students" framing.

No specific model names (per owner decision, §1.2). No comparative claims (matches existing `landing.trustNote` tone — "Not a Turnitin replacement" — and the repo's standing rule to complement, not claim superiority over, other tools).

### 2.2 Placement

[Landing.jsx](../../../draftproof-frontend/src/pages/Landing.jsx) — new line inside `.hero-copy`, directly below the existing `.trust-note` block (~line 186-189). Reuses the sparkle-icon visual already established for `.hero-free-credit` (a "new/positive update" motif) rather than the shield icon used for `.trust-note` (a "compliance/trust" motif) — keeps the two lines visually distinct in purpose.

### 2.3 i18n

Same key added to both `en/landing.js` and `zh/landing.js`. Both entries tagged with the existing `// allow-hardcode: static landing-page UI copy` convention already present in `landing.js` (e.g. above `sampleStats`) — this is static marketing text, not detection/scoring logic, so it is the correct bucket under the repo's no-hardcode rule, not an exception to it.

**Open item:** the Chinese translation is drafted by Claude, not a native reviewer — flagged for the owner's own pass before/after ship.

### 2.4 Styling

Small addition to `02-landing-hero.css` reusing existing `.hero-copy` spacing/typography tokens. No new visual system, no new component.

### 2.5 Staleness handling

"Freshly fine-tuned" is time-sensitive copy that will read stale within weeks, and a bare TODO comment rots silently (Fable 5, §6). Two things, not one:
- Visible copy includes a light date anchor so it doesn't imply "today" indefinitely — e.g. trailing "(July 2026)" or fold into a small "Updated July 2026" micro-label rather than "just."
- Inline maintenance comment at the i18n key: `// TODO: revisit/remove this fine-tune-v1 framing after 2026-08-15`.

### 2.6 Rollback coupling

The checkpoint's own rollback path is config-only (`ea5847bf`: set `representation` back to `proportion`; checkpoint/weights revert via the same 2 Koyeb env vars `647fac65` changed). **If the detector is ever rolled back, this hero line must be pulled in the same action** — it would otherwise keep claiming a fine-tune the live system no longer runs. Note this as a manual follow-up step in any future rollback runbook; not automated here (static copy, no feature flag).

## 3. Non-goals

- No new "AnnouncementBanner" component / config-driven announcement system (considered as Option 2, owner chose the smaller static-line approach for this instance).
- No change to the `trust-bar` strip, sample report, or any scoring/detection code.
- No claim naming specific competitor or specific generator models.

## 4. Validation (before merge)

1. Visual check in EN and ZH, desktop + mobile viewport (hero already has mobile-specific handling from `f82f2fe1`).
2. Confirm the new line does not visually collide with `.hero-free-credit` or `.trust-note` at narrow widths.
3. **Run `npm run build` and inspect the prerendered `/zh` HTML output directly** (not just the dev server) — this repo has a prior shipped bug where landing-page body content was not actually present in prerendered HTML despite rendering correctly in dev (`project_seo_body_not_prerendered`, fixed 2026-07-01). Grep the built output for the new copy in both locales before calling this done.
4. No automated test changes needed beyond #3 (static copy, no logic).

## 5. Rollout

Plain frontend commit to `main` → Koyeb auto-deploys the `draftproof` (WEB) service. No worker redeploy, no Modal cost, no env var changes.

**Explicit risk acceptance (owner, this session):** shipping this announcement only hours after the same-day prod incident (`18b0a7c5`) and its fix is a deliberate choice, not an oversight — 74 tests green, no further incident reports, re-weight commits independently re-confirm the ESL/RAID numbers. No additional monitoring gate is being added before publish; if a new regression surfaces post-publish, pulling the hero line is part of the rollback action (§2.6).

## 6. Fable 5 review record (REVISE → fixed)

Native `advisor()` was unavailable this session; review performed by `Agent(model: "fable")` per the documented fallback, with its own primary-source read of the commits (not just this document).

**Verdict: REVISE** — one blocking finding:

- **Blocking:** the pre-review draft's headline stat ("0% false positives ... including ESL students") was the **pre-promotion** eval from `e40f7ece`, which self-flagged as "NOT promoted: pending exact-harness re-verify." The actual live operating point (`647fac65`, `ea5847bf`) records **ESL FPR 0.74% (2/272)** — confirmed independently by re-reading both commits in full (§1.1). The deployed product's own commit trail falsified the drafted claim. **Fixed:** copy now says "under 1%," the false "0%/648/including ESL students" framing is removed (§2.1).
- "Frontier AI models" (plural) judged thin given only one model family is rigorously confirmed as a training contributor. **Fixed:** changed to "frontier AI writing" (§2.1).
- Staleness handling via TODO-comment-only risks silent rot. **Fixed:** added a visible date anchor alongside the maintenance comment (§2.5).
- Rollback path exists (config-only) but wasn't coupled to the marketing copy. **Fixed:** added §2.6 requiring the hero line to be pulled in the same action as any checkpoint rollback.
- Requested explicit recording of the "ship now, hours after an incident" decision as a deliberate risk acceptance rather than an implicit one. **Fixed:** added to §5.
- Requested prerendered-HTML verification given this repo's prior prerendering gap. **Fixed:** added to §4.

Not yet independently re-verified by a third pass: the exact wording of the final EN/ZH copy once implemented (planned as part of the writing-plans → implementation step, with a visual render check per §4).
