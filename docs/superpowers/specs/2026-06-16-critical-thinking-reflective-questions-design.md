# Design: Critical Thinking → Reflective Questions

## Problem

After a scan, the Critical Thinking section shows a **score** ("53/100, weak control") plus
dimension bars. Real-user feedback: *"only seeing the Critical Thinking Scores ... doesn't really
help or mean serious."* A score is a **verdict, not help** — it grades the student's thinking without
showing where it lapsed, what to change, or why, in their own words. It also contradicts the feature's
own thesis (keep the student *in control of their thinking*): handing them a number to optimize does
the opposite.

Two prior attempts to make it actionable both hit ceilings:
- **Deterministic per-paragraph tags** — structurally *sparse*; fire only on explicit per-sentence
  genericity/citation findings, so they show nothing on the common generic/predictability-heavy essay.
- **Phase-2 LLM scoring/highlights** — failed the variance bar (highlights Jaccard ~0.10, churn on
  re-scan), correctly left dormant.

## The reframe

Turn the section from a **verdict** into **reflective questions** that make the *student* think —
anchored to their actual claims. *"You say 'AI personalises learning' — in your own experience, when
did it fail to?"* The student reads them and revises their own draft. This matches the essay's thesis
directly and is the genuinely useful output.

**Key unlock:** the variance that killed LLM *scoring* does **not** apply to *questions*. A score that
changes every scan erodes trust; a question phrased slightly differently each run is fine — even
better (different angles). So the LLM, correctly benched for scoring, is the right tool here.

## Confirmed decisions

- **Source:** LLM, anchored to the student's actual claims (not generic templates).
- **Targeting:** dimension-steered — reuse the scan's **document-level weakest dimensions** (always
  available, even when per-paragraph is empty) to steer *what* the questions probe; the LLM finds the
  specific claim to anchor to.
- **Interaction:** passive — the student reads and revises their own draft. No input capture, no
  rewrite wiring in v1. (Answer-inline → feed-rewrite is a noted future evolution, explicitly out of
  scope here.)
- **The score:** removed from the user-facing headline. The deterministic diagnosis still computes,
  purely to steer question targeting.

## Design

### Backend
- Repurpose the dormant `poc/detect/critical_thinking_llm.py` into a question generator:
  `generate_reflective_questions(report) -> {questions: [{dimension, anchor_quote, question}], ...}`.
- Input: the draft paragraphs (`scan_intelligence.document.paragraphs`) + the document-level weakest
  dimensions (from `critical_thinking_control.dimensions`, lowest control first; `ai_dependency`
  excluded — never coach off the fluency floor).
- Output: a small set (target 3–5) of questions, each carrying the **verbatim anchor quote** from the
  draft + the question. Agnostic prompt (reason about meaning, no phrase lists, no fabricated facts,
  quote verbatim — mirrors the existing NO-HARDCODE judge prompt).
- **Kill-switch** `DRAFTPROOF_CRITICAL_THINKING_QUESTIONS` (default OFF until quality-validated).
  **Fail-open**: any disable / no key / parse error / exception → no questions, section hides.
  Temperature ~0.4 (variance acceptable). Reuse the gateway/parse/bounded-input pattern from
  `paragraph_explainer.py` already established in the dormant module.
- Wire in the worker scan task (same fail-open block that runs the dormant Phase-2 enrichment), attach
  at `ai_risk_badge.critical_thinking_control.questions[]`.

### Frontend
- `CriticalThinkingControl.jsx` becomes **"Questions to sharpen your thinking"**: drop the 0–100 score
  + band + dimension bars from the headline; render the questions list — each row = the quoted claim
  (styled as a quote) + the anchored question. en + zh i18n.
- If `questions[]` is absent/empty (kill-switch off, or LLM unavailable), the section renders nothing
  (no empty score card). The deterministic `critical_thinking_control` object still ships for internal
  use but is no longer the user-facing surface.

### What stays
- The deterministic `score_critical_thinking` + per-paragraph functions remain (internal targeting,
  and still feed the Phase-B rewrite input). Only the **UI presentation** changes from score → questions.

## Validation

- **Quality, not reproducibility.** Generate questions for 2–3 real reports via R2 (e.g. the
  7afa9f5b / bb1c08ad essays) and judge by hand: are they anchored to a real quoted claim, specific
  (not "what's the counter-argument?"), and genuinely thought-provoking? Kill-switch stays off until
  this passes.
- **Fail-open**: force the LLM to error → section hides, scan unaffected.
- **NO-HARDCODE**: no phrase lists; the prompt reasons about meaning and quotes verbatim.
- **Non-gating**: tier / ai_likelihood / credits unchanged.
- E2E through a real worker scan is CI/deploy-only (local scipy broken; the question generator itself
  is ML-stack-free and runnable locally with .env LLM creds).

## Rollout (avoid an interim empty section)

Validate question quality first with the kill-switch **off** (generate via R2 manually). Then ship the
UI swap (score card → questions) **together with** turning the switch on in prod — so users never see
an empty gap where the score used to be. Until the swap ships, the existing deterministic card stays
as-is. Sequence: (1) build generator + validate quality offline → (2) build the questions UI → (3)
enable switch + remove score card in the same deploy.

## Out of scope (noted for later)
- Interactive answers feeding the rewrite as real grounding (the powerful loop) — explicit v2.
- Removing the deterministic module entirely — it stays for internal targeting + rewrite input.
