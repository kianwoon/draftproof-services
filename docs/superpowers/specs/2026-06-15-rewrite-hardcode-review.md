# Rewrite Workflow (`rewrite_v6/`) — Hardcode Review (agnostic-blockers)

**Date:** 2026-06-15 · **Scope:** baked DOMAIN/CONTENT in the rewrite pipeline that biases the rewrite toward specific subjects/phrasings or blocks it (prevents content-agnostic rewriting). Method: 4 read-only audits (prompts / guards / naturalisation / routing), then manually verified the top claims. Default live path = `direct_rewrite.py` (`DRAFTPROOF_V6_DIRECT_REWRITE=1`); `plan.py`/`planner_llm.py` = legacy.

## Framing
"agnostic" = domain-vocabulary-free. Allowed: structural/grammatical/closed-class, data-derived anchors from the user's own text, meaning-fidelity + grammar guards. Per CLAUDE.md the rewrite ADDS concrete content and **guards must ANNOTATE, not REJECT** — a content-word guard that *rejects* (→ `source_preserved`, no change) is the worst case.

## Findings (prioritized; live-path first), with corrections to the raw audit

### P1 — Baked education examples in the DEFAULT system prompt ⭐ worst
- `direct_rewrite.py:~227-231` and `~302-307` — the author-proxy `_SYSTEM` prompt literally instructs with **education examples**: `"'In my classroom, I have watched about a third of students …', 'When I grade essays, I keep finding …'"`. **VERIFIED.** This is in the core system prompt of *every* default rewrite → nudges all rewrites toward classroom/education regardless of the document's actual subject.
- **Fix:** replace with a domain-NEUTRAL template (`"In my <setting>, I <observed> <specific particular>"`) or instruct the model to draw the concrete particular from the user's own text. Keep the structural lesson (first-person frame + concrete particular); drop the subject.

### P2 — FATAL content-word guards that REJECT (→ source_preserved)
In `integrity_guard.py`, these emit blockers NOT in `ADVISORY_BLOCKERS`, so they are **fatal** (force `source_preserved`) and key on baked content-word lists — **VERIFIED fatal**:
- `report_verbs` (~454-472, 17 verbs: adds/notes/sees/…) → `external_narrator_reporting_chain` (fatal).
- nonhuman-actor noun regex (~520-523, tools/apps/courses/platforms) → `malformed_nonhuman_activity_predicate` (fatal).
- `_malformed_tool_skill_predicate` (~700-715, tool/system/software nouns) → fatal.
- **Tension:** the *intent* (catch genuine malformations like "tools are learning") is a legitimate grammar guard, but the *implementation* is a baked content-noun list → non-agnostic + can reject valid tech/tool-domain rewrites. **Fix:** detect structurally (POS: nonhuman subject + human-cognition verb) instead of a noun list; or demote to advisory.

### P3 — `naturalisation.py` `droppable` set (live)
- `naturalisation.py:~607-620` — `droppable = {"situation","process","goal","system","model","issue", … ~48 abstract nouns + generic verbs}`, used to flag content dropped during a naturalisation patch. **Academic/business-biased** vocabulary; live path. **Fix:** compare dropped *content tokens* structurally (non-stopword nouns/verbs from the user's own sentence) rather than a baked list.

### P4 — Playbook / shape routing (liveness PARTLY confirmed — verify before acting)
- `rewrite_playbook.py` `playbook_entries(finding_tags, source_text)` IS default-path (`direct_rewrite.py:284`). Its playbook *shapes* are mostly domain-neutral (ALLOWED), but the audit flagged a `benefits/risks` domain phrase trigger (~65-68) — **confirm**.
- `finding_pattern.py:40-52` `_has_benefit_risk_shape` / `_has_old_new_mismatch_shape` (regex on opportunities/benefits/risks/old/traditional) route to domain-shaped playbooks ("benefit_risk_contrast", "old_model_current_mismatch") with baked instructions (74-99). **Liveness unclear** (audit said legacy in one place, live in another) — TRACE whether `direct_rewrite`/`playbook_entries` invokes finding_pattern before treating as live. If live: domain-biases the rewrite *shape*.
- `document_reviewer.py:47` craft-guideline text "use a concrete classroom/workflow example" — domain anchor in a reviewer guideline.

### Minor / NOT as severe as the raw audit claimed
- `prose_quality.py` `_NOT_ONLY_RESTORABLE_VERBS` (6 verbs) + repair-trace phrases — the audit called these "FATAL"; **corrected:** `fragment_or_trace_sentence` IS in `ADVISORY_BLOCKERS` → it **annotates**, not rejects. These are prose-repair internals / advisory, lower priority.
- `required_source_terms_missing` — already ADVISORY (correctly demoted per the agnostic objective).
- `prose_repair_rules.py:47` `{"used","well"}` — tiny, advisory-ish.

## ALLOWED (not violations — do not touch)
Closed-class grammatical sets (pronouns, articles, determiners, modals — naturalisation function_words/dependent), discourse-marker regex (`_ROBOTIC_TRANSITIONS`), number-words (`bracket_grounding._WORD_NUMBERS`), stop-word filters, structural sentence ops, data-derived top-k token spans (highlight_topk_repair), domain-neutral playbook shapes, citation/numeric regex.

## Tally + priority
- **Live-path agnostic-blockers: 3 clear** (P1 prompt examples, P3 naturalisation droppable, P2 ×3 fatal guards) + **P4 routing (verify liveness)**.
- **Order to fix:** P1 (prompt examples — every rewrite, trivial fix, highest impact) → P2 (fatal content-word guards — they *reject*, the worst outcome) → P3 (naturalisation droppable) → P4 (confirm + de-domain the shape routing).
- **Out of scope here:** legacy `plan.py`/`planner_llm.py` tag-keyed branches (not the default path); these are the same dead-tag branches noted in the scan work.

## Validation when fixing
No corpus AUC harness for rewrite (the scan harnesses don't cover it). Use `DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 4` (final_risk, high-variance — N≥4) + re-run a few real docs across DIFFERENT domains to confirm the rewrite no longer injects education/classroom framing into non-education text.
