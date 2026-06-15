# Scan Workflow — Complete Hardcode Audit

**Date:** 2026-06-15 · **Scope:** scan workflow only (`poc/detect/*` + `poc/report/*` scan-report path). Rewrite workflow audited separately.
**Method:** 5 parallel read-only audits, every file, classified per the project rule (`feedback_no_hardcode`): baked content-word/phrase/answer lists and magic-number cutoffs are violations; complete grammatical classes, morphological/tokenization/structural regex, and genuine versioned config are allowed.

## Classification key
- **V-list** = baked content-word/phrase list used to match/classify text (the primary violation — overfits, fails silently).
- **V-magic** = magic-number threshold/band/weight baked into detection logic (should be derived/calibrated).
- **borderline** = closed-class-ish (discourse markers / lexical-verb sets) — judgment call.
- **CONFIG / ALLOWED** = presentation labels, versioned weights, structural/morphological regex, operational enums — not violations.

## A. Content-word / phrase list violations (PRIMARY — overfit baked answers)

| file:line | literal | feeds | class |
|---|---|---|---|
| layer3_scoring.py:390 | `GENERIC_ESSAY_STARTERS` ("in the past","today","the real challenge","the goal should"…) | repeated_starter / qualifying_text_ai_density | **V-list** |
| layer3_scoring.py:363 | `FORMULAIC_PROGRESS_MARKERS` | paragraph_progression_risk | **V-list** |
| layer3_scoring.py:493 | `SIGNPOST_ANNOUNCE_PATTERNS` | signpost_paragraph_risk | **V-list** |
| layer3_scoring.py:874 | `FORMULAIC_CONCLUSION_PATTERNS` | formulaic_conclusion_risk | **V-list** |
| layer3_scoring.py:563 | `BALANCED_HEDGING_PATTERNS` ("plays a vital role","it is worth noting"…) | balanced_hedging_risk | **V-list** |
| layer3_scoring.py:378 | `BALANCED_FRAMING_PATTERNS` | balanced_generic_framing_risk | **V-list** |
| layer3_scoring.py:1009 | `lived_detail` content words `feedback`/`testing` + perception-verb list (`taught`,`treat`,`demonstrate`,`encourage`…) | lived_detail_risk → specificity_gap | **V-list** |
| scoring.py:252 | `estimate_lived_detail_risk` **domain vocabulary** (`classmate`,`interviewee`,`participant`,`prototype`,`medication`,`alarm`,`label`…) | lived_detail_risk | **V-list (domain — worst kind)** |
| scoring.py:311 | progression markers (dup of FORMULAIC_PROGRESS) | paragraph_progression_risk | **V-list** |
| scoring.py:384 | formulaic-conclusion patterns (dup) | formulaic_conclusion_risk | **V-list** |
| predictability.py:108 | `conclusion_markers` | subtype `formulaic_conclusion` | **V-list** |
| predictability.py:117 | `personal_markers` ("i believe that","in my opinion","a teacher's role should"…) | subtype `template_personal_reflection` | **V-list** |
| predictability.py:126 | `policy_markers` ("should embrace these tools","guide students on how to"…) | subtype `generic_policy_claim` | **V-list** |
| predictability.py:135 | `edu_markers` ("plays an important role in","has transformed the way","increasingly important in today's"…) | subtype `broad_education_claim` | **V-list (domain)** |
| profiles.py:187,198,209,220,231,239 | per-domain `claim_indicators` (education/healthcare/legal/engineering/business/hair_beauty: `learners`,`patient`,`clinical`,`treatment`…) | domain-profile matching | **V-list (domain × 6)** |

**Borderline (closed-class-ish — decide per policy):**
- layer3_scoring.py:763 `_REGISTER_CONNECTIVES` (curated formal connectives) · :1041 `ASSERTION_VERB_PATTERNS` lexical verbs (`creates/makes/provides/enables`) · :1033 `HEDGING_PATTERNS` "some researchers/scholars".
- document_structure.py:308/314/330 discourse connectives, enumeration words, continuation openers.
- postprocess.py:255 definition-trigger phrases (`refers to`,`is defined as`).
- ai_generation.py:284 `_criterion_to_finding_type` map · report.py:283/328 finding-**title** → actionability routing (matches internal signal keys, not text content — leans ALLOWED).

**Allowed (NOT violations):** layer3_scoring.py CONCRETE_DETAIL_PATTERNS / CONTEXTUAL_ANCHOR_PATTERNS / `_REGISTER_NOMINAL` (morphology) / `_REGISTER_FIRST_PERSON` (pronouns); profiles.py `_STOP_WORDS` + `auto_extract_domain_terms` (content-DERIVED); postprocess.py GlossaryFilter (auto-extract); utils.py `_ABBREVIATIONS`; citation/predictability biblio/URL/DOI regex; run.py finding-type routing enums.

## B. Magic-number violations (SECONDARY — pervasive)

Threshold/band/weight cutoffs baked into detection logic, by file (representative, not exhaustive):
- **layer3_scoring.py:760-761** `_TURNITIN_TOP10_THRESHOLD=0.46`, `_TURNITIN_REGISTER_THRESHOLD=1.1`; risk-band ratios `0.90/0.80/0.65/0.45`.
- **ai_generation.py:20-35** `LIKELIHOOD_THRESHOLDS`, `MULTI_SIGNAL_RULES` count cutoffs.
- **topk_calibration.py:69-80** piecewise calibration curve constants.
- **semantic_shape.py:75-149** uniformity/drift thresholds + min-sentence `6`.
- **similarity.py:17,157** `_RISK_MAP`, overlap bands `0.75/0.85`.
- **citation.py:135-145** issue-count → risk decision tree.
- **grounding_diagnosis.py:41-44** score bands `20/40/60/80`.
- **external_grouped_scoring.py:71-76** band/cap cutoffs (`50/20/65/0.45/49.9`).
- **authorship_windows.py:16-21,225** label thresholds + top-`8` slice.
- **transformation.py:209-433** dense feature weights + classification-rule cutoffs.
- **repair_units.py:132** risk-aggregation weights.
- **rewrite_targets.py:96-161** risk-tier + document-shape cutoffs.
- **postprocess.py:363** `_THRESHOLDS 0.70/0.45/0.35`; **mitigation.py:171-205** priority/target-score cutoffs.
- **report.py:689-1109** tier/escalation/downgrade cutoffs throughout; **render.py:452-917** `_CALIBRATED_AUTHORSHIP_LEVELS` brackets + dense authorship-rating grids; **authorship_evidence.py:20-21** `PRESENT_MAX/THIN_MIN`.

*Note:* some weight dicts are explicitly **versioned** (e.g. `external_grouped_scoring` v2) = calibrated config, defensible; guessed cutoffs are not.

## C. Notable findings
1. **`scoring.py` = live Authorship-Concern module** (imported at `detect/__init__.py:21`), holding a parallel set of estimators. Its `estimate_lived_detail_risk` (scoring.py:247) carries **raw topic vocabulary** — `classmate, interviewee, participant, prototype, compartment, label, transparent, alarm, medication` — the single most blatant overfit hardcode in the scan path. **Reachability:** no caller found for scoring.py's `estimate_lived_detail_risk` (the live one is `layer3_scoring.py:1639`), so this list appears **DEAD** → safe immediate delete. But `compute_structure_process_cluster` (scoring.py:408) DOES call scoring.py's progression/uniformity/starter/conclusion estimators — confirm that cluster's reachability before touching those.
2. **`profiles.py` ships 6 domain `claim_indicators` lists** — the clearest domain-vocabulary hardcode in the scan path.
3. **`predictability.py` subtype markers** are overfit to education essays ("a teacher's role should", "increasingly important in today's").

## D. Tally
- **V-list (clear): ~15 lists** across 5 files (layer3_scoring ×7 incl. lived_detail, scoring ×3, predictability ×4, profiles ×6-domain).
- **V-list (borderline): ~7.**
- **V-magic: 50+ sites** across ~15 files.

## E. Remediation (proposed; validation is CI-gated — local ML stack broken, corpus A/B mandatory per `project_detector_hardcode_audit`)
- **V-list → derive or delete:** formulaic/essay/signpost/cliché detectors are redundant with the agnostic statistical signals (predictability/topk/burstiness/uniformity/repeated-structure) → drop, lean on statistics. Grounding (lived_detail/broad_claim) → source from structural concreteness (numbers/caps/quotes/citations) already in CONCRETE_DETAIL_PATTERNS. profiles.py domain lists → rely on `auto_extract_domain_terms` (already the fallback). Delete dead `scoring.py` if confirmed legacy.
- **V-magic → single source + calibrate:** lift cutoffs into one named, versioned config module; keep only calibrated (not guessed) values.
- **Order:** profiles.py domain vocab → predictability.py subtype markers → layer3/scoring formulaic+lived_detail lists → magic-number consolidation.
- Each phase gated by a corpus A/B in CI (de-overfitting a ratio-scored list can over-flag).
