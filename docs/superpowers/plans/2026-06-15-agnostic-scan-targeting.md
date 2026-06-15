# Agnostic Scan Targeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `poc/rewrite_v6/scan.py` paragraph targeting domain-agnostic by removing the hardcoded content-word detectors and letting the detector's content-derived grounding signals decide which paragraphs get rewritten — unconditionally, no feature flag.

**Architecture:** In the default V6 direct path, a paragraph is rewritten iff it has ≥1 scan finding or a paragraph diagnosis (`direct_rewrite.py:1067`). Findings come from two sources merged in `scan_text_with_report`: (1) scan.py's own `_risk`/`_tags` heuristics, (2) the detector report's per-sentence grounding signals (`_report_findings`). We delete the arbitrary content-word heuristics from (1), keep only its structural-agnostic components, and keep (2) as the driver. We also stop the report's un-mitigable `review_only` signals (predictability/top-k) from creating a driving finding.

**Tech Stack:** Python 3, pytest. Pure-stdlib regex (no NLP deps in `poc/`). Deterministic measurement harness `poc/_measure_baseline.py`.

---

## File Structure

- `poc/rewrite_v6/scan.py` — MODIFY. Remove content-word detector functions and their use in `_risk`/`_tags`; reduce `_citation_anchor` to its structural form; drop the `_predictability_primary` coupling; make `_report_findings` not emit a *driving* finding for `review_only`-only segments.
- `poc/test_rewrite_v6_agnostic_scan.py` — CREATE. New pytest module covering: content-word detectors removed, structural-agnostic detectors retained, grounding-report findings still drive, predictability-only segment does not drive.

No other files change. `direct_rewrite.py` is **not** modified — its "has findings → rewrite" gate already yields the desired behavior once the findings set is agnostic.

---

## Background facts the implementer needs (verified)

- `scan_text(text) -> Scan`. `Scan.findings: list[Finding]`, `Finding(sentence_id, paragraph_id, tags: list[str], severity: float, evidence: dict)`.
- `scan_text_with_report(text, report) -> Scan` merges structural findings with report grounding findings.
- `findings_for_paragraph(scan, paragraph_id) -> list[Finding]`.
- `_scan_from_paragraphs` emits a structural finding only when `tags and risk >= 12.0` (`scan.py:102`). Removing content-word tags therefore removes the spurious findings.
- Tests live at `poc/` root, import `from poc.rewrite_v6.scan import ...`, run with pytest.
- Run a single test module: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py -v`.

### Detectors to REMOVE (arbitrary content-word lists)

From `_tags` (`scan.py:463`) and `_risk` (`scan.py:493`), and delete the now-unused functions:
`_predictable_start` (570), `_predictable_start_pressure` (519), `_context_anchor_gap` (574),
`_author_anchor_gap` (583), `_broad_claim` (601), `_transition_stack` (606),
`_semantic_bridge_gap` (612), `_unsupported_claim_gap` (617).

### Detectors to KEEP (structural / morphological — already agnostic)

`_list_pressure` (comma/separator density), `_abstract_pressure`/`_abstract_risk_pressure`
(nominalization morphology), `_named_anchor_count` (proper-noun count), `_paraphrase_smoothing`
(structural), `_repeated_frame_findings` (n-gram frame repetition), sentence `word_count` overload.

### `_citation_anchor` (597) — reduce to structural form

Currently: `\baccording to\b | \b\w+\s+et al\.\s*\(\d{4}\)\s+(states|indicates|argues|notes|describes)\b`.
Keep only the structural citation **form** `\b\w+\s+et al\.\s*\(\d{4}\)` (and a bare `\(\d{4}\)`
parenthetical-year already covered by `_parenthetical_citation`). Drop the reporting-verb list
(`states|indicates|...`) and the bare `according to` phrase match.

---

## Task 1: Characterization tests — lock current agnostic-target intent

**Files:**
- Create: `poc/test_rewrite_v6_agnostic_scan.py`

- [ ] **Step 1: Write tests that assert the TARGET (post-change) behavior — they will fail now**

```python
"""Targeting must be driven by content-derived grounding signals, not hardcoded word lists."""
from poc.rewrite_v6.scan import (
    scan_text,
    scan_text_with_report,
    findings_for_paragraph,
)

# --- Sentences that ONLY trip the removed content-word detectors -------------------------------

def _all_tags(scan):
    return {tag for finding in scan.findings for tag in finding.tags}

def test_broad_claim_wordlist_no_longer_tags():
    # "one of the" + "the most" tripped _broad_claim; no number/name/citation => no grounding tell.
    scan = scan_text("This is one of the most significant ideas of the modern age.")
    assert "broad_claim" not in _all_tags(scan)

def test_transition_stack_wordlist_no_longer_tags():
    scan = scan_text("However, furthermore, the situation continued to develop over the period.")
    assert "transition_stack" not in _all_tags(scan)

def test_predictable_start_wordlist_no_longer_tags():
    scan = scan_text("This shows that the overall direction of the work stayed consistent throughout.")
    assert "predictable_start" not in _all_tags(scan)
    assert "context_anchor_gap" not in _all_tags(scan)

def test_evaluative_wordlist_no_longer_tags():
    # "important" + "challenge" with no first-person tripped _author_anchor_gap.
    scan = scan_text("It is important to address the challenge that the situation presents to everyone.")
    assert "author_anchor_gap" not in _all_tags(scan)
    assert "unsupported_claim_gap" not in _all_tags(scan)

# --- Structural-agnostic detectors MUST survive -----------------------------------------------

def test_list_pressure_structural_tag_survives():
    scan = scan_text("The system tracks revenue, costs, staffing, latency, and uptime across teams.")
    assert "packed_list" in _all_tags(scan)

def test_repeated_frame_structural_finding_survives():
    text = (
        "The team measured output. The team measured latency. The team measured cost. "
        "The team measured uptime."
    )
    scan = scan_text(text)
    assert "repeated_sentence_frame" in _all_tags(scan)
```

- [ ] **Step 2: Run to confirm the content-word tests FAIL and structural tests PASS**

Run: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py -v`
Expected: `test_*_wordlist_no_longer_tags` FAIL (tags still present); `test_*_structural_*` PASS.

- [ ] **Step 3: Commit the characterization tests**

```bash
git add poc/test_rewrite_v6_agnostic_scan.py
git commit -m "test: characterize agnostic scan targeting (content-word detectors must drop)"
```

---

## Task 2: Remove content-word detectors from `_tags` and `_risk`

**Files:**
- Modify: `poc/rewrite_v6/scan.py:463-490` (`_tags`), `:493-507` (`_risk`)

- [ ] **Step 1: Rewrite `_tags` to keep only structural-agnostic tags**

Replace the body of `_tags` (`scan.py:463`) with:

```python
def _tags(sentence: Sentence) -> list[str]:
    text = sentence.text
    tags: list[str] = []
    if _list_pressure(text) >= 0.22:
        tags.append("packed_list")
    if _abstract_risk_pressure(sentence) >= 12.0:
        tags.append("abstract_density")
    if sentence.word_count >= 22:
        tags.append("sentence_overload")
    if _citation_anchor(text):
        tags.append("citation_anchor")
    if _paraphrase_smoothing(sentence):
        tags.append("paraphrase_smoothing")
    return tags
```

- [ ] **Step 2: Rewrite `_risk` to drop the content-word terms**

Replace the body of `_risk` (`scan.py:493`) with:

```python
def _risk(sentence: Sentence) -> float:
    return (
        _list_pressure(sentence.text) * 36.0
        + _abstract_risk_pressure(sentence)
        + (6.0 if _citation_anchor(sentence.text) else 0.0)
        + (6.0 if _paraphrase_smoothing(sentence) else 0.0)
        + min(12.0, max(0, sentence.word_count - 18) * 0.8)
    )
```

- [ ] **Step 3: Run the Task 1 tests**

Run: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py -v`
Expected: all `test_*_wordlist_no_longer_tags` now PASS; structural tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add poc/rewrite_v6/scan.py
git commit -m "refactor(scan): drop content-word detectors from _tags/_risk (agnostic targeting)"
```

---

## Task 3: Reduce `_citation_anchor` to its structural form and delete dead functions

**Files:**
- Modify: `poc/rewrite_v6/scan.py:595-598` (`_citation_anchor`)
- Modify: `poc/rewrite_v6/scan.py` — delete the now-unused detector functions

- [ ] **Step 1: Write a test asserting `_citation_anchor` is structural, not verb-list-based**

Add to `poc/test_rewrite_v6_agnostic_scan.py`:

```python
from poc.rewrite_v6.scan import _citation_anchor

def test_citation_anchor_is_structural_form_not_verb_list():
    # Structural citation FORM still recognized.
    assert _citation_anchor("Smith et al. (2019) reported a measurable shift.") is True
    # Bare reporting verb with no citation form is NO LONGER a citation tell.
    assert _citation_anchor("The author indicates that the result holds.") is False
    assert _citation_anchor("According to many, the trend continued.") is False
```

- [ ] **Step 2: Run to verify it fails on the second/third assertions**

Run: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py::test_citation_anchor_is_structural_form_not_verb_list -v`
Expected: FAIL (current regex matches `according to` and `indicates`).

- [ ] **Step 3: Reduce `_citation_anchor` to structural form**

Replace `_citation_anchor` (`scan.py:595`) with:

```python
def _citation_anchor(text: str) -> bool:
    # Structural citation FORM only (author + year, or parenthetical year) -- agnostic to the
    # reporting verb / phrasing. No hardcoded verb vocabulary.
    return bool(re.search(r"\b\w+\s+et al\.\s*\(\d{4}\)", text)) or _parenthetical_citation(text)
```

- [ ] **Step 4: Delete the now-unused detector functions**

Delete these functions entirely from `scan.py` (verify each has no remaining caller first with a grep):
`_predictable_start_pressure`, `_predictable_start`, `_context_anchor_gap`, `_author_anchor_gap`,
`_broad_claim`, `_transition_stack`, `_semantic_bridge_gap`, `_unsupported_claim_gap`.

Verify no references remain:

```bash
cd poc && grep -nE "_predictable_start|_context_anchor_gap|_author_anchor_gap|_broad_claim|_transition_stack|_semantic_bridge_gap|_unsupported_claim_gap" rewrite_v6/scan.py
```
Expected: no output (all deleted, no callers).

- [ ] **Step 5: Run the full module + import smoke check**

Run: `cd poc && python -c "import rewrite_v6.scan" && python -m pytest test_rewrite_v6_agnostic_scan.py -v`
Expected: import OK; all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add poc/rewrite_v6/scan.py poc/test_rewrite_v6_agnostic_scan.py
git commit -m "refactor(scan): citation_anchor structural-only; delete dead content-word detectors"
```

---

## Task 4: Drop the `_predictability_primary` coupling

**Files:**
- Modify: `poc/rewrite_v6/scan.py:14-27` (`_predictability_primary`, `_ADVISORY_SEVERITY`), `:355-392` (`_merge_findings`)

**Why:** `_predictability_primary()` (env `DRAFTPROOF_V6_SCANNER_PREDICTABILITY`) conflated two behaviors — "demote the structural layer" and "rank by raw predictability." With the content-word detectors gone, the structural layer is already agnostic and minimal; the demotion branch is no longer needed, and we do not want raw predictability ranking. Remove the switch so behavior is unconditional.

- [ ] **Step 1: Write a test that merge keeps grounding severity without the env switch**

Add to `poc/test_rewrite_v6_agnostic_scan.py`:

```python
import os

def _grounding_report(sentence_text, *, paragraph_id="p001", sentence_id="s1",
                      actionability="auto_fixable", title="low_specificity", category="genericity"):
    return {
        "scan_intelligence": {"document": {"paragraphs": [{"paragraph_id": paragraph_id}]}},
        "sentence_map": {sentence_id: {"paragraph_id": paragraph_id, "text": sentence_text}},
        "highlight_segments": [
            {
                "sentence_id": sentence_id,
                "signals": [
                    {"title": title, "category": category, "score": 0.72,
                     "actionability": actionability, "finding_id": "f1"}
                ],
            }
        ],
    }

def test_grounding_finding_drives_targeting_without_env_switch(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_SCANNER_PREDICTABILITY", raising=False)
    text = "The approach generally improves outcomes for the relevant population over time."
    scan = scan_text_with_report(text, _grounding_report(text))
    pid = scan.paragraphs[0].id
    tags = {t for f in findings_for_paragraph(scan, pid) for t in f.tags}
    assert any("specificity" in t or "genericity" in t for t in tags)
```

- [ ] **Step 2: Run to verify it passes already OR fails — record baseline**

Run: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py::test_grounding_finding_drives_targeting_without_env_switch -v`
Expected: PASS (grounding finding present). If FAIL, the report-alignment fixture is wrong — fix the fixture (sentence text must match `sentence_map` exactly), not the source, before continuing.

- [ ] **Step 3: Remove the switch and simplify `_merge_findings`**

Delete `_predictability_primary` (`scan.py:14-22`) and `_ADVISORY_SEVERITY` (`scan.py:26-27`).
Replace `_merge_findings` (`scan.py:355`) with the unconditional legacy-merge form:

```python
def _merge_findings(base: list[Finding], report_findings: list[Finding]) -> list[Finding]:
    merged: dict[str, Finding] = {finding.sentence_id: finding for finding in base}
    for finding in report_findings:
        current = merged.get(finding.sentence_id)
        if current is None:
            merged[finding.sentence_id] = finding
            continue
        report_evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
        merged_evidence = {**current.evidence, "report_evidence": report_evidence}
        if report_evidence.get("predictability"):
            merged_evidence["predictability"] = report_evidence["predictability"]
        merged[finding.sentence_id] = Finding(
            sentence_id=current.sentence_id,
            paragraph_id=current.paragraph_id,
            tags=_dedupe([*current.tags, *finding.tags]),
            severity=max(current.severity, finding.severity),
            evidence=merged_evidence,
        )
    return list(merged.values())
```

- [ ] **Step 4: Verify no remaining references to the deleted names**

```bash
cd poc && grep -nE "_predictability_primary|_ADVISORY_SEVERITY|predictability_primary" rewrite_v6/scan.py
```
Expected: no output.

- [ ] **Step 5: Run the module**

Run: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add poc/rewrite_v6/scan.py poc/test_rewrite_v6_agnostic_scan.py
git commit -m "refactor(scan): remove predictability-primary switch; unconditional grounding merge"
```

---

## Task 5: Don't let un-mitigable `review_only` signals create a driving finding

**Files:**
- Modify: `poc/rewrite_v6/scan.py:211-281` (`_report_findings`)

**Why:** A segment whose ONLY report signal is predictability is labeled `review_only` by the report (`report.py:293`, "auto-rewriting it makes things worse"). Today `_report_findings` still emits a finding for it → the direct path rewrites that paragraph. We keep the predictability detail as annotation but stop it from being the *sole* reason a paragraph is targeted.

- [ ] **Step 1: Write the test — a predictability-only segment must not drive targeting**

Add to `poc/test_rewrite_v6_agnostic_scan.py`:

```python
def test_review_only_predictability_does_not_drive_targeting():
    text = "The framework adapts to the situation and supports the people who depend on it daily."
    report = _grounding_report(
        text, actionability="review_only", title="high_predictability", category="predictability"
    )
    scan = scan_text_with_report(text, report)
    pid = scan.paragraphs[0].id
    # No structural finding (sentence is not a packed list / overload), and the only report signal
    # is review_only predictability -> paragraph must NOT be flagged for rewrite.
    assert findings_for_paragraph(scan, pid) == []

def test_mixed_segment_still_drives_when_a_mitigable_signal_present():
    text = "The framework adapts to the situation and supports the people who depend on it daily."
    report = {
        "scan_intelligence": {"document": {"paragraphs": [{"paragraph_id": "p001"}]}},
        "sentence_map": {"s1": {"paragraph_id": "p001", "text": text}},
        "highlight_segments": [
            {"sentence_id": "s1", "signals": [
                {"title": "high_predictability", "category": "predictability",
                 "score": 0.8, "actionability": "review_only", "finding_id": "f1"},
                {"title": "low_specificity", "category": "genericity",
                 "score": 0.7, "actionability": "auto_fixable", "finding_id": "f2"},
            ]},
        ],
    }
    scan = scan_text_with_report(text, report)
    pid = scan.paragraphs[0].id
    assert findings_for_paragraph(scan, pid) != []
```

- [ ] **Step 2: Run to verify the first test fails**

Run: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py::test_review_only_predictability_does_not_drive_targeting -v`
Expected: FAIL (a finding is currently emitted for the review_only segment).

- [ ] **Step 3: Suppress review_only-only segments in `_report_findings`**

In `_report_findings` (`scan.py:255`), inside the `for sentence_id, entry in grouped.items():` loop,
after `tags = _dedupe([tag for tag in entry["tags"] if tag])` and the `if not tags: continue`
guard, add a driving-signal check before appending the finding:

```python
        actionability = _dedupe(entry["actionability"])
        # A segment whose ONLY actionability is review-level (e.g. high/medium predictability the
        # detector marks "review_only") must not be the sole reason a paragraph is rewritten --
        # predictability is un-mitigable by rewrite. Keep it as annotation only when it rides
        # alongside a mitigable signal; otherwise it does not drive targeting.
        mitigable = [a for a in actionability if a and a != "review_only"]
        if actionability and not mitigable:
            continue
```

Note: this uses `entry["actionability"]`, already collected at `scan.py:252`. Place the `continue`
so the finding is skipped only when every captured actionability value is `review_only` (or empty
after filtering). Segments with no actionability at all are unaffected (they fall through and emit
as before).

- [ ] **Step 4: Run both Task 5 tests**

Run: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py -k "review_only or mixed_segment" -v`
Expected: both PASS.

- [ ] **Step 5: Run the whole new module**

Run: `cd poc && python -m pytest test_rewrite_v6_agnostic_scan.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add poc/rewrite_v6/scan.py poc/test_rewrite_v6_agnostic_scan.py
git commit -m "feat(scan): review_only-only segments annotate but don't drive targeting"
```

---

## Task 6: Regression-guard the existing V6 suite + measure final_risk

**Files:**
- No new source. Runs existing tests + the deterministic harness.

- [ ] **Step 1: Run the existing scan/production/residual V6 tests**

Run:
```bash
cd poc && python -m pytest test_rewrite_v6_production_adapter.py test_rewrite_v6_residual_checker.py -v
```
Expected: PASS. If a test asserts on a now-removed tag (e.g. `broad_claim`, `transition_stack`),
read it: if it was asserting the *old hardcoded* behavior, update the assertion to the agnostic
behavior and note it in the commit. Do NOT re-introduce a removed detector to satisfy a stale test.

- [ ] **Step 2: Capture the baseline on `main` (separate worktree/checkout)**

Run (in a clean checkout of `main`, N≥4 — single runs are noise):
```bash
DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 4
```
Record the mean `final_risk`.

- [ ] **Step 3: Measure the branch**

Run (on this branch):
```bash
DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 4
```
Record the mean `final_risk`.

- [ ] **Step 4: Decision gate (no flag, fix-don't-toggle)**

- If branch mean `final_risk` ≤ main mean (within noise): PASS — proceed.
- If branch regresses: the agnostic targeting is under-flagging real grounding gaps. Fix in code —
  e.g. confirm the report's grounding signals (`low_specificity`/`uncited_claim`) are reaching
  `_report_findings` for the affected paragraphs, and that `paragraph_diagnosis` coverage is intact.
  Re-measure. Do NOT add a flag and do NOT restore a content-word detector.

- [ ] **Step 5: Commit the measurement record**

```bash
mkdir -p poc/test_output
# write the two means + N + date into poc/test_output/agnostic_scan_measure.md
git add poc/test_output/agnostic_scan_measure.md
git commit -m "test(scan): record final_risk main-vs-branch for agnostic targeting (N=4)"
```

---

## Self-Review

**Spec coverage:**
- Spec "remove content-word detectors" → Tasks 2, 3. ✓
- Spec "targeting driven by content-derived grounding signals" → Tasks 1, 4 (merge keeps grounding severity). ✓
- Spec "honor actionability; predictability annotates, never drives" → Task 5. ✓
- Spec "no flag, unconditional" → Task 4 removes the env switch; no new flag anywhere. ✓
- Spec "re-scan passes use agnostic estimators" → covered by construction: `scan_text`/`scan_text_preserve_blocks` now emit only structural-agnostic findings, and `direct_rewrite.py:96` already calls `estimate_generic_assertion_risk`/`estimate_lived_detail_risk` for residuals (unchanged). No code change needed; Task 6 Step 1 regression-checks the residual suite. ✓
- Spec "dev-time measurement gate, fix-don't-toggle" → Task 6. ✓

**Placeholder scan:** measurement-record file content in Task 6 Step 5 is described, not templated — acceptable (it's a freeform record). No TBD/TODO/"handle edge cases" steps. ✓

**Type consistency:** `Finding(sentence_id, paragraph_id, tags, severity, evidence)` used consistently. `_report_findings` uses `entry["actionability"]` which is populated at `scan.py:252` (`entry["actionability"].append(...)`). `_citation_anchor`/`_parenthetical_citation` signatures unchanged. ✓

## Execution Handoff

(See message.)
