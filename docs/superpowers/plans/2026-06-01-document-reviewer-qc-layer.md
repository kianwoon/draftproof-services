# Document Reviewer (QC) Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-rewrite QC reviewer that reads the whole rewritten document, fixes residual AI-shaped patterns a per-paragraph writer can't see (e.g. 7/8 paragraphs opening "In my classroom"), guided by the 25 writing-craft guidelines, then runs the single final scan on the QC'd text.

**Architecture:** Three isolated units. (1) `residual_patterns.py` — pure deterministic detectors returning evidence (the safety net for cross-paragraph patterns). (2) `document_reviewer.py` — holds the 25-guideline rubric constant + one LLM call that returns corrected sentences only, spliced by verbatim match, each correction guarded against a pre-QC baseline score. (3) Wire-in in `direct_rewrite.run_direct_rewrite_all`: after best-of-N picks the winner → QC → one final scan. Order: **rewrite → QC → scan**. Writer untouched.

**Tech Stack:** Python 3, pytest, existing `poc/rewrite_v6` modules (`LLMGateway`, `scan_text`, `parse_json`, `_document_ai_risk`, `_has_broken_grammar`, `_severe_polarity_inversion`), gpt-oss-120b via Cerebras.

**Spec:** `docs/superpowers/specs/2026-06-01-document-reviewer-layer-design.md`

---

## Confirmed interfaces (do not guess — these are verified)

- `poc/rewrite_v6/pipeline.py:71` — `@dataclass DocumentResult` fields: `initial_scan: Scan`, `final_scan: Scan`, `rewritten_text: str`, `passes: list[Any]`, `pass_trace: list[dict] = field(default_factory=list)`, `final_text_before_quality_repair: str | None = None`.
- `poc/rewrite_v6/text.py` — `@dataclass Paragraph(id: str, index: int, text: str)` with `.is_heading` (≤3 words); `split_paragraphs(text) -> list[Paragraph]`; `word_count(text) -> int`.
- `poc/rewrite_v6/scan.py` — `scan_text(text) -> Scan`; `@dataclass Scan(text, paragraphs, findings)`.
- `poc/rewrite_v6/json_io.py` — `parse_json(text) -> Any` (tolerant; may raise/return non-dict).
- `poc/rewrite_v6/direct_rewrite.py` — `_document_ai_risk(text) -> float` (real-detector AI likelihood, +inf if unscorable); `_has_broken_grammar(candidate: str) -> bool`; `run_direct_rewrite_all(...)` builds `gateway` and `return best_doc` at line 358.
- `poc/rewrite_v6/selector_diagnostics.py:36` — `_severe_polarity_inversion(candidate: str, paragraph: Paragraph) -> bool`.
- `poc/detect/layer3_scoring.py` — `split_sentences(text)` (returns a list of sentence strings despite the `-> str` annotation; used as an iterable in `direct_rewrite._ungrounded_claims`).
- `poc/llm/gateway.py:30` — `@dataclass LLMResponse(content: str, raw_content: str | None = None)`. `LLMGateway.chat(prompt, *, system=None, **kwargs)` returns an object with `.content` / `.raw_content`.
- **Test stub gateway pattern** (from `poc/test_rewrite_v6_author_voice.py`):
  ```python
  class _StubGateway:
      def __init__(self, payloads):
          self._payloads = list(payloads)
          self.calls = []
      def chat(self, prompt, *, system=None, **kwargs):
          from types import SimpleNamespace
          self.calls.append({"prompt": prompt, "system": system, "kwargs": kwargs})
          payload = self._payloads.pop(0) if self._payloads else "{}"
          return SimpleNamespace(content=payload, raw_content=payload)
  ```

---

## File Structure

| File | Responsibility |
|---|---|
| `poc/rewrite_v6/residual_patterns.py` (new) | Pure deterministic detectors over full text → `list[ResidualIssue]` (evidence, no LLM). The safety net. |
| `poc/rewrite_v6/document_reviewer.py` (new) | `WRITING_CRAFT_GUIDELINES` constant (the 25); `review_document()` = build prompt → 1 LLM call → splice by verbatim match → per-correction fidelity guard vs baseline → `ReviewResult`. |
| `poc/rewrite_v6/direct_rewrite.py` (modify) | `reviewer_enabled()` kill switch; after `best_doc`, run QC then one final scan; append corrections to `pass_trace`. |
| `poc/test_rewrite_v6_document_reviewer.py` (new) | Detector tests + orchestrator tests (stub gateway) + wire-in/order tests. |

---

## Task 1: Detector scaffolding — `ResidualIssue` + `detect_residual_patterns` skeleton

**Files:**
- Create: `poc/rewrite_v6/residual_patterns.py`
- Test: `poc/test_rewrite_v6_document_reviewer.py`

- [ ] **Step 1: Write the failing test**

Create `poc/test_rewrite_v6_document_reviewer.py`:

```python
from __future__ import annotations

import json
from types import SimpleNamespace

from poc.rewrite_v6 import residual_patterns
from poc.rewrite_v6.residual_patterns import ResidualIssue, detect_residual_patterns


def test_detect_returns_list_of_issues():
    # An empty / trivially-clean doc yields no issues.
    assert detect_residual_patterns("") == []
    assert detect_residual_patterns("A single short paragraph.") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py::test_detect_returns_list_of_issues -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.rewrite_v6.residual_patterns'`.

- [ ] **Step 3: Write minimal implementation**

Create `poc/rewrite_v6/residual_patterns.py`:

```python
"""Deterministic residual-pattern detectors for the QC reviewer (no LLM, no network).

The per-paragraph writer (direct_rewrite) is blind across paragraphs, so it can trade one AI
signal (generic assertion) for another (repetitive structure) -- e.g. 7/8 paragraphs opening
"In my classroom". These pure functions measure such patterns across the FULL rewritten document
and return evidence the QC reviewer must fix. They never edit text; they only report.

Content-agnostic structural/linguistic measures only (frame repetition, sentence-length variance,
closed-class connectives) -- not a banned-phrase or domain list. Aligns with the existing
_POLARITY_MARKERS / _CITATION_MARKERS closed-set precedent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ResidualIssue:
    rule: str                                   # e.g. "opener_monoculture"
    trick_ids: list[int]                        # guideline numbers, e.g. [19, 2]
    evidence: str                               # human-readable, e.g. "paragraphs 1,2,3,7 open 'In my'"
    target_sentences: list[str] = field(default_factory=list)  # exact sentences QC must fix


def detect_residual_patterns(text: str) -> list[ResidualIssue]:
    """Run every detector over the full document; return all fired issues (may be empty)."""
    issues: list[ResidualIssue] = []
    # Detectors are added in later tasks.
    return issues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py::test_detect_returns_list_of_issues -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
git add poc/rewrite_v6/residual_patterns.py poc/test_rewrite_v6_document_reviewer.py
git commit -m "feat(rewrite): residual-pattern detector scaffolding (ResidualIssue)"
```

---

## Task 2: `opener_monoculture` detector (#19, #2) — the primary detector

**Files:**
- Modify: `poc/rewrite_v6/residual_patterns.py`
- Test: `poc/test_rewrite_v6_document_reviewer.py`

- [ ] **Step 1: Write the failing test**

Append to `poc/test_rewrite_v6_document_reviewer.py`:

```python
# The real reverted-rewrite sample: 7/8 paragraphs open "In my ...".
_MONOCULTURE_DOC = "\n\n".join([
    "In my classroom, I have seen curriculum changes outpace the tweaks my school can make.",
    "In my classroom, I see students navigating a flood of digital resources every single day.",
    "In my classroom, I have seen the shift toward project-based learning change the teacher role.",
    "In my years teaching, I keep noticing that many schools still cling to outdated routines.",
    "In my teaching, I have noticed AI platforms let students sketch essay outlines in minutes.",
    "One inequality I see is the technology gap that splits the class into haves and have-nots.",
    "In my classroom, I have seen that the education system must evolve beyond rote memorization.",
    "In my experience as a teacher, I feel we are at a crossroads about what assessment means.",
])

_VARIED_DOC = "\n\n".join([
    "Curriculum changes outpace the tweaks a school can make.",
    "Students now navigate a flood of digital resources every single day.",
    "Project-based learning has quietly changed what a teacher does.",
    "Many schools still cling to outdated routines despite the evidence.",
    "AI platforms let students sketch essay outlines in minutes.",
])


def test_opener_monoculture_fires_on_repeated_in_my():
    issues = detect_residual_patterns(_MONOCULTURE_DOC)
    monoc = [i for i in issues if i.rule == "opener_monoculture"]
    assert len(monoc) == 1
    issue = monoc[0]
    assert 19 in issue.trick_ids
    # the repeated-opener sentences are handed over as targets (the "In my" paragraphs)
    assert any("In my classroom" in s for s in issue.target_sentences)
    assert len(issue.target_sentences) >= 4


def test_opener_monoculture_silent_on_varied_doc():
    issues = detect_residual_patterns(_VARIED_DOC)
    assert [i for i in issues if i.rule == "opener_monoculture"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k opener_monoculture -v`
Expected: FAIL (`test_opener_monoculture_fires_on_repeated_in_my` — assert len == 1 fails, gets 0).

- [ ] **Step 3: Write minimal implementation**

In `poc/rewrite_v6/residual_patterns.py`, add constants near the top (after imports) and the detector, and call it from `detect_residual_patterns`:

```python
# A document needs at least this many paragraphs before opener repetition is meaningful.
_MIN_PARAGRAPHS_FOR_OPENER_CHECK = 3
# If this fraction (or more) of paragraphs share the same first-2-word opener frame, it reads as a
# monoculture (the writer's blind-spot: every paragraph independently picked the same frame).
_OPENER_SHARE_THRESHOLD = 0.5
# Number of leading words that define an "opener frame".
_OPENER_FRAME_WORDS = 2


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]


def _first_sentence(paragraph: str) -> str:
    # Cheap sentence head: up to the first ./!/? (keep the terminator), else the whole paragraph.
    match = re.search(r"^.*?[.!?](?:\s|$)", paragraph.strip())
    return (match.group(0).strip() if match else paragraph.strip())


def _opener_frame(paragraph: str, n: int = _OPENER_FRAME_WORDS) -> str:
    words = re.findall(r"[A-Za-z'’]+", paragraph)
    return " ".join(w.lower() for w in words[:n])


def _detect_opener_monoculture(text: str) -> ResidualIssue | None:
    paras = _paragraphs(text)
    if len(paras) < _MIN_PARAGRAPHS_FOR_OPENER_CHECK:
        return None
    frames = [_opener_frame(p) for p in paras]
    counts: dict[str, int] = {}
    for f in frames:
        if f:
            counts[f] = counts.get(f, 0) + 1
    if not counts:
        return None
    top_frame, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count < 2 or top_count / len(paras) < _OPENER_SHARE_THRESHOLD:
        return None
    targets = [_first_sentence(p) for p, f in zip(paras, frames) if f == top_frame]
    evidence = (
        f"{top_count} of {len(paras)} paragraphs open with the same frame "
        f"'{top_frame}' -- vary the openings (guideline 19)."
    )
    return ResidualIssue(rule="opener_monoculture", trick_ids=[19, 2],
                         evidence=evidence, target_sentences=targets)
```

Then update `detect_residual_patterns`:

```python
def detect_residual_patterns(text: str) -> list[ResidualIssue]:
    """Run every detector over the full document; return all fired issues (may be empty)."""
    issues: list[ResidualIssue] = []
    opener = _detect_opener_monoculture(text)
    if opener is not None:
        issues.append(opener)
    return issues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k opener_monoculture -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
git add poc/rewrite_v6/residual_patterns.py poc/test_rewrite_v6_document_reviewer.py
git commit -m "feat(rewrite): opener_monoculture detector (#19/#2) — catches repeated 'In my' openers"
```

---

## Task 3: `robotic_transitions` detector (#8)

**Files:**
- Modify: `poc/rewrite_v6/residual_patterns.py`
- Test: `poc/test_rewrite_v6_document_reviewer.py`

- [ ] **Step 1: Write the failing test**

Append to `poc/test_rewrite_v6_document_reviewer.py`:

```python
def test_robotic_transitions_fires():
    doc = (
        "Schools changed fast.\n\n"
        "Furthermore, the curriculum widened beyond the textbook every year.\n\n"
        "Moreover, students began learning from sources teachers could not control.\n\n"
        "In conclusion, the old model no longer matches how learners seek knowledge."
    )
    issues = detect_residual_patterns(doc)
    robotic = [i for i in issues if i.rule == "robotic_transitions"]
    assert len(robotic) == 1
    assert 8 in robotic[0].trick_ids
    # the offending sentences are handed over
    assert any(s.lower().startswith("furthermore") for s in robotic[0].target_sentences)
    assert len(robotic[0].target_sentences) >= 2


def test_robotic_transitions_silent_when_absent():
    doc = "Schools changed fast.\n\nThat created a new problem teachers had to solve themselves."
    assert [i for i in detect_residual_patterns(doc) if i.rule == "robotic_transitions"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k robotic_transitions -v`
Expected: FAIL (gets 0 issues).

- [ ] **Step 3: Write minimal implementation**

In `poc/rewrite_v6/residual_patterns.py`, add the closed set + detector, and wire it in:

```python
# Closed-class formulaic transition openers (guideline 8). Sentence-initial only.
_ROBOTIC_TRANSITIONS = (
    "furthermore", "moreover", "additionally", "in addition", "in conclusion",
    "consequently", "thus", "therefore", "as a result", "overall", "to conclude",
    "in summary", "firstly", "secondly", "thirdly", "lastly", "notably",
)
# Need at least this many robotic openers in the doc before flagging (1 is fine in good prose).
_MIN_ROBOTIC_HITS = 2


def _sentences(text: str) -> list[str]:
    # Split the whole document into sentences for transition scanning.
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").replace("\n", " ").strip())
    return [p.strip() for p in parts if p.strip()]


def _starts_with_robotic(sentence: str) -> bool:
    low = sentence.strip().lower()
    return any(low.startswith(t + " ") or low.startswith(t + ",") for t in _ROBOTIC_TRANSITIONS)


def _detect_robotic_transitions(text: str) -> ResidualIssue | None:
    hits = [s for s in _sentences(text) if _starts_with_robotic(s)]
    if len(hits) < _MIN_ROBOTIC_HITS:
        return None
    evidence = (
        f"{len(hits)} sentences open with a formulaic transition "
        f"(Furthermore/Moreover/In conclusion ...) -- use cause-based transitions instead "
        f"(guideline 8)."
    )
    return ResidualIssue(rule="robotic_transitions", trick_ids=[8],
                         evidence=evidence, target_sentences=hits)
```

Update `detect_residual_patterns` to also call it:

```python
    robotic = _detect_robotic_transitions(text)
    if robotic is not None:
        issues.append(robotic)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k robotic_transitions -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
git add poc/rewrite_v6/residual_patterns.py poc/test_rewrite_v6_document_reviewer.py
git commit -m "feat(rewrite): robotic_transitions detector (#8)"
```

---

## Task 4: `repeated_subject_starts` (#19 within-paragraph) + `balance_phrase` (#7)

**Files:**
- Modify: `poc/rewrite_v6/residual_patterns.py`
- Test: `poc/test_rewrite_v6_document_reviewer.py`

- [ ] **Step 1: Write the failing test**

Append to `poc/test_rewrite_v6_document_reviewer.py`:

```python
def test_repeated_subject_starts_fires():
    doc = (
        "Technology helps students learn. Technology also distracts them constantly. "
        "Technology shapes how teachers plan every lesson now."
    )
    issues = detect_residual_patterns(doc)
    rep = [i for i in issues if i.rule == "repeated_subject_starts"]
    assert len(rep) == 1
    assert 19 in rep[0].trick_ids


def test_balance_phrase_fires():
    doc = "AI in the classroom brings both opportunities and risks for every learner involved."
    issues = detect_residual_patterns(doc)
    bal = [i for i in issues if i.rule == "balance_phrase"]
    assert len(bal) == 1
    assert 7 in bal[0].trick_ids


def test_balance_phrase_silent_when_specific():
    doc = "AI helps students brainstorm at the planning stage, but it hides gaps they cannot explain."
    assert [i for i in detect_residual_patterns(doc) if i.rule == "balance_phrase"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "repeated_subject_starts or balance_phrase" -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

In `poc/rewrite_v6/residual_patterns.py`, add both detectors and wire them in:

```python
# Same first word more than this many times within one paragraph reads as a repeated subject start.
_MAX_SAME_FIRST_WORD = 2
# "both X and Y" abstract balance phrasing (guideline 7).
_BALANCE_RE = re.compile(r"\bboth\b[^.?!]{0,60}?\band\b", re.I)


def _first_word(sentence: str) -> str:
    words = re.findall(r"[A-Za-z'’]+", sentence)
    return words[0].lower() if words else ""


def _detect_repeated_subject_starts(text: str) -> ResidualIssue | None:
    flagged: list[str] = []
    for para in _paragraphs(text):
        sents = _sentences(para)
        counts: dict[str, int] = {}
        for s in sents:
            fw = _first_word(s)
            if fw:
                counts[fw] = counts.get(fw, 0) + 1
        for fw, c in counts.items():
            if c > _MAX_SAME_FIRST_WORD:
                flagged.extend(s for s in sents if _first_word(s) == fw)
    if not flagged:
        return None
    evidence = (
        "Several sentences in a paragraph start with the same word -- rotate the openings "
        "(guideline 19)."
    )
    return ResidualIssue(rule="repeated_subject_starts", trick_ids=[19],
                         evidence=evidence, target_sentences=flagged)


def _detect_balance_phrase(text: str) -> ResidualIssue | None:
    hits = [s for s in _sentences(text) if _BALANCE_RE.search(s)]
    if not hits:
        return None
    evidence = (
        "A 'both X and Y' balance phrase states the shape of a trade-off without the actual "
        "benefit or risk -- name the specific benefit and the specific risk (guideline 7)."
    )
    return ResidualIssue(rule="balance_phrase", trick_ids=[7],
                         evidence=evidence, target_sentences=hits)
```

Wire both into `detect_residual_patterns`:

```python
    repeated = _detect_repeated_subject_starts(text)
    if repeated is not None:
        issues.append(repeated)
    balance = _detect_balance_phrase(text)
    if balance is not None:
        issues.append(balance)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "repeated_subject_starts or balance_phrase" -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
git add poc/rewrite_v6/residual_patterns.py poc/test_rewrite_v6_document_reviewer.py
git commit -m "feat(rewrite): repeated_subject_starts (#19) + balance_phrase (#7) detectors"
```

---

## Task 5: `rhythm_sameness` detector (#13) + full false-positive guard

**Files:**
- Modify: `poc/rewrite_v6/residual_patterns.py`
- Test: `poc/test_rewrite_v6_document_reviewer.py`

- [ ] **Step 1: Write the failing test**

Append to `poc/test_rewrite_v6_document_reviewer.py`:

```python
def test_rhythm_sameness_fires_on_uniform_lengths():
    # Five sentences, all ~8 words -> very low length variance.
    doc = (
        "Students learn many new things every single day. "
        "Teachers plan many small lessons every single week. "
        "Schools change many old rules every single year. "
        "Parents ask many hard questions every single term. "
        "Leaders make many big choices every single month."
    )
    issues = detect_residual_patterns(doc)
    assert [i for i in issues if i.rule == "rhythm_sameness"], "expected rhythm_sameness to fire"
    assert 13 in next(i for i in issues if i.rule == "rhythm_sameness").trick_ids


def test_rhythm_sameness_silent_on_varied_lengths():
    doc = (
        "Schools change. "
        "When a student opens a laptop in class, the lesson the teacher planned a week earlier "
        "suddenly has to compete with a dozen brighter, faster, louder sources of information. "
        "That shift matters."
    )
    assert [i for i in detect_residual_patterns(doc) if i.rule == "rhythm_sameness"] == []


def test_clean_human_doc_has_no_issues_at_all():
    # Composite false-positive guard across all detectors.
    doc = _VARIED_DOC
    assert detect_residual_patterns(doc) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "rhythm or clean_human" -v`
Expected: FAIL (`rhythm_sameness` not implemented).

- [ ] **Step 3: Write minimal implementation**

In `poc/rewrite_v6/residual_patterns.py`, add:

```python
# A paragraph with >= this many sentences whose length coefficient-of-variation is below the
# threshold reads as mechanically uniform rhythm (guideline 13).
_MIN_SENTENCES_FOR_RHYTHM = 4
_RHYTHM_CV_THRESHOLD = 0.20  # std/mean of sentence word-counts; below this is "too even"


def _coefficient_of_variation(values: list[int]) -> float:
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 1.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (variance ** 0.5) / mean


def _detect_rhythm_sameness(text: str) -> ResidualIssue | None:
    worst: list[str] = []
    for para in _paragraphs(text):
        sents = _sentences(para)
        if len(sents) < _MIN_SENTENCES_FOR_RHYTHM:
            continue
        lengths = [len(re.findall(r"[A-Za-z'’]+", s)) for s in sents]
        if _coefficient_of_variation(lengths) < _RHYTHM_CV_THRESHOLD:
            worst.extend(sents)
    if not worst:
        return None
    evidence = (
        "A paragraph's sentences are all about the same length -- vary the rhythm with a short "
        "sentence next to a longer one (guideline 13)."
    )
    return ResidualIssue(rule="rhythm_sameness", trick_ids=[13],
                         evidence=evidence, target_sentences=worst)
```

Wire it into `detect_residual_patterns`:

```python
    rhythm = _detect_rhythm_sameness(text)
    if rhythm is not None:
        issues.append(rhythm)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -v`
Expected: PASS (all detector tests, including the composite clean-doc guard).

- [ ] **Step 5: Commit**

```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
git add poc/rewrite_v6/residual_patterns.py poc/test_rewrite_v6_document_reviewer.py
git commit -m "feat(rewrite): rhythm_sameness detector (#13) + clean-doc false-positive guard"
```

---

## Task 6: The 25-guideline rubric constant + reviewer prompt builder

**Files:**
- Create: `poc/rewrite_v6/document_reviewer.py`
- Test: `poc/test_rewrite_v6_document_reviewer.py`

- [ ] **Step 1: Write the failing test**

Append to `poc/test_rewrite_v6_document_reviewer.py`:

```python
from poc.rewrite_v6 import document_reviewer
from poc.rewrite_v6.document_reviewer import (
    WRITING_CRAFT_GUIDELINES,
    build_reviewer_prompt,
    reviewer_enabled,
)


def test_rubric_has_all_25_guidelines():
    assert len(WRITING_CRAFT_GUIDELINES) == 25


def test_prompt_includes_doc_rubric_and_must_fix_evidence():
    from poc.rewrite_v6.residual_patterns import ResidualIssue
    must_fix = [ResidualIssue(rule="opener_monoculture", trick_ids=[19],
                              evidence="4 of 8 paragraphs open 'in my'",
                              target_sentences=["In my classroom, I have seen X."])]
    prompt = build_reviewer_prompt("FULL DOC TEXT HERE", must_fix)
    assert "FULL DOC TEXT HERE" in prompt
    assert "4 of 8 paragraphs open 'in my'" in prompt          # evidence present
    assert "In my classroom, I have seen X." in prompt          # target sentence present
    assert "corrections" in prompt                              # output schema named
    # rubric is present (sample a couple of guideline keywords)
    low = prompt.lower()
    assert "vary" in low and "transition" in low


def test_reviewer_enabled_default_on(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_V6_REVIEWER", raising=False)
    assert reviewer_enabled() is True
    monkeypatch.setenv("DRAFTPROOF_V6_REVIEWER", "0")
    assert reviewer_enabled() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "rubric or prompt_includes or reviewer_enabled" -v`
Expected: FAIL (`ModuleNotFoundError: poc.rewrite_v6.document_reviewer`).

- [ ] **Step 3: Write minimal implementation**

Create `poc/rewrite_v6/document_reviewer.py`:

```python
"""QC reviewer: a document-aware pass that polishes the writer's rewrite into a teaching-grade
showcase.

Order in the pipeline: rewrite (writer) -> QC (this) -> final scan. The writer rewrites one
paragraph per LLM call, blind to the others, so it can produce cross-paragraph monocultures
(e.g. 7/8 paragraphs opening "In my classroom"). This reviewer reads the FULL document, is guided
by the 25 writing-craft guidelines, and corrects substandard sentences. Deterministic detectors
(residual_patterns) run as a safety net guaranteeing the cross-paragraph patterns are always caught.

Surgical by design: the reviewer returns only corrected sentences (not the whole doc), so output
stays small regardless of length -- avoiding the empty/truncated gpt-oss response that reverted the
prior showcase. Each correction is guarded against a pre-QC baseline score; a regressing or broken
correction is dropped (the writer's sentence is kept). NOT a humanizer/detection-evasion tool.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .json_io import parse_json
from .residual_patterns import ResidualIssue, detect_residual_patterns


# The 25 writing-craft guidelines (the user's list). Legitimate, content-agnostic craft guidance
# -- NOT a hardcode of domain answers. One source of truth, mirrored as problem -> fix.
WRITING_CRAFT_GUIDELINES: list[str] = [
    "1. Generic opening: start with pressure, not topic.",
    "2. Predictable start: avoid a broad noun followed by a broad claim.",
    "3. Weak context anchor: name the setting.",
    "4. Weak author anchor: add what the writer noticed.",
    "5. Packed list: split a long list into meaning groups (2-3 beats).",
    "6. Sentence overload: one job per sentence (claim / evidence / judgment).",
    "7. Balance-phrase filler: replace 'both opportunities and risks' with the actual benefit and the actual risk.",
    "8. Robotic transition: avoid Furthermore/Moreover/In conclusion; use cause-based transitions.",
    "9. Smooth but empty: add friction; show where the idea becomes difficult.",
    "10. Abstract nouns: convert nouns into actions (who does what).",
    "11. Weak judgment: say what should happen.",
    "12. Formulaic contrast: don't rely on a simple past-vs-present template; add the mechanism.",
    "13. Repeated sentence rhythm: vary short-long-short.",
    "14. Predictable paragraph arc: break the expected order.",
    "15. Generic benefit: attach the benefit to a user/action.",
    "16. Generic risk: attach the risk to a failure mode.",
    "17. Weak evidence: use a concrete classroom/workflow example.",
    "18. Over-polished wording: use normal human phrasing.",
    "19. Same subject starts: rotate sentence openings.",
    "20. No ownership: add a position, not just information.",
    "21. Dense academic phrasing: cut stacked modifiers.",
    "22. Weak source handling: attach the citation to the exact claim.",
    "23. AI-like conclusion: end with a consequence, not a slogan.",
    "24. Too-even paragraph shape: give each paragraph a distinct role.",
    "25. Rewrite drift: preserve the original idea first; improve expression without changing meaning.",
]

_SYSTEM = (
    "You are a writing QUALITY-CONTROL reviewer. You receive a draft that an automated rewriter "
    "produced one paragraph at a time, so it cannot see patterns that span the whole document. "
    "Your job: inspect the FULL draft against the writing-craft guidelines and correct sentences "
    "that fall short -- especially the must_fix issues, which were detected mechanically and MUST "
    "be resolved. Change ONLY what is substandard: vary repeated openings, replace robotic "
    "transitions, break uniform rhythm, and de-formulaic wording. NEVER remove or weaken the "
    "concrete, grounded specifics the draft already contains (names, figures, scenes, first-person "
    "facts) -- those are the point. Preserve every sentence's meaning and polarity. "
    "Return ONLY the corrected sentences, each as an exact-match replacement."
)


def reviewer_enabled() -> bool:
    """Kill switch. Default ON; set DRAFTPROOF_V6_REVIEWER=0 to disable (matches direct_rewrite)."""
    return os.environ.get("DRAFTPROOF_V6_REVIEWER", "1").strip().lower() not in {"0", "false", "no", "off"}


def build_reviewer_prompt(text: str, must_fix: list[ResidualIssue]) -> str:
    payload: dict[str, Any] = {
        "task": "qc_review_the_full_document",
        "guidelines": WRITING_CRAFT_GUIDELINES,
        "must_fix_issues": [
            {"issue": issue.rule, "evidence": issue.evidence,
             "sentences_to_fix": issue.target_sentences}
            for issue in must_fix
        ],
        "full_document": text,
        "instructions": [
            "Resolve every must_fix issue, and fix any other sentence that clearly falls short of "
            "the guidelines.",
            "Change only what is substandard. Keep all grounded specifics intact.",
            "Each correction's 'original' MUST be an exact substring of full_document so it can be "
            "spliced back. Quote it verbatim, including punctuation.",
            "Do not rewrite the whole document. Return only the sentences you actually change.",
        ],
        "output_schema": {
            "corrections": [{"original": "exact sentence from the document", "revised": "improved sentence"}]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "rubric or prompt_includes or reviewer_enabled" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
git add poc/rewrite_v6/document_reviewer.py poc/test_rewrite_v6_document_reviewer.py
git commit -m "feat(rewrite): QC rubric (25 guidelines) + reviewer prompt builder + kill switch"
```

---

## Task 7: `review_document` orchestrator — splice + per-correction fidelity guard

**Files:**
- Modify: `poc/rewrite_v6/document_reviewer.py`
- Test: `poc/test_rewrite_v6_document_reviewer.py`

- [ ] **Step 1: Write the failing test**

Append to `poc/test_rewrite_v6_document_reviewer.py`:

```python
def _stub(payloads):
    class _StubGateway:
        def __init__(self, payloads):
            self._payloads = list(payloads)
            self.calls = []
        def chat(self, prompt, *, system=None, **kwargs):
            self.calls.append({"prompt": prompt, "system": system})
            payload = self._payloads.pop(0) if self._payloads else "{}"
            return SimpleNamespace(content=payload, raw_content=payload)
    return _StubGateway(payloads)


def test_review_splices_correction_by_verbatim_match(monkeypatch):
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)  # everything improves/neutral
    doc = "In my classroom, I have seen change.\n\nIn my classroom, I have seen more change."
    correction = {"corrections": [
        {"original": "In my classroom, I have seen more change.",
         "revised": "Last spring, the change reached my own lesson plans."}
    ]}
    gw = _stub([json.dumps(correction)])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert "Last spring, the change reached my own lesson plans." in result.text
    assert "In my classroom, I have seen more change." not in result.text
    assert len(result.corrections) == 1


def test_review_skips_unmatched_original(monkeypatch):
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)
    doc = "In my classroom, I have seen change every term without fail."
    correction = {"corrections": [
        {"original": "A sentence that is not in the document.", "revised": "whatever"}
    ]}
    gw = _stub([json.dumps(correction)])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert result.text == doc                  # nothing spliced
    assert result.corrections == []


def test_review_drops_correction_that_raises_score(monkeypatch):
    # baseline = score of writer text; revised doc scores worse -> drop.
    calls = {"n": 0}
    def fake_score(t):
        # writer/baseline text scores 10; any doc containing the revised sentence scores 90.
        return 90.0 if "WORSE" in t else 10.0
    monkeypatch.setattr(document_reviewer, "_score", fake_score)
    doc = "In my classroom, I have seen change every single term here.\n\nIn my classroom, I have seen it twice."
    correction = {"corrections": [
        {"original": "In my classroom, I have seen it twice.", "revised": "This made things WORSE."}
    ]}
    gw = _stub([json.dumps(correction)])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert "WORSE" not in result.text          # regressing correction dropped
    assert result.text == doc


def test_review_returns_unchanged_on_bad_json(monkeypatch):
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)
    doc = "In my classroom, I have seen change happen quickly here."
    gw = _stub(["not json at all"])
    result = document_reviewer.review_document(doc, gateway=gw)
    assert result.text == doc
    assert result.corrections == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "review_splices or review_skips or review_drops or review_returns" -v`
Expected: FAIL (`review_document` / `_score` / `ReviewResult` not defined).

- [ ] **Step 3: Write minimal implementation**

In `poc/rewrite_v6/document_reviewer.py`, add the result type, the score/guard helpers, and the orchestrator. Add these imports at the top of the existing import block:

```python
from .selector_diagnostics import _severe_polarity_inversion
from .text import split_paragraphs
```

Then append:

```python
@dataclass
class Correction:
    original: str
    revised: str
    rule: str = "qc"


@dataclass
class ReviewResult:
    text: str
    corrections: list[Correction] = field(default_factory=list)
    skipped: str | None = None
    must_fix_unaddressed: list[str] = field(default_factory=list)


def _score(text: str) -> float:
    """Real-detector AI risk for the whole doc (baseline + post-correction guard). Wraps
    direct_rewrite._document_ai_risk; isolated here so tests can monkeypatch it."""
    from .direct_rewrite import _document_ai_risk
    return _document_ai_risk(text)


def _correction_is_safe(original: str, revised: str, *, doc_after: str, baseline: float) -> bool:
    """Keep a correction only if it doesn't regress score, break grammar, or invert polarity."""
    from .direct_rewrite import _has_broken_grammar
    from .text import Paragraph
    revised = (revised or "").strip()
    if not revised or len(revised.split()) < 3:
        return False
    if _has_broken_grammar(revised):
        return False
    # polarity must not flip vs the writer's original sentence
    if _severe_polarity_inversion(revised, Paragraph(id="qc", index=0, text=original)):
        return False
    # the whole doc with this correction applied must not score worse than baseline
    if _score(doc_after) > baseline:
        return False
    return True


def review_document(
    text: str,
    *,
    gateway: Any,
    cancellation_check: Callable[[], None] | None = None,
) -> ReviewResult:
    """QC the full rewritten document: detect must-fix patterns, ask the reviewer LLM for
    sentence-level corrections, splice safe ones by verbatim match. Never raises; on any failure
    returns the writer's text unchanged."""
    if cancellation_check:
        cancellation_check()
    must_fix = detect_residual_patterns(text)
    baseline = _score(text)
    prompt = build_reviewer_prompt(text, must_fix)
    try:
        response = gateway.chat(
            prompt,
            system=_SYSTEM,
            temperature=0.4,
            top_p=0.9,
            max_tokens=4000,
            response_format={"type": "json_object"},
            app_label="DocumentReviewer",
        )
        data = parse_json(getattr(response, "raw_content", "") or getattr(response, "content", "") or "")
    except Exception:
        return ReviewResult(text=text, corrections=[], skipped="llm_error")
    if not isinstance(data, dict):
        return ReviewResult(text=text, corrections=[], skipped="bad_json")

    current = text
    applied: list[Correction] = []
    for item in (data.get("corrections") or []):
        if not isinstance(item, dict):
            continue
        original = str(item.get("original") or "")
        revised = str(item.get("revised") or "")
        if not original or original not in current:
            continue  # unmatched -> skip (safe no-op)
        candidate_doc = current.replace(original, revised, 1)
        if _correction_is_safe(original, revised, doc_after=candidate_doc, baseline=baseline):
            current = candidate_doc
            applied.append(Correction(original=original, revised=revised))

    addressed = " ".join(c.original for c in applied)
    unaddressed = [
        issue.rule for issue in must_fix
        if not any(t in addressed for t in issue.target_sentences)
    ]
    return ReviewResult(text=current, corrections=applied, must_fix_unaddressed=unaddressed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "review_splices or review_skips or review_drops or review_returns" -v`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
git add poc/rewrite_v6/document_reviewer.py poc/test_rewrite_v6_document_reviewer.py
git commit -m "feat(rewrite): review_document orchestrator with splice + per-correction fidelity guard"
```

---

## Task 8: Wire QC into `direct_rewrite` — rewrite → QC → final scan

**Files:**
- Modify: `poc/rewrite_v6/direct_rewrite.py` (the `run_direct_rewrite_all` tail, around line 358 `return best_doc`)
- Test: `poc/test_rewrite_v6_document_reviewer.py`

- [ ] **Step 1: Write the failing test**

Append to `poc/test_rewrite_v6_document_reviewer.py`:

```python
from poc.rewrite_v6 import direct_rewrite as dr
from poc.rewrite_v6.pipeline import DocumentResult
from poc.rewrite_v6.scan import scan_text as _scan_text


def test_apply_reviewer_runs_qc_then_scans_qcd_text(monkeypatch):
    # writer produced a monoculture doc
    writer_text = "\n\n".join([
        "In my classroom, I have seen change arrive faster than the school can absorb it.",
        "In my classroom, I have seen students lean on AI before they ask me anything.",
        "In my classroom, I have seen the textbook lose its old authority in a single year.",
    ])
    doc = DocumentResult(
        initial_scan=_scan_text(writer_text),
        final_scan=_scan_text(writer_text),
        rewritten_text=writer_text,
        passes=[],
        pass_trace=[],
    )
    # stub QC: replace the first opener
    correction = {"corrections": [
        {"original": "In my classroom, I have seen change arrive faster than the school can absorb it.",
         "revised": "Change now arrives faster than my school can absorb it."}
    ]}
    gw = _stub([json.dumps(correction)])
    monkeypatch.setattr(document_reviewer, "_score", lambda t: 10.0)

    out = dr._apply_reviewer(doc, gw, cancellation_check=None)
    assert "Change now arrives faster than my school can absorb it." in out.rewritten_text
    # final_scan must reflect the QC'd text (order: rewrite -> QC -> scan)
    assert out.final_scan.text == out.rewritten_text
    # corrections recorded in the trace
    assert any(row.get("selected_source") == "qc_reviewer" for row in out.pass_trace)


def test_apply_reviewer_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_V6_REVIEWER", "0")
    writer_text = "A short clean paragraph that needs nothing at all."
    doc = DocumentResult(
        initial_scan=_scan_text(writer_text),
        final_scan=_scan_text(writer_text),
        rewritten_text=writer_text,
        passes=[],
        pass_trace=[],
    )
    gw = _stub(["{}"])
    out = dr._apply_reviewer(doc, gw, cancellation_check=None)
    assert out is doc            # untouched object when disabled
    assert gw.calls == []        # no LLM call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "apply_reviewer" -v`
Expected: FAIL (`dr._apply_reviewer` not defined).

- [ ] **Step 3: Write minimal implementation**

In `poc/rewrite_v6/direct_rewrite.py`, add a helper and call it from `run_direct_rewrite_all`.

First, add the helper function (place it just above `def run_direct_rewrite_all`):

```python
def _apply_reviewer(
    doc,
    gateway,
    *,
    cancellation_check: Callable[[], None] | None,
):
    """Run the QC reviewer on the writer's winning document, then re-scan the QC'd text.

    Order: rewrite -> QC -> scan. The reviewer fixes cross-paragraph patterns the per-paragraph
    writer can't see; the single authoritative final_scan is computed here, AFTER QC. On disable or
    any failure the writer's document is returned unchanged."""
    from .document_reviewer import reviewer_enabled, review_document
    from .pipeline import DocumentResult

    if not reviewer_enabled():
        return doc
    try:
        result = review_document(
            doc.rewritten_text, gateway=gateway, cancellation_check=cancellation_check
        )
    except Exception:
        return doc
    if not result.corrections:
        return doc  # nothing changed; keep writer's doc + its scan

    reviewed_text = result.text
    trace = list(doc.pass_trace)
    trace.append({
        "selected_source": "qc_reviewer",
        "status": "accepted",
        "corrections": [
            {"original": c.original, "revised": c.revised} for c in result.corrections
        ][:12],
        "must_fix_unaddressed": result.must_fix_unaddressed,
    })
    return DocumentResult(
        initial_scan=doc.initial_scan,
        final_scan=scan_text(reviewed_text),   # the ONE authoritative scan, post-QC
        rewritten_text=reviewed_text,
        passes=doc.passes,
        pass_trace=trace,
        final_text_before_quality_repair=doc.final_text_before_quality_repair,
    )
```

Then change the tail of `run_direct_rewrite_all` from:

```python
        if best_doc is None or score < best_score:
            best_doc, best_score = doc, score
    return best_doc
```

to:

```python
        if best_doc is None or score < best_score:
            best_doc, best_score = doc, score
    # rewrite -> QC -> scan: the reviewer fixes whole-document patterns the per-paragraph writer
    # can't see, then the single authoritative final scan runs on the QC'd text.
    return _apply_reviewer(best_doc, gateway, cancellation_check=cancellation_check)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -k "apply_reviewer" -v`
Expected: PASS (both).

- [ ] **Step 5: Run the FULL new test file + a smoke import**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_document_reviewer.py -v && python -c "import poc.rewrite_v6.direct_rewrite, poc.rewrite_v6.document_reviewer, poc.rewrite_v6.residual_patterns; print('imports OK')"`
Expected: all PASS, `imports OK`.

- [ ] **Step 6: Commit**

```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
git add poc/rewrite_v6/direct_rewrite.py poc/test_rewrite_v6_document_reviewer.py
git commit -m "feat(rewrite): wire QC reviewer into direct_rewrite (rewrite -> QC -> scan)"
```

---

## Task 9: Regression check — existing rewrite tests + no-regression measurement note

**Files:**
- No code change (verification task)

- [ ] **Step 1: Run the existing rewrite_v6 suite to confirm nothing broke**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_*.py -q`
Expected: all PASS (the QC layer is additive; writer path untouched).

- [ ] **Step 2: Run the direct-rewrite-adjacent tests explicitly**

Run: `cd /Users/kianwoonwong/Downloads/draftproof_services && python -m pytest poc/test_rewrite_v6_author_voice.py poc/test_rewrite_v6_fabrication.py poc/test_rewrite_v6_orchestration.py -q`
Expected: all PASS.

- [ ] **Step 3: Record the measurement command for a human to run with live LLM**

QC quality cannot be unit-tested end-to-end (needs the live gpt-oss writer). Document the manual check (do NOT run automatically; it costs LLM credits and is high-variance):

Run (manual, N≥4): `cd /Users/kianwoonwong/Downloads/draftproof_services && DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 4`
Expected: final_risk holds or improves vs the pre-QC baseline (per `project_v6_measurement_variance`, single runs are noise — compare means).

- [ ] **Step 4: Commit (if any doc note added) or skip**

No code change in this task. If you added a note to the plan/spec, commit it; otherwise nothing to commit.

---

## Self-Review

**Spec coverage:**
- §3 rewrite→QC→scan order → Task 8 (`_apply_reviewer` computes final_scan post-QC; order test asserts it). ✓
- §4 25-guideline rubric in prompt, single constant → Task 6 (`WRITING_CRAFT_GUIDELINES`, len==25 test). ✓
- §4 bucket-A detectors (safety net) → Tasks 2–5 (opener, robotic, repeated-subject, balance, rhythm). ✓
- §5.1 `residual_patterns.py` pure detectors → Tasks 1–5. ✓
- §5.2 `document_reviewer.py` orchestrator, bounded output, no short-circuit, verbatim splice → Tasks 6–7. ✓
- §5.3 wire-in after best-of-N, kill switch, pass_trace → Task 8. ✓
- §6 per-correction fidelity guard (score≤baseline, grammar, polarity, non-stub) → Task 7 (`_correction_is_safe` + drop test). ✓
- §7 trace-only visibility (no UI) → Task 8 (corrections only in `pass_trace`; no frontend touched). ✓
- §8 kill switch default-on → Task 6 (`reviewer_enabled` test) + Task 8 (disabled no-op test). ✓
- §9 failure modes (bad JSON, unmatched, regress, empty) → Task 7 tests. ✓
- §10 testing (detectors, orchestrator, order) → Tasks 1–9. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every command shows expected output. ✓

**Type consistency:** `ResidualIssue(rule, trick_ids, evidence, target_sentences)` consistent across detectors and prompt builder. `ReviewResult(text, corrections, skipped, must_fix_unaddressed)` and `Correction(original, revised, rule)` consistent across Task 7 and Task 8. `_score` monkeypatched identically in all guard tests. `review_document(text, *, gateway, cancellation_check)` signature matches caller in `_apply_reviewer`. `DocumentResult(...)` field names verified against `pipeline.py:71`. ✓

**Known caveat documented:** `split_sentences` in `layer3_scoring` is annotated `-> str` but returns a list; the plan uses local `_sentences()` helpers in `residual_patterns.py` instead, so it does not depend on that quirk. ✓
