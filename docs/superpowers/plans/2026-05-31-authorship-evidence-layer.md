# Authorship Evidence Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface an honest, descriptive authorship-evidence inventory on the scan and rewrite pages, and reuse the same data to steer the writer (protect authentic voice, target grounding at measured gaps).

**Architecture:** One pure builder (`build_authorship_evidence`) reads the *already-measured* `authorship_concern.signals` + `false_positives` and emits a block classified by risk (low = human-present marker, high = thin signal → action). Three consumers reuse the block: the scan report JSON, the rewrite report JSON (plus verbatim preserved-idea spans), and the direct writer's per-paragraph prompt (protect + target, flag-gated). No new detection; honest by construction.

**Tech Stack:** Python (`poc/`), pytest, React + Vite + i18next (`draftproof-frontend/`).

**Spec:** `docs/superpowers/specs/2026-05-31-authorship-evidence-layer-design.md`

**Environment note:** The detector needs the pyenv interpreter with deps. Run Python via `~/.pyenv/versions/3.11.0/bin/python3` and scripts that hit the model must `load_dotenv(REPO/".env")`. Frontend tooling: `export PATH="/opt/homebrew/bin:$PATH"`.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `poc/report/authorship_evidence.py` | Pure builder + helpers (`build_authorship_evidence`, `preserved_idea_spans`, `paragraph_authorship_targets`, `authorship_boost_enabled`) | Create |
| `poc/test_authorship_evidence.py` | Unit tests for the builder + helpers | Create |
| `poc/report/report.py` | Attach `authorship_evidence` to the scan report dict | Modify (~line 4172) |
| `poc/rewrite_v6/direct_rewrite.py` | Thread protect+target into the per-paragraph prompt; flag gate | Modify (`_prompt`, `_rewrite_paragraph`, `_clean_candidate`, `run_direct_rewrite_all`) |
| `poc/rewrite_v6/production.py` | Build evidence from detect_json; pass to writer; attach evidence + preserved_ideas to rewrite report | Modify (`run_rewrite_pipeline_v6` ~line 54-90) |
| `draftproof-frontend/src/i18n/resources.js` | `authorshipEvidence*` keys (en + zh) | Modify |
| `draftproof-frontend/src/pages/Report.jsx` | "Authorship evidence" panel on the scan report | Modify |
| `draftproof-frontend/src/pages/Rewrite.jsx` | "What's yours, preserved" section on the rewrite | Modify |

---

## Task 1: Authorship-evidence builder (pure function)

**Files:**
- Create: `poc/report/authorship_evidence.py`
- Test: `poc/test_authorship_evidence.py`

- [ ] **Step 1: Write the failing tests**

```python
# poc/test_authorship_evidence.py
"""The authorship-evidence builder re-presents ALREADY-MEASURED signals. It must:
(C1) treat signals as risk-oriented -- LOW risk = human-present marker, HIGH = thin -> action;
(C2) exclude the perplexity trio (predictability/surprisal_risk/topk_pattern_risk);
and never surface a verdict/score. Content-agnostic (no domain vocab dependence)."""

from poc.report.authorship_evidence import (
    build_authorship_evidence,
    preserved_idea_spans,
    paragraph_authorship_targets,
)


def _signals(**kw):
    base = {"genericity": 0.0, "specificity": 0.69, "source_grounding": 1.0,
            "draft_evolution": 0.15, "structural_reuse": 0.11, "burstiness": 0.35,
            "predictability": 0.44, "surprisal_risk": 0.11, "topk_pattern_risk": 0.79}
    base.update(kw)
    return base


def test_low_risk_signal_becomes_present_marker():
    block = build_authorship_evidence(_signals())
    present = {m["signal"] for m in block["present_markers"]}
    assert "genericity" in present          # risk 0.0 -> present
    assert "structural_reuse" in present     # risk 0.11 -> present


def test_high_risk_signal_becomes_thin_action():
    block = build_authorship_evidence(_signals())
    thin = {t["signal"] for t in block["thin_signals"]}
    assert "specificity" in thin             # risk 0.69 -> thin
    assert "source_grounding" in thin        # risk 1.0 -> thin
    assert all("action" in t for t in block["thin_signals"])


def test_perplexity_trio_excluded():
    block = build_authorship_evidence(_signals())
    seen = {m["signal"] for m in block["present_markers"] + block["mixed_markers"] + block["thin_signals"]}
    assert not ({"predictability", "surprisal_risk", "topk_pattern_risk"} & seen)


def test_no_verdict_or_score_surfaced():
    block = build_authorship_evidence(_signals())
    assert "score" not in block and "concern_tier" not in block
    assert isinstance(block["summary_line"], str) and block["summary_line"]


def test_empty_false_positives_degrades_gracefully():
    block = build_authorship_evidence(_signals(), false_positives=None)
    assert block["human_recognized_spans"] == []


def test_false_positives_become_human_recognized_spans():
    fps = [{"sentence_id": "s012", "sentence": "In other words, education today...", "reason": "downgraded high->low"}]
    block = build_authorship_evidence(_signals(), false_positives=fps)
    assert block["human_recognized_spans"][0]["text"].startswith("In other words")


def test_preserved_idea_spans_are_verbatim_survivors():
    original = "I taught in one room for years. Technology changes everything rapidly today. Students trusted us."
    final = "I taught in one room for years. Tech now reshapes the classroom in concrete ways. Students trusted us."
    kept = [p["text"] for p in preserved_idea_spans(original, final)]
    assert "I taught in one room for years." in kept
    assert "Students trusted us." in kept
    assert all("Technology changes everything" not in t for t in kept)  # rewritten -> not preserved


def test_paragraph_targets_protect_matching_spans_and_carry_actions():
    fps = [{"sentence_id": "s1", "sentence": "I taught in one room for years.", "reason": "human"}]
    block = build_authorship_evidence(_signals(), false_positives=fps)
    targets = paragraph_authorship_targets(block, "I taught in one room for years. More generic claims here.")
    assert "I taught in one room for years." in targets["protected_spans"]
    assert any("source" in g.lower() or "concrete" in g.lower() for g in targets["grounding_targets"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_authorship_evidence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.report.authorship_evidence'`

- [ ] **Step 3: Write the builder implementation**

```python
# poc/report/authorship_evidence.py
"""Authorship-evidence builder. Re-presents ALREADY-MEASURED authorship_concern signals as an
honest, descriptive inventory -- never a verdict or score.

Direction (C1): signals are risk-oriented. LOW risk = human-authorship marker PRESENT;
HIGH risk = thin signal surfaced as an ACTION. Confirmed against detect.scoring.derive_concern_tier.
Exclusions (C2): predictability / surprisal_risk / topk_pattern_risk are intrinsic LLM token-floor
stats -- not human-authorship evidence and not honestly mitigable -- so they never appear here.
"""
from __future__ import annotations

from typing import Any

try:
    from detect.layer3_scoring import split_sentences
except ImportError:  # package-qualified import path
    from poc.detect.layer3_scoring import split_sentences

SCHEMA_VERSION = "authorship_evidence.v1"
PRESENT_MAX = 0.35   # risk <= this -> genuine human marker
THIN_MIN = 0.55      # risk >= this -> thin signal -> action

# risk-oriented signals only (low = human-present). label, thin-action. (C2 trio omitted.)
SIGNAL_TABLE: dict[str, tuple[str, str]] = {
    "genericity":        ("Original, non-generic phrasing",        "Replace boilerplate phrasing with your own wording"),
    "specificity":       ("Concrete, specific detail",             "Add concrete examples, names, numbers, or scenarios you know"),
    "source_grounding":  ("Claims tied to sources",                "Tie your key claims to a source or citation"),
    "citation_integrity":("Citations support the claims",          "Add or verify citations for your claims"),
    "draft_evolution":   ("Signs of genuine revision",             "Keep or show your drafting history where possible"),
    "structural_reuse":  ("Original structure",                    "Reshape any reused or templated structure into your own"),
    "burstiness":        ("Natural sentence-length variance",      "Let sentence lengths vary as your natural voice would"),
}


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _summary_line(n_present: int, n_thin: int, confidence: str) -> str:
    soft = " (short submission -- read as indicative)" if confidence == "low" else ""
    if n_present and n_thin:
        return f"Your draft shows {n_present} human-authorship marker(s); {n_thin} area(s) would strengthen it.{soft}"
    if n_present:
        return f"Your draft shows {n_present} human-authorship marker(s).{soft}"
    if n_thin:
        return f"{n_thin} area(s) would strengthen your draft's authorship evidence.{soft}"
    return f"Authorship signals are inconclusive for this submission.{soft}"


def build_authorship_evidence(
    signals: dict[str, Any] | None,
    false_positives: list[dict[str, Any]] | None = None,
    confidence: str = "low",
    preserved_ideas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    signals = signals or {}
    present, mixed, thin = [], [], []
    for sig, (label, action) in SIGNAL_TABLE.items():
        raw = signals.get(sig)
        if raw is None:
            continue
        risk = float(raw)
        if risk <= PRESENT_MAX:
            present.append({"signal": sig, "label": label, "risk": round(risk, 3)})
        elif risk >= THIN_MIN:
            thin.append({"signal": sig, "action": action, "risk": round(risk, 3)})
        else:
            mixed.append({"signal": sig, "label": label, "risk": round(risk, 3)})

    human_spans = []
    for fp in (false_positives or []):
        text = (fp.get("sentence") or "").strip()
        if text:
            human_spans.append({
                "sentence_id": fp.get("sentence_id"),
                "text": text,
                "reason": fp.get("reason"),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "present_markers": present,
        "mixed_markers": mixed,
        "thin_signals": thin,
        "human_recognized_spans": human_spans,
        "preserved_ideas": list(preserved_ideas or []),
        "confidence": confidence or "low",
        "summary_line": _summary_line(len(present), len(thin), confidence or "low"),
    }


def preserved_idea_spans(original_text: str, final_text: str, *, min_words: int = 6) -> list[dict[str, str]]:
    """Sentences from the original kept VERBATIM in the final text (the user's surviving words)."""
    final_norm = {_norm(s) for s in split_sentences(final_text or "")}
    out: list[dict[str, str]] = []
    for sent in split_sentences(original_text or ""):
        clean = (sent or "").strip()
        if len(clean.split()) >= min_words and _norm(clean) in final_norm:
            out.append({"text": clean})
    return out


def paragraph_authorship_targets(evidence: dict[str, Any] | None, paragraph_text: str) -> dict[str, Any]:
    """Per-paragraph protect+target brief for the writer, derived from the evidence block."""
    if not evidence:
        return {}
    para_set = {_norm(s) for s in split_sentences(paragraph_text or "")}
    protected = [
        span["text"] for span in evidence.get("human_recognized_spans", [])
        if (span.get("text") or "").strip() and _norm(span["text"]) in para_set
    ]
    targets = [t["action"] for t in evidence.get("thin_signals", []) if t.get("action")]
    out: dict[str, Any] = {}
    if protected:
        out["protected_spans"] = protected[:6]
    if targets:
        out["grounding_targets"] = targets[:4]
    return out


def authorship_boost_enabled() -> bool:
    import os
    return os.environ.get("DRAFTPROOF_AUTHORSHIP_BOOST", "1").strip().lower() not in {"0", "false", "no", "off"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_authorship_evidence.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/report/authorship_evidence.py poc/test_authorship_evidence.py
git commit -m "feat(report): authorship-evidence builder (honest inventory over measured signals)"
```

---

## Task 2: Attach evidence to the scan report JSON

**Files:**
- Modify: `poc/report/report.py` (import near line 20; insert into the report dict near the `authorship_concern` block at ~line 4172)
- Test: `poc/test_authorship_evidence_report.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# poc/test_authorship_evidence_report.py
"""The scan report dict must carry an authorship_evidence block built from the same
authorship_concern signals + false_positives it already computes."""
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent / "test_output/_content11/scan/draftproof_20260530_185257.json"


def test_saved_report_signals_produce_evidence_block():
    # Builder over the saved report's measured signals -> usable, honest block.
    from poc.report.authorship_evidence import build_authorship_evidence
    d = json.loads(REPORT.read_text())
    block = build_authorship_evidence(d["authorship_concern"]["signals"], d.get("false_positives"))
    present = {m["signal"] for m in block["present_markers"]}
    thin = {t["signal"] for t in block["thin_signals"]}
    assert "genericity" in present
    assert {"specificity", "source_grounding"} <= thin
```

- [ ] **Step 2: Run test to verify it passes the builder, then add the wiring**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_authorship_evidence_report.py -v`
Expected: PASS (validates the builder against the real saved report). Now wire it into the live report dict.

- [ ] **Step 3: Add the import in `poc/report/report.py`**

Find (near line 20):
```python
from detect.scoring import extract_signals, calculate_authorship_concern, estimate_citation_risk
```
Add immediately after it:
```python
from report.authorship_evidence import build_authorship_evidence
```

- [ ] **Step 4: Insert the block into the report dict**

In the report-dict assembly, find the `authorship_concern` entry (~line 4172):
```python
        "authorship_concern": {
            "score": report.authorship_concern_score,
            "concern_tier": _concern_tier_from_score(report.authorship_concern_score),
            "confidence": report.authorship_concern_confidence,
            "weak_signal_only": _is_weak_only(report.authorship_concern_signals),
            "signals": report.authorship_concern_signals,
```
Immediately **before** the `"authorship_concern": {` line, add:
```python
        "authorship_evidence": build_authorship_evidence(
            report.authorship_concern_signals,
            false_positives=serialized_false_positives,
            confidence=report.authorship_concern_confidence,
        ),
```
Note: `serialized_false_positives` is the list already assembled for the `"false_positives"` key in this same dict. If the local variable has a different name, use the exact expression that feeds `"false_positives":` here (search upward in this function for `false_positives`).

- [ ] **Step 5: Verify the live report carries the block**

Run:
```bash
~/.pyenv/versions/3.11.0/bin/python3 -c "
import sys; sys.path.insert(0, 'poc')
from report.authorship_evidence import build_authorship_evidence, SCHEMA_VERSION
print('import OK', SCHEMA_VERSION)
import ast
src = open('poc/report/report.py').read()
assert 'authorship_evidence' in src and 'build_authorship_evidence' in src
print('wired OK')
"
~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_authorship_evidence_report.py -v
```
Expected: `import OK authorship_evidence.v1` / `wired OK` / PASS

- [ ] **Step 6: Commit**

```bash
git add poc/report/report.py poc/test_authorship_evidence_report.py
git commit -m "feat(report): attach authorship_evidence block to the scan report"
```

---

## Task 3: Writer boost — protect + target (flag-gated)

**Files:**
- Modify: `poc/rewrite_v6/direct_rewrite.py` (`_prompt` ~line 100-132; `_rewrite_paragraph` ~line 489; `_clean_candidate` ~line 457; `run_direct_rewrite_all` ~line 420-448)
- Test: `poc/test_rewrite_v6_authorship_boost.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# poc/test_rewrite_v6_authorship_boost.py
"""The direct writer's prompt must carry protect+target instructions when an authorship_targets
brief is supplied, and omit them when it is empty (no behaviour change off-flag)."""
import json
from poc.rewrite_v6.direct_rewrite import _prompt


def test_prompt_includes_protected_and_grounding_when_supplied():
    targets = {"protected_spans": ["I taught in one room for years."],
               "grounding_targets": ["Tie your key claims to a source or citation"]}
    out = _prompt("Some flagged paragraph text.", None, ["generic_assertion"], targets)
    payload = json.loads(out.split("\n", 1)[1])
    assert "protected_spans" in payload and payload["protected_spans"] == targets["protected_spans"]
    assert "grounding_targets" in payload and payload["grounding_targets"] == targets["grounding_targets"]
    assert any("verbatim" in r.lower() for r in payload["rules"])


def test_prompt_omits_fields_when_targets_empty():
    out = _prompt("Some flagged paragraph text.", None, ["generic_assertion"], {})
    payload = json.loads(out.split("\n", 1)[1])
    assert "protected_spans" not in payload and "grounding_targets" not in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_rewrite_v6_authorship_boost.py -v`
Expected: FAIL — `_prompt() takes 3 positional arguments but 4 were given`

- [ ] **Step 3: Extend `_prompt` to accept and embed the brief**

In `poc/rewrite_v6/direct_rewrite.py`, change the `_prompt` signature and payload. Find the function (it ends with `return "Return JSON only.\n" + json.dumps(payload, ...)`). Update the signature to add `authorship_targets`:
```python
def _prompt(text, diagnosis, tags, authorship_targets=None):
```
Inside, after `payload = {...}` is built and BEFORE the `return`, add:
```python
    authorship_targets = authorship_targets or {}
    protected = authorship_targets.get("protected_spans") or []
    grounding = authorship_targets.get("grounding_targets") or []
    if protected:
        payload["protected_spans"] = list(protected)
        payload["rules"].append(
            "Keep every sentence in protected_spans VERBATIM -- they are the author's own voice; "
            "rewriting them only makes them more generic. Rewrite the surrounding text only."
        )
    if grounding:
        payload["grounding_targets"] = list(grounding)
        payload["rules"].append(
            "Where you add concrete grounding, prioritise these author-owned gaps: "
            + "; ".join(grounding) + "."
        )
```
(If `payload["rules"]` is keyed differently in this file, append to the actual list of instruction strings in the payload.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_rewrite_v6_authorship_boost.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Thread the brief through `_rewrite_paragraph` and `_clean_candidate`**

In `_rewrite_paragraph` (signature `def _rewrite_paragraph(gateway, paragraph, diagnosis, findings):`), add `authorship_targets=None` and pass it to `_prompt`:
```python
def _rewrite_paragraph(gateway, paragraph, diagnosis, findings, authorship_targets=None):
    tags = sorted({tag for finding in findings for tag in (finding.tags or [])})
    try:
        response = gateway.chat(
            _prompt(paragraph.text, diagnosis, tags, authorship_targets),
            system=_SYSTEM,
            ...
```
In `_clean_candidate` (signature `def _clean_candidate(gateway, paragraph, diagnosis, findings, *, attempts=2):`), add `authorship_targets=None` and forward it:
```python
def _clean_candidate(gateway, paragraph, diagnosis, findings, *, attempts=2, authorship_targets=None):
    for _ in range(max(1, attempts)):
        candidate, review_items = _rewrite_paragraph(gateway, paragraph, diagnosis, findings, authorship_targets)
        ...
```

- [ ] **Step 6: Wire the brief into the loop in `run_direct_rewrite_all`**

At the top of `run_direct_rewrite_all`, add an `authorship_evidence=None` keyword parameter to its signature. Add this import near the other rewrite_v6 imports at the top of the file:
```python
try:
    from report.authorship_evidence import paragraph_authorship_targets, authorship_boost_enabled
except ImportError:
    from poc.report.authorship_evidence import paragraph_authorship_targets, authorship_boost_enabled
```
In the per-paragraph loop (~line 425), find:
```python
        candidate, review_items = _clean_candidate(gateway, paragraph, diagnosis, findings)
```
Replace with:
```python
        targets = (
            paragraph_authorship_targets(authorship_evidence, paragraph.text)
            if (authorship_evidence and authorship_boost_enabled())
            else {}
        )
        candidate, review_items = _clean_candidate(
            gateway, paragraph, diagnosis, findings, authorship_targets=targets
        )
```

- [ ] **Step 7: Run the full direct-rewrite test module to confirm no regression**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_rewrite_v6_authorship_boost.py poc/test_rewrite_v6_prose_repair_guard.py -v`
Expected: PASS (existing prose-repair guard tests still pass; new boost tests pass)

- [ ] **Step 8: Commit**

```bash
git add poc/rewrite_v6/direct_rewrite.py poc/test_rewrite_v6_authorship_boost.py
git commit -m "feat(rewrite): authorship boost -- protect authentic spans + target grounding (flag DRAFTPROOF_AUTHORSHIP_BOOST)"
```

---

## Task 4: Build evidence in production + attach to rewrite report

**Files:**
- Modify: `poc/rewrite_v6/production.py` (`run_rewrite_pipeline_v6`, ~line 54-95)
- Test: `poc/test_authorship_evidence_production.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# poc/test_authorship_evidence_production.py
"""run_rewrite_pipeline_v6 must build the evidence block from detect_json and attach it
(with verbatim preserved_ideas) to the rewrite report it returns."""
from poc.report.authorship_evidence import build_authorship_evidence, preserved_idea_spans


def test_evidence_built_from_detect_json_signals():
    detect_json = {"authorship_concern": {"signals": {"genericity": 0.0, "specificity": 0.69,
                   "source_grounding": 1.0}, "confidence": "medium"}, "false_positives": []}
    sig = detect_json["authorship_concern"]["signals"]
    block = build_authorship_evidence(sig, detect_json.get("false_positives"),
                                      detect_json["authorship_concern"].get("confidence", "low"))
    assert {m["signal"] for m in block["present_markers"]} == {"genericity"}
    assert {t["signal"] for t in block["thin_signals"]} == {"specificity", "source_grounding"}


def test_preserved_ideas_capture_unchanged_sentences():
    original = "I worked in one classroom for a decade. Everything is changing fast now."
    final = "I worked in one classroom for a decade. The pace of change is concrete and measurable."
    kept = [p["text"] for p in preserved_idea_spans(original, final)]
    assert "I worked in one classroom for a decade." in kept
```

- [ ] **Step 2: Run test to verify it passes the helpers**

Run: `~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_authorship_evidence_production.py -v`
Expected: PASS — confirms the helper contract before wiring production.

- [ ] **Step 3: Add the import in `poc/rewrite_v6/production.py`**

Near the existing `from .direct_rewrite import ...` (line 23), add:
```python
try:
    from report.authorship_evidence import build_authorship_evidence, preserved_idea_spans
except ImportError:
    from poc.report.authorship_evidence import build_authorship_evidence, preserved_idea_spans
```

- [ ] **Step 4: Build evidence and pass it to the writer**

In `run_rewrite_pipeline_v6`, after `original_text, rewrite_source = _rewrite_source_text(detect_json)` (~line 54), add:
```python
    _concern = (detect_json.get("authorship_concern") or {})
    authorship_evidence = build_authorship_evidence(
        _concern.get("signals"),
        false_positives=detect_json.get("false_positives"),
        confidence=_concern.get("confidence", "low"),
    )
```
In the `run_direct_rewrite_all(...)` call (~line 70), add the keyword argument:
```python
            document = run_direct_rewrite_all(
                ...,
                source_scan=source_scan,
                authorship_evidence=authorship_evidence,
                ...
            )
```

- [ ] **Step 5: Attach evidence + preserved_ideas to the rewrite report**

After the rewrite report dict is produced by `render_rewrite_report(...)` in this function (search for `render_rewrite_report`), and after `document.rewritten_text` is available, add:
```python
    authorship_evidence["preserved_ideas"] = preserved_idea_spans(original_text, document.rewritten_text)
    if isinstance(rewrite_report, dict):
        rewrite_report["authorship_evidence"] = authorship_evidence
        summary = rewrite_report.get("summary")
        if isinstance(summary, dict):
            summary["authorship_evidence"] = authorship_evidence
```
Use the actual variable name the rewrite report dict is bound to in this function (e.g. `rewrite_report` / `report_json`). The frontend reads `report.authorship_evidence` or `summary.authorship_evidence`, so set both.

- [ ] **Step 6: Verify end-to-end (deterministic) on content11**

Run:
```bash
DRAFTPROOF_V6_DETERMINISTIC=1 ~/.pyenv/versions/3.11.0/bin/python3 -m pytest poc/test_authorship_evidence_production.py -v
~/.pyenv/versions/3.11.0/bin/python3 -c "
import sys; sys.path.insert(0,'poc')
src=open('poc/rewrite_v6/production.py').read()
assert 'authorship_evidence' in src and 'preserved_idea_spans' in src
print('production wired OK')
"
```
Expected: PASS / `production wired OK`

- [ ] **Step 7: Commit**

```bash
git add poc/rewrite_v6/production.py poc/test_authorship_evidence_production.py
git commit -m "feat(rewrite): build authorship evidence + preserved_ideas, feed writer & rewrite report"
```

---

## Task 5: No-regression measurement (deterministic harness)

**Files:** none (verification only — produces a recorded result)

- [ ] **Step 1: Baseline with boost OFF**

Run:
```bash
cd /Users/kianwoonwong/Downloads/draftproof_services
DRAFTPROOF_V6_DETERMINISTIC=1 DRAFTPROOF_AUTHORSHIP_BOOST=0 ~/.pyenv/versions/3.11.0/bin/python3 poc/_measure_baseline.py 4
```
Expected: a mean `final_risk` + per-run spread. Record it (e.g. `mean 34.x`).

- [ ] **Step 2: With boost ON**

Run:
```bash
DRAFTPROOF_V6_DETERMINISTIC=1 DRAFTPROOF_AUTHORSHIP_BOOST=1 ~/.pyenv/versions/3.11.0/bin/python3 poc/_measure_baseline.py 4
```
Expected: mean within the OFF spread (no regression). Honest expectation: **headline flag unchanged** (topk floor); `generic_assertion` holds or marginally improves.

- [ ] **Step 3: Record the verdict**

Write the two means + spreads into the PR/commit body. Decision rule (from `project_v6_measurement_variance`): a real change only if the ON mean beats OFF by **more than the spread**. If ON regresses beyond spread, keep the flag default OFF and note it. Otherwise default stays ON.

```bash
git commit --allow-empty -m "test(rewrite): authorship-boost N=4 deterministic harness -- OFF mean X (spread), ON mean Y (spread); no regression"
```

---

## Task 6: Frontend i18n keys (en + zh)

**Files:**
- Modify: `draftproof-frontend/src/i18n/resources.js`

- [ ] **Step 1: Add the `authorshipEvidence` keys under the existing `en` resource block**

Find the `en` translation object (it already contains `rewritePage`). Add a sibling key:
```javascript
      authorshipEvidence: {
        scanTitle: 'Authorship evidence',
        scanKicker: 'The other side of the ledger',
        scanCopy: 'Markers of your own authorship already present in this draft. This is not a clearance — it is the evidence you can point to, plus the gaps worth closing.',
        rewriteTitle: "What's yours, preserved",
        rewriteCopy: 'Your own words and ideas the rewrite kept and protected, plus the authorship markers in your draft.',
        presentTitle: 'Present in your draft',
        thinTitle: 'Strengthen these',
        thinCopy: 'Closing these both defends your authorship and lowers the movable part of the flag. It will not change the underlying AI-likelihood floor.',
        recognizedTitle: 'Sentences recognized as yours',
        preservedTitle: 'Your words, kept verbatim',
        lowConfidenceNote: 'Short submission — read these as indicative.',
        empty: 'Authorship signals are inconclusive for this submission.',
      },
```

- [ ] **Step 2: Add the same key translated under the `zh` resource block**

Find the `zh` translation object and add:
```javascript
      authorshipEvidence: {
        scanTitle: '作者身份证据',
        scanKicker: '天平的另一端',
        scanCopy: '本稿中已经存在的、属于你自己创作的标记。这并非「通过」认证，而是你可以据以说明的证据，以及值得补强的不足之处。',
        rewriteTitle: '属于你的部分，已保留',
        rewriteCopy: '改写中保留并保护的你自己的文字与想法，以及你草稿中的作者身份标记。',
        presentTitle: '你草稿中已具备',
        thinTitle: '建议补强',
        thinCopy: '补强这些既能佐证你的作者身份，也能降低标记中可改动的部分；但不会改变底层的 AI 可能性下限。',
        recognizedTitle: '被识别为你所写的句子',
        preservedTitle: '逐字保留的你的文字',
        lowConfidenceNote: '提交内容较短 — 仅供参考。',
        empty: '本次提交的作者身份信号尚不明确。',
      },
```

- [ ] **Step 3: Verify the bundle parses**

Run:
```bash
export PATH="/opt/homebrew/bin:$PATH"
cd draftproof-frontend && node -e "import('./src/i18n/resources.js').then(()=>console.log('resources parse OK')).catch(e=>{console.error(e);process.exit(1)})"
```
Expected: `resources parse OK`

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/i18n/resources.js
git commit -m "feat(i18n): authorshipEvidence copy (en + zh)"
```

---

## Task 7: Scan-report panel (`Report.jsx`)

**Files:**
- Modify: `draftproof-frontend/src/pages/Report.jsx`

- [ ] **Step 1: Read the report data shape and pick the insertion point**

Run:
```bash
grep -n "report\?\.\|authorship\|<section\|className=\"report-" draftproof-frontend/src/pages/Report.jsx | head -40
```
Identify where the main report sections render and where `report` (the fetched JSON) is in scope. The block is at `report.authorship_evidence`.

- [ ] **Step 2: Add a render helper and the panel**

Near the other derived values in the component, add:
```jsx
  const authorshipEvidence = report?.authorship_evidence || null;
```
In the JSX, after the AI-flag/badge section (so it lands beside the flag), insert:
```jsx
        {authorshipEvidence && (
          <section className="rewrite-review-section" aria-label={t('authorshipEvidence.scanTitle')}>
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">{t('authorshipEvidence.scanKicker')}</span>
                <h3>{t('authorshipEvidence.scanTitle')}</h3>
              </div>
            </div>
            <p className="rewrite-review-copy">{t('authorshipEvidence.scanCopy')}</p>
            {authorshipEvidence.confidence === 'low' && (
              <p className="sample-reference-note">{t('authorshipEvidence.lowConfidenceNote')}</p>
            )}
            {authorshipEvidence.present_markers?.length > 0 && (
              <div className="rewrite-target-block">
                <span>{t('authorshipEvidence.presentTitle')}</span>
                <ul className="signal-list">
                  {authorshipEvidence.present_markers.map((m) => <li key={m.signal}>{m.label}</li>)}
                </ul>
              </div>
            )}
            {authorshipEvidence.thin_signals?.length > 0 && (
              <div className="rewrite-addition-block">
                <span>{t('authorshipEvidence.thinTitle')}</span>
                <p className="rewrite-review-copy">{t('authorshipEvidence.thinCopy')}</p>
                <ul className="signal-list">
                  {authorshipEvidence.thin_signals.map((tn) => <li key={tn.signal}>{tn.action}</li>)}
                </ul>
              </div>
            )}
            {authorshipEvidence.human_recognized_spans?.length > 0 && (
              <div className="rewrite-target-block">
                <span>{t('authorshipEvidence.recognizedTitle')}</span>
                {authorshipEvidence.human_recognized_spans.slice(0, 6).map((s, i) => (
                  <p key={s.sentence_id || i}>{s.text}</p>
                ))}
              </div>
            )}
          </section>
        )}
```
(Reuse existing classes `rewrite-review-section`, `signal-list`, etc. — they already exist in the stylesheet. If `signal-list` is not styled here, a plain `<ul>` is fine.)

- [ ] **Step 3: Verify the build**

Run:
```bash
export PATH="/opt/homebrew/bin:$PATH"
cd draftproof-frontend && npm run build
```
Expected: build succeeds, no JSX/i18n errors.

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/pages/Report.jsx
git commit -m "feat(frontend): authorship-evidence panel on the scan report"
```

---

## Task 8: Rewrite-report section (`Rewrite.jsx`)

**Files:**
- Modify: `draftproof-frontend/src/pages/Rewrite.jsx` (insert after the diff section, ~line 357)

- [ ] **Step 1: Derive the block**

Near the other derived values (~line 240, beside `authorReviewCards`), add:
```jsx
  const authorshipEvidence = report?.authorship_evidence || summary.authorship_evidence || null;
```

- [ ] **Step 2: Add the "What's yours, preserved" section**

After the `documentDiff` section closes (`)}` at ~line 357) and before the `report?.final_text && (` rewritten-document section, insert:
```jsx
        {authorshipEvidence && (
          <section className="rewrite-review-section" aria-label={t('authorshipEvidence.rewriteTitle')}>
            <div className="rewrite-review-heading">
              <div>
                <span className="rewrite-review-kicker">{t('authorshipEvidence.scanKicker')}</span>
                <h3>{t('authorshipEvidence.rewriteTitle')}</h3>
              </div>
            </div>
            <p className="rewrite-review-copy">{t('authorshipEvidence.rewriteCopy')}</p>
            {authorshipEvidence.preserved_ideas?.length > 0 && (
              <div className="rewrite-target-block">
                <span>{t('authorshipEvidence.preservedTitle')}</span>
                {authorshipEvidence.preserved_ideas.slice(0, 8).map((p, i) => <p key={i}>{p.text}</p>)}
              </div>
            )}
            {authorshipEvidence.present_markers?.length > 0 && (
              <div className="rewrite-target-block">
                <span>{t('authorshipEvidence.presentTitle')}</span>
                <ul className="signal-list">
                  {authorshipEvidence.present_markers.map((m) => <li key={m.signal}>{m.label}</li>)}
                </ul>
              </div>
            )}
            {authorshipEvidence.thin_signals?.length > 0 && (
              <div className="rewrite-addition-block">
                <span>{t('authorshipEvidence.thinTitle')}</span>
                <p className="rewrite-review-copy">{t('authorshipEvidence.thinCopy')}</p>
                <ul className="signal-list">
                  {authorshipEvidence.thin_signals.map((tn) => <li key={tn.signal}>{tn.action}</li>)}
                </ul>
              </div>
            )}
          </section>
        )}
```

- [ ] **Step 3: Verify the build**

Run:
```bash
export PATH="/opt/homebrew/bin:$PATH"
cd draftproof-frontend && npm run build
```
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add draftproof-frontend/src/pages/Rewrite.jsx
git commit -m "feat(frontend): 'what's yours, preserved' authorship section on the rewrite"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** Component 1 → Task 1; Component 2 → Task 2; Component 3 (preserved-ideas) → Tasks 1 (`preserved_idea_spans`) + 4 (wiring); Component 4 (writer boost) → Tasks 1 (`paragraph_authorship_targets`/flag) + 3 (prompt) + 4 (production wiring); Component 5 → Tasks 6/7/8. Verification §1 → Task 1 tests; §2 corpus A/B → covered by Task 2 saved-report test + Task 5; §3 harness → Task 5; §4 frontend render → Tasks 7/8 builds. Honesty guardrails → encoded in Task 1 (no score/verdict; C1/C2) and i18n copy (Task 6, "not a clearance / will not change the floor").
- **Placeholder scan:** none — every code step contains complete code; commands have expected output.
- **Type consistency:** block keys (`present_markers`, `mixed_markers`, `thin_signals`, `human_recognized_spans`, `preserved_ideas`, `confidence`, `summary_line`) are identical across builder (Task 1), report wiring (Task 2), production (Task 4), and both frontend renders (Tasks 7/8). `paragraph_authorship_targets` returns `protected_spans`/`grounding_targets`, consumed verbatim by `_prompt` (Task 3). `authorship_evidence=` kwarg name consistent across `run_direct_rewrite_all` (Task 3) and its caller (Task 4).
- **One known soft spot flagged for the implementer:** in Tasks 2 and 4, the exact local variable feeding `"false_positives":` (report.py) and the rewrite-report dict name (production.py) must be confirmed by reading the surrounding function — both are called out inline in the steps.
