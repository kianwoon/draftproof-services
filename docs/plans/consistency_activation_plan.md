# Consistency Dimension — ESL Validation & Activation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the already-built Phase-1 **English** stylometric-consistency detector against lower- vs higher-proficiency ESL writing (SCoCESLE), prove score-neutrality with the flag ON, and enable `DRAFTPROOF_CONSISTENCY` in production. (English-only scope is deliberate — this validates behaviour across English proficiency levels; it does NOT claim fairness for every ESL population or linguistic background, and no other-language subgroup gates are in scope for this release.)

**Architecture:** No new detection code. The full build (stylometry features → outlier detection → `ConsistencyDetector` → `consistency_display` panel → web/PDF render) shipped to `main` via merge `d8c9f002`, kill-switched OFF by `DRAFTPROOF_CONSISTENCY` (`poc/detect/consistency.py:64` `consistency_enabled()`, default `"0"`). The governing build plan (`docs/plans/consistency_defence_readiness_build_plan.md`, Global Constraints) explicitly deferred ESL validation and the default-ON decision to "a separate, later product decision" — this plan is that decision path. Production enablement is a Koyeb **runtime env** change on the worker service, NOT a repo default change (prod config = runtime env, overrides entrypoint defaults — see memory `project_modal_deepscan_koyeb_secret`).

**Tech Stack:** Python stdlib + existing `poc/detect/stylometry/` library; SCoCESLE corpus (local-only, 272 essays); Koyeb CLI.

## Global Constraints

- SCoCESLE corpus is LOCAL-ONLY (no redistribution): default path `/Users/kianwoonwong/Downloads/Small Corpus of Colombian English as a Second Language Essays (SCoCESLE)` (`poc/calibration/fpr_subgroup_gate.py:46`).
- STRICTLY NO HARDCODED score-driving values — every threshold is a named module constant with a derivation comment.
- All `poc/` commands run from the `poc/` directory (imports are rooted there).
- The consistency detector is advisory-only by contract: `DetectResult.overall_risk == 0.0` unconditionally and it is not wired into `layer3_scoring.py` (`poc/detect/consistency.py` module docstring). Nothing in this plan may change that.
- Rendered-artifact rule (memory `feedback_verify_rendered_artifacts`): the panel must be visually confirmed in rendered HTML/PDF, not grep-asserted.
- Commit after each task. Do NOT push until Task 5's checklist says so (push to `main` = Koyeb auto-deploy).

## Pre-verified facts (do not re-derive)

| Fact | Anchor |
|---|---|
| Kill switch + helper | `poc/detect/consistency.py:47` `CONSISTENCY_KILL_SWITCH_ENV = "DRAFTPROOF_CONSISTENCY"`, `:64` `consistency_enabled()`, default OFF |
| Feature extraction | `poc/detect/stylometry/features.py` `extract_fingerprints(text: str) -> list[ParagraphFingerprint]` |
| Outlier detection | `poc/detect/stylometry/outliers.py` `detect_outliers(fingerprints) -> list[OutlierResult]`; `MIN_PARAGRAPHS = 6` floor returns `[]` |
| Corpus loader to reuse | `poc/calibration/fpr_subgroup_gate.py:61` `_proficiency_groups(corpus) -> dict` (keys are proficiency-group names → lists of `.txt` paths); `:46` `DEFAULT_CORPUS`; `:77` `_ai_texts()` |
| OFF-state parity proven | `poc/test_consistency_report_parity.py` (byte-identical report + structural never-instantiated spy) |
| End-to-end render CLI | `poc/detect_pipeline.py` — `python detect_pipeline.py <file>` writes `test_output/draftproof_<ts>.{md,json,pdf}` (`:145` calls `render_pdf`) |
| Shifted-paragraph fixture | `poc/detect/test_consistency.py:74` `_document_with_shifted_paragraph() -> str` (no args) |
| ESL score gate measures ONLY `badge["ai_likelihood_score"]` | `poc/calibration/fpr_subgroup_gate.py:115` — advisory keys cannot move it; but it does NOT measure panel flag-rate disparity, hence Task 1 |
| Prod worker service | Koyeb worker svc `2af91bbc`; runtime env overrides `worker/entrypoint.sh` defaults; change via `koyeb service update` |

---

### Task 1: ESL flag-rate disparity calibration script

**Files:**
- Create: `poc/calibration/consistency_esl_rates.py`
- Test: `poc/calibration/test_consistency_esl_rates.py`

**Interfaces:**
- Consumes: `extract_fingerprints`, `detect_outliers`, `MIN_PARAGRAPHS`, `_proficiency_groups`, `DEFAULT_CORPUS`, `_ai_texts` (all pre-verified above).
- Produces: CLI script printing per-group flag rates + a PASS/FAIL disparity verdict (exit code 0/1), and `summarize_group(outlier_counts, eligibles) -> dict` (pure, unit-tested). Task 2 runs this script; no other task imports it.

- [ ] **Step 1: Write the failing test**

```python
# poc/calibration/test_consistency_esl_rates.py
"""Unit tests for the pure aggregation math in consistency_esl_rates.py.

The corpus run itself is a manual local tool (SCoCESLE is local-only); these
tests cover only the group-summary and disparity-gap arithmetic with synthetic
counts, so they run in CI without the corpus.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.consistency_esl_rates import flag_rate_gap_pts, summarize_group


def test_summarize_group_rates():
    # 4 essays: 2 eligible-and-flagged, 1 eligible-unflagged, 1 ineligible (short doc).
    s = summarize_group(outlier_counts=[2, 1, 0, 0], eligibles=[True, True, True, False])
    assert s["n_essays"] == 4
    assert s["n_eligible"] == 3
    assert s["eligibility_rate_pct"] == 75.0
    assert s["flagged_essay_rate_pct"] == round(2 / 3 * 100, 2)
    assert s["mean_outliers_per_eligible_essay"] == round(3 / 3, 3)


def test_summarize_group_no_eligible():
    s = summarize_group(outlier_counts=[0, 0], eligibles=[False, False])
    assert s["n_eligible"] == 0
    assert s["flagged_essay_rate_pct"] is None


def test_flag_rate_gap_lower_minus_higher():
    groups = {
        "lower proficiency": {"flagged_essay_rate_pct": 22.0},
        "higher proficiency": {"flagged_essay_rate_pct": 15.5},
    }
    assert flag_rate_gap_pts(groups) == 6.5


def test_flag_rate_gap_none_when_a_group_missing_rate():
    groups = {
        "lower proficiency": {"flagged_essay_rate_pct": None},
        "higher proficiency": {"flagged_essay_rate_pct": 10.0},
    }
    assert flag_rate_gap_pts(groups) is None


def test_groups_measurable_floor():
    from calibration.consistency_esl_rates import MIN_ELIGIBLE_PER_GROUP, groups_measurable

    ok = {
        "lower proficiency": {"n_eligible": MIN_ELIGIBLE_PER_GROUP},
        "higher proficiency": {"n_eligible": MIN_ELIGIBLE_PER_GROUP + 5},
    }
    tiny = {
        "lower proficiency": {"n_eligible": 3},
        "higher proficiency": {"n_eligible": MIN_ELIGIBLE_PER_GROUP + 5},
    }
    assert groups_measurable(ok) is True
    assert groups_measurable(tiny) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd poc && python -m pytest calibration/test_consistency_esl_rates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'calibration.consistency_esl_rates'`

- [ ] **Step 3: Write the implementation**

```python
# poc/calibration/consistency_esl_rates.py
"""ESL flag-rate disparity check for the Phase-1 stylometric ConsistencyDetector.

WHY THIS EXISTS: the pre-push ESL gate (fpr_subgroup_gate.py) measures ONLY
badge["ai_likelihood_score"] (line 115). The consistency panel is score-neutral by
contract (overall_risk == 0.0), so that gate can NEVER catch the failure mode that
actually matters here: the "writing-style outliers" PANEL disproportionately
flagging lower-proficiency ESL writers, whose within-document style variance is
often genuine (memory: stylometric anomaly detection is exactly where ESL false
positives live). This script measures panel flag rates by proficiency band on the
local SCoCESLE corpus and FAILS (exit 1) on excessive disparity — it is the
decision gate the build plan deferred ("full gate run is a separate, later product
decision") and must PASS before DRAFTPROOF_CONSISTENCY is enabled in production.

Usage (from poc/):
    python calibration/consistency_esl_rates.py                # full corpus
    python calibration/consistency_esl_rates.py --limit 12     # quick smoke
    python calibration/consistency_esl_rates.py --out calibration/consistency_esl_rates.json

Purely deterministic and cheap: stylometry only, no ML models, no DetectionRunner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detect.stylometry.features import extract_fingerprints
from detect.stylometry.outliers import MIN_PARAGRAPHS, detect_outliers

# Maximum permitted LOWER-PROFICIENCY OVER-FLAGGING gap: (lower − higher)
# flagged-essay-rate, in percentage points, among essays eligible for outlier
# statistics (≥MIN_PARAGRAPHS). DELIBERATELY ONE-SIDED — this is not a symmetric
# statistical-parity test. The specific product risk being controlled is
# lower-proficiency English writers being disproportionately flagged with
# style-inconsistency warnings; higher-proficiency essays flagging MORE than
# lower is not the failure mode this gate protects against, so a negative gap
# passes by design.
# Derivation: mirrors the spirit of fpr_subgroup_gate.MAX_PARITY_WIDEN_PTS (4.0 pts
# on FPR parity) with 1 pt of slack because this is an advisory review-flag rate,
# not a score/FPR — an advisory panel row costs the user a review glance, not a
# false accusation. Owner sign-off on this threshold + the measured numbers is a
# required Task 2 step before the prod flip.
MAX_FLAG_RATE_GAP_PTS = 5.0

# Below this many eligible essays in EITHER proficiency group, the flag-rate gap is
# statistical noise (SCoCESLE essays are short; many fall under MIN_PARAGRAPHS=6 and
# a 3-essay group must not PASS the flip). Verdict becomes UNMEASURABLE, exit 1.
MIN_ELIGIBLE_PER_GROUP = 20


def summarize_group(outlier_counts: list[int], eligibles: list[bool]) -> dict:
    """Pure aggregation: per-group flag rates from per-essay outlier counts.

    outlier_counts[i] = number of flagged paragraphs in essay i;
    eligibles[i] = essay i had >= MIN_PARAGRAPHS paragraphs (short docs are
    structurally unflaggable — excluding them keeps the rate honest).
    """
    n = len(outlier_counts)
    elig_counts = [c for c, e in zip(outlier_counts, eligibles) if e]
    n_elig = len(elig_counts)
    # Eligibility rate is calibration observability: a large eligibility skew between
    # groups (e.g. lower-proficiency essays mostly too short to measure) is itself
    # worth the owner's eye even when the flag-rate gap passes.
    eligibility_rate = round(n_elig / n * 100, 2) if n else None
    if n_elig == 0:
        return {
            "n_essays": n,
            "n_eligible": 0,
            "eligibility_rate_pct": eligibility_rate,
            "flagged_essay_rate_pct": None,
            "mean_outliers_per_eligible_essay": None,
        }
    flagged = sum(1 for c in elig_counts if c > 0)
    return {
        "n_essays": n,
        "n_eligible": n_elig,
        "eligibility_rate_pct": eligibility_rate,
        "flagged_essay_rate_pct": round(flagged / n_elig * 100, 2),
        "mean_outliers_per_eligible_essay": round(sum(elig_counts) / n_elig, 3),
    }


def flag_rate_gap_pts(groups: dict) -> float | None:
    """(lower − higher) flagged-essay-rate gap in pts; None if either is unmeasurable.

    Group keys come from fpr_subgroup_gate._proficiency_groups (directory names
    containing 'proficiency'); match by 'lower'/'higher' substring like the gate does.
    """
    lower = higher = None
    for name, s in groups.items():
        rate = s.get("flagged_essay_rate_pct")
        if "lower" in name.lower():
            lower = rate
        elif "higher" in name.lower():
            higher = rate
    if lower is None or higher is None:
        return None
    return round(lower - higher, 2)


def groups_measurable(groups: dict) -> bool:
    """True only if BOTH proficiency groups have >= MIN_ELIGIBLE_PER_GROUP eligibles."""
    seen = {"lower": False, "higher": False}
    for name, s in groups.items():
        for key in seen:
            if key in name.lower():
                seen[key] = s.get("n_eligible", 0) >= MIN_ELIGIBLE_PER_GROUP
    return all(seen.values())


def _essay_outliers(text: str) -> tuple[int, bool]:
    """(flagged-paragraph count, eligible?) for one essay."""
    fps = extract_fingerprints(text)
    eligible = len(fps) >= MIN_PARAGRAPHS
    if not eligible:
        return 0, False
    return len(detect_outliers(fps)), True


def measure(corpus: str | None, limit: int | None) -> dict:
    # Imported lazily: fpr_subgroup_gate pulls the full DetectionRunner ML stack at
    # import time, which this script otherwise never needs. DEFAULT_CORPUS is
    # resolved here (not in main()) for the same reason — no gate import outside
    # this function.
    from calibration.fpr_subgroup_gate import DEFAULT_CORPUS, _ai_texts, _proficiency_groups

    if corpus is None:
        corpus = DEFAULT_CORPUS
    groups_files = _proficiency_groups(corpus)
    if not any(groups_files.values()):
        print(f"No SCoCESLE essays found under {corpus!r} — pass --corpus.", file=sys.stderr)
        sys.exit(2)

    result: dict = {"corpus": corpus, "limit": limit, "groups": {}}
    for gname, files in sorted(groups_files.items()):
        use = files[:limit] if limit else files
        counts, eligs = [], []
        for fp in use:
            c, e = _essay_outliers(Path(fp).read_text(encoding="utf-8", errors="ignore"))
            counts.append(c)
            eligs.append(e)
        result["groups"][gname] = summarize_group(counts, eligs)

    # Informational control (never gates): fully-AI essays are uniform in style, so a
    # LOW flag rate here is the EXPECTED behavior — this detector finds within-doc
    # style breaks, it is not an AI detector.
    ai_texts = _ai_texts()
    ai_use = ai_texts[:limit] if limit else ai_texts
    ai_counts, ai_eligs = [], []
    for t in ai_use:
        c, e = _essay_outliers(t)
        ai_counts.append(c)
        ai_eligs.append(e)
    result["ai_cases_control"] = summarize_group(ai_counts, ai_eligs)

    result["flag_rate_gap_pts"] = flag_rate_gap_pts(result["groups"])
    result["max_flag_rate_gap_pts"] = MAX_FLAG_RATE_GAP_PTS
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=None,
                    help="defaults to fpr_subgroup_gate.DEFAULT_CORPUS (resolved lazily)")
    ap.add_argument("--limit", type=int, default=None, help="essays per group (smoke)")
    ap.add_argument("--out", default=None, help="write full JSON result here")
    args = ap.parse_args()

    res = measure(args.corpus, args.limit)

    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))

    gap = res["flag_rate_gap_pts"]
    if gap is None or not groups_measurable(res["groups"]):
        print(
            f"VERDICT: UNMEASURABLE — a proficiency group is missing or has fewer than "
            f"{MIN_ELIGIBLE_PER_GROUP} eligible essays; the gap would be noise.",
            file=sys.stderr,
        )
        sys.exit(1)
    if gap > MAX_FLAG_RATE_GAP_PTS:
        print(
            f"VERDICT: FAIL — lower-proficiency essays flagged {gap:+.2f} pts more than "
            f"higher (limit {MAX_FLAG_RATE_GAP_PTS}). Do NOT enable DRAFTPROOF_CONSISTENCY.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"VERDICT: PASS — flag-rate gap {gap:+.2f} pts within {MAX_FLAG_RATE_GAP_PTS}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd poc && python -m pytest calibration/test_consistency_esl_rates.py -v`
Expected: 5 passed

- [ ] **Step 5: Smoke the script on the real corpus**

Run: `cd poc && python calibration/consistency_esl_rates.py --limit 6`

`--limit 6` is a STRUCTURAL smoke test only — with at most 6 essays per group, `n_eligible` can never reach `MIN_ELIGIBLE_PER_GROUP = 20`, so a measurable PASS/FAIL is impossible by construction (and a higher limit wouldn't guarantee it either, since essays under `MIN_PARAGRAPHS` are structurally ineligible). Task 2's full-corpus run is the real calibration.

Expected:
- JSON produced with per-group `n_essays` / `n_eligible` / rates, no traceback
- `VERDICT: UNMEASURABLE` on stderr and **exit code 1** — this is the CORRECT smoke outcome, not a failure

(If the corpus dir is absent on this machine: exit 2 with the "No SCoCESLE essays found" message — run the smoke on the machine that has it; the corpus lives at the `DEFAULT_CORPUS` path per the gate.)

- [ ] **Step 6: Commit**

```bash
git add poc/calibration/consistency_esl_rates.py poc/calibration/test_consistency_esl_rates.py
git commit -m "feat(calibration): ESL flag-rate disparity gate for the consistency panel

The pre-push ESL gate only measures ai_likelihood_score, so it structurally
cannot catch the consistency PANEL over-flagging lower-proficiency writers.
This script measures flagged-essay rates by SCoCESLE proficiency band
(stylometry-only, no ML) and fails above a named 5.0-pt gap — the deferred
'later product decision' gate from consistency_defence_readiness_build_plan.md."
```

---

### Task 2: Full-corpus calibration run + record the decision evidence

**Files:**
- Create: `poc/calibration/consistency_esl_rates.json` (committed evidence artifact)

**Interfaces:**
- Consumes: Task 1's script.
- Produces: the committed JSON + a PASS verdict. Task 5 (prod flip) is BLOCKED unless this task's verdict is PASS.

- [ ] **Step 1: Run the full corpus**

Run: `cd poc && python calibration/consistency_esl_rates.py --out calibration/consistency_esl_rates.json`
Expected: full 272-essay run (fast — stylometry only, no ML scoring), JSON printed, `VERDICT: PASS` and exit 0. Record the exact `flag_rate_gap_pts` and per-group rates.

- [ ] **Step 2: If FAIL — stop and report, do not tune**

If the verdict is FAIL or UNMEASURABLE: do NOT adjust `MAX_FLAG_RATE_GAP_PTS` or any `poc/detect/stylometry/outliers.py` threshold to force a pass (NO-HARDCODE / no-gaming rule). Stop, report the numbers to the owner, and treat threshold redesign as a separate Opus-tier task. Tasks 3–5 remain valid but Task 5 stays blocked.

- [ ] **Step 3: Sanity-check the AI-cases control**

In the JSON, `ai_cases_control.flagged_essay_rate_pct` should be LOW (uniform-style AI essays should rarely trip a within-doc style-break detector). If it is high (>50%), the detector is noisier than designed — report to owner alongside Step 1's numbers before proceeding.

- [ ] **Step 4: Owner sign-off (BLOCKING for Task 5)**

Present to the owner: the per-group flag rates, `flag_rate_gap_pts`, `n_eligible` and `eligibility_rate_pct` per group (a large eligibility skew between groups deserves attention even on a PASS), the AI-cases control rate, and the `MAX_FLAG_RATE_GAP_PTS = 5.0` threshold with its one-sided "lower-proficiency over-flagging" rationale. The owner confirms (or tightens) the threshold and approves proceeding to the prod flip. A PASS exit code alone does NOT authorize Task 5 — this is a human product decision, per the build plan's "separate, later product decision" clause.

- [ ] **Step 5: Commit the evidence**

```bash
git add poc/calibration/consistency_esl_rates.json
git commit -m "chore(calibration): commit SCoCESLE consistency flag-rate evidence (PASS, gap <recorded value> pts)"
```

(Substitute the real gap value in the message — memory rule `feedback_commit_messages`.)

---

### Task 3: Flag-ON score-neutrality proof

**Files:** none created — verification only (evidence goes in the task report / final commit message).

**Interfaces:**
- Consumes: existing `poc/calibration/fpr_subgroup_gate.py`, existing consistency test suite.
- Produces: two gate-output JSONs proving byte-equal scores OFF vs ON. Task 5 cites them.

- [ ] **Step 1: Run the existing consistency suite (regression check)**

Run: `cd poc && python -m pytest detect/test_consistency.py test_consistency_report_parity.py report/test_consistency_panel.py detect/stylometry/test_features.py detect/stylometry/test_outliers.py -v`
Expected: all pass (these shipped green in merge `d8c9f002`; a failure here means `main` drifted — stop and investigate before proceeding).

- [ ] **Step 2: Score-neutrality smoke, OFF vs ON**

Run:
```bash
cd poc
DRAFTPROOF_CONSISTENCY=0 python calibration/fpr_subgroup_gate.py --limit 12 --out /tmp/gate_off.json
DRAFTPROOF_CONSISTENCY=1 python calibration/fpr_subgroup_gate.py --limit 12 --out /tmp/gate_on.json
python - <<'EOF'
import json
off = json.load(open("/tmp/gate_off.json"))
on = json.load(open("/tmp/gate_on.json"))
assert off == on, "consistency flag moved ai_likelihood_score — contract broken"
print("SCORE-NEUTRAL: OFF and ON gate outputs are identical")
EOF
```
Expected: `SCORE-NEUTRAL: OFF and ON gate outputs are identical`. Full-dict equality is valid: the gate's `measure()` output (fpr_subgroup_gate.py:144-159) contains only rounded scores/FPR/AUC/counts — no timestamps or paths — and the local stack has proven run-to-run determinism (2026-07-21 full-gate re-run: AUC delta +0.0000). On a mismatch, diff field-by-field FIRST before declaring the advisory contract broken — a run-nondeterminism blip is not a contract violation; only a systematic score shift with the flag ON is. If systematic: stop, report, fix before any enablement.

- [ ] **Step 3: Record evidence**

No commit (nothing changed). Paste both verdict lines + the SCORE-NEUTRAL line into the task report; Task 5's push commit message cites them.

---

### Task 4: Rendered-artifact verification (flag ON, local)

**Files:** none committed — `test_output/` artifacts are gitignored evidence.

**Interfaces:**
- Consumes: `poc/detect_pipeline.py` CLI; `_document_with_shifted_paragraph` fixture.
- Produces: a rendered PDF + markdown visually showing the "Writing-style outliers" panel. Task 5 requires this confirmation.

- [ ] **Step 1: Generate a report with the flag ON on a known-shifted document**

```bash
cd poc
python -c "from detect.test_consistency import _document_with_shifted_paragraph as d; open('/tmp/consistency_demo.txt','w').write(d())"
DRAFTPROOF_CONSISTENCY=1 python detect_pipeline.py /tmp/consistency_demo.txt
```
Expected: `test_output/draftproof_<timestamp>.{md,json,pdf}` written.

- [ ] **Step 2: Verify the JSON key**

Run: `python -c "import json,glob; p=sorted(glob.glob('test_output/draftproof_*.json'))[-1]; d=json.load(open(p)); print('consistency_display' in d and d['consistency_display'] is not None)"`
Expected: `True`

- [ ] **Step 3: RENDER and visually inspect (mandatory — not grep)**

Open the newest `test_output/draftproof_*.pdf` (Read tool on the PDF, or `open` it) and confirm the "Writing-style outliers" panel appears with the shifted paragraph's excerpt and plain-English deviating features. Per memory `feedback_verify_rendered_artifacts`, grep of the markdown does NOT satisfy this step — inspect the rendered PDF.

- [ ] **Step 4: Negative control**

Run: `cd poc && python detect_pipeline.py /tmp/consistency_demo.txt` (flag unset) and confirm the new PDF has NO outliers panel and the JSON lacks `consistency_display`.
Expected: panel absent — OFF-state parity holds end-to-end.

---

### Task 5: Production enablement (Koyeb runtime env) + live verification

**Files:** none — prod config change only. BLOCKED until Task 2 = PASS, Task 3 = SCORE-NEUTRAL, Task 4 = visually confirmed.

**Interfaces:**
- Consumes: evidence from Tasks 2–4.
- Produces: `DRAFTPROOF_CONSISTENCY=1` on the Koyeb WORKER service; live report showing the panel.

- [ ] **Step 1: Push the calibration commits to main**

Follow the `git-push-protocol` skill. Note: neither commit touches `poc/detect/`, so the pre-push ESL gate hook will correctly skip (it triggers on `poc/detect/` paths only). Push = Koyeb auto-deploy of the unchanged-behavior code.

```bash
git status && git log origin/main..HEAD --oneline
git push origin main
```

- [ ] **Step 2: Flip the worker env var**

The report is composed in the WORKER (`worker/app/tasks.py` → `poc`); the API never composes `consistency_display` (its `_composers/` has no consistency copy — scan-time only, mirroring claim_graph's posture: older reports simply won't have the panel). So only the worker service needs the var. Worker svc id `2af91bbc`; Koyeb runtime env OVERRIDES entrypoint defaults.

```bash
koyeb service update 2af91bbc --env DRAFTPROOF_CONSISTENCY=1
```
Expected: new deployment rolls out. (Do NOT change the repo default in `consistency.py` — repo default stays OFF as the documented kill-switch posture, exactly like `DRAFTPROOF_CLAIM_GRAPH`.)

- [ ] **Step 3: Live verification**

Run a REAL scan on production with a multi-paragraph document containing one deliberately different-style paragraph (Task 4's fixture text is fine). Confirm:
1. Web report page shows the ConsistencyRisk panel (`draftproof-frontend/src/pages/report/ConsistencyRisk.jsx` renders when `consistency_display` is present).
2. The downloaded PDF shows the same panel (render parity — web and PDF build from the same `poc/report/render.py` template).
3. Tier / ai_likelihood_score on this scan are plausible and unchanged in character (advisory contract).
4. Rewrite-page surface: rewrite-comparison reports are composed by the same worker → `report.py` path under the same env var, so they carry `consistency_display` automatically — run one rewrite and confirm the panel (or its absence) is coherent there too. If it does not appear on the rewrite surface, that is the claim_graph-precedent scan-only posture — record which it is in the task report so the "ALL surfaces" house rule has an explicit, stated answer rather than an assumption.

- [ ] **Step 4: Rollback line (record, don't run)**

`koyeb service update 2af91bbc --env DRAFTPROOF_CONSISTENCY=0` — single env flip, no deploy of code needed.

- [ ] **Step 5: Update memory**

Update `project_authorship_submission_risk_gap_assessment.md` (its "Consistency risk: MISSING entirely" is stale — the build shipped 2026-07-18 in merge `d8c9f002`) and add the activation outcome + evidence numbers to project memory.

---

## Explicitly OUT of scope

- **Defence-readiness activation** (`DRAFTPROOF_DEFENCE_CHECK`, Tasks 5–8 of the build plan — also fully built): separate decision; it additionally needs migrations `014`/`015` applied MANUALLY on Neon (memory: SQL migrations are manual there) and an LLM-cost review. Do not bundle with this flip.
- **`LOCAL_OUTLIER_FACTOR` strategy** (`NotImplementedError` stub in `outliers.py`): documented future work for n≥12; YAGNI now.
- **Threshold tuning of `outliers.py`**: only reopened (as an Opus-tier task) if Task 2 FAILs.
- **Read-time API backfill of `consistency_display` for older reports**: claim_graph precedent is scan-time-only; revisit only on user demand.

## Outcome (2026-07-21 — plan executed; activation NOT performed)

Tasks 1–4 executed (commits e1e95481..5a12f746). **Task 5 was NOT executed and must not be until the
threshold work below lands.** Evidence chain:

- Per-essay ESL gate: **UNMEASURABLE** — SCoCESLE eligibility ~2% (3 higher / 2 lower essays ≥6
  paragraphs; `consistency_esl_rates.json`). The MIN_ELIGIBLE_PER_GROUP floor worked as designed.
- Redesigned `--mode concat` (symmetric same-band concatenation): **PASS_SATURATED** — flag-rate gap
  0.0 pts (n 53/42, seed-robust), continuous metric agrees (4.52 vs 4.38 outliers/doc). **ESL-fairness
  blocker removed**; binary rate saturates at 100% (ceiling effect, labeled in the verdict).
- Single-author human precision check (`consistency_human_precision.py`, MEASUREMENT_ONLY): 49
  full-length Gutenberg essays → 100% of docs flagged, mean 8.49 outliers/doc. **Correct per-paragraph
  framing** (Fable review): ≈31% of paragraphs flagged on genuine single-author human prose, vs ≈67%
  on multi-author concats — the detector discriminates ~2× but its absolute false-positive rate is
  disqualifying; the one eligible modern ESL essay flagged 6/6 paragraphs, so register/era does not
  explain it away. Do NOT cite the earlier "similar magnitude to concat" per-doc comparison — it is
  wrong once normalized per paragraph.
- Likely mechanism: `OUTLIER_THRESHOLD=3.5` (poc/detect/stylometry/outliers.py) is Iglewicz–Hoaglin's
  single-comparison cutoff, but the score takes a MAX over per-feature robust-z's with leave-one-out
  on small n — structurally inflating false positives.

**Standing decision: `DRAFTPROOF_CONSISTENCY` stays OFF** (advisory-mode enablement rejected too — at
~31% per-paragraph FP most of the ≤12 displayed cards would be noise; precision-first forbids it).
Path to activation: a bounded OUTLIER_THRESHOLD derivation task — hold per-paragraph FP ≤~5% on the
human corpus (correcting the max-over-features effect), then re-run this plan's precision check AND
the ESL-parity gate before any enable decision.

## Model routing (per orchestration hierarchy)

| Task | Model |
|---|---|
| 1 (script + tests) | Sonnet |
| 2 (corpus run + evidence) | Sonnet (mechanical; escalate to owner on FAIL — no self-tuning) |
| 3 (neutrality proof) | Sonnet |
| 4 (render verify) | Sonnet |
| 5 (prod flip) | Driver inline (state-changing prod action; human-confirmed) |
| Final review | Fable ② advisor gate |
