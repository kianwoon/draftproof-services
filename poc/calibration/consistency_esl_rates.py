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

# Mirrors fpr_subgroup_gate.py's sys.path setup: `detect/__init__.py` transitively
# imports `poc.predictability...`, so both poc/ (parents[1]) and the project root
# (parents[2], which makes `poc` importable) must be on sys.path for direct script
# invocation (`python calibration/consistency_esl_rates.py`) to work — pytest masks
# this because conftest.py already adds the project root for the test collector.
_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
