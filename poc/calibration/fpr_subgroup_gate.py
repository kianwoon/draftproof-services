"""Subgroup false-positive GATE: DraftProof ai_likelihood over the SCoCESLE ESL corpus,
broken down by PROFICIENCY (higher vs lower), with AUC vs the in-repo AI set.

WHY: post-hoc detectors over-flag lower-proficiency / formulaic ESL writing — the exact
bias 12+ universities cited when disabling Turnitin (Stanford/Liang: 61% of non-native
essays flagged). This gate PUBLISHES DraftProof's ESL false-positive rate BY PROFICIENCY
SUBGROUP and FAILS if a code change raises FPR, drops AUC, or widens the higher-vs-lower
parity gap. It is the standing version of the one-off corpus checks M4/M5/M6 needed.

It does NOT change any score — it measures the existing scorer. Its value is (1) a
regression safety net for future scoring changes and (2) a publishable subgroup-FPR claim.

SCoCESLE is LOCAL-ONLY (no redistribution license), so this is a LOCAL gate (run before
shipping scorer changes), NOT a GitHub-CI gate. The numeric baseline it writes IS
committable (numbers only, no corpus text) and is the publishable claim.

Usage:
    python calibration/fpr_subgroup_gate.py --compare                  # GATE: run + check vs committed baseline (exit 1 on regression)
    python calibration/fpr_subgroup_gate.py --out PATH                 # write a fresh baseline
    python calibration/fpr_subgroup_gate.py --compare PATH             # check vs a custom baseline
    python calibration/fpr_subgroup_gate.py --limit 12                 # quick subset
    python calibration/fpr_subgroup_gate.py --corpus "/path/to/SCoCESLE"
"""
from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json  # noqa: E402

from calibration.measure_end_to_end import scan_text  # noqa: E402  reuse the full-pipeline scorer
from detect.run import DetectionRunner  # noqa: E402

HERE = Path(__file__).resolve().parent
AI_CASES = HERE / "authorship_cases"
DEFAULT_CORPUS = "/Users/kianwoonwong/Downloads/Small Corpus of Colombian English as a Second Language Essays (SCoCESLE)"
DEFAULT_OUT = HERE.parent / "test_output" / "_fpr_subgroup_baseline.json"
DEFAULT_BASELINE = HERE / "fpr_subgroup_baseline.json"  # committed reference for bare --compare

# A human essay is a FALSE POSITIVE when its ai_likelihood reads as "AI-likely". 50% is the
# natural line (validated: ~97% of SCoCESLE humans score <50%). Report several for context.
FPR_THRESHOLDS = [40.0, 50.0, 60.0]
PRIMARY_THRESHOLD = 50.0

# Regression tolerances for --compare (exit non-zero past any of these).
MAX_FPR_RISE_PTS = 3.0       # overall ESL FPR@primary may not rise by more than this
MAX_AUC_DROP = 0.05          # AI-vs-human AUC may not drop by more than this
MAX_PARITY_WIDEN_PTS = 4.0   # the lower-vs-higher proficiency gap may not widen by more than this


def _proficiency_groups(corpus: str) -> dict:
    """Map 'higher'/'lower' -> list of essay .txt paths from the SCoCESLE subdirs.

    Excludes the combined 'SCoCESLE txt' folder (no 'proficiency' in its name) so essays
    are counted once. Tolerates the trailing space in the 'higher proficiency ' dir name.
    """
    base = Path(corpus)
    groups: dict[str, list[str]] = {"higher": [], "lower": []}
    for d in base.glob("*proficiency*"):
        name = d.name.lower()
        key = "higher" if "higher" in name else ("lower" if "lower" in name else None)
        if key:
            groups[key] += sorted(glob.glob(str(d / "*.txt")))
    return groups


def _ai_texts() -> list[str]:
    out = []
    for p in sorted(glob.glob(str(AI_CASES / "*.json"))):
        d = json.loads(Path(p).read_text())
        if (d.get("authorship") or "").lower() == "ai" and d.get("text"):
            out.append(d["text"])
    return out


def _fpr(vals: list[float], thr: float):
    return round(100.0 * sum(1 for v in vals if v >= thr) / len(vals), 1) if vals else None


def _dist(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 1),
        "p50": round(s[len(s) // 2], 1),
        "p90": round(s[min(len(s) - 1, int(len(s) * 0.9))], 1),
        "max": round(max(vals), 1),
    }


def _auc(ai: list[float], human: list[float]):
    if not ai or not human:
        return None
    wins = sum(1 for x in ai for y in human if x > y) + 0.5 * sum(1 for x in ai for y in human if x == y)
    return round(wins / (len(ai) * len(human)), 4)


def _score_texts(runner: DetectionRunner, texts: list[str]) -> list[float]:
    out = []
    for text in texts:
        rep = scan_text(runner, text)
        badge = rep.get("ai_risk_badge") or {}
        out.append(float(badge.get("ai_likelihood_score") or 0.0))
    return out


def measure(corpus: str, limit: int | None) -> dict:
    runner = DetectionRunner()
    groups = _proficiency_groups(corpus)
    if not groups["higher"] and not groups["lower"]:
        raise SystemExit(f"No SCoCESLE essays found under {corpus!r} — pass --corpus.")

    scores = {}
    for g in ("higher", "lower"):
        files = groups[g][: max(1, limit // 2)] if limit else groups[g]
        scores[g] = _score_texts(runner, [Path(fp).read_text(encoding="utf-8", errors="ignore") for fp in files])

    ai_texts = _ai_texts()
    if limit:
        ai_texts = ai_texts[: max(1, limit // 2)]
    ai_scores = _score_texts(runner, ai_texts)

    all_human = scores["higher"] + scores["lower"]
    fpr_by = {g: {str(t): _fpr(scores[g], t) for t in FPR_THRESHOLDS} for g in ("higher", "lower")}
    hi = fpr_by["higher"].get(str(PRIMARY_THRESHOLD))
    lo = fpr_by["lower"].get(str(PRIMARY_THRESHOLD))
    parity_gap = round(lo - hi, 1) if (hi is not None and lo is not None) else None

    return {
        "primary_threshold": PRIMARY_THRESHOLD,
        "fpr_pct": {
            "all": {str(t): _fpr(all_human, t) for t in FPR_THRESHOLDS},
            "higher": fpr_by["higher"],
            "lower": fpr_by["lower"],
        },
        # positive => lower-proficiency writers flagged MORE often (the fairness risk).
        "parity_gap_pct": parity_gap,
        "human_dist": {"all": _dist(all_human), "higher": _dist(scores["higher"]), "lower": _dist(scores["lower"])},
        "ai_dist": _dist(ai_scores),
        "auc_ai_vs_human": _auc(ai_scores, all_human),
        "n": {"higher": len(scores["higher"]), "lower": len(scores["lower"]), "ai": len(ai_scores)},
    }


def _print_summary(res: dict) -> None:
    t = str(res["primary_threshold"])
    fpr = res["fpr_pct"]
    print(f"\n  N: higher={res['n']['higher']} lower={res['n']['lower']} ai={res['n']['ai']}")
    print(f"  ESL FALSE-POSITIVE RATE @ ai_likelihood>={t}%  (lower = better):")
    print(f"    all ESL    : {fpr['all'][t]}%   (40%: {fpr['all']['40.0']}  60%: {fpr['all']['60.0']})")
    print(f"    higher-prof: {fpr['higher'][t]}%")
    print(f"    lower-prof : {fpr['lower'][t]}%   <- the at-risk subgroup")
    print(f"  PARITY GAP (lower - higher) @ {t}%: {res['parity_gap_pct']} pts  (>0 = lower-prof penalized more)")
    hd = res["human_dist"]["all"]
    print(f"  human ai_likelihood: mean {hd.get('mean')}%  p90 {hd.get('p90')}%  max {hd.get('max')}%")
    print(f"  AUC (AI vs human separation): {res['auc_ai_vs_human']}")


def _regression_check(res: dict, base: dict) -> bool:
    """Print deltas; return True on REGRESSION (caller exits non-zero)."""
    t = str(res["primary_threshold"])
    regressed = False
    print(f"\n  vs baseline (gate fails on any regression):")

    d_fpr = (res["fpr_pct"]["all"][t] or 0) - (base["fpr_pct"]["all"][t] or 0)
    bad = d_fpr > MAX_FPR_RISE_PTS
    regressed |= bad
    print(f"    ESL FPR@{t}%   {d_fpr:+.1f} pts" + ("   REGRESSION" if bad else ""))

    a_now, a_base = res.get("auc_ai_vs_human"), base.get("auc_ai_vs_human")
    if a_now is not None and a_base is not None:
        d_auc = a_now - a_base
        bad = d_auc <= -MAX_AUC_DROP
        regressed |= bad
        print(f"    AUC           {d_auc:+.4f}" + ("   REGRESSION" if bad else ""))

    p_now, p_base = res.get("parity_gap_pct"), base.get("parity_gap_pct")
    if p_now is not None and p_base is not None:
        d_par = p_now - p_base
        bad = d_par > MAX_PARITY_WIDEN_PTS
        regressed |= bad
        print(f"    parity gap    {d_par:+.1f} pts" + ("   REGRESSION" if bad else ""))

    print("    RESULT:", "REGRESSION — gate FAILED" if regressed else "holds")
    return regressed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--compare", nargs="?", const=str(DEFAULT_BASELINE), default=None,
        help="regression-check vs a baseline; bare --compare uses the committed reference",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    res = measure(args.corpus, args.limit)
    _print_summary(res)
    print(f"  ({time.time() - t0:.0f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"  wrote {out_path}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        if _regression_check(res, base):
            sys.exit(1)


if __name__ == "__main__":
    main()
