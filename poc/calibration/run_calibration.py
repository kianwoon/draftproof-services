#!/usr/bin/env python3
"""Turnitin calibration harness.

Ground truth = REAL Turnitin AI-writing reports (what the user actually faces).
We run DraftProof's own scan on the identical text and quantify the gap.

Why this exists: a real Turnitin report (Reflection AT3) scored a genuine, grounded,
human-authored reflection at 0% / '*' (clean) -- while our internal scan flagged the same
text at ~60% external "likely to be flagged". Our number over-flags reality, which scares
honest authors and undermines our (true) grounding guidance. This harness measures that gap
on real cases so we can recalibrate or demote the misleading number.

Add a case: drop a JSON in poc/calibration/turnitin_cases/ with shape:
    {"case_id","source","turnitin":{"ai_percent":N,"band":"clean|flagged"},"text":"..."}
Especially valuable: reports Turnitin DID flag (>=20% surfaced) -- we need both classes.

Run:  ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/run_calibration.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CASES_DIR = HERE / "turnitin_cases"
OUT = REPO / "test_output" / "_turnitin_calibration.json"


def _load_env() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def _scan(text: str) -> dict:
    """Run DraftProof's scan and pull the user-facing signals."""
    from rewrite_v3.pipeline import _scan_report  # heavy import; needs torch/transformers

    rep = _scan_report(text) or {}
    badge = rep.get("ai_risk_badge") or {}
    comp = badge.get("ai_components") or {}
    ext = badge.get("external_detector_estimate") or {}
    return {
        "draftproof_pct": badge.get("ai_likelihood_score"),
        "draftproof_tier": badge.get("tier"),
        "external_pct": ext.get("score"),
        "external_band": (ext.get("band") or "").lower() or None,
        "topk": comp.get("topk_pattern_raw") or comp.get("topk_pattern"),
        "predictability": comp.get("predictability"),
        "generic_assertion": comp.get("generic_assertion_risk"),
    }


def _load_cases() -> list[dict]:
    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        cases.append(json.loads(path.read_text()))
    return cases


def _num(x):
    return x if isinstance(x, (int, float)) else None


def main() -> int:
    _load_env()
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "poc"))

    cases = _load_cases()
    if not cases:
        print(f"No cases in {CASES_DIR}. Add real Turnitin reports as JSON.")
        return 1

    rows, false_alarms, overflags = [], 0, []
    print(f"{'case':<18}{'TII%':>6}{'TII band':>10}{'ours-ext%':>11}{'our band':>10}{'DP%':>6}{'tier':>7}{'overflag':>10}{'verdict':>14}")
    print("-" * 92)
    for c in cases:
        tii = c.get("turnitin", {})
        tii_pct = _num(tii.get("ai_percent"))
        tii_band = tii.get("band", "?")
        s = _scan(c["text"])
        ext = _num(s["external_pct"])
        # A FALSE ALARM: Turnitin says clean, but our band would scare the author.
        false_alarm = tii_band == "clean" and s["external_band"] in {"high", "elevated"}
        if false_alarm:
            false_alarms += 1
        overflag = (ext - tii_pct) if (ext is not None and tii_pct is not None) else None
        if overflag is not None:
            overflags.append(overflag)
        verdict = "FALSE ALARM" if false_alarm else ("agree" if tii_band != "clean" else "ok")

        def f(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "-"

        print(f"{c['case_id']:<18}{f(tii_pct):>6}{tii_band:>10}{f(ext):>11}{str(s['external_band']):>10}"
              f"{f(s['draftproof_pct']):>6}{str(s['draftproof_tier']):>7}{f(overflag):>10}{verdict:>14}")
        rows.append({"case_id": c["case_id"], "turnitin": tii, "ours": s,
                     "false_alarm": false_alarm, "overflag_points": overflag})

    n = len(rows)
    mean_overflag = (sum(overflags) / len(overflags)) if overflags else None
    print("-" * 92)
    print(f"cases: {n}   false alarms (Turnitin clean but we'd flag): {false_alarms}/{n}")
    if mean_overflag is not None:
        print(f"mean over-flag (our external% - Turnitin%): {mean_overflag:+.1f} points")
    print("\nNOTE: calibration needs BOTH clean and Turnitin-flagged cases. Add more real")
    print("reports (esp. ones Turnitin flagged) to fit a correction rather than just show the gap.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"n": n, "false_alarms": false_alarms, "mean_overflag_points": mean_overflag, "rows": rows},
        indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
