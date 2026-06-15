"""End-to-end detector gate: full DetectionRunner -> report over the labeled corpus.

Complements measure_detector_signals.py (per-signal). This one runs the WHOLE scan
pipeline (incl. ML predictability/semantic signals) and reports the OUTPUT that ships:
overall tier + ai_likelihood_score. Use it to confirm that DROPPING an overfit hardcoded
detector (one whose per-signal AUC a structural proxy can't preserve) does not regress the
OVERALL AI-vs-human separation — the real gate for "drop & lean on statistics".

Needs the ML stack (scipy/transformers). Slower (~few sec/doc).

Usage:
    python calibration/measure_end_to_end.py --out PATH            # write baseline
    python calibration/measure_end_to_end.py --compare PATH        # diff vs baseline
    python calibration/measure_end_to_end.py --limit 8             # quick subset
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_POC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detect.run import DetectionRunner
from detect.document_structure import normalize_submitted_text
from report.report import ReportBuilder, report_to_dict

HERE = Path(__file__).resolve().parent
CASES = HERE / "authorship_cases"
DEFAULT_OUT = HERE.parent / "test_output" / "_end_to_end_baseline.json"
_TIER_RANK = {"green": 0, "amber": 1, "orange": 2, "red": 3}


def scan_text(runner: DetectionRunner, text: str) -> dict:
    text = normalize_submitted_text(str(text or ""))
    det = runner.run_all(text)
    b = ReportBuilder()
    b.add_detection_report(det)
    if getattr(det, "postprocess_results", None):
        b.add_postprocess_results(det.postprocess_results)
    b.set_meta(scan_time=0.0, original_text=text)
    return report_to_dict(b.build())


def measure(limit: int | None) -> dict:
    runner = DetectionRunner()
    rows = []
    paths = sorted(glob.glob(str(CASES / "*.json")))
    for path in paths:
        d = json.loads(Path(path).read_text())
        label = (d.get("authorship") or "").strip().lower()
        if label not in ("ai", "human") or not d.get("text"):
            continue
        rows.append((d.get("case_id"), label, d["text"]))
    if limit:
        ai = [r for r in rows if r[1] == "ai"][: limit // 2]
        hu = [r for r in rows if r[1] == "human"][: limit // 2]
        rows = ai + hu

    out = {"per_case": {}, "ai_likelihood": {}, "tier": {}}
    ai_scores, hu_scores, ai_tier, hu_tier = [], [], [], []
    for cid, label, text in rows:
        rep = scan_text(runner, text)
        badge = rep.get("ai_risk_badge") or {}
        score = float(badge.get("ai_likelihood_score") or 0.0)
        tier = str(rep.get("overall_tier") or badge.get("tier") or "").lower()
        out["per_case"][cid] = {"label": label, "ai_likelihood": round(score, 2), "tier": tier}
        (ai_scores if label == "ai" else hu_scores).append(score)
        (ai_tier if label == "ai" else hu_tier).append(_TIER_RANK.get(tier, 0))

    def auc(a, h):
        if not a or not h:
            return float("nan")
        wins = sum(1 for x in a for y in h if x > y) + 0.5 * sum(1 for x in a for y in h if x == y)
        return round(wins / (len(a) * len(h)), 4)

    out["ai_likelihood"] = {
        "ai_mean": round(statistics.mean(ai_scores), 2) if ai_scores else None,
        "human_mean": round(statistics.mean(hu_scores), 2) if hu_scores else None,
        "auc": auc(ai_scores, hu_scores),
    }
    out["tier"] = {
        "ai_mean_rank": round(statistics.mean(ai_tier), 3) if ai_tier else None,
        "human_mean_rank": round(statistics.mean(hu_tier), 3) if hu_tier else None,
        "auc": auc(ai_tier, hu_tier),
    }
    out["n"] = {"ai": len(ai_scores), "human": len(hu_scores)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--compare", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    res = measure(args.limit)
    print(f"n: {res['n']}   ({time.time()-t0:.0f}s)")
    print(f"ai_likelihood: AI {res['ai_likelihood']['ai_mean']} vs human "
          f"{res['ai_likelihood']['human_mean']}  AUC {res['ai_likelihood']['auc']}")
    print(f"tier rank:     AI {res['tier']['ai_mean_rank']} vs human "
          f"{res['tier']['human_mean_rank']}  AUC {res['tier']['auc']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        da = res["ai_likelihood"]["auc"] - base["ai_likelihood"]["auc"]
        dt = res["tier"]["auc"] - base["tier"]["auc"]
        print(f"\nvs {Path(args.compare).name}:")
        print(f"  ai_likelihood AUC delta: {da:+.4f}")
        print(f"  tier AUC delta:          {dt:+.4f}")
        print("  RESULT:", "REGRESSION" if (da <= -0.05 or dt <= -0.05) else "holds")


if __name__ == "__main__":
    main()
