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

    # Track the three SHIPPED scores: DraftProof ai_likelihood, the external-detector proxy
    # (external_grouped_v2, which the de-hardcoded writing-pattern signals feed), and writing
    # quality. Each is "higher = more AI"; report AI-vs-human separation (AUC) for all three.
    metrics = {"ai_likelihood": ([], []), "external_proxy": ([], []), "writing_quality": ([], [])}
    out = {"per_case": {}}
    for cid, label, text in rows:
        rep = scan_text(runner, text)
        badge = rep.get("ai_risk_badge") or {}
        ext = badge.get("external_detector_estimate") or {}
        rating = badge.get("authorship_rating") or {}
        vals = {
            "ai_likelihood": float(badge.get("ai_likelihood_score") or 0.0),
            "external_proxy": float(ext.get("score") or 0.0),
            "writing_quality": float(rating.get("writing_quality_score") or 0.0),
        }
        tier = str(rep.get("overall_tier") or badge.get("tier") or "").lower()
        out["per_case"][cid] = {"label": label, **{k: round(v, 2) for k, v in vals.items()}, "tier": tier}
        for k, v in vals.items():
            metrics[k][0 if label == "ai" else 1].append(v)

    def auc(a, h):
        if not a or not h:
            return float("nan")
        wins = sum(1 for x in a for y in h if x > y) + 0.5 * sum(1 for x in a for y in h if x == y)
        return round(wins / (len(a) * len(h)), 4)

    for k, (ai_v, hu_v) in metrics.items():
        out[k] = {
            "ai_mean": round(statistics.mean(ai_v), 2) if ai_v else None,
            "human_mean": round(statistics.mean(hu_v), 2) if hu_v else None,
            "auc": auc(ai_v, hu_v),
        }
    out["n"] = {"ai": len(metrics["ai_likelihood"][0]), "human": len(metrics["ai_likelihood"][1])}
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
    for k in ("ai_likelihood", "external_proxy", "writing_quality"):
        print(f"{k:<16} AI {res[k]['ai_mean']} vs human {res[k]['human_mean']}  AUC {res[k]['auc']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        print(f"\nvs {Path(args.compare).name} (AUC delta; <= -0.05 = REGRESSION):")
        worst = 0.0
        for k in ("ai_likelihood", "external_proxy", "writing_quality"):
            if k in base:
                d = res[k]["auc"] - base[k]["auc"]
                worst = min(worst, d)
                print(f"  {k:<16} {d:+.4f}" + ("  REGRESSION" if d <= -0.05 else ""))
        print("  RESULT:", "REGRESSION" if worst <= -0.05 else "holds")


if __name__ == "__main__":
    main()
