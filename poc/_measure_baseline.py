"""Deterministic measurement harness for the V6 rewrite (Cerebras / gpt-oss-120b).

WHY: the gpt-oss writer samples at temperature 0.45-0.65, so a single e2e run's final_risk
varies by several points (observed 48.67 vs 50.31 vs 54.79 on the SAME code). That made
single-run deltas unmeasurable. This harness establishes a trustworthy baseline so any future
change can be judged as signal, not sampling noise.

It runs the full production rewrite N times and reports per-run + aggregate (mean/min/max/stdev)
for the graded number (final_risk) plus source_preserved and findings.

Usage:
    # near-deterministic (recommended for measuring a single change):
    DRAFTPROOF_V6_DETERMINISTIC=1 python poc/_measure_baseline.py 3
    # natural-variance baseline (what production actually ships):
    python poc/_measure_baseline.py 5
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poc.rewrite_v6.llm_config import cerebras_model_name, deterministic_mode, using_cerebras_direct, writer_model
from poc.rewrite_v6.production import run_rewrite_pipeline_v6

# The scan-report fixture fed as detect_json. Overridable via env so the harness is not pinned to a
# single (and now absent) artifact. Bake a fresh one with `python poc/_bake_fixture_report.py`.
REPORT = os.environ.get(
    "DRAFTPROOF_BASELINE_REPORT",
    "test_output/_fixture_scan_report.json",
)
OUT = "test_output/_measure_baseline"


def _run_once(detect_json: dict, run_index: int) -> dict | None:
    envelope = run_rewrite_pipeline_v6(detect_json=detect_json, output_dir=f"{OUT}/run_{run_index}")
    result = json.loads(Path(envelope["json_path"]).read_text())
    pt = result.get("candidate_generation_status", {}).get("pass_trace", [])
    srcs = Counter(p.get("selected_source") for p in pt)
    return {
        "final_risk": result.get("final_risk"),
        "original_risk": result.get("original_risk"),
        "source_preserved": srcs.get("source_preserved", 0),
        "passes": len(pt),
        "findings": (result.get("v6_scores") or {}).get("final", {}).get("finding_count"),
        "best_of_n": result.get("best_of_n"),
    }


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    detect_json = json.loads(Path(REPORT).read_text())
    print(f"cerebras_direct={using_cerebras_direct()} model={cerebras_model_name(writer_model())} "
          f"deterministic={deterministic_mode()} runs={n}\n")

    runs: list[dict] = []
    for i in range(1, n + 1):
        row = _run_once(detect_json, i)
        if row is None:
            print(f"run {i}: FAILED")
            continue
        runs.append(row)
        bon = row.get("best_of_n") or {}
        bon_str = ""
        if bon:
            bon_str = (f"  best_of_n={bon.get('candidate_ai_risks')} -> picked #{bon.get('selected_index')} "
                       f"({bon.get('selected_ai_risk')})")
        print(f"run {i}: final_risk={row['final_risk']:.2f}  "
              f"source_preserved={row['source_preserved']}/{row['passes']}  findings={row['findings']}{bon_str}")

    risks = [r["final_risk"] for r in runs if isinstance(r.get("final_risk"), (int, float))]
    if not risks:
        print("\nNo successful runs.")
        return 1
    mean = statistics.mean(risks)
    stdev = statistics.pstdev(risks) if len(risks) > 1 else 0.0
    print("\n========== BASELINE (final_risk) ==========")
    print(f"  runs        : {len(risks)}")
    print(f"  mean        : {mean:.2f}")
    print(f"  min / max   : {min(risks):.2f} / {max(risks):.2f}")
    print(f"  spread      : {max(risks) - min(risks):.2f}")
    print(f"  stdev       : {stdev:.2f}")
    print(f"\n>>> A future change only counts as real improvement if its mean beats {mean:.2f} "
          f"by more than the spread ({max(risks) - min(risks):.2f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
