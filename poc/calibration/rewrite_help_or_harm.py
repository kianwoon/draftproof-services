#!/usr/bin/env python3
"""Does the DraftProof rewrite HELP or HARM a given document?

Runs the production rewrite N times (gpt-oss is high-variance) on a text file and reports the
before -> after delta on the signals that matter:
  - generic_assertion  (the ONLY signal that orders human<LLM<AI correctly; lower = better)
  - predictability     (a fluency tell; if the rewrite RAISES it, we harmed the text)
  - external / DP       (context)
plus source_preserved (paragraphs the rewrite left untouched).

Motivating case: run it on a doc Turnitin already cleared at 0% — if our rewrite raises risk on
its own signals, the pipeline should NOT touch already-clean content.

Usage:
    DRAFTPROOF_V6_DETERMINISTIC=1 ~/.pyenv/versions/3.11.0/bin/python3 \\
        poc/calibration/rewrite_help_or_harm.py test_content12.txt 3
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "poc"))

from rewrite_v3.pipeline import _scan_report  # noqa: E402
from rewrite_v6.production import run_rewrite_pipeline_v6  # noqa: E402

OUT = str(REPO / "test_output" / "_rewrite_help_or_harm")


def _metrics(scan: dict) -> dict:
    b = (scan or {}).get("ai_risk_badge") or {}
    c = b.get("ai_components") or {}
    e = b.get("external_detector_estimate") or {}
    return {
        "dp": b.get("ai_likelihood_score"),
        "ext": e.get("score"),
        "gen": c.get("generic_assertion_risk"),
        "pred": c.get("predictability"),
        "topk": c.get("topk_pattern_raw") or c.get("topk_pattern"),
    }


def _f(x):
    return f"{x:.0f}" if isinstance(x, (int, float)) else "-"


def main() -> int:
    text_path = sys.argv[1] if len(sys.argv) > 1 else "test_content12.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    text = (REPO / text_path).read_text() if not Path(text_path).is_absolute() else Path(text_path).read_text()

    detect_json = _scan_report(text)
    before = _metrics(detect_json)
    print(f"INPUT: {text_path}  ({len(text.split())} words)")
    print(f"BEFORE (original):  DP={_f(before['dp'])}  ext={_f(before['ext'])}  "
          f"gen={_f(before['gen'])}  pred={_f(before['pred'])}  topk={_f(before['topk'])}\n")

    afters, finals, preserved = [], [], []
    for i in range(1, n + 1):
        env = run_rewrite_pipeline_v6(detect_json=detect_json, output_dir=f"{OUT}/run_{i}")
        res = json.loads(Path(env["json_path"]).read_text())
        s = res.get("summary") or res
        a = _metrics(s.get("detect_scan_rewritten") or {})
        afters.append(a)
        finals.append(res.get("final_risk"))
        pt = res.get("candidate_generation_status", {}).get("pass_trace", [])
        srcs = Counter(p.get("selected_source") for p in pt)
        preserved.append((srcs.get("source_preserved", 0), len(pt)))
        print(f"run {i}: DP={_f(a['dp'])}  ext={_f(a['ext'])}  gen={_f(a['gen'])}  "
              f"pred={_f(a['pred'])}  topk={_f(a['topk'])}  preserved={srcs.get('source_preserved',0)}/{len(pt)}  "
              f"final_risk={_f(res.get('final_risk'))}")

    def mean(key):
        vals = [a[key] for a in afters if isinstance(a[key], (int, float))]
        return statistics.mean(vals) if vals else None

    print("\n========== HELP OR HARM (before -> after mean of N) ==========")
    for key, better in (("gen", "lower"), ("pred", "lower"), ("ext", "lower"), ("dp", "lower")):
        b, a = before[key], mean(key)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = a - b
            verdict = "HELP" if delta < -1 else ("HARM" if delta > 1 else "flat")
            print(f"  {key:<5} {b:5.0f} -> {a:5.1f}   ({delta:+.1f})  [{verdict}]  (lower=better)")
    pres = sum(p[0] for p in preserved)
    tot = sum(p[1] for p in preserved)
    print(f"  source_preserved: {pres}/{tot} paragraph-passes across {n} runs")
    print("\nKEY: 'gen' is the discriminative signal. If gen drops we genuinely helped; if 'pred'")
    print("rises we manufactured a fluency tell. On an already-clean doc, flat+preserved is fine;")
    print("HARM (pred up / gen up) means the pipeline should leave clean content alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
