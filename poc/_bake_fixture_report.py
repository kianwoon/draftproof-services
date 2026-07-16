"""Bake a real scan-report fixture for the deterministic rewrite measurement harness.

The harness (`_measure_baseline.py`) needs a full scan report JSON to feed as `detect_json`. The
previously hardcoded fixture (`production_rewrite_ad62c7f1_20260529/...report.json`) is absent from
this worktree AND predates the enhanced-scan signals (sentence_issue_tags, grounding/reasoning
findings). This script scans a sample essay locally (composite-only, no Modal) and writes a fresh
report that carries those signals, so P1/P2/P3 can be measured.

Usage:
    python poc/_bake_fixture_report.py [input.txt] [output_report.json]
Defaults: input  = poc/test_output/_fixture_input.txt
          output = poc/test_output/_fixture_scan_report.json
Point the harness at it:  DRAFTPROOF_BASELINE_REPORT=<output> python poc/_measure_baseline.py 6
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from poc.detect_pipeline import run_detect  # noqa: E402

DEFAULT_IN = ROOT / "poc/test_output/_fixture_input.txt"
DEFAULT_OUT = ROOT / "poc/test_output/_fixture_scan_report.json"
WORK_DIR = ROOT / "poc/test_output/_fixture_bake"


def _findings_rows(report: dict) -> list[dict]:
    f = report.get("findings")
    if isinstance(f, list):
        return [x for x in f if isinstance(x, dict)]
    if isinstance(f, dict):
        return [x for v in f.values() if isinstance(v, list) for x in v if isinstance(x, dict)]
    return []


def main() -> int:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    text = in_path.read_text()

    result = run_detect(text, output_dir=str(WORK_DIR))
    report = json.loads(Path(result["json_path"]).read_text())
    out_path.write_text(json.dumps(report, indent=2, default=str))

    rows = _findings_rows(report)
    cats = Counter(x.get("category") for x in rows)
    with_sid = sum(1 for x in rows if x.get("sentence_id"))
    sit = report.get("sentence_issue_tags") or {}
    tagged = sit.get("sentences") if isinstance(sit, dict) else None
    paras = ((report.get("scan_intelligence") or {}).get("document") or {}).get("paragraphs") or []
    hs = report.get("highlight_segments") or []
    hs_with_pid_sid = sum(1 for s in hs if isinstance(s, dict) and s.get("sentence_id") and s.get("paragraph_id"))

    print("\n================= FIXTURE BAKED =================")
    print(f"  wrote        : {out_path}")
    print(f"  tier         : {result.get('tier')}")
    print(f"  paragraphs   : {len(paras)}")
    print(f"  findings     : {len(rows)}  (with sentence_id: {with_sid})")
    print(f"  categories   : {dict(cats)}")
    print(f"  highlight_seg: {len(hs)}  (with sid+pid: {hs_with_pid_sid})")
    print(f"  issue_tags   : present={bool(sit.get('present'))}  tagged_sentences={len(tagged) if isinstance(tagged, dict) else 0}")
    if isinstance(tagged, dict) and tagged:
        sample_types = Counter(t.get("type") for tags in tagged.values() for t in tags if isinstance(t, dict))
        print(f"  tag types    : {dict(sample_types)}")
    ok = bool(rows) and bool(paras) and bool(hs)
    print(f"\n  usable as measurement fixture: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
