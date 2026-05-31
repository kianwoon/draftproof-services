#!/usr/bin/env python3
"""CLI for the calibrated 2-signal Turnitin estimator (matches production exactly).

The estimator itself lives in detect.layer3_scoring.estimate_external_detector_segment_fraction
(predictability × formal-academic register, flagged share of qualifying sentences). This script is
a thin wrapper: it scans a text file through the FULL detector (same per-sentence data production
uses -- not the report dict's truncated `text` field), prints the estimate, and lists the flagged
formal sentences for coaching.

CALIBRATION HONESTY: thresholds were fit on ONE labeled doc (Reflection AT3 = Turnitin 27%, flagged
in the formal Critical-Analysis middle, conclusion spared). UNVALIDATED for other documents -- the
output is an estimate. Add labeled reports to turnitin_cases/ and re-fit via calibrate.py.

Run:  ~/.pyenv/versions/3.11.0/bin/python3 poc/calibration/segment_fraction_estimate.py test_content12.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def main() -> int:
    _load_env()
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "poc"))
    path = sys.argv[1] if len(sys.argv) > 1 else "test_content12.txt"
    text = (REPO / path).read_text() if not Path(path).is_absolute() else Path(path).read_text()

    from detect.run import DetectionRunner
    from report.report import ReportBuilder
    from detect.layer3_scoring import (
        estimate_external_detector_segment_fraction, register_score,
        _TURNITIN_TOP10_THRESHOLD, _TURNITIN_REGISTER_THRESHOLD,
    )

    builder = ReportBuilder()
    builder.add_detection_report(DetectionRunner().run_all(text))
    sentences = builder._pred_summary.sentences if builder._pred_summary else []

    est = estimate_external_detector_segment_fraction(sentences) or {"score": 0.0, "band": "low"}
    print(f"Turnitin estimate (2-signal segment-fraction): ~{est['score']:.0f}%  band={est['band']}  "
          f"[UNVALIDATED — 1-doc calibration; thresholds top10>={_TURNITIN_TOP10_THRESHOLD}, "
          f"register>={_TURNITIN_REGISTER_THRESHOLD}]")
    print("\nFlagged (formal + predictable) sentences — the AI-tell to rewrite in plain voice:")
    for s in sentences:
        t = s.get("sentence") or s.get("text") or ""
        p = s.get("top10_ratio")
        if not isinstance(p, (int, float)):
            continue
        reg = register_score(t)
        if p >= _TURNITIN_TOP10_THRESHOLD and reg >= _TURNITIN_REGISTER_THRESHOLD:
            print(f"  {s.get('sentence_id','?')}  top10={p:.2f} reg={reg:.1f}  {t[:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
