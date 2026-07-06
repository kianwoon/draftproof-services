"""Paid re-calibration orchestration — Phase 2 detail of the V7 re-tune cycle.

Runs the two Modal-cost scripts (`v7_deberta_academic_calibrate.py`,
`v7_fused_gate_run.py`) against a STAGING dir, never the committed repo paths.
`deberta_fit_calibrator.py` is EXCLUDED by design: superseded, source of the
production 0%-bug (see its own docstring). Injectable `runner` so tests never
spawn a real subprocess / hit Modal.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_CALIBRATION_DIR = Path(__file__).resolve().parents[1]
ACADEMIC_SCRIPT = _CALIBRATION_DIR / "v7_deberta_academic_calibrate.py"
FUSED_SCRIPT = _CALIBRATION_DIR / "v7_fused_gate_run.py"


@dataclass(frozen=True)
class CalibrationResult:
    ran: bool
    fused_verdict: object
    staging_dir: str
    steps: list[str] = field(default_factory=list)


def run_calibration(staging_dir: Path, corpus: Path | None, weights_path: Path | None,
                    limit: int | None, runner=subprocess.run) -> CalibrationResult:
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    steps: list[str] = []

    academic_out = staging_dir / "academic.json"
    academic_cmd = [sys.executable, str(ACADEMIC_SCRIPT), "--out", str(academic_out)]
    if corpus is not None:
        academic_cmd += ["--corpus", str(corpus)]
    if limit is not None:
        academic_cmd += ["--limit-per-group", str(limit)]

    proc = runner(academic_cmd, capture_output=True, text=True)
    steps.append("academic")
    if proc.returncode != 0:
        return CalibrationResult(ran=True, fused_verdict="error:academic",
                                 staging_dir=str(staging_dir), steps=steps)

    fused_out = staging_dir / "fused.json"
    fused_progress = staging_dir / "fused_progress.jsonl"
    fused_cmd = [sys.executable, str(FUSED_SCRIPT), "--out", str(fused_out),
                 "--progress", str(fused_progress)]
    if weights_path is not None:
        fused_cmd += ["--weights", str(weights_path)]
    if corpus is not None:
        fused_cmd += ["--corpus", str(corpus)]

    proc = runner(fused_cmd, capture_output=True, text=True)
    steps.append("fused")
    if proc.returncode != 0:
        return CalibrationResult(ran=True, fused_verdict="error:fused",
                                 staging_dir=str(staging_dir), steps=steps)

    fused_verdict: object = "unknown"
    if fused_out.exists():
        try:
            data = json.loads(fused_out.read_text())
            fused_verdict = data.get("verdict", "unknown")
        except (json.JSONDecodeError, OSError):
            fused_verdict = "unknown"

    return CalibrationResult(ran=True, fused_verdict=fused_verdict,
                             staging_dir=str(staging_dir), steps=steps)
