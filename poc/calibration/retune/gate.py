"""Thin wrapper over fpr_subgroup_gate.py — the single acceptance oracle. It owns the
exit-code contract (0 pass / 1 regression / 2 corpus-missing) so callers read a verdict,
not a return code."""
from __future__ import annotations
import subprocess, sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE_SCRIPT = HERE.parent / "fpr_subgroup_gate.py"

@dataclass(frozen=True)
class GateResult:
    passed: bool
    exit_code: int
    corpus_available: bool
    stdout: str

def run_fpr_gate(corpus: Path | None = None, baseline: Path | None = None,
                 limit: int | None = None, runner=subprocess.run) -> GateResult:
    cmd = [sys.executable, str(GATE_SCRIPT), "--compare"]
    if baseline is not None:
        cmd.append(str(baseline))
    if corpus is not None:
        cmd += ["--corpus", str(corpus)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    proc = runner(cmd, capture_output=True, text=True)
    code = proc.returncode
    return GateResult(
        passed=(code == 0),
        exit_code=code,
        corpus_available=(code != 2),
        stdout=(proc.stdout or ""),
    )
