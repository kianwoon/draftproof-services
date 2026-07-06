import json
from pathlib import Path

from poc.calibration.retune.recalibrate import run_calibration, CalibrationResult


class _FakeCompleted:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


def test_run_calibration_builds_commands_and_parses_verdict(tmp_path):
    staging = tmp_path / "staging"
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        # second call is the fused gate — write the fused.json it "produced"
        if "v7_fused_gate_run.py" in cmd[1]:
            fused = staging / "fused.json"
            fused.write_text(json.dumps({"verdict": {"gate_pass": True}}))
        return _FakeCompleted(0)

    weights_path = tmp_path / "weights.json"
    result = run_calibration(staging, corpus=Path("/some/corpus"), weights_path=weights_path,
                             limit=10, runner=fake_runner)

    assert isinstance(result, CalibrationResult)
    assert result.ran is True
    assert result.staging_dir == str(staging)
    assert staging.exists()
    assert len(calls) == 2

    academic_cmd, fused_cmd = calls
    assert "v7_deberta_academic_calibrate.py" in academic_cmd[1]
    assert "--out" in academic_cmd and str(staging / "academic.json") in academic_cmd
    assert "--corpus" in academic_cmd and str(Path("/some/corpus")) in academic_cmd
    assert "--limit-per-group" in academic_cmd and "10" in academic_cmd

    assert "v7_fused_gate_run.py" in fused_cmd[1]
    assert "--out" in fused_cmd and str(staging / "fused.json") in fused_cmd
    assert "--progress" in fused_cmd and str(staging / "fused_progress.jsonl") in fused_cmd
    assert "--weights" in fused_cmd and str(weights_path) in fused_cmd
    assert "--corpus" in fused_cmd and str(Path("/some/corpus")) in fused_cmd

    assert result.fused_verdict == {"gate_pass": True}


def test_run_calibration_no_optional_args(tmp_path):
    staging = tmp_path / "staging2"
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        return _FakeCompleted(0)

    result = run_calibration(staging, corpus=None, weights_path=None, limit=None, runner=fake_runner)
    academic_cmd, fused_cmd = calls
    assert "--corpus" not in academic_cmd
    assert "--limit-per-group" not in academic_cmd
    assert "--weights" not in fused_cmd
    assert "--corpus" not in fused_cmd
    assert result.fused_verdict == "unknown"


def test_run_calibration_subprocess_failure_sets_error_verdict(tmp_path):
    staging = tmp_path / "staging3"

    def fake_runner(cmd, **kwargs):
        return _FakeCompleted(1)

    result = run_calibration(staging, corpus=None, weights_path=None, limit=None, runner=fake_runner)
    assert result.ran is True
    assert result.fused_verdict.startswith("error:")
