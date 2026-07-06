import json
from pathlib import Path
from poc.calibration.retune import run_cycle
from poc.calibration.retune.gate import GateResult

def _seed_ai(tmp):
    ai = tmp / "authorship_cases"; ai.mkdir()
    (ai / "ai_x_00.json").write_text(json.dumps(
        {"case_id": "ai_x_00", "authorship": "ai", "source": "openai/gpt-5-mini",
         "family": "gpt-5", "text": "word " * 150}))
    return ai

def test_append_log_creates_header(tmp_path):
    log = tmp_path / "RETUNE_LOG.md"
    run_cycle.append_log(log, {"version": "v1", "n_rows": 5, "families": "gpt-5",
                               "gate": "PASS", "auc_line": "AUC 0.75"})
    txt = log.read_text()
    assert "| version |" in txt and "| v1 |" in txt and "PASS" in txt

def test_run_cycle_passes_and_logs(tmp_path):
    ai = _seed_ai(tmp_path)
    manifest = tmp_path / "manifest.json"
    log = tmp_path / "RETUNE_LOG.md"
    fake_gate = lambda **kw: GateResult(passed=True, exit_code=0, corpus_available=True, stdout="AUC 0.75")
    res = run_cycle.run_cycle(ai, None, manifest, log, now_iso="2026-07-06T00:00:00Z",
                              generate=False, gate_fn=fake_gate)
    assert res.passed
    assert manifest.exists()
    assert "PASS" in log.read_text()

def test_run_cycle_records_fail(tmp_path):
    ai = _seed_ai(tmp_path)
    fake_gate = lambda **kw: GateResult(passed=False, exit_code=1, corpus_available=True, stdout="")
    res = run_cycle.run_cycle(ai, None, tmp_path / "m.json", tmp_path / "L.md",
                              now_iso="2026-07-06T00:00:00Z", generate=False, gate_fn=fake_gate)
    assert not res.passed
    assert "FAIL" in (tmp_path / "L.md").read_text()
