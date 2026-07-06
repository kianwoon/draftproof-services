import json
from pathlib import Path
from poc.calibration.retune import run_cycle
from poc.calibration.retune.gate import GateResult
from poc.calibration.retune.recalibrate import CalibrationResult
from poc.calibration.retune import deepscan_cache

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

def test_run_cycle_paid_calls_calibrate_and_logs_verdict(tmp_path):
    ai = _seed_ai(tmp_path)
    manifest = tmp_path / "manifest.json"
    log = tmp_path / "RETUNE_LOG.md"
    fake_gate = lambda **kw: GateResult(passed=True, exit_code=0, corpus_available=True, stdout="AUC 0.75")
    fake_result = CalibrationResult(ran=True, fused_verdict="candidate-pass",
                                     staging_dir=str(tmp_path / "staging"), steps=["academic", "fused"])
    captured = {}
    def fake_calibrate(staging_dir, corpus, weights_path, limit, cache_path):
        captured["args"] = (staging_dir, corpus, weights_path, limit, cache_path)
        return fake_result

    res = run_cycle.run_cycle(ai, None, manifest, log, now_iso="2026-07-06T00:00:00Z",
                              generate=False, gate_fn=fake_gate, paid=True,
                              staging_dir=tmp_path / "staging", weights_path=tmp_path / "w.json",
                              limit=5, calibrate_fn=fake_calibrate)
    assert res.passed
    assert "candidate-pass" in log.read_text()
    received_staging, corpus, weights_path, limit, cache_path = captured["args"]
    assert Path(received_staging) == tmp_path / "staging" / "run-2026-07-06T00-00-00Z"
    assert corpus is None and weights_path == tmp_path / "w.json" and limit == 5
    assert Path(cache_path) == deepscan_cache.DEFAULT_CACHE
    assert not str(cache_path).startswith(str(received_staging))

def test_run_cycle_paid_uses_per_run_staging_subdir(tmp_path):
    ai = _seed_ai(tmp_path)
    manifest = tmp_path / "manifest.json"
    log = tmp_path / "RETUNE_LOG.md"
    fake_gate = lambda **kw: GateResult(passed=True, exit_code=0, corpus_available=True, stdout="AUC 0.75")
    received = []
    def fake_calibrate(staging_dir, corpus, weights_path, limit, cache_path):
        received.append(staging_dir)
        return CalibrationResult(ran=True, fused_verdict="candidate-pass",
                                  staging_dir=str(staging_dir), steps=["academic", "fused"])

    run_cycle.run_cycle(ai, None, manifest, log, now_iso="2026-07-06T00:00:00Z",
                        generate=False, gate_fn=fake_gate, paid=True,
                        staging_dir=tmp_path / "staging", calibrate_fn=fake_calibrate)

    assert len(received) == 1
    received_dir = Path(received[0])
    assert received_dir.name.startswith("run-")
    assert received_dir.exists()
    assert received_dir.parent == tmp_path / "staging"

def test_run_cycle_paid_different_timestamps_get_different_staging_dirs(tmp_path):
    ai = _seed_ai(tmp_path)
    fake_gate = lambda **kw: GateResult(passed=True, exit_code=0, corpus_available=True, stdout="AUC 0.75")
    received = []
    def fake_calibrate(staging_dir, corpus, weights_path, limit, cache_path):
        received.append(Path(staging_dir))
        return CalibrationResult(ran=True, fused_verdict="candidate-pass",
                                  staging_dir=str(staging_dir), steps=["academic", "fused"])

    run_cycle.run_cycle(ai, None, tmp_path / "m1.json", tmp_path / "L1.md",
                        now_iso="2026-07-06T00:00:00Z", generate=False, gate_fn=fake_gate,
                        paid=True, staging_dir=tmp_path / "staging", calibrate_fn=fake_calibrate)
    run_cycle.run_cycle(ai, None, tmp_path / "m2.json", tmp_path / "L2.md",
                        now_iso="2026-07-06T01:30:00Z", generate=False, gate_fn=fake_gate,
                        paid=True, staging_dir=tmp_path / "staging", calibrate_fn=fake_calibrate)

    assert len(received) == 2
    assert received[0] != received[1]
    assert received[0].exists() and received[1].exists()

def test_run_cycle_free_path_never_calls_calibrate(tmp_path):
    ai = _seed_ai(tmp_path)
    manifest = tmp_path / "manifest.json"
    log = tmp_path / "RETUNE_LOG.md"
    fake_gate = lambda **kw: GateResult(passed=True, exit_code=0, corpus_available=True, stdout="AUC 0.75")
    def exploding_calibrate(*a, **kw):
        raise AssertionError("calibrate_fn must not be called when paid=False")

    res = run_cycle.run_cycle(ai, None, manifest, log, now_iso="2026-07-06T00:00:00Z",
                              generate=False, gate_fn=fake_gate, paid=False,
                              calibrate_fn=exploding_calibrate)
    assert res.passed
    assert "skipped" in log.read_text()

def test_run_cycle_paid_defaults_cache_path_to_persistent_default(tmp_path):
    ai = _seed_ai(tmp_path)
    manifest = tmp_path / "manifest.json"
    log = tmp_path / "RETUNE_LOG.md"
    fake_gate = lambda **kw: GateResult(passed=True, exit_code=0, corpus_available=True, stdout="AUC 0.75")
    captured = {}
    def fake_calibrate(staging_dir, corpus, weights_path, limit, cache_path):
        captured["cache_path"] = cache_path
        captured["staging_dir"] = staging_dir
        return CalibrationResult(ran=True, fused_verdict="candidate-pass",
                                  staging_dir=str(staging_dir), steps=["academic", "fused"])

    run_cycle.run_cycle(ai, None, manifest, log, now_iso="2026-07-06T00:00:00Z",
                        generate=False, gate_fn=fake_gate, paid=True,
                        staging_dir=tmp_path / "staging", calibrate_fn=fake_calibrate)

    assert Path(captured["cache_path"]) == deepscan_cache.DEFAULT_CACHE
    assert Path(captured["cache_path"]) != Path(captured["staging_dir"])
    assert not str(captured["cache_path"]).startswith(str(captured["staging_dir"]))

def test_run_cycle_paid_explicit_cache_path_overrides_default(tmp_path):
    ai = _seed_ai(tmp_path)
    manifest = tmp_path / "manifest.json"
    log = tmp_path / "RETUNE_LOG.md"
    fake_gate = lambda **kw: GateResult(passed=True, exit_code=0, corpus_available=True, stdout="AUC 0.75")
    custom_cache = tmp_path / "custom_cache" / "deepscan_scores.jsonl"
    captured = {}
    def fake_calibrate(staging_dir, corpus, weights_path, limit, cache_path):
        captured["cache_path"] = cache_path
        return CalibrationResult(ran=True, fused_verdict="candidate-pass",
                                  staging_dir=str(staging_dir), steps=["academic", "fused"])

    run_cycle.run_cycle(ai, None, manifest, log, now_iso="2026-07-06T00:00:00Z",
                        generate=False, gate_fn=fake_gate, paid=True,
                        staging_dir=tmp_path / "staging", calibrate_fn=fake_calibrate,
                        cache_path=custom_cache)

    assert Path(captured["cache_path"]) == custom_cache
