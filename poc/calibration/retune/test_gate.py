from types import SimpleNamespace
from poc.calibration.retune import gate

def _fake_runner(code, out=""):
    def run(cmd, **kw):
        return SimpleNamespace(returncode=code, stdout=out, stderr="")
    return run

def test_pass():
    r = gate.run_fpr_gate(runner=_fake_runner(0, "AUC 0.75"))
    assert r.passed and r.exit_code == 0 and r.corpus_available

def test_regression():
    r = gate.run_fpr_gate(runner=_fake_runner(1))
    assert not r.passed and r.exit_code == 1 and r.corpus_available

def test_corpus_missing_is_not_regression():
    r = gate.run_fpr_gate(runner=_fake_runner(2))
    assert not r.passed and not r.corpus_available and r.exit_code == 2
