from contextlib import contextmanager

import app.db as db
import app.tasks as tasks


# --------------------------------------------------------------------------
# claim_scan_job + completed-CAS fake DB
#
# A tiny in-memory stand-in for Postgres modelling the ONE property that
# matters for the billed-but-failed fix: the conditional scan_jobs UPDATE only
# "wins" (RETURNING a row / rowcount > 0) when the row's current status matches
# the CAS predicate. claim -> status IN ('pending','retrying');
# update_job_status -> status NOT IN ('failed','canceled','completed').
# --------------------------------------------------------------------------
class _FakeScanCursor:
    def __init__(self, state):
        self.state = state
        self._last = None
        self.rowcount = 0

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        status = self.state["status"]
        if "status IN ('pending', 'retrying')" in s:
            # claim_scan_job CAS
            if status in ("pending", "retrying"):
                self.state["status"] = "processing"
                self._last = dict(self.state)
                self.rowcount = 1
            else:
                self._last = None
                self.rowcount = 0
        elif "status NOT IN ('failed', 'canceled', 'completed')" in s:
            # update_job_status CAS — first param is the new status
            new_status = params[0]
            if status not in ("failed", "canceled", "completed"):
                self.state["status"] = new_status
                self.rowcount = 1
            else:
                self.rowcount = 0
            self._last = None
        else:  # pragma: no cover - defensive
            self._last = None
            self.rowcount = 0

    def fetchone(self):
        return self._last


@contextmanager
def _fake_scan_conn(state):
    yield type("C", (), {"cursor": lambda self: _FakeScanCursor(state)})()


def _install_scan_db(monkeypatch, status):
    state = {"id": "sc1", "user_id": "u1", "status": status}
    monkeypatch.setattr(db, "get_conn", lambda: _fake_scan_conn(state))
    return state


def test_claim_scan_job_wins_from_pending(monkeypatch):
    state = _install_scan_db(monkeypatch, "pending")
    row = db.claim_scan_job("sc1", progress_percent=10, progress_message="Preparing scan")
    assert row is not None
    assert state["status"] == "processing"


def test_claim_scan_job_wins_from_retrying(monkeypatch):
    state = _install_scan_db(monkeypatch, "retrying")
    assert db.claim_scan_job("sc1") is not None
    assert state["status"] == "processing"


def test_claim_scan_job_rejects_already_processing(monkeypatch):
    # A redelivered / duplicate acks_late run must NOT re-claim a running job.
    state = _install_scan_db(monkeypatch, "processing")
    assert db.claim_scan_job("sc1") is None
    assert state["status"] == "processing"


def test_claim_scan_job_rejects_terminal_states(monkeypatch):
    for terminal in ("completed", "failed", "canceled"):
        state = _install_scan_db(monkeypatch, terminal)
        assert db.claim_scan_job("sc1") is None, terminal
        assert state["status"] == terminal


def test_completed_cas_wins_from_processing(monkeypatch):
    # The tasks.py gate: capture only runs when this returns True.
    _install_scan_db(monkeypatch, "processing")
    assert db.update_job_status("sc1", "completed", tier="amber") is True


def test_completed_cas_lost_when_already_failed(monkeypatch):
    # Concurrent stale-recovery flipped the job to 'failed' (and released credits).
    # The completed write must LOSE so tasks.py skips capture (no billed-but-failed).
    state = _install_scan_db(monkeypatch, "failed")
    assert db.update_job_status("sc1", "completed", tier="amber") is False
    assert state["status"] == "failed"


def test_scan_status_update_is_best_effort(monkeypatch):
    def fail_update(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(tasks, "update_job_status", fail_update)

    assert tasks._best_effort_scan_status_update("job-1", "retrying", error="db unavailable") is False


def test_scan_status_update_reports_success(monkeypatch):
    calls = []

    def record_update(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(tasks, "update_job_status", record_update)

    assert tasks._best_effort_scan_status_update("job-1", "retrying", error="db unavailable") is True
    assert calls == [(("job-1", "retrying"), {"error": "db unavailable"})]
