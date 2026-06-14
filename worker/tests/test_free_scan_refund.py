"""Idempotency tests for the durable free-scan counter refund.

Free (0-cost) scans have no credit reservation, so refund_free_scan is the
free-path analog of release_scan_credits: on failure it decrements the durable
users.free_scans_used counter. The CAS on scan_jobs.free_scan_counted must make
that decrement fire at most once, even if a redelivered task and the API-side
stale-recovery both refund the same job.

A tiny in-memory fake stands in for Postgres — it models exactly the property
that matters: the conditional UPDATE returns a row only while the scan is still
counted.
"""

from contextlib import contextmanager

import app.db as db


class _FakeCursor:
    def __init__(self, state, log):
        self.state = state
        self.log = log
        self._last = None

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        self.log.append(s)
        job = self.state["scan_job"]

        if s.startswith("UPDATE scan_jobs SET free_scan_counted = FALSE"):
            # Atomic CAS: flip only if still counted, RETURNING the user_id.
            if job["free_scan_counted"]:
                job["free_scan_counted"] = False
                self._last = {"user_id": job["user_id"]}
            else:
                self._last = None
        elif s.startswith("UPDATE users SET free_scans_used = GREATEST"):
            (_uid,) = params
            self.state["free_scans_used"] = max(0, self.state["free_scans_used"] - 1)
            self._last = None
        else:  # pragma: no cover - defensive
            self._last = None

    def fetchone(self):
        return self._last


class _FakeConn:
    def __init__(self, state, log):
        self.state = state
        self.log = log

    def cursor(self):
        return _FakeCursor(self.state, self.log)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _install_fake_db(monkeypatch, *, free_scans_used=5, counted=True, user_id="user-1"):
    state = {
        "scan_job": {"free_scan_counted": counted, "user_id": user_id},
        "free_scans_used": free_scans_used,
    }
    log: list[str] = []

    @contextmanager
    def _fake_get_conn():
        yield _FakeConn(state, log)

    monkeypatch.setattr(db, "get_conn", _fake_get_conn)
    return state, log


def test_refund_is_atomic_conditional_update(monkeypatch):
    _state, log = _install_fake_db(monkeypatch)
    db.refund_free_scan("job-1")
    assert log[0].startswith("UPDATE scan_jobs SET free_scan_counted = FALSE")
    assert "WHERE id = %s AND free_scan_counted = TRUE" in log[0]
    assert "RETURNING user_id" in log[0]
    assert not any(l.startswith("SELECT") for l in log)


def test_refund_decrements_counter_once(monkeypatch):
    state, _log = _install_fake_db(monkeypatch, free_scans_used=5, counted=True)
    db.refund_free_scan("job-1")
    assert state["free_scans_used"] == 4
    assert state["scan_job"]["free_scan_counted"] is False


def test_refund_no_double_decrement_on_redelivery(monkeypatch):
    state, _log = _install_fake_db(monkeypatch, free_scans_used=5, counted=True)
    db.refund_free_scan("job-1")  # real run
    db.refund_free_scan("job-1")  # redelivered task / concurrent recovery
    assert state["free_scans_used"] == 4  # decremented exactly once


def test_refund_noop_for_paid_scan_never_counted(monkeypatch):
    state, log = _install_fake_db(monkeypatch, free_scans_used=0, counted=False)
    db.refund_free_scan("job-1")
    assert state["free_scans_used"] == 0
    assert not any("UPDATE users" in l for l in log)


def test_refund_noop_when_user_id_missing(monkeypatch):
    state, log = _install_fake_db(monkeypatch, free_scans_used=3, counted=True, user_id=None)
    db.refund_free_scan("job-1")
    assert state["free_scans_used"] == 3
    assert not any("UPDATE users" in l for l in log)
