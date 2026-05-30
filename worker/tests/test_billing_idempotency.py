"""Idempotency / race-safety tests for worker-side credit billing.

These guard the fix that collapsed the capture/release "check-then-act" into an
atomic compare-and-swap (UPDATE ... WHERE status='active' RETURNING). The risk
they prevent: a redelivered task, or a concurrent API-side stale-recovery
capture, double-charging the user or double-releasing reserved tokens.

A tiny in-memory fake stands in for Postgres so the test needs no database. The
fake models exactly the one property that matters: the conditional UPDATE only
"returns" a row when the reservation is still 'active'.
"""

from contextlib import contextmanager

import app.db as db


class _FakeCursor:
    def __init__(self, state, log):
        self.state = state
        self.log = log
        self._last = None

    def execute(self, sql, params=()):
        s = " ".join(sql.split())  # normalise whitespace for matching
        self.log.append(s)
        res = self.state["reservation"]

        if s.startswith("UPDATE credit_reservations SET status = 'captured'"):
            # Atomic CAS: flip only if still active, RETURNING the row.
            if res["status"] == "active":
                res["status"] = "captured"
                self._last = {
                    "credit_account_id": res["credit_account_id"],
                    "tokens_reserved": res["tokens_reserved"],
                }
            else:
                self._last = None
        elif s.startswith("UPDATE credit_reservations SET status = 'released'"):
            if res["status"] == "active":
                res["status"] = "released"
                self._last = {
                    "credit_account_id": res["credit_account_id"],
                    "tokens_reserved": res["tokens_reserved"],
                }
            else:
                self._last = None
        elif s.startswith("UPDATE credit_accounts SET balance_tokens"):
            tokens, reserved, _acct = params
            self.state["balance"] -= tokens
            self.state["reserved"] -= reserved
            self._last = None
        elif s.startswith("UPDATE credit_accounts SET reserved_tokens"):
            tokens, _acct = params
            self.state["reserved"] -= tokens
            self._last = None
        elif s.startswith("INSERT INTO usage_events"):
            self.state["usage_events"].append(params)
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


def _install_fake_db(monkeypatch, *, balance=100, reserved=5, tokens=5):
    state = {
        "reservation": {
            "status": "active",
            "credit_account_id": "acct-1",
            "tokens_reserved": tokens,
        },
        "balance": balance,
        "reserved": reserved,
        "usage_events": [],
    }
    log: list[str] = []

    @contextmanager
    def _fake_get_conn():
        yield _FakeConn(state, log)

    monkeypatch.setattr(db, "get_conn", _fake_get_conn)
    return state, log


def test_capture_rewrite_credits_is_atomic_conditional_update(monkeypatch):
    state, log = _install_fake_db(monkeypatch)
    db.capture_rewrite_credits("user-1", "job-1")
    # The very first statement must be the conditional CAS, not an unlocked SELECT.
    assert log[0].startswith("UPDATE credit_reservations SET status = 'captured'")
    assert "WHERE job_id = %s AND status = 'active'" in log[0]
    assert "RETURNING" in log[0]
    assert not any(l.startswith("SELECT") for l in log)


def test_capture_rewrite_credits_no_double_charge_on_redelivery(monkeypatch):
    state, log = _install_fake_db(monkeypatch, balance=100, reserved=5, tokens=5)

    db.capture_rewrite_credits("user-1", "job-1")  # first (real) run
    assert state["reservation"]["status"] == "captured"
    assert state["balance"] == 95
    assert state["reserved"] == 0
    assert len(state["usage_events"]) == 1

    db.capture_rewrite_credits("user-1", "job-1")  # redelivered duplicate
    # No second debit, no second usage_event.
    assert state["balance"] == 95
    assert state["reserved"] == 0
    assert len(state["usage_events"]) == 1


def test_capture_scan_credits_no_double_charge_on_redelivery(monkeypatch):
    state, log = _install_fake_db(monkeypatch, balance=40, reserved=1, tokens=1)

    db.capture_credits("user-1", "job-1", word_count=1200)
    assert state["balance"] == 39
    assert len(state["usage_events"]) == 1
    assert state["usage_events"][0] == ("user-1", "acct-1", 1, "job-1", 1200)

    db.capture_credits("user-1", "job-1", word_count=1200)
    assert state["balance"] == 39
    assert len(state["usage_events"]) == 1


def test_capture_then_release_cannot_both_apply(monkeypatch):
    """Saga race: a job that completes (capture) while a cancel/stale-recovery
    tries to release. Whichever flips 'active' first wins; the other no-ops."""
    state, log = _install_fake_db(monkeypatch, balance=100, reserved=5, tokens=5)

    db.capture_rewrite_credits("user-1", "job-1")  # capture wins the race
    assert state["balance"] == 95 and state["reserved"] == 0

    db.release_rewrite_credits("job-1")  # must NOT refund reserved tokens
    assert state["balance"] == 95
    assert state["reserved"] == 0  # unchanged — no double-decrement
    assert state["reservation"]["status"] == "captured"


def test_release_then_capture_cannot_both_apply(monkeypatch):
    """Reverse order: release wins, a late capture must not charge."""
    state, log = _install_fake_db(monkeypatch, balance=100, reserved=5, tokens=5)

    db.release_rewrite_credits("job-1")  # release wins
    assert state["reservation"]["status"] == "released"
    assert state["reserved"] == 0
    assert state["balance"] == 100  # balance untouched on release

    db.capture_rewrite_credits("user-1", "job-1")  # late capture must no-op
    assert state["balance"] == 100  # NOT charged
    assert len(state["usage_events"]) == 0


def test_release_is_idempotent(monkeypatch):
    state, log = _install_fake_db(monkeypatch, balance=100, reserved=5, tokens=5)

    db.release_rewrite_credits("job-1")
    db.release_rewrite_credits("job-1")  # duplicate
    assert state["reserved"] == 0  # decremented once, not -5
