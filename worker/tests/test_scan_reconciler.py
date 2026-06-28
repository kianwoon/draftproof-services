"""Tests for the worker scan startup reconciler (recovery.reconcile_interrupted_scans).

No DB / Redis / R2: the DB layer, the R2 report fetch, and the Redis heartbeat are
monkeypatched so the recover / fail / skip decision logic is exercised in isolation.
Mirrors test_startup_reconciler.py (the rewrite reconciler) — findings H1 / M1 / M3 / L4.
"""

from datetime import datetime, timedelta, timezone

import app.recovery as recovery

UTC = timezone.utc
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


# ----------------------------- staleness -----------------------------------

def test_scan_stale_when_heartbeat_old(monkeypatch):
    monkeypatch.setattr(recovery, "_latest_scan_heartbeat", lambda _sid: NOW - timedelta(minutes=20))
    assert recovery._scan_processing_is_stale(NOW - timedelta(hours=2), "sc1", NOW) is True


def test_scan_not_stale_when_heartbeat_recent(monkeypatch):
    monkeypatch.setattr(recovery, "_latest_scan_heartbeat", lambda _sid: NOW - timedelta(seconds=30))
    assert recovery._scan_processing_is_stale(NOW - timedelta(hours=2), "sc1", NOW) is False


def test_scan_no_heartbeat_falls_back_to_started_at(monkeypatch):
    monkeypatch.setattr(recovery, "_latest_scan_heartbeat", lambda _sid: None)
    # recently started, no heartbeat -> not yet stale
    assert recovery._scan_processing_is_stale(NOW - timedelta(minutes=3), "sc1", NOW) is False
    # long since start, no heartbeat -> stale
    assert recovery._scan_processing_is_stale(NOW - timedelta(hours=2), "sc1", NOW) is True


def test_scan_not_started_is_never_stale(monkeypatch):
    # started_at is None (queued, not yet running) -> never stale, even behind a long
    # rewrite. This is the M1 fix: don't age a job that hasn't run by enqueue time.
    monkeypatch.setattr(recovery, "_latest_scan_heartbeat", lambda _sid: None)
    assert recovery._scan_processing_is_stale(None, "sc1", NOW) is False


# ----------------------------- reconcile flow ------------------------------

class _ScanDbSpy:
    def __init__(self, jobs):
        self.jobs = jobs
        self.updates = []
        self.captures = []
        self.releases = []
        self.refunds = []

    def list_processing_scan_jobs(self):
        return self.jobs

    def update_job_status(self, job_id, status, **kw):
        self.updates.append((job_id, status, kw))
        return True

    def capture_credits(self, user_id, job_id, word_count):
        self.captures.append((user_id, job_id, word_count))

    def release_scan_credits(self, job_id):
        self.releases.append(job_id)

    def refund_free_scan(self, job_id):
        self.refunds.append(job_id)


def _wire(monkeypatch, jobs, *, stale=True, report=None):
    spy = _ScanDbSpy(jobs)
    monkeypatch.setenv("DRAFTPROOF_STARTUP_RECONCILER_ENABLED", "true")
    monkeypatch.setattr(recovery, "db", spy)
    monkeypatch.setattr(recovery, "_scan_processing_is_stale", lambda *a, **k: stale)
    monkeypatch.setattr(recovery, "_fetch_scan_report_json", lambda _sid: report)
    return spy


def test_scan_reconcile_completes_when_report_in_r2(monkeypatch):
    # Worker uploaded the report to R2 but died before the DB write -> recover as
    # completed + capture (M3), never failed-for-free.
    spy = _wire(
        monkeypatch,
        [{"id": "sc1", "user_id": "u1", "started_at": NOW, "created_at": NOW}],
        stale=True,
        report={"ai_risk_badge": {"tier": "amber", "ai_likelihood_score": 32.0}},
    )
    stats = recovery.reconcile_interrupted_scans(now=NOW)
    assert stats == {"scanned": 1, "completed": 1, "failed": 0, "skipped": 0}
    assert spy.updates[0][1] == "completed"
    assert spy.captures == [("u1", "sc1", 0)]
    assert spy.releases == []


def test_scan_reconcile_fails_when_no_report(monkeypatch):
    spy = _wire(
        monkeypatch,
        [{"id": "sc1", "user_id": "u1", "started_at": NOW, "created_at": NOW}],
        stale=True,
        report=None,
    )
    stats = recovery.reconcile_interrupted_scans(now=NOW)
    assert stats == {"scanned": 1, "completed": 0, "failed": 1, "skipped": 0}
    assert spy.updates[0][1] == "failed"
    assert spy.releases == ["sc1"]
    assert spy.refunds == ["sc1"]
    assert spy.captures == []


def test_scan_reconcile_skips_live_job(monkeypatch):
    spy = _wire(
        monkeypatch,
        [{"id": "sc1", "user_id": "u1", "started_at": NOW, "created_at": NOW}],
        stale=False,
    )
    stats = recovery.reconcile_interrupted_scans(now=NOW)
    assert stats == {"scanned": 1, "completed": 0, "failed": 0, "skipped": 1}
    assert spy.updates == []


def test_scan_reconcile_disabled_is_noop(monkeypatch):
    spy = _ScanDbSpy([{"id": "sc1", "user_id": "u1", "started_at": NOW}])
    monkeypatch.setenv("DRAFTPROOF_STARTUP_RECONCILER_ENABLED", "0")
    monkeypatch.setattr(recovery, "db", spy)
    stats = recovery.reconcile_interrupted_scans(now=NOW)
    assert stats == {"scanned": 0, "completed": 0, "failed": 0, "skipped": 0}
    assert spy.updates == []
