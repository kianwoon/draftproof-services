"""Tests for the worker startup reconciler (recovery.py).

No DB / Redis / R2: the DB layer, the R2 report fetch, and the Redis heartbeat
are monkeypatched so the recover / fail / skip decision logic is exercised in
isolation.
"""

from datetime import datetime, timedelta, timezone

import app.recovery as recovery

UTC = timezone.utc
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


# ----------------------------- pure helpers --------------------------------

def test_delivered_content_true_for_mitigated_changed_text():
    assert recovery._rewrite_report_has_delivered_content(
        {"status": "ai_mitigated", "original_text": "old draft", "final_text": "new improved draft"}
    ) is True


def test_delivered_content_false_when_text_unchanged():
    assert recovery._rewrite_report_has_delivered_content(
        {"status": "ai_mitigated", "original_text": "same text", "final_text": "same text"}
    ) is False


def test_delivered_content_false_when_final_text_empty():
    assert recovery._rewrite_report_has_delivered_content(
        {"status": "ai_mitigated", "original_text": "old", "final_text": ""}
    ) is False


def test_delivered_content_reads_outcome_from_summary():
    assert recovery._rewrite_report_has_delivered_content(
        {"final_text": "new", "summary": {"outcome": "rewrite_candidate_generated_needs_external_review"}}
    ) is True


def test_delivered_content_false_for_non_delivered_status():
    assert recovery._rewrite_report_has_delivered_content(
        {"status": "mitigation_failed_no_safe_candidate", "final_text": "x", "original_text": "y"}
    ) is False


def test_delivered_content_false_for_none():
    assert recovery._rewrite_report_has_delivered_content(None) is False


# ----------------------------- staleness -----------------------------------

def test_stale_when_heartbeat_old(monkeypatch):
    monkeypatch.setattr(recovery, "_latest_heartbeat", lambda _rid: NOW - timedelta(minutes=10))
    assert recovery._processing_is_stale(NOW - timedelta(hours=2), "rw1", NOW) is True


def test_not_stale_when_heartbeat_recent(monkeypatch):
    monkeypatch.setattr(recovery, "_latest_heartbeat", lambda _rid: NOW - timedelta(seconds=30))
    assert recovery._processing_is_stale(NOW - timedelta(hours=2), "rw1", NOW) is False


def test_no_heartbeat_falls_back_to_wallclock(monkeypatch):
    monkeypatch.setattr(recovery, "_latest_heartbeat", lambda _rid: None)
    # young job, no heartbeat -> not yet stale
    assert recovery._processing_is_stale(NOW - timedelta(minutes=10), "rw1", NOW) is False
    # old job, no heartbeat -> stale
    assert recovery._processing_is_stale(NOW - timedelta(hours=2), "rw1", NOW) is True


def test_naive_created_at_is_treated_as_utc(monkeypatch):
    monkeypatch.setattr(recovery, "_latest_heartbeat", lambda _rid: None)
    naive_old = (NOW - timedelta(hours=2)).replace(tzinfo=None)
    assert recovery._processing_is_stale(naive_old, "rw1", NOW) is True


# ----------------------------- reconcile flow ------------------------------

class _DbSpy:
    def __init__(self, jobs):
        self.jobs = jobs
        self.updates = []
        self.captures = []
        self.releases = []

    def list_processing_rewrite_jobs(self):
        return self.jobs

    def update_rewrite_status(self, job_id, status, **kw):
        self.updates.append((job_id, status, kw))
        return True

    def capture_rewrite_credits(self, user_id, job_id):
        self.captures.append((user_id, job_id))

    def release_rewrite_credits(self, job_id):
        self.releases.append(job_id)


def _wire(monkeypatch, jobs, *, stale=True, report=None):
    spy = _DbSpy(jobs)
    monkeypatch.setenv("DRAFTPROOF_STARTUP_RECONCILER_ENABLED", "true")
    monkeypatch.setattr(recovery, "db", spy)
    monkeypatch.setattr(recovery, "_processing_is_stale", lambda *a, **k: stale)
    monkeypatch.setattr(recovery, "_fetch_rewrite_report_json", lambda _sid: report)
    return spy


def test_reconcile_completes_job_with_delivered_checkpoint(monkeypatch):
    spy = _wire(
        monkeypatch,
        [{"id": "rw1", "scan_id": "sc1", "user_id": "u1", "created_at": NOW}],
        stale=True,
        report={"status": "ai_mitigated", "original_text": "old", "final_text": "new better"},
    )
    stats = recovery.reconcile_interrupted_rewrites(now=NOW)
    assert stats == {"scanned": 1, "completed": 1, "failed": 0, "skipped": 0}
    assert spy.updates[0][1] == "completed"
    assert spy.captures == [("u1", "rw1")]
    assert spy.releases == []


def test_reconcile_fails_job_without_delivered_checkpoint(monkeypatch):
    spy = _wire(
        monkeypatch,
        [{"id": "rw1", "scan_id": "sc1", "user_id": "u1", "created_at": NOW}],
        stale=True,
        report=None,
    )
    stats = recovery.reconcile_interrupted_rewrites(now=NOW)
    assert stats == {"scanned": 1, "completed": 0, "failed": 1, "skipped": 0}
    assert spy.updates[0][1] == "failed"
    assert spy.releases == ["rw1"]
    assert spy.captures == []


def test_reconcile_skips_live_job(monkeypatch):
    spy = _wire(
        monkeypatch,
        [{"id": "rw1", "scan_id": "sc1", "user_id": "u1", "created_at": NOW}],
        stale=False,
    )
    stats = recovery.reconcile_interrupted_rewrites(now=NOW)
    assert stats == {"scanned": 1, "completed": 0, "failed": 0, "skipped": 1}
    assert spy.updates == []
    assert spy.captures == []
    assert spy.releases == []


def test_reconcile_disabled_is_noop(monkeypatch):
    spy = _DbSpy([{"id": "rw1", "scan_id": "sc1", "user_id": "u1", "created_at": NOW}])
    monkeypatch.setenv("DRAFTPROOF_STARTUP_RECONCILER_ENABLED", "0")
    monkeypatch.setattr(recovery, "db", spy)
    stats = recovery.reconcile_interrupted_rewrites(now=NOW)
    assert stats == {"scanned": 0, "completed": 0, "failed": 0, "skipped": 0}
    assert spy.updates == []
