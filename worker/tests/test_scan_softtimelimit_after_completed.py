"""SoftTimeLimitExceeded racing the completed-CAS write must not refund a free
scan or touch status/credits for a job that already finished (see the
`except SoftTimeLimitExceeded` handler in app.tasks.scan_document).

Before the fix, refund_free_scan has no status check -- it unconditionally
flips scan_jobs.free_scan_counted TRUE->FALSE for the job_id, so a soft-limit
fire that lands just after the completion write wrongly refunds a scan that
was actually billed and delivered. The fix checks get_scan_job(job_id).status
first and short-circuits release/refund/status-update/publish when it is
already 'completed'.

We force the SoftTimeLimitExceeded to fire at the very first line inside
scan_document's try block (claim_scan_job) so this test never needs to mock
the detect pipeline, R2 upload, or LLM calls.
"""
from celery.exceptions import SoftTimeLimitExceeded

import app.tasks as tasks


def _install_no_op_credit_calls(monkeypatch, calls):
    monkeypatch.setattr(tasks, "release_scan_credits", lambda job_id: calls.append(("release", job_id)))
    monkeypatch.setattr(tasks, "refund_free_scan", lambda job_id: calls.append(("refund", job_id)))
    monkeypatch.setattr(
        tasks,
        "_best_effort_scan_status_update",
        lambda job_id, status, **fields: calls.append(("status_update", job_id, status)),
    )
    monkeypatch.setattr(
        tasks,
        "publish_scan_progress",
        lambda job_id, **fields: calls.append(("publish", job_id, fields.get("status"))),
    )


def test_softtimelimit_after_completed_skips_release_refund_and_status(monkeypatch):
    calls = []

    def _raise_soft_limit(*args, **kwargs):
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(tasks, "claim_scan_job", _raise_soft_limit)
    monkeypatch.setattr(
        tasks,
        "get_scan_job",
        lambda job_id: {"status": "completed", "tier": "clean", "finding_count": 3},
    )
    _install_no_op_credit_calls(monkeypatch, calls)

    result = tasks.scan_document.run("job-1", "some text")

    assert result == {"status": "completed", "tier": "clean", "findings": 3}
    assert calls == []  # no release, refund, status update, or progress publish


def test_softtimelimit_before_completion_still_releases_and_refunds(monkeypatch):
    """Non-completed status (e.g. still 'processing') is the pre-existing,
    byte-identical behavior: release/refund/status-update/publish all still run.
    """
    calls = []

    def _raise_soft_limit(*args, **kwargs):
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(tasks, "claim_scan_job", _raise_soft_limit)
    monkeypatch.setattr(
        tasks,
        "get_scan_job",
        lambda job_id: {"status": "processing", "tier": None, "finding_count": None},
    )
    _install_no_op_credit_calls(monkeypatch, calls)

    result = tasks.scan_document.run("job-2", "some text")

    assert result == {"status": "failed", "error": "timeout"}
    call_kinds = [c[0] for c in calls]
    assert call_kinds == ["release", "refund", "status_update", "publish"]
