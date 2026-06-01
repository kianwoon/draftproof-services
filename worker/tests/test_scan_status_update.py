import app.tasks as tasks


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
