"""regenerate_rewrite_report_assets blocks a thread for up to 45s (task.get(timeout=45)
inside asyncio.to_thread) on the process-wide default executor shared by every other
asyncio.to_thread call in the API. Unbounded concurrent regenerate requests could
starve that shared pool (self-DoS). A per-process counter
(_MAX_CONCURRENT_REWRITE_REGENERATIONS) bounds in-flight regenerations and raises
RewriteRegenerationBusyError -- mapped to HTTP 429 by the router -- once saturated.

These tests are white-box: they set/inspect the module-level counter directly rather
than simulating real thread concurrency, which would need a live Celery broker/worker.
"""
from types import SimpleNamespace

import pytest

from app.services import rewrite_service


def _completed_job_info():
    return {"status": "completed", "scan_id": "scan-1", "id": "rewrite-1"}


class _FakeAsyncResult:
    def __init__(self, value):
        self._value = value

    def get(self, timeout=None):
        return self._value


class _FakeRegenTask:
    def __init__(self, value):
        self._value = value
        self.delay_calls = 0

    def delay(self, *args, **kwargs):
        self.delay_calls += 1
        return _FakeAsyncResult(self._value)


@pytest.fixture(autouse=True)
def _reset_counter():
    """Guard against cross-test pollution of the module-level counter."""
    original = rewrite_service._active_rewrite_regenerations
    yield
    rewrite_service._active_rewrite_regenerations = original


async def test_regenerate_rejects_when_saturated(monkeypatch):
    async def _fake_get_rewrite(*a, **k):
        return _completed_job_info()

    monkeypatch.setattr(rewrite_service, "get_rewrite", _fake_get_rewrite)
    rewrite_service._active_rewrite_regenerations = rewrite_service._MAX_CONCURRENT_REWRITE_REGENERATIONS

    with pytest.raises(rewrite_service.RewriteRegenerationBusyError):
        await rewrite_service.regenerate_rewrite_report_assets("rewrite-1", "user-1")


async def test_regenerate_succeeds_and_releases_slot(monkeypatch):
    async def _fake_get_rewrite(*a, **k):
        return _completed_job_info()

    monkeypatch.setattr(rewrite_service, "get_rewrite", _fake_get_rewrite)
    fake_task = _FakeRegenTask({"status": "regenerated"})
    monkeypatch.setattr("app.services.celery_client.regenerate_rewrite_report_assets", fake_task)

    rewrite_service._active_rewrite_regenerations = 0
    result = await rewrite_service.regenerate_rewrite_report_assets("rewrite-1", "user-1")

    assert result == {"status": "regenerated"}
    assert fake_task.delay_calls == 1
    # Slot released after completion -- next call is not spuriously rejected.
    assert rewrite_service._active_rewrite_regenerations == 0


async def test_regenerate_releases_slot_even_when_task_raises(monkeypatch):
    async def _fake_get_rewrite(*a, **k):
        return _completed_job_info()

    class _RaisingTask:
        def delay(self, *a, **k):
            raise ConnectionError("broker unreachable")

    monkeypatch.setattr(rewrite_service, "get_rewrite", _fake_get_rewrite)
    monkeypatch.setattr("app.services.celery_client.regenerate_rewrite_report_assets", _RaisingTask())

    rewrite_service._active_rewrite_regenerations = 0
    with pytest.raises(ConnectionError):
        await rewrite_service.regenerate_rewrite_report_assets("rewrite-1", "user-1")

    # finally-block must decrement even on an unhandled exception, or a single
    # broker hiccup would permanently eat a concurrency slot.
    assert rewrite_service._active_rewrite_regenerations == 0
