import sys
from types import SimpleNamespace

import pytest

from app.services import scan_service


def words(count: int) -> str:
    return " ".join(f"word{i}" for i in range(count))


class FakeSession:
    def __init__(self):
        self.added = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, item):
        self.added.append(item)

    async def execute(self, *_args, **_kwargs):
        raise AssertionError("free scans should not query credit accounts")

    async def commit(self):
        self.committed = True


class FakeScanTask:
    def __init__(self):
        self.calls = []

    def delay(self, *args):
        self.calls.append(args)


def test_scan_cost_is_free_through_300_words():
    assert scan_service._scan_cost(1) == 0
    assert scan_service._scan_cost(300) == 0


def test_scan_cost_charges_from_301_words():
    assert scan_service._scan_cost(301) == 1
    assert scan_service._scan_cost(1000) == 1
    assert scan_service._scan_cost(1001) == 2


@pytest.mark.asyncio
async def test_create_free_scan_does_not_require_credit_account(monkeypatch):
    fake_session = FakeSession()
    fake_task = FakeScanTask()

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules,
        "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    result = await scan_service.create_scan(
        "paste",
        user_id="00000000-0000-0000-0000-000000000001",
        text=words(300),
    )

    assert result["status"] == "pending"
    assert fake_session.committed is True
    assert len(fake_session.added) == 1
    assert fake_session.added[0].word_count == 300
    assert len(fake_task.calls) == 1
