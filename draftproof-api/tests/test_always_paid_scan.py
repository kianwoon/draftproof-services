"""always_paid=True forces the paid rate and skips the free-scan quota.

This is the billing contract for extension scans: they never consume the free
allowance and always reserve >=1 credit, reusing the existing reserve→capture
machinery. Fake-session pattern mirrors test_scan_service.py.
"""

import sys
from types import SimpleNamespace

import pytest

from app.services import scan_service


def words(count: int) -> str:
    return " ".join(f"word{i}" for i in range(count))


class FakeScanTask:
    def __init__(self):
        self.calls = []

    def delay(self, *args):
        self.calls.append(args)


class AlwaysPaidSession:
    """Fails loudly if the free-quota UPDATE is ever issued; otherwise returns
    the credit account on the paid-path SELECT."""

    def __init__(self, *, account):
        self.account = account
        self.added = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, item):
        self.added.append(item)

    async def execute(self, statement, params=None):
        assert "free_scans_used" not in str(statement), \
            "always_paid must skip the free-quota path entirely"
        return SimpleNamespace(scalar_one_or_none=lambda: self.account)

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_always_paid_short_scan_charges_and_skips_free_quota(monkeypatch):
    acct = SimpleNamespace(id="acct-1", balance_tokens=50, reserved_tokens=0)
    fake_session = AlwaysPaidSession(account=acct)
    fake_task = FakeScanTask()

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules,
        "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    result = await scan_service.create_scan(
        "ext-paste",
        user_id="00000000-0000-0000-0000-000000000001",
        text=words(50),           # well under the 800-word free threshold
        always_paid=True,
    )

    assert result["status"] == "pending"
    job = next(a for a in fake_session.added if isinstance(a, scan_service.ScanJob))
    assert job.free_scan_counted is False          # never a free scan
    assert acct.reserved_tokens == 1               # 1 credit for a <=1000 word doc
    reservations = [a for a in fake_session.added if isinstance(a, scan_service.CreditReservation)]
    assert len(reservations) == 1 and reservations[0].tokens_reserved == 1
    assert len(fake_task.calls) == 1               # enqueued the same Celery task


@pytest.mark.asyncio
async def test_title_override_sets_document_title(monkeypatch):
    """An explicit title (the Word file name) becomes the report's document_title."""
    acct = SimpleNamespace(id="acct-1", balance_tokens=50, reserved_tokens=0)
    fake_session = AlwaysPaidSession(account=acct)
    fake_task = FakeScanTask()
    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules, "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    await scan_service.create_scan(
        "ext-x",
        user_id="00000000-0000-0000-0000-000000000001",
        text=words(50),
        always_paid=True,
        title="Assignment 3.docx",
    )

    job = next(a for a in fake_session.added if isinstance(a, scan_service.ScanJob))
    assert job.document_title == "Assignment 3.docx"


@pytest.mark.asyncio
async def test_long_title_is_truncated(monkeypatch):
    acct = SimpleNamespace(id="acct-1", balance_tokens=50, reserved_tokens=0)
    fake_session = AlwaysPaidSession(account=acct)
    fake_task = FakeScanTask()
    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules, "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    await scan_service.create_scan(
        "ext-x",
        user_id="00000000-0000-0000-0000-000000000001",
        text=words(50),
        always_paid=True,
        title="X" * 200,
    )

    job = next(a for a in fake_session.added if isinstance(a, scan_service.ScanJob))
    assert len(job.document_title) <= scan_service.DOCUMENT_TITLE_MAX_CHARS
