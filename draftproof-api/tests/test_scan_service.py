import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.services import scan_service


def words(count: int) -> str:
    return " ".join(f"word{i}" for i in range(count))


class FakeScanTask:
    def __init__(self):
        self.calls = []

    def delay(self, *args, **kwargs):
        # Mirrors the real Celery .delay(*args, **kwargs) signature (Task 7's
        # test_defence.py precedent notwithstanding, scan_document is dispatched
        # by celery_client.py's pure name-string signature() call -- no static
        # arg-count check exists anywhere except this fake, so it must accept
        # whatever scan_service.py actually passes, e.g. ai_policy as a kwarg).
        self.calls.append((args, kwargs))


class FakeResult:
    def __init__(self, *, rows=None, scalar_value=None):
        self.rows = rows or []
        self.scalar_value = scalar_value

    def scalar(self):
        return self.scalar_value

    def scalars(self):
        return SimpleNamespace(all=lambda: self.rows)


class FakeListSession:
    def __init__(self, results):
        self.results = list(results)
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, *_args, **_kwargs):
        return self.results.pop(0)

    async def commit(self):
        self.committed = True


class RefundCaptureSession:
    """Captures the SQL that _refund_free_scan issues. The first execute() is the
    CAS flip (returns a user_id row only if the scan was still counted); the
    second is the decrement, recorded so tests can assert it fired exactly once."""

    def __init__(self, *, was_counted: bool):
        self.was_counted = was_counted
        self.statements = []

    async def execute(self, statement, params=None):
        self.statements.append((str(statement), params))
        if "UPDATE scan_jobs" in str(statement):
            row = ("11111111-1111-1111-1111-111111111111",) if self.was_counted else None
            return SimpleNamespace(first=lambda: row)
        return SimpleNamespace(first=lambda: None)

    @property
    def decrement_calls(self):
        return [s for s, _ in self.statements if "free_scans_used = GREATEST" in s]


class PaidScanSession:
    """Models the paid billing path: the only query create_scan issues for a user
    scan is the credit-account ``SELECT ... FOR UPDATE``. `account` is the fake
    acct (or None to model a user with no credit account)."""

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
        return SimpleNamespace(scalar_one_or_none=lambda: self.account)

    async def commit(self):
        self.committed = True


def _stream_id(at: datetime) -> str:
    return f"{int(at.timestamp() * 1000)}-0"


def test_paid_scan_cost_is_one_credit_per_started_1000_words():
    assert scan_service._paid_scan_cost(1) == 1
    assert scan_service._paid_scan_cost(800) == 1
    assert scan_service._paid_scan_cost(1000) == 1
    assert scan_service._paid_scan_cost(1001) == 2
    assert scan_service._paid_scan_cost(2500) == 3


def test_rewrite_cost_charges_five_tokens_per_started_1000_words():
    assert scan_service._rewrite_cost(1) == 5
    assert scan_service._rewrite_cost(1000) == 5
    assert scan_service._rewrite_cost(1001) == 10


def test_scan_report_metadata_compacts_and_bounds_text():
    text = "\n\n  This is the first useful sentence.   This continues with extra detail.\nSecond line."

    metadata = scan_service.build_scan_report_metadata(text)

    assert metadata["document_title"] == "This is the first useful sentence."
    assert metadata["content_preview"] == "This is the first useful sentence. This continues with extra detail. Second line."


def test_scan_report_metadata_ignores_punctuation_only_title():
    metadata = scan_service.build_scan_report_metadata("\n ... !!! \n\n")

    assert metadata["document_title"] is None
    assert metadata["content_preview"] == "... !!!"


def test_scan_report_metadata_truncates_long_values():
    text = " ".join(["Alpha"] * 80)

    metadata = scan_service.build_scan_report_metadata(text)

    assert len(metadata["document_title"]) <= scan_service.DOCUMENT_TITLE_MAX_CHARS
    assert len(metadata["content_preview"]) <= scan_service.CONTENT_PREVIEW_MAX_CHARS
    assert metadata["document_title"].endswith("…")
    assert metadata["content_preview"].endswith("…")


@pytest.mark.asyncio
async def test_short_scan_reserves_one_credit(monkeypatch):
    """A <=1,000 word scan costs 1 credit: it reserves the credit, is NOT flagged
    free_scan_counted (no per-scan free path exists), and is enqueued."""
    acct = SimpleNamespace(id="acct-1", balance_tokens=5, reserved_tokens=0)
    fake_session = PaidScanSession(account=acct)
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
        text=words(518),
    )

    assert result["status"] == "pending"
    job = next(a for a in fake_session.added if isinstance(a, scan_service.ScanJob))
    assert job.word_count == 518
    assert job.free_scan_counted is False
    assert job.document_title.startswith("word0 word1")
    assert acct.reserved_tokens == 1
    reservations = [a for a in fake_session.added if isinstance(a, scan_service.CreditReservation)]
    assert len(reservations) == 1 and reservations[0].tokens_reserved == 1
    assert len(fake_task.calls) == 1


@pytest.mark.asyncio
async def test_long_scan_reserves_one_credit_per_started_1000_words(monkeypatch):
    """A 2,500 word scan reserves 3 credits (ceil(2500/1000))."""
    acct = SimpleNamespace(id="acct-1", balance_tokens=10, reserved_tokens=0)
    fake_session = PaidScanSession(account=acct)
    fake_task = FakeScanTask()

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules,
        "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    await scan_service.create_scan(
        "paste",
        user_id="00000000-0000-0000-0000-000000000001",
        text=words(2500),
    )

    assert acct.reserved_tokens == 3
    reservations = [a for a in fake_session.added if isinstance(a, scan_service.CreditReservation)]
    assert len(reservations) == 1 and reservations[0].tokens_reserved == 3


# ── ai_policy capture (Phase 1 batch 2, docs/plans/policy_risk_external_review_response.md) ──

@pytest.mark.asyncio
async def test_ai_policy_is_stored_on_the_job_and_sent_to_the_worker(monkeypatch):
    acct = SimpleNamespace(id="acct-1", balance_tokens=5, reserved_tokens=0)
    fake_session = PaidScanSession(account=acct)
    fake_task = FakeScanTask()

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules,
        "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    await scan_service.create_scan(
        "paste",
        user_id="00000000-0000-0000-0000-000000000001",
        text=words(518),
        ai_policy="prohibited",
    )

    job = next(a for a in fake_session.added if isinstance(a, scan_service.ScanJob))
    assert job.ai_policy == "prohibited"
    (args, kwargs) = fake_task.calls[0]
    assert kwargs.get("ai_policy") == "prohibited"


@pytest.mark.asyncio
async def test_ai_policy_absent_normalizes_to_unknown_not_null(monkeypatch):
    # Explicit ai_policy=None on the ScanJob constructor would insert a real SQL
    # NULL, bypassing the column's DEFAULT 'unknown' -- create_scan must
    # normalize before constructing ScanJob, not rely on the DB default.
    acct = SimpleNamespace(id="acct-1", balance_tokens=5, reserved_tokens=0)
    fake_session = PaidScanSession(account=acct)
    fake_task = FakeScanTask()

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules,
        "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    await scan_service.create_scan(
        "paste",
        user_id="00000000-0000-0000-0000-000000000001",
        text=words(518),
    )

    job = next(a for a in fake_session.added if isinstance(a, scan_service.ScanJob))
    assert job.ai_policy == "unknown"
    (args, kwargs) = fake_task.calls[0]
    assert kwargs.get("ai_policy") == "unknown"


@pytest.mark.asyncio
async def test_scan_blocked_when_no_credit_account(monkeypatch):
    """No credit account: the scan is blocked with the 'buy credits' error and
    nothing is enqueued."""
    fake_session = PaidScanSession(account=None)
    fake_task = FakeScanTask()

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules,
        "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    with pytest.raises(ValueError, match="No credit account"):
        await scan_service.create_scan(
            "paste",
            user_id="00000000-0000-0000-0000-000000000001",
            text=words(518),
        )

    assert len(fake_task.calls) == 0


@pytest.mark.asyncio
async def test_scan_blocked_when_insufficient_credits(monkeypatch):
    """Account exists but the available balance is below the cost: blocked with
    'Insufficient tokens', nothing enqueued."""
    acct = SimpleNamespace(id="acct-1", balance_tokens=0, reserved_tokens=0)
    fake_session = PaidScanSession(account=acct)
    fake_task = FakeScanTask()

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules,
        "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    with pytest.raises(ValueError, match="Insufficient"):
        await scan_service.create_scan(
            "paste",
            user_id="00000000-0000-0000-0000-000000000001",
            text=words(518),
        )

    assert len(fake_task.calls) == 0


@pytest.mark.asyncio
async def test_refund_free_scan_decrements_once_when_counted():
    """A failed free scan that was still counted refunds the durable counter."""
    session = RefundCaptureSession(was_counted=True)
    job = SimpleNamespace(
        id="22222222-2222-2222-2222-222222222222",
        user_id="11111111-1111-1111-1111-111111111111",
    )

    await scan_service._refund_free_scan(session, job)

    assert len(session.decrement_calls) == 1


@pytest.mark.asyncio
async def test_refund_free_scan_is_noop_when_already_refunded():
    """Idempotent: if the CAS flip finds nothing to flip (already refunded or
    paid scan), no decrement is issued."""
    session = RefundCaptureSession(was_counted=False)
    job = SimpleNamespace(
        id="22222222-2222-2222-2222-222222222222",
        user_id="11111111-1111-1111-1111-111111111111",
    )

    await scan_service._refund_free_scan(session, job)

    assert session.decrement_calls == []


@pytest.mark.asyncio
async def test_refund_free_scan_skips_anonymous_scan():
    """No user means no durable counter to touch — nothing is queried."""
    session = RefundCaptureSession(was_counted=True)
    job = SimpleNamespace(id="22222222-2222-2222-2222-222222222222", user_id=None)

    await scan_service._refund_free_scan(session, job)

    assert session.statements == []


@pytest.mark.asyncio
async def test_list_scans_returns_report_metadata(monkeypatch):
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000005",
        status="completed",
        document_title="Essay introduction",
        content_preview="This essay introduces the argument.",
        tier="low",
        ai_score=12,
        writing_score=88,
        finding_count=2,
        progress_percent=100,
        progress_message="Complete",
        word_count=640,
        created_at=datetime(2026, 5, 11, 1, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 11, 1, 2, tzinfo=timezone.utc),
    )
    fake_session = FakeListSession([
        FakeResult(rows=[]),
        FakeResult(scalar_value=1),
        FakeResult(rows=[job]),
    ])

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)

    result = await scan_service.list_scans(
        "00000000-0000-0000-0000-000000000001",
        page=1,
        per_page=10,
    )

    assert result["total"] == 1
    assert result["items"][0]["document_title"] == "Essay introduction"
    assert result["items"][0]["content_preview"] == "This essay introduces the argument."


@pytest.mark.asyncio
async def test_processing_scan_is_stale_when_heartbeat_stops(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        status="processing",
        created_at=now - timedelta(minutes=8),
        started_at=now - timedelta(minutes=8),
    )

    async def fake_latest(_scan_id):
        # Heartbeat older than the 8-min heartbeat-stale threshold -> worker reaped.
        return (_stream_id(now - timedelta(minutes=10)), {"status": "processing"})

    monkeypatch.setattr(
        scan_service.progress_stream,
        "read_latest_scan_progress",
        fake_latest,
    )

    assert await scan_service._processing_scan_is_stale(job, now=now) is True


@pytest.mark.asyncio
async def test_processing_scan_is_not_stale_with_recent_heartbeat(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000002",
        status="processing",
        created_at=now - timedelta(minutes=8),
        started_at=now - timedelta(minutes=8),
    )

    async def fake_latest(_scan_id):
        return (_stream_id(now - timedelta(minutes=1)), {"status": "processing"})

    monkeypatch.setattr(
        scan_service.progress_stream,
        "read_latest_scan_progress",
        fake_latest,
    )

    assert await scan_service._processing_scan_is_stale(job, now=now) is False


@pytest.mark.asyncio
async def test_processing_scan_falls_back_to_hard_stale_threshold(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000003",
        status="processing",
        created_at=now - scan_service._STALE_THRESHOLD - timedelta(seconds=1),
        started_at=now - scan_service._STALE_THRESHOLD - timedelta(seconds=1),
    )

    async def fake_latest(_scan_id):
        return None

    monkeypatch.setattr(
        scan_service.progress_stream,
        "read_latest_scan_progress",
        fake_latest,
    )

    assert await scan_service._processing_scan_is_stale(job, now=now) is True


@pytest.mark.asyncio
async def test_processing_scan_without_heartbeat_uses_conservative_fallback(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000004",
        status="processing",
        created_at=now - timedelta(minutes=8),
        started_at=now - timedelta(minutes=8),
    )

    async def fake_latest(_scan_id):
        return None

    monkeypatch.setattr(
        scan_service.progress_stream,
        "read_latest_scan_progress",
        fake_latest,
    )

    assert await scan_service._processing_scan_is_stale(job, now=now) is False
