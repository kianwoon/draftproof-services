import sys
from datetime import datetime, timedelta, timezone
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
        return SimpleNamespace(scalar=lambda: 0)

    async def commit(self):
        self.committed = True


class FakeScanTask:
    def __init__(self):
        self.calls = []

    def delay(self, *args):
        self.calls.append(args)


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


class CountingFreeScanSession:
    """Simulates SQLAlchemy autoflush: the free-scan count query sees any job
    already ``add``-ed to the session (real bug mechanism). If the limit check
    runs after ``session.add(job)``, the in-flight pending scan counts against
    itself — the off-by-one this guards."""

    def __init__(self, prior_free_scans: int):
        self.prior = prior_free_scans
        self.added = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, item):
        self.added.append(item)

    async def execute(self, *_args, **_kwargs):
        flushed = sum(
            1
            for j in self.added
            if getattr(j, "word_count", 10 ** 9) <= scan_service.FREE_SCAN_WORD_LIMIT
        )
        return SimpleNamespace(scalar=lambda: self.prior + flushed)

    async def commit(self):
        self.committed = True


def _stream_id(at: datetime) -> str:
    return f"{int(at.timestamp() * 1000)}-0"


def test_scan_cost_is_free_through_800_words():
    assert scan_service._scan_cost(1) == 0
    assert scan_service._scan_cost(800) == 0


def test_scan_cost_charges_from_801_words():
    assert scan_service._scan_cost(801) == 1
    assert scan_service._scan_cost(1000) == 1
    assert scan_service._scan_cost(1001) == 2


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
        text=words(500),
    )

    assert result["status"] == "pending"
    assert fake_session.committed is True
    assert len(fake_session.added) == 1
    assert fake_session.added[0].word_count == 500
    assert len(fake_session.added[0].document_title) <= scan_service.DOCUMENT_TITLE_MAX_CHARS
    assert fake_session.added[0].document_title.startswith("word0 word1")
    assert fake_session.added[0].document_title.endswith("…")
    assert fake_session.added[0].content_preview.endswith("…")
    assert len(fake_task.calls) == 1


@pytest.mark.asyncio
async def test_fifth_free_scan_allowed_when_four_used(monkeypatch):
    """Regression: the in-flight pending scan must not count against itself.
    With 4 free scans already used, the legitimate 5th must be allowed."""
    fake_session = CountingFreeScanSession(prior_free_scans=4)
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
    assert len(fake_task.calls) == 1


@pytest.mark.asyncio
async def test_sixth_free_scan_blocked_when_five_used(monkeypatch):
    """The limit still holds: a 6th short scan with 5 genuinely used is blocked."""
    fake_session = CountingFreeScanSession(prior_free_scans=5)
    fake_task = FakeScanTask()

    monkeypatch.setattr(scan_service, "async_session", lambda: fake_session)
    monkeypatch.setitem(
        sys.modules,
        "app.services.celery_client",
        SimpleNamespace(scan_document=fake_task),
    )

    with pytest.raises(ValueError, match="Free scan limit reached"):
        await scan_service.create_scan(
            "paste",
            user_id="00000000-0000-0000-0000-000000000001",
            text=words(300),
        )

    assert len(fake_task.calls) == 0


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
    )

    async def fake_latest(_scan_id):
        return (_stream_id(now - timedelta(minutes=6)), {"status": "processing"})

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
    )

    async def fake_latest(_scan_id):
        return None

    monkeypatch.setattr(
        scan_service.progress_stream,
        "read_latest_scan_progress",
        fake_latest,
    )

    assert await scan_service._processing_scan_is_stale(job, now=now) is False
