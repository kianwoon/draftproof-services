from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.services import rewrite_service


def test_completed_external_review_candidate_is_not_reused():
    assert rewrite_service._completed_rewrite_report_is_reusable(
        {
            "status": "rewrite_candidate_generated_needs_external_review",
            "summary": {
                "rewrite_pipeline_version": "rewrite_v4_normalized_repair",
                "outcome": "rewrite_candidate_generated_needs_external_review",
                "strict_goal_status": "mitigation_failed_no_safe_candidate",
                "best_candidate_external_review_required": True,
                "detect_scores": {
                    "original_ai": 48.32,
                    "rewritten_ai": 34.02,
                },
            },
        }
    ) is False


def test_completed_author_review_candidate_is_not_reused():
    assert rewrite_service._completed_rewrite_report_is_reusable(
        {
            "status": "rewrite_candidate_generated_needs_author_review",
            "summary": {
                "rewrite_pipeline_version": "rewrite_v4_normalized_repair",
                "outcome": "rewrite_candidate_generated_needs_author_review",
                "best_candidate_author_review_required": True,
                "author_proxy_context": {
                    "active": True,
                    "review_required": True,
                    "review_cards": [{"card_id": "target-01"}],
                },
                "detect_scores": {
                    "original_ai": 48.32,
                    "rewritten_ai": 34.02,
                },
            },
        }
    ) is False


def _stream_id(at: datetime) -> str:
    return f"{int(at.timestamp() * 1000)}-0"


@pytest.mark.asyncio
async def test_processing_rewrite_is_stale_when_heartbeat_stops(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        status="processing",
        created_at=now - timedelta(minutes=8),
    )

    async def fake_latest(_rewrite_id):
        return (_stream_id(now - timedelta(minutes=6)), {"status": "processing"})

    monkeypatch.setattr(
        rewrite_service.progress_stream,
        "read_latest_rewrite_progress",
        fake_latest,
    )

    assert await rewrite_service._processing_rewrite_is_stale(job, now=now) is True


@pytest.mark.asyncio
async def test_processing_rewrite_is_not_stale_with_recent_heartbeat(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000002",
        status="processing",
        created_at=now - timedelta(minutes=8),
    )

    async def fake_latest(_rewrite_id):
        return (_stream_id(now - timedelta(minutes=1)), {"status": "processing"})

    monkeypatch.setattr(
        rewrite_service.progress_stream,
        "read_latest_rewrite_progress",
        fake_latest,
    )

    assert await rewrite_service._processing_rewrite_is_stale(job, now=now) is False


@pytest.mark.asyncio
async def test_old_processing_rewrite_is_not_stale_with_recent_heartbeat(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000005",
        status="processing",
        created_at=now - rewrite_service._STALE_THRESHOLD - timedelta(minutes=10),
    )

    async def fake_latest(_rewrite_id):
        return (_stream_id(now - timedelta(minutes=1)), {"status": "processing"})

    monkeypatch.setattr(
        rewrite_service.progress_stream,
        "read_latest_rewrite_progress",
        fake_latest,
    )

    assert await rewrite_service._processing_rewrite_is_stale(job, now=now) is False


@pytest.mark.asyncio
async def test_processing_rewrite_falls_back_to_hard_stale_threshold(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000003",
        status="processing",
        created_at=now - rewrite_service._STALE_THRESHOLD - timedelta(seconds=1),
    )

    async def fake_latest(_rewrite_id):
        return None

    monkeypatch.setattr(
        rewrite_service.progress_stream,
        "read_latest_rewrite_progress",
        fake_latest,
    )

    assert await rewrite_service._processing_rewrite_is_stale(job, now=now) is True


@pytest.mark.asyncio
async def test_old_processing_rewrite_with_invalid_heartbeat_uses_hard_stale_threshold(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000006",
        status="processing",
        created_at=now - rewrite_service._STALE_THRESHOLD - timedelta(seconds=1),
    )

    async def fake_latest(_rewrite_id):
        return ("not-a-redis-stream-id", {"status": "processing"})

    monkeypatch.setattr(
        rewrite_service.progress_stream,
        "read_latest_rewrite_progress",
        fake_latest,
    )

    assert await rewrite_service._processing_rewrite_is_stale(job, now=now) is True


def _patch_recovery_boundary(monkeypatch, *, saved_report):
    """Isolate _mark_rewrite_interrupted's branch logic from the DB/R2 boundary."""
    captured = {"completed": False, "released": False}

    async def fake_fetch(_scan_id):
        return saved_report

    async def fake_capture(_session, _job_id):
        captured["completed"] = True

    async def fake_release(_session, _job_id):
        captured["released"] = True

    monkeypatch.setattr(rewrite_service, "_fetch_rewrite_report_json", fake_fetch)
    monkeypatch.setattr(rewrite_service, "_capture_active_rewrite_reservation", fake_capture)
    monkeypatch.setattr(rewrite_service, "_release_active_reservation", fake_release)
    return captured


def _delivered_report(rewrite_id: str) -> dict:
    return {
        "rewrite_id": rewrite_id,
        "status": "rewrite_candidate_generated_needs_external_review",
        "final_text": "a rewritten paragraph with real content",
        "original_text": "the original paragraph",
        "summary": {"outcome": "ai_mitigated"},
    }


@pytest.mark.asyncio
async def test_mark_interrupted_captures_when_artifact_belongs_to_this_job(monkeypatch):
    job = SimpleNamespace(id="job-b", scan_id="scan-1", status="processing")
    captured = _patch_recovery_boundary(monkeypatch, saved_report=_delivered_report("job-b"))

    await rewrite_service._mark_rewrite_interrupted(session=None, job=job)

    assert job.status == "completed"
    assert captured["completed"] is True
    assert captured["released"] is False


@pytest.mark.asyncio
async def test_mark_interrupted_does_not_bill_a_different_jobs_artifact(monkeypatch):
    """Regression test: a stale retry job (B) must not be billed for a prior
    job's (A) already-delivered, already-billed artifact just because they
    share the same scan_id R2 key."""
    job = SimpleNamespace(id="job-b", scan_id="scan-1", status="processing")
    captured = _patch_recovery_boundary(monkeypatch, saved_report=_delivered_report("job-a"))

    await rewrite_service._mark_rewrite_interrupted(session=None, job=job)

    assert job.status == "failed"
    assert captured["completed"] is False
    assert captured["released"] is True


@pytest.mark.asyncio
async def test_mark_interrupted_does_not_bill_pre_fix_artifact_without_rewrite_id(monkeypatch):
    """Fail-safe direction: an artifact uploaded before this fix shipped has no
    rewrite_id at all — must not be trusted as evidence of delivery either."""
    job = SimpleNamespace(id="job-b", scan_id="scan-1", status="processing")
    report_without_id = _delivered_report("job-a")
    del report_without_id["rewrite_id"]
    captured = _patch_recovery_boundary(monkeypatch, saved_report=report_without_id)

    await rewrite_service._mark_rewrite_interrupted(session=None, job=job)

    assert job.status == "failed"
    assert captured["completed"] is False
    assert captured["released"] is True


@pytest.mark.asyncio
async def test_mark_interrupted_fails_when_no_artifact_saved(monkeypatch):
    job = SimpleNamespace(id="job-b", scan_id="scan-1", status="processing")
    captured = _patch_recovery_boundary(monkeypatch, saved_report=None)

    await rewrite_service._mark_rewrite_interrupted(session=None, job=job)

    assert job.status == "failed"
    assert captured["completed"] is False
    assert captured["released"] is True


@pytest.mark.asyncio
async def test_processing_rewrite_without_heartbeat_uses_conservative_fallback(monkeypatch):
    now = datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000004",
        status="processing",
        created_at=now - timedelta(minutes=8),
    )

    async def fake_latest(_rewrite_id):
        return None

    monkeypatch.setattr(
        rewrite_service.progress_stream,
        "read_latest_rewrite_progress",
        fake_latest,
    )

    assert await rewrite_service._processing_rewrite_is_stale(job, now=now) is False
