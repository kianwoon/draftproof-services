"""Rewrite service — create rewrite jobs, fetch results from R2."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from app.config import REWRITE_STALE_THRESHOLD_MINUTES
from app.models.db import async_session, RewriteJob, ScanJob, CreditAccount, CreditReservation
from app.services.scan_service import _rewrite_cost
from app.services import progress_stream

logger = logging.getLogger("rewrite_service")

_STALE_THRESHOLD = timedelta(minutes=REWRITE_STALE_THRESHOLD_MINUTES)
_PROCESSING_HEARTBEAT_STALE_THRESHOLD = timedelta(minutes=5)
_ACTIVE_REWRITE_STATUSES = ("pending", "processing", "retrying")
_STALE_RECOVERY_STATUSES = ("pending", "retrying")
_REPHRASABLE_TYPES = {
    "high_predictability",
    "medium_predictability",
    "review_predictability",
    "high_topk_predictability",
    "low_predictability",
    "formulaic_sentence",
    "generic_phrase",
    "style_shift",
    "low_specificity",
}

REVIEW_ONLY_REWRITE_MESSAGE = (
    "No rewriteable AI sections were found. The findings on this report are review-only, "
    "so DraftProof did not start a rewrite or reserve any tokens."
)


class NoRewriteableFindingsError(ValueError):
    """Raised when a completed report has findings, but none can be rewritten automatically."""


def _flatten_report_findings(report_json: dict) -> list[dict]:
    findings = []
    for tier_findings in (report_json.get("findings") or {}).values():
        if isinstance(tier_findings, list):
            findings.extend(f for f in tier_findings if isinstance(f, dict))
    return findings


def _has_rewriteable_findings(report_json: dict) -> bool:
    rewrite_decision = report_json.get("rewrite_decision")
    if isinstance(rewrite_decision, dict) and rewrite_decision.get("run_rewrite") is False:
        return False

    for finding in _flatten_report_findings(report_json):
        if finding.get("actionability") in ("auto_fixable", "auto_rewrite_candidate"):
            return True
        if finding.get("title") in _REPHRASABLE_TYPES and finding.get("recommendation"):
            return True
    return False


async def _fetch_scan_report_json(scan_id: str) -> dict | None:
    from app.services.report_service import _r2, _fetch_report_json_sync

    if not _r2:
        return None

    try:
        return await asyncio.to_thread(_fetch_report_json_sync, f"reports/{scan_id}/report.json")
    except Exception as exc:
        logger.warning("Failed to preflight rewrite findings for scan %s: %s", scan_id, exc)
        return None


async def _release_active_reservation(session, job_id: uuid.UUID) -> None:
    reservation_result = await session.execute(
        select(CreditReservation).where(
            CreditReservation.job_type == "rewrite",
            CreditReservation.job_id == job_id,
            CreditReservation.status == "active",
        ).with_for_update()
    )
    reservation = reservation_result.scalar_one_or_none()
    if not reservation:
        return

    acct_result = await session.execute(
        select(CreditAccount).where(CreditAccount.id == reservation.credit_account_id).with_for_update()
    )
    acct = acct_result.scalar_one_or_none()
    if acct:
        acct.reserved_tokens = max(0, acct.reserved_tokens - reservation.tokens_reserved)
    reservation.status = "released"


def _redis_stream_event_time(event_id: str) -> datetime | None:
    """Parse a Redis stream ID into UTC time."""
    try:
        millis = int(str(event_id).split("-", 1)[0])
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)


async def _processing_rewrite_is_stale(job: RewriteJob, *, now: datetime | None = None) -> bool:
    """Detect a processing rewrite whose worker heartbeat has stopped.

    Redis progress events are the lightweight heartbeat. If Redis is unavailable
    or the stream is missing, fall back to the conservative hard stale threshold
    so active long rewrites are not killed incorrectly.
    """
    if job.status != "processing" or not job.created_at:
        return False

    now = now or datetime.now(timezone.utc)
    age = now - job.created_at
    if age > _STALE_THRESHOLD:
        return True

    latest = await progress_stream.read_latest_rewrite_progress(str(job.id))
    if latest is None:
        return False

    event_time = _redis_stream_event_time(latest[0])
    if event_time is None:
        return False

    return now - event_time > _PROCESSING_HEARTBEAT_STALE_THRESHOLD


async def _mark_rewrite_interrupted(session, job: RewriteJob) -> None:
    job.status = "failed"
    job.error = "Rewrite interrupted during worker restart"
    job.progress_message = "Rewrite worker restarted. Please retry."
    job.completed_at = datetime.now(timezone.utc)
    await _release_active_reservation(session, job.id)


async def create_rewrite(scan_id: str, user_id: str) -> dict:
    """Create a rewrite job: validate scan, check balance, deduct tokens, enqueue."""
    uid = uuid.UUID(user_id)
    scan_uuid = uuid.UUID(scan_id)

    async with async_session() as session:
        result = await session.execute(
            select(ScanJob).where(
                ScanJob.id == scan_uuid,
                ScanJob.user_id == uid,
                ScanJob.status == "completed",
            ).with_for_update()
        )
        scan = result.scalar_one_or_none()
        if not scan:
            raise ValueError("Completed scan not found")

        # Recover stale queued/retry jobs first. Processing rewrites are handled
        # below with the Redis heartbeat so deploy-killed workers do not leave
        # a permanently active job row.
        stale_cutoff = datetime.now(timezone.utc) - _STALE_THRESHOLD
        stale = await session.execute(
            select(RewriteJob).where(
                RewriteJob.scan_id == scan_uuid,
                RewriteJob.status.in_(_STALE_RECOVERY_STATUSES),
                RewriteJob.created_at < stale_cutoff,
            )
        )
        stale_jobs = stale.scalars().all()
        for stale_job in stale_jobs:
            await _mark_rewrite_interrupted(session, stale_job)

        # Check for actively running rewrites.
        existing = await session.execute(
            select(RewriteJob).where(
                RewriteJob.scan_id == scan_uuid,
                RewriteJob.status.in_(_ACTIVE_REWRITE_STATUSES),
            ).order_by(RewriteJob.created_at.desc()).limit(1)
        )
        existing_job = existing.scalar_one_or_none()
        if existing_job:
            if await _processing_rewrite_is_stale(existing_job):
                await _mark_rewrite_interrupted(session, existing_job)
                await session.commit()
            else:
                await session.commit()
                return _rewrite_to_dict(existing_job)

        if stale_jobs:
            await session.commit()

        completed = await session.execute(
            select(RewriteJob).where(
                RewriteJob.scan_id == scan_uuid,
                RewriteJob.status == "completed",
            ).order_by(RewriteJob.completed_at.desc(), RewriteJob.created_at.desc()).limit(1)
        )
        completed_job = completed.scalar_one_or_none()
        if completed_job:
            await session.commit()
            return _rewrite_to_dict(completed_job)

        report_json = await _fetch_scan_report_json(scan_id)
        if report_json is not None and not _has_rewriteable_findings(report_json):
            if stale_jobs:
                await session.commit()
            raise NoRewriteableFindingsError(REVIEW_ONLY_REWRITE_MESSAGE)

        # Get word count from scan job for cost calculation
        scan_result = await session.execute(
            select(ScanJob.word_count).where(ScanJob.id == scan_uuid)
        )
        word_count = scan_result.scalar_one_or_none() or 0
        cost = _rewrite_cost(word_count)

        acct_result = await session.execute(
            select(CreditAccount).where(CreditAccount.user_id == uid).with_for_update()
        )
        acct = acct_result.scalar_one_or_none()
        if not acct:
            raise ValueError("No credit account found")
        if acct.balance_tokens - acct.reserved_tokens < cost:
            raise ValueError(f"Insufficient tokens (need {cost})")

        acct.reserved_tokens += cost
        job_id = uuid.uuid4()
        rewrite_job = RewriteJob(
            id=job_id,
            scan_id=scan_uuid,
            user_id=uid,
            status="pending",
            progress_percent=0,
            progress_message="Queued",
        )
        session.add(rewrite_job)

        reservation = CreditReservation(
            user_id=uid,
            credit_account_id=acct.id,
            job_type="rewrite",
            job_id=job_id,
            tokens_reserved=cost,
            status="active",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        session.add(reservation)
        await session.commit()

    from app.services.celery_client import run_rewrite
    run_rewrite.apply_async(args=[str(job_id), scan_id], task_id=str(job_id))

    return {
        "id": str(job_id),
        "scan_id": scan_id,
        "status": "pending",
        "progress_percent": 0,
        "progress_message": "Queued",
    }


async def cancel_rewrite(rewrite_id: str, user_id: str) -> dict | None:
    """Cancel an active rewrite and release its reserved credits."""
    rewrite_uuid = uuid.UUID(rewrite_id)
    uid = uuid.UUID(user_id)
    async with async_session() as session:
        result = await session.execute(
            select(RewriteJob)
            .where(RewriteJob.id == rewrite_uuid, RewriteJob.user_id == uid)
            .with_for_update()
        )
        job = result.scalar_one_or_none()
        if not job:
            return None

        if job.status in _ACTIVE_REWRITE_STATUSES:
            job.status = "canceled"
            job.error = "Rewrite canceled by user"
            job.progress_message = "Rewrite canceled"
            job.completed_at = datetime.now(timezone.utc)
            await _release_active_reservation(session, job.id)
            await session.commit()
            await session.refresh(job)
        else:
            await session.commit()

    return await get_rewrite(rewrite_id, user_id)


async def get_rewrite(rewrite_id: str, user_id: str | None = None) -> dict | None:
    """Get rewrite job status.

    Stale recovery uses the Redis heartbeat for ``processing`` rewrites. This
    prevents deploy-killed worker tasks from leaving jobs stuck indefinitely.
    """
    async with async_session() as session:
        q = select(RewriteJob).where(RewriteJob.id == uuid.UUID(rewrite_id))
        if user_id:
            q = q.where(RewriteJob.user_id == uuid.UUID(user_id))
        q = q.with_for_update()
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return None

        if job.status in _STALE_RECOVERY_STATUSES and job.created_at:
            age = datetime.now(timezone.utc) - job.created_at
            if age > _STALE_THRESHOLD:
                await _mark_rewrite_interrupted(session, job)
                await session.commit()
                await session.refresh(job)
        elif await _processing_rewrite_is_stale(job):
            await _mark_rewrite_interrupted(session, job)
            await session.commit()
            await session.refresh(job)

        return _rewrite_to_dict(job)


async def get_rewrite_report(rewrite_id: str, user_id: str) -> dict | None:
    """Fetch rewrite report JSON from R2."""
    job_info = await get_rewrite(rewrite_id, user_id)
    if not job_info or job_info["status"] != "completed":
        return None

    from app.services.report_service import _r2, _fetch_report_json_sync
    if not _r2:
        return None

    scan_id = job_info["scan_id"]
    r2_key = f"reports/{scan_id}/rewrite/rewrite.json"
    try:
        data = await asyncio.to_thread(_fetch_report_json_sync, r2_key)
        return data
    except Exception as e:
        logger.warning("Failed to fetch rewrite JSON from R2: %s", e)
        return None


async def regenerate_rewrite_report_assets(rewrite_id: str, user_id: str) -> dict | None:
    """Regenerate stored rewrite.md/rewrite.pdf through the worker."""
    job_info = await get_rewrite(rewrite_id, user_id)
    if not job_info:
        return None
    if job_info["status"] != "completed":
        raise ValueError("Rewrite is not completed")

    from celery.exceptions import NotRegistered, TimeoutError as CeleryTimeoutError
    from app.services.celery_client import regenerate_rewrite_report_assets as regen_task

    def _enqueue_and_wait():
        task = regen_task.delay(rewrite_id, job_info["scan_id"])
        return task.get(timeout=45)

    try:
        return await asyncio.to_thread(_enqueue_and_wait)
    except NotRegistered:
        logger.warning(
            "Rewrite report regeneration worker task is not registered yet for rewrite %s",
            rewrite_id,
        )
        return {
            "status": "worker_unavailable",
            "rewrite_id": rewrite_id,
            "scan_id": job_info["scan_id"],
        }
    except CeleryTimeoutError:
        return {"status": "queued", "rewrite_id": rewrite_id, "scan_id": job_info["scan_id"]}
    except (NotImplementedError, RuntimeError, AttributeError) as e:
        message = str(e).lower()
        if "result backend" not in message and "disabledbackend" not in message:
            raise
        logger.info(
            "Rewrite report regeneration queued without waiting for result backend for rewrite %s",
            rewrite_id,
        )
        return {"status": "queued", "rewrite_id": rewrite_id, "scan_id": job_info["scan_id"]}


async def get_rewrite_download_url(rewrite_id: str, fmt: str, user_id: str) -> str | None:
    """Generate presigned download URL for rewrite output."""
    job_info = await get_rewrite(rewrite_id, user_id)
    if not job_info or job_info["status"] != "completed":
        return None

    from app.services.report_service import _r2
    from app.config import R2_BUCKET_NAME
    if not _r2:
        return None

    scan_id = job_info["scan_id"]
    fmt_map = {
        "pdf": "rewrite.pdf",
        "md": "rewrite.md",
        "txt": "rewritten.txt",
        "log": "rewrite.log",
    }
    filename = fmt_map.get(fmt)
    if not filename:
        return None

    key = f"reports/{scan_id}/rewrite/{filename}"
    try:
        url = await asyncio.to_thread(
            _r2.generate_presigned_url,
            "get_object",
            {"Bucket": R2_BUCKET_NAME, "Key": key},
            ExpiresIn=3600,
        )
        return url
    except Exception:
        return None


async def get_detect_json_url(rewrite_id: str, user_id: str) -> str | None:
    """Generate presigned download URL for the original detect scan report.json."""
    job_info = await get_rewrite(rewrite_id, user_id)
    if not job_info:
        return None

    from app.services.report_service import _r2
    from app.config import R2_BUCKET_NAME
    if not _r2:
        return None

    scan_id = job_info["scan_id"]
    key = f"reports/{scan_id}/report.json"
    try:
        url = await asyncio.to_thread(
            _r2.generate_presigned_url,
            "get_object",
            {"Bucket": R2_BUCKET_NAME, "Key": key},
            ExpiresIn=3600,
        )
        return url
    except Exception:
        return None


def _rewrite_to_dict(job: RewriteJob) -> dict:
    return {
        "id": str(job.id),
        "scan_id": str(job.scan_id),
        "status": job.status,
        "error": job.error,
        "progress_percent": job.progress_percent,
        "progress_message": _normalize_rewrite_progress_message(job.progress_message),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def _normalize_rewrite_progress_message(message: str | None) -> str | None:
    if not message:
        return message

    normalized = message.strip().lower()
    legacy_messages = (
        "rewriting your document",
        "this may take 1-3 minutes",
    )
    if any(legacy_message in normalized for legacy_message in legacy_messages):
        return "Rewriting AI sections"

    return message
