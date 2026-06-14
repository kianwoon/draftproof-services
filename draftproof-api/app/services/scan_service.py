"""Scan service — creates scan jobs and dispatches to Celery worker."""

import asyncio
import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.config import UPLOAD_DIR
from app.models.db import async_session, ScanJob, CreditAccount, CreditReservation
from app.services import progress_stream


FREE_SCAN_WORD_LIMIT = 800
FREE_SCAN_LIMIT = 5
DOCUMENT_TITLE_MAX_CHARS = 90
CONTENT_PREVIEW_MAX_CHARS = 220
_STALE_THRESHOLD = timedelta(minutes=10)
_PROCESSING_HEARTBEAT_STALE_THRESHOLD = timedelta(minutes=5)
_ACTIVE_SCAN_STATUSES = ("pending", "processing", "retrying")
_PROCESSING_SCAN_STATUSES = ("processing",)
_MEANINGFUL_TEXT_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def _compact_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _truncate_display_text(value: str, limit: int) -> str:
    compact = _compact_text(value)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _first_meaningful_segment(text: str) -> str:
    for raw_line in str(text or "").splitlines():
        line = _compact_text(raw_line)
        if _MEANINGFUL_TEXT_RE.search(line):
            sentence_match = re.match(r"^(.+?[.!?。！？])(?:\s|$)", line)
            return sentence_match.group(1) if sentence_match else line
    compact = _compact_text(text)
    return compact if _MEANINGFUL_TEXT_RE.search(compact) else ""


def build_scan_report_metadata(text: str) -> dict:
    title_source = _first_meaningful_segment(text)
    preview_source = _compact_text(text)
    return {
        "document_title": _truncate_display_text(title_source, DOCUMENT_TITLE_MAX_CHARS) or None,
        "content_preview": _truncate_display_text(preview_source, CONTENT_PREVIEW_MAX_CHARS) or None,
    }


def _scan_cost(word_count: int) -> int:
    """Free through 800 words, then 1 token per started 1,000 words."""
    if word_count <= FREE_SCAN_WORD_LIMIT:
        return 0
    return max(1, -(-word_count // 1000))


async def _count_free_scans_used(session, user_id: uuid.UUID) -> int:
    from sqlalchemy import func
    result = await session.execute(
        select(func.count()).select_from(ScanJob).where(
            ScanJob.user_id == user_id,
            ScanJob.word_count <= FREE_SCAN_WORD_LIMIT,
            ScanJob.status.notin_(["failed", "canceled"]),
        )
    )
    return result.scalar() or 0


def _rewrite_cost(word_count: int) -> int:
    """5 tokens per 1,000 words (ceiling). 1-1000 = 5, 1001-2000 = 10, etc."""
    return max(5, -(-word_count // 1000) * 5)


def _read_document_text_sync(document_id: str) -> str:
    """Read uploaded document text from disk (sync — call via to_thread)."""
    for ext in (".txt", ".pdf", ".docx"):
        path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
    return ""


def _redis_stream_event_time(event_id: str) -> datetime | None:
    try:
        millis = int(str(event_id).split("-", 1)[0])
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)


async def _processing_scan_is_stale(job: ScanJob, *, now: datetime | None = None) -> bool:
    """Detect a processing scan whose worker heartbeat has stopped.

    Redis progress events are the lightweight worker heartbeat. If Redis is
    unavailable or the stream is missing, keep the existing hard stale timeout
    so live scans are not killed incorrectly.
    """
    if job.status not in _PROCESSING_SCAN_STATUSES or not job.created_at:
        return False

    now = now or datetime.now(timezone.utc)
    age = now - job.created_at
    if age > _STALE_THRESHOLD:
        return True

    latest = await progress_stream.read_latest_scan_progress(str(job.id))
    if latest is None:
        return False

    event_time = _redis_stream_event_time(latest[0])
    if event_time is None:
        return False

    return now - event_time > _PROCESSING_HEARTBEAT_STALE_THRESHOLD


async def _release_active_scan_reservations(session, job_id: uuid.UUID) -> int:
    reservations = await session.execute(
        select(CreditReservation).where(
            CreditReservation.job_type == "scan",
            CreditReservation.job_id == job_id,
            CreditReservation.status == "active",
        )
    )
    released_tokens = 0
    for reservation in reservations.scalars().all():
        reservation.status = "released"
        acct_result = await session.execute(
            select(CreditAccount).where(CreditAccount.id == reservation.credit_account_id)
        )
        account = acct_result.scalar_one_or_none()
        if account:
            account.reserved_tokens = max(0, account.reserved_tokens - reservation.tokens_reserved)
            released_tokens += reservation.tokens_reserved
    return released_tokens


async def _mark_scan_interrupted(session, job: ScanJob) -> int:
    job.status = "failed"
    job.error = "Scan interrupted during worker restart"
    job.progress_message = "Scan worker restarted. Please retry."
    job.completed_at = datetime.now(timezone.utc)
    return await _release_active_scan_reservations(session, job.id)


async def create_scan(document_id: str, user_id: str | None = None, text: str | None = None) -> dict:
    """Create a scan_job row, enqueue Celery task, return scan info."""
    if not text:
        text = await asyncio.to_thread(_read_document_text_sync, document_id)
    if not text:
        raise ValueError("Document text not found or empty")

    text_hash = hashlib.sha256(text.encode()).hexdigest()
    word_count = len(text.split())
    job_id = uuid.uuid4()
    report_metadata = build_scan_report_metadata(text)

    async with async_session() as session:
        # Reserve tokens based on word count. Short scans are free and should
        # not require an existing credit account.
        cost = _scan_cost(word_count)
        # Enforce the free-scan limit BEFORE adding this job to the session.
        # Otherwise SQLAlchemy autoflushes the new pending row before the count
        # query runs, so the in-flight scan counts against itself — an
        # off-by-one that blocks the user's legitimate Nth free scan even though
        # only N-1 have actually been used.
        if user_id and cost == 0:
            used = await _count_free_scans_used(session, uuid.UUID(user_id))
            if used >= FREE_SCAN_LIMIT:
                raise ValueError(
                    f"Free scan limit reached ({FREE_SCAN_LIMIT} scans). "
                    "Please purchase credits to continue scanning."
                )

        job = ScanJob(
            id=job_id,
            user_id=uuid.UUID(user_id) if user_id else None,
            input_text_hash=text_hash,
            word_count=word_count,
            document_title=report_metadata["document_title"],
            content_preview=report_metadata["content_preview"],
            scan_type="scan",
            status="pending",
        )
        session.add(job)

        if user_id and cost > 0:
            uid = uuid.UUID(user_id)
            result = await session.execute(
                select(CreditAccount).where(CreditAccount.user_id == uid).with_for_update()
            )
            acct = result.scalar_one_or_none()
            if not acct:
                raise ValueError("No credit account found — please purchase tokens first")
            if acct.balance_tokens - acct.reserved_tokens < cost:
                raise ValueError("Insufficient tokens — please purchase more")

            acct.reserved_tokens += cost
            reservation = CreditReservation(
                user_id=uid,
                credit_account_id=acct.id,
                job_type="scan",
                job_id=job_id,
                tokens_reserved=cost,
                status="active",
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            )
            session.add(reservation)

        await session.commit()

    from app.services.celery_client import scan_document
    scan_document.delay(str(job_id), text)

    return {
        "id": str(job_id),
        "document_id": document_id,
        "status": "pending",
        "progress_percent": 0,
        "progress_message": "Queued",
        "report_id": None,
    }


async def get_free_scan_usage(user_id: str) -> dict:
    uid = uuid.UUID(user_id)
    async with async_session() as session:
        used = await _count_free_scans_used(session, uid)
    return {"used": used, "limit": FREE_SCAN_LIMIT, "remaining": max(0, FREE_SCAN_LIMIT - used)}


async def list_scans(user_id: str, page: int = 1, per_page: int = 10) -> dict:
    """List scan_jobs for a user with pagination, newest first."""
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    offset = (page - 1) * per_page
    uid = uuid.UUID(user_id)

    async with async_session() as session:
        from sqlalchemy import func

        # Only run stale recovery if user has active jobs.
        active_result = await session.execute(
            select(ScanJob)
            .where(ScanJob.user_id == uid)
            .where(ScanJob.status.in_(_ACTIVE_SCAN_STATUSES))
        )
        active_jobs = active_result.scalars().all()
        if active_jobs:
            now = datetime.now(timezone.utc)
            for active_job in active_jobs:
                age = now - active_job.created_at if active_job.created_at else timedelta(0)
                if age > _STALE_THRESHOLD or await _processing_scan_is_stale(active_job, now=now):
                    await _mark_scan_interrupted(session, active_job)
            await session.commit()

        count_result = await session.execute(
            select(func.count()).select_from(ScanJob)
            .where(ScanJob.user_id == uid)
        )
        total = count_result.scalar() or 0

        result = await session.execute(
            select(ScanJob)
            .where(ScanJob.user_id == uid)
            .order_by(ScanJob.created_at.desc())
            .offset(offset)
            .limit(per_page)
        )
        jobs = result.scalars().all()
        return {
            "items": [
                {
                    "id": str(j.id),
                    "document_id": "",
                    "status": j.status,
                    "report_id": str(j.id) if j.status == "completed" else None,
                    "document_title": j.document_title,
                    "content_preview": j.content_preview,
                    "tier": j.tier,
                    "ai_score": float(j.ai_score) if j.ai_score is not None else None,
                    "writing_score": float(j.writing_score) if j.writing_score is not None else None,
                    "finding_count": j.finding_count,
                    "progress_percent": j.progress_percent,
                    "progress_message": j.progress_message,
                    "word_count": j.word_count,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                }
                for j in jobs
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page,
        }

async def _mark_stale_jobs_failed(user_id: uuid.UUID | None = None) -> None:
    """Bulk-mark processing/pending jobs older than threshold as failed.

    Also releases any active credit reservations for those jobs so tokens
    are not permanently locked.
    """
    import logging
    log = logging.getLogger("scan_service.stale")
    cutoff = datetime.now(timezone.utc) - _STALE_THRESHOLD
    async with async_session() as session:
        # Find stale jobs first
        q = select(ScanJob).where(
            ScanJob.status.in_(_ACTIVE_SCAN_STATUSES),
            ScanJob.created_at < cutoff,
        )
        if user_id:
            q = q.where(ScanJob.user_id == user_id)
        result = await session.execute(q)
        stale_jobs = result.scalars().all()

        if not stale_jobs:
            return

        released_count = 0
        for stale_job in stale_jobs:
            released_count += await _mark_scan_interrupted(session, stale_job)

        await session.commit()
        if released_count:
            log.info("Stale job cleanup: %d jobs, %d tokens released", len(stale_jobs), released_count)


async def get_scan(scan_id: str, user_id: str | None = None) -> dict | None:
    """Look up scan_job by ID, optionally scoped to a user.

    Auto-marks jobs stuck in active states as failed. Processing scans use the
    Redis progress heartbeat so deploy-killed workers do not leave scans stuck.
    """
    async with async_session() as session:
        q = select(ScanJob).where(ScanJob.id == uuid.UUID(scan_id))
        if user_id:
            q = q.where(ScanJob.user_id == uuid.UUID(user_id))
        q = q.with_for_update()
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return None

        if job.status in _ACTIVE_SCAN_STATUSES and job.created_at:
            age = datetime.now(timezone.utc) - job.created_at
            if age > _STALE_THRESHOLD or await _processing_scan_is_stale(job):
                await _mark_scan_interrupted(session, job)
                await session.commit()
                await session.refresh(job)

        return {
            "id": str(job.id),
            "document_id": "",
            "status": job.status,
            "report_id": str(job.id) if job.status == "completed" else None,
            "tier": job.tier,
            "ai_score": float(job.ai_score) if job.ai_score is not None else None,
            "writing_score": float(job.writing_score) if job.writing_score is not None else None,
            "finding_count": job.finding_count,
            "progress_percent": job.progress_percent,
            "progress_message": job.progress_message,
        }


async def delete_scan(scan_id: str, user_id: str) -> bool:
    """Delete a scan job and its R2 report files. Returns True if found."""
    import logging
    log = logging.getLogger("scan_service.delete")
    async with async_session() as session:
        q = select(ScanJob).where(
            ScanJob.id == uuid.UUID(scan_id),
            ScanJob.user_id == uuid.UUID(user_id),
        )
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return False

        # Clean up R2 report files
        report_urls = job.report_urls or {}
        keys = [f"reports/{scan_id}/{name}" for name in ("report.json", "report.md", "report.pdf")]
        await asyncio.to_thread(_delete_r2_objects, keys, log)

        await session.delete(job)
        await session.commit()
        log.info("Deleted scan %s for user %s", scan_id, user_id)
        return True


def _delete_r2_objects(keys: list[str], log) -> None:
    try:
        from app.services.report_service import _r2
        from app.config import R2_BUCKET_NAME
        if not _r2:
            return
        for key in keys:
            try:
                _r2.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
            except Exception:
                pass  # R2 key may not exist for all file types
    except Exception as e:
        log.warning("R2 cleanup skipped: %s", e)
