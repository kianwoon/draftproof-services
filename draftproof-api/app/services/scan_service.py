"""Scan service — creates scan jobs and dispatches to Celery worker."""

import asyncio
import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

from app.config import UPLOAD_DIR
from app.models.db import async_session, ScanJob, CreditAccount, CreditReservation


def _scan_cost(word_count: int) -> int:
    """1 token per 1,000 words (ceiling). 1–1000 = 1, 1001–2000 = 2, etc."""
    return max(1, -(-word_count // 1000))


def _read_document_text_sync(document_id: str) -> str:
    """Read uploaded document text from disk (sync — call via to_thread)."""
    for ext in (".txt", ".pdf", ".docx"):
        path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
    return ""


async def create_scan(document_id: str, user_id: str | None = None, text: str | None = None) -> dict:
    """Create a scan_job row, enqueue Celery task, return scan info."""
    if not text:
        text = await asyncio.to_thread(_read_document_text_sync, document_id)
    if not text:
        raise ValueError("Document text not found or empty")

    text_hash = hashlib.sha256(text.encode()).hexdigest()
    word_count = len(text.split())
    job_id = uuid.uuid4()

    async with async_session() as session:
        job = ScanJob(
            id=job_id,
            user_id=uuid.UUID(user_id) if user_id else None,
            input_text_hash=text_hash,
            word_count=word_count,
            scan_type="scan",
            status="pending",
        )
        session.add(job)

        # Reserve tokens based on word count
        cost = _scan_cost(word_count)
        if user_id:
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


async def list_scans(user_id: str, page: int = 1, per_page: int = 10) -> dict:
    """List scan_jobs for a user with pagination, newest first."""
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    offset = (page - 1) * per_page
    uid = uuid.UUID(user_id)

    async with async_session() as session:
        from sqlalchemy import func

        # Only run stale recovery if user has active jobs
        active_count = await session.scalar(
            select(func.count()).select_from(ScanJob)
            .where(ScanJob.user_id == uid)
            .where(ScanJob.status.in_(["processing", "pending"]))
        )
        if active_count and active_count > 0:
            cutoff = datetime.now(timezone.utc) - _STALE_THRESHOLD
            await session.execute(
                update(ScanJob)
                .where(ScanJob.user_id == uid)
                .where(ScanJob.status.in_(["processing", "pending"]))
                .where(ScanJob.created_at < cutoff)
                .values(status="failed", progress_message="Scan timed out")
            )
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


_STALE_THRESHOLD = timedelta(minutes=10)


async def _mark_stale_jobs_failed(user_id: uuid.UUID | None = None) -> None:
    """Bulk-mark processing/pending jobs older than threshold as failed."""
    cutoff = datetime.now(timezone.utc) - _STALE_THRESHOLD
    async with async_session() as session:
        q = (
            update(ScanJob)
            .where(ScanJob.status.in_(["processing", "pending"]))
            .where(ScanJob.created_at < cutoff)
            .values(status="failed", progress_message="Scan timed out")
        )
        if user_id:
            q = q.where(ScanJob.user_id == user_id)
        await session.execute(q)
        await session.commit()


async def get_scan(scan_id: str, user_id: str | None = None) -> dict | None:
    """Look up scan_job by ID, optionally scoped to a user.

    Auto-marks jobs stuck in 'processing' or 'pending' for >10 min as 'failed'.
    """
    async with async_session() as session:
        q = select(ScanJob).where(ScanJob.id == uuid.UUID(scan_id))
        if user_id:
            q = q.where(ScanJob.user_id == uuid.UUID(user_id))
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return None

        # Auto-recover stale processing/pending jobs
        if job.status in ("processing", "pending") and job.created_at:
            age = datetime.now(timezone.utc) - job.created_at
            if age > _STALE_THRESHOLD:
                job.status = "failed"
                job.progress_message = "Scan timed out"
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
