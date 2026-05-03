"""Rewrite service — create rewrite jobs, fetch results from R2."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from app.config import REWRITE_TOKEN_COST
from app.models.db import async_session, RewriteJob, ScanJob, CreditAccount, CreditReservation

logger = logging.getLogger("rewrite_service")

_STALE_THRESHOLD = timedelta(minutes=5)


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
            )
        )
        scan = result.scalar_one_or_none()
        if not scan:
            raise ValueError("Completed scan not found")

        # Mark stale pending/processing rewrites as failed (older than threshold)
        stale_cutoff = datetime.now(timezone.utc) - _STALE_THRESHOLD
        stale = await session.execute(
            select(RewriteJob).where(
                RewriteJob.scan_id == scan_uuid,
                RewriteJob.status.in_(["pending", "processing"]),
                RewriteJob.created_at < stale_cutoff,
            )
        )
        for stale_job in stale.scalars().all():
            stale_job.status = "failed"
            stale_job.error = "Stale rewrite — timed out"
            stale_job.completed_at = datetime.now(timezone.utc)

        # Check for actively running rewrites (recent, not stale)
        existing = await session.execute(
            select(RewriteJob).where(
                RewriteJob.scan_id == scan_uuid,
                RewriteJob.status.in_(["pending", "processing"]),
            )
        )
        existing_job = existing.scalar_one_or_none()
        if existing_job:
            # Return existing rewrite instead of erroring — allows frontend to resume polling
            return {
                "id": str(existing_job.id),
                "scan_id": str(existing_job.scan_id),
                "status": existing_job.status,
                "error": existing_job.error,
                "created_at": existing_job.created_at.isoformat() if existing_job.created_at else None,
                "completed_at": existing_job.completed_at.isoformat() if existing_job.completed_at else None,
            }

        acct_result = await session.execute(
            select(CreditAccount).where(CreditAccount.user_id == uid).with_for_update()
        )
        acct = acct_result.scalar_one_or_none()
        if not acct:
            raise ValueError("No credit account found")
        if acct.balance_tokens - acct.reserved_tokens < REWRITE_TOKEN_COST:
            raise ValueError(f"Insufficient tokens (need {REWRITE_TOKEN_COST})")

        acct.reserved_tokens += REWRITE_TOKEN_COST
        job_id = uuid.uuid4()
        rewrite_job = RewriteJob(
            id=job_id,
            scan_id=scan_uuid,
            user_id=uid,
            status="pending",
        )
        session.add(rewrite_job)

        reservation = CreditReservation(
            user_id=uid,
            credit_account_id=acct.id,
            job_type="rewrite",
            job_id=job_id,
            tokens_reserved=REWRITE_TOKEN_COST,
            status="active",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        session.add(reservation)
        await session.commit()

    from app.services.celery_client import run_rewrite
    run_rewrite.delay(str(job_id), scan_id)

    return {"id": str(job_id), "scan_id": scan_id, "status": "pending"}


async def get_rewrite(rewrite_id: str, user_id: str | None = None) -> dict | None:
    """Get rewrite job status. Auto-fails stale jobs."""
    async with async_session() as session:
        q = select(RewriteJob).where(RewriteJob.id == uuid.UUID(rewrite_id))
        if user_id:
            q = q.where(RewriteJob.user_id == uuid.UUID(user_id))
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return None

        if job.status in ("pending", "processing") and job.created_at:
            age = datetime.now(timezone.utc) - job.created_at
            if age > _STALE_THRESHOLD:
                job.status = "failed"
                job.error = "Rewrite timed out"
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
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
