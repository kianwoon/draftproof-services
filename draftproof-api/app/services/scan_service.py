"""Scan service — creates scan jobs and dispatches to Celery worker."""

import asyncio
import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

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
                select(CreditAccount).where(CreditAccount.user_id == uid)
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
                expires_at=datetime.utcnow() + timedelta(minutes=30),
            )
            session.add(reservation)

        await session.commit()

    from app.services.celery_client import scan_document
    scan_document.delay(str(job_id), text)

    return {
        "id": str(job_id),
        "document_id": document_id,
        "status": "pending",
        "report_id": None,
    }


async def list_scans(user_id: str, page: int = 1, per_page: int = 10) -> dict:
    """List scan_jobs for a user with pagination, newest first."""
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    offset = (page - 1) * per_page

    async with async_session() as session:
        # Count total
        from sqlalchemy import func
        count_result = await session.execute(
            select(func.count()).select_from(ScanJob)
            .where(ScanJob.user_id == uuid.UUID(user_id))
        )
        total = count_result.scalar() or 0

        result = await session.execute(
            select(ScanJob)
            .where(ScanJob.user_id == uuid.UUID(user_id))
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
                    "finding_count": j.finding_count,
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


async def get_scan(scan_id: str, user_id: str | None = None) -> dict | None:
    """Look up scan_job by ID, optionally scoped to a user."""
    async with async_session() as session:
        q = select(ScanJob).where(ScanJob.id == uuid.UUID(scan_id))
        if user_id:
            q = q.where(ScanJob.user_id == uuid.UUID(user_id))
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return None
        return {
            "id": str(job.id),
            "document_id": "",
            "status": job.status,
            "report_id": str(job.id) if job.status == "completed" else None,
            "tier": job.tier,
            "finding_count": job.finding_count,
        }
