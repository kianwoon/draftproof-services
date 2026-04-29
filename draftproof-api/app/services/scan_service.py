"""Scan service — creates scan jobs and dispatches to Celery worker."""

import hashlib
import os
import uuid

from sqlalchemy import select

from app.config import UPLOAD_DIR
from app.models.db import async_session, ScanJob


def _read_document_text(document_id: str) -> str:
    """Read uploaded document text from disk."""
    for ext in (".txt", ".pdf", ".docx"):
        path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
    return ""


async def create_scan(document_id: str, user_id: str | None = None) -> dict:
    """Create a scan_job row, enqueue Celery task, return scan info."""
    text = _read_document_text(document_id)
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
        await session.commit()

    from app.services.celery_client import scan_document
    scan_document.delay(str(job_id), text)

    return {
        "id": str(job_id),
        "document_id": document_id,
        "status": "pending",
        "report_id": None,
    }


async def list_scans(user_id: str) -> list[dict]:
    """List all scan_jobs for a user, newest first."""
    async with async_session() as session:
        result = await session.execute(
            select(ScanJob)
            .where(ScanJob.user_id == uuid.UUID(user_id))
            .order_by(ScanJob.created_at.desc())
        )
        jobs = result.scalars().all()
        return [
            {
                "id": str(j.id),
                "status": j.status,
                "tier": j.tier,
                "finding_count": j.finding_count,
                "word_count": j.word_count,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ]


async def get_scan(scan_id: str) -> dict | None:
    """Look up scan_job by ID."""
    async with async_session() as session:
        result = await session.execute(
            select(ScanJob).where(ScanJob.id == uuid.UUID(scan_id))
        )
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
