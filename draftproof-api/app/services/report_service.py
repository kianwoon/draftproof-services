"""Report service — fetches completed scan results."""

import uuid

from sqlalchemy import select

from app.models.db import async_session, ScanJob


async def get_report(report_id: str) -> dict | None:
    """Fetch a completed scan report by scan job ID."""
    async with async_session() as session:
        result = await session.execute(
            select(ScanJob).where(ScanJob.id == uuid.UUID(report_id))
        )
        job = result.scalar_one_or_none()
        if not job or job.status != "completed":
            return None

        report_urls = job.report_urls or {}
        return {
            "id": str(job.id),
            "document_name": f"scan_{str(job.id)[:8]}",
            "issues": [],
            "created_at": job.completed_at or job.created_at,
            "tier": job.tier,
            "report_md_url": report_urls.get("md"),
            "report_pdf_url": report_urls.get("pdf"),
        }
