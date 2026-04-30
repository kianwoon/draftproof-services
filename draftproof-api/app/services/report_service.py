"""Report service — fetches completed scan results."""

import uuid
import httpx

from sqlalchemy import select

from app.models.db import async_session, ScanJob


def _flatten_findings(results_json: dict) -> list[dict]:
    """Convert tiered findings dict into flat issues list for the frontend."""
    issues = []
    findings_by_tier = results_json.get("findings", {})
    for severity in ("critical", "high", "medium", "low"):
        for f in findings_by_tier.get(severity, []):
            issues.append({
                "id": f.get("finding_id", str(len(issues) + 1)),
                "severity": severity,
                "title": f.get("title", ""),
                "description": f.get("detail") or f.get("title", ""),
                "location": f.get("sentence_id"),
                "scanner": f.get("scanner", ""),
                "category": f.get("category", ""),
                "signal_category": f.get("signal_category"),
                "score": f.get("score"),
                "top10_ratio": f.get("top10_ratio"),
                "raw_risk": f.get("raw_risk", ""),
                "adjusted_risk": f.get("adjusted_risk", ""),
                "actionability": f.get("actionability", ""),
                "evidence": f.get("evidence"),
                "recommendation": f.get("recommendation", ""),
                "adjustment": f.get("adjustment"),
            })
    return issues


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
        results_json = None
        issues = []
        json_url = report_urls.get("json")
        if json_url:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(json_url)
                    if resp.status_code == 200:
                        results_json = resp.json()
                        issues = _flatten_findings(results_json)
            except Exception:
                pass

        return {
            "id": str(job.id),
            "document_name": f"scan_{str(job.id)[:8]}",
            "issues": issues,
            "created_at": job.completed_at or job.created_at,
            "tier": job.tier,
            "report_md_url": report_urls.get("md"),
            "report_pdf_url": report_urls.get("pdf"),
            "results_json": results_json,
        }
