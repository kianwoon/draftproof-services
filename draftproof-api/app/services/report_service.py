"""Report service — fetches completed scan results."""

import json
import logging
import uuid

import boto3
from botocore.config import Config as BotoConfig
from sqlalchemy import select

from app.config import R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
from app.models.db import async_session, ScanJob

logger = logging.getLogger("report_service")


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
    )


def _presign(key: str, expires: int = 3600) -> str:
    s3 = _r2_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": key},
        ExpiresIn=expires,
    )


def _flatten_findings(results_json: dict) -> list[dict]:
    """Convert tiered findings dict into flat issues list for the frontend."""
    # Build sentence_id -> full sentence text lookup
    sentence_map = {}
    pred = results_json.get("predictability", {})
    input_text = results_json.get("input_text", "")
    for s in pred.get("sentences", []):
        sid = s.get("sentence_id", "")
        # Use full text from input_text via char offsets if available
        start = s.get("start_char")
        end = s.get("end_char")
        if sid and start is not None and end is not None and input_text:
            sentence_map[sid] = input_text[start:end]
        elif sid:
            sentence_map[sid] = s.get("text", s.get("sentence", ""))

    issues = []
    findings_by_tier = results_json.get("findings", {})
    for severity in ("critical", "high", "medium", "low"):
        for f in findings_by_tier.get(severity, []):
            sentence_id = f.get("sentence_id", "")
            sentence_text = sentence_map.get(sentence_id, "")

            # Normalize evidence: can be string, dict, or None
            raw_evidence = f.get("evidence")
            evidence = None
            if isinstance(raw_evidence, dict):
                evidence = raw_evidence
            elif isinstance(raw_evidence, str) and raw_evidence:
                evidence = {"summary": raw_evidence}
            # Attach sentence text to evidence
            if evidence and sentence_text and not evidence.get("sentence"):
                evidence["sentence"] = sentence_text

            # Sanitize legacy "top-10" labels in description
            description = f.get("detail") or f.get("title", "")
            description = description.replace("top-10 ratio", "common ratio").replace("Top-10 ratio", "Common ratio")

            issues.append({
                "id": f.get("finding_id", str(len(issues) + 1)),
                "severity": severity,
                "title": f.get("title", ""),
                "description": description,
                "location": sentence_id or None,
                "sentence_text": sentence_text or None,
                "scanner": f.get("scanner", ""),
                "category": f.get("category", ""),
                "signal_category": f.get("signal_category"),
                "score": f.get("score"),
                "top10_ratio": f.get("top10_ratio"),
                "raw_risk": f.get("raw_risk", ""),
                "adjusted_risk": f.get("adjusted_risk", ""),
                "actionability": f.get("actionability", ""),
                "evidence": evidence,
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

        results_json = None
        issues = []

        # Fetch JSON directly from R2 (avoids stale presigned URLs)
        r2_key = f"reports/{report_id}/report.json"
        try:
            s3 = _r2_client()
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
            results_json = json.loads(obj["Body"].read())
            issues = _flatten_findings(results_json)
        except Exception as e:
            logger.warning("Failed to fetch report JSON from R2 for %s: %s", report_id, e)

        # Generate fresh presigned URLs for downloads
        report_md_url = None
        report_pdf_url = None
        try:
            report_md_url = _presign(f"reports/{report_id}/report.md")
            report_pdf_url = _presign(f"reports/{report_id}/report.pdf")
        except Exception:
            pass

        return {
            "id": str(job.id),
            "document_name": f"scan_{str(job.id)[:8]}",
            "issues": issues,
            "created_at": job.completed_at or job.created_at,
            "tier": job.tier,
            "report_md_url": report_md_url,
            "report_pdf_url": report_pdf_url,
            "results_json": results_json,
        }
