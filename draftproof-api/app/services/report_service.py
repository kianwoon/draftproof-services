"""Report service — fetches completed scan results."""

import asyncio
import json
import logging
import uuid

import boto3
from botocore.config import Config as BotoConfig
from sqlalchemy import select

from app.config import R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
from app.models.db import async_session, ScanJob

logger = logging.getLogger("report_service")

_r2 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=BotoConfig(
        signature_version="s3v4",
        connect_timeout=5,
        read_timeout=15,
        retries={"max_attempts": 2},
    ),
) if R2_ENDPOINT_URL else None


def _presign_sync(key: str, expires: int = 3600) -> str:
    return _r2.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": key},
        ExpiresIn=expires,
    )


_MAX_REPORT_BYTES = 20 * 1024 * 1024  # 20 MB safety cap


def _fetch_report_json_sync(r2_key: str) -> dict | None:
    obj = _r2.get_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
    content_length = obj.get("ContentLength", 0)
    if content_length > _MAX_REPORT_BYTES:
        raise ValueError(f"Report too large ({content_length} bytes), max {_MAX_REPORT_BYTES}")
    return json.loads(obj["Body"].read(_MAX_REPORT_BYTES))


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


async def get_report(report_id: str, user_id: str | None = None) -> dict | None:
    """Fetch a completed scan report by scan job ID, optionally scoped to a user."""
    async with async_session() as session:
        q = select(ScanJob).where(ScanJob.id == uuid.UUID(report_id))
        if user_id:
            q = q.where(ScanJob.user_id == uuid.UUID(user_id))
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job or job.status != "completed":
            return None

        results_json = None
        issues = []

        # Fetch JSON directly from R2 (avoids stale presigned URLs)
        r2_key = f"reports/{report_id}/report.json"
        if _r2:
            try:
                results_json = await asyncio.to_thread(_fetch_report_json_sync, r2_key)
                issues = _flatten_findings(results_json)
            except Exception as e:
                logger.warning("Failed to fetch report JSON from R2 for %s: %s", report_id, e)

        # Generate fresh presigned URLs for downloads
        report_md_url = None
        report_pdf_url = None
        if _r2:
            try:
                report_md_url = await asyncio.to_thread(_presign_sync, f"reports/{report_id}/report.md")
                report_pdf_url = await asyncio.to_thread(_presign_sync, f"reports/{report_id}/report.pdf")
            except Exception:
                pass

        # Extract AI risk badge from results_json for display alignment
        ai_score = None
        writing_score = None
        ai_badge_tier = None
        if results_json:
            badge = results_json.get("ai_risk_badge")
            if badge:
                ai_score = badge.get("ai_likelihood_score")
                writing_score = badge.get("writing_quality_score")
                ai_badge_tier = badge.get("tier", "").lower()

        # Prefer AI badge tier over findings-based tier (matches PDF)
        display_tier = ai_badge_tier or job.tier

        return {
            "id": str(job.id),
            "document_name": f"scan_{str(job.id)[:8]}",
            "issues": issues,
            "created_at": job.completed_at or job.created_at,
            "tier": display_tier,
            "ai_score": ai_score,
            "writing_score": writing_score,
            "report_md_url": report_md_url,
            "report_pdf_url": report_pdf_url,
            "results_json": results_json,
        }
