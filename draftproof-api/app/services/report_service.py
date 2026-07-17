"""Report service — fetches completed scan results."""

import asyncio
import json
import logging
import os
import uuid

import boto3
from botocore.config import Config as BotoConfig
from sqlalchemy import select

from app.config import R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
from app.models.db import async_session, ScanJob, RewriteJob
from app.services.rewrite_service import has_rewriteable_findings
from app.services.scan_service import _rewrite_cost

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


def _fetch_optional_report_json_sync(r2_key: str) -> dict | None:
    try:
        return _fetch_report_json_sync(r2_key)
    except Exception:
        return None


# Rewrite-pipeline scaffolding the report PAGE never reads (verified zero frontend refs).
# The worker reads rewrite inputs straight from R2, so trimming them from the API response is
# payload-only — measured ~330 KB (~30%) off a 1.16 MB report, dominated by the 186 KB
# scan_intelligence.mitigation_inputs. Kill switch: DRAFTPROOF_REPORT_PAYLOAD_TRIM=0.
_REPORT_TRIM_ENABLED = os.getenv("DRAFTPROOF_REPORT_PAYLOAD_TRIM", "1") == "1"
_TRIM_TOP_LEVEL_KEYS = frozenset({
    "rewrite_edit_briefs",
    "ai_mitigation",
    "repair_units_v2",
    "rewrite_constraints",
    "generation_handoff",
})
_TRIM_SCAN_INTELLIGENCE_KEYS = frozenset({
    "mitigation_inputs",
})


def _strip_rewrite_only_fields(results_json: dict | None) -> None:
    """Drop rewrite-only fields from the report-page response, in place."""
    if not _REPORT_TRIM_ENABLED or not isinstance(results_json, dict):
        return
    for key in _TRIM_TOP_LEVEL_KEYS:
        results_json.pop(key, None)
    intel = results_json.get("scan_intelligence")
    if isinstance(intel, dict):
        for key in _TRIM_SCAN_INTELLIGENCE_KEYS:
            intel.pop(key, None)


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


async def get_report(report_id: str, user_id: str | None = None) -> dict | None:
    """Fetch a completed scan report by scan job ID, optionally scoped to a user."""
    async with async_session() as session:
        try:
            report_uuid = uuid.UUID(report_id)
        except (ValueError, TypeError):
            return None  # malformed id -> not found (404), not a 500 (L13)
        q = select(ScanJob).where(ScanJob.id == report_uuid)
        if user_id:
            q = q.where(ScanJob.user_id == uuid.UUID(user_id))
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job or job.status != "completed":
            return None

        results_json = None
        issues = []
        report_md_url = None
        report_pdf_url = None

        # Fetch report JSON + paragraph explanations + fresh download presigns concurrently.
        # These are independent R2 round-trips; serializing them was the dominant read latency
        # (a single report.json read alone measured ~700 ms). Presigns avoid stale URLs.
        if _r2:
            async def _load_report_json():
                try:
                    return await asyncio.to_thread(
                        _fetch_report_json_sync, f"reports/{report_id}/report.json"
                    )
                except Exception as e:
                    logger.warning("Failed to fetch report JSON from R2 for %s: %s", report_id, e)
                    return None

            async def _presign_or_none(key: str):
                try:
                    return await asyncio.to_thread(_presign_sync, key)
                except Exception:
                    return None

            results_json, paragraph_explanations, report_md_url, report_pdf_url = await asyncio.gather(
                _load_report_json(),
                asyncio.to_thread(
                    _fetch_optional_report_json_sync,
                    f"reports/{report_id}/paragraph_explanations.json",
                ),
                _presign_or_none(f"reports/{report_id}/report.md"),
                _presign_or_none(f"reports/{report_id}/report.pdf"),
            )

            if results_json is not None:
                if paragraph_explanations:
                    results_json["paragraph_explanations"] = paragraph_explanations
                issues = _flatten_findings(results_json)

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

        # Read-time authenticity dashboard backfill: recompute with full predictability so the
        # page gets the rich CI (scan-time stored a degraded copy with predictability=None).
        if results_json:
            from app._composers.authenticity_dashboard import maybe_attach as _attach_dashboard
            _badge = results_json.get("ai_risk_badge")
            if isinstance(_badge, dict):
                _dash = _attach_dashboard(_badge, predictability=results_json.get("predictability"))
                if _dash is not None:
                    _badge["authenticity_dashboard"] = _dash

        # Prefer AI badge tier over findings-based tier (matches PDF)
        display_tier = ai_badge_tier or job.tier

        # Check for the latest rewrite so the report page can show active progress
        rewrite_info = None
        rw_result = await session.execute(
            select(RewriteJob).where(
                RewriteJob.scan_id == job.id,
            ).order_by(RewriteJob.created_at.desc()).limit(1)
        )
        rw_job = rw_result.scalar_one_or_none()
        if rw_job:
            rewrite_info = {
                "id": str(rw_job.id),
                "status": rw_job.status,
                "error": rw_job.error,
                "progress_percent": rw_job.progress_percent,
                "progress_message": _normalize_rewrite_progress_message(rw_job.progress_message),
                "created_at": rw_job.created_at.isoformat() if rw_job.created_at else None,
                "completed_at": rw_job.completed_at.isoformat() if rw_job.completed_at else None,
            }

        # Trim rewrite-only scaffolding from the page payload (after all internal uses above,
        # e.g. has_rewriteable_findings / dashboard, which read the full results_json).
        _strip_rewrite_only_fields(results_json)

        return {
            "id": str(job.id),
            "document_name": f"scan_{str(job.id)[:8]}",
            "issues": issues,
            "created_at": job.completed_at or job.created_at,
            "tier": display_tier,
            "ai_score": ai_score,
            "writing_score": writing_score,
            "word_count": job.word_count,
            "rewrite_token_cost": _rewrite_cost(job.word_count),
            "can_start_rewrite": has_rewriteable_findings(results_json or {}),
            "ai_risk_badge": results_json.get("ai_risk_badge") if results_json else None,
            "report_md_url": report_md_url,
            "report_pdf_url": report_pdf_url,
            "results_json": results_json,
            "rewrite": rewrite_info,
        }
