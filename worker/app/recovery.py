"""Startup reconciler — recover rewrite jobs orphaned by a killed worker.

When a worker is killed mid-rewrite (deploy/restart, OOM, crash), its job row is
left in ``processing`` and the in-flight Celery message may never be cleanly
redelivered. On worker boot this sweep finds those rows and resolves each one,
mirroring the API-side recovery (``rewrite_service._processing_rewrite_is_stale``
+ ``_mark_rewrite_interrupted``) so behaviour is identical wherever recovery runs:

* a job whose delivered rewrite report is already in R2 -> ``completed`` (+capture)
* otherwise -> ``failed`` with a retry message (+release of reserved credits)

Staleness is decided by the Redis progress stream acting as a heartbeat, so a
rewrite still running on a briefly-overlapping worker during a rolling deploy is
never killed. Credit capture/release are idempotent (atomic CAS in db.py), so a
race with the original task or the API recovery cannot double-charge.

Scans are intentionally out of scope: they are short and the rewrite task itself
resumes from its R2 checkpoint on redelivery.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

from . import db, progress
from .config import settings

logger = logging.getLogger(__name__)

_DELIVERED_STATUSES = {"rewrite_candidate_generated_needs_external_review", "ai_mitigated"}


def _enabled() -> bool:
    return os.getenv("DRAFTPROOF_STARTUP_RECONCILER_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _stale_threshold() -> timedelta:
    minutes = int(os.getenv("REWRITE_STALE_THRESHOLD_MINUTES", "50") or "50")
    return timedelta(minutes=max(1, minutes))


def _heartbeat_stale_threshold() -> timedelta:
    minutes = int(os.getenv("REWRITE_HEARTBEAT_STALE_MINUTES", "5") or "5")
    return timedelta(minutes=max(1, minutes))


def _stream_event_time(event_id) -> datetime | None:
    """Parse a Redis stream ID (``<millis>-<seq>``) into a UTC datetime."""
    try:
        millis = int(str(event_id).split("-", 1)[0])
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)


def _latest_heartbeat(rewrite_id: str) -> datetime | None:
    """Newest progress-stream event time, or None if missing/unavailable.

    None collapses both 'no stream' and 'Redis down' into the conservative
    wall-clock fallback, exactly like the API recovery path.
    """
    try:
        client = progress._redis_client()
        entries = client.xrevrange(progress.rewrite_progress_key(rewrite_id), count=1)
    except Exception:
        return None
    if not entries:
        return None
    return _stream_event_time(entries[0][0])


def _rewrite_report_has_delivered_content(report_json) -> bool:
    """Ported from rewrite_service so worker recovery matches API recovery."""
    if not isinstance(report_json, dict):
        return False
    status = str(report_json.get("status") or "").strip()
    summary = report_json.get("summary") if isinstance(report_json.get("summary"), dict) else {}
    outcome = str(summary.get("outcome") or summary.get("public_status") or "").strip()
    final_text = " ".join(str(report_json.get("final_text") or summary.get("final_text") or "").split())
    original_text = " ".join(str(report_json.get("original_text") or "").split())
    if not final_text:
        return False
    if original_text and final_text == original_text:
        return False
    return status in _DELIVERED_STATUSES or outcome in _DELIVERED_STATUSES


def _processing_is_stale(created_at, rewrite_id: str, now: datetime) -> bool:
    if not created_at:
        return False
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    event_time = _latest_heartbeat(rewrite_id)
    if event_time is None:
        return now - created_at > _stale_threshold()
    return now - event_time > _heartbeat_stale_threshold()


def _fetch_rewrite_report_json(scan_id: str) -> dict | None:
    try:
        from .storage import _client as r2_client

        s3 = r2_client()
        obj = s3.get_object(
            Bucket=settings.R2_BUCKET_NAME,
            Key=f"reports/{scan_id}/rewrite/rewrite.json",
        )
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def reconcile_interrupted_rewrites(now: datetime | None = None) -> dict:
    """Resolve rewrite jobs orphaned in 'processing'. Best-effort; never raises."""
    stats = {"scanned": 0, "completed": 0, "failed": 0, "skipped": 0}
    if not _enabled():
        logger.info("Startup reconciler disabled (DRAFTPROOF_STARTUP_RECONCILER_ENABLED=0)")
        return stats

    now = now or datetime.now(timezone.utc)
    try:
        jobs = db.list_processing_rewrite_jobs()
    except Exception:
        logger.warning("Startup reconciler: could not list processing rewrite jobs", exc_info=True)
        return stats

    stats["scanned"] = len(jobs)
    for job in jobs:
        rewrite_id = str(job["id"])
        scan_id = str(job["scan_id"])
        user_id = job.get("user_id")
        try:
            if not _processing_is_stale(job.get("created_at"), rewrite_id, now):
                stats["skipped"] += 1
                continue

            report = _fetch_rewrite_report_json(scan_id)
            if _rewrite_report_has_delivered_content(report):
                db.update_rewrite_status(
                    rewrite_id,
                    "completed",
                    progress_percent=100,
                    progress_message="Rewrite recovered from saved checkpoint",
                )
                db.capture_rewrite_credits(str(user_id) if user_id else "", rewrite_id)
                stats["completed"] += 1
                logger.info("Startup reconciler: recovered rewrite %s from checkpoint", rewrite_id)
            else:
                db.update_rewrite_status(
                    rewrite_id,
                    "failed",
                    error="Rewrite interrupted during worker restart",
                    progress_message="Rewrite worker restarted. Please retry.",
                )
                db.release_rewrite_credits(rewrite_id)
                stats["failed"] += 1
                logger.info("Startup reconciler: failed interrupted rewrite %s", rewrite_id)
        except Exception:
            logger.warning("Startup reconciler: error resolving rewrite %s", rewrite_id, exc_info=True)

    logger.info("Startup reconciler complete: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# Scan reconciler — mirror of the rewrite reconciler above.
#
# Scans do NOT resume from an R2 checkpoint (the rewrite rationale for leaving them
# out of scope), so a SIGKILL'd scan whose Celery message is never cleanly redelivered
# would otherwise sit 'processing' with its reservation 'active' forever (finding L4).
# Resolve them on boot, mirroring the API-side scan recovery (scan_service.
# _processing_scan_is_stale + _mark_scan_interrupted): R2-delivered -> completed+capture,
# else failed+release. Staleness is measured from started_at (the worker task-start), not
# enqueue time, so a scan that merely sat queued is never reaped (finding M1).
# ---------------------------------------------------------------------------

def _scan_stale_threshold() -> timedelta:
    return timedelta(minutes=max(1, int(os.getenv("SCAN_STALE_THRESHOLD_MINUTES", "10") or "10")))


def _scan_heartbeat_stale_threshold() -> timedelta:
    return timedelta(minutes=max(1, int(os.getenv("SCAN_HEARTBEAT_STALE_MINUTES", "8") or "8")))


def _latest_scan_heartbeat(scan_id: str) -> datetime | None:
    try:
        client = progress._redis_client()
        entries = client.xrevrange(progress.scan_progress_key(scan_id), count=1)
    except Exception:
        return None
    if not entries:
        return None
    return _stream_event_time(entries[0][0])


def _scan_processing_is_stale(started_at, scan_id: str, now: datetime) -> bool:
    if not started_at:
        return False
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    event_time = _latest_scan_heartbeat(scan_id)
    if event_time is not None:
        return now - event_time > _scan_heartbeat_stale_threshold()
    return now - started_at > _scan_stale_threshold()


def _fetch_scan_report_json(scan_id: str) -> dict | None:
    try:
        from .storage import _client as r2_client

        s3 = r2_client()
        obj = s3.get_object(
            Bucket=settings.R2_BUCKET_NAME,
            Key=f"reports/{scan_id}/report.json",
        )
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def reconcile_interrupted_scans(now: datetime | None = None) -> dict:
    """Resolve scan jobs orphaned in 'processing'. Best-effort; never raises."""
    stats = {"scanned": 0, "completed": 0, "failed": 0, "skipped": 0}
    if not _enabled():
        return stats

    now = now or datetime.now(timezone.utc)
    try:
        jobs = db.list_processing_scan_jobs()
    except Exception:
        logger.warning("Startup reconciler: could not list processing scan jobs", exc_info=True)
        return stats

    stats["scanned"] = len(jobs)
    for job in jobs:
        scan_id = str(job["id"])
        user_id = job.get("user_id")
        try:
            if not _scan_processing_is_stale(job.get("started_at"), scan_id, now):
                stats["skipped"] += 1
                continue

            report = _fetch_scan_report_json(scan_id)
            if isinstance(report, dict) and report:
                badge = report.get("ai_risk_badge") or {}
                db.update_job_status(
                    scan_id,
                    "completed",
                    tier=badge.get("tier"),
                    ai_score=badge.get("ai_likelihood_score"),
                    writing_score=badge.get("writing_quality_score"),
                    progress_percent=100,
                    progress_message="Scan recovered from saved report",
                )
                db.capture_credits(str(user_id) if user_id else "", scan_id, 0)
                stats["completed"] += 1
                logger.info("Startup reconciler: recovered scan %s from saved report", scan_id)
            else:
                db.update_job_status(
                    scan_id,
                    "failed",
                    error="Scan interrupted during worker restart",
                    progress_message="Scan worker restarted. Please retry.",
                )
                db.release_scan_credits(scan_id)
                db.refund_free_scan(scan_id)
                stats["failed"] += 1
                logger.info("Startup reconciler: failed interrupted scan %s", scan_id)
        except Exception:
            logger.warning("Startup reconciler: error resolving scan %s", scan_id, exc_info=True)

    logger.info("Startup scan reconciler complete: %s", stats)
    return stats
