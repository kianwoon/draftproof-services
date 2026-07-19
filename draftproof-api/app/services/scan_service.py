"""Scan service — creates scan jobs and dispatches to Celery worker."""

import asyncio
import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text as sql_text

from app.config import UPLOAD_DIR
from app.models.db import async_session, ScanJob, CreditAccount, CreditReservation
from app.services import progress_stream


DOCUMENT_TITLE_MAX_CHARS = 90
CONTENT_PREVIEW_MAX_CHARS = 220
def _stale_minutes(env_name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(env_name, str(default)) or default))
    except (TypeError, ValueError):
        return default


# Hard wall-clock cap, measured from the worker's started_at (NOT enqueue created_at).
_STALE_THRESHOLD = timedelta(minutes=_stale_minutes("SCAN_STALE_THRESHOLD_MINUTES", 10))
# Heartbeat-gap cap. MUST exceed the worst-case SILENT tail: the paragraph-explainer LLM
# call can stack provider retries (~3x90s + backoff ~= 4.5min) between the 96% and 97%
# heartbeats, so a 5-min cap would reap a live worker mid-completion (findings H1/M1).
_PROCESSING_HEARTBEAT_STALE_THRESHOLD = timedelta(
    minutes=_stale_minutes("SCAN_HEARTBEAT_STALE_MINUTES", 8)
)
_ACTIVE_SCAN_STATUSES = ("pending", "processing", "retrying")
_PROCESSING_SCAN_STATUSES = ("processing",)
_MEANINGFUL_TEXT_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def _compact_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _truncate_display_text(value: str, limit: int) -> str:
    compact = _compact_text(value)
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _first_meaningful_segment(text: str) -> str:
    for raw_line in str(text or "").splitlines():
        line = _compact_text(raw_line)
        if _MEANINGFUL_TEXT_RE.search(line):
            sentence_match = re.match(r"^(.+?[.!?。！？])(?:\s|$)", line)
            return sentence_match.group(1) if sentence_match else line
    compact = _compact_text(text)
    return compact if _MEANINGFUL_TEXT_RE.search(compact) else ""


def build_scan_report_metadata(text: str) -> dict:
    title_source = _first_meaningful_segment(text)
    preview_source = _compact_text(text)
    return {
        "document_title": _truncate_display_text(title_source, DOCUMENT_TITLE_MAX_CHARS) or None,
        "content_preview": _truncate_display_text(preview_source, CONTENT_PREVIEW_MAX_CHARS) or None,
    }


def _paid_scan_cost(word_count: int) -> int:
    """Credit cost for a scan: 1 token per started 1,000 words (min 1).

    A scan up to 1,000 words costs 1 credit; 1,001-2,000 costs 2; and so on.
    Every scan is billed at this rate — new users instead receive a one-time
    welcome grant of credits at signup (see config.WELCOME_CREDITS).
    """
    return max(1, -(-word_count // 1000))


async def _refund_free_scan(session, job: ScanJob) -> None:
    """Refund a legacy free scan's durable counter exactly once on failure/cancel.

    The per-scan free allowance was retired in favour of a signup welcome grant,
    so new scans are never flagged free_scan_counted and this is a safe no-op for
    them. It is retained to correctly settle any free scan still in flight across
    the deploy: the CAS on scan_jobs.free_scan_counted guarantees the decrement
    fires at most once even if multiple recovery paths run.
    """
    if job.user_id is None:
        return
    flipped = await session.execute(
        sql_text(
            "UPDATE scan_jobs SET free_scan_counted = FALSE "
            "WHERE id = :jid AND free_scan_counted = TRUE "
            "RETURNING user_id"
        ),
        {"jid": job.id},
    )
    row = flipped.first()
    if row is None:
        return
    await session.execute(
        sql_text(
            "UPDATE users SET free_scans_used = GREATEST(free_scans_used - 1, 0) "
            "WHERE id = :uid"
        ),
        {"uid": row[0]},
    )


def _rewrite_cost(word_count: int) -> int:
    """5 tokens per 1,000 words (ceiling). 1-1000 = 5, 1001-2000 = 10, etc."""
    return max(5, -(-word_count // 1000) * 5)


def _read_document_text_sync(document_id: str) -> str:
    """Read uploaded document text from disk (sync — call via to_thread)."""
    # document_id is attacker-controllable (ScanRequest.document_id) and is
    # interpolated straight into a filesystem path, so anything other than a UUID
    # (e.g. "../../../../etc/passwd") is a path-traversal / arbitrary-file-read
    # vector — the file contents get echoed back into the scan report. Documents
    # are always created with uuid4 ids; a string that parses as a UUID cannot
    # contain "/" or "..", so the join is guaranteed to stay inside UPLOAD_DIR.
    try:
        uuid.UUID(str(document_id))
    except (ValueError, TypeError):
        return ""
    for ext in (".txt", ".pdf", ".docx"):
        path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
    return ""


def _redis_stream_event_time(event_id: str) -> datetime | None:
    try:
        millis = int(str(event_id).split("-", 1)[0])
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)


async def _processing_scan_is_stale(job: ScanJob, *, now: datetime | None = None) -> bool:
    """Detect a processing scan whose worker heartbeat has stopped.

    Redis progress events are the lightweight worker heartbeat. If Redis is
    unavailable or the stream is missing, keep the existing hard stale timeout
    so live scans are not killed incorrectly.
    """
    if job.status not in _PROCESSING_SCAN_STATUSES:
        return False
    # Measure from started_at (worker task-start), NOT created_at (enqueue): a scan can
    # sit queued for minutes behind a long rewrite, and aging it by enqueue time would
    # reap a job that has not started yet (finding M1). A processing scan always has
    # started_at (the worker sets it on its first 'processing' write); if it is somehow
    # missing, treat it as not-started -> NOT stale rather than falling back to created_at.
    started_at = job.started_at
    if started_at is None:
        return False
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    now = now or datetime.now(timezone.utc)

    # Heartbeat governs: a live worker republishes Redis progress, so a stopped heartbeat
    # is the real death signal. The wall-clock age (from started_at) is only the
    # conservative fallback when Redis is unavailable or the stream is missing.
    latest = await progress_stream.read_latest_scan_progress(str(job.id))
    if latest is not None:
        event_time = _redis_stream_event_time(latest[0])
        if event_time is not None:
            return now - event_time > _PROCESSING_HEARTBEAT_STALE_THRESHOLD

    return now - started_at > _STALE_THRESHOLD


async def _release_active_scan_reservations(session, job_id: uuid.UUID) -> int:
    """Release a scan's active reservation back to available balance — a real CAS.

    Mirrors the worker's release_scan_credits: a single guarded UPDATE flips only rows
    still 'active' and decrements reserved_tokens ONLY for the rows it actually changed,
    so a concurrent worker capture/release can never be overwritten and tokens are never
    double-decremented (finding M2). The prior ORM read-modify-write had no status='active'
    re-check at write time.
    """
    result = await session.execute(
        sql_text(
            """UPDATE credit_reservations
               SET status = 'released'
               WHERE job_id = :job_id AND job_type = 'scan' AND status = 'active'
               RETURNING credit_account_id, tokens_reserved"""
        ),
        {"job_id": job_id},
    )
    released_tokens = 0
    for credit_account_id, tokens_reserved in result.all():
        await session.execute(
            sql_text(
                "UPDATE credit_accounts "
                "SET reserved_tokens = GREATEST(reserved_tokens - :tok, 0) WHERE id = :acct"
            ),
            {"tok": tokens_reserved, "acct": credit_account_id},
        )
        released_tokens += tokens_reserved
    return released_tokens


async def _capture_active_scan_reservations(session, job: ScanJob) -> int:
    """Capture (charge) a recovered scan's active reservation — the API-side mirror of the
    worker's capture_credits. Used when stale-recovery finds the report already in R2 (the
    worker finished but was killed before its completion write). Real CAS: only an 'active'
    row flips to 'captured', so it cannot double-charge a concurrent worker capture (M3)."""
    result = await session.execute(
        sql_text(
            """UPDATE credit_reservations
               SET status = 'captured'
               WHERE job_id = :job_id AND job_type = 'scan' AND status = 'active'
               RETURNING credit_account_id, tokens_reserved"""
        ),
        {"job_id": job.id},
    )
    captured_tokens = 0
    for credit_account_id, tokens_reserved in result.all():
        await session.execute(
            sql_text(
                "UPDATE credit_accounts "
                "SET balance_tokens = balance_tokens - :tok, "
                "reserved_tokens = GREATEST(reserved_tokens - :tok, 0) WHERE id = :acct"
            ),
            {"tok": tokens_reserved, "acct": credit_account_id},
        )
        await session.execute(
            sql_text(
                """INSERT INTO usage_events
                   (user_id, credit_account_id, event_type, tokens_charged, job_id, word_count)
                   VALUES (:uid, :acct, 'scan', :tok, :job_id, :wc)"""
            ),
            {
                "uid": job.user_id,
                "acct": credit_account_id,
                "tok": tokens_reserved,
                "job_id": job.id,
                "wc": job.word_count or 0,
            },
        )
        captured_tokens += tokens_reserved
    return captured_tokens


async def _scan_report_in_r2(scan_id: uuid.UUID) -> bool:
    """True if a finished scan report already exists in R2 (the work completed before the
    worker died). Lets stale-recovery mark such a job 'completed' + capture instead of
    failing it for free (finding M3). Best-effort: any R2 error -> False (fail the job)."""
    from app.services import report_service
    try:
        report = await asyncio.to_thread(
            report_service._fetch_optional_report_json_sync, f"reports/{scan_id}/report.json"
        )
    except Exception:
        return False
    return isinstance(report, dict) and bool(report)


async def _mark_scan_interrupted(session, job: ScanJob) -> int:
    """Resolve a stale processing scan. If its report already reached R2 the work is done,
    so recover it as completed + capture credits (M3); otherwise fail it and release/refund.

    The R2 check is a single bounded GET per job (only stale jobs reach here). list_scans
    runs this in an un-locked session; get_scan holds a single-row FOR UPDATE lock across
    the one GET, which is acceptable for a single row.
    """
    if await _scan_report_in_r2(job.id):
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.progress_percent = 100
        job.progress_message = "Scan recovered from saved report"
        job.error = None
        return await _capture_active_scan_reservations(session, job)

    job.status = "failed"
    job.error = "Scan interrupted during worker restart"
    job.progress_message = "Scan worker restarted. Please retry."
    job.completed_at = datetime.now(timezone.utc)
    await _refund_free_scan(session, job)
    return await _release_active_scan_reservations(session, job.id)


async def create_scan(
    document_id: str,
    user_id: str | None = None,
    text: str | None = None,
    *,
    always_paid: bool = False,
    title: str | None = None,
) -> dict:
    """Create a scan_job row, enqueue Celery task, return scan info.

    Every scan is billed at the paid rate (>=1 credit). New users instead
    receive a one-time welcome grant of credits at signup. always_paid is kept
    for call-site compatibility (extension/API-key scans) but no longer changes
    the cost — all scans reserve >=1 credit via the same reserve→capture path.
    """
    if not text:
        text = await asyncio.to_thread(_read_document_text_sync, document_id)
    if not text:
        raise ValueError("Document text not found or empty")

    text_hash = hashlib.sha256(text.encode()).hexdigest()
    word_count = len(text.split())
    job_id = uuid.uuid4()
    report_metadata = build_scan_report_metadata(text)
    # An explicit title (e.g. the Word file name from the add-in) overrides the
    # text-derived one so reports are identifiable by their source document.
    document_title = (
        _truncate_display_text(title, DOCUMENT_TITLE_MAX_CHARS)
        if title and title.strip()
        else report_metadata["document_title"]
    )

    async with async_session() as session:
        # Every scan is billed at the paid rate (1 credit per started 1,000
        # words, min 1). New users instead start with a one-time welcome grant of
        # credits at signup, so there is no per-scan free path. always_paid is
        # retained for call-site compatibility but no longer changes the cost.
        cost = _paid_scan_cost(word_count)

        job = ScanJob(
            id=job_id,
            user_id=uuid.UUID(user_id) if user_id else None,
            input_text_hash=text_hash,
            word_count=word_count,
            document_title=document_title,
            content_preview=report_metadata["content_preview"],
            scan_type="scan",
            status="pending",
            free_scan_counted=False,
        )
        session.add(job)

        if user_id and cost > 0:
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

        # Only run stale recovery if user has active jobs.
        active_result = await session.execute(
            select(ScanJob)
            .where(ScanJob.user_id == uid)
            .where(ScanJob.status.in_(_ACTIVE_SCAN_STATUSES))
        )
        active_jobs = active_result.scalars().all()
        if active_jobs:
            now = datetime.now(timezone.utc)
            for active_job in active_jobs:
                if await _processing_scan_is_stale(active_job, now=now):
                    await _mark_scan_interrupted(session, active_job)
            await session.commit()

        # Select only the columns the response actually serializes (skips the
        # JSONB report_urls blob, input_text_hash, etc.) and fold the total
        # count into the same round trip via a window function instead of a
        # separate COUNT(*) query -- list_scans() is called on every
        # dashboard badge and reports-page load, so cutting one DB round
        # trip per call matters more than the query cost itself.
        list_cols = (
            ScanJob.id, ScanJob.status, ScanJob.document_title,
            ScanJob.content_preview, ScanJob.tier, ScanJob.ai_score,
            ScanJob.writing_score, ScanJob.finding_count,
            ScanJob.progress_percent, ScanJob.progress_message,
            ScanJob.word_count, ScanJob.created_at, ScanJob.completed_at,
        )
        result = await session.execute(
            select(*list_cols, func.count().over().label("total_count"))
            .where(ScanJob.user_id == uid)
            .order_by(ScanJob.created_at.desc())
            .offset(offset)
            .limit(per_page)
        )
        rows = result.all()

        if rows:
            total = rows[0].total_count
        else:
            # Requested page is past the end (e.g. a stale page param) -- the
            # window function has no rows to count from, so fall back to a
            # real COUNT(*) to report the true total.
            count_result = await session.execute(
                select(func.count()).select_from(ScanJob).where(ScanJob.user_id == uid)
            )
            total = count_result.scalar() or 0

        return {
            "items": [
                {
                    "id": str(r.id),
                    "document_id": "",
                    "status": r.status,
                    "report_id": str(r.id) if r.status == "completed" else None,
                    "document_title": r.document_title,
                    "content_preview": r.content_preview,
                    "tier": r.tier,
                    "ai_score": float(r.ai_score) if r.ai_score is not None else None,
                    "writing_score": float(r.writing_score) if r.writing_score is not None else None,
                    "finding_count": r.finding_count,
                    "progress_percent": r.progress_percent,
                    "progress_message": r.progress_message,
                    "word_count": r.word_count,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in rows
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page,
        }

async def _mark_stale_jobs_failed(user_id: uuid.UUID | None = None) -> None:
    """Bulk-mark processing/pending jobs older than threshold as failed.

    Also releases any active credit reservations for those jobs so tokens
    are not permanently locked.
    """
    import logging
    log = logging.getLogger("scan_service.stale")
    cutoff = datetime.now(timezone.utc) - _STALE_THRESHOLD
    async with async_session() as session:
        # Find stale jobs first
        q = select(ScanJob).where(
            ScanJob.status.in_(_ACTIVE_SCAN_STATUSES),
            ScanJob.created_at < cutoff,
        )
        if user_id:
            q = q.where(ScanJob.user_id == user_id)
        result = await session.execute(q)
        stale_jobs = result.scalars().all()

        if not stale_jobs:
            return

        released_count = 0
        for stale_job in stale_jobs:
            released_count += await _mark_scan_interrupted(session, stale_job)

        await session.commit()
        if released_count:
            log.info("Stale job cleanup: %d jobs, %d tokens released", len(stale_jobs), released_count)


async def get_scan(scan_id: str, user_id: str | None = None) -> dict | None:
    """Look up scan_job by ID, optionally scoped to a user.

    Auto-marks jobs stuck in active states as failed. Processing scans use the
    Redis progress heartbeat so deploy-killed workers do not leave scans stuck.
    """
    async with async_session() as session:
        try:
            scan_uuid = uuid.UUID(scan_id)
        except (ValueError, TypeError):
            return None  # malformed id -> not found (404), not a 500 (L13)
        q = select(ScanJob).where(ScanJob.id == scan_uuid)
        if user_id:
            q = q.where(ScanJob.user_id == uuid.UUID(user_id))
        q = q.with_for_update()
        result = await session.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return None

        if await _processing_scan_is_stale(job):
            await _mark_scan_interrupted(session, job)
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
        try:
            scan_uuid = uuid.UUID(scan_id)
        except (ValueError, TypeError):
            return False  # malformed id -> not found (404) (L13)
        q = select(ScanJob).where(
            ScanJob.id == scan_uuid,
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
