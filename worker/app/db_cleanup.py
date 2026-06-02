"""Database retention cleanup for report-owned scan/rewrite rows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import get_conn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DbReportCleanupResult:
    cutoff: datetime
    dry_run: bool
    old_scan_jobs: int = 0
    old_rewrite_jobs: int = 0
    deleted_scan_jobs: int = 0
    deleted_rewrite_jobs: int = 0
    released_orphan_reservations: int = 0
    released_orphan_tokens: int = 0


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _retention_days(value: int) -> int:
    if value < 1:
        raise ValueError("DB report retention must be at least 1 day")
    return value


def _count(cur: Any, sql: str, params: tuple) -> int:
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(row["n"] if isinstance(row, dict) else row[0])


def cleanup_old_report_rows(
    *,
    retention_days: int = 3,
    dry_run: bool = True,
    now: datetime | None = None,
) -> DbReportCleanupResult:
    """Purge scan/rewrite job rows older than retention and release stale reservations."""

    retention = _retention_days(retention_days)
    current_time = _utc_datetime(now or datetime.now(timezone.utc))
    cutoff = current_time - timedelta(days=retention)

    with get_conn() as conn:
        cur = conn.cursor()
        old_scan_jobs = _count(
            cur,
            "SELECT count(*) AS n FROM scan_jobs WHERE created_at < %s",
            (cutoff,),
        )
        old_rewrite_jobs = _count(
            cur,
            "SELECT count(*) AS n FROM rewrite_jobs WHERE created_at < %s",
            (cutoff,),
        )

        if dry_run:
            result = DbReportCleanupResult(
                cutoff=cutoff,
                dry_run=True,
                old_scan_jobs=old_scan_jobs,
                old_rewrite_jobs=old_rewrite_jobs,
            )
            logger.info("DB report cleanup dry-run: %s", result)
            return result

        cur.execute(
            """
            WITH stale AS (
                SELECT cr.id
                FROM credit_reservations cr
                LEFT JOIN scan_jobs s ON cr.job_type = 'scan' AND s.id = cr.job_id
                LEFT JOIN rewrite_jobs r ON cr.job_type = 'rewrite' AND r.id = cr.job_id
                WHERE cr.status = 'active'
                  AND cr.created_at < %s
                  AND cr.expires_at < now()
                  AND (
                    (cr.job_type = 'scan' AND s.id IS NULL)
                    OR (cr.job_type = 'rewrite' AND r.id IS NULL)
                  )
            ), released AS (
                UPDATE credit_reservations cr
                SET status = 'released', updated_at = now()
                FROM stale
                WHERE cr.id = stale.id
                RETURNING cr.tokens_reserved
            )
            SELECT count(*) AS n, COALESCE(sum(tokens_reserved), 0) AS tokens FROM released
            """,
            (cutoff,),
        )
        released_row = cur.fetchone()
        released_orphan_reservations = int(released_row["n"])
        released_orphan_tokens = int(released_row["tokens"])

        cur.execute(
            """
            WITH deleted AS (
                DELETE FROM rewrite_jobs
                WHERE created_at < %s
                RETURNING id
            )
            SELECT count(*) AS n FROM deleted
            """,
            (cutoff,),
        )
        deleted_rewrite_jobs = int(cur.fetchone()["n"])

        cur.execute(
            """
            WITH deleted AS (
                DELETE FROM scan_jobs
                WHERE created_at < %s
                RETURNING id
            )
            SELECT count(*) AS n FROM deleted
            """,
            (cutoff,),
        )
        deleted_scan_jobs = int(cur.fetchone()["n"])

    result = DbReportCleanupResult(
        cutoff=cutoff,
        dry_run=False,
        old_scan_jobs=old_scan_jobs,
        old_rewrite_jobs=old_rewrite_jobs,
        deleted_scan_jobs=deleted_scan_jobs,
        deleted_rewrite_jobs=deleted_rewrite_jobs,
        released_orphan_reservations=released_orphan_reservations,
        released_orphan_tokens=released_orphan_tokens,
    )
    logger.info("DB report cleanup: %s", result)
    return result
