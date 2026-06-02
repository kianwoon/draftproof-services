"""R2 report-object retention cleanup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import settings
from .storage import _client

logger = logging.getLogger(__name__)

_MAX_DELETE_BATCH = 1000


@dataclass(frozen=True)
class R2CleanupResult:
    bucket: str
    prefix: str
    cutoff: datetime
    dry_run: bool
    scanned: int = 0
    eligible: int = 0
    deleted: int = 0
    errors: int = 0


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _retention_days(value: int | None = None) -> int:
    days = settings.DRAFTPROOF_R2_REPORT_RETENTION_DAYS if value is None else value
    if days < 1:
        raise ValueError("R2 report retention must be at least 1 day")
    return days


def _report_prefix(value: str | None = None) -> str:
    prefix = (settings.DRAFTPROOF_R2_REPORT_PREFIX if value is None else value).strip()
    if not prefix:
        raise ValueError("R2 report cleanup prefix must not be empty")
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _iter_report_objects(s3: Any, *, bucket: str, prefix: str):
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        page = s3.list_objects_v2(**kwargs)
        for item in page.get("Contents", []) or []:
            yield item
        if not page.get("IsTruncated"):
            break
        continuation_token = page.get("NextContinuationToken")
        if not continuation_token:
            break


def _delete_batch(s3: Any, *, bucket: str, keys: list[str]) -> tuple[int, int]:
    if not keys:
        return 0, 0
    response = s3.delete_objects(
        Bucket=bucket,
        Delete={
            "Objects": [{"Key": key} for key in keys],
            "Quiet": True,
        },
    )
    error_count = len(response.get("Errors", []) or [])
    if "Deleted" in response:
        deleted_count = len(response.get("Deleted") or [])
    else:
        deleted_count = len(keys) - error_count
    return deleted_count, error_count


def cleanup_old_report_objects(
    *,
    retention_days: int | None = None,
    prefix: str | None = None,
    dry_run: bool = True,
    now: datetime | None = None,
    s3: Any | None = None,
) -> R2CleanupResult:
    """Delete R2 report objects older than the configured retention window."""

    if not settings.R2_BUCKET_NAME:
        raise ValueError("R2_BUCKET_NAME is required for report cleanup")

    retention = _retention_days(retention_days)
    report_prefix = _report_prefix(prefix)
    current_time = _utc_datetime(now or datetime.now(timezone.utc))
    cutoff = current_time - timedelta(days=retention)
    client = s3 or _client()

    scanned = 0
    eligible_keys: list[str] = []
    deleted = 0
    errors = 0

    for item in _iter_report_objects(client, bucket=settings.R2_BUCKET_NAME, prefix=report_prefix):
        scanned += 1
        key = item.get("Key")
        last_modified = item.get("LastModified")
        if not key or not isinstance(last_modified, datetime):
            continue
        if _utc_datetime(last_modified) >= cutoff:
            continue
        eligible_keys.append(key)

        if not dry_run and len(eligible_keys) >= _MAX_DELETE_BATCH:
            batch_deleted, batch_errors = _delete_batch(
                client,
                bucket=settings.R2_BUCKET_NAME,
                keys=eligible_keys,
            )
            deleted += batch_deleted
            errors += batch_errors
            eligible_keys.clear()

    if not dry_run and eligible_keys:
        batch_deleted, batch_errors = _delete_batch(
            client,
            bucket=settings.R2_BUCKET_NAME,
            keys=eligible_keys,
        )
        deleted += batch_deleted
        errors += batch_errors

    result = R2CleanupResult(
        bucket=settings.R2_BUCKET_NAME,
        prefix=report_prefix,
        cutoff=cutoff,
        dry_run=dry_run,
        scanned=scanned,
        eligible=len(eligible_keys) if dry_run else deleted + errors,
        deleted=0 if dry_run else deleted,
        errors=errors,
    )
    logger.info(
        "R2 report cleanup: bucket=%s prefix=%s cutoff=%s dry_run=%s scanned=%d eligible=%d deleted=%d errors=%d",
        result.bucket,
        result.prefix,
        result.cutoff.isoformat(),
        result.dry_run,
        result.scanned,
        result.eligible,
        result.deleted,
        result.errors,
    )
    return result
