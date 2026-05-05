"""Ephemeral progress publishing for worker jobs."""

import logging
from datetime import datetime, timezone

import redis

from .config import settings

logger = logging.getLogger(__name__)

_client = None
_STREAM_TTL_SECONDS = 60 * 60
_STREAM_MAXLEN = 100


def _redis_client():
    global _client
    if _client is None:
        kwargs = {"decode_responses": True}
        if settings.REDIS_URL.startswith("rediss://"):
            kwargs["ssl_cert_reqs"] = None
        _client = redis.Redis.from_url(settings.REDIS_URL, **kwargs)
    return _client


def rewrite_progress_key(rewrite_id: str) -> str:
    return f"rewrite_progress:{rewrite_id}"


def publish_rewrite_progress(
    rewrite_id: str,
    *,
    status: str,
    progress_percent: int | None = None,
    progress_message: str | None = None,
    scan_id: str | None = None,
    error: str | None = None,
) -> None:
    """Publish a lightweight progress event to Redis.

    Redis is best-effort here. A Redis outage must never fail the rewrite task.
    """
    fields = {
        "id": str(rewrite_id),
        "status": status or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if scan_id is not None:
        fields["scan_id"] = str(scan_id)
    if progress_percent is not None:
        fields["progress_percent"] = str(max(0, min(100, int(progress_percent))))
    if progress_message is not None:
        fields["progress_message"] = progress_message
    if error is not None:
        fields["error"] = error

    try:
        client = _redis_client()
        key = rewrite_progress_key(rewrite_id)
        client.xadd(key, fields, maxlen=_STREAM_MAXLEN, approximate=True)
        client.expire(key, _STREAM_TTL_SECONDS)
    except Exception:
        logger.warning("Failed to publish rewrite progress to Redis", exc_info=True)
