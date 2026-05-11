"""Redis-backed progress streams for live job updates."""

import logging
import ssl

import redis.asyncio as redis

from app.config import REDIS_URL

logger = logging.getLogger(__name__)

_client = None


def scan_progress_key(scan_id: str) -> str:
    return f"scan_progress:{scan_id}"


def rewrite_progress_key(rewrite_id: str) -> str:
    return f"rewrite_progress:{rewrite_id}"


def _redis_client():
    global _client
    if _client is None:
        kwargs = {"decode_responses": True}
        if REDIS_URL.startswith("rediss://"):
            kwargs["ssl_cert_reqs"] = ssl.CERT_NONE
        _client = redis.Redis.from_url(REDIS_URL, **kwargs)
    return _client


async def read_rewrite_progress(
    rewrite_id: str,
    last_id: str = "$",
    *,
    block_ms: int = 3000,
    count: int = 10,
) -> list[tuple[str, dict]] | None:
    """Read rewrite progress events.

    Returns None when Redis is unavailable so callers can fall back to durable
    database polling without failing the SSE connection.
    """
    try:
        response = await _redis_client().xread(
            {rewrite_progress_key(rewrite_id): last_id},
            block=block_ms,
            count=count,
        )
    except Exception:
        logger.warning("Failed to read rewrite progress from Redis", exc_info=True)
        return None

    events = []
    for _, entries in response:
        for event_id, fields in entries:
            events.append((event_id, fields))
    return events


async def read_latest_rewrite_progress(rewrite_id: str) -> tuple[str, dict] | None:
    """Return the newest rewrite progress event, or None if unavailable/missing."""
    try:
        entries = await _redis_client().xrevrange(
            rewrite_progress_key(rewrite_id),
            count=1,
        )
    except Exception:
        logger.warning("Failed to read latest rewrite progress from Redis", exc_info=True)
        return None

    if not entries:
        return None
    event_id, fields = entries[0]
    return event_id, fields


async def read_latest_scan_progress(scan_id: str) -> tuple[str, dict] | None:
    """Return the newest scan progress event, or None if unavailable/missing."""
    try:
        entries = await _redis_client().xrevrange(
            scan_progress_key(scan_id),
            count=1,
        )
    except Exception:
        logger.warning("Failed to read latest scan progress from Redis", exc_info=True)
        return None

    if not entries:
        return None
    event_id, fields = entries[0]
    return event_id, fields


async def read_scan_progress(
    scan_id: str,
    last_id: str = "$",
    *,
    block_ms: int = 3000,
    count: int = 10,
) -> list[tuple[str, dict]] | None:
    """Read scan progress events from Redis stream.

    Returns None when Redis is unavailable (fallback to DB polling).
    """
    try:
        response = await _redis_client().xread(
            {scan_progress_key(scan_id): last_id},
            block=block_ms,
            count=count,
        )
    except Exception:
        logger.warning("Failed to read scan progress from Redis", exc_info=True)
        return None

    events = []
    for _, entries in response:
        for event_id, fields in entries:
            events.append((event_id, fields))
    return events
