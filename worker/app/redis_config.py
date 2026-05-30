"""Redis connection options shared by worker broker and progress clients."""

import logging
import ssl

from .config import settings

logger = logging.getLogger(__name__)


def redis_ssl_options(redis_url: str) -> dict:
    if not redis_url.startswith("rediss://"):
        return {}
    return {"ssl_cert_reqs": ssl.CERT_NONE}


def redis_connection_options(*, decode_responses: bool = False) -> dict:
    options = {
        "socket_timeout": settings.REDIS_SOCKET_TIMEOUT_SECONDS,
        "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
        "socket_keepalive": settings.REDIS_SOCKET_KEEPALIVE,
        "health_check_interval": settings.REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
        **redis_ssl_options(settings.REDIS_URL),
    }
    if decode_responses:
        options["decode_responses"] = True
    return options


def redis_broker_transport_options() -> dict:
    visibility_timeout = max(
        settings.CELERY_VISIBILITY_TIMEOUT_SECONDS,
        settings.REWRITE_TIME_LIMIT_SECONDS + 120,
    )
    # NB: kombu's redis transport ignores a `polling_interval` transport option
    # (it is not in Channel.from_transport_options). The idle-poll cost is driven
    # by kombu's hardcoded 1s BRPOP timeout instead -- widen it via
    # apply_broker_brpop_timeout() rather than setting a no-op option here.
    return {
        "visibility_timeout": visibility_timeout,
        "socket_timeout": settings.REDIS_SOCKET_TIMEOUT_SECONDS,
        "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
        "socket_keepalive": settings.REDIS_SOCKET_KEEPALIVE,
        "health_check_interval": settings.REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
        "retry_on_timeout": True,
    }


def apply_broker_brpop_timeout(seconds: int) -> bool:
    """Widen kombu's hardcoded 1s Redis BRPOP timeout to cut idle broker commands.

    kombu's redis transport (``kombu.transport.redis.Channel._brpop_start``)
    issues ``BRPOP`` with a fixed 1s server-side timeout and re-issues it ~once a
    second per idle worker -- roughly 86k Redis commands/day/worker even on an
    empty queue. The timeout is not exposed via ``transport_options``, so we raise
    the default here.

    A newly-enqueued task still wakes ``BRPOP`` instantly (blocking-pop
    semantics), so this adds NO task-pickup latency; it only removes empty-poll
    round-trips. ``seconds`` must stay below the broker socket timeout so the
    blocking read returns before the socket read times out. Idempotent; returns
    True when the patch is in place.
    """
    try:
        from kombu.transport import redis as kombu_redis
    except Exception:  # pragma: no cover - kombu is always present in the worker
        logger.warning("Could not import kombu redis transport; BRPOP timeout unchanged")
        return False

    channel = getattr(kombu_redis, "Channel", None)
    original = getattr(channel, "_brpop_start", None)
    if channel is None or not callable(original):
        logger.warning("kombu Channel._brpop_start missing; BRPOP timeout unchanged")
        return False
    if getattr(original, "_draftproof_patched", False):
        return True

    timeout = max(1, int(seconds))

    def _brpop_start(self, timeout=timeout, _orig=original):
        return _orig(self, timeout=timeout)

    _brpop_start._draftproof_patched = True
    channel._brpop_start = _brpop_start
    logger.info("Broker BRPOP timeout set to %ss (kombu default is 1s)", timeout)
    return True
