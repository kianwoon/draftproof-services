"""Redis connection options shared by worker broker and progress clients."""

import ssl

from .config import settings


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
    return {
        "visibility_timeout": visibility_timeout,
        "socket_timeout": settings.REDIS_SOCKET_TIMEOUT_SECONDS,
        "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
        "socket_keepalive": settings.REDIS_SOCKET_KEEPALIVE,
        "health_check_interval": settings.REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
        "retry_on_timeout": True,
        "polling_interval": settings.CELERY_BROKER_POLLING_INTERVAL_SECONDS,
    }
