"""Redis connection options shared by API broker and progress clients."""

import ssl

from app.config import (
    REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
    REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
    REDIS_SOCKET_KEEPALIVE,
    REDIS_SOCKET_TIMEOUT_SECONDS,
    REDIS_URL,
)


def redis_ssl_options(redis_url: str) -> dict:
    if not redis_url.startswith("rediss://"):
        return {}
    return {"ssl_cert_reqs": ssl.CERT_NONE}


def redis_connection_options(*, decode_responses: bool = False) -> dict:
    options = {
        "socket_timeout": REDIS_SOCKET_TIMEOUT_SECONDS,
        "socket_connect_timeout": REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
        "socket_keepalive": REDIS_SOCKET_KEEPALIVE,
        "health_check_interval": REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
        **redis_ssl_options(REDIS_URL),
    }
    if decode_responses:
        options["decode_responses"] = True
    return options


def redis_broker_transport_options(visibility_timeout: int) -> dict:
    return {
        "visibility_timeout": visibility_timeout,
        "socket_timeout": REDIS_SOCKET_TIMEOUT_SECONDS,
        "socket_connect_timeout": REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
        "socket_keepalive": REDIS_SOCKET_KEEPALIVE,
        "health_check_interval": REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
        "retry_on_timeout": True,
    }
