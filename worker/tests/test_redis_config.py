from app.celery_app import app
from app.config import settings
from app.redis_config import redis_connection_options


def test_worker_broker_transport_has_resilient_redis_options():
    options = app.conf.broker_transport_options

    assert options["visibility_timeout"] >= settings.REWRITE_TIME_LIMIT_SECONDS + 120
    assert options["socket_timeout"] == settings.REDIS_SOCKET_TIMEOUT_SECONDS
    assert options["socket_connect_timeout"] == settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS
    assert options["socket_keepalive"] is settings.REDIS_SOCKET_KEEPALIVE
    assert options["health_check_interval"] == settings.REDIS_HEALTH_CHECK_INTERVAL_SECONDS
    assert options["retry_on_timeout"] is True
    # polling_interval is intentionally NOT set: kombu's redis transport ignores
    # it (not in from_transport_options). Idle-poll cost is controlled by widening
    # kombu's hardcoded 1s BRPOP timeout instead (see test below).
    assert "polling_interval" not in options


def test_worker_brpop_timeout_is_widened_below_socket_timeout():
    from kombu.transport import redis as kombu_redis

    fn = kombu_redis.Channel._brpop_start
    assert getattr(fn, "_draftproof_patched", False) is True
    expected = min(
        settings.CELERY_BROKER_POLLING_INTERVAL_SECONDS,
        max(1, settings.REDIS_SOCKET_TIMEOUT_SECONDS - 5),
    )
    assert fn.__defaults__[0] == expected
    assert fn.__defaults__[0] < settings.REDIS_SOCKET_TIMEOUT_SECONDS


def test_worker_celery_reconnects_without_retry_ceiling():
    assert app.conf.broker_connection_retry is True
    assert app.conf.broker_connection_retry_on_startup is True
    assert app.conf.broker_connection_max_retries is None
    assert app.conf.broker_channel_error_retry is True


def test_worker_progress_client_uses_same_redis_socket_policy():
    options = redis_connection_options(decode_responses=True)

    assert options["decode_responses"] is True
    assert options["socket_timeout"] == settings.REDIS_SOCKET_TIMEOUT_SECONDS
    assert options["socket_connect_timeout"] == settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS
    assert options["socket_keepalive"] is settings.REDIS_SOCKET_KEEPALIVE
    assert options["health_check_interval"] == settings.REDIS_HEALTH_CHECK_INTERVAL_SECONDS
