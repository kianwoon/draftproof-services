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
    assert options["polling_interval"] == settings.CELERY_BROKER_POLLING_INTERVAL_SECONDS


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
