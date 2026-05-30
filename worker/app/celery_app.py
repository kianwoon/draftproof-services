"""Celery application — broker via Redis."""

from celery import Celery

from .config import settings
from .redis_config import (
    apply_broker_brpop_timeout,
    redis_broker_transport_options,
    redis_ssl_options,
)

# kombu hardcodes a 1s BRPOP timeout, re-issued ~once a second per idle worker
# (~86k Redis commands/day/worker on an empty queue -- costly on per-request
# Upstash). Widen it to cut idle broker commands ~10x. A pushed task still wakes
# BRPOP instantly, so this adds no pickup latency. Stay below the broker socket
# timeout so the blocking read returns before the socket read times out.
apply_broker_brpop_timeout(
    min(
        settings.CELERY_BROKER_POLLING_INTERVAL_SECONDS,
        max(1, settings.REDIS_SOCKET_TIMEOUT_SECONDS - 5),
    )
)

app = Celery(
    "draftproof",
    broker=settings.REDIS_URL,
    backend=None,
    include=["app.tasks"],
)

broker_ssl_options = redis_ssl_options(settings.REDIS_URL)
if broker_ssl_options:
    app.conf.update(broker_use_ssl=broker_ssl_options)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=False,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # Disable Celery's internal event stream — single worker, no monitoring.
    # Event polling was generating ~11K reads/sec on Redis.
    worker_send_task_events=False,
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,
    broker_channel_error_retry=True,
    task_ignore_result=True,
    task_routes={
        "app.tasks.scan_document": {"queue": "scan"},
        "app.tasks.run_rewrite": {"queue": "scan"},
        "app.tasks.regenerate_rewrite_report_assets": {"queue": "scan"},
    },
    task_default_queue="default",
    # Keep the visibility timeout longer than the longest task. Re-delivering
    # a live rewrite task causes duplicate executions while the first worker is
    # still doing final detector scans.
    broker_transport_options=redis_broker_transport_options(),
)
