"""Celery application — broker via Upstash Redis (TLS)."""

import ssl

from celery import Celery

from .config import settings

app = Celery(
    "draftproof",
    broker=settings.REDIS_URL,
    backend=None,
    include=["app.tasks"],
)

# Upstash requires TLS
if settings.REDIS_URL.startswith("rediss://"):
    app.conf.update(
        broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    )

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
    broker_transport_options={
        "visibility_timeout": max(
            settings.CELERY_VISIBILITY_TIMEOUT_SECONDS,
            settings.REWRITE_TIME_LIMIT_SECONDS + 120,
        ),
        # Reduce broker heartbeats from default 2s → 120s.
        # Single-worker setup doesn't need frequent health checks.
        # Saves ~40K Redis commands/day.
        "health_check_interval": 120,
        # BRPOP polling interval. Default 1s = ~86K commands/day.
        # 10s = ~8.6K/day. Adds up to 9s latency on task pickup — acceptable
        # for scan/rewrite jobs that run for minutes.
        "polling_interval": 10,
    },
)
