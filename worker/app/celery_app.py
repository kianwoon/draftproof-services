"""Celery application — broker via Upstash Redis (TLS)."""

import ssl

from celery import Celery

from .config import settings

app = Celery(
    "draftproof",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks"],
)

# Upstash requires TLS
if settings.REDIS_URL.startswith("rediss://"):
    app.conf.update(
        broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
        redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    )

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "app.tasks.scan_document": {"queue": "scan"},
        "app.tasks.run_rewrite": {"queue": "scan"},
    },
    task_default_queue="default",
    # Re-deliver unacked tasks after 2 minutes if worker crashed
    broker_transport_options={"visibility_timeout": 120},
)
