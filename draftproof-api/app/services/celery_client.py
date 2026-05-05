"""Celery client — lets the API enqueue tasks on the same broker as the worker."""

import ssl

from celery import Celery
from app.config import CELERY_VISIBILITY_TIMEOUT_SECONDS, REDIS_URL

celery_app = Celery("draftproof-api", broker=REDIS_URL, backend=None)

if REDIS_URL.startswith("rediss://"):
    celery_app.conf.update(
        broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    )

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_routes={
        "app.tasks.scan_document": {"queue": "scan"},
        "app.tasks.run_rewrite": {"queue": "scan"},
        "app.tasks.regenerate_rewrite_report_assets": {"queue": "scan"},
    },
    broker_transport_options={
        "visibility_timeout": CELERY_VISIBILITY_TIMEOUT_SECONDS,
        # API is a producer only — no need for frequent health checks.
        "health_check_interval": 120,
    },
)

# The actual task lives in worker/app/tasks.py — same broker, same serializer
# Task name follows Celery convention: <module>.<function>
scan_document = celery_app.signature("app.tasks.scan_document")
run_rewrite = celery_app.signature("app.tasks.run_rewrite")
regenerate_rewrite_report_assets = celery_app.signature("app.tasks.regenerate_rewrite_report_assets")
