"""Celery client — lets the API enqueue tasks on the same broker as the worker."""

import ssl

from celery import Celery
from app.config import REDIS_URL

celery_app = Celery("draftproof-api", broker=REDIS_URL, backend=REDIS_URL)

if REDIS_URL.startswith("rediss://"):
    celery_app.conf.update(
        broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
        redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    )

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_routes={
        "app.tasks.scan_document": {"queue": "scan"},
        "app.tasks.run_rewrite": {"queue": "scan"},
    },
)

# The actual task lives in worker/app/tasks.py — same broker, same serializer
# Task name follows Celery convention: <module>.<function>
scan_document = celery_app.signature("app.tasks.scan_document")
run_rewrite = celery_app.signature("app.tasks.run_rewrite")
