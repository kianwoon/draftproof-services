"""Celery client — lets the API enqueue and control tasks on the worker broker."""

import logging

from celery import Celery
from app.config import (
    CELERY_VISIBILITY_TIMEOUT_SECONDS,
    REDIS_URL,
    REWRITE_CANCEL_TERMINATE,
    REWRITE_CANCEL_TERMINATE_SIGNAL,
)
from app.services.redis_config import redis_broker_transport_options, redis_ssl_options

logger = logging.getLogger("celery_client")

celery_app = Celery("draftproof-api", broker=REDIS_URL, backend=None)

broker_ssl_options = redis_ssl_options(REDIS_URL)
if broker_ssl_options:
    celery_app.conf.update(broker_use_ssl=broker_ssl_options)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_publish_retry=True,
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,
    broker_channel_error_retry=True,
    task_routes={
        "app.tasks.scan_document": {"queue": "scan"},
        "app.tasks.run_rewrite": {"queue": "scan"},
        "app.tasks.regenerate_rewrite_report_assets": {"queue": "scan"},
    },
    broker_transport_options=redis_broker_transport_options(CELERY_VISIBILITY_TIMEOUT_SECONDS),
)

# The actual task lives in worker/app/tasks.py — same broker, same serializer
# Task name follows Celery convention: <module>.<function>
scan_document = celery_app.signature("app.tasks.scan_document")
run_rewrite = celery_app.signature("app.tasks.run_rewrite")
regenerate_rewrite_report_assets = celery_app.signature("app.tasks.regenerate_rewrite_report_assets")


def cancel_rewrite_task(task_id: str) -> bool:
    """Revoke a rewrite task and let the DB cancellation flag stop active work.

    Termination is an opt-in emergency setting because killing a late-ack worker
    child can redeliver the task before the UI has settled on the terminal row.
    """
    try:
        celery_app.control.revoke(
            str(task_id),
            terminate=REWRITE_CANCEL_TERMINATE,
            signal=REWRITE_CANCEL_TERMINATE_SIGNAL,
        )
        return True
    except Exception:
        logger.warning("Failed to revoke rewrite task %s", task_id, exc_info=True)
        return False
