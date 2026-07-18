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

# Named queues (single source of truth for THIS process's task_routes -- worker/app/celery_app.py
# keeps its own mirrored copy, see the comment there for why both need it).
_SCAN_QUEUE = "scan"
# Dedicated queue for judge_defence_answer (final-review Finding 3, 2026-07-18). scan/run_rewrite
# jobs are multi-minute; this deployment runs the worker at CELERY_WORKER_CONCURRENCY=1 (single
# concurrency, see worker/entrypoint.sh), so a quick defence-answer judgment previously queued
# behind a long-running scan/rewrite on the shared "scan" queue -- easily long enough to blow past
# DefenceCheck.jsx's ~2-minute poll cap (DEFENCE_POLL_MAX_ATTEMPTS * DEFENCE_POLL_INTERVAL_MS)
# before the judge task even started. Splitting it onto its own queue means it is never blocked
# behind a scan/rewrite that happens to be running when it's enqueued.
# REQUIRES the worker process to actually consume this queue -- see worker/entrypoint.sh's
# `-Q` flag (must list this queue name) and worker/app/celery_app.py's mirrored task_routes entry.
_DEFENCE_QUEUE = "defence"

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
        "app.tasks.scan_document": {"queue": _SCAN_QUEUE},
        "app.tasks.run_rewrite": {"queue": _SCAN_QUEUE},
        "app.tasks.regenerate_rewrite_report_assets": {"queue": _SCAN_QUEUE},
        "app.tasks.judge_defence_answer": {"queue": _DEFENCE_QUEUE},
    },
    broker_transport_options=redis_broker_transport_options(CELERY_VISIBILITY_TIMEOUT_SECONDS),
)

# The actual task lives in worker/app/tasks.py — same broker, same serializer
# Task name follows Celery convention: <module>.<function>
scan_document = celery_app.signature("app.tasks.scan_document")
run_rewrite = celery_app.signature("app.tasks.run_rewrite")
regenerate_rewrite_report_assets = celery_app.signature("app.tasks.regenerate_rewrite_report_assets")
# Task 7 (not yet implemented as of this commit) owns worker/app/tasks.py::judge_defence_answer.
# Enqueued by name string only — the API image never copies poc/ or worker/ (root Dockerfile),
# so this task cannot be imported directly. Task 7's implementer: this name string
# ("app.tasks.judge_defence_answer") is the contract — the worker task MUST be named exactly
# this for draftproof-api/app/routes/defence.py's enqueue call to resolve.
judge_defence_answer = celery_app.signature("app.tasks.judge_defence_answer")


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
