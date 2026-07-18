"""Celery task-routing regression test (final-review Finding 3, 2026-07-18).

judge_defence_answer must route to its OWN queue, distinct from the "scan" queue used by
scan_document/run_rewrite -- those are multi-minute jobs and this worker deployment runs at
CELERY_WORKER_CONCURRENCY=1 (see worker/entrypoint.sh), so sharing a queue let a quick
defence-answer judgment queue behind a long-running scan/rewrite for long enough to blow past
DefenceCheck.jsx's ~2-minute poll cap before the judge task even started.

This test only pins the ROUTING CONFIG (task_routes), not actual concurrent execution -- with
concurrency=1 a dedicated queue alone does not guarantee the judge task runs promptly while a
scan is mid-execution (see the comment block in worker/entrypoint.sh and
.superpowers/sdd/final-review-fixes-report.md Finding 3). This test exists so the routing
half of the fix can't silently regress back onto the shared "scan" queue.
"""
from app.services.celery_client import celery_app


def test_judge_defence_answer_routes_to_its_own_queue():
    routes = celery_app.conf.task_routes
    assert routes["app.tasks.judge_defence_answer"]["queue"] == "defence"


def test_judge_defence_answer_queue_distinct_from_scan_queue():
    routes = celery_app.conf.task_routes
    scan_tasks = (
        "app.tasks.scan_document",
        "app.tasks.run_rewrite",
        "app.tasks.regenerate_rewrite_report_assets",
    )
    defence_queue = routes["app.tasks.judge_defence_answer"]["queue"]
    for task_name in scan_tasks:
        assert routes[task_name]["queue"] == "scan"
        assert routes[task_name]["queue"] != defence_queue
