from app import config
from app.services import celery_client, report_service, rewrite_service
from app.services.redis_config import redis_connection_options


def test_rewrite_status_normalizes_legacy_document_progress_message():
    assert (
        rewrite_service._normalize_rewrite_progress_message("Rewriting your document...")
        == "Rewriting AI sections"
    )


def test_rewrite_status_normalizes_legacy_wait_progress_message():
    assert (
        rewrite_service._normalize_rewrite_progress_message("This may take 1-3 minutes")
        == "Rewriting AI sections"
    )


def test_rewrite_status_normalizes_combined_legacy_progress_message():
    assert (
        rewrite_service._normalize_rewrite_progress_message(
            "Rewriting your document...\n\nThis may take 1-3 minutes"
        )
        == "Rewriting AI sections"
    )


def test_report_embedded_rewrite_normalizes_legacy_document_progress_message():
    assert (
        report_service._normalize_rewrite_progress_message("Rewriting your document...")
        == "Rewriting AI sections"
    )


def test_report_embedded_rewrite_normalizes_legacy_wait_progress_message():
    assert (
        report_service._normalize_rewrite_progress_message("This may take 1-3 minutes")
        == "Rewriting AI sections"
    )


def test_rewrite_progress_message_keeps_current_worker_message():
    assert (
        rewrite_service._normalize_rewrite_progress_message("Rewriting AI sections")
        == "Rewriting AI sections"
    )


def test_completed_v4_rewrite_with_comparison_scores_is_reusable():
    assert (
        rewrite_service._completed_rewrite_report_is_reusable(
            {
                "status": "rewrite_candidate_generated_needs_external_review",
                "summary": {
                    "rewrite_pipeline_version": "rewrite_v4_normalized_repair",
                    "outcome": "rewrite_candidate_generated_needs_external_review",
                    "no_text_change": False,
                    "detect_scan_rewritten": {"overall_tier": "medium"},
                    "detect_scores": {
                        "original_ai": 54.6,
                        "rewritten_ai": 53.1,
                    },
                },
            }
        )
        is True
    )


def test_completed_v4_original_preserved_rewrite_is_not_reusable():
    assert (
        rewrite_service._completed_rewrite_report_is_reusable(
            {
                "status": "original_preserved",
                "summary": {
                    "rewrite_pipeline_version": "rewrite_v4_normalized_repair",
                    "outcome": "original_preserved",
                    "no_text_change": True,
                },
            }
        )
        is False
    )


def test_completed_v4_rewrite_without_comparison_scores_is_not_reusable():
    assert (
        rewrite_service._completed_rewrite_report_is_reusable(
            {
                "status": "rewrite_candidate_generated_needs_external_review",
                "summary": {
                    "rewrite_pipeline_version": "rewrite_v4_normalized_repair",
                    "outcome": "rewrite_candidate_generated_needs_external_review",
                    "no_text_change": False,
                },
            }
        )
        is False
    )


def test_completed_non_v4_rewrite_reuse_contract_is_unchanged():
    assert (
        rewrite_service._completed_rewrite_report_is_reusable(
            {
                "status": "completed",
                "summary": {
                    "rewrite_pipeline_version": "rewrite_v3",
                },
            }
        )
        is True
    )


def test_canceled_rewrite_does_not_block_new_rewrite_job():
    assert "canceled" not in rewrite_service._ACTIVE_REWRITE_STATUSES


def test_cancel_rewrite_task_revokes_without_terminating_by_default(monkeypatch):
    calls = []

    def fake_revoke(task_id, *, terminate, signal):
        calls.append({"task_id": task_id, "terminate": terminate, "signal": signal})

    monkeypatch.setattr(celery_client.celery_app.control, "revoke", fake_revoke)

    assert celery_client.cancel_rewrite_task("rewrite-123") is True
    assert calls == [{
        "task_id": "rewrite-123",
        "terminate": False,
        "signal": celery_client.REWRITE_CANCEL_TERMINATE_SIGNAL,
    }]


def test_cancel_rewrite_task_is_best_effort(monkeypatch):
    def fake_revoke(task_id, *, terminate, signal):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(celery_client.celery_app.control, "revoke", fake_revoke)

    assert celery_client.cancel_rewrite_task("rewrite-123") is False


def test_api_celery_client_uses_resilient_redis_transport_options():
    options = celery_client.celery_app.conf.broker_transport_options

    assert options["visibility_timeout"] == config.CELERY_VISIBILITY_TIMEOUT_SECONDS
    assert options["socket_timeout"] == config.REDIS_SOCKET_TIMEOUT_SECONDS
    assert options["socket_connect_timeout"] == config.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS
    assert options["socket_keepalive"] is config.REDIS_SOCKET_KEEPALIVE
    assert options["health_check_interval"] == config.REDIS_HEALTH_CHECK_INTERVAL_SECONDS
    assert options["retry_on_timeout"] is True


def test_api_celery_client_retries_broker_connections():
    assert celery_client.celery_app.conf.task_publish_retry is True
    assert celery_client.celery_app.conf.broker_connection_retry is True
    assert celery_client.celery_app.conf.broker_connection_retry_on_startup is True
    assert celery_client.celery_app.conf.broker_connection_max_retries is None
    assert celery_client.celery_app.conf.broker_channel_error_retry is True


def test_api_progress_client_uses_same_redis_socket_policy():
    options = redis_connection_options(decode_responses=True)

    assert options["decode_responses"] is True
    assert options["socket_timeout"] == config.REDIS_SOCKET_TIMEOUT_SECONDS
    assert options["socket_connect_timeout"] == config.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS
    assert options["socket_keepalive"] is config.REDIS_SOCKET_KEEPALIVE
    assert options["health_check_interval"] == config.REDIS_HEALTH_CHECK_INTERVAL_SECONDS


def test_saved_rewrite_checkpoint_with_changed_text_is_delivered_content():
    assert (
        rewrite_service._rewrite_report_has_delivered_content(
            {
                "status": "rewrite_candidate_generated_needs_external_review",
                "original_text": "Original text.",
                "final_text": "Changed rewritten text.",
                "summary": {
                    "partial_rewrite_preserved": True,
                },
            }
        )
        is True
    )


def test_saved_rewrite_checkpoint_without_text_change_is_not_delivered_content():
    assert (
        rewrite_service._rewrite_report_has_delivered_content(
            {
                "status": "rewrite_candidate_generated_needs_external_review",
                "original_text": "Same text.",
                "final_text": "Same text.",
            }
        )
        is False
    )
