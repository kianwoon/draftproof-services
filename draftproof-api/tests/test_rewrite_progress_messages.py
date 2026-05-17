from app.services import report_service, rewrite_service


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
