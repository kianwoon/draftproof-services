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
