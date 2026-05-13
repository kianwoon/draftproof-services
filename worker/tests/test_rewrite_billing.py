from app.tasks import _rewrite_billing_decision


def test_rewrite_billing_captures_changed_rewritten_content():
    decision = _rewrite_billing_decision(
        {"status": "rewritten"},
        {
            "status": "rewritten",
            "original_text": "This is the original draft.",
            "final_text": "This is the revised draft with clearer authorship.",
            "summary": {"outcome": "ai_mitigated"},
        },
    )

    assert decision["billable"] is True
    assert decision["reason"] == "rewritten_content_delivered"
    assert decision["text_changed"] is True


def test_rewrite_billing_releases_no_safe_candidate_even_with_diagnostic_text():
    decision = _rewrite_billing_decision(
        {"status": "mitigation_failed_no_safe_candidate"},
        {
            "status": "mitigation_failed_no_safe_candidate",
            "original_text": "This is the original draft.",
            "final_text": "This is a diagnostic candidate that was not safe.",
            "summary": {"outcome": "mitigation_failed_no_safe_candidate"},
        },
    )

    assert decision["billable"] is False
    assert decision["reason"] == "non_billable_status:mitigation_failed_no_safe_candidate"


def test_rewrite_billing_releases_original_preserved_text():
    decision = _rewrite_billing_decision(
        {"status": "rewritten"},
        {
            "status": "rewritten",
            "original_text": "This draft stayed the same.",
            "final_text": "This draft stayed the same.",
            "summary": {"outcome": "cleanup_only"},
        },
    )

    assert decision["billable"] is False
    assert decision["reason"] == "final_text_unchanged"


def test_rewrite_billing_releases_empty_final_text():
    decision = _rewrite_billing_decision(
        {"status": "rewritten"},
        {
            "status": "rewritten",
            "original_text": "This draft had no delivered rewrite.",
            "final_text": "",
            "summary": {"outcome": "ai_mitigated"},
        },
    )

    assert decision["billable"] is False
    assert decision["reason"] == "empty_final_text"
