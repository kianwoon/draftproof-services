from types import SimpleNamespace

from app.tasks import _rewrite_billing_decision, _selected_rewrite_pipeline


def test_rewrite_pipeline_selection_prefers_v5_when_enabled():
    settings = SimpleNamespace(
        DRAFTPROOF_REWRITE_V5_ENABLED=True,
        DRAFTPROOF_REWRITE_V4_ENABLED=True,
        DRAFTPROOF_REWRITE_V3_ENABLED=True,
        DRAFTPROOF_REWRITE_V2_ENABLED=True,
    )

    assert _selected_rewrite_pipeline(settings) == "v5"


def test_rewrite_pipeline_selection_falls_back_to_v4_when_v5_disabled():
    settings = SimpleNamespace(
        DRAFTPROOF_REWRITE_V5_ENABLED=False,
        DRAFTPROOF_REWRITE_V4_ENABLED=True,
        DRAFTPROOF_REWRITE_V3_ENABLED=True,
        DRAFTPROOF_REWRITE_V2_ENABLED=True,
    )

    assert _selected_rewrite_pipeline(settings) == "v4"


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


def test_rewrite_billing_releases_external_review_candidate_even_when_text_changed():
    decision = _rewrite_billing_decision(
        {"status": "rewrite_candidate_generated_needs_external_review"},
        {
            "status": "rewrite_candidate_generated_needs_external_review",
            "original_text": "This is the original draft.",
            "final_text": "This is a changed candidate that still needs external review.",
            "summary": {
                "outcome": "rewrite_candidate_generated_needs_external_review",
                "strict_goal_status": "mitigation_failed_no_safe_candidate",
                "best_candidate_external_review_required": True,
                "public_candidate_warning": "best_candidate_requires_external_review",
            },
        },
    )

    assert decision["billable"] is False
    assert decision["reason"] == "external_review_required_status:rewrite_candidate_generated_needs_external_review"


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
