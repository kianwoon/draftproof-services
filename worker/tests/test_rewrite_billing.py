from types import SimpleNamespace

import app.tasks as tasks
from app.tasks import (
    _bounded_json_debug_log,
    _bounded_rewrite_json_payload,
    _deliver_rewrite_completion_email,
    _historical_rewrite_seed_texts,
    _rewrite_billing_decision,
    _selected_rewrite_pipeline,
)


def test_completion_email_lookup_raise_is_swallowed(monkeypatch):
    # Runs AFTER status='completed' + credits captured. An email-lookup raise must
    # NOT propagate (the generic run_rewrite handler would flip the billed job to
    # 'failed'). The helper must swallow it so the completed status survives.
    def boom(_rid):
        raise RuntimeError("user email lookup failed")

    sent = []
    monkeypatch.setattr(tasks, "get_rewrite_user_email", boom)
    monkeypatch.setattr(tasks, "send_rewrite_completion_email", lambda **kw: sent.append(kw))

    # Must not raise.
    _deliver_rewrite_completion_email(
        rewrite_id="rw1",
        scan_id="sc1",
        rewritten_text="new text",
        rewrite_summary={"outcome": "ai_mitigated"},
        pdf_bytes=b"",
        settings_obj=SimpleNamespace(),
    )
    assert sent == []  # send never reached because lookup raised first


def test_completion_email_send_raise_is_swallowed(monkeypatch):
    def boom(**_kw):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(tasks, "get_rewrite_user_email", lambda _rid: "u@example.com")
    monkeypatch.setattr(tasks, "send_rewrite_completion_email", boom)

    # Must not raise even when the send itself fails.
    _deliver_rewrite_completion_email(
        rewrite_id="rw1",
        scan_id="sc1",
        rewritten_text="new text",
        rewrite_summary={"outcome": "ai_mitigated"},
        pdf_bytes=b"",
        settings_obj=SimpleNamespace(),
    )


def test_rewrite_pipeline_selection_returns_v6_when_enabled():
    settings = SimpleNamespace(DRAFTPROOF_REWRITE_V6_ENABLED=True)

    assert _selected_rewrite_pipeline(settings) == "v6"


def test_rewrite_pipeline_selection_falls_back_to_legacy_when_v6_disabled():
    settings = SimpleNamespace(DRAFTPROOF_REWRITE_V6_ENABLED=False)

    assert _selected_rewrite_pipeline(settings) == "legacy"


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


def test_rewrite_billing_releases_author_review_candidate_even_when_text_changed():
    decision = _rewrite_billing_decision(
        {"status": "rewrite_candidate_generated_needs_author_review"},
        {
            "status": "rewrite_candidate_generated_needs_author_review",
            "original_text": "This is the original draft.",
            "final_text": "This is a changed author-proxy candidate.",
            "summary": {
                "outcome": "rewrite_candidate_generated_needs_author_review",
                "best_candidate_author_review_required": True,
                "public_candidate_warning": "author_proxy_candidate_requires_review",
            },
        },
    )

    assert decision["billable"] is False
    assert decision["reason"] == "external_review_required_status:rewrite_candidate_generated_needs_author_review"


def test_rewrite_billing_releases_partial_candidate_not_strict_safe():
    decision = _rewrite_billing_decision(
        {"status": "partial_candidate_not_strict_safe"},
        {
            "status": "partial_candidate_not_strict_safe",
            "original_text": "This is the original draft.",
            "final_text": "This is a changed partial candidate.",
            "summary": {
                "outcome": "partial_candidate_not_strict_safe",
                "public_candidate_warning": "candidate_not_strict_safe",
                "strict_safe_band_achieved": False,
                "kpi_finalization_status": "partial_candidate_not_strict_safe",
            },
        },
    )

    assert decision["billable"] is False
    assert decision["reason"] == "non_billable_status:partial_candidate_not_strict_safe"


def test_historical_rewrite_seed_texts_extracts_distinct_changed_candidates():
    seeds = _historical_rewrite_seed_texts(
        {
            "final_text": "Rewritten candidate with stronger author detail.",
            "summary": {
                "final_text": "Rewritten candidate with stronger author detail.",
                "selected_candidate": {"text": "Alternative candidate from selected row."},
            },
        },
        "Original draft.",
    )

    assert seeds == [
        "Rewritten candidate with stronger author detail.",
        "Alternative candidate from selected row.",
    ]


def test_historical_rewrite_seed_texts_prefers_ranked_candidate_ledger():
    original = " ".join(["original"] * 220)
    best = " ".join(["best"] * 218)
    second = " ".join(["second"] * 216)
    delivered = " ".join(["delivered"] * 214)

    seeds = _historical_rewrite_seed_texts(
        {
            "final_text": delivered,
            "summary": {
                "candidate_ledger": [
                    {
                        "rank": 2,
                        "source": "final_text",
                        "text": delivered,
                        "scores": {"topk_calibrated_risk": 31.0},
                    },
                    {
                        "rank": 1,
                        "source": "global_best_fallback",
                        "text": best,
                        "scores": {"topk_calibrated_risk": 24.9},
                    },
                ],
                "rewrite_layers": {
                    "v5_residual_cluster_comb": {
                        "seed_candidate_rows": [
                            {
                                "section_id": "full_document",
                                "text": second,
                                "scores": {"topk_calibrated_risk": 25.2},
                            }
                        ]
                    }
                },
            },
        },
        original,
        limit=3,
    )

    assert seeds == [best, delivered, second]


def test_historical_rewrite_seed_texts_ignores_paragraph_hard_stop_candidates():
    original = " ".join(["original"] * 220)
    failed_candidate = " ".join(["failed"] * 220)

    seeds = _historical_rewrite_seed_texts(
        {
            "final_text": original,
            "summary": {
                "paragraph_obligation_hard_stop": {
                    "active": True,
                    "blocked_findings": ["unsafe_density"],
                },
                "candidate_generation_status": {
                    "reason": "unresolved_paragraph_findings",
                },
                "candidate_ledger": [{
                    "rank": 1,
                    "source": "failed_paragraph_candidate",
                    "section_id": "full_document",
                    "text": failed_candidate,
                }],
                "rewrite_layers": {
                    "v5_residual_cluster_comb": {
                        "global_best_fallback": {
                            "selected": {
                                "section_id": "full_document",
                                "text": failed_candidate,
                            }
                        }
                    }
                },
            },
        },
        original,
        limit=3,
    )

    assert seeds == []


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


def test_bounded_rewrite_json_preserves_author_proxy_kpi_status():
    ledger_text = " ".join(["candidate-ledger-text"] * 100)
    payload = {
        "status": "rewrite_candidate_generated_needs_author_review",
        "elapsed": 1.0,
        "original_text": "source",
        "final_text": "final",
        "summary": {
            "outcome": "rewrite_candidate_generated_needs_author_review",
            "public_status": "rewrite_candidate_generated_needs_author_review",
            "public_candidate_warning": "author_proxy_candidate_requires_review",
            "best_candidate_author_review_required": True,
            "best_candidate_external_review_required": False,
            "strict_safe_band_achieved": True,
            "kpi_finalization_status": "strict_safe_author_review_required",
            "strict_goal_status": "ai_mitigated",
            "rewrite_goal_status": {"status": "rewrite_candidate_generated_needs_author_review", "goal_met": True},
            "author_review_cards": [{"card_id": "target-01", "target_text": "confirm source detail"}],
            "candidate_ledger": [
                {
                    "rank": 1,
                    "source": "global_best_fallback",
                    "section_id": "full_document",
                    "variant_id": "seed_01",
                    "text": ledger_text,
                    "scores": {"ai": 31.0, "topk_calibrated_risk": 24.8},
                    "goal": {"status": "ai_mitigated", "goal_met": True},
                }
            ],
            "candidate_loop_trace": [{"candidate_report": {"blob": "x" * 2000}} for _ in range(300)],
            "rewrite_layers": {"full_trace": [{"candidate_report": "y" * 2000} for _ in range(300)]},
        },
        "sentence_comparison": [{"original": "a" * 2000, "rewritten": "b" * 2000} for _ in range(300)],
    }

    bounded = _bounded_rewrite_json_payload(payload, max_bytes=12000)
    summary = bounded["summary"]

    assert bounded["rewrite_json_truncated"] is True
    assert summary["strict_safe_band_achieved"] is True
    assert summary["kpi_finalization_status"] == "strict_safe_author_review_required"
    assert summary["best_candidate_author_review_required"] is True
    assert summary["best_candidate_external_review_required"] is False
    assert summary["author_review_cards"][0]["card_id"] == "target-01"
    assert summary["candidate_ledger"][0]["text"] == ledger_text
    assert summary["candidate_ledger"][0]["scores"]["topk_calibrated_risk"] == 24.8
    assert "rewrite_layers" not in summary


def test_bounded_rewrite_json_preserves_paragraph_hard_stop():
    payload = {
        "status": "mitigation_failed_no_safe_candidate",
        "elapsed": 1.0,
        "original_text": "source",
        "final_text": "source",
        "summary": {
            "outcome": "mitigation_failed_no_safe_candidate",
            "public_status": "mitigation_failed_no_safe_candidate",
            "candidate_generation_status": {
                "reason": "unresolved_paragraph_findings",
                "blocked_findings": ["unsafe_density"],
            },
            "paragraph_obligation_hard_stop": {
                "active": True,
                "blocked_findings": ["unsafe_density"],
                "evidence_ledger": [{"finding_tag": "unsafe_density"}],
            },
            "candidate_loop_trace": [{"candidate_report": {"blob": "x" * 2000}} for _ in range(300)],
        },
        "sentence_comparison": [{"original": "a" * 2000, "rewritten": "b" * 2000} for _ in range(300)],
    }

    bounded = _bounded_rewrite_json_payload(payload, max_bytes=5000)
    summary = bounded["summary"]

    assert bounded["rewrite_json_truncated"] is True
    assert summary["paragraph_obligation_hard_stop"]["active"] is True
    assert summary["paragraph_obligation_hard_stop"]["blocked_findings"] == ["unsafe_density"]
    assert summary["candidate_generation_status"]["reason"] == "unresolved_paragraph_findings"


def test_bounded_rewrite_json_preserves_rewritten_score_profile_signals():
    core_signals = [
        {"key": "patchwork_variance", "name": "Patchwork variance", "value": 85},
        {"key": "topk_pattern_raw", "name": "Raw Top-k Predictability", "value": 71},
        {"key": "semantic_uniformity", "name": "Semantic uniformity", "value": 48},
        {"key": "expansion_pattern", "name": "Expansion pattern", "value": 46},
        {"key": "ai_likelihood", "name": "AI likelihood", "value": 43},
        {"key": "discourse_regularity", "name": "Discourse regularity", "value": 42},
        {"key": "rewrite_smoothness", "name": "Rewrite smoothness", "value": 37},
        {"key": "topk_calibrated_risk", "name": "Calibrated Top-k Risk", "value": 28},
        {"key": "human_anchor", "name": "Human anchor", "value": 36},
        {"key": "grounding_risk", "name": "Grounding risk", "value": 80},
    ]
    payload = {
        "status": "rewritten",
        "elapsed": 1.0,
        "original_text": "source",
        "final_text": "rewritten",
        "summary": {
            "outcome": "ai_mitigated",
            "detect_scan_rewritten": {
                "ai_score": 43,
                "writing_score": 72,
                "ai_risk_badge": {
                    "ai_likelihood_score": 43,
                    "writing_quality_score": 72,
                    "authorship_rating": "ai_mitigated",
                    "transformation_classification": {"pattern": "rewritten"},
                    "ai_components": {
                        "topk_pattern_raw": 71,
                        "topk_calibrated_risk": 28,
                        "ai_likelihood": 43,
                    },
                },
                "scan_intelligence": {
                    "transformation": {
                        "classification": {"pattern": "rewritten"},
                        "contribution": {"human": 83, "ai": 17},
                        "core_signals": core_signals,
                    },
                    "document": {"word_count": 500, "sentence_count": 30},
                },
                "findings": {"low": [{"finding_id": "f1", "title": "Low signal", "text": "x" * 5000}]},
            },
            "candidate_loop_trace": [{"candidate_report": {"blob": "x" * 2000}} for _ in range(300)],
        },
        "sentence_comparison": [{"original": "a" * 2000, "rewritten": "b" * 2000} for _ in range(300)],
    }

    bounded = _bounded_rewrite_json_payload(payload, max_bytes=12000)
    rewritten_scan = bounded["summary"]["detect_scan_rewritten"]
    rewritten_signals = rewritten_scan["scan_intelligence"]["transformation"]["core_signals"]

    assert bounded["rewrite_json_truncated"] is True
    assert [signal["key"] for signal in rewritten_signals] == [signal["key"] for signal in core_signals]
    assert rewritten_scan["ai_risk_badge"]["ai_components"]["topk_pattern_raw"] == 71
    assert rewritten_scan["scan_intelligence"]["transformation"]["contribution"]["human"] == 83


def test_bounded_debug_log_preserves_author_proxy_kpi_status():
    debug_text = _bounded_json_debug_log(
        {
            "debug_export_version": "test",
            "rewrite_id": "rw_test",
            "scan_id": "scan_test",
            "input_scan": {"findings": [{"detail": "x" * 2000} for _ in range(300)]},
            "rewrite_summary": {
                "outcome": "rewrite_candidate_generated_needs_author_review",
                "public_status": "rewrite_candidate_generated_needs_author_review",
                "public_candidate_warning": "author_proxy_candidate_requires_review",
                "best_candidate_author_review_required": True,
                "best_candidate_external_review_required": False,
                "strict_safe_band_achieved": True,
                "kpi_finalization_status": "strict_safe_author_review_required",
                "strict_goal_status": "ai_mitigated",
                "candidate_loop_trace": [{"candidate_report": {"blob": "y" * 2000}} for _ in range(300)],
            },
        },
        max_bytes=4096,
    )

    assert len(debug_text.encode("utf-8")) <= 4096
    assert '"debug_truncated": true' in debug_text
    assert '"strict_safe_band_achieved": true' in debug_text
    assert '"kpi_finalization_status": "strict_safe_author_review_required"' in debug_text
