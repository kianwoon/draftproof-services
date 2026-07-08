from types import SimpleNamespace

from app.email_service import (
    build_rewrite_completion_email,
    build_scan_completion_email,
    send_email,
    send_rewrite_completion_email,
    send_scan_completion_email,
)


def _settings(**overrides):
    defaults = {
        "REWRITE_COMPLETION_EMAIL_ENABLED": True,
        "SCAN_COMPLETION_EMAIL_ENABLED": True,
        "EMAIL_PROVIDER": "cloudflare",
        "EMAIL_FROM_ADDRESS": "support@draftproof.app",
        "EMAIL_FROM_NAME": "DraftProof Support",
        "CLOUDFLARE_ACCOUNT_ID": "account-id",
        "CLOUDFLARE_ACCOUNT_API_TOKEN": "token",
        "CLOUDFLARE_EMAIL_SEND_ENDPOINT": "https://example.test/send",
        "REWRITE_COMPLETION_EMAIL_MAX_CHARS": 50000,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_rewrite_completion_email_includes_final_text():
    payload = build_rewrite_completion_email(
        recipient_email="student@example.com",
        rewrite_id="rewrite-1",
        scan_id="scan-1",
        final_text="Rewritten content here.",
        settings=_settings(),
    )

    assert payload["to"] == "student@example.com"
    assert payload["from"] == "DraftProof Support <support@draftproof.app>"
    assert payload["subject"] == "Your DraftProof rewrite is complete"
    assert "Rewritten content here." in payload["text"]
    assert "rewrite-1" in payload["text"]
    assert "scan-1" in payload["text"]
    assert "keeps rewrite report records in the system for 3 days" in payload["text"]


def test_build_rewrite_completion_email_attaches_pdf():
    payload = build_rewrite_completion_email(
        recipient_email="student@example.com",
        rewrite_id="rewrite-1",
        scan_id="scan-1",
        final_text="Rewritten content here.",
        pdf_bytes=b"%PDF-1.7 example",
        settings=_settings(),
    )

    assert payload["attachments"] == [
        {
            "filename": "draftproof-rewrite-rewrite-1.pdf",
            "type": "application/pdf",
            "disposition": "attachment",
            "content": "JVBERi0xLjcgZXhhbXBsZQ==",
        }
    ]


def test_build_rewrite_completion_email_leads_with_v7_fused_before_after():
    # Mirrors the /rewrite page + PDF hero lead (poc/report/render_rewrite.py _ai_score /
    # _deep_scan_pct): fused ai_likelihood_score before->after, plus composite + deep-scan
    # evidence when tier_authority (V7) is present. Fixture matches the live rewrite
    # 9a29e56a numbers from commit cb8ddfa7 (fused 16.29 -> 6.82, deep 23.8% -> 8.7%).
    rewrite_summary = {
        "detect_scan_original_saved": {
            "ai_score": 16.29,
            "ai_risk_badge": {
                "ai_likelihood_score": 16.29,
                "tier_authority": {
                    "source": "v7_fused", "fused_score": 16.29, "composite_score": 5.0, "proportion": 0.238,
                },
            },
        },
        "detect_scan_rewritten": {
            "ai_score": 6.82,
            "ai_risk_badge": {
                "ai_likelihood_score": 6.82,
                "tier_authority": {
                    "source": "v7_fused", "fused_score": 6.82, "composite_score": 4.0, "proportion": 0.087,
                },
            },
        },
    }
    payload = build_rewrite_completion_email(
        recipient_email="student@example.com",
        rewrite_id="rewrite-1",
        scan_id="scan-1",
        final_text="Rewritten content here.",
        rewrite_summary=rewrite_summary,
        settings=_settings(),
    )

    assert "AI likelihood: 16% -> 7%" in payload["text"]
    assert "Evidence: composite 5 -> 4, deep-scan 23.8% -> 8.7%" in payload["text"]


def test_build_rewrite_completion_email_legacy_before_after_without_tier_authority():
    # Legacy rewrite (no V7 tier_authority block): still shows the before/after AI-likelihood
    # lead from the plain ai_score, but never fabricates composite/deep-scan evidence lines.
    rewrite_summary = {
        "detect_scan_original_saved": {"ai_score": 40.0, "ai_risk_badge": {"ai_likelihood_score": 40.0}},
        "detect_scan_rewritten": {"ai_score": 12.0, "ai_risk_badge": {"ai_likelihood_score": 12.0}},
    }
    payload = build_rewrite_completion_email(
        recipient_email="student@example.com",
        rewrite_id="rewrite-1",
        scan_id="scan-1",
        final_text="Rewritten content here.",
        rewrite_summary=rewrite_summary,
        settings=_settings(),
    )

    assert "AI likelihood: 40% -> 12%" in payload["text"]
    assert "Evidence:" not in payload["text"]


def test_build_rewrite_completion_email_no_summary_omits_fused_lead():
    # No rewrite_summary at all (absent detect-scan comparison) -> fails open to the plain
    # rewritten-content email; must not crash and must not print a fabricated lead.
    payload = build_rewrite_completion_email(
        recipient_email="student@example.com",
        rewrite_id="rewrite-1",
        scan_id="scan-1",
        final_text="Rewritten content here.",
        rewrite_summary=None,
        settings=_settings(),
    )

    assert "AI likelihood:" not in payload["text"]
    assert "Evidence:" not in payload["text"]
    assert "Rewritten content here." in payload["text"]


def test_build_rewrite_completion_email_truncates_large_content():
    payload = build_rewrite_completion_email(
        recipient_email="student@example.com",
        rewrite_id="rewrite-1",
        scan_id="scan-1",
        final_text="x" * 1200,
        settings=_settings(REWRITE_COMPLETION_EMAIL_MAX_CHARS=1000),
    )

    assert "[Content truncated in email." in payload["text"]


def test_build_scan_completion_email_attaches_report_pdf():
    # Email leads with the web report's lead: Submission risk + policy scores.
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="scan-1",
        tier="possible AI-assisted",
        ai_score=42.5,
        authorship_rating_label="Possible AI-Assisted",
        submission_risk={"overall": {"level": "medium", "main_reason": "Several unattributed claims"}},
        policy_risk={
            "ai_allowed": {"level": "moderate", "score": 41.4},
            "ai_restricted": {"level": "high", "score": 67.6},
        },
        writing_score=81,
        finding_count=3,
        pdf_bytes=b"%PDF-1.7 scan",
        settings=_settings(),
    )

    assert payload["to"] == "student@example.com"
    assert payload["from"] == "DraftProof Support <support@draftproof.app>"
    assert payload["subject"] == "Your DraftProof scan report is ready"
    assert "Scan ID: scan-1" in payload["text"]
    assert "Submission risk: Medium — Several unattributed claims" in payload["text"]
    assert "If AI is allowed (with declaration): Moderate (41)" in payload["text"]
    assert "If AI is not allowed: High (68)" in payload["text"]
    assert "Writing score: 81%" in payload["text"]
    assert "Findings: 3" in payload["text"]
    assert "keeps scan report records in the system for 3 days" in payload["text"]
    assert "attached PDF" in payload["text"]
    assert payload["attachments"] == [
        {
            "filename": "draftproof-scan-scan-1.pdf",
            "type": "application/pdf",
            "disposition": "attachment",
            "content": "JVBERi0xLjcgc2Nhbg==",
        }
    ]


def test_build_scan_completion_email_formats_page_aligned_lead():
    # Policy scores are rounded; the non-accusatory disclaimer accompanies them.
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="4957c85d-f913-40eb-892c-addf0850b02f",
        tier="amber",
        ai_score=36.66,
        authorship_rating_label=None,
        submission_risk={"overall": {"level": "high", "main_reason": "Generic, unanchored assertions"}},
        policy_risk={
            "ai_allowed": {"level": "low", "score": 12.6},
            "ai_restricted": {"level": "severe", "score": 88.4},
        },
        writing_score=44.05,
        finding_count=37,
        pdf_bytes=b"%PDF-1.7 scan",
        settings=_settings(),
    )

    assert "Submission risk: High — Generic, unanchored assertions" in payload["text"]
    assert "If AI is allowed (with declaration): Low (13)" in payload["text"]  # round(12.6)
    assert "If AI is not allowed: Severe (88)" in payload["text"]  # round(88.4)
    assert "These scores do not prove AI use" in payload["text"]
    assert "Writing score: 44.05%" in payload["text"]
    assert "Findings: 37" in payload["text"]


def test_build_scan_completion_email_does_not_surface_legacy_ai_score_lines():
    # ai_score / authorship_rating_label are accepted for backward compatibility
    # but intentionally NOT rendered (email mirrors the V7 page lead instead).
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="181952b4-73bd-4a66-bb4f-c89d0e8d2aa9",
        tier="amber",
        ai_score=42,
        authorship_rating_label="Possible AI-Assisted",
        writing_score=59.33,
        finding_count=28,
        pdf_bytes=b"%PDF-1.7 scan",
        settings=_settings(),
    )

    assert "Report outcome:" not in payload["text"]
    assert "AI likelihood score:" not in payload["text"]
    assert "Possible AI-Assisted" not in payload["text"]
    assert "Writing score: 59.33%" in payload["text"]
    assert "Findings: 28" in payload["text"]


def test_build_scan_completion_email_does_not_surface_turnitin_estimate():
    # external_estimate is accepted for backward compatibility but no longer rendered.
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="scan-ext",
        tier="amber",
        ai_score=42,
        authorship_rating_label="Possible AI-Assisted",
        external_estimate={"score": 59.8, "band": "high"},
        writing_score=81,
        finding_count=3,
        pdf_bytes=b"%PDF-1.7 scan",
        settings=_settings(),
    )

    assert "Turnitin" not in payload["text"]
    assert "~60%" not in payload["text"]
    assert "Scan ID: scan-ext" in payload["text"]


def test_build_scan_completion_email_omits_turnitin_when_no_estimate():
    # no external_estimate → no Turnitin line
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="scan-noext",
        tier="amber",
        ai_score=42,
        authorship_rating_label="Possible AI-Assisted",
        writing_score=81,
        finding_count=3,
        pdf_bytes=b"%PDF-1.7 scan",
        settings=_settings(),
    )

    assert "Turnitin / external estimate:" not in payload["text"]


def test_send_rewrite_completion_email_skips_when_disabled(monkeypatch):
    sent = []
    monkeypatch.setattr("app.email_service.send_email", lambda payload, *, settings: sent.append(payload))

    result = send_rewrite_completion_email(
        recipient_email="student@example.com",
        rewrite_id="rewrite-1",
        scan_id="scan-1",
        final_text="Rewritten content here.",
        pdf_bytes=b"%PDF-1.7 example",
        settings=_settings(REWRITE_COMPLETION_EMAIL_ENABLED=False),
    )

    assert result is False
    assert sent == []


def test_send_scan_completion_email_skips_when_disabled(monkeypatch):
    sent = []
    monkeypatch.setattr("app.email_service.send_email", lambda payload, *, settings: sent.append(payload))

    result = send_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="scan-1",
        pdf_bytes=b"%PDF-1.7 scan",
        settings=_settings(SCAN_COMPLETION_EMAIL_ENABLED=False),
    )

    assert result is False
    assert sent == []


def test_send_scan_completion_email_skips_without_pdf(monkeypatch):
    sent = []
    monkeypatch.setattr("app.email_service.send_email", lambda payload, *, settings: sent.append(payload))

    result = send_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="scan-1",
        pdf_bytes=b"",
        settings=_settings(),
    )

    assert result is False
    assert sent == []


def test_send_email_posts_to_cloudflare(monkeypatch):
    calls = []

    class Response:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {"success": True}

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr("app.email_service.requests.post", fake_post)

    result = send_email({"to": "student@example.com"}, settings=_settings())

    assert result is True
    assert calls[0]["url"] == "https://example.test/send"
    assert calls[0]["headers"]["Authorization"] == "Bearer token"
    assert calls[0]["json"] == {"to": "student@example.com"}
    assert calls[0]["timeout"] == 20


def test_send_email_returns_false_on_cloudflare_failure(monkeypatch):
    class Response:
        ok = False
        status_code = 401

        @staticmethod
        def json():
            return {"success": False, "errors": [{"message": "auth failed"}]}

    monkeypatch.setattr("app.email_service.requests.post", lambda *args, **kwargs: Response())

    assert send_email({"to": "student@example.com"}, settings=_settings()) is False


def test_scan_email_asserts_ai_category_when_reliable_and_deep_scan_ran():
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="s-1",
        pdf_bytes=b"%PDF-1.7 scan",
        authorship_breakdown={
            "primary_category": "ai_generated_like",
            "primary_category_reliable": True,
            "deep_scan": {"proportion": 0.8, "band": "red", "calibrated": True},
        },
        settings=_settings(),
    )
    assert "Authorship read: mostly AI-generated-like" in payload["text"]


def test_scan_email_withholds_ai_category_without_deep_scan():
    # Quick-scan-only (Modal fallback): never assert an AI-flavored category.
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="s-1b",
        pdf_bytes=b"%PDF-1.7 scan",
        authorship_breakdown={
            "primary_category": "ai_generated_like",
            "primary_category_reliable": True,
        },
        settings=_settings(),
    )
    assert "mostly AI-generated-like" not in payload["text"]
    assert "Authorship read: mixed signals" in payload["text"]


def test_scan_email_student_owned_needs_no_deep_scan():
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="s-1c",
        pdf_bytes=b"%PDF-1.7 scan",
        authorship_breakdown={
            "primary_category": "student_owned",
            "primary_category_reliable": True,
        },
        settings=_settings(),
    )
    assert "Authorship read: mostly student-owned" in payload["text"]


def test_scan_email_neutral_line_when_category_unreliable():
    # Measured 2026-07-06: 58% of human ESL essays get ai_generated_like as
    # primary on the quick-scan path — the email must not assert it when the
    # breakdown's own guards (primary_category_reliable=False) say not to.
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="s-2",
        pdf_bytes=b"%PDF-1.7 scan",
        authorship_breakdown={
            "primary_category": "ai_generated_like",
            "primary_category_reliable": False,
        },
        settings=_settings(),
    )
    assert "mostly AI-generated-like" not in payload["text"]
    assert "Authorship read: mixed signals" in payload["text"]


def test_scan_email_legacy_breakdown_without_reliability_field_keeps_old_behavior():
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="s-3",
        pdf_bytes=b"%PDF-1.7 scan",
        authorship_breakdown={"primary_category": "student_owned"},
        settings=_settings(),
    )
    assert "Authorship read: mostly student-owned" in payload["text"]


def test_scan_email_uses_merged_display_primary_when_three_way():
    # V8 three-way display fallback (poc/detect_v7/pipeline_bridge.py::
    # _compose_display_fallback): display_taxonomy == "three_way" -> the email claim
    # uses the merged display_primary term ("ai_transformed"), not the raw four-way
    # primary_category, so it matches what the report page/PDF show.
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="s-4",
        pdf_bytes=b"%PDF-1.7 scan",
        authorship_breakdown={
            "primary_category": "ai_generated_like",
            "primary_category_reliable": True,
            "display_taxonomy": "three_way",
            "display_primary": "ai_transformed",
            "deep_scan": {"proportion": 0.8, "band": "red", "calibrated": True},
        },
        settings=_settings(),
    )
    assert "Authorship read: mostly AI-transformed" in payload["text"]
    assert "mostly AI-generated-like" not in payload["text"]


def test_scan_email_three_way_absent_keeps_four_way_category():
    # No display_* fields (legacy payload / mode "four_way") -> byte-identical
    # four-way behavior, unaffected by the three-way branch.
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="s-5",
        pdf_bytes=b"%PDF-1.7 scan",
        authorship_breakdown={
            "primary_category": "ai_generated_like",
            "primary_category_reliable": True,
            "deep_scan": {"proportion": 0.8, "band": "red", "calibrated": True},
        },
        settings=_settings(),
    )
    assert "Authorship read: mostly AI-generated-like" in payload["text"]
