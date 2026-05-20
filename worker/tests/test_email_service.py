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
    payload = build_scan_completion_email(
        recipient_email="student@example.com",
        scan_id="scan-1",
        tier="possible AI-assisted",
        ai_score=42.5,
        writing_score=81,
        finding_count=3,
        pdf_bytes=b"%PDF-1.7 scan",
        settings=_settings(),
    )

    assert payload["to"] == "student@example.com"
    assert payload["from"] == "DraftProof Support <support@draftproof.app>"
    assert payload["subject"] == "Your DraftProof scan report is ready"
    assert "Scan ID: scan-1" in payload["text"]
    assert "Report outcome: possible AI-assisted" in payload["text"]
    assert "AI likelihood score: 42.5%" in payload["text"]
    assert "Writing score: 81%" in payload["text"]
    assert "Findings: 3" in payload["text"]
    assert payload["attachments"] == [
        {
            "filename": "draftproof-scan-scan-1.pdf",
            "type": "application/pdf",
            "disposition": "attachment",
            "content": "JVBERi0xLjcgc2Nhbg==",
        }
    ]


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
