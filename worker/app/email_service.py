"""Best-effort transactional email helpers for worker-side events."""

import base64
import logging
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


class EmailConfigurationError(RuntimeError):
    """Raised when email sending is enabled but required settings are missing."""


def _cloudflare_send_endpoint(settings) -> str:
    if settings.CLOUDFLARE_EMAIL_SEND_ENDPOINT:
        return settings.CLOUDFLARE_EMAIL_SEND_ENDPOINT
    if not settings.CLOUDFLARE_ACCOUNT_ID:
        raise EmailConfigurationError("CLOUDFLARE_ACCOUNT_ID is required")
    account_id = quote(settings.CLOUDFLARE_ACCOUNT_ID, safe="")
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/email/sending/send"


def _from_header(settings) -> str:
    address = (settings.EMAIL_FROM_ADDRESS or "").strip()
    if not address:
        raise EmailConfigurationError("EMAIL_FROM_ADDRESS is required")
    name = (settings.EMAIL_FROM_NAME or "").strip()
    return f"{name} <{address}>" if name else address


def build_rewrite_completion_email(
    *,
    recipient_email: str,
    rewrite_id: str,
    scan_id: str,
    final_text: str,
    pdf_bytes: bytes | None = None,
    pdf_filename: str | None = None,
    settings,
) -> dict:
    max_chars = max(1000, int(settings.REWRITE_COMPLETION_EMAIL_MAX_CHARS or 50000))
    normalized_text = str(final_text or "").strip()
    truncated = len(normalized_text) > max_chars
    delivered_text = normalized_text[:max_chars].rstrip()
    if truncated:
        delivered_text = f"{delivered_text}\n\n[Content truncated in email. Please open DraftProof to view the full rewrite.]"

    subject = "Your DraftProof rewrite is complete"
    text = (
        "Hi,\n\n"
        "Your DraftProof rewrite is complete. The rewritten content is below.\n\n"
        f"Rewrite ID: {rewrite_id}\n"
        f"Scan ID: {scan_id}\n\n"
        "Rewritten content\n"
        "-----------------\n"
        f"{delivered_text}\n\n"
        "DraftProof Support\n"
        f"{settings.EMAIL_FROM_ADDRESS}"
    )
    payload = {
        "to": recipient_email,
        "from": _from_header(settings),
        "subject": subject,
        "text": text,
    }
    if pdf_bytes:
        payload["attachments"] = [
            {
                "filename": pdf_filename or f"draftproof-rewrite-{rewrite_id}.pdf",
                "type": "application/pdf",
                "disposition": "attachment",
                "content": base64.b64encode(pdf_bytes).decode("ascii"),
            }
        ]
    return payload


def build_scan_completion_email(
    *,
    recipient_email: str,
    scan_id: str,
    tier: str | None = None,
    ai_score: float | int | None = None,
    writing_score: float | int | None = None,
    finding_count: int | None = None,
    pdf_bytes: bytes | None = None,
    pdf_filename: str | None = None,
    settings,
) -> dict:
    details = [f"Scan ID: {scan_id}"]
    if tier:
        details.append(f"Report outcome: {tier}")
    if ai_score is not None:
        details.append(f"AI likelihood score: {ai_score}%")
    if writing_score is not None:
        details.append(f"Writing score: {writing_score}%")
    if finding_count is not None:
        details.append(f"Findings: {finding_count}")
    details_text = "\n".join(details)

    subject = "Your DraftProof scan report is ready"
    text = (
        "Hi,\n\n"
        "Your DraftProof scan is complete. A PDF copy of your report is attached.\n\n"
        f"{details_text}\n\n"
        "You can also open DraftProof to review the full interactive report and start a guided rewrite when available.\n\n"
        "DraftProof Support\n"
        f"{settings.EMAIL_FROM_ADDRESS}"
    )
    payload = {
        "to": recipient_email,
        "from": _from_header(settings),
        "subject": subject,
        "text": text,
    }
    if pdf_bytes:
        payload["attachments"] = [
            {
                "filename": pdf_filename or f"draftproof-scan-{scan_id}.pdf",
                "type": "application/pdf",
                "disposition": "attachment",
                "content": base64.b64encode(pdf_bytes).decode("ascii"),
            }
        ]
    return payload


def send_email(payload: dict, *, settings) -> bool:
    if (settings.EMAIL_PROVIDER or "").strip().lower() != "cloudflare":
        raise EmailConfigurationError("Only EMAIL_PROVIDER=cloudflare is supported")
    if not settings.CLOUDFLARE_ACCOUNT_API_TOKEN:
        raise EmailConfigurationError("CLOUDFLARE_ACCOUNT_API_TOKEN is required")

    response = requests.post(
        _cloudflare_send_endpoint(settings),
        headers={
            "Authorization": f"Bearer {settings.CLOUDFLARE_ACCOUNT_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=20,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"success": False, "errors": [{"message": response.text[:500]}]}

    if not response.ok or not body.get("success"):
        logger.warning(
            "Cloudflare email send failed: status=%s errors=%s",
            response.status_code,
            body.get("errors"),
        )
        return False
    return True


def send_rewrite_completion_email(
    *,
    recipient_email: str | None,
    rewrite_id: str,
    scan_id: str,
    final_text: str,
    pdf_bytes: bytes | None = None,
    settings,
) -> bool:
    if not settings.REWRITE_COMPLETION_EMAIL_ENABLED:
        return False
    if not recipient_email:
        logger.info("Skipping rewrite completion email for %s: missing recipient", rewrite_id)
        return False
    if not str(final_text or "").strip():
        logger.info("Skipping rewrite completion email for %s: missing final text", rewrite_id)
        return False

    try:
        payload = build_rewrite_completion_email(
            recipient_email=recipient_email,
            rewrite_id=rewrite_id,
            scan_id=scan_id,
            final_text=final_text,
            pdf_bytes=pdf_bytes,
            settings=settings,
        )
        return send_email(payload, settings=settings)
    except Exception:
        logger.warning("Rewrite completion email failed for %s", rewrite_id, exc_info=True)
        return False


def send_scan_completion_email(
    *,
    recipient_email: str | None,
    scan_id: str,
    tier: str | None = None,
    ai_score: float | int | None = None,
    writing_score: float | int | None = None,
    finding_count: int | None = None,
    pdf_bytes: bytes | None = None,
    settings,
) -> bool:
    if not settings.SCAN_COMPLETION_EMAIL_ENABLED:
        return False
    if not recipient_email:
        logger.info("Skipping scan completion email for %s: missing recipient", scan_id)
        return False
    if not pdf_bytes:
        logger.info("Skipping scan completion email for %s: missing PDF", scan_id)
        return False

    try:
        payload = build_scan_completion_email(
            recipient_email=recipient_email,
            scan_id=scan_id,
            tier=tier,
            ai_score=ai_score,
            writing_score=writing_score,
            finding_count=finding_count,
            pdf_bytes=pdf_bytes,
            settings=settings,
        )
        return send_email(payload, settings=settings)
    except Exception:
        logger.warning("Scan completion email failed for %s", scan_id, exc_info=True)
        return False
