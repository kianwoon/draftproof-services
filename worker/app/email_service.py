"""Best-effort transactional email helpers for worker-side events."""

import base64
import logging
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

SCAN_REPORT_OUTCOME_LABELS = {
    "low": "Low Risk",
    "moderate": "Moderate Risk",
    "high": "High Risk",
    "green": "Low Risk",
    "amber": "Moderate Risk",
    "orange": "High Risk",
    "red": "Critical Risk",
}
_EXTERNAL_BAND_LABELS = {
    "low": "unlikely to be flagged",
    "elevated": "possibly flagged",
    "high": "likely to be flagged",
}

# Mirrors draftproof-frontend/src/i18n/en/report.js authorshipBreakdown.categories —
# keep the wording in sync with AuthorshipClarityBreakdown.jsx CATEGORY_ORDER labels.
_AUTHORSHIP_CATEGORY_LABELS = {
    "student_owned": "mostly student-owned",
    "ai_assisted_polished": "mostly AI-assisted / polished",
    "ai_paraphrased": "mostly AI-paraphrased",
    "ai_generated_like": "mostly AI-generated-like",
}

# Keep in sync with report.render.EXTERNAL_ESTIMATE_DISPLAY_ENABLED. The badge carries DraftProof's
# grouped external-detector proxy, so email surfaces it only as an estimate.
EXTERNAL_ESTIMATE_DISPLAY_ENABLED = True


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


def _metric_percent(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number * 100 if abs(number) <= 1 else number


def _format_score(value: float | int | None) -> str | None:
    percent = _metric_percent(value)
    if percent is None:
        return None
    return f"{percent:.2f}".rstrip("0").rstrip(".")


def _format_display_ai_score(value: float | int | None) -> str | None:
    percent = _metric_percent(value)
    if percent is None:
        return None
    return f"{percent:.0f}"


def _scan_report_outcome_label(tier: str | None) -> str | None:
    if not tier:
        return None
    normalized = str(tier).strip().lower()
    return SCAN_REPORT_OUTCOME_LABELS.get(normalized) or str(tier).strip()


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
        "DraftProof keeps rewrite report records in the system for 3 days. Keep this email "
        "as your own copy after the system copy expires.\n\n"
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


# allow-hardcode: presentation level labels (machine level -> display), not a
# scoring/matching oracle. Mirror render.py / the web report.
_SR_LEVEL_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "unknown": "Unknown"}
_POLICY_LEVEL = {"low": "Low", "moderate": "Moderate", "high": "High", "severe": "Severe", "unknown": "Unknown"}


def build_scan_completion_email(
    *,
    recipient_email: str,
    scan_id: str,
    tier: str | None = None,
    ai_score: float | int | None = None,
    authorship_rating_label: str | None = None,
    external_estimate: dict | None = None,
    writing_score: float | int | None = None,
    finding_count: int | None = None,
    submission_risk: dict | None = None,
    policy_risk: dict | None = None,
    authorship_breakdown: dict | None = None,
    pdf_bytes: bytes | None = None,
    pdf_filename: str | None = None,
    settings,
) -> dict:
    # Mirror the web report's LEAD: Submission risk + the two policy scores. The raw
    # AI-likelihood %, Turnitin estimate, and authorship rating are no longer surfaced
    # (ai_score / external_estimate / authorship_rating_label kept in the signature for
    # backward compatibility but intentionally not rendered).
    details = [f"Scan ID: {scan_id}"]
    sr = (submission_risk or {}).get("overall") or {}
    if sr.get("level") and sr["level"] != "unknown":
        line = f"Submission risk: {_SR_LEVEL_LABELS.get(sr['level'], sr['level'])}"
        if sr.get("main_reason"):
            line += f" — {sr['main_reason']}"
        details.append(line)
    pr = policy_risk or {}
    pa, prr = pr.get("ai_allowed") or {}, pr.get("ai_restricted") or {}
    if pa.get("level") and pa["level"] != "unknown":
        details.append(
            f"If AI is allowed (with declaration): {_POLICY_LEVEL.get(pa['level'], pa['level'])} ({round(pa.get('score') or 0)})"
        )
        details.append(
            f"If AI is not allowed: {_POLICY_LEVEL.get(prr.get('level'), prr.get('level'))} ({round(prr.get('score') or 0)})"
        )
        details.append(
            "These scores do not prove AI use; they estimate how the draft may read under different school policies."
        )
    formatted_writing_score = _format_score(writing_score)
    if formatted_writing_score is not None:
        details.append(f"Writing score: {formatted_writing_score}%")
    if finding_count is not None:
        details.append(f"Findings: {finding_count}")
    # Additive one-line authorship-clarity summary (fail-open: absent on older
    # reports / flag off). Mirrors the /report page's Authorship Clarity
    # Breakdown centerpiece without dumping all 4 category shares into the email.
    # Only ASSERT the category when the breakdown itself marks it reliable
    # (primary_category_reliable, derived from its confidence + degraded-display
    # guards): measured 2026-07-06, 58% of human ESL essays got
    # "ai_generated_like" as primary on the quick-scan path — stating that in
    # an email is a false accusation the report page's banded display avoids.
    # Older reports without the field keep the previous behavior (is not False).
    bd = authorship_breakdown or {}
    primary_category = bd.get("primary_category")
    if primary_category and primary_category in _AUTHORSHIP_CATEGORY_LABELS:
        if bd.get("primary_category_reliable") is not False:
            details.append(f"Authorship read: {_AUTHORSHIP_CATEGORY_LABELS[primary_category]}")
        else:
            details.append(
                "Authorship read: mixed signals — no single category is reliable; "
                "see the attached report for the banded breakdown."
            )
    details_text = "\n".join(details)

    subject = "Your DraftProof scan report is ready"
    text = (
        "Hi,\n\n"
        "Your DraftProof scan is complete. A PDF copy of your report is attached.\n\n"
        f"{details_text}\n\n"
        "DraftProof keeps scan report records in the system for 3 days. Keep this email or the attached PDF "
        "as your own copy after the system copy expires.\n\n"
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

    # success=true means Cloudflare ACCEPTED the message. The per-recipient
    # result.{delivered,queued,permanent_bounces} arrays are often EMPTY on a normal
    # accept (delivery is reported asynchronously), so empty arrays are NOT a failure.
    # Only an explicit permanent_bounce for this recipient is a real failure.
    result = body.get("result") or {}
    recipient = payload.get("to")
    delivered = result.get("delivered") or []
    queued = result.get("queued") or []
    bounces = result.get("permanent_bounces") or []
    if recipient and recipient in bounces:
        logger.warning(
            "Cloudflare permanently bounced %s (delivered=%s queued=%s)",
            recipient, delivered, queued,
        )
        return False
    logger.info(
        "Cloudflare email accepted for %s: delivered=%s queued=%s permanent_bounces=%s",
        recipient, delivered, queued, bounces,
    )
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
        logger.warning("Skipping rewrite completion email for %s: missing recipient", rewrite_id)
        return False
    if not str(final_text or "").strip():
        logger.warning("Skipping rewrite completion email for %s: missing final text", rewrite_id)
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
    authorship_rating_label: str | None = None,
    external_estimate: dict | None = None,
    writing_score: float | int | None = None,
    finding_count: int | None = None,
    submission_risk: dict | None = None,
    policy_risk: dict | None = None,
    authorship_breakdown: dict | None = None,
    pdf_bytes: bytes | None = None,
    settings,
) -> bool:
    if not settings.SCAN_COMPLETION_EMAIL_ENABLED:
        return False
    if not recipient_email:
        logger.warning("Skipping scan completion email for %s: missing recipient", scan_id)
        return False
    if not pdf_bytes:
        logger.warning("Skipping scan completion email for %s: missing PDF", scan_id)
        return False

    try:
        payload = build_scan_completion_email(
            recipient_email=recipient_email,
            scan_id=scan_id,
            tier=tier,
            ai_score=ai_score,
            authorship_rating_label=authorship_rating_label,
            external_estimate=external_estimate,
            writing_score=writing_score,
            finding_count=finding_count,
            submission_risk=submission_risk,
            policy_risk=policy_risk,
            authorship_breakdown=authorship_breakdown,
            pdf_bytes=pdf_bytes,
            settings=settings,
        )
        return send_email(payload, settings=settings)
    except Exception:
        logger.warning("Scan completion email failed for %s", scan_id, exc_info=True)
        return False
