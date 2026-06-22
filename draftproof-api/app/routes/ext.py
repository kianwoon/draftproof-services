"""Extension API — key-authenticated scanning for the Google Docs / MS Word add-ins.

Mounted at /api/ext. Reuses scan_service end to end: submit text, poll status.
Scans always bill credits (always_paid=True) and never touch the free quota.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.models.db import async_session, CreditAccount
from app.policy_enrich import _enrich_badge
from app.services.api_key_service import get_api_key_user
from app.services.report_service import get_report
from app.services.scan_service import create_scan, get_scan

# How many top findings to surface as "signal highlights" in the add-in.
EXT_SIGNAL_HIGHLIGHT_LIMIT = 8

router = APIRouter()

# Detector signals are unreliable below this; we still scan but flag the result
# so the client can label it low-confidence (honours expose-the-ugly-side).
MIN_CONFIDENT_WORDS = 100
# Match the web text-document cap (documents.py) so a key holder can't submit an
# unbounded body that reserves credits + loads the worker before any size check.
MAX_SCAN_CHARS = 50000


class ExtScanRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_SCAN_CHARS)
    # Optional source document name (e.g. the Word file name) — tags the report.
    document_name: str | None = Field(None, max_length=200)


@router.post("/scan")
async def ext_create_scan(req: ExtScanRequest, user: dict = Depends(get_api_key_user)):
    text = (req.text or "").strip()
    word_count = len(text.split())
    if word_count == 0:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        result = await create_scan(
            f"ext-{uuid.uuid4()}",      # free-form label, like the web "paste" id — text is inline, never read from disk
            user_id=user["id"],
            text=text,
            always_paid=True,
            title=req.document_name,    # tag the report with the source file name
        )
    except ValueError as exc:
        msg = str(exc)
        if "token" in msg.lower() or "credit" in msg.lower():
            raise HTTPException(status_code=402, detail=msg)
        raise HTTPException(status_code=400, detail=msg)

    return {
        "scan_id": result["id"],
        "status": result["status"],
        "word_count": word_count,
        "low_confidence": word_count < MIN_CONFIDENT_WORDS,
    }


@router.get("/scan/{scan_id}")
async def ext_scan_status(scan_id: str, user: dict = Depends(get_api_key_user)):
    result = await get_scan(scan_id, user_id=user["id"])  # already user-scoped
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found")
    return result


@router.get("/scan/{scan_id}/report")
async def ext_scan_report(scan_id: str, user: dict = Depends(get_api_key_user)):
    """Richer result for a completed scan: the two headline scores plus the
    Critical Thinking, Submitted-content (submission risk), and signal-highlight
    sections the web report shows. get_report is user-scoped; _enrich_badge fills
    the additive composers at read time (idempotent) if the badge lacks them."""
    report = await get_report(scan_id, user_id=user["id"])
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    badge = report.get("ai_risk_badge") or {}
    try:
        _enrich_badge(badge)  # no-op if already enriched
    except Exception:
        pass

    ctc = badge.get("critical_thinking_control") or {}
    sub_overall = (badge.get("submission_risk") or {}).get("overall") or {}
    issues = report.get("issues") or []

    return {
        "scan_id": scan_id,
        "tier": report.get("tier"),
        "ai_score": report.get("ai_score"),
        "writing_score": report.get("writing_score"),
        "critical_thinking": {
            "score": ctc.get("score"),
            "status": ctc.get("status"),
            "band": ctc.get("band"),
            "lead": ctc.get("lead_dimension_label"),
            "action": ctc.get("lead_dimension_action"),
            "caveat": ctc.get("caveat") or None,
            "dimensions": [
                {"label": d.get("label"), "control": d.get("control"), "gap": d.get("gap")}
                for d in (ctc.get("dimensions") or {}).values()
                if isinstance(d, dict) and d.get("control") is not None
            ],
        } if ctc else None,
        "submission_risk": {
            "level": sub_overall.get("level"),
            "label": sub_overall.get("label"),
            "risk": sub_overall.get("risk"),
            "reason": sub_overall.get("main_reason"),
        } if sub_overall else None,
        "signal_highlights": [
            {
                "severity": i.get("severity"),
                "title": i.get("title"),
                "description": i.get("description"),
                "recommendation": i.get("recommendation"),
            }
            for i in issues[:EXT_SIGNAL_HIGHLIGHT_LIMIT]
        ],
        "report_url": f"/report/{scan_id}",
    }


@router.get("/credits")
async def ext_credits(user: dict = Depends(get_api_key_user)):
    async with async_session() as session:
        res = await session.execute(
            select(CreditAccount).where(CreditAccount.user_id == uuid.UUID(user["id"]))
        )
        acct = res.scalar_one_or_none()
    if not acct:
        return {"balance": 0, "reserved": 0}
    return {"balance": acct.balance_tokens, "reserved": acct.reserved_tokens}
