"""Extension API — key-authenticated scanning for the Google Docs / MS Word add-ins.

Mounted at /api/ext. Reuses scan_service end to end: submit text, poll status.
Scans always bill credits (always_paid=True) and never touch the free quota.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.models.db import async_session, CreditAccount
from app.services.api_key_service import get_api_key_user
from app.services.scan_service import create_scan, get_scan

router = APIRouter()

# Detector signals are unreliable below this; we still scan but flag the result
# so the client can label it low-confidence (honours expose-the-ugly-side).
MIN_CONFIDENT_WORDS = 100
# Match the web text-document cap (documents.py) so a key holder can't submit an
# unbounded body that reserves credits + loads the worker before any size check.
MAX_SCAN_CHARS = 50000


class ExtScanRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_SCAN_CHARS)


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
