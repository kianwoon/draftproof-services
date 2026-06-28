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


_TIER_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _build_paragraph_issues(results_json: dict, limit: int) -> list[dict]:
    """Reproduce the web report's paragraph-grouped 'Issues' for flagged paragraphs:
    join sentence segments (by paragraph_id) with the per-paragraph explanations
    (main issue / how to improve / reader summary). Mirrors poc/report/render.py."""
    segments = (
        results_json.get("highlight_segments")
        or (((results_json.get("scan_intelligence") or {}).get("document") or {}).get("segments"))
        or []
    )
    expl_by_pid = {
        str(e.get("paragraph_id") or ""): e
        for e in (((results_json.get("paragraph_explanations") or {}).get("paragraphs")) or [])
        if isinstance(e, dict) and e.get("paragraph_id")
    }

    grouped: dict[str, list] = {}
    order: list[str] = []
    for seg in segments:
        pid = str((seg or {}).get("paragraph_id") or "")
        if not pid:
            continue
        if pid not in grouped:
            grouped[pid] = []
            order.append(pid)
        grouped[pid].append(seg)

    issues = []
    for pid in order:
        segs = sorted(grouped[pid], key=lambda x: x.get("start_char") or 0)
        flagged = [x for x in segs if x.get("primary_signal")]
        expl = expl_by_pid.get(pid)
        if not flagged and not expl:
            continue  # only surface paragraphs that carry an issue
        top = None
        for x in flagged:
            ps = x.get("primary_signal") or {}
            if top is None or _TIER_RANK.get(ps.get("tier", ""), -1) > _TIER_RANK.get(top.get("tier", ""), -1):
                top = ps
        top = top or {}
        expl = expl or {}
        text = " ".join((x.get("text") or "") for x in segs).strip()
        issues.append({
            "snippet": text[:220],
            "tier": top.get("tier") or None,
            "signal_label": top.get("label") or None,
            "reader_summary": expl.get("reader_summary") or expl.get("summary") or None,
            "main_issue": expl.get("main_issue") or None,
            "recommendation": expl.get("recommendation") or top.get("recommendation") or None,
        })
        if len(issues) >= limit:
            break
    return issues


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
    # _enrich_badge can RECOMPUTE critical_thinking_control via the deterministic
    # composer (no LLM questions). Preserve any questions the worker already wrote.
    pre_questions = (badge.get("critical_thinking_control") or {}).get("questions")
    try:
        _enrich_badge(badge)  # no-op if already enriched
    except Exception:
        pass

    ctc = dict(badge.get("critical_thinking_control") or {})
    if pre_questions and not ctc.get("questions"):
        ctc["questions"] = pre_questions
    sub_overall = (badge.get("submission_risk") or {}).get("overall") or {}
    issues = report.get("issues") or []
    results_json = report.get("results_json") or {}
    paragraph_issues = _build_paragraph_issues(results_json, EXT_SIGNAL_HIGHLIGHT_LIMIT)
    # The text that was scanned, so the add-in can show it in the "Selected text"
    # box alongside a (re-fetched) result. Capped to bound the payload.
    scanned_text = results_json.get("input_text") or ""
    if len(scanned_text) > 8000:
        scanned_text = scanned_text[:8000] + "…"

    return {
        "scan_id": scan_id,
        "tier": report.get("tier"),
        "ai_score": report.get("ai_score"),
        # Decouple the % from the Turnitin mental model: students read any % as a Turnitin
        # pass/fail (<20% = passed), so a bare number falsely alarms or reassures. It is NOT
        # a Turnitin score, and detectors over-flag fluent writing -- a heads-up, not a verdict.
        "ai_score_note": "Not a Turnitin score — don't compare it to the 20% line. Detectors over-flag fluent writing, so this is a heads-up, not a verdict.",
        "writing_score": report.get("writing_score"),
        "critical_thinking": {
            "score": ctc.get("score"),
            "status": ctc.get("status"),
            "band": ctc.get("band"),
            "questions": [
                {"quote": q.get("anchor_quote"), "question": q.get("question")}
                for q in (ctc.get("questions") or [])
                if isinstance(q, dict) and q.get("question")
            ],
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
        "scanned_text": scanned_text,
        "word_count": report.get("word_count"),
        "paragraph_issues": paragraph_issues,
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
