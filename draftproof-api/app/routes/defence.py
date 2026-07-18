"""Defence-readiness API — reflective-question answers + LLM-judged readiness (Task 6).

Mounted at /api/scans (alongside routes/scans.py) so the paths read as
`/api/scans/{scan_id}/defence...`, matching the plan brief exactly.

Kill switch `DRAFTPROOF_DEFENCE_CHECK` (app.config), default OFF: both routes return 404
("feature disabled") AT REQUEST TIME via a route-level dependency, so this is gated even
before the `get_current_user` auth dependency runs — an unauthenticated probe gets the same
404 as everyone else while the feature is off, not a 401 that would reveal the route exists.

Free-capped v1 (product decision, see migrations/014_defence_responses.sql): NO
credit_ledger / credit_reservations integration. Only simple env-configured attempt/length
caps at this layer (DRAFTPROOF_DEFENCE_MAX_ANSWER_CHARS, DRAFTPROOF_DEFENCE_MAX_ATTEMPTS).

Judging (Task 7, not implemented yet) is enqueued by TASK NAME STRING ONLY —
"app.tasks.judge_defence_answer" — never by importing the worker task module directly: the
root Dockerfile never copies poc/ or worker/ into the API image. Task 7's Celery task in
worker/app/tasks.py MUST be named exactly `judge_defence_answer` for this to resolve, and can
load everything it needs (question/anchor_quote/dimension/answer_text/scan_id) from the
defence_responses row via the single `response_id` argument passed here.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import (
    DRAFTPROOF_DEFENCE_CHECK,
    DRAFTPROOF_DEFENCE_MAX_ANSWER_CHARS,
    DRAFTPROOF_DEFENCE_MAX_ATTEMPTS,
)
from app.routes.auth import get_current_user
from app.services import defence_service
from app.services.report_service import _fetch_optional_report_json_sync
from app.services.scan_service import get_scan

router = APIRouter()


class DefenceAnswerIn(BaseModel):
    question_index: int = Field(..., ge=0)
    answer_text: str = Field(..., min_length=1, max_length=DRAFTPROOF_DEFENCE_MAX_ANSWER_CHARS)


def _require_enabled() -> None:
    """Route-level dependency (see module docstring for why this must run before auth)."""
    if not DRAFTPROOF_DEFENCE_CHECK:
        raise HTTPException(status_code=404, detail="feature disabled")


async def _load_questions(scan_id: str) -> list[dict]:
    """Read the scan's report JSON from R2 (same read path as report_service.get_report /
    scan_service._scan_report_in_r2) and return its critical_thinking_control.questions
    array. Empty list if the report / badge / questions block is missing or the scan
    hasn't completed yet — callers treat that as "no valid question_index"."""
    report = await asyncio.to_thread(
        _fetch_optional_report_json_sync, f"reports/{scan_id}/report.json"
    )
    if not isinstance(report, dict):
        return []
    badge = report.get("ai_risk_badge")
    if not isinstance(badge, dict):
        return []
    ctc = badge.get("critical_thinking_control")
    if not isinstance(ctc, dict):
        return []
    questions = ctc.get("questions")
    return questions if isinstance(questions, list) else []


@router.post(
    "/{scan_id}/defence/answers",
    status_code=202,
    dependencies=[Depends(_require_enabled)],
)
async def submit_defence_answer(
    scan_id: str,
    body: DefenceAnswerIn,
    user: dict = Depends(get_current_user),
):
    # Ownership: reuse scan_service.get_scan(scan_id, user_id=...) exactly as
    # routes/scans.py::get_scan_route does — None covers both "no such scan" and "not
    # yours" (this codebase's ownership pattern never distinguishes the two, to avoid
    # leaking existence of other users' scans), so 404 either way.
    scan = await get_scan(scan_id, user_id=user["id"])
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    questions = await _load_questions(scan_id)
    if body.question_index >= len(questions) or not isinstance(questions[body.question_index], dict):
        raise HTTPException(status_code=400, detail="Invalid question_index")
    q = questions[body.question_index]

    prior_attempts = await defence_service.count_attempts(scan_id, body.question_index)
    if prior_attempts >= DRAFTPROOF_DEFENCE_MAX_ATTEMPTS:
        # 409 (not 429): this is a permanent per-question count cap, not a time-windowed
        # rate limit — 429 implies "retry later" which doesn't apply here. Matches the
        # analogous permanent count-cap precedent in routes/keys.py::create_key_route.
        raise HTTPException(status_code=409, detail="Attempt limit reached for this question")

    created = await defence_service.create_response(
        scan_id=scan_id,
        user_id=user["id"],
        question_index=body.question_index,
        dimension=q.get("dimension"),
        question=str(q.get("question") or ""),
        anchor_quote=q.get("anchor_quote"),
        answer_text=body.answer_text,
        attempt=prior_attempts + 1,
    )

    from app.services.celery_client import judge_defence_answer
    judge_defence_answer.delay(created["id"])

    return {"response_id": created["id"], "status": created["status"]}


@router.get("/{scan_id}/defence", dependencies=[Depends(_require_enabled)])
async def get_defence_responses(scan_id: str, user: dict = Depends(get_current_user)):
    scan = await get_scan(scan_id, user_id=user["id"])
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    rows = await defence_service.list_responses(scan_id)
    readiness = defence_service.aggregate_readiness(rows)
    return {"responses": rows, "readiness": readiness}
