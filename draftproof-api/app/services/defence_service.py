"""Defence-readiness service — DB CRUD for `defence_responses` (Task 6).

Free-capped v1 (product decision, see migrations/014_defence_responses.sql): NO
credit_ledger / credit_reservations integration. The route layer (app/routes/defence.py)
enforces the attempt/length caps using count_attempts() below before calling create_response().

Judging itself (populating `judgment`/`status='judged'`) is Task 7's Celery task — this
service only ever writes `status='pending'` rows and reads whatever the worker later wrote.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models.db import async_session, DefenceResponse

# Best-of-attempts level ranking, worst first — used by aggregate_readiness() to pick the
# single best judged attempt per dimension.
_LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}


class AttemptCapRace(Exception):
    """Raised when create_response()'s insert loses a race against the DB-level unique
    constraint on (scan_id, question_index, attempt) added in migrations/015 — i.e. a
    concurrent request for the same question committed the same attempt number first.
    count_attempts() + this check-then-insert is not atomic, so this is the backstop that
    makes the attempt cap a hard guarantee rather than a best-effort one. Callers (see
    app/routes/defence.py) should treat this identically to the pre-check 409 path."""


async def count_attempts(scan_id: str, question_index: int) -> int:
    """Number of existing defence_responses rows for this (scan_id, question_index).

    Counts ALL attempts (pending/judged/failed) — a retry while a prior attempt is still
    `pending` (judging in flight) still consumes an attempt slot, so a user cannot bypass
    the cap by firing several answers before the first is judged.
    """
    async with async_session() as session:
        result = await session.execute(
            select(DefenceResponse).where(
                DefenceResponse.scan_id == uuid.UUID(scan_id),
                DefenceResponse.question_index == question_index,
            )
        )
        return len(result.scalars().all())


async def create_response(
    *,
    scan_id: str,
    user_id: str,
    question_index: int,
    dimension: str | None,
    question: str,
    anchor_quote: str | None,
    answer_text: str,
    attempt: int,
) -> dict:
    """Insert a pending defence_responses row. Returns {id, status}.

    Raises AttemptCapRace if the DB-level unique constraint on
    (scan_id, question_index, attempt) rejects this insert — a concurrent request already
    committed this exact attempt number first (see migrations/015 and the class docstring).
    """
    row = DefenceResponse(
        id=uuid.uuid4(),
        scan_id=uuid.UUID(scan_id),
        user_id=uuid.UUID(user_id),
        question_index=question_index,
        dimension=dimension,
        question=question,
        anchor_quote=anchor_quote,
        answer_text=answer_text,
        attempt=attempt,
        status="pending",
    )
    async with async_session() as session:
        session.add(row)
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise AttemptCapRace(
                f"attempt {attempt} for (scan_id={scan_id}, question_index={question_index}) "
                "already exists"
            ) from exc
        await session.refresh(row)
    return {"id": str(row.id), "status": row.status}


async def list_responses(scan_id: str) -> list[dict]:
    """All defence_responses rows for a scan, ordered by question then attempt."""
    async with async_session() as session:
        result = await session.execute(
            select(DefenceResponse)
            .where(DefenceResponse.scan_id == uuid.UUID(scan_id))
            .order_by(DefenceResponse.question_index, DefenceResponse.attempt)
        )
        rows = result.scalars().all()
        return [
            {
                "id": str(r.id),
                "question_index": r.question_index,
                "dimension": r.dimension,
                "question": r.question,
                "anchor_quote": r.anchor_quote,
                "answer_text": r.answer_text,
                "attempt": r.attempt,
                "status": r.status,
                "judgment": r.judgment,
                "judge_model": r.judge_model,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "judged_at": r.judged_at.isoformat() if r.judged_at else None,
            }
            for r in rows
        ]


def aggregate_readiness(rows: list[dict]) -> dict[str, dict]:
    """Per-dimension readiness: the BEST judged `overall` {level, score} across all attempts
    for that dimension. Rows that are not yet judged, or whose judgment is missing/malformed,
    do not count toward readiness (a pending/failed attempt proves nothing)."""
    best: dict[str, dict] = {}
    for row in rows:
        if row.get("status") != "judged":
            continue
        judgment = row.get("judgment")
        if not isinstance(judgment, dict):
            continue
        overall = judgment.get("overall")
        if not isinstance(overall, dict):
            continue
        level = overall.get("level")
        score = overall.get("score")
        if level not in _LEVEL_RANK or not isinstance(score, (int, float)):
            continue
        dimension = row.get("dimension") or "unspecified"
        current = best.get(dimension)
        is_better = current is None or (
            _LEVEL_RANK[level] > _LEVEL_RANK[current["level"]]
            or (_LEVEL_RANK[level] == _LEVEL_RANK[current["level"]] and score > current["score"])
        )
        if is_better:
            best[dimension] = {
                "level": level,
                "score": score,
                "question_index": row.get("question_index"),
                "attempt": row.get("attempt"),
            }
    return best
