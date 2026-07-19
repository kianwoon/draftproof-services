"""Unit tests for defence_service.create_response's DB-level attempt-cap backstop
(migrations/015_defence_responses_unique_attempt.sql).

count_attempts() + create_response() in routes/defence.py is a non-atomic check-then-insert:
two near-simultaneous requests for the same (scan_id, question_index) can both read the same
prior_attempts and both pass the application-level cap check. The unique index on
(scan_id, question_index, attempt) makes the loser's INSERT fail at the DB level instead of
silently exceeding the attempt cap; create_response() must translate that IntegrityError into
AttemptCapRace so the route can return the same 409 the pre-check path already returns.

No real DB — async_session is monkeypatched to a fake async context manager whose commit()
raises sqlalchemy.exc.IntegrityError on demand.
"""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.services import defence_service


class _FakeSession:
    def __init__(self, raise_on_commit=None):
        self._raise_on_commit = raise_on_commit
        self.added = []
        self.committed = False
        self.rolled_back = False

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        if self._raise_on_commit is not None:
            raise self._raise_on_commit
        self.committed = True

    async def rollback(self):
        self.rolled_back = True

    async def refresh(self, obj):
        pass


class _FakeSessionCtx:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _unique_violation():
    # orig can be any exception for test purposes — create_response only inspects the
    # IntegrityError wrapper type, matching what SQLAlchemy raises for a real unique-index
    # violation regardless of the underlying driver's specific exception class.
    return IntegrityError("INSERT INTO defence_responses ...", {}, Exception("unique violation"))


@pytest.mark.asyncio
async def test_create_response_raises_attempt_cap_race_on_unique_violation(monkeypatch):
    fake_session = _FakeSession(raise_on_commit=_unique_violation())
    monkeypatch.setattr(
        defence_service, "async_session", lambda: _FakeSessionCtx(fake_session)
    )

    with pytest.raises(defence_service.AttemptCapRace):
        await defence_service.create_response(
            scan_id="11111111-1111-1111-1111-111111111111",
            user_id="22222222-2222-2222-2222-222222222222",
            question_index=0,
            dimension="d",
            question="Q?",
            anchor_quote="q",
            answer_text="a",
            attempt=1,
        )
    # The failed commit must be rolled back, not left dangling.
    assert fake_session.rolled_back is True
    assert fake_session.committed is False


@pytest.mark.asyncio
async def test_create_response_succeeds_when_no_race(monkeypatch):
    fake_session = _FakeSession(raise_on_commit=None)
    monkeypatch.setattr(
        defence_service, "async_session", lambda: _FakeSessionCtx(fake_session)
    )

    result = await defence_service.create_response(
        scan_id="11111111-1111-1111-1111-111111111111",
        user_id="22222222-2222-2222-2222-222222222222",
        question_index=0,
        dimension="d",
        question="Q?",
        anchor_quote="q",
        answer_text="a",
        attempt=1,
    )
    assert result["status"] == "pending"
    assert fake_session.committed is True
    assert fake_session.rolled_back is False
