"""Tests for OAuth DB-connect retry.

Root cause (Koyeb/Neon): the managed Postgres endpoint autosuspends, so the first
connection after idle can exceed the 10s connect timeout and raise a bare
TimeoutError. The OAuth callback is single-shot, so one unlucky connect failed the
whole sign-in. These tests pin the retry behavior that absorbs that transient.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import OperationalError

import app.routes.auth as auth


@pytest.fixture
def fake_db():
    db = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_retries_on_connect_timeout_then_succeeds(fake_db):
    sentinel_user = object()
    inner = AsyncMock(side_effect=[TimeoutError(), sentinel_user])

    with patch.object(auth, "_upsert_user", inner), \
         patch.object(auth.asyncio, "sleep", AsyncMock()):
        result = await auth._upsert_user_with_retry(
            fake_db, "microsoft", {"email": "a@b.com"}, attempts=3, base_delay=0
        )

    assert result is sentinel_user
    assert inner.await_count == 2          # failed once, retried, succeeded
    fake_db.rollback.assert_awaited()      # cleared partial state before retry


@pytest.mark.asyncio
async def test_gives_up_and_reraises_after_max_attempts(fake_db):
    inner = AsyncMock(side_effect=TimeoutError())

    with patch.object(auth, "_upsert_user", inner), \
         patch.object(auth.asyncio, "sleep", AsyncMock()):
        with pytest.raises(TimeoutError):
            await auth._upsert_user_with_retry(
                fake_db, "microsoft", {"email": "a@b.com"}, attempts=3, base_delay=0
            )

    assert inner.await_count == 3


@pytest.mark.asyncio
async def test_retries_on_sqlalchemy_operational_error(fake_db):
    sentinel_user = object()
    op_err = OperationalError("connect", None, Exception("timeout"))
    inner = AsyncMock(side_effect=[op_err, sentinel_user])

    with patch.object(auth, "_upsert_user", inner), \
         patch.object(auth.asyncio, "sleep", AsyncMock()):
        result = await auth._upsert_user_with_retry(
            fake_db, "google", {"email": "a@b.com"}, attempts=3, base_delay=0
        )

    assert result is sentinel_user
    assert inner.await_count == 2


@pytest.mark.asyncio
async def test_does_not_retry_application_errors(fake_db):
    """A non-connection error (e.g. bad data) must fail fast, not burn retries."""
    inner = AsyncMock(side_effect=ValueError("bad email"))

    with patch.object(auth, "_upsert_user", inner), \
         patch.object(auth.asyncio, "sleep", AsyncMock()):
        with pytest.raises(ValueError):
            await auth._upsert_user_with_retry(
                fake_db, "google", {"email": "a@b.com"}, attempts=3, base_delay=0
            )

    assert inner.await_count == 1
