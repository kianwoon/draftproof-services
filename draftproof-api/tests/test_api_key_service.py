"""Tests for the API-key service: generation, hashing, CRUD, authentication.

Auth + key material is security-critical, so the crypto/format logic gets real
unit tests and the DB flows use the repo's fake-session pattern (see
test_scan_service.py). No real DB — async_session is monkeypatched.
"""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.services import api_key_service


# ── Fake sessions (mirror test_scan_service.py) ───────────────────────────────

class FakeWriteSession:
    """Captures added objects + commit; execute returns a configurable result."""

    def __init__(self, result=None):
        self.added = []
        self.committed = False
        self._result = result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, item):
        self.added.append(item)

    async def execute(self, *_args, **_kwargs):
        return self._result

    async def commit(self):
        self.committed = True


def _result(*, first=None, scalar=None, count=None, rows=None):
    return SimpleNamespace(
        first=lambda: first,
        scalar=lambda: count,                       # .scalar() — used for COUNT(*)
        scalar_one_or_none=lambda: scalar,
        scalars=lambda: SimpleNamespace(all=lambda: (rows or [])),
    )


# ── Pure crypto / format logic ────────────────────────────────────────────────

def test_generate_key_format():
    raw, key_hash, key_prefix = api_key_service.generate_key()
    assert raw.startswith("dp_live_")
    assert len(raw) > 24  # prefix + ~43 url-safe chars of entropy
    assert key_hash == hashlib.sha256(raw.encode()).hexdigest()
    assert key_prefix == raw[:16]
    assert key_prefix in raw


def test_generate_key_is_unique():
    a, _, _ = api_key_service.generate_key()
    b, _, _ = api_key_service.generate_key()
    assert a != b


def test_hash_key_is_deterministic():
    raw = "dp_live_abc123"
    assert api_key_service._hash_key(raw) == api_key_service._hash_key(raw)
    assert api_key_service._hash_key(raw) == hashlib.sha256(raw.encode()).hexdigest()


# ── create_key ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_key_returns_plaintext_once_and_persists_only_hash(monkeypatch):
    session = FakeWriteSession(result=_result(count=0))  # under the active-key cap
    monkeypatch.setattr(api_key_service, "async_session", lambda: session)

    user_id = str(uuid.uuid4())
    out = await api_key_service.create_key(user_id, name="My Word add-in")

    # Plaintext returned to the caller exactly here.
    assert out["key"].startswith("dp_live_")
    assert out["name"] == "My Word add-in"
    assert out["key_prefix"] == out["key"][:16]
    assert "id" in out and "created_at" in out

    # Persisted row stores the HASH, never the clear-text key.
    assert len(session.added) == 1
    row = session.added[0]
    assert row.key_hash == hashlib.sha256(out["key"].encode()).hexdigest()
    assert getattr(row, "key", None) is None
    assert session.committed is True


@pytest.mark.asyncio
async def test_create_key_rejects_when_active_cap_reached(monkeypatch):
    at_cap = api_key_service.MAX_ACTIVE_KEYS_PER_USER
    session = FakeWriteSession(result=_result(count=at_cap))
    monkeypatch.setattr(api_key_service, "async_session", lambda: session)

    with pytest.raises(ValueError, match="limit reached"):
        await api_key_service.create_key(str(uuid.uuid4()), name="one too many")

    assert session.added == []        # no key minted
    assert session.committed is False


# ── authenticate ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_authenticate_valid_key_returns_user(monkeypatch):
    user_id = uuid.uuid4()
    api_key_row = SimpleNamespace(last_used_at=None)
    user_row = SimpleNamespace(id=user_id, email="owner@example.com")
    session = FakeWriteSession(result=_result(first=(api_key_row, user_row)))
    monkeypatch.setattr(api_key_service, "async_session", lambda: session)

    out = await api_key_service.authenticate("dp_live_validkey")
    assert out == {"id": str(user_id), "email": "owner@example.com"}


@pytest.mark.asyncio
async def test_authenticate_unknown_key_returns_none(monkeypatch):
    session = FakeWriteSession(result=_result(first=None))
    monkeypatch.setattr(api_key_service, "async_session", lambda: session)
    assert await api_key_service.authenticate("dp_live_nope") is None


@pytest.mark.asyncio
async def test_authenticate_rejects_bad_prefix_without_db(monkeypatch):
    def _boom():
        raise AssertionError("DB must not be touched for a malformed key")
    monkeypatch.setattr(api_key_service, "async_session", _boom)
    assert await api_key_service.authenticate("not-a-dp-key") is None
    assert await api_key_service.authenticate("") is None
    assert await api_key_service.authenticate(None) is None


# ── list_keys / revoke_key ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_keys_never_exposes_hash(monkeypatch):
    rows = [
        SimpleNamespace(
            id=uuid.uuid4(), name="k1", key_prefix="dp_live_aaaa1111",
            key_hash="SECRET", last_used_at=None, revoked_at=None,
            created_at=datetime.now(timezone.utc),
        )
    ]
    session = FakeWriteSession(result=_result(rows=rows))
    monkeypatch.setattr(api_key_service, "async_session", lambda: session)

    out = await api_key_service.list_keys(str(uuid.uuid4()))
    assert len(out) == 1
    assert out[0]["key_prefix"] == "dp_live_aaaa1111"
    assert "key_hash" not in out[0]
    assert "key" not in out[0]


@pytest.mark.asyncio
async def test_revoke_key_sets_revoked_at(monkeypatch):
    api_key_row = SimpleNamespace(revoked_at=None)
    session = FakeWriteSession(result=_result(scalar=api_key_row))
    monkeypatch.setattr(api_key_service, "async_session", lambda: session)

    ok = await api_key_service.revoke_key(str(uuid.uuid4()), str(uuid.uuid4()))
    assert ok is True
    assert api_key_row.revoked_at is not None
    assert session.committed is True


@pytest.mark.asyncio
async def test_revoke_missing_key_returns_false(monkeypatch):
    session = FakeWriteSession(result=_result(scalar=None))
    monkeypatch.setattr(api_key_service, "async_session", lambda: session)
    assert await api_key_service.revoke_key(str(uuid.uuid4()), str(uuid.uuid4())) is False


# ── get_api_key_user dependency ───────────────────────────────────────────────

def _request(headers: dict):
    return SimpleNamespace(headers=headers)


@pytest.mark.asyncio
async def test_dependency_accepts_bearer(monkeypatch):
    async def _auth(presented):
        assert presented == "dp_live_x"
        return {"id": "u1", "email": "e@x.com"}
    monkeypatch.setattr(api_key_service, "authenticate", _auth)
    user = await api_key_service.get_api_key_user(_request({"Authorization": "Bearer dp_live_x"}))
    assert user["id"] == "u1"


@pytest.mark.asyncio
async def test_dependency_accepts_x_api_key(monkeypatch):
    async def _auth(presented):
        assert presented == "dp_live_y"
        return {"id": "u2", "email": "e@x.com"}
    monkeypatch.setattr(api_key_service, "authenticate", _auth)
    user = await api_key_service.get_api_key_user(_request({"X-API-Key": "dp_live_y"}))
    assert user["id"] == "u2"


@pytest.mark.asyncio
async def test_dependency_missing_key_401():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await api_key_service.get_api_key_user(_request({}))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_dependency_invalid_key_401(monkeypatch):
    from fastapi import HTTPException
    async def _auth(_presented):
        return None
    monkeypatch.setattr(api_key_service, "authenticate", _auth)
    with pytest.raises(HTTPException) as exc:
        await api_key_service.get_api_key_user(_request({"Authorization": "Bearer dp_live_bad"}))
    assert exc.value.status_code == 401
