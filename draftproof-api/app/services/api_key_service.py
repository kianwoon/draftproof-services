"""API-key service — mint, list, revoke, and authenticate personal API keys.

Keys authenticate the Google Docs / MS Word add-ins. They resolve to the owning
user (same {"id", "email"} shape as the cookie/JWT `get_current_user`) and bill
the user's existing credit balance — there is no separate currency or cost path.

Security model:
  * Keys are high-entropy random tokens (`dp_live_` + 32 url-safe bytes), so a
    fast hash (SHA-256) is the correct primitive — bcrypt is for low-entropy
    passwords. Only the hash is stored; the clear-text key is returned once.
  * Auth looks the key up by hash, filtered to non-revoked rows.
  * `last_used_at` is updated lazily (throttled) to avoid a write per request.
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from sqlalchemy import func, select

from app.models.db import async_session, ApiKey, User


logger = logging.getLogger("draftproof.api_keys")

KEY_PREFIX = "dp_live_"
KEY_PREFIX_DISPLAY_LEN = 16  # "dp_live_" + first 8 entropy chars, safe to show
LAST_USED_THROTTLE = timedelta(seconds=60)
# Cap active keys per user so a leaked session can't mint unboundedly (table
# growth + auth-lookup overhead). Generous enough for several devices/add-ins.
MAX_ACTIVE_KEYS_PER_USER = 20


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_key() -> tuple[str, str, str]:
    """Return (raw_key, key_hash, key_prefix). `raw_key` is shown to the user once."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, _hash_key(raw), raw[:KEY_PREFIX_DISPLAY_LEN]


async def create_key(user_id: str, name: str | None = None) -> dict:
    """Mint a key for a user. Returns the clear-text `key` exactly once."""
    raw, key_hash, key_prefix = generate_key()
    label = (name or "").strip() or "API key"
    record_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)
    uid = uuid.UUID(user_id)

    async with async_session() as session:
        active_count = await session.execute(
            select(func.count())
            .select_from(ApiKey)
            .where(ApiKey.user_id == uid, ApiKey.revoked_at.is_(None))
        )
        if (active_count.scalar() or 0) >= MAX_ACTIVE_KEYS_PER_USER:
            raise ValueError(
                f"API key limit reached (max {MAX_ACTIVE_KEYS_PER_USER}). "
                "Revoke an unused key first."
            )
        session.add(ApiKey(
            id=record_id,
            user_id=uid,
            name=label,
            key_prefix=key_prefix,
            key_hash=key_hash,
            created_at=created_at,
        ))
        await session.commit()

    return {
        "id": str(record_id),
        "name": label,
        "key": raw,  # shown once — never retrievable again
        "key_prefix": key_prefix,
        "created_at": created_at.isoformat(),
    }


async def list_keys(user_id: str) -> list[dict]:
    """List a user's keys, newest first. Never returns the hash or clear-text key."""
    async with async_session() as session:
        result = await session.execute(
            select(ApiKey)
            .where(ApiKey.user_id == uuid.UUID(user_id))
            .order_by(ApiKey.created_at.desc())
        )
        rows = result.scalars().all()

    return [
        {
            "id": str(k.id),
            "name": k.name,
            "key_prefix": k.key_prefix,
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            "revoked_at": k.revoked_at.isoformat() if k.revoked_at else None,
            "created_at": k.created_at.isoformat() if k.created_at else None,
        }
        for k in rows
    ]


async def revoke_key(user_id: str, key_id: str) -> bool:
    """Soft-revoke a key the user owns. Returns False if missing or already revoked."""
    async with async_session() as session:
        result = await session.execute(
            select(ApiKey).where(
                ApiKey.id == uuid.UUID(key_id),
                ApiKey.user_id == uuid.UUID(user_id),
            )
        )
        api_key = result.scalar_one_or_none()
        if api_key is None or api_key.revoked_at is not None:
            return False
        api_key.revoked_at = datetime.now(timezone.utc)
        await session.commit()
    return True


async def authenticate(presented: str | None) -> dict | None:
    """Resolve a presented key to {"id", "email"}, or None if invalid/revoked."""
    if not presented or not presented.startswith(KEY_PREFIX):
        return None  # malformed — never hit the DB

    key_hash = _hash_key(presented)
    async with async_session() as session:
        result = await session.execute(
            select(ApiKey, User)
            .join(User, ApiKey.user_id == User.id)
            .where(ApiKey.key_hash == key_hash, ApiKey.revoked_at.is_(None))
        )
        row = result.first()
        if row is None:
            return None
        api_key, user = row

        now = datetime.now(timezone.utc)
        if api_key.last_used_at is None or (now - api_key.last_used_at) > LAST_USED_THROTTLE:
            api_key.last_used_at = now
            await session.commit()

        return {"id": str(user.id), "email": user.email}


async def get_api_key_user(request: Request) -> dict:
    """FastAPI dependency for key-authenticated routes.

    Accepts `Authorization: Bearer dp_live_…` or `X-API-Key: dp_live_…`.
    Kept separate from the web cookie/JWT dependency so the two never cross.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        presented = auth_header[7:].strip()
    else:
        presented = request.headers.get("X-API-Key")

    if not presented:
        raise HTTPException(status_code=401, detail="API key required")

    user = await authenticate(presented)
    if not user:
        # Log the safe prefix only (never the full key) so brute-force / replay
        # attempts on /api/ext/* are observable.
        logger.warning("API key auth failed (prefix=%s)", str(presented)[:KEY_PREFIX_DISPLAY_LEN])
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    return user
