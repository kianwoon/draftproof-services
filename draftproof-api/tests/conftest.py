import asyncio
import uuid
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from jose import jwt

from app.config import SECRET_KEY, JWT_ALGORITHM
from app.models.db import Base, User, UserIdentity, CreditAccount


# ── Test DB ──

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def engine():
    eng = create_async_engine(TEST_DB_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest.fixture
async def db_session(engine):
    """Fresh DB session with per-test rollback isolation."""
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def db_with_user(db_session):
    """Insert a test user + credit account and return (session, user_id)."""
    user_id = str(uuid.uuid4())
    user = User(id=user_id, email="test@example.com", display_name="Test User", status="active")
    db_session.add(user)

    identity = UserIdentity(
        user_id=user_id, provider="google", provider_user_id="google_123",
        email="test@example.com", email_verified=True,
    )
    db_session.add(identity)

    account = CreditAccount(user_id=user_id, balance_tokens=10, reserved_tokens=0, status="active")
    db_session.add(account)

    await db_session.commit()
    return db_session, user_id


# ── Auth helpers ──

def make_jwt(user_id: str, email: str = "test@example.com") -> str:
    from datetime import datetime, timedelta
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=1),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


@pytest.fixture
def auth_cookie(db_with_user):
    """Return {'token': jwt} for the test user."""
    _, user_id = db_with_user
    return {"token": make_jwt(user_id)}


# ── HTTP client ──

@pytest.fixture
async def client(db_session):
    """ASGI test client with DB dependency overridden."""
    from app.main import app
    from app.models.db import get_db

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
