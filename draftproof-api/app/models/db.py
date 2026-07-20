from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, DateTime, Integer, Boolean, Text, ForeignKey, CheckConstraint, UniqueConstraint, Numeric, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime, timezone
import uuid

from app.config import DATABASE_CONNECT_TIMEOUT_SECONDS, DATABASE_POOL_RECYCLE_SECONDS, DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"ssl": "require", "timeout": DATABASE_CONNECT_TIMEOUT_SECONDS},
    pool_pre_ping=True,
    pool_recycle=DATABASE_POOL_RECYCLE_SECONDS,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, nullable=False)
    email_normalized = Column(Text, nullable=False, unique=True)
    display_name = Column(Text)
    avatar_url = Column(Text)
    status = Column(Text, nullable=False, default="active")
    # Durable lifetime count of free (<=800 word) scans consumed. Authoritative
    # for the free-scan quota — must NOT be derived from scan_jobs, which the
    # report-retention cleanup purges. See migration 012.
    free_scans_used = Column(Integer, nullable=False, default=0, server_default="0")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class UserIdentity(Base):
    __tablename__ = "user_identities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(Text, nullable=False)
    provider_user_id = Column(Text, nullable=False)
    provider_email = Column(Text)
    provider_email_verified = Column(Boolean)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_login_at = Column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("provider", "provider_user_id"),)


class CreditAccount(Base):
    __tablename__ = "credit_accounts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    balance_tokens = Column(Integer, nullable=False, default=0)
    reserved_tokens = Column(Integer, nullable=False, default=0)
    currency = Column(Text, nullable=False, default="SGD")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (CheckConstraint("balance_tokens >= 0"), CheckConstraint("reserved_tokens >= 0"))


class CreditLedger(Base):
    __tablename__ = "credit_ledger"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    credit_account_id = Column(UUID(as_uuid=True), ForeignKey("credit_accounts.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    entry_type = Column(Text, nullable=False)
    token_delta = Column(Integer, nullable=False)
    balance_after = Column(Integer, nullable=False)
    reference_type = Column(Text)
    reference_id = Column(UUID(as_uuid=True))
    idempotency_key = Column(Text, unique=True)
    note = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class CreditReservation(Base):
    __tablename__ = "credit_reservations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    credit_account_id = Column(UUID(as_uuid=True), ForeignKey("credit_accounts.id", ondelete="CASCADE"), nullable=False)
    job_type = Column(Text, nullable=False)
    job_id = Column(UUID(as_uuid=True), nullable=False)
    tokens_reserved = Column(Integer, nullable=False)
    status = Column(Text, nullable=False, default="active", index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (UniqueConstraint("job_type", "job_id"),)


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    credit_account_id = Column(UUID(as_uuid=True), ForeignKey("credit_accounts.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(Text, nullable=False)
    tokens_charged = Column(Integer, nullable=False, default=0)
    document_id = Column(UUID(as_uuid=True))
    job_id = Column(UUID(as_uuid=True))
    word_count = Column(Integer)
    file_type = Column(Text)
    status = Column(Text, nullable=False, default="completed")
    event_metadata = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Payment(Base):
    __tablename__ = "payments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(Text, nullable=False)
    provider_payment_id = Column(Text)
    provider_customer_id = Column(Text)
    amount_cents = Column(Integer, nullable=False, default=0)
    currency = Column(Text, nullable=False, default="SGD")
    tokens_purchased = Column(Integer, nullable=False, default=0)
    status = Column(Text, nullable=False, index=True)
    idempotency_key = Column(Text, unique=True)
    payment_metadata = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (UniqueConstraint("provider", "provider_payment_id"),)


class PricingPlan(Base):
    __tablename__ = "pricing_plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(Text, nullable=False, unique=True)
    name = Column(Text, nullable=False)
    action_type = Column(Text, nullable=False)
    tokens_required = Column(Integer, nullable=False)
    max_words = Column(Integer, nullable=False, default=1000)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ScanJob(Base):
    __tablename__ = "scan_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    input_text_hash = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False, default=0)
    document_title = Column(Text)
    content_preview = Column(Text)
    scan_type = Column(Text, nullable=False, default="scan")
    # This ASSIGNMENT's AI policy (not institution-wide -- see migration 016 and
    # docs/plans/policy_risk_external_review_response.md Phase 1). One of
    # prohibited/editing_only/allowed_with_declaration/collaboration_allowed/unknown.
    # Nullable/defaulted so absent = today's byte-identical behavior; not yet read or
    # written by any route (Phase 1 batch 1 -- schema/model only, wiring is batch 2).
    ai_policy = Column(Text, nullable=True, default="unknown")
    # CAS token: TRUE while this free scan's increment is live on
    # users.free_scans_used. Refund flips it TRUE->FALSE exactly once so a
    # failed/canceled free scan decrements the counter at most once.
    free_scan_counted = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    status = Column(Text, nullable=False, default="pending", index=True)
    tier = Column(Text)
    ai_score = Column(Numeric(6, 2))
    writing_score = Column(Numeric(6, 2))
    finding_count = Column(Integer)
    progress_percent = Column(Integer, nullable=False, default=0)
    progress_message = Column(Text)
    report_urls = Column(JSONB, default=dict)
    error = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))


class RewriteJob(Base):
    __tablename__ = "rewrite_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scan_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(Text, nullable=False, default="pending", index=True)
    error = Column(Text)
    progress_percent = Column(Integer, nullable=False, default=0)
    progress_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))


class DefenceResponse(Base):
    __tablename__ = "defence_responses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scan_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    question_index = Column(Integer, nullable=False)
    dimension = Column(Text)
    question = Column(Text, nullable=False)
    anchor_quote = Column(Text)
    answer_text = Column(Text, nullable=False)
    attempt = Column(Integer, nullable=False, default=1)
    # pending (created, not yet judged) / judged (judgment populated) / failed
    # (judge_defence_answer returned None or judging errored).
    status = Column(Text, nullable=False, default="pending")
    judgment = Column(JSONB)
    judge_model = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    judged_at = Column(DateTime(timezone=True))


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(Text, nullable=False, default="API key")
    # Display-only prefix (e.g. "dp_live_a1b2c3d4"). The full secret is shown
    # exactly once at creation and is never stored in clear text.
    key_prefix = Column(Text, nullable=False)
    # SHA-256 hex digest of the full key. Keys are high-entropy random tokens, so
    # a fast hash is the correct primitive (bcrypt is for low-entropy passwords).
    # Unique + indexed for O(1) authentication lookup.
    key_hash = Column(Text, nullable=False, unique=True, index=True)
    last_used_at = Column(DateTime(timezone=True))
    # Soft-delete / revocation timestamp; NULL means the key is active.
    revoked_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


async def init_db():
    pass  # Tables managed by migrations — no auto-create needed


async def get_db():
    async with async_session() as session:
        yield session
