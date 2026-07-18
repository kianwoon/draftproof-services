import os
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Default raised 10 → 20: the Koyeb/Neon endpoint autosuspends, so the first
# connect after idle must wake compute + negotiate SSL, which can exceed 10s and
# raise a bare TimeoutError. 20s absorbs the cold-start; retry logic is the net.
DATABASE_CONNECT_TIMEOUT_SECONDS = max(1, int(os.getenv("DATABASE_CONNECT_TIMEOUT_SECONDS", "20")))

# pool_recycle 300 dropped connections every 5 min — right as Neon suspends — so
# requests routinely paid the cold-connect cost. 1800s + pool_pre_ping (which
# transparently reconnects stale connections) cuts fresh connects without risk.
DATABASE_POOL_RECYCLE_SECONDS = max(60, int(os.getenv("DATABASE_POOL_RECYCLE_SECONDS", "1800")))


def _normalize_database_url(raw_db_url: str) -> str:
    raw_db_url = (raw_db_url or "").strip()
    if not raw_db_url:
        raise RuntimeError("DATABASE_URL environment variable is required")

    if "://" in raw_db_url:
        # Normalize any postgres:// or postgresql:// to postgresql+asyncpg:// for SQLAlchemy async.
        after_scheme = raw_db_url.split("://", 1)[1]
        database_url = f"postgresql+asyncpg://{after_scheme}"
    else:
        # Bare hostname — build from Koyeb env vars.
        db_user = os.getenv("DATABASE_USER", "koyeb-adm")
        db_pass = os.getenv("DATABASE_PASSWORD", "")
        db_name = os.getenv("DATABASE_NAME", "koyebdb")
        db_port = os.getenv("DATABASE_PORT", "5432")
        database_url = f"postgresql+asyncpg://{db_user}:{db_pass}@{raw_db_url}:{db_port}/{db_name}"

    parsed = urlsplit(database_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    # asyncpg accepts sslmode in its own DSN parser, but SQLAlchemy's asyncpg
    # dialect forwards URL query keys as keyword args in this app's pinned
    # runtime. Keep SSL enforcement in create_async_engine(connect_args).
    query.pop("sslmode", None)
    return urlunsplit(parsed._replace(query=urlencode(query)))


DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL", ""))

# Auth
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is required")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

ALLOWED_EMAIL_DOMAINS = {"gmail.com", "googlemail.com", "hotmail.com", "outlook.com", "live.com"}

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# Microsoft OAuth
MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
MICROSOFT_TENANT = os.getenv("MICROSOFT_TENANT", "common")

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Token pricing (SGD)
TOKEN_CURRENCY_CODE = os.getenv("TOKEN_CURRENCY_CODE", "SGD").strip().upper() or "SGD"
TOKEN_STRIPE_CURRENCY = TOKEN_CURRENCY_CODE.lower()
TOKEN_PRICE_SGD = float(os.getenv("TOKEN_PRICE_SGD", "0.90"))
TOKEN_PRO_PACK_PRICE_SGD = float(os.getenv("TOKEN_PRO_PACK_PRICE_SGD", "20.00"))
TOKEN_PACKS = {
    "single": {"tokens": 1, "name": "Single Token", "price_sgd": round(1 * TOKEN_PRICE_SGD, 2)},
    "starter": {"tokens": 5, "name": "Starter Pack", "price_sgd": round(5 * TOKEN_PRICE_SGD, 2)},
    "standard": {"tokens": 10, "name": "Standard Pack", "price_sgd": round(10 * TOKEN_PRICE_SGD, 2)},
    "pro": {"tokens": 25, "name": "Pro Pack", "price_sgd": round(TOKEN_PRO_PACK_PRICE_SGD, 2)},
}


def get_token_pack_price_sgd(pack: dict) -> float:
    return round(float(pack["price_sgd"]), 2)

# One-time welcome grant: new accounts are created with this many free credits at
# signup (replaces the old per-scan free allowance). Env-tunable so the bonus can
# be changed without a code deploy. Set to 0 to disable the grant.
WELCOME_CREDITS = max(0, int(os.getenv("WELCOME_CREDITS", "5")))

# Frontend URL for redirects
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
_frontend_origin = urlparse(FRONTEND_URL.split(",")[0].strip())
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "").lower() in {"1", "true", "yes"} if os.getenv("COOKIE_SECURE") else _frontend_origin.scheme == "https"

# Redis / Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_VISIBILITY_TIMEOUT_SECONDS = max(
    3000,
    int(os.getenv("CELERY_VISIBILITY_TIMEOUT_SECONDS", "3000")),
)
REDIS_SOCKET_TIMEOUT_SECONDS = max(1, int(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "30")))
REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = max(1, int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS", "10")))
REDIS_SOCKET_KEEPALIVE = os.getenv("REDIS_SOCKET_KEEPALIVE", "true").strip().lower() in {"1", "true", "yes", "on"}
REDIS_HEALTH_CHECK_INTERVAL_SECONDS = max(1, int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL_SECONDS", "120")))
# SSE progress-stream read window (ms). XREAD BLOCK returns immediately when a
# new event is published, so widening this only removes idle round-trips (Upstash
# bills per command) -- it does NOT delay event delivery. Default 5000ms matches
# the 5s Postgres fallback cadence in the SSE routes, keeping that safety net
# tight. Clamped below the socket timeout so the blocking read always returns first.
SSE_XREAD_BLOCK_MS = min(
    REDIS_SOCKET_TIMEOUT_SECONDS * 1000 - 1000,
    max(1000, int(os.getenv("SSE_XREAD_BLOCK_MS", "5000"))),
)

# R2 Storage (for fetching report JSON)
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "draftproof-reports")

# Rewrite
REWRITE_STALE_THRESHOLD_MINUTES = max(
    50,
    int(os.getenv("REWRITE_STALE_THRESHOLD_MINUTES", "50")),
)
# Cancellation is DB-authoritative and cooperative. Hard termination can cause
# late-ack task redelivery, so keep it opt-in for emergency operations only.
REWRITE_CANCEL_TERMINATE = os.getenv("REWRITE_CANCEL_TERMINATE", "false").strip().lower() in {"1", "true", "yes", "on"}
REWRITE_CANCEL_TERMINATE_SIGNAL = os.getenv("REWRITE_CANCEL_TERMINATE_SIGNAL", "SIGTERM").strip() or "SIGTERM"

# Defence-readiness check (Task 6). Free-capped v1 per product decision: NO credit_ledger /
# credit_reservations integration here, just simple env-configured caps at this API layer (see
# migrations/014_defence_responses.sql). Kill switch, default OFF until Task 7 (judging) and
# Task 8 (frontend) land — both /api/scans/{id}/defence routes 404 at request time while off.
DRAFTPROOF_DEFENCE_CHECK = os.getenv("DRAFTPROOF_DEFENCE_CHECK", "false").strip().lower() in {"1", "true", "yes", "on"}
# Max characters accepted in a single defence answer.
DRAFTPROOF_DEFENCE_MAX_ANSWER_CHARS = max(1, int(os.getenv("DRAFTPROOF_DEFENCE_MAX_ANSWER_CHARS", "2000")))
# Max attempts allowed per (scan_id, question_index) — a simple free-tier guard, not real
# credit metering.
DRAFTPROOF_DEFENCE_MAX_ATTEMPTS = max(1, int(os.getenv("DRAFTPROOF_DEFENCE_MAX_ATTEMPTS", "2")))

# In-app feedback → GitHub Issues. The repo is private, so visitors can't file
# issues directly; the API files them server-side with a fine-grained PAT that
# has only Issues:write on FEEDBACK_GITHUB_REPO. Bot abuse is fenced off with a
# Cloudflare Turnstile challenge verified server-side before any issue is cut.
# All three must be set for the /api/feedback endpoint to accept submissions.
FEEDBACK_GITHUB_TOKEN = os.getenv("FEEDBACK_GITHUB_TOKEN", "").strip()
FEEDBACK_GITHUB_REPO = os.getenv("FEEDBACK_GITHUB_REPO", "kianwoon/draftproof-services").strip()
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "").strip()
# Public Turnstile SITE key — served to the browser at runtime via
# GET /api/feedback/config so it stays a plain Koyeb env var (no Vite build-arg).
TURNSTILE_SITE_KEY = os.getenv("TURNSTILE_SITE_KEY", "").strip()
