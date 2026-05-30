import os
from urllib.parse import urlparse

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

_raw_db_url = os.getenv("DATABASE_URL", "").strip()
if not _raw_db_url:
    raise RuntimeError("DATABASE_URL environment variable is required")

# Normalize any postgres:// or postgresql:// to postgresql+asyncpg:// for SQLAlchemy async
if "://" in _raw_db_url:
    # Strip any existing scheme, then prepend the correct async one
    _after_scheme = _raw_db_url.split("://", 1)[1]
    DATABASE_URL = f"postgresql+asyncpg://{_after_scheme}"
else:
    # Bare hostname — build from Koyeb env vars
    _db_user = os.getenv("DATABASE_USER", "koyeb-adm")
    _db_pass = os.getenv("DATABASE_PASSWORD", "")
    _db_name = os.getenv("DATABASE_NAME", "koyebdb")
    _db_port = os.getenv("DATABASE_PORT", "5432")
    DATABASE_URL = f"postgresql+asyncpg://{_db_user}:{_db_pass}@{_raw_db_url}:{_db_port}/{_db_name}"

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

# Token pricing (USD)
TOKEN_PRICE_USD = float(os.getenv("TOKEN_PRICE_USD", "0.90"))
TOKEN_PACKS = {
    "single": {"tokens": 1, "name": "Single Token"},
    "starter": {"tokens": 5, "name": "Starter Pack"},
    "standard": {"tokens": 10, "name": "Standard Pack"},
    "pro": {"tokens": 25, "name": "Pro Pack"},
}

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
