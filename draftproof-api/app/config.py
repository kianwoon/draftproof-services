import os

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

# Token pricing (SGD)
TOKEN_PRICE_SGD = 2.90
TOKEN_PACKS = {
    "single": {"tokens": 1, "name": "Single Token"},
    "starter": {"tokens": 5, "name": "Starter Pack"},
    "standard": {"tokens": 10, "name": "Standard Pack"},
    "pro": {"tokens": 25, "name": "Pro Pack"},
}

# Frontend URL for redirects
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Redis / Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# R2 Storage (for fetching report JSON)
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "draftproof-reports")

# Rewrite
REWRITE_TOKEN_COST = 2
