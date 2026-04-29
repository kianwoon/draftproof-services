import os

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://koyeb-adm:npg_a0Xkjwc4nYtA@ep-autumn-pond-anvor8lu.c-6.us-east-1.pg.koyeb.app:5432/koyebdb?ssl=require")

# Auth
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
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
