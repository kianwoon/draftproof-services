"""Worker configuration - reads from environment / .env."""

import os

from pydantic_settings import BaseSettings
from dotenv import load_dotenv


# Several rewrite modules read role-specific V6 settings directly from
# os.environ. Loading .env here keeps local worker runs aligned with production
# service environment variables, without overriding real production env values.
load_dotenv(override=False)

# Enable the last-stage top-k features for the WORKER (poc code defaults are OFF; this turns them on
# in the running worker). setdefault keeps any explicit service-env override (set =0 to disable).
# worker/app/ is git-pulled live at startup, so this takes effect on redeploy without an image rebuild.
# The bracket pass also needs OPENROUTER_API_KEY (service env) and the qwen grammar model below.
os.environ.setdefault("DRAFTPROOF_TOPK_GROUNDING_GATE", "1")
os.environ.setdefault("DRAFTPROOF_V6_BRACKET_GROUNDING", "1")
os.environ.setdefault("DRAFTPROOF_V6_GRAMMAR_MODEL", "qwen/qwen-2.5-7b-instruct")
os.environ.setdefault("DRAFTPROOF_V6_GRAMMAR_BASE_URL", "https://openrouter.ai/api/v1")


class Settings(BaseSettings):
    DATABASE_URL: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"
    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "draftproof-reports"
    R2_PUBLIC_URL: str = ""
    DRAFTPROOF_R2_REPORT_PREFIX: str = "reports/"
    DRAFTPROOF_R2_REPORT_RETENTION_DAYS: int = 3
    PREDICTABILITY_MODEL: str = "gpt2"
    SEMANTIC_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    CEREBRAS_API_KEY: str = ""
    LLM_MODEL: str = ""
    DRAFTPROOF_PLANNER_MODEL: str = ""
    DRAFTPROOF_GENERATOR_MODEL: str = ""
    DRAFTPROOF_REWRITE_MODEL_LOCK: str = ""
    DRAFTPROOF_RETRY_MODEL: str = ""
    DRAFTPROOF_RETRY_MODEL_ENABLED: bool = False
    DRAFTPROOF_RETRY_MODEL_MAX_CALLS: int = 1
    DRAFTPROOF_REWRITE_V5_MODEL: str = ""
    DRAFTPROOF_REWRITE_V5_PLANNER_MODEL: str = ""
    DRAFTPROOF_REWRITE_V4_MODEL: str = ""
    DRAFTPROOF_V6_PLANNER_MODEL: str = ""
    DRAFTPROOF_V6_WRITER_MODEL: str = ""
    DRAFTPROOF_V6_GRAMMAR_MODEL: str = ""
    DRAFTPROOF_V6_GRAMMAR_BASE_URL: str = ""
    DRAFTPROOF_V6_GRAMMAR_API_KEY: str = ""
    DRAFTPROOF_V6_GRAMMAR_TIMEOUT: int = 120
    DRAFTPROOF_V6_SELECTOR_MODEL: str = ""
    DRAFTPROOF_V6_CEREBRAS_DIRECT: bool = False
    DRAFTPROOF_V6_PROVIDER_DEFAULT_ORDER: str = ""
    DRAFTPROOF_V6_PLANNER_PROVIDER_ORDER: str = ""
    DRAFTPROOF_V6_PLANNER_PROVIDER_SORT: str = ""
    DRAFTPROOF_V6_PLANNER_ALLOW_FALLBACKS: bool = True
    DRAFTPROOF_V6_WRITER_PROVIDER_ORDER: str = ""
    DRAFTPROOF_V6_WRITER_PROVIDER_SORT: str = ""
    DRAFTPROOF_V6_WRITER_ALLOW_FALLBACKS: bool = True
    DRAFTPROOF_V6_SELECTOR_PROVIDER_ORDER: str = ""
    DRAFTPROOF_V6_SELECTOR_PROVIDER_SORT: str = ""
    DRAFTPROOF_V6_SELECTOR_ALLOW_FALLBACKS: bool = True
    DRAFTPROOF_V6_WRITER_VARIANTS: int = 3
    DRAFTPROOF_V6_WRITER_FEEDBACK_ROUNDS: int = 1
    DRAFTPROOF_V6_MIN_WRITER_FEEDBACK_ROUNDS: int = 1
    DRAFTPROOF_V6_MAX_PASSES: str = ""
    DRAFTPROOF_V6_GRAMMER_REPAIR_ENABLED: bool = False
    DRAFTPROOF_V6_NATURALISATION_ENABLED: bool = False
    LLM_BASE_URL: str = ""
    SCAN_SOFT_TIME_LIMIT_SECONDS: int = 300
    SCAN_TIME_LIMIT_SECONDS: int = 330
    REWRITE_SOFT_TIME_LIMIT_SECONDS: int = 1100
    REWRITE_TIME_LIMIT_SECONDS: int = 1200
    CELERY_VISIBILITY_TIMEOUT_SECONDS: int = 1320
    REDIS_SOCKET_TIMEOUT_SECONDS: int = 30
    REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS: int = 10
    REDIS_SOCKET_KEEPALIVE: bool = True
    REDIS_HEALTH_CHECK_INTERVAL_SECONDS: int = 120
    CELERY_BROKER_POLLING_INTERVAL_SECONDS: int = 10
    DATABASE_CONNECT_TIMEOUT_SECONDS: int = 10
    DRAFTPROOF_REWRITE_V6_ENABLED: bool = True
    DRAFTPROOF_REWRITE_V5_ENABLED: bool = True
    DRAFTPROOF_REWRITE_V4_ENABLED: bool = True
    DRAFTPROOF_REWRITE_V2_ENABLED: bool = True
    DRAFTPROOF_REWRITE_V3_ENABLED: bool = False
    SCAN_COMPLETION_EMAIL_ENABLED: bool = False
    REWRITE_COMPLETION_EMAIL_ENABLED: bool = False
    EMAIL_PROVIDER: str = "cloudflare"
    EMAIL_FROM_ADDRESS: str = "support@draftproof.app"
    EMAIL_FROM_NAME: str = "DraftProof Support"
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_ACCOUNT_API_TOKEN: str = ""
    CLOUDFLARE_EMAIL_SEND_ENDPOINT: str = ""
    REWRITE_COMPLETION_EMAIL_MAX_CHARS: int = 50000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
settings.REWRITE_SOFT_TIME_LIMIT_SECONDS = max(settings.REWRITE_SOFT_TIME_LIMIT_SECONDS, 60)
settings.REWRITE_TIME_LIMIT_SECONDS = max(
    settings.REWRITE_TIME_LIMIT_SECONDS,
    settings.REWRITE_SOFT_TIME_LIMIT_SECONDS + 60,
)
settings.CELERY_VISIBILITY_TIMEOUT_SECONDS = max(
    settings.CELERY_VISIBILITY_TIMEOUT_SECONDS,
    settings.REWRITE_TIME_LIMIT_SECONDS + 300,
)
settings.REDIS_SOCKET_TIMEOUT_SECONDS = max(settings.REDIS_SOCKET_TIMEOUT_SECONDS, 1)
settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS = max(settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS, 1)
settings.REDIS_HEALTH_CHECK_INTERVAL_SECONDS = max(settings.REDIS_HEALTH_CHECK_INTERVAL_SECONDS, 1)
settings.CELERY_BROKER_POLLING_INTERVAL_SECONDS = max(settings.CELERY_BROKER_POLLING_INTERVAL_SECONDS, 1)
settings.DATABASE_CONNECT_TIMEOUT_SECONDS = max(settings.DATABASE_CONNECT_TIMEOUT_SECONDS, 1)
