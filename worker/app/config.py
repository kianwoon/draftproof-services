"""Worker configuration — reads from environment / .env."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"
    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "draftproof-reports"
    R2_PUBLIC_URL: str = ""
    PREDICTABILITY_MODEL: str = "gpt2"
    SEMANTIC_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = ""
    DRAFTPROOF_PLANNER_MODEL: str = ""
    DRAFTPROOF_GENERATOR_MODEL: str = ""
    DRAFTPROOF_RETRY_MODEL: str = ""
    DRAFTPROOF_RETRY_MODEL_ENABLED: bool = False
    DRAFTPROOF_RETRY_MODEL_MAX_CALLS: int = 1
    LLM_BASE_URL: str = ""
    SCAN_SOFT_TIME_LIMIT_SECONDS: int = 300
    SCAN_TIME_LIMIT_SECONDS: int = 330
    REWRITE_SOFT_TIME_LIMIT_SECONDS: int = 2400
    REWRITE_TIME_LIMIT_SECONDS: int = 2700
    CELERY_VISIBILITY_TIMEOUT_SECONDS: int = 3000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
settings.REWRITE_SOFT_TIME_LIMIT_SECONDS = max(settings.REWRITE_SOFT_TIME_LIMIT_SECONDS, 2400)
settings.REWRITE_TIME_LIMIT_SECONDS = max(
    settings.REWRITE_TIME_LIMIT_SECONDS,
    settings.REWRITE_SOFT_TIME_LIMIT_SECONDS + 300,
)
settings.CELERY_VISIBILITY_TIMEOUT_SECONDS = max(
    settings.CELERY_VISIBILITY_TIMEOUT_SECONDS,
    settings.REWRITE_TIME_LIMIT_SECONDS + 300,
)
