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
    PREDICTABILITY_MODEL: str = "gpt2-medium"
    LLM_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL: str = ""
    LLM_BASE_URL: str = ""
    SCAN_SOFT_TIME_LIMIT_SECONDS: int = 300
    SCAN_TIME_LIMIT_SECONDS: int = 330
    REWRITE_SOFT_TIME_LIMIT_SECONDS: int = 720
    REWRITE_TIME_LIMIT_SECONDS: int = 780
    CELERY_VISIBILITY_TIMEOUT_SECONDS: int = 900

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
