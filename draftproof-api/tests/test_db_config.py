from app.config import _normalize_database_url


KOYEB_HOST = "ep-autumn-pond-anvor8lu.c-6.us-east-1.pg.koyeb.app"


def test_koyeb_postgres_url_gets_asyncpg_scheme_and_sslmode():
    url = _normalize_database_url(f"postgres://u:p@{KOYEB_HOST}/koyebdb")

    assert url == f"postgresql+asyncpg://u:p@{KOYEB_HOST}/koyebdb?sslmode=require"


def test_existing_sslmode_is_preserved():
    url = _normalize_database_url(f"postgres://u:p@{KOYEB_HOST}/koyebdb?sslmode=verify-full")

    assert url == f"postgresql+asyncpg://u:p@{KOYEB_HOST}/koyebdb?sslmode=verify-full"


def test_non_koyeb_url_only_gets_asyncpg_scheme():
    url = _normalize_database_url("postgres://u:p@db.internal.example.com/app")

    assert url == "postgresql+asyncpg://u:p@db.internal.example.com/app"
