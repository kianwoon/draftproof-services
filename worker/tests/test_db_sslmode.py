"""Neon/Koyeb managed Postgres rejects non-SSL connections:
    'connection is insecure (try using `sslmode=require`)'.
The worker's psycopg2 connection must force SSL. _sslmode_connect_kwargs adds sslmode=require unless
the DSN already sets one, format-agnostically (URI or keyword DSN). Worker-scoped (the API uses a
separate asyncpg engine)."""
from app.db import _sslmode_connect_kwargs

URI = "postgresql://u:p@ep-autumn-pond-anvor8lu.c-6.us-east-1.pg.koyeb.app:5432/db"
KEYWORD = "host=ep-autumn-pond-anvor8lu.c-6.us-east-1.pg.koyeb.app dbname=db user=u password=p"


def test_adds_sslmode_require_for_uri_without_ssl():
    assert _sslmode_connect_kwargs(URI) == {"sslmode": "require"}


def test_adds_sslmode_require_for_keyword_dsn_without_ssl():
    assert _sslmode_connect_kwargs(KEYWORD) == {"sslmode": "require"}


def test_respects_existing_sslmode_uri():
    assert _sslmode_connect_kwargs(URI + "?sslmode=verify-full") == {}


def test_respects_existing_sslmode_keyword():
    assert _sslmode_connect_kwargs(KEYWORD + " sslmode=disable") == {}


def test_empty_url_still_forces_require():
    # misconfigured/empty url: still force require so we never connect silently-insecure
    assert _sslmode_connect_kwargs("") == {"sslmode": "require"}
