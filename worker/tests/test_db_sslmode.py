"""Neon/Koyeb managed Postgres needs two things the worker's bundled libpq may not provide on its
own:
  - sslmode=require: the endpoint rejects non-SSL ('connection is insecure (try using
    `sslmode=require`)').
  - options=endpoint=<id>: Neon routes to the right compute via the TLS SNI extension; libpq only
    sends SNI from Postgres 14+, and an older bundled libpq triggers 'Control plane request failed'.
    Passing the compute id in libpq `options` is Neon's documented SNI-less workaround (and is
    harmless when SNI already works -- verified against the live endpoint).
_neon_connect_kwargs adds each only when the URL doesn't already specify it. Worker-scoped -- the
API uses a separate asyncpg engine."""
from app.db import _neon_connect_kwargs

NEON = "postgresql://u:p@ep-autumn-pond-anvor8lu.c-6.us-east-1.pg.koyeb.app:5432/koyebdb"
EP = "endpoint=ep-autumn-pond-anvor8lu"
TIMEOUT = {"connect_timeout": 10}


def test_neon_url_without_ssl_or_options_gets_both():
    assert _neon_connect_kwargs(NEON) == {**TIMEOUT, "sslmode": "require", "options": EP}


def test_existing_sslmode_kept_options_added():
    assert _neon_connect_kwargs(NEON + "?sslmode=verify-full") == {**TIMEOUT, "options": EP}


def test_existing_options_kept_sslmode_added():
    assert _neon_connect_kwargs(NEON + "?options=" + EP) == {**TIMEOUT, "sslmode": "require"}


def test_both_present_no_extra():
    assert _neon_connect_kwargs(NEON + f"?sslmode=require&options={EP}") == TIMEOUT


def test_existing_unrelated_query_still_gets_endpoint():
    assert _neon_connect_kwargs(NEON + "?application_name=draftproof") == {
        **TIMEOUT,
        "sslmode": "require",
        "options": EP,
    }


def test_existing_unrelated_options_still_gets_endpoint():
    assert _neon_connect_kwargs(NEON + "?options=-c%20statement_timeout%3D5000") == {
        **TIMEOUT,
        "sslmode": "require",
        "options": f"-c statement_timeout=5000 {EP}",
    }


def test_non_neon_host_gets_sslmode_only():
    assert _neon_connect_kwargs("postgresql://u:p@db.internal.example.com:5432/app") == {
        **TIMEOUT,
        "sslmode": "require",
    }


def test_empty_url_forces_sslmode_only():
    assert _neon_connect_kwargs("") == {**TIMEOUT, "sslmode": "require"}
