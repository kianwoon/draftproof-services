from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from app import db_cleanup
from app.db_cleanup import _retention_days, _utc_datetime


def test_retention_days_rejects_non_positive_values():
    with pytest.raises(ValueError, match="retention"):
        _retention_days(0)


def test_utc_datetime_treats_naive_value_as_utc():
    value = datetime(2026, 6, 3, 12, 0, 0)

    assert _utc_datetime(value).tzinfo == timezone.utc


def test_utc_datetime_converts_aware_value_to_utc():
    sg = timezone.utc
    value = datetime(2026, 6, 3, 12, 0, 0, tzinfo=sg)

    assert _utc_datetime(value).tzinfo == timezone.utc


class FakeCursor:
    def __init__(self):
        self.operations = []
        self._next_row = None
        self._rewrite_count_calls = 0

    def execute(self, sql, params=None):
        compact_sql = " ".join(sql.split())
        self.operations.append(compact_sql)

        if compact_sql.startswith("SELECT count(*) AS n FROM scan_jobs"):
            self._next_row = {"n": 1}
            return
        if compact_sql.startswith("SELECT count(*) AS n FROM rewrite_jobs"):
            self._rewrite_count_calls += 1
            self._next_row = {"n": 2 if self._rewrite_count_calls == 1 else 0}
            return
        if "WITH old_jobs AS" in compact_sql:
            self._next_row = {"n": 2, "tokens": 5, "accounts": 1}
            return
        if "WITH stale AS" in compact_sql:
            self._next_row = {"n": 1, "tokens": 2, "accounts": 1}
            return
        if "DELETE FROM scan_jobs" in compact_sql:
            self._next_row = {"n": 1}
            return
        if compact_sql.startswith("DELETE FROM rewrite_jobs"):
            self._next_row = None
            return

        raise AssertionError(f"Unexpected SQL: {compact_sql}")

    def fetchone(self):
        if self._next_row is None:
            raise AssertionError("No row prepared for fetchone")
        return self._next_row


class FakeConn:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_cleanup_dry_run_counts_old_rows_without_mutation(monkeypatch):
    fake_conn = FakeConn()

    @contextmanager
    def fake_get_conn():
        yield fake_conn

    monkeypatch.setattr(db_cleanup, "get_conn", fake_get_conn)

    result = db_cleanup.cleanup_old_report_rows(
        retention_days=3,
        dry_run=True,
        now=datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert result.dry_run is True
    assert result.old_scan_jobs == 1
    assert result.old_rewrite_jobs == 2
    assert result.deleted_scan_jobs == 0
    assert result.deleted_rewrite_jobs == 0
    assert not any("DELETE FROM" in operation for operation in fake_conn.cursor_obj.operations)
    assert not any("UPDATE credit_accounts" in operation for operation in fake_conn.cursor_obj.operations)


def test_cleanup_delete_releases_reservations_updates_accounts_and_deletes_in_order(monkeypatch):
    fake_conn = FakeConn()

    @contextmanager
    def fake_get_conn():
        yield fake_conn

    monkeypatch.setattr(db_cleanup, "get_conn", fake_get_conn)

    result = db_cleanup.cleanup_old_report_rows(
        retention_days=3,
        dry_run=False,
        now=datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert result.dry_run is False
    assert result.old_scan_jobs == 1
    assert result.old_rewrite_jobs == 2
    assert result.released_orphan_reservations == 3
    assert result.released_orphan_tokens == 7
    assert result.updated_credit_accounts == 2
    assert result.deleted_scan_jobs == 1
    assert result.deleted_rewrite_jobs == 2

    operations = fake_conn.cursor_obj.operations
    release_index = next(i for i, sql in enumerate(operations) if "WITH old_jobs AS" in sql)
    scan_delete_index = next(i for i, sql in enumerate(operations) if "DELETE FROM scan_jobs" in sql)
    rewrite_delete_index = next(i for i, sql in enumerate(operations) if sql.startswith("DELETE FROM rewrite_jobs"))
    assert release_index < scan_delete_index < rewrite_delete_index
    assert any("UPDATE credit_accounts ca SET reserved_tokens = GREATEST" in sql for sql in operations)
