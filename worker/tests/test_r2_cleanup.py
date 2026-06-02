from datetime import datetime, timedelta, timezone

import pytest

from app import r2_cleanup

UTC = timezone.utc
NOW = datetime(2026, 6, 3, 12, 0, 0, tzinfo=UTC)


class FakeR2:
    def __init__(self, pages):
        self.pages = pages
        self.list_calls = []
        self.deleted_batches = []

    def list_objects_v2(self, **kwargs):
        self.list_calls.append(kwargs)
        index = len(self.list_calls) - 1
        return self.pages[index]

    def delete_objects(self, **kwargs):
        self.deleted_batches.append(kwargs)
        return {"Deleted": kwargs["Delete"]["Objects"]}


def _object(key, age_days):
    return {"Key": key, "LastModified": NOW - timedelta(days=age_days)}


def test_cleanup_dry_run_counts_only_report_objects_older_than_retention(monkeypatch):
    monkeypatch.setattr(r2_cleanup.settings, "R2_BUCKET_NAME", "bucket")
    fake = FakeR2([
        {
            "Contents": [
                _object("reports/old/report.json", 4),
                _object("reports/new/report.json", 2),
            ],
            "IsTruncated": False,
        }
    ])

    result = r2_cleanup.cleanup_old_report_objects(
        retention_days=3,
        dry_run=True,
        now=NOW,
        s3=fake,
    )

    assert result.scanned == 2
    assert result.eligible == 1
    assert result.deleted == 0
    assert fake.deleted_batches == []
    assert fake.list_calls[0]["Prefix"] == "reports/"


def test_cleanup_delete_batches_eligible_objects(monkeypatch):
    monkeypatch.setattr(r2_cleanup.settings, "R2_BUCKET_NAME", "bucket")
    fake = FakeR2([
        {
            "Contents": [
                _object("reports/a/report.json", 3.1),
                _object("reports/a/report.pdf", 4),
                _object("reports/b/report.json", 1),
            ],
            "IsTruncated": False,
        }
    ])

    result = r2_cleanup.cleanup_old_report_objects(
        retention_days=3,
        dry_run=False,
        now=NOW,
        s3=fake,
    )

    assert result.scanned == 3
    assert result.eligible == 2
    assert result.deleted == 2
    assert result.errors == 0
    deleted_keys = [item["Key"] for item in fake.deleted_batches[0]["Delete"]["Objects"]]
    assert deleted_keys == ["reports/a/report.json", "reports/a/report.pdf"]


def test_cleanup_follows_pagination(monkeypatch):
    monkeypatch.setattr(r2_cleanup.settings, "R2_BUCKET_NAME", "bucket")
    fake = FakeR2([
        {
            "Contents": [_object("reports/a/report.json", 4)],
            "IsTruncated": True,
            "NextContinuationToken": "next",
        },
        {
            "Contents": [_object("reports/b/report.json", 5)],
            "IsTruncated": False,
        },
    ])

    result = r2_cleanup.cleanup_old_report_objects(
        retention_days=3,
        dry_run=False,
        now=NOW,
        s3=fake,
    )

    assert result.scanned == 2
    assert result.deleted == 2
    assert fake.list_calls[1]["ContinuationToken"] == "next"


def test_cleanup_rejects_empty_prefix():
    with pytest.raises(ValueError, match="prefix"):
        r2_cleanup.cleanup_old_report_objects(prefix=" ", s3=FakeR2([]))


def test_cleanup_rejects_non_positive_retention():
    with pytest.raises(ValueError, match="retention"):
        r2_cleanup.cleanup_old_report_objects(retention_days=0, s3=FakeR2([]))
