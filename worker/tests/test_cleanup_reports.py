import json
from datetime import datetime, timezone

import cleanup_reports
from app.db_cleanup import DbReportCleanupResult
from app.r2_cleanup import R2CleanupResult


NOW = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_combined_cleanup_dry_run_outputs_json_summary(monkeypatch, capsys):
    calls = {}

    def fake_r2(**kwargs):
        calls["r2"] = kwargs
        return R2CleanupResult(
            bucket="bucket",
            prefix="reports/",
            cutoff=NOW,
            dry_run=True,
            scanned=3,
            eligible=2,
        )

    def fake_db(**kwargs):
        calls["db"] = kwargs
        return DbReportCleanupResult(
            cutoff=NOW,
            dry_run=True,
            old_scan_jobs=1,
            old_rewrite_jobs=1,
        )

    monkeypatch.setattr(cleanup_reports, "cleanup_old_report_objects", fake_r2)
    monkeypatch.setattr(cleanup_reports, "cleanup_old_report_rows", fake_db)

    exit_code = cleanup_reports.main(["--retention-days", "3"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["retention_days"] == 3
    assert output["dry_run"] is True
    assert output["r2"]["eligible"] == 2
    assert output["db"]["old_scan_jobs"] == 1
    assert calls["r2"] == {"retention_days": 3, "prefix": None, "dry_run": True}
    assert calls["db"] == {"retention_days": 3, "dry_run": True}


def test_db_only_skips_r2(monkeypatch, capsys):
    def fail_r2(**kwargs):
        raise AssertionError("R2 cleanup should not run")

    def fake_db(**kwargs):
        return DbReportCleanupResult(cutoff=NOW, dry_run=True, old_scan_jobs=4)

    monkeypatch.setattr(cleanup_reports, "cleanup_old_report_objects", fail_r2)
    monkeypatch.setattr(cleanup_reports, "cleanup_old_report_rows", fake_db)

    exit_code = cleanup_reports.main(["--retention-days", "3", "--db-only"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["r2"] is None
    assert output["db"]["old_scan_jobs"] == 4


def test_delete_mode_validates_required_config(monkeypatch, capsys):
    monkeypatch.setattr(cleanup_reports.settings, "DATABASE_URL", "")
    monkeypatch.setattr(cleanup_reports.settings, "R2_ENDPOINT_URL", "")
    monkeypatch.setattr(cleanup_reports.settings, "R2_ACCESS_KEY_ID", "")
    monkeypatch.setattr(cleanup_reports.settings, "R2_SECRET_ACCESS_KEY", "")
    monkeypatch.setattr(cleanup_reports.settings, "R2_BUCKET_NAME", "")

    exit_code = cleanup_reports.main(["--retention-days", "3", "--delete"])

    captured = capsys.readouterr()
    assert exit_code == 1
    output = json.loads(captured.err)
    assert output["ok"] is False
    assert "DATABASE_URL" in output["error"]
    assert "R2_BUCKET_NAME" in output["error"]


def test_delete_mode_passes_delete_flags_when_config_is_present(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(cleanup_reports.settings, "DATABASE_URL", "postgresql://db")
    monkeypatch.setattr(cleanup_reports.settings, "R2_ENDPOINT_URL", "https://r2.example")
    monkeypatch.setattr(cleanup_reports.settings, "R2_ACCESS_KEY_ID", "access")
    monkeypatch.setattr(cleanup_reports.settings, "R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(cleanup_reports.settings, "R2_BUCKET_NAME", "bucket")

    def fake_r2(**kwargs):
        calls["r2"] = kwargs
        return R2CleanupResult(bucket="bucket", prefix="reports/", cutoff=NOW, dry_run=False)

    def fake_db(**kwargs):
        calls["db"] = kwargs
        return DbReportCleanupResult(cutoff=NOW, dry_run=False)

    monkeypatch.setattr(cleanup_reports, "cleanup_old_report_objects", fake_r2)
    monkeypatch.setattr(cleanup_reports, "cleanup_old_report_rows", fake_db)

    exit_code = cleanup_reports.main(["--retention-days", "3", "--prefix", "reports", "--delete"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is False
    assert calls["r2"] == {"retention_days": 3, "prefix": "reports", "dry_run": False}
    assert calls["db"] == {"retention_days": 3, "dry_run": False}
