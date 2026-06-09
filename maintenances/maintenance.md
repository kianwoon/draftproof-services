---
name: draftproof-maintenance
description: Use this when maintaining DraftProof operational housekeeping, especially local report cleanup for Cloudflare R2 and Postgres report rows, retention windows, dry-run/delete safety, config validation, and cleanup tests.
---

# DraftProof Maintenance Skill

This file is written as an AI-agent skill. Load it before changing operational cleanup, retention, storage, database housekeeping, or administrator scripts in this repo.

## Operating Rules

- Treat housekeeping as production-risk work: inspect current code and schema before changing behavior.
- Keep cleanup local, explicit, and reversible by default. Destructive cleanup must require an explicit delete flag.
- Do not hardcode bucket names, retention windows, database URLs, credentials, provider endpoints, or report prefixes. Use existing settings and CLI parameters.
- Do not delete payment, credit ledger, usage history, users, or identities as part of report housekeeping.
- Fix defective cleanup/accounting behavior before adding wrapper scripts or new automation.
- Preserve unrelated worktree changes. The repo may already be dirty.

## Current Report Housekeeping

Primary local entrypoint:

```bash
python worker/cleanup_reports.py --retention-days 3
python worker/cleanup_reports.py --retention-days 3 --delete
```

The script:

- defaults to dry-run;
- requires `--delete` before mutating R2 or Postgres;
- supports `--r2-only`, `--db-only`, and `--prefix`;
- emits JSON suitable for logs;
- reads default retention from `DRAFTPROOF_REPORT_RETENTION_DAYS`, falling back to the existing 3-day default;
- validates destructive-run config before delete mode.

Required config for delete mode:

- DB cleanup: `DATABASE_URL`
- R2 cleanup: `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`

## Cleanup Semantics

- “Older than N days” means:
  - Postgres rows: `created_at < now - N days`
  - R2 objects: `LastModified < now - N days`
- R2 cleanup only targets objects under the configured report prefix, defaulting to `DRAFTPROOF_R2_REPORT_PREFIX`.
- DB cleanup targets report-owned `scan_jobs` and `rewrite_jobs`.
- Before deleting DB rows, release active scan/rewrite reservations attached to those rows and decrement `credit_accounts.reserved_tokens`.
- Delete old `scan_jobs` first and rely on `rewrite_jobs.scan_id ON DELETE CASCADE`; then delete remaining old rewrite rows attached to newer scans.
- Count rewrites attached to old scans as cleanup targets even if their own `created_at` is newer, because scan deletion cascades them.

## Files To Inspect First

- `worker/cleanup_reports.py`
- `worker/app/db_cleanup.py`
- `worker/app/r2_cleanup.py`
- `worker/app/config.py`
- `draftproof-api/migrations/001_initial_schema.sql`
- `draftproof-api/migrations/002_scan_jobs.sql`
- `draftproof-api/migrations/005_rewrite_jobs.sql`
- `worker/tests/test_cleanup_reports.py`
- `worker/tests/test_db_cleanup.py`
- `worker/tests/test_r2_cleanup.py`

## Verification

Run targeted cleanup tests after any change:

```bash
PYTHONPATH=worker pytest worker/tests/test_r2_cleanup.py worker/tests/test_db_cleanup.py worker/tests/test_cleanup_reports.py
```

Run a compile check on touched cleanup scripts/modules:

```bash
python -m py_compile worker/app/config.py worker/app/db_cleanup.py worker/app/r2_cleanup.py worker/cleanup_reports.py worker/cleanup_r2_reports.py worker/cleanup_db_reports.py
```

Run whitespace validation before handoff:

```bash
git diff --check -- maintenances/maintenance.md worker/app/config.py worker/app/db_cleanup.py worker/app/r2_cleanup.py worker/cleanup_reports.py worker/tests/test_cleanup_reports.py worker/tests/test_db_cleanup.py worker/tests/test_r2_cleanup.py
```

## Administrator Runbook

1. Confirm environment variables point at the intended environment.
2. Run dry-run first:

```bash
python worker/cleanup_reports.py --retention-days 3
```

3. Review JSON output:
   - `dry_run` must be `true`;
   - `retention_days` must match the intended retention;
   - R2 `prefix` must match the intended report namespace;
   - DB and R2 eligible counts must be plausible.
4. Run delete only after dry-run is reviewed:

```bash
python worker/cleanup_reports.py --retention-days 3 --delete
```

5. Save the JSON output in operational logs when running against production.

## Common Failure Modes

- Missing R2 or DB config: delete mode exits with JSON error and does not run cleanup.
- Non-positive retention: rejected before cleanup starts.
- Active reservations on deleted jobs: must be released and reflected in `credit_accounts.reserved_tokens`.
- R2 delete errors: command exits non-zero when R2 reports delete errors.
- Prefix mistakes: never run delete unless the dry-run output shows the expected report prefix.
