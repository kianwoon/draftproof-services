"""CLI for purging old report-owned scan/rewrite rows from Postgres."""

from __future__ import annotations

import argparse
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.db_cleanup import cleanup_old_report_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean up old DB report rows.")
    parser.add_argument("--retention-days", type=int, default=3)
    parser.add_argument("--delete", action="store_true", help="Actually delete eligible rows.")
    args = parser.parse_args()

    result = cleanup_old_report_rows(
        retention_days=args.retention_days,
        dry_run=not args.delete,
    )
    print(json.dumps({
        "cutoff": result.cutoff.isoformat(),
        "dry_run": result.dry_run,
        "old_scan_jobs": result.old_scan_jobs,
        "old_rewrite_jobs": result.old_rewrite_jobs,
        "deleted_scan_jobs": result.deleted_scan_jobs,
        "deleted_rewrite_jobs": result.deleted_rewrite_jobs,
        "released_orphan_reservations": result.released_orphan_reservations,
        "released_orphan_tokens": result.released_orphan_tokens,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
