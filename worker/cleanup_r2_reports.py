"""CLI for deleting old DraftProof report artifacts from R2."""

from __future__ import annotations

import argparse
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.r2_cleanup import cleanup_old_report_objects


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean up old R2 report objects.")
    parser.add_argument("--retention-days", type=int, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--delete", action="store_true", help="Actually delete eligible objects.")
    args = parser.parse_args()

    result = cleanup_old_report_objects(
        retention_days=args.retention_days,
        prefix=args.prefix,
        dry_run=not args.delete,
    )
    print(json.dumps({
        "bucket": result.bucket,
        "prefix": result.prefix,
        "cutoff": result.cutoff.isoformat(),
        "dry_run": result.dry_run,
        "scanned": result.scanned,
        "eligible": result.eligible,
        "deleted": result.deleted,
        "errors": result.errors,
    }, indent=2))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
