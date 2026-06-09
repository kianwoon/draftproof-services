"""Local administrator CLI for DraftProof report housekeeping.

Dry-run:
    python worker/cleanup_reports.py --retention-days 3

Delete eligible R2 objects and database rows:
    python worker/cleanup_reports.py --retention-days 3 --delete
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.config import settings
from app.db_cleanup import cleanup_old_report_rows
from app.r2_cleanup import cleanup_old_report_objects


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _configured_retention_days() -> int:
    return settings.DRAFTPROOF_REPORT_RETENTION_DAYS or settings.DRAFTPROOF_R2_REPORT_RETENTION_DAYS


def _missing_config(*names: str) -> list[str]:
    return [name for name in names if not str(getattr(settings, name, "") or "").strip()]


def _validate_delete_config(*, include_r2: bool, include_db: bool) -> None:
    missing: list[str] = []
    if include_db:
        missing.extend(_missing_config("DATABASE_URL"))
    if include_r2:
        missing.extend(_missing_config(
            "R2_ENDPOINT_URL",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET_NAME",
        ))
    if missing:
        raise ValueError(f"Missing required cleanup config for delete mode: {', '.join(missing)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean up old DraftProof report artifacts and rows.")
    parser.add_argument(
        "--retention-days",
        type=int,
        default=_configured_retention_days(),
        help="Delete reports older than this many days. Defaults to DRAFTPROOF_REPORT_RETENTION_DAYS or 3.",
    )
    parser.add_argument("--prefix", default=None, help="R2 report prefix override.")
    parser.add_argument("--delete", action="store_true", help="Actually delete eligible objects and rows.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--r2-only", action="store_true", help="Only clean R2 report objects.")
    mode.add_argument("--db-only", action="store_true", help="Only clean database report rows.")
    return parser


def run_cleanup(args: argparse.Namespace) -> dict[str, Any]:
    include_r2 = not args.db_only
    include_db = not args.r2_only
    dry_run = not args.delete

    if args.retention_days < 1:
        raise ValueError("Report retention must be at least 1 day")
    if args.delete:
        _validate_delete_config(include_r2=include_r2, include_db=include_db)

    result: dict[str, Any] = {
        "retention_days": args.retention_days,
        "dry_run": dry_run,
        "r2": None,
        "db": None,
    }

    if include_r2:
        result["r2"] = asdict(cleanup_old_report_objects(
            retention_days=args.retention_days,
            prefix=args.prefix,
            dry_run=dry_run,
        ))
    if include_db:
        result["db"] = asdict(cleanup_old_report_rows(
            retention_days=args.retention_days,
            dry_run=dry_run,
        ))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_cleanup(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **result}, indent=2, default=_json_default))
    errors = result["r2"]["errors"] if isinstance(result.get("r2"), dict) else 0
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
