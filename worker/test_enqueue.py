"""Enqueue a test scan job for local worker testing.

Usage:
    python worker/test_enqueue.py --text "Your text here" --user-id <UUID>
    python worker/test_enqueue.py --file input.txt --user-id <UUID>
"""

import sys
import os
import argparse
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.db import create_scan_job
from app.celery_app import app as celery_app
from app.tasks import scan_document


def main():
    parser = argparse.ArgumentParser(description="Enqueue a DraftProof scan job")
    parser.add_argument("--text", help="Text to scan")
    parser.add_argument("--file", "-f", help="File to scan")
    parser.add_argument("--user-id", required=True, help="User UUID")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--sync", action="store_true", help="Run synchronously (no worker needed)")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide --text or --file")

    user_id = args.user_id
    if not user_id.count("-") == 4:
        # Not a UUID format — try to look up
        print(f"Warning: '{user_id}' doesn't look like a UUID")

    print(f"Creating scan job for user {user_id} ({len(text.split())} words)...")

    job_id = create_scan_job(user_id, text, verbose=args.verbose, do_rewrite=args.rewrite)
    print(f"Job created: {job_id}")

    if args.sync:
        print("Running synchronously (blocking)...")
        result = scan_document(job_id, text)
        print(f"Result: {result}")
    else:
        result = scan_document.delay(job_id, text)
        print(f"Task enqueued: {result.id}")
        print(f"Job ID: {job_id}")
        print(f"Monitor: celery -A app.celery_app events")


if __name__ == "__main__":
    main()
