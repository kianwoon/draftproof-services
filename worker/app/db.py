"""PostgreSQL operations — scan_jobs CRUD + billing integration."""

import hashlib
import json
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras

from .config import settings


@contextmanager
def get_conn():
    conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_scan_job(user_id: str, input_text: str, verbose: bool = False, do_rewrite: bool = False) -> str:
    """Insert a new scan_job row, return the job_id."""
    word_count = len(input_text.split())
    scan_type = "scan_rewrite" if do_rewrite else "scan"
    text_hash = hashlib.sha256(input_text.encode()).hexdigest()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO scan_jobs (user_id, input_text_hash, word_count, scan_type)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (user_id, text_hash, word_count, scan_type),
        )
        return str(cur.fetchone()["id"])


def get_scan_job(job_id: str) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM scan_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def update_job_status(job_id: str, status: str, **fields):
    """Update scan_job status and optional fields."""
    sets = []
    vals = []
    for key in (
        "tier",
        "ai_score",
        "writing_score",
        "finding_count",
        "report_urls",
        "error",
        "progress_percent",
        "progress_message",
    ):
        if key in fields:
            sets.append(f"{key} = %s")
            val = psycopg2.extras.Json(fields[key]) if key == "report_urls" else fields[key]
            vals.append(val)
    if status == "processing":
        sets.append("started_at = COALESCE(started_at, now())")
    if status == "completed":
        sets.append("completed_at = now()")
    if sets:
        with get_conn() as conn:
            conn.cursor().execute(
                f"UPDATE scan_jobs SET status = %s, {', '.join(sets)} WHERE id = %s",
                [status] + vals + [job_id],
            )
    else:
        with get_conn() as conn:
            conn.cursor().execute(
                "UPDATE scan_jobs SET status = %s WHERE id = %s",
                [status, job_id],
            )


def capture_credits(user_id: str, job_id: str, word_count: int):
    """Capture a credit reservation and create a usage_event."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, credit_account_id, tokens_reserved
               FROM credit_reservations
               WHERE job_id = %s AND status = 'active'
               LIMIT 1""",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return

        res_id = row["id"]
        acct_id = row["credit_account_id"]
        tokens_reserved = row["tokens_reserved"]

        cur.execute(
            "UPDATE credit_reservations SET status = 'captured' WHERE id = %s",
            (res_id,),
        )
        cur.execute(
            "UPDATE credit_accounts SET balance_tokens = balance_tokens - %s, reserved_tokens = reserved_tokens - %s WHERE id = %s",
            (tokens_reserved, tokens_reserved, acct_id),
        )
        cur.execute(
            """INSERT INTO usage_events
               (user_id, credit_account_id, event_type, tokens_charged, job_id, word_count)
               VALUES (%s, %s, 'scan', %s, %s, %s)""",
            (user_id, acct_id, tokens_reserved, job_id, word_count),
        )


def get_rewrite_job(job_id: str) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM rewrite_jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def claim_rewrite_job(job_id: str) -> Optional[dict]:
    """Atomically move a rewrite job into processing if it has not run yet."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE rewrite_jobs
               SET status = 'processing'
               WHERE id = %s
                 AND status IN ('pending', 'retrying')
               RETURNING *""",
            (job_id,),
        )
        return cur.fetchone()


def update_rewrite_status(job_id: str, status: str, error: str = None):
    sets = ["status = %s"]
    vals = [status]
    if status == "completed":
        sets.append("completed_at = now()")
    if error:
        sets.append("error = %s")
        vals.append(error)
    with get_conn() as conn:
        conn.cursor().execute(
            f"UPDATE rewrite_jobs SET {', '.join(sets)} WHERE id = %s",
            vals + [job_id],
        )


def release_rewrite_credits(job_id: str):
    """Release reserved tokens back to available balance (on failure/cancellation)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, credit_account_id, tokens_reserved
               FROM credit_reservations
               WHERE job_id = %s AND status = 'active'
               LIMIT 1""",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return
        res_id = row["id"]
        acct_id = row["credit_account_id"]
        tokens = row["tokens_reserved"]

        cur.execute("UPDATE credit_reservations SET status = 'released' WHERE id = %s", (res_id,))
        cur.execute(
            "UPDATE credit_accounts SET reserved_tokens = reserved_tokens - %s WHERE id = %s",
            (tokens, acct_id),
        )


def capture_rewrite_credits(user_id: str, job_id: str):
    """Capture credit reservation for a rewrite job."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, credit_account_id, tokens_reserved
               FROM credit_reservations
               WHERE job_id = %s AND status = 'active'
               LIMIT 1""",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return
        res_id = row["id"]
        acct_id = row["credit_account_id"]
        tokens = row["tokens_reserved"]

        cur.execute("UPDATE credit_reservations SET status = 'captured' WHERE id = %s", (res_id,))
        cur.execute(
            "UPDATE credit_accounts SET balance_tokens = balance_tokens - %s, reserved_tokens = reserved_tokens - %s WHERE id = %s",
            (tokens, tokens, acct_id),
        )
        cur.execute(
            """INSERT INTO usage_events
               (user_id, credit_account_id, event_type, tokens_charged, job_id)
               VALUES (%s, %s, 'rewrite', %s, %s)""",
            (user_id, acct_id, tokens, job_id),
        )
