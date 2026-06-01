"""PostgreSQL operations — scan_jobs CRUD + billing integration."""

import hashlib
import json
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras

from .config import settings


def _sslmode_connect_kwargs(url: str) -> dict:
    """psycopg2 connect kwargs that force an SSL connection. Neon/Koyeb managed Postgres rejects
    non-SSL connections ('connection is insecure (try using `sslmode=require`)'), which makes every
    scan_document DB call fail at get_conn and retry forever. Add sslmode=require UNLESS the DSN
    already sets one (matches both URI `?sslmode=` and keyword `sslmode=` formats). Passing it as a
    kwarg is format-agnostic and avoids a libpq duplicate-parameter error. Worker-scoped -- the API
    uses a separate asyncpg engine."""
    return {} if (url and "sslmode=" in url) else {"sslmode": "require"}


@contextmanager
def get_conn():
    conn = psycopg2.connect(
        settings.DATABASE_URL,
        cursor_factory=psycopg2.extras.RealDictCursor,
        **_sslmode_connect_kwargs(settings.DATABASE_URL),
    )
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


def get_scan_user_email(job_id: str) -> Optional[str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT users.email
               FROM scan_jobs
               JOIN users ON users.id = scan_jobs.user_id
               WHERE scan_jobs.id = %s""",
            (job_id,),
        )
        row = cur.fetchone()
        return row["email"] if row and row.get("email") else None


def update_job_status(job_id: str, status: str, **fields):
    """Update scan_job status and optional fields."""
    def _fields(include_progress: bool = True):
        sets = []
        vals = []
        allowed = (
            "tier",
            "ai_score",
            "writing_score",
            "finding_count",
            "report_urls",
            "error",
            "progress_percent",
            "progress_message",
        )
        for key in allowed:
            if key in fields:
                if not include_progress and key in ("progress_percent", "progress_message"):
                    continue
                sets.append(f"{key} = %s")
                val = psycopg2.extras.Json(fields[key]) if key == "report_urls" else fields[key]
                vals.append(val)
        if status == "processing":
            sets.append("started_at = COALESCE(started_at, now())")
        if status == "completed":
            sets.append("completed_at = now()")
        return sets, vals

    def _execute(include_progress: bool = True):
        sets, vals = _fields(include_progress)
        with get_conn() as conn:
            if sets:
                conn.cursor().execute(
                    f"UPDATE scan_jobs SET status = %s, {', '.join(sets)} WHERE id = %s",
                    [status] + vals + [job_id],
                )
            else:
                conn.cursor().execute(
                    "UPDATE scan_jobs SET status = %s WHERE id = %s",
                    [status, job_id],
                )

    try:
        _execute()
    except psycopg2.errors.UndefinedColumn as exc:
        if "progress_" not in str(exc):
            raise
        _execute(include_progress=False)


def capture_credits(user_id: str, job_id: str, word_count: int):
    """Capture a scan credit reservation and create a usage_event (idempotent).

    The status flip is an atomic compare-and-swap: ``UPDATE ... WHERE
    status='active' RETURNING``. Only one caller can win, so a redelivered scan
    task -- or a concurrent API-side stale-recovery capture -- re-evaluates the
    WHERE against the already-captured row, matches zero rows, and no-ops. The
    balance is debited and the usage_event written exactly once (one active
    reservation per job is the upstream invariant).
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE credit_reservations
               SET status = 'captured'
               WHERE job_id = %s AND status = 'active'
               RETURNING credit_account_id, tokens_reserved""",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return

        acct_id = row["credit_account_id"]
        tokens_reserved = row["tokens_reserved"]

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


def get_rewrite_user_email(job_id: str) -> Optional[str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT users.email
               FROM rewrite_jobs
               JOIN users ON users.id = rewrite_jobs.user_id
               WHERE rewrite_jobs.id = %s""",
            (job_id,),
        )
        row = cur.fetchone()
        return row["email"] if row and row.get("email") else None


def is_rewrite_canceled(job_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM rewrite_jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
        return bool(row and row.get("status") == "canceled")


def list_processing_rewrite_jobs() -> list:
    """Return rewrite jobs currently in 'processing' (for the startup reconciler).

    At worker boot these are candidates orphaned by a previously-killed worker;
    the reconciler decides per-job whether each is genuinely stale.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, scan_id, user_id, created_at, status
               FROM rewrite_jobs
               WHERE status = 'processing'"""
        )
        return cur.fetchall()


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


def update_rewrite_status(
    job_id: str,
    status: str,
    error: str = None,
    progress_percent: int = None,
    progress_message: str = None,
) -> bool:
    def _fields(include_progress: bool = True):
        sets = ["status = %s"]
        vals = [status]
        if status == "completed":
            sets.append("completed_at = now()")
        if error is not None:
            sets.append("error = %s")
            vals.append(error)
        elif status == "completed":
            sets.append("error = NULL")
        if include_progress and progress_percent is not None:
            sets.append("progress_percent = %s")
            vals.append(max(0, min(100, int(progress_percent))))
        if include_progress and progress_message is not None:
            sets.append("progress_message = %s")
            vals.append(progress_message)
        return sets, vals

    def _execute(include_progress: bool = True):
        sets, vals = _fields(include_progress)
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE rewrite_jobs SET {', '.join(sets)} WHERE id = %s AND status <> 'canceled'",
                vals + [job_id],
            )
            return cur.rowcount > 0

    try:
        return _execute()
    except psycopg2.errors.UndefinedColumn as exc:
        if "progress_" not in str(exc):
            raise
        return _execute(include_progress=False)


def release_rewrite_credits(job_id: str):
    """Release reserved tokens back to available balance on failure/cancel (idempotent).

    Atomic compare-and-swap mirrors capture: only one caller flips an 'active'
    reservation to 'released', so release races safely against a concurrent
    capture (whichever flips status first wins; the other no-ops) and against a
    redelivered task -- reserved tokens are returned exactly once.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE credit_reservations
               SET status = 'released'
               WHERE job_id = %s AND status = 'active'
               RETURNING credit_account_id, tokens_reserved""",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return

        cur.execute(
            "UPDATE credit_accounts SET reserved_tokens = reserved_tokens - %s WHERE id = %s",
            (row["tokens_reserved"], row["credit_account_id"]),
        )


def capture_rewrite_credits(user_id: str, job_id: str):
    """Capture credit reservation for a rewrite job (idempotent).

    Atomic compare-and-swap: only one caller flips an 'active' reservation to
    'captured'. A redelivered rewrite task, or a concurrent API-side
    stale-recovery capture, re-evaluates WHERE status='active' against the
    already-captured row, matches zero rows, and no-ops -- so the rewrite is
    billed exactly once.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE credit_reservations
               SET status = 'captured'
               WHERE job_id = %s AND status = 'active'
               RETURNING credit_account_id, tokens_reserved""",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return

        acct_id = row["credit_account_id"]
        tokens = row["tokens_reserved"]

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
