"""Defence-judging Celery task (Task 7).

Worker-side counterpart to poc/detect/defence_judge.py (Task 5, the LLM judge) and
draftproof-api/app/routes/defence.py (Task 6, the API routes that insert a `pending`
defence_responses row and enqueue this task).

TASK NAME CONTRACT -- read this before touching the decorator below. The API never imports this
module directly (the root Dockerfile does not copy poc/ or worker/ into the API image); it
enqueues purely by name string via app/services/celery_client.py:

    judge_defence_answer = celery_app.signature("app.tasks.judge_defence_answer")

Celery's default task-name derivation is `<module>.<function name>`, using the module path the
function was actually DEFINED in. This file's natural module path is `app.defence` (it lives at
worker/app/defence.py, and the worker runs Celery from cwd=worker/ with `app/` as the top-level
package -- see worker/app/celery_app.py's `Celery("draftproof", include=["app.tasks"])` and
worker/entrypoint.sh's `cd /app/worker && celery -A app.celery_app worker`). A task defined here
with no override would therefore register as `app.defence.judge_defence_answer_task`, NOT the
`app.tasks.judge_defence_answer` the API expects -- an enqueued task would then be silently
unroutable (Celery has no handler under that name) rather than merely "not yet built".

Two things fix this, both required together:
  1. `name="app.tasks.judge_defence_answer"` explicitly overrides the derived name below.
  2. worker/app/tasks.py imports this module (`from . import defence`, alongside the existing
     `from . import model_preload` line) so the @app.task decorator actually RUNS at worker boot
     -- celery_app.py's `include=["app.tasks"]` only eagerly imports app.tasks itself, not this
     file, so without that import this task would never register at all, name override or not.
test_defence.py's test_task_registers_under_exact_contract_name asserts both halves: the task
object's own `.name`, AND its presence in the real Celery app's task registry.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Optional

# Mirrors tasks.py's own sys.path bootstrap (worker/app/tasks.py lines ~11-18) so `poc.*` imports
# below work even if this module is imported standalone (e.g. a test importing `app.defence`
# directly, without `app.tasks` -- which normally does this same setup -- having run first). The
# membership checks make re-running this a no-op when tasks.py already did it in-process.
_app_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_app_dir, "..", "..")
if _repo_root not in sys.path:
    sys.path.insert(0, os.path.abspath(_repo_root))
_poc_dir = os.path.join(_repo_root, "poc")
if _poc_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_poc_dir))

import psycopg2.extras

from .celery_app import app
from .config import settings
from .db import get_conn
from .storage import _client as _r2_client
from poc.detect.defence_judge import judge_defence_answer

logger = logging.getLogger(__name__)

# Context-paragraph char cap. Reuses the SAME env var poc/detect/defence_judge.py's own _prompt()
# caps the (already-assembled) context with -- one knob governs both layers, not a second
# disconnected constant. defence_judge.py's default for this env var is also 6000.
_CONTEXT_CHARS_ENV = "DRAFTPROOF_DEFENCE_CONTEXT_CHARS"
_DEFAULT_CONTEXT_CHARS = 6000


# ── DB access (raw SQL via psycopg2, matching worker/app/db.py's established pattern --
# the worker is a separate process from the API and does NOT import the API's SQLAlchemy models) ──


def get_defence_response(response_id: str) -> Optional[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM defence_responses WHERE id = %s", (response_id,))
        return cur.fetchone()


def update_defence_response(
    response_id: str,
    *,
    status: str,
    judgment: dict | None = None,
    judge_model: str | None = None,
) -> bool:
    """Persist a judging outcome (status in {'judged','failed'} -- the only two terminal states
    the defence_responses migration defines; there is no 'processing'/'retrying' state for this
    table, unlike scan_jobs/rewrite_jobs, so a mid-retry attempt intentionally leaves the row
    untouched at 'pending' -- see _perform_judging's retry branch).

    CAS-guarded (`WHERE status = 'pending'`), mirroring the compare-and-swap idiom in
    worker/app/db.py's update_job_status / update_rewrite_status: a redelivered/raced task
    execution landing here after another execution already resolved the row is a no-op (rowcount
    0) instead of clobbering an already-judged/failed result.

    On a fail-open (judge_defence_answer returned None) or a report-fetch failure, callers pass
    only status="failed" -- judgment/judge_model stay None here, so those columns are never
    included in the SET clause and the row's `judgment` column is left exactly as it was
    (NULL), per the brief: "leave judgment null, answer_text untouched (re-judgeable)".
    """
    sets = ["status = %s", "judged_at = now()"]
    vals: list[Any] = [status]
    if judgment is not None:
        sets.append("judgment = %s")
        vals.append(psycopg2.extras.Json(judgment))
    if judge_model is not None:
        sets.append("judge_model = %s")
        vals.append(judge_model)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE defence_responses SET {', '.join(sets)} WHERE id = %s AND status = 'pending'",
            vals + [response_id],
        )
        return cur.rowcount > 0


def _best_effort_defence_status_update(response_id: str, **kwargs) -> bool:
    """Persist a terminal status without blocking Celery retry/failure handling -- mirrors
    tasks.py's _best_effort_scan_status_update. Needed because this is called from the
    retries-exhausted branch of _perform_judging's own except block: if the DB write itself
    fails there too, we must not let that raise and mask the original exception being re-raised.
    """
    try:
        return update_defence_response(response_id, **kwargs)
    except Exception:
        logger.warning("Defence response status update failed for %s", response_id, exc_info=True)
        return False


# ── Context-paragraph assembly (anchor quote's paragraph +/- 1) ────────────────────────────────


def _context_chars_cap() -> int:
    try:
        return max(1, int(os.environ.get(_CONTEXT_CHARS_ENV, _DEFAULT_CONTEXT_CHARS)))
    except (TypeError, ValueError):
        return _DEFAULT_CONTEXT_CHARS


def _normalize_for_match(text: Any) -> str:
    """Same whitespace/case normalization poc/detect/critical_thinking_llm.py's anchoring guard
    uses (_normalize_questions' _norm) -- the anchor_quote stored on the defence_responses row
    was itself validated with that exact normalization against this exact paragraph list when it
    was generated, so matching it back the same way is what makes the match reliable."""
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _fetch_report_json(scan_id: str) -> dict:
    """Fetch the scan's report JSON from R2 (same key convention as storage.upload_report_files /
    tasks.py's run_rewrite report re-read: reports/{scan_id}/report.json), reusing the SAME R2
    client helper storage.py already exposes rather than hand-rolling a new boto3 client. Raises
    on any failure -- the caller (_perform_judging) treats that as a PERMANENT failure for this
    response, not a retryable one (mirrors run_rewrite's "Original report not found in R2"
    handling)."""
    s3 = _r2_client()
    resp = s3.get_object(Bucket=settings.R2_BUCKET_NAME, Key=f"reports/{scan_id}/report.json")
    return json.loads(resp["Body"].read())


def _paragraph_window_text(report: Any, anchor_quote: str) -> str:
    """Pure: locate the paragraph containing anchor_quote in report["scan_intelligence"]
    ["document"]["paragraphs"] (the SAME list, in document order, that
    poc/detect/critical_thinking_llm.py's _document_paragraphs() reads to generate anchor_quote
    in the first place -- poc/report/report.py's _paragraph_map sorts these rows by start_char),
    and return that paragraph plus its immediate neighbours (index-1, index, index+1), joined and
    capped. Returns "" (never raises) when the structure is missing, the quote is empty, or no
    paragraph matches -- none of those are errors, just "no extra context available"."""
    if not isinstance(report, dict):
        return ""
    document = (report.get("scan_intelligence") or {}).get("document")
    paragraphs = document.get("paragraphs") if isinstance(document, dict) else None
    if not isinstance(paragraphs, list) or not paragraphs:
        return ""

    quote_norm = _normalize_for_match(anchor_quote)
    if not quote_norm:
        return ""

    match_idx = -1
    for i, para in enumerate(paragraphs):
        if isinstance(para, dict) and quote_norm in _normalize_for_match(para.get("text")):
            match_idx = i
            break
    if match_idx < 0:
        return ""

    window = paragraphs[max(0, match_idx - 1): match_idx + 2]
    texts = [str(p.get("text") or "").strip() for p in window if isinstance(p, dict) and p.get("text")]
    joined = "\n\n".join(texts)
    cap = _context_chars_cap()
    return joined[:cap].rstrip() if len(joined) > cap else joined


def _fetch_context_paragraphs(scan_id: str, anchor_quote: str) -> str:
    """Fetch the report and extract the anchor quote's paragraph window. Raises on a
    fetch/parse failure (see _fetch_report_json); returns "" (no context, not an error) when the
    report loads fine but has no matching paragraph."""
    report = _fetch_report_json(scan_id)
    return _paragraph_window_text(report, anchor_quote)


# ── The task itself ──────────────────────────────────────────────────────────────────────────


def _perform_judging(self, response_id: str) -> dict:
    """All business logic, factored out of the @app.task-decorated function below so tests can
    call it directly with a lightweight fake `self` (duck-typing .request.retries, .max_retries,
    .retry()) instead of exercising real Celery broker/eager-mode machinery."""
    try:
        row = get_defence_response(response_id)
        if not row:
            logger.info("judge_defence_answer: response %s not found; skipping", response_id)
            return {"status": "skipped", "reason": "response_not_found"}
        if row.get("status") != "pending":
            # Redelivered task (task_acks_late=True) racing a run that already resolved this
            # row, or a stray re-enqueue -- abort rather than re-judge (mirrors scan_document's
            # "job already resolved; skipping redelivered/raced run" guard in tasks.py).
            logger.info(
                "judge_defence_answer: response %s already %s; skipping redelivered/raced run",
                response_id, row.get("status"),
            )
            return {"status": "skipped", "reason": f"already_{row.get('status')}"}

        scan_id = str(row.get("scan_id"))
        anchor_quote = row.get("anchor_quote") or ""

        try:
            context_paragraphs = _fetch_context_paragraphs(scan_id, anchor_quote)
        except Exception:
            # Permanent, not transient: a missing/unreadable report won't reappear on retry
            # (mirrors run_rewrite's report-not-found handling in tasks.py, which also fails
            # immediately without going through the retry ladder below).
            logger.warning(
                "judge_defence_answer: report fetch failed for scan %s (response %s); marking "
                "failed (not retried)",
                scan_id, response_id, exc_info=True,
            )
            _best_effort_defence_status_update(response_id, status="failed")
            return {"status": "failed", "response_id": response_id, "error": "report not found"}

        llm_api_key = settings.LLM_API_KEY or settings.OPENROUTER_API_KEY or settings.CEREBRAS_API_KEY or None
        llm_base_url = settings.LLM_BASE_URL or None

        # No `model=` passed -- judge_defence_answer resolves its own model priority chain
        # internally (DRAFTPROOF_DEFENCE_JUDGE_MODEL / _V6_PLANNER_MODEL / _PLANNER_MODEL /
        # LLM_MODEL / default), the same pattern scan_enrichment.py's run_critical_thinking /
        # run_reflective_questions use for the sibling critical_thinking_llm.py functions.
        judgment = judge_defence_answer(
            question=row.get("question") or "",
            anchor_quote=anchor_quote,
            dimension=row.get("dimension") or "",
            answer_text=row.get("answer_text") or "",
            context_paragraphs=context_paragraphs,
            api_key=llm_api_key,
            base_url=llm_base_url,
        )

        if judgment is None:
            # Fail-open per judge_defence_answer's contract (disable/no key/malformed
            # JSON/gateway error/any exception -> None, never raises). Per the brief: leave
            # judgment null, answer_text untouched -- the row stays re-judgeable via a new
            # attempt row (defence_responses is one row per attempt, not overwritten in place).
            _best_effort_defence_status_update(response_id, status="failed")
            return {"status": "failed", "response_id": response_id}

        update_defence_response(
            response_id,
            status="judged",
            judgment=judgment,
            judge_model=judgment.get("model"),
        )
        return {"status": "judged", "response_id": response_id}

    except Exception as e:
        # Unexpected exception anywhere above (DB read/write error, or any other bug) -- matches
        # scan_document/run_rewrite's outer retry-then-fail escalation in tasks.py exactly.
        if self.request.retries < self.max_retries:
            logger.warning(
                "judge_defence_answer: transient failure for %s (attempt %d/%d); retrying",
                response_id, self.request.retries + 1, self.max_retries, exc_info=True,
            )
            raise self.retry(exc=e)
        logger.warning(
            "judge_defence_answer: giving up on %s after %d retries",
            response_id, self.max_retries, exc_info=True,
        )
        _best_effort_defence_status_update(response_id, status="failed")
        raise  # Re-raise original -- Celery marks as FAILURE, not RETRY.


@app.task(
    bind=True,
    max_retries=1,
    default_retry_delay=30,
    name="app.tasks.judge_defence_answer",
)
def judge_defence_answer_task(self, response_id: str) -> dict:
    """Judge one defence-readiness answer (Task 7).

    Loads the defence_responses row by response_id, fetches the scan's report JSON from R2 for
    context_paragraphs (the anchor quote's paragraph +/- 1), resolves LLM creds via the worker's
    settings pattern, calls poc.detect.defence_judge.judge_defence_answer(...), and writes
    judgment/status/judged_at back to the row. See _perform_judging for the full flow and the
    module docstring above for why this MUST resolve to exactly `app.tasks.judge_defence_answer`.
    """
    return _perform_judging(self, response_id)
