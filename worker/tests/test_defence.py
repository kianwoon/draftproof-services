"""Tests for the defence-judging Celery task (Task 7).

Task-name resolution is the single most safety-critical fact this task must get right: the API
enqueues by NAME STRING ONLY ("app.tasks.judge_defence_answer", never importing the worker task
module directly -- see draftproof-api/app/routes/defence.py + app/services/celery_client.py). If
the resolved name here doesn't match exactly, the enqueued task silently vanishes or errors --
never executing. test_task_registers_under_exact_contract_name is the load-bearing assertion for
that contract.

No real DB/R2/LLM/network calls anywhere in this file: get_defence_response / update_defence_
response / _fetch_report_json / judge_defence_answer are all monkeypatched. The DB-helper tests
use a tiny in-memory fake Postgres cursor, mirroring test_billing_idempotency.py /
test_free_scan_refund.py elsewhere in this suite.
"""

from contextlib import contextmanager

import app.defence as defence
import app.tasks  # noqa: F401  -- importing tasks.py is what wires app.defence onto the real app
from app.celery_app import app as celery_app


# ── Fakes ──────────────────────────────────────────────────────────────────


class _Request:
    pass


class _FakeSelf:
    """Duck-types the bound-task `self` Celery passes when bind=True, without touching real
    Celery broker/eager-mode machinery."""

    def __init__(self, retries=0, max_retries=1):
        self.request = _Request()
        self.request.retries = retries
        self.max_retries = max_retries
        self.retry_calls = []

    def retry(self, exc=None):
        self.retry_calls.append(exc)
        return _RetrySignal(exc)


class _RetrySignal(Exception):
    """Stand-in for celery.exceptions.Retry. Production code does `raise self.retry(...)`; this
    fake's retry() returns an instance of this so the raise actually raises something."""

    def __init__(self, exc):
        super().__init__(str(exc))
        self.exc = exc


PENDING_ROW = {
    "id": "resp-1",
    "scan_id": "scan-1",
    "user_id": "user-1",
    "question_index": 0,
    "dimension": "reasoning_depth",
    "question": "Why does X follow from Y?",
    "anchor_quote": "the second paragraph's claim",
    "answer_text": "Because Y directly implies X given Z.",
    "attempt": 1,
    "status": "pending",
    "judgment": None,
    "judge_model": None,
}


# ── Task name resolution (the critical contract) ────────────────────────────


def test_task_registers_under_exact_contract_name():
    assert defence.judge_defence_answer_task.name == "app.tasks.judge_defence_answer"
    assert "app.tasks.judge_defence_answer" in celery_app.tasks
    assert celery_app.tasks["app.tasks.judge_defence_answer"] is defence.judge_defence_answer_task


# ── _perform_judging: success / fail-open / skip / retry paths ─────────────


def test_success_path_writes_judged_status_and_judgment(monkeypatch):
    monkeypatch.setattr(defence, "get_defence_response", lambda rid: dict(PENDING_ROW))
    monkeypatch.setattr(defence, "_fetch_report_json", lambda scan_id: {})
    calls = []
    monkeypatch.setattr(
        defence, "update_defence_response", lambda rid, **kw: calls.append((rid, kw)) or True
    )
    judgment = {
        "schema_version": "defence_judge_v1",
        "model": "openai/gpt-oss-120b",
        "generated_at": 123,
        "axes": {},
        "overall": {"level": "high", "score": 80, "derived": False},
        "flags": [],
    }
    monkeypatch.setattr(defence, "judge_defence_answer", lambda **kwargs: judgment)

    result = defence._perform_judging(_FakeSelf(), "resp-1")

    assert result["status"] == "judged"
    assert calls == [
        ("resp-1", {"status": "judged", "judgment": judgment, "judge_model": "openai/gpt-oss-120b"})
    ]


def test_fail_open_on_none_judgment_marks_failed_without_touching_judgment(monkeypatch):
    monkeypatch.setattr(defence, "get_defence_response", lambda rid: dict(PENDING_ROW))
    monkeypatch.setattr(defence, "_fetch_report_json", lambda scan_id: {})
    calls = []
    monkeypatch.setattr(
        defence, "update_defence_response", lambda rid, **kw: calls.append((rid, kw)) or True
    )
    monkeypatch.setattr(defence, "judge_defence_answer", lambda **kwargs: None)

    result = defence._perform_judging(_FakeSelf(), "resp-1")

    assert result["status"] == "failed"
    assert len(calls) == 1
    rid, kwargs = calls[0]
    assert rid == "resp-1"
    assert kwargs["status"] == "failed"
    assert kwargs.get("judgment") is None  # never writes a judgment on fail-open


def test_missing_response_row_is_skipped_without_any_write(monkeypatch):
    monkeypatch.setattr(defence, "get_defence_response", lambda rid: None)
    update_calls = []
    monkeypatch.setattr(
        defence, "update_defence_response", lambda rid, **kw: update_calls.append((rid, kw))
    )
    judge_calls = []
    monkeypatch.setattr(defence, "judge_defence_answer", lambda **kw: judge_calls.append(kw))

    result = defence._perform_judging(_FakeSelf(), "missing-id")

    assert result["status"] == "skipped"
    assert update_calls == []
    assert judge_calls == []


def test_already_resolved_row_is_skipped_idempotently(monkeypatch):
    """Redelivered task (task_acks_late=True) landing on an already-judged row must not re-judge."""
    row = dict(PENDING_ROW)
    row["status"] = "judged"
    monkeypatch.setattr(defence, "get_defence_response", lambda rid: row)
    judge_calls = []
    monkeypatch.setattr(defence, "judge_defence_answer", lambda **kw: judge_calls.append(kw))
    update_calls = []
    monkeypatch.setattr(
        defence, "update_defence_response", lambda rid, **kw: update_calls.append((rid, kw))
    )

    result = defence._perform_judging(_FakeSelf(), "resp-1")

    assert result["status"] == "skipped"
    assert judge_calls == []
    assert update_calls == []


def test_report_fetch_failure_marks_failed_without_retry(monkeypatch):
    """Mirrors run_rewrite's 'Original report not found in R2' handling: a missing/unreadable
    report is a PERMANENT failure for this response (retrying won't make R2 have the file), not
    a transient one -- must NOT go through the self.retry() escalation ladder."""
    monkeypatch.setattr(defence, "get_defence_response", lambda rid: dict(PENDING_ROW))

    def boom(scan_id):
        raise RuntimeError("R2 object not found")

    monkeypatch.setattr(defence, "_fetch_report_json", boom)
    calls = []
    monkeypatch.setattr(
        defence, "update_defence_response", lambda rid, **kw: calls.append((rid, kw)) or True
    )
    judge_calls = []
    monkeypatch.setattr(defence, "judge_defence_answer", lambda **kw: judge_calls.append(kw))

    fake_self = _FakeSelf(retries=0, max_retries=1)
    result = defence._perform_judging(fake_self, "resp-1")

    assert result["status"] == "failed"
    assert fake_self.retry_calls == []  # no retry attempted
    assert len(calls) == 1
    assert calls[0][1]["status"] == "failed"
    assert judge_calls == []  # never reached the judge call


def test_transient_exception_retries_when_attempts_remain(monkeypatch):
    def boom(rid):
        raise ConnectionError("db unreachable mid-flow")

    monkeypatch.setattr(defence, "get_defence_response", boom)

    fake_self = _FakeSelf(retries=0, max_retries=1)
    raised = None
    try:
        defence._perform_judging(fake_self, "resp-1")
    except _RetrySignal as exc:
        raised = exc

    assert raised is not None, "expected the retry sentinel to propagate"
    assert len(fake_self.retry_calls) == 1


def test_transient_exception_marks_failed_after_retries_exhausted(monkeypatch):
    def boom(rid):
        raise ConnectionError("db unreachable mid-flow")

    monkeypatch.setattr(defence, "get_defence_response", boom)
    calls = []
    monkeypatch.setattr(
        defence, "update_defence_response", lambda rid, **kw: calls.append((rid, kw)) or True
    )

    fake_self = _FakeSelf(retries=1, max_retries=1)  # already at the ceiling
    raised = None
    try:
        defence._perform_judging(fake_self, "resp-1")
    except ConnectionError as exc:
        raised = exc

    assert raised is not None, "expected the original exception to propagate (Celery FAILURE)"
    assert fake_self.retry_calls == []
    assert len(calls) == 1
    assert calls[0][1]["status"] == "failed"


# ── _paragraph_window_text: pure paragraph-matching logic ──────────────────


def test_paragraph_window_text_returns_neighbors_around_match():
    report = {
        "scan_intelligence": {
            "document": {
                "paragraphs": [
                    {"paragraph_id": "p001", "text": "First paragraph text."},
                    {"paragraph_id": "p002", "text": "Second paragraph has the important claim here."},
                    {"paragraph_id": "p003", "text": "Third paragraph text."},
                    {"paragraph_id": "p004", "text": "Fourth paragraph text."},
                ]
            }
        }
    }
    result = defence._paragraph_window_text(report, "the important claim")
    assert "First paragraph" in result
    assert "important claim" in result
    assert "Third paragraph" in result
    assert "Fourth paragraph" not in result


def test_paragraph_window_text_clamps_at_document_start():
    report = {
        "scan_intelligence": {
            "document": {
                "paragraphs": [
                    {"paragraph_id": "p001", "text": "Opening claim right here."},
                    {"paragraph_id": "p002", "text": "Second paragraph text."},
                ]
            }
        }
    }
    result = defence._paragraph_window_text(report, "opening claim")
    assert "Opening claim" in result
    assert "Second paragraph" in result


def test_paragraph_window_text_no_match_returns_empty():
    report = {
        "scan_intelligence": {
            "document": {"paragraphs": [{"paragraph_id": "p001", "text": "Nothing relevant here."}]}
        }
    }
    assert defence._paragraph_window_text(report, "a quote that does not appear anywhere") == ""


def test_paragraph_window_text_missing_structure_returns_empty():
    assert defence._paragraph_window_text({}, "anything") == ""
    assert defence._paragraph_window_text(None, "anything") == ""
    assert defence._paragraph_window_text({"scan_intelligence": {}}, "anything") == ""


def test_paragraph_window_text_empty_anchor_quote_returns_empty():
    report = {
        "scan_intelligence": {
            "document": {"paragraphs": [{"paragraph_id": "p001", "text": "Some paragraph text."}]}
        }
    }
    assert defence._paragraph_window_text(report, "") == ""


def test_paragraph_window_text_caps_length(monkeypatch):
    monkeypatch.setattr(defence, "_context_chars_cap", lambda: 20)
    report = {
        "scan_intelligence": {
            "document": {
                "paragraphs": [
                    {"paragraph_id": "p001", "text": "A" * 50 + " match-quote " + "B" * 50}
                ]
            }
        }
    }
    result = defence._paragraph_window_text(report, "match-quote")
    assert len(result) <= 20


def test_context_chars_cap_reads_env_and_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("DRAFTPROOF_DEFENCE_CONTEXT_CHARS", raising=False)
    assert defence._context_chars_cap() == 6000
    monkeypatch.setenv("DRAFTPROOF_DEFENCE_CONTEXT_CHARS", "500")
    assert defence._context_chars_cap() == 500


# ── _best_effort_defence_status_update ──────────────────────────────────────


def test_best_effort_defence_status_update_swallows_exceptions(monkeypatch):
    def fail_update(*_a, **_kw):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(defence, "update_defence_response", fail_update)
    assert defence._best_effort_defence_status_update("resp-1", status="failed") is False


def test_best_effort_defence_status_update_reports_success(monkeypatch):
    monkeypatch.setattr(defence, "update_defence_response", lambda rid, **kw: True)
    assert defence._best_effort_defence_status_update("resp-1", status="failed") is True


# ── get_defence_response / update_defence_response: fake-Postgres DB layer ─


class _FakeCursor:
    def __init__(self, state, log):
        self.state = state
        self.log = log
        self._last = None
        self.rowcount = 0

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        self.log.append((s, params))
        row = self.state.get("row")
        if s.startswith("SELECT * FROM defence_responses WHERE id"):
            self._last = row if row and row["id"] == params[0] else None
            self.rowcount = 0
        elif s.startswith("UPDATE defence_responses SET"):
            self._last = None
            self.rowcount = 1 if row and row.get("status") == "pending" else 0
        else:  # pragma: no cover - defensive
            self._last = None
            self.rowcount = 0

    def fetchone(self):
        return self._last


class _FakeConn:
    def __init__(self, state, log):
        self.state = state
        self.log = log

    def cursor(self):
        return _FakeCursor(self.state, self.log)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _install_fake_db(monkeypatch, row=None):
    state = {"row": row}
    log = []

    @contextmanager
    def _fake_get_conn():
        yield _FakeConn(state, log)

    monkeypatch.setattr(defence, "get_conn", _fake_get_conn)
    return state, log


def test_get_defence_response_reads_by_id(monkeypatch):
    row = dict(PENDING_ROW)
    _state, log = _install_fake_db(monkeypatch, row=row)

    result = defence.get_defence_response("resp-1")

    assert result == row
    assert log[0][0].startswith("SELECT * FROM defence_responses WHERE id")
    assert log[0][1] == ("resp-1",)


def test_get_defence_response_returns_none_when_missing(monkeypatch):
    _install_fake_db(monkeypatch, row=None)
    assert defence.get_defence_response("missing") is None


def test_update_defence_response_is_cas_guarded_on_pending(monkeypatch):
    row = dict(PENDING_ROW)
    _state, log = _install_fake_db(monkeypatch, row=row)

    ok = defence.update_defence_response(
        "resp-1", status="judged", judgment={"overall": {}}, judge_model="m"
    )

    assert ok is True
    sql, params = log[0]
    assert "WHERE id = %s AND status = 'pending'" in sql
    assert "judgment = %s" in sql
    assert "judge_model = %s" in sql
    assert "judged_at = now()" in sql
    assert params[-1] == "resp-1"


def test_update_defence_response_noop_when_already_resolved(monkeypatch):
    row = dict(PENDING_ROW)
    row["status"] = "judged"
    _install_fake_db(monkeypatch, row=row)

    ok = defence.update_defence_response("resp-1", status="failed")

    assert ok is False


def test_update_defence_response_failed_status_omits_judgment_write(monkeypatch):
    row = dict(PENDING_ROW)
    _state, log = _install_fake_db(monkeypatch, row=row)

    defence.update_defence_response("resp-1", status="failed")

    sql, _params = log[0]
    assert "judgment = %s" not in sql
    assert "judge_model = %s" not in sql
