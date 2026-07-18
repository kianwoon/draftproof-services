"""HTTP tests for the defence-readiness routes (/api/scans/{scan_id}/defence...), Task 6.

Auth + service/DB layer are mocked (TestClient + dependency_overrides + monkeypatch/patch),
matching the convention in tests/test_ext_routes.py — no real DB/Redis required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

TEST_USER = {"id": "11111111-1111-1111-1111-111111111111", "email": "u@x.com"}

SCAN = {"id": "scan-1", "status": "completed"}

QUESTIONS = [
    {"dimension": "generic_assertion_risk", "anchor_quote": "quote one", "question": "Q1?"},
    {"dimension": "citation_grounding_risk", "anchor_quote": "quote two", "question": "Q2?"},
]

REPORT_JSON = {
    "ai_risk_badge": {
        "critical_thinking_control": {"questions": QUESTIONS},
    },
}


@pytest.fixture
def client():
    from app.main import app
    from app.routes.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: TEST_USER
    tc = TestClient(app, raise_server_exceptions=False)
    yield tc
    app.dependency_overrides.clear()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr("app.routes.defence.DRAFTPROOF_DEFENCE_CHECK", True)


def _mock_report(monkeypatch, report=REPORT_JSON):
    monkeypatch.setattr(
        "app.routes.defence._fetch_optional_report_json_sync",
        MagicMock(return_value=report),
    )


# ── Kill switch ──────────────────────────────────────────────────────────────

def test_post_404_when_flag_off(client, monkeypatch):
    monkeypatch.setattr("app.routes.defence.DRAFTPROOF_DEFENCE_CHECK", False)
    r = client.post("/api/scans/scan-1/defence/answers", json={"question_index": 0, "answer_text": "hi"})
    assert r.status_code == 404
    assert r.json()["detail"] == "feature disabled"


def test_get_404_when_flag_off(client, monkeypatch):
    monkeypatch.setattr("app.routes.defence.DRAFTPROOF_DEFENCE_CHECK", False)
    r = client.get("/api/scans/scan-1/defence")
    assert r.status_code == 404
    assert r.json()["detail"] == "feature disabled"


def test_flag_off_gates_before_auth(client, monkeypatch):
    """Even an unauthenticated caller must get 404, not 401 — the feature is fully
    invisible while off, not merely 'requires login'."""
    from app.main import app
    from app.routes.auth import get_current_user

    app.dependency_overrides.pop(get_current_user, None)  # real auth dependency runs
    monkeypatch.setattr("app.routes.defence.DRAFTPROOF_DEFENCE_CHECK", False)
    r = client.get("/api/scans/scan-1/defence")
    assert r.status_code == 404
    assert r.json()["detail"] == "feature disabled"


def test_flag_on_existing_routes_unaffected(client, monkeypatch):
    """Sanity check for the acceptance criterion: toggling the flag must not change any
    existing route's behavior. GET /api/documents/{id} always 404s synchronously (no DB
    touch), so it's a clean existing-route probe unaffected by test-env DB reachability."""
    monkeypatch.setattr("app.routes.defence.DRAFTPROOF_DEFENCE_CHECK", False)
    r_off = client.get("/api/documents/some-id")
    monkeypatch.setattr("app.routes.defence.DRAFTPROOF_DEFENCE_CHECK", True)
    r_on = client.get("/api/documents/some-id")
    assert r_off.status_code == r_on.status_code == 404
    assert r_off.json() == r_on.json() == {"detail": "Document not found"}


# ── Ownership ────────────────────────────────────────────────────────────────

def test_post_404_for_other_users_scan(client, enabled, monkeypatch):
    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=None))
    r = client.post("/api/scans/scan-1/defence/answers", json={"question_index": 0, "answer_text": "hi"})
    assert r.status_code == 404
    assert r.json()["detail"] == "Scan not found"


def test_get_404_for_other_users_scan(client, enabled, monkeypatch):
    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=None))
    r = client.get("/api/scans/scan-1/defence")
    assert r.status_code == 404
    assert r.json()["detail"] == "Scan not found"


# ── question_index validation ────────────────────────────────────────────────

def test_invalid_question_index_returns_400(client, enabled, monkeypatch):
    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=SCAN))
    _mock_report(monkeypatch)
    r = client.post("/api/scans/scan-1/defence/answers", json={"question_index": 5, "answer_text": "hi"})
    assert r.status_code == 400


def test_negative_question_index_returns_400(client, enabled, monkeypatch):
    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=SCAN))
    _mock_report(monkeypatch)
    r = client.post("/api/scans/scan-1/defence/answers", json={"question_index": -1, "answer_text": "hi"})
    assert r.status_code == 400


def test_question_index_invalid_when_report_missing(client, enabled, monkeypatch):
    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=SCAN))
    _mock_report(monkeypatch, report=None)
    r = client.post("/api/scans/scan-1/defence/answers", json={"question_index": 0, "answer_text": "hi"})
    assert r.status_code == 400


# ── answer_text length ───────────────────────────────────────────────────────

def test_answer_text_too_long_returns_400(client, enabled, monkeypatch):
    from app.config import DRAFTPROOF_DEFENCE_MAX_ANSWER_CHARS

    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=SCAN))
    _mock_report(monkeypatch)
    too_long = "a" * (DRAFTPROOF_DEFENCE_MAX_ANSWER_CHARS + 1)
    r = client.post("/api/scans/scan-1/defence/answers", json={"question_index": 0, "answer_text": too_long})
    assert r.status_code == 400


# ── attempt cap ──────────────────────────────────────────────────────────────

def test_attempt_cap_reached_returns_409(client, enabled, monkeypatch):
    from app.config import DRAFTPROOF_DEFENCE_MAX_ATTEMPTS

    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=SCAN))
    _mock_report(monkeypatch)
    monkeypatch.setattr(
        "app.routes.defence.defence_service.count_attempts",
        AsyncMock(return_value=DRAFTPROOF_DEFENCE_MAX_ATTEMPTS),
    )
    r = client.post("/api/scans/scan-1/defence/answers", json={"question_index": 0, "answer_text": "hi"})
    assert r.status_code == 409


def test_attempt_cap_counts_per_question_index_not_per_scan(client, enabled, monkeypatch):
    """A maxed-out attempt count on question_index=0 must not block question_index=1."""
    from app.config import DRAFTPROOF_DEFENCE_MAX_ATTEMPTS

    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=SCAN))
    _mock_report(monkeypatch)

    async def fake_count(scan_id, question_index):
        return DRAFTPROOF_DEFENCE_MAX_ATTEMPTS if question_index == 0 else 0

    monkeypatch.setattr("app.routes.defence.defence_service.count_attempts", fake_count)
    monkeypatch.setattr(
        "app.routes.defence.defence_service.create_response",
        AsyncMock(return_value={"id": "resp-2", "status": "pending"}),
    )
    with patch("app.services.celery_client.judge_defence_answer") as mock_task:
        r = client.post("/api/scans/scan-1/defence/answers", json={"question_index": 1, "answer_text": "hi"})
    assert r.status_code == 202
    mock_task.delay.assert_called_once_with("resp-2")


# ── happy path ───────────────────────────────────────────────────────────────

def test_submit_answer_happy_path(client, enabled, monkeypatch):
    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=SCAN))
    _mock_report(monkeypatch)
    monkeypatch.setattr("app.routes.defence.defence_service.count_attempts", AsyncMock(return_value=0))
    create_mock = AsyncMock(return_value={"id": "resp-1", "status": "pending"})
    monkeypatch.setattr("app.routes.defence.defence_service.create_response", create_mock)

    with patch("app.services.celery_client.judge_defence_answer") as mock_task:
        r = client.post(
            "/api/scans/scan-1/defence/answers",
            json={"question_index": 0, "answer_text": "My answer"},
        )

    assert r.status_code == 202
    assert r.json() == {"response_id": "resp-1", "status": "pending"}
    # Enqueued by NAME (app.services.celery_client.judge_defence_answer), never imported
    # from worker/ directly — Task 7 does not exist in this codebase yet.
    mock_task.delay.assert_called_once_with("resp-1")
    _, kwargs = create_mock.call_args
    assert kwargs["question_index"] == 0
    assert kwargs["dimension"] == "generic_assertion_risk"
    assert kwargs["question"] == "Q1?"
    assert kwargs["anchor_quote"] == "quote one"
    assert kwargs["answer_text"] == "My answer"
    assert kwargs["attempt"] == 1


# ── GET aggregate readiness ──────────────────────────────────────────────────

def test_get_defence_returns_rows_and_best_of_attempts_readiness(client, enabled, monkeypatch):
    monkeypatch.setattr("app.routes.defence.get_scan", AsyncMock(return_value=SCAN))
    rows = [
        {
            "id": "r1", "question_index": 0, "dimension": "generic_assertion_risk",
            "question": "Q1?", "anchor_quote": "q", "answer_text": "a1", "attempt": 1,
            "status": "judged", "judgment": {"overall": {"level": "low", "score": 20}},
            "judge_model": "m", "created_at": None, "judged_at": None,
        },
        {
            "id": "r2", "question_index": 0, "dimension": "generic_assertion_risk",
            "question": "Q1?", "anchor_quote": "q", "answer_text": "a2", "attempt": 2,
            "status": "judged", "judgment": {"overall": {"level": "high", "score": 85}},
            "judge_model": "m", "created_at": None, "judged_at": None,
        },
        {
            "id": "r3", "question_index": 1, "dimension": "citation_grounding_risk",
            "question": "Q2?", "anchor_quote": "q2", "answer_text": "a3", "attempt": 1,
            "status": "pending", "judgment": None, "judge_model": None,
            "created_at": None, "judged_at": None,
        },
    ]
    monkeypatch.setattr("app.routes.defence.defence_service.list_responses", AsyncMock(return_value=rows))

    r = client.get("/api/scans/scan-1/defence")
    assert r.status_code == 200
    body = r.json()
    assert len(body["responses"]) == 3
    # Best-of-attempts: attempt 2 (high/85) beats attempt 1 (low/20) for the same dimension.
    assert body["readiness"]["generic_assertion_risk"] == {
        "level": "high", "score": 85, "question_index": 0, "attempt": 2,
    }
    # citation_grounding_risk has no judged attempt yet -> absent from readiness.
    assert "citation_grounding_risk" not in body["readiness"]
