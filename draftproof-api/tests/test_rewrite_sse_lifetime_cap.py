"""SSE absolute-lifetime-cap test for /api/rewrites/{id}/events.

Mirrors tests/test_scan_sse_lifetime_cap.py: TestClient + dependency_overrides
for auth, service/DB layer mocked -- no real DB/Redis.

Without the cap, a rewrite stuck in "processing" (Redis down, DB still
returning "processing") would hold the SSE connection open forever, relying
only on client disconnect or the stale-job reaper. This pins that the stream
now closes on its own once SSE_REWRITE_STREAM_MAX_SECONDS elapses, without
ever emitting a fabricated terminal status.
"""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

TEST_USER = {"id": "11111111-1111-1111-1111-111111111111", "email": "u@x.com"}

PROCESSING_REWRITE = {
    "id": "33333333-3333-3333-3333-333333333333",
    "scan_id": "22222222-2222-2222-2222-222222222222",
    "status": "processing",
    "error": None,
    "progress_percent": 40,
    "progress_message": "Rewriting AI sections",
    "created_at": "2026-07-20T00:00:00+00:00",
    "completed_at": None,
}


@pytest.fixture
def client():
    from app.main import app
    from app.routes.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: TEST_USER
    tc = TestClient(app, raise_server_exceptions=False)
    yield tc
    app.dependency_overrides.clear()


def test_sse_stream_closes_at_lifetime_cap_without_fake_terminal_status(client, monkeypatch):
    monkeypatch.setattr("app.routes.rewrites.SSE_REWRITE_STREAM_MAX_SECONDS", 0)
    monkeypatch.setattr(
        "app.routes.rewrites.rewrite_service.get_rewrite", AsyncMock(return_value=PROCESSING_REWRITE)
    )
    monkeypatch.setattr(
        "app.routes.rewrites.progress_stream.read_rewrite_progress", AsyncMock(return_value=None)
    )

    with client.stream("GET", f"/api/rewrites/{PROCESSING_REWRITE['id']}/events") as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())

    payload_lines = [l for l in lines if l.startswith("data: ")]
    assert payload_lines, "expected at least the initial progress event"
    for line in payload_lines:
        status = json.loads(line[len("data: "):]).get("status")
        assert status == "processing", (
            "the lifetime cap must close the connection as-is, never synthesize "
            "a terminal status the DB never reported"
        )


async def test_get_rewrite_service_rejects_non_uuid_without_raising():
    """A malformed rewrite_id must resolve to None (-> 404 at the route), not an
    uncaught uuid.UUID() ValueError (-> unhandled 500). Runs the real service
    function; the UUID parse fails before any DB query, so no DB is needed."""
    from app.services import rewrite_service

    assert await rewrite_service.get_rewrite("not-a-uuid") is None
    assert await rewrite_service.get_rewrite("not-a-uuid", user_id="11111111-1111-1111-1111-111111111111") is None


async def test_cancel_rewrite_service_rejects_non_uuid_without_raising():
    from app.services import rewrite_service

    assert await rewrite_service.cancel_rewrite("not-a-uuid", "11111111-1111-1111-1111-111111111111") is None


def test_get_rewrite_route_returns_404_for_non_uuid_id(client, monkeypatch):
    monkeypatch.setattr(
        "app.routes.rewrites.rewrite_service.get_rewrite", AsyncMock(return_value=None)
    )
    resp = client.get("/api/rewrites/not-a-uuid")
    assert resp.status_code == 404
