"""SSE absolute-lifetime-cap test for /api/scans/{id}/events.

Mirrors the mocking convention in tests/test_defence_routes.py: TestClient +
dependency_overrides for auth, service/DB layer mocked -- no real DB/Redis.

Without the cap, a scan stuck in "processing" (Redis down, DB still returning
"processing") would hold the SSE connection open forever, relying only on
client disconnect or the stale-job reaper. This pins that the stream now closes
on its own once SSE_SCAN_STREAM_MAX_SECONDS elapses, without ever emitting a
fabricated "completed"/"failed" status.
"""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

TEST_USER = {"id": "11111111-1111-1111-1111-111111111111", "email": "u@x.com"}

PROCESSING_SCAN = {
    "id": "22222222-2222-2222-2222-222222222222",
    "status": "processing",
    "report_id": None,
    "progress_percent": 40,
    "progress_message": "Scanning",
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
    monkeypatch.setattr("app.routes.scans.SSE_SCAN_STREAM_MAX_SECONDS", 0)
    monkeypatch.setattr("app.routes.scans.get_scan", AsyncMock(return_value=PROCESSING_SCAN))
    monkeypatch.setattr(
        "app.routes.scans.progress_stream.read_scan_progress", AsyncMock(return_value=None)
    )

    with client.stream("GET", f"/api/scans/{PROCESSING_SCAN['id']}/events") as resp:
        assert resp.status_code == 200
        lines = list(resp.iter_lines())

    payload_lines = [l for l in lines if l.startswith("data: ")]
    assert payload_lines, "expected at least the initial progress event"
    for line in payload_lines:
        status = json.loads(line[len("data: "):]).get("status")
        assert status == "processing", (
            "the lifetime cap must close the connection as-is, never synthesize "
            "a completed/failed status the DB never reported"
        )
