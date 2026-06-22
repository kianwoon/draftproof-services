"""HTTP tests for the key-management (/api/keys) and extension (/api/ext) routers.

Auth dependencies are overridden and the service layer is mocked — these assert
the wiring, status codes, and the low_confidence / 402 billing contract.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

TEST_USER = {"id": "11111111-1111-1111-1111-111111111111", "email": "u@x.com"}


@pytest.fixture
def client():
    from app.main import app
    from app.routes.auth import get_current_user
    from app.services.api_key_service import get_api_key_user

    app.dependency_overrides[get_current_user] = lambda: TEST_USER
    app.dependency_overrides[get_api_key_user] = lambda: TEST_USER
    tc = TestClient(app, raise_server_exceptions=False)
    yield tc
    app.dependency_overrides.clear()


# ── /api/keys (web cookie/JWT auth) ───────────────────────────────────────────

def test_create_key_returns_plaintext_once(client):
    minted = {
        "id": "k1", "name": "My Word add-in", "key": "dp_live_supersecret",
        "key_prefix": "dp_live_superse", "created_at": "2026-01-01T00:00:00+00:00",
    }
    with patch("app.routes.keys.api_key_service.create_key",
               new=AsyncMock(return_value=minted)):
        r = client.post("/api/keys/", json={"name": "My Word add-in"})
    assert r.status_code in (200, 201)
    assert r.json()["key"] == "dp_live_supersecret"


def test_list_keys_returns_prefixes(client):
    rows = [{"id": "k1", "name": "k", "key_prefix": "dp_live_aaaa1111",
             "last_used_at": None, "revoked_at": None, "created_at": None}]
    with patch("app.routes.keys.api_key_service.list_keys",
               new=AsyncMock(return_value=rows)):
        r = client.get("/api/keys/")
    assert r.status_code == 200
    assert r.json()[0]["key_prefix"] == "dp_live_aaaa1111"
    assert "key_hash" not in r.json()[0]


def test_revoke_key_found(client):
    with patch("app.routes.keys.api_key_service.revoke_key",
               new=AsyncMock(return_value=True)):
        r = client.delete("/api/keys/k1")
    assert r.status_code == 200


def test_revoke_key_missing_404(client):
    with patch("app.routes.keys.api_key_service.revoke_key",
               new=AsyncMock(return_value=False)):
        r = client.delete("/api/keys/k1")
    assert r.status_code == 404


def test_create_key_cap_reached_returns_409(client):
    with patch("app.routes.keys.api_key_service.create_key",
               new=AsyncMock(side_effect=ValueError("API key limit reached (max 20)."))):
        r = client.post("/api/keys/", json={"name": "too many"})
    assert r.status_code == 409


# ── /api/ext (API-key auth) ───────────────────────────────────────────────────

def test_ext_scan_flags_low_confidence_for_short_text(client):
    with patch("app.routes.ext.create_scan",
               new=AsyncMock(return_value={"id": "scan-1", "status": "pending"})):
        r = client.post("/api/ext/scan", json={"text": " ".join(["w"] * 40)})
    assert r.status_code == 200
    body = r.json()
    assert body["scan_id"] == "scan-1"
    assert body["word_count"] == 40
    assert body["low_confidence"] is True


def test_ext_scan_not_low_confidence_when_long(client):
    with patch("app.routes.ext.create_scan",
               new=AsyncMock(return_value={"id": "scan-2", "status": "pending"})):
        r = client.post("/api/ext/scan", json={"text": " ".join(["w"] * 150)})
    assert r.status_code == 200
    assert r.json()["low_confidence"] is False


def test_ext_scan_insufficient_credits_returns_402(client):
    with patch("app.routes.ext.create_scan",
               new=AsyncMock(side_effect=ValueError("Insufficient tokens — please purchase more"))):
        r = client.post("/api/ext/scan", json={"text": " ".join(["w"] * 150)})
    assert r.status_code == 402


def test_ext_scan_empty_text_returns_400(client):
    r = client.post("/api/ext/scan", json={"text": "   "})
    assert r.status_code == 400


def test_ext_scan_rejects_oversized_body(client):
    # Over the 50k-char cap → Pydantic validation → 400 (app's validation handler).
    r = client.post("/api/ext/scan", json={"text": "w " * 30000})  # 60k chars
    assert r.status_code == 400


def test_ext_scan_status_passthrough(client):
    done = {"id": "scan-1", "status": "completed", "tier": "clean", "ai_score": 12.0}
    with patch("app.routes.ext.get_scan", new=AsyncMock(return_value=done)):
        r = client.get("/api/ext/scan/scan-1")
    assert r.status_code == 200
    assert r.json()["tier"] == "clean"


def test_ext_scan_status_404(client):
    with patch("app.routes.ext.get_scan", new=AsyncMock(return_value=None)):
        r = client.get("/api/ext/scan/missing")
    assert r.status_code == 404
