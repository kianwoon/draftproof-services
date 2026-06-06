"""Tests for payment routes: packs, checkout, balance, webhook."""

import json
import hashlib
import hmac
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import stripe as stripe_lib
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_jwt():
    """Return a valid-looking JWT cookie for test user."""
    def _make(user_id="test-user-123"):
        return user_id
    return _make


@pytest.fixture
def auth_cookie():
    """Patch _get_user_id to always return a test user."""
    with patch("app.routes.payments._get_user_id", return_value="test-user-123"):
        yield "token=fake-jwt"


@pytest.fixture
def client():
    """Create test client with DB dependency overridden."""
    from app.main import app
    from app.models.db import get_db

    # Mock async session
    mock_session = MagicMock()

    async def _override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = _override_get_db

    tc = TestClient(app, raise_server_exceptions=False)
    tc.mock_db = mock_session
    yield tc

    app.dependency_overrides.clear()


# ── GET /packs ────────────────────────────────────────────────────

class TestGetPacks:
    def test_returns_all_packs(self, client):
        resp = client.get("/api/payments/packs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4

        names = {p["id"] for p in data}
        assert names == {"single", "starter", "standard", "pro"}

    def test_pack_has_sgd_price(self, client):
        resp = client.get("/api/payments/packs")
        data = resp.json()
        expected_prices = {
            "single": 0.90,
            "starter": 4.50,
            "standard": 9.00,
            "pro": 20.00,
        }

        for pack in data:
            assert "price_sgd" in pack
            assert "tokens" in pack
            assert "name" in pack
            assert pack["price_sgd"] == expected_prices[pack["id"]]

    def test_single_pack_price(self, client):
        resp = client.get("/api/payments/packs")
        data = resp.json()
        single = next(p for p in data if p["id"] == "single")
        assert single["price_sgd"] == 0.90

    def test_pro_pack_price(self, client):
        resp = client.get("/api/payments/packs")
        data = resp.json()
        pro = next(p for p in data if p["id"] == "pro")
        assert pro["price_sgd"] == 20.00


# ── POST /checkout ───────────────────────────────────────────────

class TestCheckout:
    def test_requires_auth(self, client):
        resp = client.post("/api/payments/checkout", json={"pack_id": "single"})
        assert resp.status_code == 401

    def test_invalid_pack(self, client, auth_cookie):
        resp = client.post(
            "/api/payments/checkout",
            json={"pack_id": "mega_pack"},
            headers={"Cookie": auth_cookie},
        )
        assert resp.status_code in (400, 422)

    @patch("app.routes.payments.stripe.checkout.Session.create")
    def test_single_pack_stripe_params(self, mock_create, client, auth_cookie):
        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/test"
        mock_create.return_value = mock_session

        resp = client.post(
            "/api/payments/checkout",
            json={"pack_id": "single"},
            headers={"Cookie": auth_cookie},
        )

        assert resp.status_code == 200
        assert resp.json()["url"] == "https://checkout.stripe.com/test"

        # Verify Stripe was called with correct params
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["mode"] == "payment"
        assert call_kwargs["payment_method_types"] == ["card"]

        line_item = call_kwargs["line_items"][0]
        assert line_item["price_data"]["currency"] == "sgd"
        assert line_item["price_data"]["unit_amount"] == 90  # SGD $0.90 = 90 cents
        assert "Single Token" in line_item["price_data"]["product_data"]["name"]
        assert line_item["quantity"] == 1

        # No timeout param (was causing "unknown parameter" error)
        assert "timeout" not in call_kwargs

    @patch("app.routes.payments.stripe.checkout.Session.create")
    def test_pro_pack_stripe_cents(self, mock_create, client, auth_cookie):
        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/test"
        mock_create.return_value = mock_session

        resp = client.post(
            "/api/payments/checkout",
            json={"pack_id": "pro"},
            headers={"Cookie": auth_cookie},
        )

        assert resp.status_code == 200
        line_item = mock_create.call_args[1]["line_items"][0]
        assert line_item["price_data"]["unit_amount"] == 2000  # SGD $20.00 = 2000 cents

    @patch("app.routes.payments.stripe.checkout.Session.create")
    def test_metadata_contains_user_and_pack(self, mock_create, client, auth_cookie):
        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/test"
        mock_create.return_value = mock_session

        client.post(
            "/api/payments/checkout",
            json={"pack_id": "starter"},
            headers={"Cookie": auth_cookie},
        )

        metadata = mock_create.call_args[1]["metadata"]
        assert metadata["user_id"] == "test-user-123"
        assert metadata["pack_id"] == "starter"
        assert metadata["tokens"] == "5"

    @patch("app.routes.payments.stripe.checkout.Session.create")
    def test_success_cancel_urls(self, mock_create, client, auth_cookie):
        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/test"
        mock_create.return_value = mock_session

        client.post(
            "/api/payments/checkout",
            json={"pack_id": "single"},
            headers={"Cookie": auth_cookie},
        )

        call_kwargs = mock_create.call_args[1]
        assert "/buy?success=1" in call_kwargs["success_url"]
        assert "/buy?canceled=1" in call_kwargs["cancel_url"]

    @patch("app.routes.payments.stripe.checkout.Session.create")
    def test_stripe_api_error_returns_502(self, mock_create, client, auth_cookie):
        mock_create.side_effect = stripe_lib.error.APIError("Stripe is down")

        resp = client.post(
            "/api/payments/checkout",
            json={"pack_id": "single"},
            headers={"Cookie": auth_cookie},
        )

        assert resp.status_code == 502

    @patch("app.routes.payments.stripe.checkout.Session.create")
    def test_stripe_auth_error_returns_502(self, mock_create, client, auth_cookie):
        mock_create.side_effect = stripe_lib.error.AuthenticationError("Invalid API key")

        resp = client.post(
            "/api/payments/checkout",
            json={"pack_id": "single"},
            headers={"Cookie": auth_cookie},
        )

        assert resp.status_code == 502

    @patch("app.routes.payments.stripe.checkout.Session.create")
    def test_stripe_invalid_request_error(self, mock_create, client, auth_cookie):
        mock_create.side_effect = stripe_lib.error.InvalidRequestError(
            message="Received unknown parameter: timeout",
            param="timeout",
        )

        resp = client.post(
            "/api/payments/checkout",
            json={"pack_id": "single"},
            headers={"Cookie": auth_cookie},
        )

        assert resp.status_code == 502
        # Verify error was logged (not crash/500)

    @patch("app.routes.payments.stripe.checkout.Session.create")
    def test_all_packs_valid_cents(self, mock_create, client, auth_cookie):
        """Verify every pack produces integer cents with no floating point issues."""
        mock_session = MagicMock()
        mock_session.url = "https://checkout.stripe.com/test"
        mock_create.return_value = mock_session

        expected = {
            "single": 90,
            "starter": 450,
            "standard": 900,
            "pro": 2000,
        }

        for pack_id, expected_cents in expected.items():
            client.post(
                "/api/payments/checkout",
                json={"pack_id": pack_id},
                headers={"Cookie": auth_cookie},
            )
            line_item = mock_create.call_args[1]["line_items"][0]
            actual = line_item["price_data"]["unit_amount"]
            assert actual == expected_cents, f"{pack_id}: expected {expected_cents}, got {actual}"


# ── GET /balance ──────────────────────────────────────────────────

class TestGetBalance:
    def test_requires_auth(self, client):
        resp = client.get("/api/payments/balance")
        assert resp.status_code == 401


# ── Price Calculation ────────────────────────────────────────────

class TestPriceCalculation:
    """Unit tests for price math — no Stripe calls, no API."""

    def test_no_floating_point_rounding_errors(self):
        """Ensure price_cents is always an exact integer."""
        from app.config import TOKEN_PACKS, get_token_pack_price_sgd

        for pack_id, pack in TOKEN_PACKS.items():
            price_sgd = get_token_pack_price_sgd(pack)
            price_cents = int(price_sgd * 100)
            assert price_cents == price_sgd * 100, (
                f"{pack_id}: floating point mismatch — {price_sgd} * 100 != {price_cents}"
            )

    def test_minimum_amount_stripe_sgd(self):
        """Stripe minimum for SGD is S$0.50 (50 cents)."""
        from app.config import TOKEN_PRICE_SGD

        assert TOKEN_PRICE_SGD >= 0.50, f"Price SGD ${TOKEN_PRICE_SGD} below Stripe minimum"
