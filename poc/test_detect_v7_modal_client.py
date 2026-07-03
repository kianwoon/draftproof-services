"""Tests for detect_v7.modal_client — the fail-open HTTP client wrapping the
live Modal deep-scan detector endpoint. NEVER calls the real endpoint here;
all HTTP is mocked via monkeypatch on ``requests.post``.
"""
from __future__ import annotations

import requests

from detect_v7 import modal_client

_URL_ENV_VAR = "DRAFTPROOF_MODAL_ENDPOINT_URL"
_TOKEN_ENV_VAR = "DRAFTPROOF_MODAL_ENDPOINT_TOKEN"


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, raise_json_error=False):
        self.status_code = status_code
        self._json_data = json_data
        self._raise_json_error = raise_json_error

    def json(self):
        if self._raise_json_error:
            raise ValueError("not json")
        return self._json_data


def _set_env(monkeypatch, url="https://example.modal.run", token="secret-token"):
    monkeypatch.setenv(_URL_ENV_VAR, url)
    monkeypatch.setenv(_TOKEN_ENV_VAR, token)


class TestEnvNotConfigured:
    def test_missing_url_returns_none_no_http_call(self, monkeypatch):
        monkeypatch.delenv(_URL_ENV_VAR, raising=False)
        monkeypatch.setenv(_TOKEN_ENV_VAR, "token")

        def _boom(*args, **kwargs):
            raise AssertionError("should not call requests.post when env unset")

        monkeypatch.setattr(requests, "post", _boom)
        assert modal_client.call_deep_scan(["chunk"]) is None

    def test_missing_token_returns_none(self, monkeypatch):
        monkeypatch.setenv(_URL_ENV_VAR, "https://example.modal.run")
        monkeypatch.delenv(_TOKEN_ENV_VAR, raising=False)
        assert modal_client.call_deep_scan(["chunk"]) is None

    def test_both_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv(_URL_ENV_VAR, raising=False)
        monkeypatch.delenv(_TOKEN_ENV_VAR, raising=False)
        assert modal_client.call_deep_scan(["chunk"]) is None


class TestEmptyChunks:
    def test_empty_chunks_returns_none(self, monkeypatch):
        _set_env(monkeypatch)
        assert modal_client.call_deep_scan([]) is None


class TestSuccessfulCall:
    def test_success_returns_parsed_payload(self, monkeypatch):
        _set_env(monkeypatch)
        payload = {
            "available": True,
            "calibrated": False,
            "chunk_scores": [0.99],
            "document_score": 0.965,
        }
        captured = {}

        def _fake_post(url, json, headers, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _FakeResponse(status_code=200, json_data=payload)

        monkeypatch.setattr(requests, "post", _fake_post)
        result = modal_client.call_deep_scan(["some text"], timeout_s=30.0)

        assert result == payload
        assert captured["url"] == "https://example.modal.run"
        assert captured["json"] == {"chunks": ["some text"]}
        assert captured["headers"]["Authorization"] == "Bearer secret-token"
        assert captured["timeout"] == 30.0


class TestFailureModes:
    def test_timeout_returns_none(self, monkeypatch):
        _set_env(monkeypatch)

        def _fake_post(*args, **kwargs):
            raise requests.exceptions.Timeout("timed out")

        monkeypatch.setattr(requests, "post", _fake_post)
        assert modal_client.call_deep_scan(["chunk"]) is None

    def test_connection_error_returns_none(self, monkeypatch):
        _set_env(monkeypatch)

        def _fake_post(*args, **kwargs):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", _fake_post)
        assert modal_client.call_deep_scan(["chunk"]) is None

    def test_non_200_returns_none(self, monkeypatch):
        _set_env(monkeypatch)
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=500))
        assert modal_client.call_deep_scan(["chunk"]) is None

    def test_malformed_json_returns_none(self, monkeypatch):
        _set_env(monkeypatch)
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: _FakeResponse(status_code=200, raise_json_error=True)
        )
        assert modal_client.call_deep_scan(["chunk"]) is None

    def test_non_dict_json_returns_none(self, monkeypatch):
        _set_env(monkeypatch)
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: _FakeResponse(status_code=200, json_data=[1, 2, 3])
        )
        assert modal_client.call_deep_scan(["chunk"]) is None

    def test_unexpected_exception_returns_none(self, monkeypatch):
        _set_env(monkeypatch)

        def _fake_post(*args, **kwargs):
            raise RuntimeError("something else broke")

        monkeypatch.setattr(requests, "post", _fake_post)
        assert modal_client.call_deep_scan(["chunk"]) is None

    def test_token_never_leaks_into_exception_message(self, monkeypatch, caplog):
        _set_env(monkeypatch, token="super-secret-token-xyz")

        def _fake_post(*args, **kwargs):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", _fake_post)
        with caplog.at_level("WARNING"):
            modal_client.call_deep_scan(["chunk"])
        assert "super-secret-token-xyz" not in caplog.text
