"""Tests for detect_v7.modal_client — the fail-open HTTP client wrapping the
live Modal deep-scan detector endpoint. NEVER calls the real endpoint here;
all HTTP is mocked via monkeypatch on ``requests.post``.
"""
from __future__ import annotations

import pytest
import requests

from detect_v7 import modal_client

_URL_ENV_VAR = "DRAFTPROOF_MODAL_ENDPOINT_URL"
_TOKEN_ENV_VAR = "DRAFTPROOF_MODAL_ENDPOINT_TOKEN"
_MAX_WINDOWS_ENV_VAR = "DRAFTPROOF_DEEP_SCAN_MAX_WINDOWS"


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    # FIX 1's retry backoff is a real time.sleep; zero it so transient-failure
    # tests don't each pause the full backoff. Retry BEHAVIOR is asserted via
    # request call counts, not wall-clock, so this changes nothing under test.
    monkeypatch.setattr(modal_client, "_RETRY_BACKOFF_S", 0)


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


class _ScriptedPost:
    """requests.post stand-in returning/raising a scripted per-call sequence.

    The last item repeats for calls past the sequence end (so a 1-item timeout
    sequence models "times out on every attempt"). Records call count and the
    chunks POSTed on each call.
    """

    def __init__(self, sequence):
        self._sequence = list(sequence)
        self.calls = 0
        self.chunks_seen: list = []

    def __call__(self, url, json, headers, timeout):
        self.chunks_seen.append(json.get("chunks"))
        item = self._sequence[min(self.calls, len(self._sequence) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class TestRetryOnTransientFailures:
    """FIX 1: exactly one retry (2 attempts total) on transient failures only."""

    def test_timeout_then_success_retries_once(self, monkeypatch):
        _set_env(monkeypatch)
        payload = {"available": True, "chunk_scores": [0.9]}
        post = _ScriptedPost([requests.exceptions.Timeout("t"), _FakeResponse(200, payload)])
        monkeypatch.setattr(requests, "post", post)
        assert modal_client.call_deep_scan(["chunk"]) == payload
        assert post.calls == 2

    def test_connection_error_then_success_retries_once(self, monkeypatch):
        _set_env(monkeypatch)
        payload = {"available": True, "chunk_scores": [0.9]}
        post = _ScriptedPost([requests.exceptions.ConnectionError("c"), _FakeResponse(200, payload)])
        monkeypatch.setattr(requests, "post", post)
        assert modal_client.call_deep_scan(["chunk"]) == payload
        assert post.calls == 2

    def test_5xx_then_success_retries_once(self, monkeypatch):
        _set_env(monkeypatch)
        payload = {"available": True, "chunk_scores": [0.9]}
        post = _ScriptedPost([_FakeResponse(503), _FakeResponse(200, payload)])
        monkeypatch.setattr(requests, "post", post)
        assert modal_client.call_deep_scan(["chunk"]) == payload
        assert post.calls == 2

    def test_timeout_on_every_attempt_gives_up_after_max_attempts(self, monkeypatch):
        _set_env(monkeypatch)
        post = _ScriptedPost([requests.exceptions.Timeout("t")])
        monkeypatch.setattr(requests, "post", post)
        assert modal_client.call_deep_scan(["chunk"]) is None
        assert post.calls == modal_client._MAX_ATTEMPTS == 2


class TestNoRetryOnDeterministicFailures:
    """FIX 1: 4xx/auth and malformed JSON are deterministic — never retried."""

    def test_4xx_not_retried(self, monkeypatch):
        _set_env(monkeypatch)
        post = _ScriptedPost([_FakeResponse(401)])
        monkeypatch.setattr(requests, "post", post)
        assert modal_client.call_deep_scan(["chunk"]) is None
        assert post.calls == 1

    def test_404_not_retried(self, monkeypatch):
        _set_env(monkeypatch)
        post = _ScriptedPost([_FakeResponse(404)])
        monkeypatch.setattr(requests, "post", post)
        assert modal_client.call_deep_scan(["chunk"]) is None
        assert post.calls == 1

    def test_malformed_json_not_retried(self, monkeypatch):
        _set_env(monkeypatch)
        post = _ScriptedPost([_FakeResponse(200, raise_json_error=True)])
        monkeypatch.setattr(requests, "post", post)
        assert modal_client.call_deep_scan(["chunk"]) is None
        assert post.calls == 1

    def test_non_dict_json_not_retried(self, monkeypatch):
        _set_env(monkeypatch)
        post = _ScriptedPost([_FakeResponse(200, json_data=[1, 2, 3])])
        monkeypatch.setattr(requests, "post", post)
        assert modal_client.call_deep_scan(["chunk"]) is None
        assert post.calls == 1


class TestWindowCap:
    """FIX 3: env-configurable long-doc window cap. Deterministic first-N
    truncation + a ``capped`` provenance flag; typical docs never hit it."""

    def test_default_cap_high_enough_for_typical_docs(self):
        # A 50k-char doc yields at most ~1.6k windows; the default must clear it.
        assert modal_client._DEFAULT_MAX_WINDOWS >= 2000

    def test_over_cap_truncates_to_first_n_and_flags(self, monkeypatch):
        _set_env(monkeypatch)
        monkeypatch.setenv(_MAX_WINDOWS_ENV_VAR, "2")
        post = _ScriptedPost([_FakeResponse(200, {"available": True, "chunk_scores": [0.9, 0.9]})])
        monkeypatch.setattr(requests, "post", post)
        result = modal_client.call_deep_scan(["a", "b", "c", "d", "e"])
        assert post.chunks_seen[0] == ["a", "b"]  # deterministic first-N
        assert result["capped"] is True

    def test_under_cap_not_flagged(self, monkeypatch):
        _set_env(monkeypatch)
        monkeypatch.setenv(_MAX_WINDOWS_ENV_VAR, "10")
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: _FakeResponse(200, {"available": True, "chunk_scores": [0.9]})
        )
        result = modal_client.call_deep_scan(["a"])
        assert "capped" not in result

    def test_invalid_cap_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(_MAX_WINDOWS_ENV_VAR, "not-an-int")
        assert modal_client.resolve_max_windows() == modal_client._DEFAULT_MAX_WINDOWS

    def test_nonpositive_cap_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(_MAX_WINDOWS_ENV_VAR, "0")
        assert modal_client.resolve_max_windows() == modal_client._DEFAULT_MAX_WINDOWS

    def test_unset_cap_env_uses_default(self, monkeypatch):
        monkeypatch.delenv(_MAX_WINDOWS_ENV_VAR, raising=False)
        assert modal_client.resolve_max_windows() == modal_client._DEFAULT_MAX_WINDOWS
