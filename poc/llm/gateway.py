"""LLM Gateway — provider-agnostic chat completion client.

Supports any OpenAI-compatible API (OpenRouter, OpenAI, Together, etc.)
via configurable base_url. Uses smart error classification for retries.
"""

from __future__ import annotations

import os
import time
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class _RetryAction(Enum):
    RETRY = "retry"
    FAIL = "fail"


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None       # None → resolved from LLM_BASE_URL env var (fallback: OpenRouter)
    model: Optional[str] = None          # None → resolved from LLM_MODEL env var at runtime
    max_tokens: int = 4096
    temperature: float = 0.3
    max_retries: int = 3
    timeout: int = 120
    site_url: Optional[str] = None       # OpenRouter optional headers
    site_name: Optional[str] = None


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def is_empty(self) -> bool:
        return not self.content or not self.content.strip()


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

# Retryable HTTP status codes (transient / rate-limit)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Permanent error status codes (auth, bad request, not found)
_PERMANENT_STATUS_CODES = {400, 401, 403, 404, 405, 422}

# Retryable error substrings in exception messages
_RETRYABLE_PATTERNS = [
    "rate_limit",
    "rate limit",
    "overloaded",
    "capacity",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
]


def _classify_error(error: Exception, attempt: int, max_retries: int) -> _RetryAction:
    """Decide whether to retry or fail based on error type and attempt count."""
    if attempt >= max_retries:
        return _RetryAction.FAIL

    status = None
    if isinstance(error, requests.HTTPError) and error.response is not None:
        status = error.response.status_code

    if status in _PERMANENT_STATUS_CODES:
        return _RetryAction.FAIL

    if status in _RETRYABLE_STATUS_CODES:
        return _RetryAction.RETRY

    msg = str(error).lower()
    if any(p in msg for p in _RETRYABLE_PATTERNS):
        return _RetryAction.RETRY

    # Unknown errors: retry up to max, then fail
    return _RetryAction.RETRY


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class LLMGateway:
    """Provider-agnostic LLM chat completion gateway.

    Usage:
        gw = LLMGateway()  # reads from env vars
        resp = gw.chat("Explain quantum entanglement in one paragraph.")
        print(resp.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        cfg = config or LLMConfig()

        # Resolve from env if not explicitly set
        self.api_key = cfg.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
        raw_base = cfg.base_url or os.environ.get("LLM_BASE_URL") or "https://openrouter.ai/api/v1"
        self.base_url = raw_base.rstrip("/")
        logger.info(f"LLM Gateway initialized: base_url={self.base_url}, model={self.model}")
        self.model = cfg.model or os.environ.get("LLM_MODEL") or "google/gemma-3-12b-it"
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.max_retries = cfg.max_retries
        self.timeout = cfg.timeout
        self.site_url = cfg.site_url or os.environ.get("LLM_SITE_URL")
        self.site_name = cfg.site_name or os.environ.get("LLM_SITE_NAME")

        if not self.api_key:
            raise ValueError(
                "No API key found. Set OPENROUTER_API_KEY env var or pass config.api_key."
            )

    # --- Public API ---

    def chat(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> LLMResponse:
        """Send a single-turn chat completion request."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._complete(messages, temperature=temperature, max_tokens=max_tokens)

    def chat_multi(self, messages: list[dict], *, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> LLMResponse:
        """Send a multi-turn chat completion request with full message history."""
        return self._complete(messages, temperature=temperature, max_tokens=max_tokens)

    # --- Internal ---

    def _build_headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-OpenRouter-Title"] = self.site_name
        return headers

    def _complete(self, messages: list[dict], *, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        headers = self._build_headers()

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                content = self._extract_content(data)
                usage = data.get("usage", {})
                model_used = data.get("model", self.model)

                return LLMResponse(content=content, model=model_used, usage=usage, raw=data)

            except Exception as exc:
                action = _classify_error(exc, attempt, self.max_retries)

                if action == _RetryAction.FAIL:
                    logger.error("LLM call failed permanently on attempt %d: %s", attempt, exc)
                    raise

                backoff = min(2 ** attempt, 30)  # cap at 30s
                logger.warning("LLM call failed (attempt %d/%d), retrying in %ds: %s", attempt, self.max_retries, backoff, exc)
                time.sleep(backoff)

        # Should not reach here, but safety net
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts")

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract text content from OpenAI-compatible response JSON."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning("Unexpected response structure: %s", json.dumps(data)[:500])
            return ""
