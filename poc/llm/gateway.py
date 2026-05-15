"""LLM Gateway — provider-agnostic chat completion client.

Supports any OpenAI-compatible API (OpenRouter, OpenAI, Together, etc.)
via configurable base_url. Uses smart error classification for retries.
"""

from __future__ import annotations

import os
import time
import json
import logging
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

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
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    response_format: Optional[dict[str, Any]] = None
    provider: Optional[dict[str, Any]] = None
    extra_body: Optional[dict[str, Any]] = None
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

    @property
    def finish_reason(self) -> str | None:
        choices = self.raw.get("choices") if isinstance(self.raw, dict) else None
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0] if isinstance(choices[0], dict) else {}
        value = first.get("finish_reason")
        return str(value) if value is not None else None

    @property
    def native_finish_reason(self) -> str | None:
        choices = self.raw.get("choices") if isinstance(self.raw, dict) else None
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0] if isinstance(choices[0], dict) else {}
        value = first.get("native_finish_reason")
        return str(value) if value is not None else None


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


_MODEL_CAPABILITIES = {
    "openai/gpt-4.1-mini": {
        "top_k": False,
        "presence_penalty": True,
        "frequency_penalty": True,
        "repetition_penalty": False,
        "structured_outputs": True,
    },
    "openai/gpt-4o-mini": {
        "top_k": False,
        "presence_penalty": True,
        "frequency_penalty": True,
        "repetition_penalty": False,
        "structured_outputs": True,
    },
    "openai/gpt-5-mini": {
        "top_k": False,
        "presence_penalty": True,
        "frequency_penalty": True,
        "repetition_penalty": False,
        "structured_outputs": True,
    },
    "openai/gpt-5.4-nano": {
        "top_k": False,
        "presence_penalty": True,
        "frequency_penalty": True,
        "repetition_penalty": False,
        "structured_outputs": True,
    },
    "deepseek/deepseek-chat": {
        "top_k": True,
        "presence_penalty": True,
        "frequency_penalty": True,
        "repetition_penalty": True,
        "structured_outputs": True,
    },
    "deepseek/deepseek-v4-flash": {
        "top_k": False,
        "presence_penalty": False,
        "frequency_penalty": False,
        "repetition_penalty": False,
        "structured_outputs": True,
    },
    "deepseek/deepseek-v4-flash-20260423": {
        "top_k": False,
        "presence_penalty": False,
        "frequency_penalty": False,
        "repetition_penalty": False,
        "structured_outputs": True,
    },
}


def _model_capabilities(model: str) -> dict:
    normalized = str(model or "").strip().lower()
    if normalized in _MODEL_CAPABILITIES:
        return dict(_MODEL_CAPABILITIES[normalized])
    if normalized.startswith("openai/") or normalized.startswith("gpt-"):
        return {
            "top_k": False,
            "presence_penalty": True,
            "frequency_penalty": True,
            "repetition_penalty": False,
            "structured_outputs": True,
        }
    repetition_supported = any(
        provider in normalized
        for provider in ("deepseek", "qwen", "mistral", "llama", "anthropic")
    )
    return {
        "top_k": True,
        "presence_penalty": True,
        "frequency_penalty": True,
        "repetition_penalty": repetition_supported,
        "structured_outputs": normalized.startswith(("deepseek/", "qwen/", "mistral/", "meta-llama/", "anthropic/")),
    }


def model_supports_presence_frequency_penalties(model: str | None) -> bool:
    caps = _model_capabilities(str(model or ""))
    return bool(caps.get("presence_penalty") and caps.get("frequency_penalty"))


def model_supports_repetition_penalty(model: str | None) -> bool:
    return bool(_model_capabilities(str(model or "")).get("repetition_penalty"))


def model_supports_structured_outputs(model: str | None) -> bool:
    return bool(_model_capabilities(str(model or "")).get("structured_outputs"))


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _list_env(*names: str) -> list[str]:
    value = _first_env(*names)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool_env(*names: str) -> bool | None:
    value = _first_env(*names)
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _provider_from_env() -> dict[str, Any] | None:
    raw_json = _first_env(
        "DRAFTPROOF_OPENROUTER_PROVIDER_ROUTING_JSON",
        "OPENROUTER_PROVIDER_ROUTING_JSON",
        "LLM_PROVIDER_ROUTING_JSON",
    )
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError("OpenRouter provider routing JSON must be an object")
        return parsed

    provider: dict[str, Any] = {}
    order = _list_env("DRAFTPROOF_OPENROUTER_PROVIDER_ORDER", "OPENROUTER_PROVIDER_ORDER")
    only = _list_env("DRAFTPROOF_OPENROUTER_PROVIDER_ONLY", "OPENROUTER_PROVIDER_ONLY")
    ignore = _list_env("DRAFTPROOF_OPENROUTER_PROVIDER_IGNORE", "OPENROUTER_PROVIDER_IGNORE")
    if order:
        provider["order"] = order
    if only:
        provider["only"] = only
    if ignore:
        provider["ignore"] = ignore

    allow_fallbacks = _bool_env("DRAFTPROOF_OPENROUTER_ALLOW_FALLBACKS", "OPENROUTER_ALLOW_FALLBACKS")
    require_parameters = _bool_env("DRAFTPROOF_OPENROUTER_REQUIRE_PARAMETERS", "OPENROUTER_REQUIRE_PARAMETERS")
    zdr = _bool_env("DRAFTPROOF_OPENROUTER_ZDR", "OPENROUTER_ZDR")
    enforce_distillable_text = _bool_env(
        "DRAFTPROOF_OPENROUTER_ENFORCE_DISTILLABLE_TEXT",
        "OPENROUTER_ENFORCE_DISTILLABLE_TEXT",
    )
    if allow_fallbacks is not None:
        provider["allow_fallbacks"] = allow_fallbacks
    if require_parameters is not None:
        provider["require_parameters"] = require_parameters
    if zdr is not None:
        provider["zdr"] = zdr
    if enforce_distillable_text is not None:
        provider["enforce_distillable_text"] = enforce_distillable_text

    sort = _first_env("DRAFTPROOF_OPENROUTER_PROVIDER_SORT", "OPENROUTER_PROVIDER_SORT")
    data_collection = _first_env("DRAFTPROOF_OPENROUTER_DATA_COLLECTION", "OPENROUTER_DATA_COLLECTION")
    if sort:
        provider["sort"] = sort
    if data_collection:
        provider["data_collection"] = data_collection
    return provider or None


def _extra_body_from_env() -> dict[str, Any] | None:
    raw_json = _first_env(
        "DRAFTPROOF_LLM_EXTRA_BODY_JSON",
        "OPENROUTER_EXTRA_BODY_JSON",
        "LLM_EXTRA_BODY_JSON",
    )
    extra: dict[str, Any] = {}
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError("LLM extra body JSON must be an object")
        extra.update(parsed)

    reasoning_effort = _first_env("DRAFTPROOF_OPENROUTER_REASONING_EFFORT", "OPENROUTER_REASONING_EFFORT")
    if reasoning_effort:
        current_reasoning = extra.get("reasoning") if isinstance(extra.get("reasoning"), dict) else {}
        extra["reasoning"] = {**current_reasoning, "effort": reasoning_effort}
    return extra or None


def _safe_extra_body(extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
    if not extra_body:
        return None
    reserved = {"model", "messages", "max_tokens"}
    safe: dict[str, Any] = {}
    for key, value in extra_body.items():
        if key in reserved:
            logger.warning("Ignoring reserved LLM extra body key: %s", key)
            continue
        safe[key] = value
    return safe or None


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
        self.model = cfg.model or os.environ.get("LLM_MODEL") or "google/gemma-3-12b-it"
        logger.info(f"LLM Gateway initialized: base_url={self.base_url}, model={self.model}")
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k
        self.presence_penalty = cfg.presence_penalty
        self.frequency_penalty = cfg.frequency_penalty
        self.repetition_penalty = cfg.repetition_penalty
        self.seed = cfg.seed
        self.response_format = cfg.response_format
        self.provider = cfg.provider if cfg.provider is not None else _provider_from_env()
        self.extra_body = _safe_extra_body(cfg.extra_body if cfg.extra_body is not None else _extra_body_from_env())
        self.max_retries = cfg.max_retries
        self.timeout = cfg.timeout
        self.site_url = cfg.site_url or os.environ.get("LLM_SITE_URL")
        self.site_name = cfg.site_name or os.environ.get("LLM_SITE_NAME")

        if not self.api_key:
            raise ValueError(
                "No API key found. Set OPENROUTER_API_KEY env var or pass config.api_key."
            )

    # --- Public API ---

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        provider: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """Send a single-turn chat completion request."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
            response_format=response_format,
            provider=provider,
        )

    def chat_multi(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        provider: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """Send a multi-turn chat completion request with full message history."""
        return self._complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
            response_format=response_format,
            provider=provider,
        )

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

    def _complete(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        provider: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        effective_top_p = top_p if top_p is not None else self.top_p
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_presence_penalty = (
            presence_penalty if presence_penalty is not None else self.presence_penalty
        )
        effective_frequency_penalty = (
            frequency_penalty if frequency_penalty is not None else self.frequency_penalty
        )
        effective_repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        )
        effective_seed = seed if seed is not None else self.seed
        effective_response_format = (
            response_format if response_format is not None else self.response_format
        )
        effective_provider = provider if provider is not None else self.provider
        requested_sampling = {
            "temperature": payload["temperature"],
            "top_p": effective_top_p,
            "top_k": effective_top_k,
            "presence_penalty": effective_presence_penalty,
            "frequency_penalty": effective_frequency_penalty,
            "repetition_penalty": effective_repetition_penalty,
            "seed": effective_seed,
        }
        caps = _model_capabilities(self.model)
        if effective_top_p is not None:
            payload["top_p"] = effective_top_p
        if effective_top_k is not None and caps.get("top_k", True):
            payload["top_k"] = effective_top_k
        if effective_presence_penalty is not None and caps.get("presence_penalty", True):
            payload["presence_penalty"] = effective_presence_penalty
        if effective_frequency_penalty is not None and caps.get("frequency_penalty", True):
            payload["frequency_penalty"] = effective_frequency_penalty
        if effective_repetition_penalty is not None and caps.get("repetition_penalty", True):
            payload["repetition_penalty"] = effective_repetition_penalty
        if effective_seed is not None:
            payload["seed"] = effective_seed
        if effective_response_format is not None:
            payload["response_format"] = effective_response_format
        if effective_provider is not None:
            payload["provider"] = effective_provider
        if self.extra_body is not None:
            payload.update(self.extra_body)
        effective_sampling = {
            key: payload[key]
            for key in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "repetition_penalty", "seed")
            if key in payload
        }

        headers = self._build_headers()
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "LLM request: url=%s, model=%s, messages=%d, prompt_chars=%d, max_tokens=%s, requested_sampling=%s, effective_sampling=%s, provider=%s, extra_body_keys=%s",
            url,
            self.model,
            len(messages),
            prompt_chars,
            payload["max_tokens"],
            json.dumps(requested_sampling, sort_keys=True),
            json.dumps(effective_sampling, sort_keys=True),
            json.dumps(effective_provider, sort_keys=True) if effective_provider is not None else None,
            sorted(self.extra_body.keys()) if self.extra_body is not None else [],
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                wall_s = time.monotonic() - t0
                logger.info(f"LLM response: status={resp.status_code}, latency={wall_s:.1f}s, attempt={attempt}")
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
    def _normalize_quotes(text: str) -> str:
        """Replace Unicode curly quotes/apostrophes with ASCII equivalents."""
        return text.replace('‘', "'").replace('’', "'") \
                   .replace('“', '"').replace('”', '"') \
                   .replace('‛', "'").replace('‚', "'") \
                   .replace('„', '"').replace('‟', '"') \
                   .replace('—', ' -- ').replace('–', '-') \
                   .replace('…', '...')

    @staticmethod
    def _fix_mojibake(text: str) -> str:
        """Fix UTF-8 bytes that were decoded as latin-1 (mojibake).

        Handles: â€™ -> ', â€" -> --, â€œ -> ", â€ -> ..., etc.
        """
        return text.replace('\xc3\xa2\xe2\x82\xac\xe2\x84\xa2', "'") \
                   .replace('\xe2\x80\x99', "'") \
                   .replace('\xe2\x80\x98', "'") \
                   .replace('\xe2\x80\x9c', '"') \
                   .replace('\xe2\x80\x9d', '"') \
                   .replace('\xe2\x80\x93', '-') \
                   .replace('\xe2\x80\x94', ' -- ') \
                   .replace('\xe2\x80\xa6', '...') \
                   .replace('\xc3\xa2\xe2\x82\xac\xe2\x80\x9c', '"') \
                   .replace('\xc3\xa2\xe2\x82\xac\xe2\x80\x9d', '"') \
                   .replace('\xc3\xa2\xe2\x82\xac\xe2\x80\x9d', '"') \
                   .replace('â€"', ' -- ') \
                   .replace('â€™', "'") \
                   .replace('â€˜', "'") \
                   .replace('â€œ', '"') \
                   .replace('â€\x9d', '"') \
                   .replace('â€\xa6', '...') \
                   .replace('â€"', '-')

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract text content from OpenAI-compatible response JSON."""
        try:
            raw = data["choices"][0]["message"]["content"]
            if raw is None:
                logger.warning("LLM response content was null: %s", json.dumps(data)[:500])
                return ""
            if not isinstance(raw, str):
                raw = str(raw)
            normalized = LLMGateway._normalize_quotes(raw)
            return LLMGateway._fix_mojibake(normalized)
        except (KeyError, IndexError, TypeError):
            logger.warning("Unexpected response structure: %s", json.dumps(data)[:500])
            return ""
