from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from poc.llm.gateway import LLMConfig, LLMGateway


DEFAULT_V6_MODEL = "openai/gpt-oss-120b"
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


def writer_model() -> str:
    return os.environ.get("DRAFTPROOF_V6_WRITER_MODEL") or DEFAULT_V6_MODEL


def planner_model() -> str:
    return (
        os.environ.get("DRAFTPROOF_V6_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_REWRITE_V5_PLANNER_MODEL")
        or DEFAULT_V6_MODEL
    )


def planner_gateway(
    *,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None = None,
) -> LLMGateway:
    model = resolve_v6_model(planner_model()) or planner_model()
    return LLMGateway(
        LLMConfig(
            model=model,
            api_key=resolve_v6_api_key(api_key),
            base_url=resolve_v6_base_url(base_url),
            **planner_llm_profile(model),
            provider=provider_from_env("PLANNER", model),
            extra_body=planner_extra_body(model),
            cancellation_check=cancellation_check,
        )
    )


def selector_gateway(
    *,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None = None,
) -> LLMGateway:
    model = resolve_v6_model(_selector_model()) or _selector_model()
    return LLMGateway(
        LLMConfig(
            model=model,
            api_key=resolve_v6_api_key(api_key),
            base_url=resolve_v6_base_url(base_url),
            **selector_llm_profile(model),
            provider=provider_from_env("SELECTOR", model),
            extra_body=selector_extra_body(model),
            cancellation_check=cancellation_check,
        )
    )


def provider_from_env(role: str, model: str) -> dict[str, Any] | None:
    if using_cerebras_direct():
        return None
    prefix = f"DRAFTPROOF_V6_{role}_PROVIDER"
    raw_json = _first_env(f"{prefix}_ROUTING_JSON")
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError(f"V6 {role.casefold()} provider routing JSON must be an object")
        return parsed
    provider: dict[str, Any] = {}
    order = _csv_env(f"{prefix}_ORDER")
    only = _csv_env(f"{prefix}_ONLY")
    ignore = _csv_env(f"{prefix}_IGNORE")
    if not order and str(model or "").casefold() == "z-ai/glm-4.7":
        order = ["Cerebras"]
    if order:
        provider["order"] = order
    if only:
        provider["only"] = only
    if ignore:
        provider["ignore"] = ignore
    allow_fallbacks = _bool_env(f"{prefix}_ALLOW_FALLBACKS")
    if allow_fallbacks is None:
        allow_fallbacks = _bool_env(f"DRAFTPROOF_V6_{role}_ALLOW_FALLBACKS")
    if allow_fallbacks is None and order:
        allow_fallbacks = True
    if allow_fallbacks is not None:
        provider["allow_fallbacks"] = allow_fallbacks
    sort = _first_env(f"{prefix}_SORT")
    if sort:
        provider["sort"] = sort
    return provider or None


def planner_extra_body(model: str) -> dict[str, Any] | None:
    if using_cerebras_direct():
        return None
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return {"reasoning": {"effort": "medium", "exclude": True}, "include_reasoning": False}
    if "thinking" in normalized:
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 128}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False}


def selector_extra_body(model: str) -> dict[str, Any] | None:
    if using_cerebras_direct():
        return None
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return {"reasoning": {"effort": "low", "exclude": True}, "include_reasoning": False}
    if "thinking" in normalized:
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 32}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False}


def writer_extra_body(model: str) -> dict[str, Any] | None:
    if using_cerebras_direct():
        return None
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return {"reasoning": {"effort": "medium", "exclude": True}, "include_reasoning": False}
    if "thinking" not in normalized:
        return None
    return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}


def planner_llm_profile(model: str) -> dict[str, Any]:
    if "gpt-oss" in str(model or "").casefold():
        return {"max_tokens": None, "temperature": 0.2, "top_p": 0.9, "top_k": 0, "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1.0}
    return {"max_tokens": None, "temperature": 0.1, "top_p": 0.75}


def selector_llm_profile(model: str) -> dict[str, Any]:
    profile = dict(planner_llm_profile(model))
    profile["temperature"] = 0.0
    profile["top_p"] = 1.0
    return profile


def writer_llm_profile(model: str, text: str = "") -> dict[str, Any]:
    if "gpt-oss" not in str(model or "").casefold():
        return {"max_tokens": None, "temperature": 0.12, "top_p": 0.75}
    source_sensitive = _source_sensitive_text(text)
    return {
        "max_tokens": None,
        "temperature": 0.45 if source_sensitive else 0.65,
        "top_p": 0.9 if source_sensitive else 0.95,
        "top_k": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0.1 if source_sensitive else 0.15,
        "repetition_penalty": 1.03 if source_sensitive else 1.05,
    }


def resolve_v6_api_key(api_key: str | None) -> str | None:
    return (os.environ.get("CEREBRAS_API_KEY") or api_key) if using_cerebras_direct() else api_key


def resolve_v6_base_url(base_url: str | None) -> str | None:
    return _CEREBRAS_BASE_URL if using_cerebras_direct() else base_url


def resolve_v6_model(model: str | None) -> str | None:
    return cerebras_model_name(model or writer_model()) if using_cerebras_direct() else model


def cerebras_model_name(model: str | None) -> str:
    value = str(model or DEFAULT_V6_MODEL).strip()
    if value == DEFAULT_V6_MODEL:
        return "gpt-oss-120b"
    return value.split("/", 1)[1] if value.startswith("openai/") else value


def using_cerebras_direct() -> bool:
    explicit = _bool_env("DRAFTPROOF_V6_CEREBRAS_DIRECT")
    if explicit is not None:
        return explicit
    return bool(os.environ.get("CEREBRAS_API_KEY"))


def _selector_model() -> str:
    return (
        os.environ.get("DRAFTPROOF_V6_SELECTOR_MODEL")
        or os.environ.get("DRAFTPROOF_V6_PLANNER_MODEL")
        or os.environ.get("DRAFTPROOF_PLANNER_MODEL")
        or DEFAULT_V6_MODEL
    )


def _source_sensitive_text(text: str) -> bool:
    return bool(re.search(r"\([A-Z][A-Za-z .,&;'-]*\b\d{4}\)|\b(?:Act|Standards?|assessment|competency|citation|legal|VET|TAFE|unit)\b", str(text or ""), flags=re.I))


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _csv_env(name: str) -> list[str]:
    value = _first_env(name)
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _bool_env(name: str) -> bool | None:
    value = _first_env(name)
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None
