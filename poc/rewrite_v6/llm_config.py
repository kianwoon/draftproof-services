from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from poc.llm.gateway import LLMConfig, LLMGateway


DEFAULT_V6_MODEL = "openai/gpt-oss-120b"
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


def deterministic_mode() -> bool:
    """Whether to pin sampling to a near-deterministic profile (measurement/eval use).

    The gpt-oss writer normally samples at temperature 0.45-0.65, so a single rewrite run varies
    by several risk points. Setting DRAFTPROOF_V6_DETERMINISTIC=1 forces temperature 0 / top_p 1
    so a change can be measured as signal rather than sampling noise. Off (production) by default.
    """
    return os.environ.get("DRAFTPROOF_V6_DETERMINISTIC", "").strip().lower() in {"1", "true", "yes", "on"}


def _as_deterministic(profile: dict[str, Any]) -> dict[str, Any]:
    pinned = dict(profile)
    pinned["temperature"] = 0.0
    pinned["top_p"] = 1.0
    for greedy_neutral in ("presence_penalty", "frequency_penalty"):
        if greedy_neutral in pinned:
            pinned[greedy_neutral] = 0
    if "repetition_penalty" in pinned:
        pinned["repetition_penalty"] = 1.0
    return pinned


def writer_model() -> str:
    return os.environ.get("DRAFTPROOF_V6_WRITER_MODEL") or DEFAULT_V6_MODEL


def full_doc_rewrite_model() -> str | None:
    return os.environ.get("DRAFTPROOF_V6_FULL_DOC_REWRITE_MODEL") or None


def grammar_model() -> str | None:
    return os.environ.get("DRAFTPROOF_V6_GRAMMAR_MODEL") or None


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


def full_doc_rewrite_gateway(
    *,
    api_key: str | None,
    base_url: str | None,
    text: str = "",
    cancellation_check: Callable[[], None] | None = None,
) -> LLMGateway | None:
    model = full_doc_rewrite_model()
    if not model:
        return None
    return LLMGateway(
        LLMConfig(
            model=model,
            api_key=_first_env("DRAFTPROOF_V6_FULL_DOC_REWRITE_API_KEY", "OPENROUTER_API_KEY") or api_key,
            base_url=_first_env("DRAFTPROOF_V6_FULL_DOC_REWRITE_BASE_URL", "LLM_BASE_URL") or base_url,
            **full_doc_rewrite_llm_profile(model, text),
            provider=provider_from_env("FULL_DOC_REWRITE", model),
            extra_body=full_doc_rewrite_extra_body(model),
            app_label="FullDocRewrite",
            max_retries=1,
            timeout=_int_env("DRAFTPROOF_V6_FULL_DOC_REWRITE_TIMEOUT", 180),
            cancellation_check=cancellation_check,
        )
    )


def grammar_gateway(
    *,
    api_key: str | None,
    base_url: str | None,
    cancellation_check: Callable[[], None] | None = None,
) -> LLMGateway | None:
    model = grammar_model()
    if not model:
        return None
    return LLMGateway(
        LLMConfig(
            model=model,
            api_key=_first_env("DRAFTPROOF_V6_GRAMMAR_API_KEY", "OPENROUTER_API_KEY") or api_key,
            base_url=_first_env("DRAFTPROOF_V6_GRAMMAR_BASE_URL", "LLM_BASE_URL") or base_url,
            **grammar_llm_profile(model),
            provider=provider_from_env("GRAMMAR", model),
            extra_body=grammar_extra_body(model),
            app_label="GrammarRepair",
            max_retries=1,
            timeout=_int_env("DRAFTPROOF_V6_GRAMMAR_TIMEOUT", 120),
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
    if not order:
        order = _csv_env("DRAFTPROOF_V6_PROVIDER_DEFAULT_ORDER")
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


# gpt-oss reasoning depth per role. The complex restructure-while-preserving-meaning task the
# writer/planner do benefits from deeper reasoning; the selector only picks an id, so it stays low.
# Cerebras (OpenAI-compatible) takes a TOP-LEVEL `reasoning_effort`; OpenRouter takes nested
# `reasoning.effort`. Both were previously OFF on Cerebras (returned None + gateway force-disable).
# reasoning_effort is OPT-IN on Cerebras (env only) -- gpt-oss spends its token budget on reasoning,
# which truncates output unless max_completion_tokens is raised accordingly, so it must be enabled
# deliberately. OpenRouter keeps its prior medium default. Set DRAFTPROOF_V6_<ROLE>_REASONING_EFFORT.
_OPENROUTER_REASONING_EFFORT = {"planner": "medium", "writer": "medium", "selector": "low"}


def _reasoning_effort(role: str, default: str) -> str | None:
    raw = os.environ.get(f"DRAFTPROOF_V6_{role.upper()}_REASONING_EFFORT")
    value = (raw if raw is not None else default).strip().lower()
    return value if value in {"low", "medium", "high"} else None


def _gpt_oss_reasoning_extra(role: str) -> dict[str, Any] | None:
    if using_cerebras_direct():
        effort = _reasoning_effort(role, "")  # opt-in: only when env explicitly sets it
        return {"reasoning_effort": effort} if effort else None
    effort = _reasoning_effort(role, _OPENROUTER_REASONING_EFFORT.get(role, "medium"))
    return {"reasoning": {"effort": effort, "exclude": True}, "include_reasoning": False} if effort else None


def planner_extra_body(model: str) -> dict[str, Any] | None:
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return _gpt_oss_reasoning_extra("planner")
    if using_cerebras_direct():
        return None
    if "thinking" in normalized:
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 128}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False}


def selector_extra_body(model: str) -> dict[str, Any] | None:
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return _gpt_oss_reasoning_extra("selector")
    if using_cerebras_direct():
        return None
    if "thinking" in normalized:
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 32}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False}


def writer_extra_body(model: str) -> dict[str, Any] | None:
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return _gpt_oss_reasoning_extra("writer")
    if using_cerebras_direct():
        return None
    if "thinking" not in normalized:
        return None
    return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}


def full_doc_rewrite_extra_body(model: str) -> dict[str, Any] | None:
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return _gpt_oss_reasoning_extra("full_doc_rewrite")
    if "thinking" in normalized:
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False} if not using_cerebras_direct() else None


def grammar_extra_body(model: str) -> dict[str, Any] | None:
    normalized = str(model or "").casefold()
    if "gpt-oss" in normalized:
        return _gpt_oss_reasoning_extra("grammar")
    if "thinking" in normalized:
        return {"reasoning": {"enabled": True, "exclude": True, "max_tokens": 64}, "include_reasoning": False}
    return {"reasoning": {"enabled": False}, "include_reasoning": False} if not using_cerebras_direct() else None


def planner_llm_profile(model: str) -> dict[str, Any]:
    if "gpt-oss" in str(model or "").casefold():
        profile = {"max_tokens": None, "temperature": 0.2, "top_p": 0.9, "top_k": 0, "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1.0}
    else:
        profile = {"max_tokens": None, "temperature": 0.1, "top_p": 0.75}
    return _as_deterministic(profile) if deterministic_mode() else profile


def selector_llm_profile(model: str) -> dict[str, Any]:
    profile = dict(planner_llm_profile(model))
    profile["temperature"] = 0.0
    profile["top_p"] = 1.0
    return profile


def writer_llm_profile(model: str, text: str = "") -> dict[str, Any]:
    if "gpt-oss" not in str(model or "").casefold():
        profile = {"max_tokens": None, "temperature": 0.12, "top_p": 0.75}
    else:
        source_sensitive = _source_sensitive_text(text)
        profile = {
            "max_tokens": None,
            "temperature": 0.45 if source_sensitive else 0.65,
            "top_p": 0.9 if source_sensitive else 0.95,
            "top_k": 0,
            "presence_penalty": 0,
            "frequency_penalty": 0.1 if source_sensitive else 0.15,
            "repetition_penalty": 1.03 if source_sensitive else 1.05,
        }
    return _as_deterministic(profile) if deterministic_mode() else profile


def full_doc_rewrite_llm_profile(model: str, text: str = "") -> dict[str, Any]:
    if "gpt-oss" in str(model or "").casefold():
        profile = writer_llm_profile(model, text)
        profile["temperature"] = min(float(profile.get("temperature") or 0.45), 0.45)
        profile["top_p"] = min(float(profile.get("top_p") or 0.9), 0.9)
    else:
        profile = {"max_tokens": None, "temperature": 0.2, "top_p": 0.85}
    return _as_deterministic(profile) if deterministic_mode() else profile


def grammar_llm_profile(model: str) -> dict[str, Any]:
    if "gpt-oss" in str(model or "").casefold():
        profile = {"max_tokens": None, "temperature": 0.1, "top_p": 0.9, "top_k": 0}
    else:
        profile = {"max_tokens": None, "temperature": 0.1, "top_p": 0.9}
    return _as_deterministic(profile) if deterministic_mode() else profile


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
    value = str(text or "")
    return bool(
        re.search(r"\([A-Z][A-Za-z .,&;'-]*\b\d{4}\)", value)
        or re.search(r"\b[A-Z]{2,}\b", value)
        or re.search(r"\b[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,6}\s+(?:Act|Standards?|Code|Policy|Framework)\b", value)
    )


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


def _int_env(name: str, default: int) -> int:
    value = _first_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
