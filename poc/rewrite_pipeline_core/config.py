from __future__ import annotations

import os
import re

from detect.topk_calibration import TOPK_CALIBRATED_SAFE_LIMIT


TOPK_SAFE_LIMIT = TOPK_CALIBRATED_SAFE_LIMIT


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default

def _float_env_with_fallback(name: str, fallback: float) -> float:
    value = _float_env_optional(name)
    return float(value) if value is not None else float(fallback)

def _safe_topk_calibrated_limit() -> float:
    """Hard product ceiling for calibrated Top-k mitigation safety.

    Raw GPT-2 Top-k remains diagnostic. Safe-band decisions use the fixed
    calibrated risk scale and are not environment-tuned.
    """
    return TOPK_SAFE_LIMIT

def _safe_topk_limit() -> float:
    """Compatibility alias for older tests/callers."""
    return _safe_topk_calibrated_limit()

def _rewrite_sampling_profile(prefix: str = "DRAFTPROOF_AI_SEARCH") -> dict:
    """Effective default sampling controls for rewrite-generation calls.

    Phase-specific calls may override these values, but generation should not
    silently collapse to temperature-only sampling when env vars are unset.
    """
    return {
        "temperature": _float_env(f"{prefix}_TEMPERATURE", 0.45),
        "top_p": _float_env_with_fallback(f"{prefix}_TOP_P", 0.82),
        "top_k": _int_env_optional(f"{prefix}_TOP_K"),
        "presence_penalty": _float_env_with_fallback(f"{prefix}_PRESENCE_PENALTY", 0.15),
        "frequency_penalty": _float_env_with_fallback(f"{prefix}_FREQUENCY_PENALTY", 0.25),
    }

def _phase_sampling_arg(phase_prefix: str, key: str, fallback_prefix: str = "DRAFTPROOF_AI_SEARCH"):
    env_name = f"{phase_prefix}_{key}"
    fallback = _rewrite_sampling_profile(fallback_prefix)
    phase_defaults = {
        ("DRAFTPROOF_TOPK_ROUTE", "TOP_P"): 0.72,
        ("DRAFTPROOF_TOPK_ROUTE", "PRESENCE_PENALTY"): 0.10,
        ("DRAFTPROOF_TOPK_ROUTE", "FREQUENCY_PENALTY"): 0.35,
    }
    if key == "TOP_K":
        value = _int_env_optional(env_name)
        return value if value is not None else fallback.get("top_k")
    field = {
        "TOP_P": "top_p",
        "PRESENCE_PENALTY": "presence_penalty",
        "FREQUENCY_PENALTY": "frequency_penalty",
    }.get(key)
    value = _float_env_optional(env_name)
    if value is not None:
        return value
    if (phase_prefix, key) in phase_defaults:
        return phase_defaults[(phase_prefix, key)]
    return fallback.get(field)

def _phase_chat_sampling_kwargs(
    phase_prefix: str,
    *,
    temperature_env: str,
    temperature_default: float,
    max_tokens_env: str,
    max_tokens_default: int,
    fallback_prefix: str = "DRAFTPROOF_AI_SEARCH",
) -> dict:
    """Build normalized sampling kwargs for generation calls."""
    return {
        "temperature": float(os.environ.get(temperature_env, str(temperature_default))),
        "max_tokens": int(os.environ.get(max_tokens_env, str(max_tokens_default))),
        "top_p": _phase_sampling_arg(phase_prefix, "TOP_P", fallback_prefix=fallback_prefix),
        "top_k": _phase_sampling_arg(phase_prefix, "TOP_K", fallback_prefix=fallback_prefix),
        "presence_penalty": _phase_sampling_arg(
            phase_prefix,
            "PRESENCE_PENALTY",
            fallback_prefix=fallback_prefix,
        ),
        "frequency_penalty": _phase_sampling_arg(
            phase_prefix,
            "FREQUENCY_PENALTY",
            fallback_prefix=fallback_prefix,
        ),
    }

def _llm_call_budget_exhausted_before_send(optimistic_llm_calls: int, max_llm_calls: int) -> bool:
    """Return true only after the optimistic pre-send count exceeds budget."""
    return int(optimistic_llm_calls or 0) > int(max_llm_calls or 0)

def _resolve_stage_llm_budget(
    primary_env: str,
    fallback_env: str | None = None,
    *,
    default: int = 0,
) -> int:
    """Resolve an LLM call budget with an explicit fallback env.

    Several rewrite stages share the same user-visible budget intent. This
    helper prevents a later stage from silently ignoring a cap that was set for
    the broader rewrite/search controller.
    """
    for name in (primary_env, fallback_env):
        if not name:
            continue
        value = _int_env_optional(name)
        if value is not None:
            return max(0, int(value))
    return max(0, int(default))

def _float_env_optional(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

def _int_env_optional(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None

def _safe_index(value, default: int = -1) -> int:
    """Parse indexes without treating zero as missing."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _load_local_env(env_path: str | None = None) -> list[str]:
    """Load simple KEY=VALUE pairs from repo .env without overriding exports."""
    if env_path is None:
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    loaded = []
    if not os.path.exists(env_path):
        return loaded
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                    continue
                if key in os.environ:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                loaded.append(key)
    except OSError:
        return loaded
    return loaded

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}

def _role_model(role: str, fallback_model: str | None = None) -> str | None:
    """Resolve stage-specific LLM model names from env with legacy fallback."""
    role = (role or "").strip().lower()
    model_lock = _rewrite_model_lock()
    if model_lock:
        return model_lock
    role_env = {
        "planner": ("DRAFTPROOF_PLANNER_MODEL", "PLANNER_MODEL", "LLM_PLANNER_MODEL"),
        "generator": ("DRAFTPROOF_GENERATOR_MODEL", "GENERATOR_MODEL", "LLM_GENERATOR_MODEL"),
        "retry": ("DRAFTPROOF_RETRY_MODEL", "RETRY_MODEL", "LLM_RETRY_MODEL"),
    }.get(role, ())
    for name in role_env:
        value = os.environ.get(name)
        if value and str(value).strip():
            return str(value).strip()
    if role == "retry":
        return _role_model("generator", fallback_model)
    return fallback_model or os.environ.get("LLM_MODEL")

def _rewrite_model_lock() -> str | None:
    """Optional hard lock for all rewrite LLM roles."""
    raw = os.environ.get("DRAFTPROOF_REWRITE_MODEL_LOCK", "openai/gpt-4.1-mini")
    value = str(raw or "").strip()
    if not value or value.lower() in {"0", "off", "false", "none", "disabled"}:
        return None
    return value

def _retry_model_enabled() -> bool:
    """Kill switch for expensive retry-model escalation."""
    return _env_flag("DRAFTPROOF_RETRY_MODEL_ENABLED", False) or _env_flag(
        "RETRY_MODEL_ENABLED",
        False,
    )

def _retry_model_max_calls() -> int:
    raw = (
        os.environ.get("DRAFTPROOF_RETRY_MODEL_MAX_CALLS")
        or os.environ.get("RETRY_MODEL_MAX_CALLS")
    )
    try:
        return max(0, int(raw if raw is not None else "1"))
    except ValueError:
        return 1

def _llm_role_config(fallback_model: str | None = None) -> dict:
    retry_enabled = _retry_model_enabled()
    retry_model = _role_model("retry", fallback_model)
    return {
        "planner_model": _role_model("planner", fallback_model),
        "generator_model": _role_model("generator", fallback_model),
        "retry_model": retry_model,
        "retry_model_enabled": retry_enabled,
        "retry_model_max_calls": _retry_model_max_calls() if retry_enabled else 0,
    }
