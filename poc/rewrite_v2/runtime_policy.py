"""Runtime budget policy for rewrite V2."""

from __future__ import annotations

import os
from typing import Any


MODE_RUNTIME_CAPS: dict[str, dict[str, int | float]] = {
    "academic_cited_text": {"generation_budget_max": 150, "runtime_fraction": 0.58, "reserved_tail_seconds": 75},
    "broad_explanatory_essay": {"generation_budget_max": 180, "runtime_fraction": 0.65, "reserved_tail_seconds": 60},
    "generic_expository": {"generation_budget_max": 150, "runtime_fraction": 0.60, "reserved_tail_seconds": 60},
    "technical_content": {"generation_budget_max": 90, "runtime_fraction": 0.45, "reserved_tail_seconds": 90},
    "regulated_policy_content": {"generation_budget_max": 75, "runtime_fraction": 0.40, "reserved_tail_seconds": 90},
    "structured_list_table": {"generation_budget_max": 60, "runtime_fraction": 0.35, "reserved_tail_seconds": 90},
    "quote_heavy": {"generation_budget_max": 90, "runtime_fraction": 0.45, "reserved_tail_seconds": 90},
    "short_text": {"generation_budget_max": 45, "runtime_fraction": 0.35, "reserved_tail_seconds": 45},
    "personal_reflection": {"generation_budget_max": 120, "runtime_fraction": 0.55, "reserved_tail_seconds": 60},
    "creative_marketing": {"generation_budget_max": 100, "runtime_fraction": 0.50, "reserved_tail_seconds": 60},
    "hybrid_guarded": {"generation_budget_max": 60, "runtime_fraction": 0.35, "reserved_tail_seconds": 90},
}


def _content_mode(content_route: Any | None) -> str:
    if content_route is None:
        return "generic_expository"
    if isinstance(content_route, dict):
        return str(content_route.get("content_mode") or "generic_expository")
    return str(getattr(content_route, "content_mode", "") or "generic_expository")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def llm_call_timeout_seconds(default: int = 30) -> int:
    value = _int_env("DRAFTPROOF_REWRITE_V2_LLM_TIMEOUT_SECONDS", default)
    return max(5, min(60, value))


def phase_start_margin_seconds(default: int = 5) -> int:
    value = _int_env("DRAFTPROOF_REWRITE_V2_PHASE_START_MARGIN_SECONDS", default)
    return max(1, min(60, value))


def runtime_policy(content_route: Any | None, *, max_runtime_seconds: int) -> dict[str, Any]:
    mode = _content_mode(content_route)
    caps = MODE_RUNTIME_CAPS.get(mode) or MODE_RUNTIME_CAPS["generic_expository"]
    raw_override = os.environ.get("DRAFTPROOF_REWRITE_V2_GENERATION_BUDGET_SECONDS")
    override = _int_env("DRAFTPROOF_REWRITE_V2_GENERATION_BUDGET_SECONDS", 0) if raw_override else 0
    upper_bound = max(30, int(max_runtime_seconds) - int(caps.get("reserved_tail_seconds") or 60))
    if override > 0:
        budget = max(30, min(override, upper_bound))
        source = "env_override"
    else:
        mode_cap = int(caps.get("generation_budget_max") or 150)
        fraction = float(caps.get("runtime_fraction") or 0.6)
        budget = max(30, min(mode_cap, int(max_runtime_seconds * fraction), upper_bound))
        source = "content_mode_policy"
    return {
        "content_mode": mode,
        "generation_budget_seconds": budget,
        "max_generation_budget_seconds": int(caps.get("generation_budget_max") or budget),
        "runtime_fraction": float(caps.get("runtime_fraction") or 0.6),
        "reserved_tail_seconds": int(caps.get("reserved_tail_seconds") or 60),
        "source": source,
    }
