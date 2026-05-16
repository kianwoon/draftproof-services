"""Layer 3 budget-lane controls for V4 rewrite experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from llm.gateway import LLMGateway
from rewrite_v2.structured_output import structured_json_request_options

from .models import CandidateVariant, ClusterRepairUnit
from .validation import parse_generator_variants


@dataclass(frozen=True)
class BudgetLane:
    lane_id: str
    changed_source_ratio_max: float
    growth_ratio_max: float
    shrink_ratio_max: float
    operation_families: tuple[str, ...]
    instruction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "changed_source_ratio_max": self.changed_source_ratio_max,
            "growth_ratio_max": self.growth_ratio_max,
            "shrink_ratio_max": self.shrink_ratio_max,
            "operation_families": list(self.operation_families),
            "instruction": self.instruction,
        }


def budget_lanes_for_unit(unit_word_count: int, *, risk_score: float = 0.0) -> list[BudgetLane]:
    """Return conservative-to-aggressive edit lanes for a repair unit.

    The ratios come from the measured V4 experiments: conservative captures low-risk
    wins cleanly, route widens for continuity repairs, and aggressive is reserved
    for persistent high-risk pockets.
    """
    _ = max(0, int(unit_word_count or 0))
    high_pressure = float(risk_score or 0.0) >= _float_env(
        "DRAFTPROOF_REWRITE_V4_LAYER3_HIGH_RISK_SCORE",
        0.55,
        minimum=0.0,
        maximum=1.0,
    )
    lanes = [
        BudgetLane(
            lane_id="conservative",
            changed_source_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_CONSERVATIVE_CHANGED_MAX", 0.25, minimum=0.05, maximum=0.80),
            growth_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_CONSERVATIVE_GROW_MAX", 0.12, minimum=0.0, maximum=0.80),
            shrink_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_CONSERVATIVE_SHRINK_MAX", 0.12, minimum=0.0, maximum=0.80),
            operation_families=("replace_only", "light_delete", "small_bump"),
            instruction="Make the smallest measurable route change first.",
        ),
        BudgetLane(
            lane_id="route",
            changed_source_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_ROUTE_CHANGED_MAX", 0.35, minimum=0.05, maximum=0.80),
            growth_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_ROUTE_GROW_MAX", 0.20, minimum=0.0, maximum=0.80),
            shrink_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_ROUTE_SHRINK_MAX", 0.20, minimum=0.0, maximum=0.80),
            operation_families=("replace_bump", "route_reorder", "delete_repath"),
            instruction="Repair the sentence route when small local edits do not move the score.",
        ),
    ]
    if high_pressure or _bool_env("DRAFTPROOF_REWRITE_V4_LAYER3_ALWAYS_INCLUDE_AGGRESSIVE", True):
        lanes.append(
            BudgetLane(
                lane_id="aggressive",
                changed_source_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_AGGRESSIVE_CHANGED_MAX", 0.45, minimum=0.05, maximum=0.80),
                growth_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_AGGRESSIVE_GROW_MAX", 0.20, minimum=0.0, maximum=0.80),
                shrink_ratio_max=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_AGGRESSIVE_SHRINK_MAX", 0.20, minimum=0.0, maximum=0.80),
                operation_families=("strong_replace", "strong_delete", "paragraph_route_rebuild"),
                instruction="Use a wider edit budget only for unresolved high-value targets.",
            )
        )
    return lanes


def edit_budget_profile(source_text: str, replacement_text: str) -> dict[str, Any]:
    """Compute word-level replace/delete/insert pressure for a candidate."""
    source_words = _words(source_text)
    replacement_words = _words(replacement_text)
    old_count = len(source_words)
    new_count = len(replacement_words)
    inserted = deleted = replaced_source = replaced_replacement = 0
    matcher = SequenceMatcher(None, source_words, replacement_words, autojunk=False)
    for tag, source_start, source_end, replacement_start, replacement_end in matcher.get_opcodes():
        if tag == "insert":
            inserted += replacement_end - replacement_start
        elif tag == "delete":
            deleted += source_end - source_start
        elif tag == "replace":
            replaced_source += source_end - source_start
            replaced_replacement += replacement_end - replacement_start
    denominator = max(old_count, 1)
    changed_source_words = replaced_source + deleted
    return {
        "source_words": old_count,
        "replacement_words": new_count,
        "inserted_words": inserted,
        "deleted_words": deleted,
        "replaced_source_words": replaced_source,
        "replaced_replacement_words": replaced_replacement,
        "changed_source_words": changed_source_words,
        "changed_source_ratio": round(changed_source_words / denominator, 4),
        "growth_ratio": round(max(0, new_count - old_count) / denominator, 4),
        "shrink_ratio": round(max(0, old_count - new_count) / denominator, 4),
        "net_length_delta_ratio": round((new_count - old_count) / denominator, 4),
    }


def budget_profile_passes(profile: dict[str, Any], lane: BudgetLane) -> bool:
    return (
        _num(profile.get("changed_source_ratio")) <= lane.changed_source_ratio_max
        and _num(profile.get("growth_ratio")) <= lane.growth_ratio_max
        and _num(profile.get("shrink_ratio")) <= lane.shrink_ratio_max
    )


def build_budget_lane_prompt(*, unit: ClusterRepairUnit, lane: BudgetLane, variant_count: int = 2) -> str:
    variants = max(1, min(3, int(variant_count or 1)))
    payload = {
        "task": "budget_lane_route_optimizer",
        "cluster_id": unit.cluster_id,
        "repair_unit": {
            "cluster_text": unit.text,
            "before_context": unit.before_context,
            "after_context": unit.after_context,
            "sentence_count": unit.sentence_count,
            "word_count": unit.word_count,
            "risk_score": unit.risk_score,
        },
        "budget_lane": lane.to_dict(),
        "operation_space": [
            "replace words",
            "bump in words",
            "delete words",
            "combine these only when it changes the route more cleanly",
        ],
        "repair_goal": [
            lane.instruction,
            "Reduce predictable token route and generic AI-style movement.",
            "Keep the same facts, sequence, citations, and source viewpoint.",
            "Return only replacements for cluster_text, not the full document.",
        ],
        "constraints": [
            f"Keep changed_source_ratio at or below {lane.changed_source_ratio_max:.2f}.",
            f"Keep growth_ratio at or below {lane.growth_ratio_max:.2f}.",
            f"Keep shrink_ratio at or below {lane.shrink_ratio_max:.2f}.",
            "Do not use detector or AI-authorship language in the replacement.",
            "Do not make the text casual, jokey, broken, or intentionally flawed.",
            f"Return exactly {variants} {'variant' if variants == 1 else 'variants'}.",
            "Return exactly the allowed keys for each variant: variant_id, text.",
        ],
        "forbidden": [
            "new facts",
            "new examples",
            "new names",
            "new dates",
            "new numbers",
            "new citations",
            "headings",
            "bullets",
            "markdown",
            "HTML",
            "first-person conversion unless already present in cluster_text",
            "grammar damage or random errors",
            "full document rewrite",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, variants + 1)
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_budget_lane_variants(
    *,
    unit: ClusterRepairUnit,
    lane: BudgetLane,
    gateway: LLMGateway,
    variant_count: int = 2,
) -> tuple[list[CandidateVariant], dict[str, Any], str, str]:
    prompt = build_budget_lane_prompt(unit=unit, lane=lane, variant_count=variant_count)
    response_format = _variants_response_format(max(1, min(3, int(variant_count or 1))))
    structured_options = structured_json_request_options(getattr(gateway, "model", None), response_format)
    provider = _merge_provider_options(getattr(gateway, "provider", None), structured_options.get("provider"))
    max_tokens = _int_env("DRAFTPROOF_REWRITE_V4_LAYER3_MAX_TOKENS", 3000, minimum=800, maximum=8000)
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured_options.get("response_format") or {"type": "json_object"},
        provider=provider,
        temperature=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_TEMPERATURE", 0.32, minimum=0.0, maximum=1.0),
        top_p=_float_env("DRAFTPROOF_REWRITE_V4_LAYER3_TOP_P", 0.82, minimum=0.1, maximum=1.0),
        max_tokens=max_tokens,
    )
    raw_completion = response.raw_content or response.content
    min_words = max(1, round(unit.word_count * (1.0 - lane.shrink_ratio_max)))
    max_words = max(min_words + 1, round(unit.word_count * (1.0 + lane.growth_ratio_max)))
    variants, diagnostics = parse_generator_variants(
        raw_completion,
        min_words=min_words,
        max_words=max_words,
        source_text=unit.text,
    )
    return variants, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "max_tokens": max_tokens,
        "structured_output_mode": structured_options.get("structured_output_mode"),
        "budget_lane": lane.to_dict(),
    }, prompt, raw_completion


def _words(text: str) -> list[str]:
    return str(text or "").replace("\n", " ").split()


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _variants_response_format(variant_count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rewrite_v4_budget_lane_variants",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variants": {
                        "type": "array",
                        "minItems": variant_count,
                        "maxItems": variant_count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "variant_id": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["variant_id", "text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["variants"],
                "additionalProperties": False,
            },
        },
    }


def _merge_provider_options(base: Any, required: Any) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(required, dict):
        merged.update(required)
    return merged or None
