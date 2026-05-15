"""Prompt compiler and candidate generator for V4 repair briefs."""

from __future__ import annotations

import json
import os
from typing import Any

from llm.gateway import LLMGateway
from rewrite_v3.document_units import word_count

from .models import CandidateVariant, RepairBrief
from .validation import parse_generator_variants


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


def word_bounds(source_text: str, *, tolerance: float | None = None) -> tuple[int, int]:
    count = max(1, word_count(source_text))
    if tolerance is None:
        tolerance = _float_env("DRAFTPROOF_REWRITE_V4_WORD_TOLERANCE", 0.18, minimum=0.05, maximum=0.35)
    return round(count * (1.0 - tolerance)), round(count * (1.0 + tolerance))


def build_generator_prompt(*, group: Any, repair_brief: RepairBrief, variant_count: int = 3) -> str:
    source_text = str(getattr(group, "source_text", "") or "")
    min_words, max_words = word_bounds(source_text)
    source_words = max(1, word_count(source_text))
    payload = {
        "task": "editorial_repair_candidate_generation",
        "paragraph_role": repair_brief.paragraph_role,
        "original_paragraph": source_text,
        "repair_tasks": list(repair_brief.repair_tasks),
        "constraints": [
            *repair_brief.constraints,
            f"Stay near the original length of about {source_words} words; do not compress into a summary or expand with new material.",
            f"Return exactly {max(1, int(variant_count))} variants.",
            "Keep it as one paragraph.",
        ],
        "avoid": [
            *repair_brief.avoid,
            "new facts",
            "new examples",
            "personal stories",
            "slang",
            "headings",
            "bullets",
            "markdown",
            "HTML",
            "commentary",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, max(1, int(variant_count)) + 1)
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_variants(
    *,
    group: Any,
    repair_brief: RepairBrief,
    gateway: LLMGateway,
    variant_count: int = 3,
) -> tuple[list[CandidateVariant], dict[str, Any], str, str]:
    prompt = build_generator_prompt(group=group, repair_brief=repair_brief, variant_count=variant_count)
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format={"type": "json_object"},
        temperature=0.35,
        top_p=0.85,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V4_GENERATOR_MAX_TOKENS", 6000, minimum=800, maximum=12000),
    )
    min_words, max_words = word_bounds(str(getattr(group, "source_text", "") or ""))
    variants, diagnostics = parse_generator_variants(response.content, min_words=min_words, max_words=max_words)
    return variants, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
    }, prompt, response.content
