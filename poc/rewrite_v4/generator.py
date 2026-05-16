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
        tolerance = _float_env("DRAFTPROOF_REWRITE_V4_WORD_TOLERANCE", 0.35, minimum=0.05, maximum=0.35)
    return round(count * (1.0 - tolerance)), round(count * (1.0 + tolerance))


def default_voice_profile() -> dict[str, Any]:
    override = os.environ.get("DRAFTPROOF_REWRITE_V4_VOICE_PROFILE_JSON", "").strip()
    if override:
        try:
            parsed = json.loads(override)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return {
        "education_level": os.environ.get("DRAFTPROOF_REWRITE_V4_EDUCATION_LEVEL", "bachelor_degree"),
        "voice_register": "clear undergraduate essay",
        "tone": "plain, serious, and readable",
        "vocabulary_level": "standard academic, not advanced scholarly",
        "sentence_texture": "mostly direct medium-length sentences with some shorter sentences",
        "claim_style": "explain claims with simple reasoning, not dense theory",
        "specificity_style": "student-plausible concrete context when allowed; no unsupported expert detail",
        "transition_style": "natural essay transitions, not formulaic textbook transitions",
        "preserve_traits": [
            "plain wording",
            "clear essay tone",
            "source viewpoint",
            "paragraph role",
        ],
        "repair_traits": [
            "too-even sentence route",
            "generic transitions",
            "broad unsupported claims",
            "over-polished academic phrasing",
        ],
        "avoid": [
            "PhD-level phrasing",
            "journal-article tone",
            "rare vocabulary",
            "casual slang",
            "fake personal anecdotes",
            "excessive abstraction",
        ],
    }


def build_generator_prompt(*, group: Any, repair_brief: RepairBrief, variant_count: int = 3) -> str:
    source_text = str(getattr(group, "source_text", "") or "")
    min_words, max_words = word_bounds(source_text)
    source_words = max(1, word_count(source_text))
    payload = {
        "task": "editorial_repair_candidate_generation",
        "repair_mode": repair_brief.repair_mode,
        "paragraph_role": repair_brief.paragraph_role,
        "target_voice_profile": default_voice_profile(),
        "original_paragraph": source_text,
        "tutor_feedback": {
            "diagnosis": repair_brief.tutor_diagnosis,
            "student_explanation": repair_brief.student_explanation,
            "source_examples": list(repair_brief.source_examples),
            "repair_assignment": repair_brief.repair_assignment,
            "coverage_hint": repair_brief.coverage_hint,
        },
        "repair_tasks": list(repair_brief.repair_tasks),
        "constraints": [
            *repair_brief.constraints,
            "If tutor feedback asks for specificity, use only source wording already present; narrow or connect existing claims instead of adding examples.",
            "Preserve the paragraph's central claims and scope; do not turn a broad overview into a single example or narrower topic.",
            "Every original sentence's central claim must still be represented in the replacement.",
            "Do not merge sentences in a way that drops a source claim, list item group, or qualifying detail.",
            "Source-near specificity means tighter wording and clearer links among existing ideas, not new content.",
            f"Stay near the original length of about {source_words} words; do not compress into a summary or expand with new material.",
            f"Return exactly {max(1, int(variant_count))} variants.",
            "Keep it as one paragraph.",
            "Follow target_voice_profile; do not upgrade the paragraph into a scholarly or professional article voice.",
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
    if repair_brief.repair_mode == "controlled_enrichment_repair":
        payload["constraints"].extend([
            "This is controlled enrichment: you may add one or two short grounding phrases if source-only repair would stay generic.",
            "Any new phrase must be general context that connects existing claims, not a new named fact or example.",
            "Do not add proper nouns, locations, organizations, events, dates, statistics, citations, quotes, or specific case details absent from the original paragraph.",
            "Do not present enrichment as observed personal experience.",
        ])
        payload["avoid"].extend([
            "new named facts",
            "new locations",
            "new organizations",
            "new events",
            "new numbers or statistics",
            "new citations",
        ])
    else:
        payload["constraints"].append(
            "Do not introduce nouns, named entities, fields, mechanisms, dates, numbers, or examples that are absent from the original paragraph."
        )
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
