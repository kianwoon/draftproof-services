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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
        "editing_depth": "light-to-moderate student essay editing; preserve ordinary learner phrasing when it is understandable",
        "preserve_traits": [
            "plain wording",
            "clear essay tone",
            "source viewpoint",
            "paragraph role",
            "student writer texture",
            "existing concrete course or workplace anchors",
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
    structure = _structure_contract(source_text)
    unit_label = "section" if structure["nonempty_line_count"] > 1 else "paragraph"
    payload = {
        "task": "editorial_repair_candidate_generation",
        "repair_mode": repair_brief.repair_mode,
        "paragraph_role": repair_brief.paragraph_role,
        "unit_kind": unit_label,
        "structure_contract": structure,
        "route_hint": _route_hint(repair_brief.paragraph_role),
        "target_voice_profile": default_voice_profile(),
        "original_unit": source_text,
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
            _structure_instruction(structure),
            "Do not globally grammar-polish the writing. Preserve the writer's undergraduate/student voice unless a wording problem blocks meaning.",
            "Keep existing concrete anchors prominent, especially course names, unit codes, workplace context, and first-person professional-practice context already present in the source.",
            "For opening/background framing, start from an existing concrete anchor when the source supplies one; do not begin with a generic field-wide overview if the paragraph already names the course, unit, or workplace context.",
            "If you move a concrete anchor into the opening, make that anchor the grammatical subject of the sentence. Do not keep a generic 'the importance of...' or 'this is particularly noticeable...' route as the main frame.",
            "When repairing a broad opening, avoid the becoming-increasingly-important/clear/obvious frame; use the existing course or unit to show why the topic matters.",
            "When using an anchor-first opening, still represent the original broad field claim and the original AI/information-era claim; connect them to the anchor instead of deleting them.",
            "Prefer small route and linkage changes over replacing the paragraph with smoother academic phrasing.",
            "JSON safety: preserve curly quotation marks from the source where possible. If you use straight double quotes inside text values, they must be escaped as JSON.",
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
            "global grammar polish",
            "professional copyediting voice",
            "generic importance-of opening",
            "becoming increasingly important opening",
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
    if (
        not variants
        and diagnostics.get("status") == "json_parse_failed"
        and _bool_env("DRAFTPROOF_REWRITE_V4_REPAIR_MALFORMED_JSON", True)
    ):
        repair_prompt = _json_repair_prompt(
            raw_completion=response.content,
            variant_count=variant_count,
        )
        repair_response = gateway.chat(
            repair_prompt,
            system="Return only valid JSON with a variants array.",
            response_format={"type": "json_object"},
            temperature=0.0,
            top_p=0.5,
            max_tokens=_int_env("DRAFTPROOF_REWRITE_V4_JSON_REPAIR_MAX_TOKENS", 6000, minimum=800, maximum=12000),
        )
        repaired_variants, repair_diagnostics = parse_generator_variants(
            repair_response.content,
            min_words=min_words,
            max_words=max_words,
        )
        if repaired_variants:
            return repaired_variants, {
                **repair_diagnostics,
                "status": "ok_after_json_repair",
                "first_parse": diagnostics,
                "repair_model": repair_response.model,
                "repair_provider": repair_response.raw.get("provider"),
                "repair_usage": repair_response.usage,
                "model": response.model,
                "provider": response.raw.get("provider"),
                "usage": response.usage,
            }, prompt, response.content
        diagnostics = {
            **diagnostics,
            "repair_attempt": {
                **repair_diagnostics,
                "model": repair_response.model,
                "provider": repair_response.raw.get("provider"),
                "usage": repair_response.usage,
            },
        }
    return variants, {
        **diagnostics,
        "model": response.model,
        "provider": response.raw.get("provider"),
        "usage": response.usage,
    }, prompt, response.content


def _structure_contract(source_text: str) -> dict[str, Any]:
    lines = str(source_text or "").splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    first = nonempty[0] if nonempty else ""
    return {
        "nonempty_line_count": len(nonempty) or 1,
        "has_heading_like_first_line": _looks_like_heading(first),
        "first_nonempty_line": first,
        "preserve_line_breaks": len(nonempty) > 1,
    }


def _structure_instruction(structure: dict[str, Any]) -> str:
    if int(structure.get("nonempty_line_count") or 1) > 1:
        heading = str(structure.get("first_nonempty_line") or "").strip()
        if structure.get("has_heading_like_first_line") and heading:
            return f"Preserve the same section structure and keep the first line as the heading: {heading}"
        return "Preserve the same section structure and line breaks; do not flatten separate blocks into one paragraph."
    return "Keep it as one paragraph."


def _route_hint(paragraph_role: str) -> dict[str, Any]:
    if str(paragraph_role or "").strip() == "opening/background framing":
        return {
            "preferred_route": [
                "Start with an existing concrete course, unit, or workplace anchor.",
                "Use the next sentence to connect that anchor back to the broader field or era claim already in the source.",
                "Then keep the writer's observed classroom/practice problem and report purpose.",
                "A useful shape is: existing course or unit shows why the topic matters; existing era/context changes how students learn; existing practice observation explains the problem.",
            ],
            "avoid_route": [
                "Do not simply prepend the anchor to the old generic importance-of sentence.",
                "Do not keep a becoming-increasingly-important frame when the source can instead say what the course or unit shows.",
                "Do not delete the broad vocational education or AI/information-era claim.",
            ],
        }
    return {}


def _looks_like_heading(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    return len(stripped.split()) <= 8 and stripped[-1] not in ".?!)]"


def _json_repair_prompt(*, raw_completion: str, variant_count: int) -> str:
    payload = {
        "task": "repair_malformed_variants_json",
        "invalid_completion": str(raw_completion or ""),
        "rules": [
            "Do not rewrite, shorten, expand, or improve any candidate text.",
            "Only convert the invalid completion into valid JSON.",
            "Escape straight double quote characters inside text values.",
            "Return exactly the variants that are present in the invalid completion.",
            f"The expected number of variants is {max(1, int(variant_count))}.",
            "Return only JSON with one top-level key: variants.",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, max(1, int(variant_count)) + 1)
            ]
        },
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)
