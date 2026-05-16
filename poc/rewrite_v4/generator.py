"""Prompt compiler and candidate generator for V4 repair briefs."""

from __future__ import annotations

import json
import os
from typing import Any

from llm.gateway import LLMGateway
from rewrite_v2.structured_output import structured_json_request_options
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
    scope = _candidate_generation_scope(source_text, repair_brief)
    prompt_unit = scope["editable_text"]
    source_words = max(1, word_count(prompt_unit))
    structure = _structure_contract(prompt_unit)
    unit_label = "section" if structure["nonempty_line_count"] > 1 else "paragraph"
    effective_variant_count = _effective_variant_count(variant_count, repair_brief.mitigation_strategy)
    payload = {
        "task": "editorial_repair_candidate_generation",
        "repair_mode": repair_brief.repair_mode,
        "paragraph_role": repair_brief.paragraph_role,
        "unit_kind": unit_label,
        "structure_contract": structure,
        "route_hint": _route_hint(repair_brief.paragraph_role),
        "target_voice_profile": default_voice_profile(),
        "generation_scope": {
            "mode": scope["mode"],
            "return_text_for": "editable_unit_only" if scope["locked_suffix"] else "full_unit",
            "locked_suffix_context": scope["locked_suffix"][:700],
        },
        "original_unit": prompt_unit,
        "mitigation_strategy": repair_brief.mitigation_strategy,
        "route_plan": _route_plan(repair_brief.mitigation_strategy),
        "rewrite_sequence": _strategy_steps(repair_brief.mitigation_strategy),
        "tutor_feedback": {
            "diagnosis": repair_brief.tutor_diagnosis,
            "student_explanation": repair_brief.student_explanation,
            "source_examples": list(repair_brief.source_examples),
            "repair_assignment": repair_brief.repair_assignment,
            "coverage_hint": repair_brief.coverage_hint,
        },
        "repair_tasks": list(repair_brief.repair_tasks),
        "constraints": [
            *_structure_aware_constraints(repair_brief.constraints, structure),
            *_strategy_controller_constraints(repair_brief.mitigation_strategy),
            "Execute route_plan.better_route as the paragraph route; do not merely polish the old route.",
            "Use route_plan.current_route to understand what needs to change, then apply route_plan.route_moves.",
            "Route moves are moves, not copies: if you move a source claim earlier, remove the old repeated occurrence.",
            "Do not repeat the same course, unit, difficulty, or benefit claim in two places.",
            "Every route_plan.route_must_preserve item must remain represented unless it is absent from original_unit and locked_suffix_context.",
            "Avoid route_plan.route_forbidden patterns.",
            "If tutor feedback asks for specificity, use only source wording already present; narrow or connect existing claims instead of adding examples.",
            "Preserve the paragraph's central claims and scope; do not turn a broad overview into a single example or narrower topic.",
            "Every original sentence's central claim must still be represented in the replacement.",
            "Do not merge sentences in a way that drops a source claim, list item group, or qualifying detail.",
            "Source-near specificity means tighter wording and clearer links among existing ideas, not new content.",
            f"Stay near the original length of about {source_words} words; do not compress into a summary or expand with new material.",
            *_generation_scope_constraints(scope),
            f"Return exactly {effective_variant_count} {'variant' if effective_variant_count == 1 else 'variants'}.",
            _structure_instruction(structure),
            "Do not globally grammar-polish the writing. Preserve the writer's undergraduate/student voice unless a wording problem blocks meaning.",
            "Keep existing concrete anchors prominent, especially course names, unit codes, workplace context, and first-person professional-practice context already present in the source.",
            "For opening/background framing, start from an existing concrete anchor when the source supplies one; do not begin with a generic field-wide overview if the paragraph already names the course, unit, or workplace context.",
            "If you move a concrete anchor into the opening, make that anchor the grammatical subject of the sentence. Do not keep a generic 'the importance of...' or 'this is particularly noticeable...' route as the main frame.",
            "When repairing a broad opening, avoid the becoming-increasingly-important/clear/obvious frame; use the existing course or unit to show why the topic matters.",
            "When using an anchor-first opening, still represent the original broad field claim and the original AI/information-era claim; connect them to the anchor instead of deleting them.",
            "For abstract_to_source_anchor, do not write that the topic is important, increasingly important, or matters. Write what the source anchor shows or exposes.",
            "For claim_bridge_repair, do not use a broad different-from-traditional-methods bridge by itself. Tie the bridge to a practical skill problem already in the unit.",
            "Prefer small route and linkage changes over replacing the paragraph with smoother academic phrasing.",
            "JSON safety: preserve curly quotation marks from the source where possible. If you use straight double quotes inside text values, they must be escaped as JSON.",
            "Follow target_voice_profile; do not upgrade the paragraph into a scholarly or professional article voice.",
        ],
        "avoid": [
            *_structure_aware_avoid(repair_brief.avoid, structure),
            "new facts",
            "new examples",
            "personal stories",
            "slang",
            "new headings" if structure.get("has_heading_like_first_line") else "headings",
            "bullets",
            "markdown",
            "HTML",
            "commentary",
            "global grammar polish",
            "professional copyediting voice",
            "generic importance-of opening",
            "becoming increasingly important opening",
            "important because opening",
            "different from traditional methods bridge",
            "this course shows why opening",
        ],
        "output_schema": {
            "variants": [
                {"variant_id": f"v{index}", "text": "..."}
                for index in range(1, effective_variant_count + 1)
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
    if scope.get("mode") == "focused_editable_prefix":
        payload = _compact_focused_payload(payload)
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def generate_variants(
    *,
    group: Any,
    repair_brief: RepairBrief,
    gateway: LLMGateway,
    variant_count: int = 3,
) -> tuple[list[CandidateVariant], dict[str, Any], str, str]:
    source_text = str(getattr(group, "source_text", "") or "")
    scope = _candidate_generation_scope(source_text, repair_brief)
    prompt = build_generator_prompt(group=group, repair_brief=repair_brief, variant_count=variant_count)
    effective_variant_count = _effective_variant_count(variant_count, repair_brief.mitigation_strategy)
    structured_options = structured_json_request_options(
        getattr(gateway, "model", None),
        _variants_response_format(effective_variant_count),
    )
    structured_provider = _merge_provider_options(
        getattr(gateway, "provider", None),
        structured_options.get("provider"),
    )
    max_tokens = _int_env("DRAFTPROOF_REWRITE_V4_GENERATOR_MAX_TOKENS", 8000, minimum=800, maximum=16000)
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with a variants array.",
        response_format=structured_options.get("response_format") or {"type": "json_object"},
        provider=structured_provider,
        temperature=0.35,
        top_p=0.85,
        max_tokens=max_tokens,
    )
    min_words, max_words = word_bounds(scope["editable_text"])
    raw_completion = response.raw_content or response.content
    parsed_variants, diagnostics = parse_generator_variants(
        raw_completion,
        min_words=min_words,
        max_words=max_words,
        source_text=scope["editable_text"],
    )
    variants = _assemble_generation_scope_variants(parsed_variants, scope)
    if (
        not variants
        and diagnostics.get("status") == "json_parse_failed"
        and _bool_env("DRAFTPROOF_REWRITE_V4_REPAIR_MALFORMED_JSON", True)
    ):
        repair_prompt = _json_repair_prompt(
            raw_completion=raw_completion,
            variant_count=effective_variant_count,
        )
        repair_max_tokens = _int_env("DRAFTPROOF_REWRITE_V4_JSON_REPAIR_MAX_TOKENS", 8000, minimum=800, maximum=16000)
        repair_response = gateway.chat(
            repair_prompt,
            system="Return only valid JSON with a variants array.",
            response_format=structured_options.get("response_format") or {"type": "json_object"},
            provider=structured_provider,
            temperature=0.0,
            top_p=0.5,
            max_tokens=repair_max_tokens,
        )
        repair_raw_completion = repair_response.raw_content or repair_response.content
        repaired_parsed_variants, repair_diagnostics = parse_generator_variants(
            repair_raw_completion,
            min_words=min_words,
            max_words=max_words,
            source_text=scope["editable_text"],
        )
        repaired_variants = _assemble_generation_scope_variants(repaired_parsed_variants, scope)
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
                "finish_reason": response.finish_reason,
                "native_finish_reason": response.native_finish_reason,
                "max_tokens": max_tokens,
                "structured_output_mode": structured_options.get("structured_output_mode"),
                "generation_scope": _scope_diagnostics(scope),
                "repair_finish_reason": repair_response.finish_reason,
                "repair_native_finish_reason": repair_response.native_finish_reason,
                "repair_max_tokens": repair_max_tokens,
            }, prompt, raw_completion
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
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "max_tokens": max_tokens,
        "structured_output_mode": structured_options.get("structured_output_mode"),
        "generation_scope": _scope_diagnostics(scope),
    }, prompt, raw_completion


def _candidate_generation_scope(source_text: str, repair_brief: RepairBrief) -> dict[str, str]:
    text = str(source_text or "")
    threshold = _int_env("DRAFTPROOF_REWRITE_V4_FOCUSED_PREFIX_WORD_THRESHOLD", 320, minimum=120, maximum=900)
    if word_count(text) < threshold:
        return {"mode": "full_unit", "editable_text": text, "locked_suffix": ""}
    structure = _structure_contract(text)
    if int(structure.get("nonempty_line_count") or 1) <= 1:
        return {"mode": "full_unit", "editable_text": text, "locked_suffix": ""}
    boundary = _first_blank_line_boundary(text)
    if boundary <= 0:
        return {"mode": "full_unit", "editable_text": text, "locked_suffix": ""}
    editable = text[:boundary].rstrip()
    suffix = text[boundary:]
    if word_count(editable) < 40 or word_count(suffix) < 40:
        return {"mode": "full_unit", "editable_text": text, "locked_suffix": ""}
    return {
        "mode": "focused_editable_prefix",
        "editable_text": editable,
        "locked_suffix": suffix,
    }


def _first_blank_line_boundary(text: str) -> int:
    offset = 0
    previous_line_ending = 0
    for line in str(text or "").splitlines(keepends=True):
        if not line.strip():
            return max(0, offset - previous_line_ending)
        if line.endswith("\r\n"):
            previous_line_ending = 2
        elif line.endswith("\n") or line.endswith("\r"):
            previous_line_ending = 1
        else:
            previous_line_ending = 0
        offset += len(line)
    return -1


def _generation_scope_constraints(scope: dict[str, str]) -> list[str]:
    if not scope.get("locked_suffix"):
        return []
    return [
        "Return only a replacement for original_unit, not the locked suffix.",
        "Do not include locked_suffix_context in the returned text; the system will append that unchanged after validation.",
        "Use locked_suffix_context only to keep the transition natural.",
    ]


def _assemble_generation_scope_variants(variants: list[CandidateVariant], scope: dict[str, str]) -> list[CandidateVariant]:
    suffix = scope.get("locked_suffix") or ""
    if not suffix:
        return variants
    assembled: list[CandidateVariant] = []
    for variant in variants:
        text = variant.text.rstrip() + suffix
        assembled.append(CandidateVariant(
            variant_id=variant.variant_id,
            text=text,
            word_count=word_count(text),
        ))
    return assembled


def _scope_diagnostics(scope: dict[str, str]) -> dict[str, Any]:
    return {
        "mode": scope.get("mode"),
        "editable_words": word_count(scope.get("editable_text") or ""),
        "locked_suffix_words": word_count(scope.get("locked_suffix") or ""),
    }


def _compact_focused_payload(payload: dict[str, Any]) -> dict[str, Any]:
    strategy = payload.get("mitigation_strategy") if isinstance(payload.get("mitigation_strategy"), dict) else {}
    steps = _compact_strategy_steps(payload.get("rewrite_sequence"))
    compact_strategy = {
        key: strategy.get(key)
        for key in ("scope", "strategy_id", "primary_problem", "rewrite_depth", "candidate_count_hint")
        if strategy.get(key)
    }
    for key in ("required_moves", "forbidden_moves", "must_preserve_claims", "success_checks"):
        rows = strategy.get(key)
        if isinstance(rows, list) and rows:
            compact_strategy[key] = rows[:4]
    route_plan = _route_plan(strategy)
    if steps:
        compact_strategy["strategy_steps"] = steps

    constraints = _unique_strings([
        "Return only the replacement text for original_unit; do not include locked_suffix_context.",
        "The system appends locked_suffix_context unchanged after validation.",
        "Execute route_plan.better_route as the route; do not keep the old current_route order unless a claim would otherwise be lost.",
        "Apply route_plan.route_moves using only original_unit and locked_suffix_context material.",
        "Route moves are moves, not copies: if a claim is moved earlier, remove its old repeated occurrence.",
        "Do not repeat the same course, unit, difficulty, or benefit claim in two places.",
        "Every route_plan.route_must_preserve item must remain represented.",
        "Avoid route_plan.route_forbidden patterns.",
        "Preserve meaning, citations, anchors, paragraph role, and source viewpoint.",
        "Do not add new facts, examples, names, dates, numbers, headings, bullets, markdown, HTML, or commentary.",
        "Keep the same line-break structure inside original_unit.",
        "Keep close to the original_unit length; do not compress into a summary.",
        "Follow mitigation_strategy and rewrite_sequence; skip only a step whose target is absent.",
        "Preserve the writer's undergraduate/student voice; do not upgrade into a professional article voice.",
        *[str(item) for item in payload.get("constraints") or [] if str(item).strip()][:8],
    ], limit=14)
    avoid = _unique_strings([
        *[str(item) for item in payload.get("avoid") or [] if str(item).strip()],
        "notes",
        "comments",
        "debug text",
    ], limit=14)
    tutor_feedback = payload.get("tutor_feedback") if isinstance(payload.get("tutor_feedback"), dict) else {}
    compact_feedback = {
        "diagnosis": tutor_feedback.get("diagnosis", ""),
        "repair_assignment": tutor_feedback.get("repair_assignment", ""),
    }
    return {
        "task": payload.get("task"),
        "repair_mode": payload.get("repair_mode"),
        "paragraph_role": payload.get("paragraph_role"),
        "unit_kind": payload.get("unit_kind"),
        "generation_scope": payload.get("generation_scope"),
        "target_voice_profile": {
            "education_level": (payload.get("target_voice_profile") or {}).get("education_level"),
            "voice_register": (payload.get("target_voice_profile") or {}).get("voice_register"),
            "tone": (payload.get("target_voice_profile") or {}).get("tone"),
        },
        "original_unit": payload.get("original_unit"),
        "mitigation_strategy": compact_strategy,
        "route_plan": route_plan,
        "rewrite_sequence": steps,
        "tutor_feedback": compact_feedback,
        "repair_tasks": [str(item) for item in payload.get("repair_tasks") or [] if str(item).strip()][:3],
        "constraints": constraints,
        "avoid": avoid,
        "output_schema": payload.get("output_schema"),
    }


def _compact_strategy_steps(value: Any) -> list[dict[str, Any]]:
    steps = value if isinstance(value, list) else []
    cleaned: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        op = str(step.get("op") or "")
        if op in {"reaction_reason_link", "list_to_process_repair"}:
            continue
        cleaned.append({
            "op": op,
            "target": str(step.get("target") or ""),
            "instruction": str(step.get("instruction") or ""),
            "must_preserve": [str(item) for item in step.get("must_preserve") or [] if str(item).strip()][:3],
            "avoid": [str(item) for item in step.get("avoid") or [] if str(item).strip()][:3],
        })
        if len(cleaned) >= 3:
            break
    return cleaned


def _route_plan(strategy: dict[str, Any]) -> dict[str, list[str]]:
    if not isinstance(strategy, dict):
        return {}
    keys = ("current_route", "better_route", "route_moves", "route_must_preserve", "route_forbidden")
    plan: dict[str, list[str]] = {}
    for key in keys:
        rows = strategy.get(key)
        if not isinstance(rows, list):
            continue
        cleaned = [str(item).strip() for item in rows if str(item).strip()][:6]
        if cleaned:
            plan[key] = cleaned
    return plan


def _unique_strings(items: list[str], *, limit: int) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item or "").split())
        key = text.casefold()
        if not text or key in seen:
            continue
        rows.append(text)
        seen.add(key)
        if len(rows) >= limit:
            break
    return rows


def _variants_response_format(variant_count: int) -> dict[str, Any]:
    variant_schema = {
        "type": "object",
        "properties": {
            "variant_id": {
                "type": "string",
                "description": "Variant id matching the requested variant slot, such as v1.",
            },
            "text": {
                "type": "string",
                "description": "Replacement unit text only, with no notes or commentary.",
            },
        },
        "required": ["variant_id", "text"],
        "additionalProperties": False,
    }
    count = max(1, int(variant_count or 1))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rewrite_v4_variants",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variants": {
                        "type": "array",
                        "items": variant_schema,
                        "minItems": count,
                        "maxItems": count,
                    },
                },
                "required": ["variants"],
                "additionalProperties": False,
            },
        },
    }


def _merge_provider_options(base: Any, override: Any) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    return merged or None


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


def _effective_variant_count(variant_count: int, strategy: dict[str, Any]) -> int:
    requested = max(1, int(variant_count))
    if not isinstance(strategy, dict):
        return requested
    try:
        hinted = int(str(strategy.get("candidate_count_hint") or "").strip())
    except ValueError:
        return requested
    return max(1, min(requested, hinted))


def _strategy_controller_constraints(strategy: dict[str, Any]) -> list[str]:
    if not isinstance(strategy, dict) or not strategy:
        return []
    return [
            "Follow mitigation_strategy as the repair controller; do not treat repair_tasks as independent unrelated edits.",
            "Apply rewrite_sequence in order where the target text exists; skip a step only if its target is absent.",
            "For each rewrite_sequence step, preserve its must_preserve items and avoid its avoid items.",
            "Reject your own draft mentally if a rewrite_sequence step deletes a source claim or source block.",
        ]


def _strategy_steps(strategy: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(strategy, dict):
        return []
    steps = strategy.get("strategy_steps")
    if not isinstance(steps, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in steps[:5]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or "").strip()
        target = str(item.get("target") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        if not op or not target or not instruction:
            continue
        cleaned.append({
            "op": op,
            "target": target,
            "instruction": instruction,
            "must_preserve": [str(row).strip() for row in item.get("must_preserve") or [] if str(row).strip()][:4],
            "avoid": [str(row).strip() for row in item.get("avoid") or [] if str(row).strip()][:4],
        })
    return cleaned


def _structure_aware_constraints(items: tuple[str, ...] | list[str], structure: dict[str, Any]) -> list[str]:
    preserve_blocks = int(structure.get("nonempty_line_count") or 1) > 1
    cleaned: list[str] = []
    for item in items or ():
        text = str(item or "").strip()
        if preserve_blocks and text == "Keep one paragraph.":
            continue
        if text:
            cleaned.append(text)
    return cleaned


def _structure_aware_avoid(items: tuple[str, ...] | list[str], structure: dict[str, Any]) -> list[str]:
    has_source_heading = bool(structure.get("has_heading_like_first_line"))
    cleaned: list[str] = []
    for item in items or ():
        text = str(item or "").strip()
        if has_source_heading and text == "headings":
            text = "new headings"
        if text:
            cleaned.append(text)
    return cleaned


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
