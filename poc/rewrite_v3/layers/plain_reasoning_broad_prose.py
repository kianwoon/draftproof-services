"""Plain-reasoning broad prose layer for V3."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.compression_policy import CompressionPolicy
from rewrite_v3.document_units import compact_document_inventory, structural_shape_contract, word_count
from rewrite_v3.prompt_contract import profile_action_contracts


FAMILY = "plain_reasoning_broad_prose"


def _limit_text(text: Any, limit: int = 420) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    clipped = value[: max(0, limit - 1)].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."


def _compact_target_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    payload = profile if isinstance(profile, dict) else {}
    targets = []
    for target in payload.get("targets") or []:
        if not isinstance(target, dict):
            continue
        targets.append({
            "target_id": target.get("target_id"),
            "unit_id": target.get("unit_id"),
            "scope_level": target.get("scope_level"),
            "risk_level": target.get("risk_level"),
            "dominant_drivers": list(target.get("dominant_drivers") or [])[:3],
            "required_movement": target.get("required_movement") or {},
            "recommended_operation": target.get("recommended_operation"),
            "source_excerpt": _limit_text(target.get("source_excerpt") or target.get("source_text"), 180),
            "word_count_guide": target.get("word_count_guide") or {},
        })
        if len(targets) >= 4:
            break
    return {
        "schema_version": payload.get("schema_version"),
        "document_shape": payload.get("document_shape"),
        "target_scope_policy": payload.get("target_scope_policy"),
        "target_count": len(payload.get("targets") or []),
        "driver_summary": payload.get("driver_summary") or {},
        "operation_mix": payload.get("operation_mix") or {},
        "targets": targets,
    }


def _compact_central_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    payload = plan if isinstance(plan, dict) else {}
    objective = payload.get("generation_objective") if isinstance(payload.get("generation_objective"), dict) else {}
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    return {
        "strategy_id": payload.get("strategy_id"),
        "generation_objective": {
            "weights": objective.get("weights") or {},
            "primary_drivers": list(objective.get("primary_drivers") or [])[:6],
            "target": objective.get("target"),
        },
        "constraints": {
            "source_word_count": constraints.get("source_word_count"),
            "minimum_words": constraints.get("minimum_words"),
            "preferred_sentence_count": constraints.get("preferred_sentence_count"),
            "do_not_add_unsupported_facts": constraints.get("do_not_add_unsupported_facts"),
        },
        "content_operations": list(payload.get("content_operations") or [])[:4],
    }


def _compact_failed_candidates(candidates: list[str]) -> list[dict[str, Any]]:
    compact = []
    for index, candidate in enumerate(candidates or [], start=1):
        if not str(candidate or "").strip():
            continue
        compact.append({
            "candidate_index": index,
            "word_count": word_count(candidate),
            "excerpt": _limit_text(candidate, 220),
        })
        if len(compact) >= 2:
            break
    return compact


def _compact_style_examples(examples: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for example in examples or []:
        if not isinstance(example, dict):
            continue
        compact.append({
            "external_ai_percent": example.get("external_ai_percent"),
            "label": example.get("label") or example.get("source"),
            "excerpt": _limit_text(example.get("text") or example.get("content"), 180),
        })
        if len(compact) >= 2:
            break
    return compact


def build_plain_reasoning_broad_prose_prompt(
    *,
    original_text: str,
    failed_candidates: list[str],
    compression_policy: CompressionPolicy,
    style_examples: dict[str, list[dict[str, Any]]],
    rewrite_target_profile: dict[str, Any] | None = None,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    central_judgment_plan: dict[str, Any] | None = None,
) -> str:
    payload = {
        "source_excerpt": _limit_text(original_text, 600),
        "source_word_count": word_count(original_text),
        "document_inventory": compact_document_inventory(original_text, max_units=10, preview_chars=160),
        "source_structure_contract": structural_shape_contract(original_text),
        "rewrite_target_profile_summary": _compact_target_profile(rewrite_target_profile),
        "scanner_action_contracts": profile_action_contracts(
            rewrite_target_profile=rewrite_target_profile,
            predictability_briefs=predictability_briefs,
            max_contracts=3,
            compact=True,
        ),
        "central_judgment_plan_summary": _compact_central_plan(central_judgment_plan),
        "failed_candidate_summaries": _compact_failed_candidates(failed_candidates),
        "positive_external_boundaries": _compact_style_examples(style_examples.get("positive")),
        "negative_external_boundaries": _compact_style_examples(style_examples.get("negative")),
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "requirements": [
            "Rewrite as broad prose with plain reasoning, not as a formal survey.",
            "Preserve source paragraph order and source paragraph count.",
            "Preserve source_structure_contract exactly: same block_count, same blank_line_boundary_count, and same heading_like_lines.",
            "Do not add blank-line paragraph splits or merge source blocks.",
            "Preserve the source meaning, factual claims, entities, and examples.",
            "Represent the source document inventory; do not collapse the document into a summary.",
            "Use rewrite_target_profile_summary as scanner-derived repair guidance.",
            "Use scanner_action_contracts for exact target operations and predictable spans when present.",
            "Use scanner_action_contracts.ownership_contract when present: do not just change point of view; add source-supported author trace, specific context, and real judgment.",
            "Use direct language and human judgment lines where the source supports them.",
            "Avoid textbook openings, balanced report phrasing, and polished summary transitions.",
            "Do not add unsupported facts, numbers, names, examples, headings, bullets, labels, or markdown.",
            "Aim near the preferred word count, not the minimum word count.",
            "Return only the rewritten document as plain text.",
        ],
    }
    return (
        "Rewrite this broad prose document using V3 plain-reasoning style.\n"
        "The goal is a natural argument/overview that keeps the source facts but avoids formal generated-survey texture.\n"
        "Learn from positive and negative external boundaries without copying them.\n"
        "Return only plain text, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
