"""Unit-preserving prune/bridge strategy for V3 problem groups."""

from __future__ import annotations

import json
from typing import Any

from .target_executor import TargetGroup, apply_target_replacements, target_execution_trace


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _group_rank(group: TargetGroup) -> tuple[float, float, float]:
    level_weight = {"high": 3.0, "medium": 2.0, "low": 1.0, "minimal": 0.25}
    risk = 0.0
    driver_score = 0.0
    source_words = 0.0
    for target in group.targets:
        risk = max(risk, level_weight.get(str(target.get("risk_level") or "minimal"), 0.25))
        source_words = max(source_words, _number((target.get("word_count_guide") or {}).get("source_words")))
        for driver in target.get("dominant_drivers") or []:
            if isinstance(driver, dict):
                driver_score += min(_number(driver.get("score")), 1.0)
    return (risk, driver_score, source_words)


def _target_ids_from_problem_groups(problem_groups: list[dict[str, Any]]) -> set[str]:
    target_ids: set[str] = set()
    for group in problem_groups:
        if not isinstance(group, dict):
            continue
        if "unit_preserving_prune_bridge" not in {str(item) for item in group.get("allowed_operations") or []}:
            continue
        target_ids.update(str(item) for item in group.get("target_ids") or [] if str(item))
    return target_ids


def filter_prune_bridge_groups(
    *,
    target_groups: list[TargetGroup],
    problem_inventory: dict[str, Any] | None,
) -> list[TargetGroup]:
    inventory = problem_inventory if isinstance(problem_inventory, dict) else {}
    problem_groups = inventory.get("problem_groups") if isinstance(inventory.get("problem_groups"), list) else []
    eligible_target_ids = _target_ids_from_problem_groups(problem_groups)
    if not eligible_target_ids:
        return []
    filtered: list[TargetGroup] = []
    for group in target_groups:
        targets = tuple(
            target for target in group.targets
            if str(target.get("target_id") or "") in eligible_target_ids
        )
        if not targets:
            continue
        filtered.append(TargetGroup(
            group_id=group.group_id,
            unit_id=group.unit_id,
            operation="unit_preserving_prune_bridge",
            start_index=group.start_index,
            end_index=group.end_index,
            source_text=group.source_text,
            before_context=group.before_context,
            after_context=group.after_context,
            targets=targets,
            protected_anchors=group.protected_anchors,
            word_count_guide=group.word_count_guide,
        ))
    return sorted(filtered, key=_group_rank, reverse=True)


def build_prune_bridge_prompt(*, target_groups: list[TargetGroup]) -> str:
    payload = {
        "strategy": "unit_preserving_prune_bridge",
        "repair_scope": "scanner_problem_groups_only",
        "target_groups": [
            {
                "group_id": group.group_id,
                "unit_id": group.unit_id,
                "source_text": group.source_text,
                "before_context": group.before_context[-320:],
                "after_context": group.after_context[:320],
                "protected_anchors": list(group.protected_anchors),
                "word_count_guide": dict(group.word_count_guide),
                "preferred_reduction": {
                    "source_words": group.word_count_guide.get("source_words"),
                    "preferred_words": group.word_count_guide.get("preferred_words"),
                    "instruction": "Use fewer words than source_text unless a protected anchor makes that impossible.",
                },
                "targets": [
                    {
                        "target_id": target.get("target_id"),
                        "source_text": target.get("source_text"),
                        "risk_level": target.get("risk_level"),
                        "dominant_drivers": target.get("dominant_drivers") or [],
                        "required_movement": target.get("required_movement") or {},
                    }
                    for target in group.targets
                ],
            }
            for group in target_groups
        ],
        "requirements": [
            "Return JSON only with a replacements array.",
            "Return one replacement per group_id.",
            "Replace only the source_text for each target group.",
            "Do not paraphrase every sentence.",
            "Remove, shorten, or bridge low-value risky wording when nearby text already carries the meaning.",
            "For this strategy, replacement_text should usually be materially shorter than source_text.",
            "If replacement_text is about the same length as source_text, it is probably a failed prune/bridge.",
            "If the source_text contains a heading plus prose, preserve the heading and revise only the prose under it.",
            "If a standalone source_text would become empty, keep one short bridge sentence so the document unit remains present.",
            "Preserve protected anchors exactly when present.",
            "Do not add facts, citations, dates, names, numbers, examples, bullets, labels, markdown, or commentary.",
            "Use plain local wording; avoid polished summary style.",
        ],
        "response_schema": {
            "replacements": [
                {"group_id": "tg001", "replacement_text": "replacement prose only"}
            ]
        },
    }
    return "Execute the scanner-selected V3 prune/bridge strategy.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def parse_prune_bridge_replacements(raw: str) -> list[dict[str, str]]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    rows = payload.get("replacements") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    replacements: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        group_id = str(row.get("group_id") or "").strip()
        replacement = str(row.get("replacement_text") or "").strip()
        if group_id and replacement:
            replacements.append({"group_id": group_id, "replacement_text": replacement})
    return replacements


def apply_prune_bridge_replacements(
    *,
    original_text: str,
    target_groups: list[TargetGroup],
    replacements: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    text, apply_status = apply_target_replacements(
        original_text=original_text,
        target_groups=target_groups,
        replacements=replacements,
    )
    trace = target_execution_trace(
        attempted=True,
        target_groups=target_groups,
        replacements=replacements,
        apply_status=apply_status,
        batches=[],
        error=None if any(row.get("applied") for row in apply_status) else "no_prune_bridge_replacement_applied",
    )
    return text, {**trace, "problem_strategy": "unit_preserving_prune_bridge"}


__all__ = [
    "apply_prune_bridge_replacements",
    "build_prune_bridge_prompt",
    "filter_prune_bridge_groups",
    "parse_prune_bridge_replacements",
]
