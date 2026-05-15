"""Broad AI-assisted footprint executor for rewrite V3.

This layer is intentionally separate from the target executor. Target executor
handles high-risk scanner targets; this module handles the broader assisted
window footprint that remains when most of the document is still labelled as
AI-assisted.
"""

from __future__ import annotations

import json
from typing import Any

from .prompt_contract import group_action_contract
from .target_executor import (
    TargetGroup,
    _looks_like_structural_label,
    _paragraph_spans,
    apply_target_replacements,
    batch_target_groups,
    parse_target_replacements,
    target_execution_trace,
)


ASSISTED_FOOTPRINT_LABELS = {
    "ai_generated",
    "moderately_ai_assisted",
    "lightly_ai_assisted",
}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _target_anchors_by_unit(rewrite_target_profile: dict[str, Any] | None) -> dict[str, tuple[dict[str, Any], ...]]:
    profile = rewrite_target_profile if isinstance(rewrite_target_profile, dict) else {}
    anchors_by_unit: dict[str, list[dict[str, Any]]] = {}
    seen_by_unit: dict[str, set[str]] = {}
    for target in profile.get("targets") or []:
        if not isinstance(target, dict):
            continue
        unit_id = str(target.get("unit_id") or target.get("paragraph_id") or "")
        if not unit_id:
            continue
        anchors_by_unit.setdefault(unit_id, [])
        seen_by_unit.setdefault(unit_id, set())
        for anchor in target.get("protected_anchors") or []:
            if not isinstance(anchor, dict):
                continue
            text = str(anchor.get("text") or "").strip()
            if not text or text in seen_by_unit[unit_id]:
                continue
            seen_by_unit[unit_id].add(text)
            anchors_by_unit[unit_id].append(dict(anchor))
    return {unit: tuple(anchors) for unit, anchors in anchors_by_unit.items()}


def group_assisted_footprint_windows(
    *,
    original_text: str,
    authorship_window_profile: dict[str, Any] | None,
    rewrite_target_profile: dict[str, Any] | None = None,
    max_groups: int = 10,
    context_chars: int = 420,
) -> list[TargetGroup]:
    profile = authorship_window_profile if isinstance(authorship_window_profile, dict) else {}
    windows = [
        window for window in profile.get("windows") or []
        if isinstance(window, dict)
        and str(window.get("label") or "") in ASSISTED_FOOTPRINT_LABELS
        and str(window.get("paragraph_id") or "")
    ]
    paragraph_spans = _paragraph_spans(original_text)
    anchor_map = _target_anchors_by_unit(rewrite_target_profile)
    by_paragraph: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        paragraph_id = str(window.get("paragraph_id") or "")
        if paragraph_id not in paragraph_spans:
            continue
        by_paragraph.setdefault(paragraph_id, []).append(window)

    ranked = sorted(
        by_paragraph.items(),
        key=lambda item: (
            max(_number(window.get("ai_assistance_score")) for window in item[1]),
            sum(_number(window.get("word_count")) for window in item[1]),
        ),
        reverse=True,
    )
    text = str(original_text or "")
    groups: list[TargetGroup] = []
    for index, (paragraph_id, rows) in enumerate(ranked, start=1):
        start, end = paragraph_spans[paragraph_id]
        source = text[start:end].strip()
        if _looks_like_structural_label(source):
            continue
        before = text[max(0, start - context_chars):start]
        after = text[end:min(len(text), end + context_chars)]
        preferred_words = _word_count(source)
        group_targets = tuple({
            "target_id": str(window.get("window_id") or f"aw{index:03d}"),
            "risk_level": "assisted",
            "dominant_drivers": [
                {
                    "key": key,
                    "score": round(_number(value), 3),
                    "source": "authorship_window_profile.score_components",
                }
                for key, value in ((window.get("score_components") or {}).items())
                if _number(value) > 0
            ],
            "required_movement": {
                "ai_assistance_score_drop": 0.08,
                "predictability_drop": 0.10,
                "human_signal_gain": 0.12,
            },
            "rewrite_constraints": {
                "preserve_unit_meaning": True,
                "avoid_compressed_summary": True,
                "avoid_generic_academic_smoothing": True,
            },
        } for window in rows[:3])
        groups.append(TargetGroup(
            group_id=f"af{len(groups) + 1:03d}",
            unit_id=paragraph_id,
            operation="assisted_footprint_paragraph_rewrite",
            start_index=start,
            end_index=end,
            source_text=source,
            before_context=before,
            after_context=after,
            targets=group_targets,
            protected_anchors=anchor_map.get(paragraph_id, ()),
            word_count_guide={
                "source_words": preferred_words,
                "preferred_words": preferred_words,
            },
        ))
        if len(groups) >= max(1, int(max_groups or 1)):
            break
    return groups


def build_assisted_footprint_prompt(
    *,
    target_groups: list[TargetGroup],
    content_mode: str,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> str:
    groups = []
    for group in target_groups:
        groups.append({
            "group_id": group.group_id,
            "unit_id": group.unit_id,
            "operation": group.operation,
            "source_text": group.source_text,
            "before_context": group.before_context,
            "after_context": group.after_context,
            "protected_anchors": list(group.protected_anchors),
            "word_count_guide": dict(group.word_count_guide),
            "window_drivers": [target.get("dominant_drivers") for target in group.targets],
            "scanner_action_contract": group_action_contract(
                group=group,
                predictability_briefs=predictability_briefs,
            ),
        })
    payload = {
        "content_mode": content_mode,
        "repair_scope": "assisted_footprint_paragraph_groups",
        "target_groups": groups,
        "requirements": [
            "Return JSON only with a replacements array.",
            "Rewrite each source_text as a whole local paragraph.",
            "Do not rewrite the full document.",
            "Preserve protected_anchors exactly when present.",
            "Use scanner_action_contract as the execution contract for each group.",
            "Use scanner_action_contract.ownership_contract as the authorship ownership contract for each group.",
            "Do not solve ownership by changing point of view only; add source-supported author trace, specific context, and real judgment.",
            "Preserve source viewpoint unless source_text or nearby context already supports author experience, action, observation, or decision.",
            "Patch scanner_action_contract.topk_repair_contract.predictable_spans_in_source when span_source is scanner_exact.",
            "Use raw/rejected predictable spans only as diagnostics; count movement only on valid phrase spans.",
            "Modify at least scanner_action_contract.topk_repair_contract.required_modified_spans phrase spans when span_source is scanner_exact.",
            "Stay inside scanner_action_contract.topk_repair_contract.locality_limits for local predictability repair.",
            "Keep the same claims, citations, codes, names, and paragraph role.",
            "Do not summarize or compress the paragraph.",
            "Do not do synonym swapping or clean paraphrase.",
            "Do not simply reverse sentence order.",
            "Avoid abstract academic openings when the source can start with the concrete classroom issue, learner action, or teaching decision.",
            "Use plainer connective movement: a short setup, then the specific consequence, then the reason or citation when needed.",
            "Change the paragraph's internal movement: split or combine sentences, vary sentence length, move one concrete cause or consequence into its own sentence, and make one relation explicit when supported by source_text.",
            "For practice-based education text, sound like a teacher explaining what happens in class, not a journal abstract.",
            "Use word_count_guide as preferred length only.",
            "Return no markdown, labels, commentary, or bullets inside replacement_text.",
        ],
        "response_schema": {
            "replacements": [
                {
                    "group_id": "af001",
                    "replacement_text": "replacement paragraph only"
                }
            ]
        },
    }
    return (
        "Repair the broad AI-assisted footprint in selected paragraphs.\n"
        "The scanner has already selected the paragraphs. Change paragraph movement, not just wording.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def apply_assisted_footprint_replacements(
    *,
    original_text: str,
    target_groups: list[TargetGroup],
    replacements: list[dict[str, str]],
) -> tuple[str, list[dict[str, Any]]]:
    return apply_target_replacements(
        original_text=original_text,
        target_groups=target_groups,
        replacements=replacements,
    )


__all__ = [
    "ASSISTED_FOOTPRINT_LABELS",
    "apply_assisted_footprint_replacements",
    "batch_target_groups",
    "build_assisted_footprint_prompt",
    "group_assisted_footprint_windows",
    "parse_target_replacements",
    "target_execution_trace",
]
