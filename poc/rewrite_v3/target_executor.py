"""Scanner-target executor for rewrite V3.

This layer turns ``rewrite_target_profile.v1`` into bounded replacement
requests. It routes from scanner-provided target fields only: spans, ids,
drivers, operations, anchors, and word guides.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from .prompt_contract import group_action_contract


SUPPORTED_TARGET_OPERATIONS = {
    "protected_section_rewrite",
    "citation_preserving_window_repair",
    "grounded_author_reasoning_rewrite",
    "light_texture_rewrite",
    "paragraph_preserving_broad_reconstruction",
}


@dataclass(frozen=True)
class TargetGroup:
    group_id: str
    unit_id: str
    operation: str
    start_index: int | None
    end_index: int | None
    source_text: str
    before_context: str
    after_context: str
    targets: tuple[dict[str, Any], ...]
    protected_anchors: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    soft_guidance_anchors: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    word_count_guide: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetApplyStatus:
    group_id: str
    applied: bool
    method: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _anchor_compare_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return (
        normalized
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )


def _anchor_present(text: str, anchor_text: str) -> bool:
    if not anchor_text:
        return True
    normalized_text = _anchor_compare_text(text)
    normalized_anchor = _anchor_compare_text(anchor_text)
    if normalized_anchor.isdigit():
        search_from = 0
        while True:
            start = normalized_text.find(normalized_anchor, search_from)
            if start < 0:
                return False
            end = start + len(normalized_anchor)
            before = normalized_text[start - 1] if start > 0 else ""
            after = normalized_text[end] if end < len(normalized_text) else ""
            if not before.isdigit() and not after.isdigit():
                return True
            search_from = start + 1
    return normalized_anchor in normalized_text


def required_protected_anchors_for_source(source_text: str, protected_anchors: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    required: list[dict[str, Any]] = []
    for anchor in protected_anchors or ():
        if not isinstance(anchor, dict):
            continue
        anchor_text = str(anchor.get("text") or "").strip()
        if anchor_text and _anchor_present(source_text, anchor_text):
            required.append(anchor)
    return tuple(required)


def required_protected_anchors_for_group(group: TargetGroup) -> tuple[dict[str, Any], ...]:
    return required_protected_anchors_for_source(group.source_text, group.protected_anchors)


def missing_required_protected_anchors(text: str, group: TargetGroup) -> list[str]:
    missing: list[str] = []
    for anchor in required_protected_anchors_for_group(group):
        anchor_text = str(anchor.get("text") or "").strip()
        if anchor_text and not _anchor_present(text, anchor_text):
            missing.append(anchor_text)
    return missing


def _span_from_target(target: dict[str, Any]) -> tuple[int | None, int | None, bool]:
    span = target.get("span") if isinstance(target.get("span"), dict) else {}
    integrity = span.get("integrity") if isinstance(span.get("integrity"), dict) else {}
    start = span.get("start_index")
    end = span.get("end_index")
    if start is None:
        start = integrity.get("start_index")
    if end is None:
        end = integrity.get("end_index")
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None, None, False
    return start_i, end_i, bool(integrity.get("passed"))


def _span_in_bounds(span: tuple[int | None, int | None, bool], text_length: int) -> bool:
    start, end, _ = span
    return start is not None and end is not None and 0 <= start < end <= text_length


def _target_sort_key(target: dict[str, Any]) -> tuple[int, str]:
    start, _, _ = _span_from_target(target)
    return (start if start is not None else 10**12, str(target.get("target_id") or ""))


def _group_key(target: dict[str, Any]) -> str:
    return str(
        target.get("unit_id")
        or target.get("paragraph_id")
        or target.get("window_id")
        or target.get("target_id")
        or ""
    )


def _paragraph_spans(text: str) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    block_start: int | None = None
    position = 0
    index = 0
    for line in str(text or "").splitlines(keepends=True):
        is_blank = not line.strip()
        if is_blank:
            if block_start is not None:
                index += 1
                spans[f"p{index:03d}"] = (block_start, position)
                block_start = None
            position += len(line)
            continue
        if block_start is None:
            block_start = position
        position += len(line)
    if block_start is not None:
        index += 1
        spans[f"p{index:03d}"] = (block_start, position)
    return spans


def _snap_span_to_paragraph_boundaries(
    start: int,
    end: int,
    paragraph_spans: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    intersecting = [
        (span_start, span_end)
        for span_start, span_end in sorted(paragraph_spans.values())
        if not (span_end <= start or span_start >= end)
    ]
    if not intersecting:
        return start, end
    return intersecting[0][0], intersecting[-1][1]


def _target_uses_chunk_scope(target: dict[str, Any]) -> bool:
    operations = target.get("operation_candidates") if isinstance(target.get("operation_candidates"), list) else []
    return (
        str(target.get("scope_level") or "") == "chunk"
        or str(target.get("recommended_operation") or "") == "chunk_reconstruction"
        or "chunk_reconstruction" in {str(item) for item in operations}
    )


def _target_operation(targets: list[dict[str, Any]]) -> str:
    for target in targets:
        operation = str(target.get("recommended_operation") or "")
        if operation in SUPPORTED_TARGET_OPERATIONS:
            return operation
    return "light_texture_rewrite"


def _dedupe_anchors(targets: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    seen: set[str] = set()
    anchors: list[dict[str, Any]] = []
    for target in targets:
        for anchor in target.get("protected_anchors") or []:
            if not isinstance(anchor, dict):
                continue
            text = str(anchor.get("text") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            anchors.append(dict(anchor))
    return tuple(anchors)


def _dedupe_soft_guidance_anchors(targets: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    seen: set[str] = set()
    anchors: list[dict[str, Any]] = []
    for target in targets:
        for anchor in target.get("soft_guidance_anchors") or []:
            if not isinstance(anchor, dict):
                continue
            text = str(anchor.get("text") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            anchors.append(dict(anchor))
    return tuple(anchors)


def _anchors_present_in_source(anchors: tuple[dict[str, Any], ...], source_text: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        anchor for anchor in anchors
        if _anchor_present(source_text, str(anchor.get("text") or "").strip())
    )


def _combined_word_guide(targets: list[dict[str, Any]], source_text: str) -> dict[str, Any]:
    actual_words = _word_count(source_text)
    return {
        "source_words": actual_words,
        "preferred_words": actual_words,
    }


def _looks_like_structural_label(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if _word_count(stripped) > 8:
        return False
    return stripped[-1] not in ".?!)]"


def group_rewrite_targets(
    *,
    original_text: str,
    rewrite_target_profile: dict[str, Any] | None,
    max_groups: int = 4,
    context_chars: int = 420,
) -> list[TargetGroup]:
    profile = rewrite_target_profile if isinstance(rewrite_target_profile, dict) else {}
    targets = [
        target for target in profile.get("targets") or []
        if isinstance(target, dict)
        and str(target.get("recommended_operation") or "") in SUPPORTED_TARGET_OPERATIONS
        and str(target.get("source_text") or "").strip()
    ]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for target in sorted(targets, key=_target_sort_key):
        key = _group_key(target)
        if not key:
            key = str(target.get("target_id") or len(buckets) + 1)
        buckets.setdefault(key, []).append(target)

    text = str(original_text or "")
    paragraph_spans = _paragraph_spans(text)
    groups: list[TargetGroup] = []
    for index, (unit_id, rows) in enumerate(buckets.items(), start=1):
        spans = [_span_from_target(target) for target in rows]
        bounded_spans = [(start, end, passed) for start, end, passed in spans if _span_in_bounds((start, end, passed), len(text))]
        valid_spans = [(start, end) for start, end, passed in bounded_spans if passed]
        paragraph_id = str(rows[0].get("paragraph_id") or "")
        paragraph_span = paragraph_spans.get(paragraph_id)
        uses_chunk_scope = any(_target_uses_chunk_scope(target) for target in rows)
        if uses_chunk_scope and bounded_spans:
            raw_start = min(start for start, _, _ in bounded_spans)
            raw_end = max(end for _, end, _ in bounded_spans)
            start_i, end_i = _snap_span_to_paragraph_boundaries(raw_start, raw_end, paragraph_spans)
            source = text[start_i:end_i].strip()
        elif paragraph_span is not None:
            start_i, end_i = paragraph_span
            source = text[start_i:end_i].strip()
        elif valid_spans:
            start_i = min(start for start, _ in valid_spans)
            end_i = max(end for _, end in valid_spans)
            source = text[start_i:end_i].strip()
        elif bounded_spans:
            start_i = min(start for start, _, _ in bounded_spans)
            end_i = max(end for _, end, _ in bounded_spans)
            source = text[start_i:end_i].strip()
        else:
            start_i = None
            end_i = None
            source = "\n\n".join(str(target.get("source_text") or "").strip() for target in rows if str(target.get("source_text") or "").strip())
        if paragraph_id and _looks_like_structural_label(source):
            continue
        before = text[max(0, (start_i or 0) - context_chars):(start_i or 0)] if start_i is not None else ""
        after = text[(end_i or 0):min(len(text), (end_i or 0) + context_chars)] if end_i is not None else ""
        protected_anchors = _anchors_present_in_source(_dedupe_anchors(rows), source)
        soft_guidance_anchors = _anchors_present_in_source(_dedupe_soft_guidance_anchors(rows), source)
        groups.append(TargetGroup(
            group_id=f"tg{index:03d}",
            unit_id=str(unit_id),
            operation=_target_operation(rows),
            start_index=start_i,
            end_index=end_i,
            source_text=source,
            before_context=before,
            after_context=after,
            targets=tuple(rows),
            protected_anchors=protected_anchors,
            soft_guidance_anchors=soft_guidance_anchors,
            word_count_guide=_combined_word_guide(rows, source),
        ))
        if len(groups) >= max(1, int(max_groups or 1)):
            break
    return groups


def batch_target_groups(target_groups: list[TargetGroup], *, batch_size: int = 4) -> list[list[TargetGroup]]:
    size = max(1, int(batch_size or 1))
    return [target_groups[index:index + size] for index in range(0, len(target_groups), size)]


def build_target_executor_prompt(
    *,
    target_groups: list[TargetGroup],
    content_mode: str,
    strategy_family: str,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> str:
    compact_groups = []
    for group in target_groups:
        compact_targets = []
        for target in group.targets:
            compact_targets.append({
                "target_id": target.get("target_id"),
                "risk_level": target.get("risk_level"),
                "dominant_drivers": target.get("dominant_drivers") or [],
                "required_movement": target.get("required_movement") or {},
                "rewrite_constraints": target.get("rewrite_constraints") or {},
            })
        compact_groups.append({
            "group_id": group.group_id,
            "unit_id": group.unit_id,
            "operation": group.operation,
            "source_text": group.source_text,
            "before_context": group.before_context,
            "after_context": group.after_context,
            "dominant_drivers": compact_targets[0].get("dominant_drivers") if compact_targets else [],
            "required_movement": compact_targets[0].get("required_movement") if compact_targets else {},
            "protected_anchors": list(group.protected_anchors),
            "soft_guidance_anchors": list(group.soft_guidance_anchors),
            "word_count_guide": dict(group.word_count_guide),
            "targets": compact_targets,
            "scanner_action_contract": group_action_contract(
                group=group,
                predictability_briefs=predictability_briefs,
            ),
        })
    paragraph_preserving = any(group.operation == "paragraph_preserving_broad_reconstruction" for group in target_groups)
    payload = {
        "content_mode": content_mode,
        "strategy_family": strategy_family,
        "repair_scope": "target_groups_only",
        "target_groups": compact_groups,
        "requirements": [
            "Return JSON only with a replacements array.",
            "Rewrite only source_text for each target group.",
            "Use before_context and after_context only for continuity.",
            "Preserve protected_anchors exactly when present.",
            "Use soft_guidance_anchors as coverage hints, not exact strings.",
            "Use scanner_action_contract as the execution contract for each group.",
            "Use scanner_action_contract.ownership_contract as the authorship ownership contract for each group.",
            "Do not solve ownership by changing point of view only; add source-supported author trace, specific context, and real judgment.",
            "Preserve source viewpoint unless source_text or nearby context already supports author experience, action, observation, or decision.",
            "Patch scanner_action_contract.topk_repair_contract.predictable_spans_in_source when span_source is scanner_exact.",
            "Use raw/rejected predictable spans only as diagnostics; count movement only on valid phrase spans.",
            "Modify at least scanner_action_contract.topk_repair_contract.required_modified_spans phrase spans when span_source is scanner_exact.",
            "Stay inside scanner_action_contract.topk_repair_contract.locality_limits for local predictability repair.",
            "Address dominant_drivers and required_movement from the targets.",
            "Do not perform synonym swapping or tidy paraphrase only.",
            "Rebuild the target as a natural local passage with clearer cause, observation, judgement, or classroom-specific reasoning when the source supports it.",
            "You may split, combine, or reorder sentences inside the same target group when that makes the passage less formulaic.",
            "For cited academic or practice-based writing, keep citations but make the surrounding claim sound like a writer explaining a decision, not a polished abstract.",
            "Prefer concrete relations already present in source_text over broad summary language.",
            "Keep the same local meaning, claims, citations, technical codes, and paragraph role.",
            "Use word_count_guide as a preferred length guide, not a hard min or max.",
            "Do not add unsupported facts, sources, names, numbers, headings, bullets, markdown, labels, or commentary.",
        ],
        "response_schema": {
            "replacements": [
                {
                    "group_id": "tg001",
                    "replacement_text": "replacement prose only"
                }
            ]
        },
    }
    if paragraph_preserving:
        payload["requirements"].extend([
            "For paragraph_preserving_broad_reconstruction, return one replacement paragraph for each source paragraph.",
            "Do not compress the paragraph into a summary. Stay near the preferred_words guide while prioritizing natural movement.",
            "Avoid polished textbook paraphrase. Keep the facts, but make the paragraph read like a person deciding how the pieces connect.",
            "Vary sentence length inside the paragraph and include one grounded judgement or relation that is already supported by source_text.",
        ])
    return (
        "Rewrite only the scanner-targeted passages for DraftProof V3.\n"
        "The scanner has already identified the risky spans. Do not rewrite the whole document.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_target_replacements(raw: str) -> list[dict[str, str]]:
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
        if not group_id or not replacement:
            continue
        replacements.append({"group_id": group_id, "replacement_text": replacement})
    return replacements


def apply_target_replacements(
    *,
    original_text: str,
    target_groups: list[TargetGroup],
    replacements: list[dict[str, str]],
) -> tuple[str, list[dict[str, Any]]]:
    text = str(original_text or "")
    by_id = {str(row.get("group_id") or ""): str(row.get("replacement_text") or "").strip() for row in replacements}
    replace_ranges: list[tuple[int, int, str, str]] = []
    statuses: list[TargetApplyStatus] = []
    occupied: list[tuple[int, int]] = []
    for group in target_groups:
        replacement = by_id.get(group.group_id, "")
        if not replacement:
            statuses.append(TargetApplyStatus(group.group_id, False, "none", "missing_replacement"))
            continue
        missing_anchors = missing_required_protected_anchors(replacement, group)
        if missing_anchors:
            statuses.append(TargetApplyStatus(group.group_id, False, "none", "protected_anchor_missing"))
            continue
        start = group.start_index
        end = group.end_index
        def fit_boundary(replacement_text: str, range_start: int, range_end: int) -> str:
            original_segment = text[range_start:range_end]
            leading = original_segment[:len(original_segment) - len(original_segment.lstrip())]
            trailing = original_segment[len(original_segment.rstrip()):]
            fitted = f"{leading}{replacement_text.strip()}{trailing}"
            previous_char = text[range_start - 1] if range_start > 0 else ""
            next_char = text[range_end] if range_end < len(text) else ""
            if previous_char and not previous_char.isspace() and fitted and not fitted[0].isspace():
                fitted = " " + fitted
            if next_char and not next_char.isspace() and fitted and not fitted[-1].isspace():
                fitted = fitted + " "
            return fitted
        if start is not None and end is not None and 0 <= start < end <= len(text):
            replace_ranges.append((start, end, fit_boundary(replacement, start, end), group.group_id))
            occupied.append((start, end))
            statuses.append(TargetApplyStatus(group.group_id, True, "span"))
            continue
        source = str(group.source_text or "").strip()
        found = text.find(source) if source else -1
        if found >= 0:
            fallback_start = found
            fallback_end = found + len(source)
            overlaps = any(fallback_start < used_end and fallback_end > used_start for used_start, used_end in occupied)
            if overlaps:
                statuses.append(TargetApplyStatus(group.group_id, False, "source_text", "overlapping_fallback_span"))
                continue
            replace_ranges.append((fallback_start, fallback_end, fit_boundary(replacement, fallback_start, fallback_end), group.group_id))
            occupied.append((fallback_start, fallback_end))
            statuses.append(TargetApplyStatus(group.group_id, True, "source_text"))
            continue
        statuses.append(TargetApplyStatus(group.group_id, False, "none", "source_span_not_found"))
    for start, end, replacement, _ in sorted(replace_ranges, key=lambda item: item[0], reverse=True):
        text = text[:start] + replacement + text[end:]
    return text.strip(), [status.to_dict() for status in statuses]


def target_execution_trace(
    *,
    attempted: bool,
    target_groups: list[TargetGroup],
    replacements: list[dict[str, str]] | None = None,
    apply_status: list[dict[str, Any]] | None = None,
    batches: list[dict[str, Any]] | None = None,
    llm_calls: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    unresolved = []
    for status in apply_status or []:
        if not status.get("applied"):
            unresolved.append(status.get("group_id"))
    return {
        "target_execution_attempted": bool(attempted),
        "target_groups": [group.to_dict() for group in target_groups],
        "target_replacements": list(replacements or []),
        "target_apply_status": list(apply_status or []),
        "target_batches": list(batches or []),
        "llm_calls": list(llm_calls or []),
        "llm_call_count": len(llm_calls or []),
        "unresolved_targets": unresolved,
        "error": error,
    }
