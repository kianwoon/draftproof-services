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


SUPPORTED_TARGET_OPERATIONS = {
    "protected_section_rewrite",
    "citation_preserving_window_repair",
    "grounded_author_reasoning_rewrite",
    "light_texture_rewrite",
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


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


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
    if anchor_text in text:
        return True
    return _anchor_compare_text(anchor_text) in _anchor_compare_text(text)


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


def _combined_word_guide(targets: list[dict[str, Any]], source_text: str) -> dict[str, Any]:
    source_words = 0
    preferred_words = 0
    for target in targets:
        guide = target.get("word_count_guide") if isinstance(target.get("word_count_guide"), dict) else {}
        source_words += int(_number(guide.get("source_words"), _word_count(str(target.get("source_text") or ""))))
        preferred_words += int(_number(guide.get("preferred_words"), _word_count(str(target.get("source_text") or ""))))
    return {
        "source_words": source_words or _word_count(source_text),
        "preferred_words": preferred_words or _word_count(source_text),
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
        valid_spans = [(start, end) for start, end, passed in spans if passed and start is not None and end is not None and 0 <= start < end <= len(text)]
        paragraph_id = str(rows[0].get("paragraph_id") or "")
        paragraph_span = paragraph_spans.get(paragraph_id)
        if paragraph_span is not None:
            start_i, end_i = paragraph_span
            source = text[start_i:end_i].strip()
        elif valid_spans:
            start_i = min(start for start, _ in valid_spans)
            end_i = max(end for _, end in valid_spans)
            source = text[start_i:end_i].strip()
        else:
            start_i = None
            end_i = None
            source = "\n\n".join(str(target.get("source_text") or "").strip() for target in rows if str(target.get("source_text") or "").strip())
        if paragraph_id and _looks_like_structural_label(source):
            continue
        before = text[max(0, (start_i or 0) - context_chars):(start_i or 0)] if start_i is not None else ""
        after = text[(end_i or 0):min(len(text), (end_i or 0) + context_chars)] if end_i is not None else ""
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
            protected_anchors=_dedupe_anchors(rows),
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
            "word_count_guide": dict(group.word_count_guide),
            "targets": compact_targets,
        })
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
        missing_anchors = [
            str(anchor.get("text") or "").strip()
            for anchor in group.protected_anchors
            if str(anchor.get("text") or "").strip() and not _anchor_present(replacement, str(anchor.get("text") or "").strip())
        ]
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
        "unresolved_targets": unresolved,
        "error": error,
    }
