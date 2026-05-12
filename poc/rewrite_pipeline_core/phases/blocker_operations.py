from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BlockerOperationCandidateDeps:
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    text_word_count: Callable[[str], int]
    blocker_operation_plan: Callable[..., dict]
    safe_index: Callable[[Any, int], int]
    compress_score_drag_paragraph: Callable[..., str]
    narrow_generic_claim_text: Callable[[str], str]


def blocker_operation_candidates(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 6,
    deps: BlockerOperationCandidateDeps,
) -> list[tuple[str, str, dict]]:
    """Generate deterministic candidates from the blocker operation compiler."""
    if not deps.env_flag("DRAFTPROOF_BLOCKER_OPERATION_COMPILER", True):
        return []
    paragraphs = deps.logical_paragraphs(source_text)
    if len(paragraphs) < 2:
        return []
    source_words = deps.text_word_count(source_text)
    min_words = max(1, int(source_words * deps.float_env("DRAFTPROOF_BLOCKER_OPERATION_MIN_WORD_RATIO", 0.60)))
    plan = deps.blocker_operation_plan(source_text, raw_json or {}, limit=max(limit * 2, 6))
    operations = plan.get("operations") or []
    decision_by_index = {
        deps.safe_index(decision.get("paragraph_index"), -1): decision
        for decision in plan.get("block_decisions") or []
        if isinstance(decision, dict)
    }
    candidates: list[tuple[str, str, dict]] = []
    seen: set[str] = set()

    def add_candidate(strategy: str, candidate_paragraphs: list[str], meta: dict) -> None:
        candidate_text = deps.join_logical_paragraphs(candidate_paragraphs)
        if candidate_text.strip() == source_text.strip():
            return
        if deps.text_word_count(candidate_text) < min_words:
            return
        if candidate_text in seen:
            return
        seen.add(candidate_text)
        candidates.append((strategy, candidate_text, {**meta, "operation_plan": plan}))

    for op in operations:
        if len(candidates) >= max(1, limit):
            break
        index = deps.safe_index(op.get("paragraph_index"), -1)
        if index < 0 or index >= len(paragraphs):
            continue
        paragraph = paragraphs[index]
        operation = str(op.get("operation") or "")
        decision = decision_by_index.get(index, {})
        decision_name = str(decision.get("decision") or "")
        allowed_operations = set(decision.get("allowed_operations") or [])
        has_protected = bool(op.get("has_protected_anchor"))
        if (
            operation in {"delete_or_compress", "compress_or_delete"}
            and not has_protected
            and decision_name == "remove_or_compress"
            and "delete_paragraph" in allowed_operations
        ):
            deleted = list(paragraphs)
            deleted.pop(index)
            add_candidate(
                f"blocker_compiler_delete_p{index + 1}",
                deleted,
                {
                    "operation": "delete_paragraph",
                    "paragraph_index": index,
                    "compiled_from": op,
                    "block_decision": decision,
                },
            )
        compressed_text = deps.compress_score_drag_paragraph(
            paragraph,
            max_remove=2 if operation in {"delete_or_compress", "compress_or_delete"} else 1,
        )
        if (
            compressed_text.strip()
            and compressed_text.strip() != paragraph.strip()
            and "compress_paragraph" in allowed_operations
        ):
            compressed = list(paragraphs)
            compressed[index] = compressed_text
            add_candidate(
                f"blocker_compiler_compress_p{index + 1}",
                compressed,
                {
                    "operation": "compress_or_narrow_paragraph",
                    "paragraph_index": index,
                    "compiled_from": op,
                    "block_decision": decision,
                },
            )
        narrowed_text = deps.narrow_generic_claim_text(paragraph)
        if (
            narrowed_text.strip()
            and narrowed_text.strip() != paragraph.strip()
            and "claim_narrow" in allowed_operations
        ):
            narrowed = list(paragraphs)
            narrowed[index] = narrowed_text
            add_candidate(
                f"blocker_compiler_narrow_p{index + 1}",
                narrowed,
                {
                    "operation": "claim_narrow",
                    "paragraph_index": index,
                    "compiled_from": op,
                    "block_decision": decision,
                },
            )

    if len(candidates) < max(1, limit):
        combined = list(paragraphs)
        changed_indexes = []
        for op in operations:
            index = deps.safe_index(op.get("paragraph_index"), -1)
            if index < 0 or index >= len(combined) or bool(op.get("has_protected_anchor")):
                continue
            if len(changed_indexes) >= 3:
                break
            operation = str(op.get("operation") or "")
            if operation in {"delete_or_compress", "compress_or_delete", "claim_narrow"}:
                replacement = deps.compress_score_drag_paragraph(combined[index], max_remove=2)
                if replacement.strip() and replacement.strip() != combined[index].strip():
                    combined[index] = replacement
                    changed_indexes.append(index)
        if changed_indexes:
            add_candidate(
                "blocker_compiler_multi_compress",
                combined,
                {"operation": "multi_compress_or_narrow", "paragraph_indexes": changed_indexes},
            )

    return candidates[:max(1, limit)]
