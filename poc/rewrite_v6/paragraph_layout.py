from __future__ import annotations

import re
from typing import Protocol

from .scan import scan_text
from .text import Paragraph, split_paragraphs


class ParagraphRewriteResult(Protocol):
    scan: object
    plan: object
    selected: object | None
    rewritten_text: str


def restore_original_paragraph_layout(
    original_text: str,
    rewritten_text: str,
    passes: list[ParagraphRewriteResult],
) -> str:
    original_paragraphs = split_paragraphs(original_text)
    rewritten_paragraphs = split_paragraphs(rewritten_text)
    if not original_paragraphs or not rewritten_paragraphs:
        return rewritten_text
    if len(original_paragraphs) == len(rewritten_paragraphs):
        return rewritten_text

    origins = _paragraph_origins_after_passes(original_paragraphs, passes)
    if len(origins) != len(rewritten_paragraphs):
        return _fold_by_count(original_paragraphs, rewritten_paragraphs)

    slots: list[list[str]] = [[] for _ in original_paragraphs]
    for origin_index, paragraph in zip(origins, rewritten_paragraphs, strict=False):
        if 0 <= origin_index < len(slots):
            slots[origin_index].append(paragraph.text)

    if not all(slots):
        return _fold_by_count(original_paragraphs, rewritten_paragraphs)
    return "\n\n".join(_join_blocks(blocks) for blocks in slots)


def _paragraph_origins_after_passes(
    original_paragraphs: list[Paragraph],
    passes: list[ParagraphRewriteResult],
) -> list[int]:
    origins = list(range(len(original_paragraphs)))
    for result in passes:
        before_paragraphs = getattr(getattr(result, "scan", None), "paragraphs", None)
        plan = getattr(result, "plan", None)
        target_id = getattr(plan, "paragraph_id", "")
        if not before_paragraphs or not target_id:
            continue
        target_index = _paragraph_index(before_paragraphs, target_id)
        if target_index is None or target_index >= len(origins):
            continue

        after_paragraphs = scan_text(getattr(result, "rewritten_text", "")).paragraphs
        expected_children = _replacement_block_count(getattr(result, "selected", None))
        delta = len(after_paragraphs) - len(before_paragraphs)
        child_count = max(1, expected_children, delta + 1 if delta > 0 else 1)
        child_count = min(child_count, max(1, len(after_paragraphs) - target_index))
        origin = origins[target_index]
        origins = [
            *origins[:target_index],
            *([origin] * child_count),
            *origins[target_index + 1:],
        ]
    return origins


def _paragraph_index(paragraphs: list[Paragraph], paragraph_id: str) -> int | None:
    for index, paragraph in enumerate(paragraphs):
        if paragraph.id == paragraph_id:
            return index
    return None


def _replacement_block_count(selected: object | None) -> int:
    text = str(getattr(selected, "text", "") or "") if selected is not None else ""
    return max(1, len([block for block in re.split(r"\n\s*\n+", text.strip()) if block.strip()]))


def _fold_by_count(original_paragraphs: list[Paragraph], rewritten_paragraphs: list[Paragraph]) -> str:
    if len(rewritten_paragraphs) <= len(original_paragraphs):
        return "\n\n".join(paragraph.text for paragraph in rewritten_paragraphs)
    slots: list[list[str]] = [[paragraph.text] for paragraph in rewritten_paragraphs[:len(original_paragraphs)]]
    overflow = rewritten_paragraphs[len(original_paragraphs):]
    if overflow:
        slots[-1].extend(paragraph.text for paragraph in overflow)
    return "\n\n".join(_join_blocks(blocks) for blocks in slots)


def _join_blocks(blocks: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(block.strip() for block in blocks if block.strip())).strip()
