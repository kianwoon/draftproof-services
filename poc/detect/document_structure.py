"""Document structure helpers shared by scan and rewrite report layers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .utils import split_sentences


@dataclass(frozen=True)
class StructuredSentence:
    sentence_id: str
    paragraph_id: str
    source_paragraph_id: str
    virtual_paragraph_id: str
    sentence_index: int
    start_char: int
    end_char: int
    sentence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "paragraph_id": self.paragraph_id,
            "source_paragraph_id": self.source_paragraph_id,
            "virtual_paragraph_id": self.virtual_paragraph_id,
            "sentence_index": self.sentence_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "sentence": self.sentence,
        }


def structured_sentence_segments(text: str) -> list[dict[str, Any]]:
    return [row.to_dict() for row in structured_sentences(text)]


def structured_paragraph_texts(text: str) -> list[str]:
    segments = structured_sentences(text)
    paragraphs: dict[str, list[StructuredSentence]] = {}
    for segment in segments:
        paragraphs.setdefault(segment.paragraph_id, []).append(segment)
    rows: list[str] = []
    for paragraph_id in sorted(paragraphs):
        sentence_rows = sorted(paragraphs[paragraph_id], key=lambda item: item.start_char)
        paragraph_text = " ".join(row.sentence.strip() for row in sentence_rows if row.sentence.strip()).strip()
        if paragraph_text:
            rows.append(paragraph_text)
    return rows


def structured_sentences(text: str) -> list[StructuredSentence]:
    source = str(text or "")
    if not source.strip():
        return []
    physical_blocks = _physical_blocks(source)
    rows: list[StructuredSentence] = []
    sentence_no = 0
    paragraph_no = 0
    for source_paragraph_no, block in enumerate(physical_blocks, start=1):
        sentence_rows = _sentence_rows_for_block(source, block)
        if not sentence_rows:
            continue
        groups = _virtual_paragraph_groups(sentence_rows)
        for group in groups:
            paragraph_no += 1
            paragraph_id = f"p{paragraph_no:03d}"
            source_paragraph_id = f"src_p{source_paragraph_no:03d}"
            for row in group:
                sentence_no += 1
                rows.append(StructuredSentence(
                    sentence_id=f"s{sentence_no:03d}",
                    paragraph_id=paragraph_id,
                    source_paragraph_id=source_paragraph_id,
                    virtual_paragraph_id=paragraph_id,
                    sentence_index=sentence_no - 1,
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    sentence=str(row["sentence"]),
                ))
    return rows


def _physical_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    search_from = 0
    for match in re.finditer(r"\S(?:.*?\S)?(?=(?:\n\s*\n+)|\s*\Z)", text, flags=re.DOTALL):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            search_from = match.end()
            continue
        start = text.find(stripped, search_from)
        if start < 0:
            start = match.start() + len(raw) - len(raw.lstrip())
        blocks.extend(_split_heading_lines(stripped, start))
        search_from = match.end()
    return blocks


def _split_heading_lines(block: str, block_start: int) -> list[dict[str, Any]]:
    lines = block.splitlines()
    if len(lines) < 2:
        return [{"text": block, "start_char": block_start}]
    first = lines[0].strip()
    rest = "\n".join(lines[1:]).strip()
    if first and rest and _looks_like_heading(first):
        rest_start = block_start + block.find(lines[1])
        return [
            {"text": first, "start_char": block_start},
            {"text": rest, "start_char": rest_start},
        ]
    return [{"text": block, "start_char": block_start}]


def _looks_like_heading(line: str) -> bool:
    value = " ".join(str(line or "").split())
    if not value:
        return False
    max_words = _int_env("DRAFTPROOF_STRUCTURE_HEADING_MAX_WORDS", 14, minimum=3, maximum=30)
    return (
        len(value.split()) <= max_words
        and not re.search(r"[.!?:;]\s*$", value)
        and bool(re.search(r"[A-Za-z]", value))
    )


def _sentence_rows_for_block(source: str, block: dict[str, Any]) -> list[dict[str, Any]]:
    block_text = str(block.get("text") or "")
    block_start = int(block.get("start_char") or 0)
    sentences = split_sentences(block_text)
    if not sentences and block_text.strip():
        sentences = [block_text.strip()]
    rows: list[dict[str, Any]] = []
    local_cursor = 0
    for sentence in sentences:
        sentence_text = str(sentence or "").strip()
        if not sentence_text:
            continue
        local_offset = block_text.find(sentence_text[:40], local_cursor)
        if local_offset < 0:
            local_offset = block_text.find(sentence_text, local_cursor)
        if local_offset < 0:
            local_offset = local_cursor
        start = block_start + local_offset
        end = start + len(sentence_text)
        if start < 0 or end <= start or end > len(source):
            start = block_start + local_cursor
            end = min(len(source), start + len(sentence_text))
        rows.append({
            "sentence": sentence_text,
            "start_char": start,
            "end_char": end,
            "word_count": len(sentence_text.split()),
        })
        local_cursor = max(local_cursor, local_offset + len(sentence_text))
    return rows


def _virtual_paragraph_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    max_words = _int_env("DRAFTPROOF_STRUCTURE_MAX_PARAGRAPH_WORDS", 170, minimum=60, maximum=320)
    max_sentences = _int_env("DRAFTPROOF_STRUCTURE_MAX_PARAGRAPH_SENTENCES", 8, minimum=3, maximum=16)
    min_words = _int_env("DRAFTPROOF_STRUCTURE_MIN_PARAGRAPH_WORDS", 70, minimum=20, maximum=max_words)
    total_words = sum(int(row.get("word_count") or 0) for row in rows)
    if len(rows) <= max_sentences and total_words <= max_words:
        return [rows]

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    for row in rows:
        row_words = int(row.get("word_count") or 0)
        should_break = bool(current) and (
            len(current) >= max_sentences
            or current_words + row_words > max_words
        )
        if should_break and current_words < min_words and len(current) < max_sentences + 2:
            should_break = False
        if should_break:
            groups.append(current)
            current = []
            current_words = 0
        current.append(row)
        current_words += row_words
    if current:
        if groups and current_words < min_words:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))
