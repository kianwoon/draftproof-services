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


def normalize_submitted_text(text: str) -> str:
    """Return the scan-facing text with stable semantic paragraph breaks.

    The normalizer changes only paragraph boundaries. It preserves sentence text
    and order so scanner evidence, report highlights, and rewrite references use
    the same submitted wording.
    """
    source = str(text or "")
    if not source.strip():
        return ""
    blocks = _semantic_blocks(source)
    return "\n\n".join(block.strip() for block in blocks if block.strip()).strip()


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
    """Group one physical block's sentences into virtual paragraphs (the p001/s001 IDs).

    This is the ID-producing path the scanner and report consume, so it MUST NOT depend on
    ``normalize_submitted_text`` having run first. If a caller scans un-normalized text, a lumped
    block still has to break at semantic boundaries -- not at an arbitrary sentence count -- or the
    findings the planner acts on land on count-sliced paragraphs. The splitting logic is therefore
    shared with the normalizer via ``_semantic_sentence_groups`` (single source of truth for where a
    paragraph breaks). A block already within the size caps -- the normal case once normalize has run
    -- is returned whole, so already-normalized paragraph boundaries are reproduced exactly.
    """
    if not rows:
        return []
    max_words = _int_env("DRAFTPROOF_STRUCTURE_MAX_PARAGRAPH_WORDS", 170, minimum=60, maximum=320)
    max_sentences = _int_env("DRAFTPROOF_STRUCTURE_MAX_PARAGRAPH_SENTENCES", 8, minimum=3, maximum=16)
    total_words = sum(int(row.get("word_count") or 0) for row in rows)
    if len(rows) <= max_sentences and total_words <= max_words:
        return [rows]
    return _semantic_sentence_groups(rows)


def _semantic_blocks(text: str) -> list[str]:
    raw_blocks = _physical_blocks(text)
    blocks: list[str] = []
    pending: list[str] = []
    pending_words = 0
    max_words = _int_env("DRAFTPROOF_STRUCTURE_MAX_PARAGRAPH_WORDS", 170, minimum=60, maximum=320)
    min_words = _int_env("DRAFTPROOF_STRUCTURE_MIN_PARAGRAPH_WORDS", 70, minimum=20, maximum=max_words)
    max_sentences = _int_env("DRAFTPROOF_STRUCTURE_MAX_PARAGRAPH_SENTENCES", 8, minimum=3, maximum=16)

    def flush_pending() -> None:
        nonlocal pending, pending_words
        if not pending:
            return
        text = " ".join(part.strip() for part in pending if part.strip()).strip()
        orphan_words = pending_words
        pending = []
        pending_words = 0
        if not text:
            return
        # A short orphan -- e.g. a lone sentence sitting between two paragraphs -- attaches to the
        # preceding real paragraph instead of standing alone, so the scanner and rewrite see a full
        # paragraph rather than a one-sentence fragment. Caps still apply (see _can_absorb_orphan).
        if blocks and _can_absorb_orphan(blocks[-1], text, orphan_words, min_words, max_words, max_sentences):
            blocks[-1] = (blocks[-1] + " " + text).strip()
        else:
            blocks.append(text)

    for block in raw_blocks:
        block_text = str(block.get("text") or "").strip()
        if not block_text:
            continue
        if _looks_like_heading(block_text):
            flush_pending()
            blocks.append(block_text)
            continue

        sentence_rows = _sentence_rows_for_block(text, block)
        if not sentence_rows:
            continue
        groups = _semantic_sentence_groups(sentence_rows)
        for group in groups:
            paragraph_text = " ".join(
                str(row.get("sentence") or "").strip()
                for row in group
                if str(row.get("sentence") or "").strip()
            ).strip()
            if not paragraph_text:
                continue
            group_words = sum(int(row.get("word_count") or 0) for row in group)
            group_sentences = len(group)
            if (
                pending
                and pending_words + group_words <= max_words
                and len(_split_pending_sentences(pending)) + group_sentences <= max_sentences
                and _looks_like_continuation(paragraph_text)
            ):
                pending.append(paragraph_text)
                pending_words += group_words
                continue
            flush_pending()
            if group_words < min_words and group_sentences <= 2:
                pending.append(paragraph_text)
                pending_words = group_words
            else:
                blocks.append(paragraph_text)
    flush_pending()
    return blocks


def _semantic_sentence_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    max_words = _int_env("DRAFTPROOF_STRUCTURE_MAX_PARAGRAPH_WORDS", 170, minimum=60, maximum=320)
    max_sentences = _int_env("DRAFTPROOF_STRUCTURE_MAX_PARAGRAPH_SENTENCES", 8, minimum=3, maximum=16)
    min_words = _int_env("DRAFTPROOF_STRUCTURE_MIN_PARAGRAPH_WORDS", 70, minimum=20, maximum=max_words)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    for index, row in enumerate(rows):
        row_words = int(row.get("word_count") or 0)
        should_break = bool(current) and (
            len(current) >= max_sentences
            or current_words + row_words > max_words
            or (
                current_words >= min_words
                and len(current) >= 3
                and _semantic_boundary_score(current[-1], row, index) >= 2
            )
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


def _semantic_boundary_score(previous: dict[str, Any], current: dict[str, Any], index: int) -> int:
    previous_text = str(previous.get("sentence") or "").strip()
    current_text = str(current.get("sentence") or "").strip()
    score = 0
    if re.match(
        r"^(however|therefore|nevertheless|instead|in contrast|by contrast|on the other hand|another|finally|overall|in conclusion|as a result)\b",
        current_text,
        flags=re.I,
    ):
        score += 2
    if re.match(
        r"^(first|second|third|next|then|also|in addition|for example|for instance)\b",
        current_text,
        flags=re.I,
    ):
        score += 1
    if re.search(r"\([A-Z][^)]*(?:19|20)\d{2}[^)]*\)\.?$", previous_text):
        score += 1
    if index >= 5 and re.match(r"^(this|that|these|those|it)\b", current_text, flags=re.I):
        score += 1
    return score


def _looks_like_continuation(text: str) -> bool:
    value = str(text or "").strip()
    return bool(
        re.match(
            r"^(this|that|these|those|it|they|such|however|therefore|also|in addition|for example|for instance|because|when|while)\b",
            value,
            flags=re.I,
        )
    )


def _can_absorb_orphan(
    previous: str,
    orphan: str,
    orphan_words: int,
    min_words: int,
    max_words: int,
    max_sentences: int,
) -> bool:
    """True when a short trailing orphan should merge into the previous paragraph.

    Targets the lone-sentence-between-paragraphs case: a short fragment (under the paragraph-size
    floor, at most two sentences) attaches to the preceding paragraph when that paragraph is a real,
    substantial one -- a multi-sentence paragraph or one already past the word floor, not a heading
    or a peer short line -- and the combined paragraph stays within the word/sentence caps. This
    keeps a one-sentence orphan from becoming its own paragraph while leaving genuinely independent
    short paragraphs apart.
    """
    if orphan_words >= min_words or len(split_sentences(orphan)) > 2:
        return False
    if _looks_like_heading(previous):
        return False
    previous_sentences = len(split_sentences(previous))
    if previous_sentences < 3 and len(previous.split()) < min_words:
        return False
    if len(previous.split()) + orphan_words > max_words:
        return False
    if previous_sentences + len(split_sentences(orphan)) > max_sentences:
        return False
    return True


def _split_pending_sentences(parts: list[str]) -> list[str]:
    return split_sentences(" ".join(parts))


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))
