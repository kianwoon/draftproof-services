from __future__ import annotations

import re

from .text import split_sentences, word_count


def normalize_paragraph_blocks(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    normalized: list[str] = []
    for block in blocks:
        normalized.extend(_split_oversized_block(block))
    return "\n\n".join(normalized).strip()


def _split_oversized_block(block: str) -> list[str]:
    heading, body = _split_heading(block)
    sentences = split_sentences(body, paragraph_id="p000")
    body_words = word_count(body)
    if body_words < 120:
        return [block.strip()]
    if len(sentences) < 9 and word_count(body) < 220:
        return [block.strip()]
    chunks: list[list[str]] = []
    current: list[str] = []
    for index, sentence in enumerate(sentences):
        if current and _should_start_new_chunk(current, sentence.text, index):
            chunks.append(current)
            current = []
        current.append(sentence.text)
    if current:
        chunks.append(current)
    if len(chunks) <= 1:
        return [block.strip()]
    rows = [" ".join(chunk).strip() for chunk in chunks if chunk]
    if heading and rows:
        rows[0] = f"{heading}\n{rows[0]}"
    return rows


def _split_heading(block: str) -> tuple[str, str]:
    lines = block.splitlines()
    if len(lines) >= 2 and lines[0].strip() and not re.search(r"[.!?]$", lines[0].strip()):
        return lines[0].strip(), "\n".join(lines[1:]).strip()
    return "", block.strip()


def _should_start_new_chunk(current: list[str], next_sentence: str, index: int) -> bool:
    current_words = word_count(" ".join(current))
    if current_words < 120 and len(current) < 10:
        return False
    if len(current) >= 6:
        return True
    if current_words >= 150 and len(current) >= 3:
        return True
    if len(current) < 3:
        return False
    if _boundary_score(current[-1], next_sentence, index) >= 2:
        return True
    return len(current) >= 4 and _boundary_score(current[-1], next_sentence, index) >= 1


def _boundary_score(previous_sentence: str, next_sentence: str, index: int) -> int:
    score = 0
    next_text = next_sentence.strip()
    previous = previous_sentence.strip()
    if re.match(r"^(later|after that|afterwards|in addition|however|for this reason|in my view|from this|this experience|this example)\b", next_text, flags=re.I):
        score += 2
    if re.match(r"^(for\s+[A-Z][A-Za-z'’-]+|during|when|one\s+of|another\s+issue|to\s+promote|to\s+address)\b", next_text, flags=re.I):
        score += 1
    if re.search(r"\([A-Z][^)]*(?:19|20)\d{2}[^)]*\)\.?$", previous):
        score += 1
    if index >= 5 and re.match(r"^(this|that|it)\b", next_text, flags=re.I):
        score += 1
    return score
