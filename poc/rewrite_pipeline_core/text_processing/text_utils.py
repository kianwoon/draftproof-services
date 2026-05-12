"""General text utilities used by rewrite pipeline phases."""

from __future__ import annotations

import re


def _split_sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        if item.strip()
    ]


def _brief_sentences(text: str, limit: int = 10) -> list[str]:
    if not isinstance(text, str):
        return []
    rows = []
    for sentence in re.findall(r"[^.!?\n]+(?:[.!?]+|$)", text):
        cleaned = " ".join(sentence.split()).strip()
        if len(cleaned.split()) < 6:
            continue
        rows.append(cleaned)
        if len(rows) >= limit:
            break
    return rows


def _text_word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\b[\w’'-]+\b", text))


def _normalize_protected_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def _protected_number_set(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text or "")))


def _logical_paragraphs(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraphs) != 1:
        return paragraphs

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 4:
        return paragraphs
    heading_like = sum(
        1
        for line in lines
        if len(line.split()) <= 9 and not re.search(r"[.!?:;]\s*$", line)
    )
    prose_like = sum(1 for line in lines if len(line.split()) >= 14)
    if heading_like >= 2 and prose_like >= 2:
        return lines
    return paragraphs


def _join_logical_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join(p.strip() for p in paragraphs if str(p or "").strip())
