from __future__ import annotations

import re
from typing import Any


def compile_sentence_rows(row: dict[str, Any]) -> str:
    """Compile writer sentence rows into paragraph text.

    The writer may still return a paragraph-level ``text`` field, but V6 should
    not let that field recombine carefully planned sentence rows into a smooth
    paragraph route. Rows are the execution unit; paragraph text is compiled.
    """
    rows = _row_list(row.get("sentence_rows")) or _row_list(row.get("coverage_map"))
    sentences: list[str] = []
    seen: set[str] = set()
    for item in rows:
        for sentence in _row_sentences(item):
            key = _sentence_key(sentence)
            if key in seen:
                continue
            sentences.append(sentence)
            seen.add(key)
    return " ".join(sentences).strip()


def compile_or_fallback_text(row: dict[str, Any]) -> str:
    compiled = compile_sentence_rows(row)
    return compiled or str(row.get("text") or "").strip()


def _row_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clean_sentence(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    text = text[:1].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _row_sentences(item: dict[str, Any]) -> list[str]:
    sentence = _clean_sentence(item.get("sentence"))
    if not sentence:
        return []
    beat_count = len(item.get("coverage_beat_ids") or [])
    if beat_count <= 1 and _word_count(sentence) <= 24:
        return [sentence]
    return [_clean_sentence(part) for part in _split_overloaded_sentence(sentence, force=beat_count > 1)]


def _split_overloaded_sentence(sentence: str, *, force: bool = False) -> list[str]:
    text = str(sentence or "").strip().rstrip(".!?")
    if not force and _word_count(text) <= 24:
        return [_clean_sentence(text)]
    parts = _split_framework_working_row(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_evaluates_between_row(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_first_person_working_to(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_once(text, r"\s+\bbecause\b\s+")
    if len(parts) == 2:
        left, right = parts
        return [_clean_sentence(left), *_split_overloaded_sentence(_ensure_subject(right, left), force=force)]
    parts = _split_once(text, r"\s+\bwhile\b\s+")
    if len(parts) == 2:
        left, right = parts
        return [_clean_sentence(left), _clean_sentence(_ensure_subject(right, left))]
    parts = _split_support_clause(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    return [_clean_sentence(text)]


def _split_once(text: str, pattern: str) -> list[str]:
    parts = re.split(pattern, text, maxsplit=1, flags=re.I)
    return [part.strip(" ,;:") for part in parts if part.strip(" ,;:")]


def _split_first_person_working_to(text: str) -> list[str]:
    match = re.match(r"\s*((?:I|We)\s+(?:am|are)\s+[^.]{4,120}?)\s+working\s+to\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    subject = match.group(1).split()[0]
    auxiliary = "am" if subject.casefold() == "i" else "are"
    return [match.group(1), f"{subject} {auxiliary} working to {match.group(2)}"]


def _split_framework_working_row(text: str) -> list[str]:
    match = re.match(
        r"\s*((?:I|We)\s+(?:am|are)\s+[^.]{4,120}?)\s+working\s+to\s+design\s+and\s+implement\s+(.+?)\s+for\s+(.+?)\s+that\s+(.+)$",
        str(text or ""),
        flags=re.I,
    )
    if not match:
        return [text]
    role = match.group(1).strip(" ,;:")
    object_text = match.group(2).strip(" ,;:")
    context = match.group(3).strip(" ,;:")
    clause = match.group(4).strip(" ,;:")
    return [role, f"{context} uses {object_text}", f"{context} {clause}"]


def _split_evaluates_between_row(text: str) -> list[str]:
    match = re.match(
        r"\s*(.+?)\s+(?:critically\s+)?evaluates\s+(?:the\s+tension\s+)?between\s+(.+?)\s+and\s+(.+?)\s+required\s+by\s+(.+)$",
        str(text or ""),
        flags=re.I,
    )
    if match:
        subject = match.group(1).strip(" ,;:")
        first = _plain_compare_side(match.group(2))
        second = _plain_compare_side(match.group(3))
        requirement = match.group(4).strip(" ,;:")
        return [f"{subject} compares {first} with {second}", f"The second side is required by {requirement}"]
    match = re.match(
        r"\s*(.+?)\s+(?:critically\s+)?evaluates\s+(?:the\s+tension\s+)?between\s+(.+?)\s+and\s+(.+)$",
        str(text or ""),
        flags=re.I,
    )
    if not match:
        return [text]
    return [
        f"{match.group(1).strip(' ,;:')} compares {_plain_compare_side(match.group(2))} with {_plain_compare_side(match.group(3))}"
    ]


def _plain_compare_side(text: str) -> str:
    return re.sub(r"^\s*(?:providing|upholding|maintaining|using|applying)\s+", "", str(text or ""), flags=re.I).strip(" ,;:")


def _split_support_clause(text: str) -> list[str]:
    match = re.match(r"\s*(.{12,180}?)\s+\bto\s+support\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    return [match.group(1), f"The same support focused on {match.group(2)}"]


def _ensure_subject(fragment: str, previous: str) -> str:
    text = str(fragment or "").strip()
    if not text:
        return text
    first = text.split()[0].casefold()
    subject = _first_subject(previous)
    if first in {"it", "this", "that"} and subject and subject.casefold() not in {"it", "this", "that"}:
        return re.sub(r"^\w+", subject, text, count=1)
    if first in {"i", "we", "the", "this", "that", "these", "those", "it", "they", "learners", "students"}:
        return text
    if subject and first.endswith("ing"):
        return f"{subject} involves {text}"
    return f"{subject} {text}" if subject else text


def _first_subject(text: str) -> str:
    match = re.match(r"\s*((?:I|We|This|That|These|Those|It|They)\b)", str(text or ""), flags=re.I)
    if match:
        return match.group(1).strip().capitalize()
    match = re.match(r"\s*((?:The)\s+[A-Za-z0-9'’-]+)", str(text or ""), flags=re.I)
    return match.group(1).strip() if match else ""


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "")))


def _sentence_key(sentence: str) -> str:
    return re.sub(r"\W+", " ", sentence).strip().casefold()
