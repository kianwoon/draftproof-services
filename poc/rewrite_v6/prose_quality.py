from __future__ import annotations

import re


def drop_redundant_adjacent_sentence_intent(text: str) -> str:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    cleaned: list[str] = []
    for block in paragraphs:
        heading, body = _split_heading(block)
        kept: list[str] = []
        for sentence in _sentences(body):
            if kept and _same_sentence_intent(kept[-1], sentence):
                continue
            kept.append(sentence)
        rebuilt = " ".join(kept).strip()
        cleaned.append("\n".join(part for part in [heading, rebuilt] if part).strip())
    return "\n\n".join(cleaned).strip()


def has_fragment_or_trace_sentences(text: str) -> bool:
    sentences = _sentences(text)
    if not sentences:
        return True
    flagged = sum(1 for sentence in sentences if _fragment_like(sentence) or _repair_trace_like(sentence))
    return flagged >= 2 or flagged / max(1, len(sentences)) >= 0.18


def fragment_trace_penalty(text: str) -> float:
    sentences = _sentences(text)
    if not sentences:
        return 8.0
    flagged = sum(1 for sentence in sentences if _fragment_like(sentence) or _repair_trace_like(sentence))
    return round((flagged / max(1, len(sentences))) * 8.0, 3)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]


def _split_heading(block: str) -> tuple[str, str]:
    lines = block.splitlines()
    if len(lines) >= 2 and lines[0].strip() and not re.search(r"[.!?]$", lines[0].strip()):
        return lines[0].strip(), "\n".join(lines[1:]).strip()
    return "", block


def _same_sentence_intent(left: str, right: str) -> bool:
    left_words, right_words = _content_words(left), _content_words(right)
    if not left_words or not right_words:
        return False
    overlap = len(left_words & right_words) / max(1, min(len(left_words), len(right_words)))
    new_right = right_words - left_words
    return overlap >= 0.7 and len(new_right) <= 2 and _restates_comparison_limit(left, right)


def _restates_comparison_limit(left: str, right: str) -> bool:
    pair = f"{left} {right}".casefold()
    return bool(re.search(r"\bmore\s+\w+\s+than\b|\bmatters\s+more\s+than\b|\bnot\s+enough\s+by\s+itself\b|\bby\s+itself\b", pair))


def _content_words(sentence: str) -> set[str]:
    stop = {
        "about", "above", "after", "again", "against", "also", "because", "being",
        "between", "could", "does", "from", "have", "into", "itself", "more",
        "only", "should", "that", "their", "there", "these", "those", "through",
        "under", "when", "where", "which", "while", "would",
    }
    return {
        word.casefold()
        for word in re.findall(r"[A-Za-z][A-Za-z'’-]{3,}", sentence)
        if word.casefold() not in stop
    }


def _fragment_like(sentence: str) -> bool:
    value = sentence.strip(" .!?")
    lowered = value.casefold()
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", value)
    if lowered.endswith((" that", " because", " while", " when", " with", " by", " to", " of", " for")):
        return True
    if re.match(r"^(?:when|while|whilst|although|though|because|if)\b", lowered) and "," not in value:
        return True
    if re.match(r"^i\s+[a-z]+ly\s+[a-z]+ed$", lowered):
        return True
    if re.match(r"^(?:in|under|during|according to|by|with|for)\b", lowered) and len(words) <= 10 and not _has_finite_verb(lowered):
        return True
    if len(words) <= 5 and not _has_finite_verb(lowered):
        return True
    return bool(re.search(r"\b(?:the|a|an)\s+(?:includes|carries|applies)\b", lowered))


def _repair_trace_like(sentence: str) -> bool:
    lowered = sentence.strip(" .!?").casefold()
    return bool(re.search(r"\b(?:is the context|carries the same point|keeps both sides visible|same point)\b", lowered))


def _has_finite_verb(text: str) -> bool:
    return bool(re.search(r"\b(?:am|are|is|was|were|be|being|been|have|has|had|do|does|did|must|should|would|could|can|will|may|might|[a-z]+ed|[a-z]+s)\b", text))
