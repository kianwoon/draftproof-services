from __future__ import annotations

import re


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
