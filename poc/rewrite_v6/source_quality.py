from __future__ import annotations

import re

from .text import Paragraph


def scope_marker_reused_as_content(candidate_text: str, paragraph: Paragraph) -> bool:
    source = str(paragraph.text or "").casefold()
    candidate = str(candidate_text or "").casefold()
    if "no longer" not in source:
        return False
    for match in re.finditer(r"\blonger\b", candidate):
        before = candidate[max(0, match.start() - 4):match.start()]
        if not before.endswith("no "):
            return True
    return False


def unsupported_semantic_padding(candidate_text: str, paragraph: Paragraph) -> bool:
    source_bases = _content_bases(paragraph.text)
    if not source_bases:
        return False
    source_sentences = [sentence.text for sentence in paragraph.sentences if sentence.text.strip()]
    for sentence in _candidate_sentences(candidate_text):
        candidate_tokens = _content_tokens(sentence)
        if len(candidate_tokens) < 8:
            continue
        if _max_unsupported_run(candidate_tokens, source_bases) >= 5:
            return True
        matched_source = _best_source_sentence(candidate_tokens, source_sentences)
        if not matched_source:
            continue
        matched_bases = _content_bases(matched_source)
        if len(matched_bases) < 5:
            continue
        overlap = sum(1 for token in candidate_tokens if _token_base(token) in matched_bases)
        if overlap < 5:
            continue
        if _max_unsupported_run(candidate_tokens, source_bases) >= 5:
            return True
    return False


def _max_unsupported_run(tokens: list[str], source_bases: set[str]) -> int:
    unsupported_run = 0
    max_run = 0
    for token in tokens:
        base = _token_base(token)
        if base not in source_bases:
            unsupported_run += 1
            max_run = max(max_run, unsupported_run)
        else:
            unsupported_run = 0
    return max_run


def source_quality_blockers(candidate_text: str, paragraph: Paragraph) -> list[str]:
    blockers: list[str] = []
    if scope_marker_reused_as_content(candidate_text, paragraph):
        blockers.append("source_scope_marker_reused_as_content")
    if unsupported_source_channel_list(candidate_text, paragraph):
        blockers.append("unsupported_semantic_padding")
    if source_contrast_reframed(candidate_text, paragraph):
        blockers.append("source_contrast_reframed")
    return blockers


def unsupported_source_channel_list(candidate_text: str, paragraph: Paragraph) -> bool:
    source_bases = _content_bases(paragraph.text)
    if not source_bases:
        return False
    for sentence in _candidate_sentences(candidate_text):
        lowered = sentence.casefold()
        marker = re.search(r"\b(?:include|includes|including|such\s+as|range\s+from|ranges\s+from|from)\b", lowered)
        if not marker:
            continue
        tail = sentence[marker.end():]
        tokens = _content_tokens(tail)
        if len(tokens) < 5:
            continue
        unsupported = [token for token in tokens if _token_base(token) not in source_bases]
        supported = [token for token in tokens if _token_base(token) in source_bases]
        if len(unsupported) >= 4 and len(unsupported) > len(supported):
            return True
    return False


def source_contrast_reframed(candidate_text: str, paragraph: Paragraph) -> bool:
    source = str(paragraph.text or "").casefold()
    candidate = str(candidate_text or "").casefold()
    if "not less important" not in source:
        return False
    return bool(re.search(r"\bmore\s+important\s+and\s+less\s+(?:overlooked|ignored|visible|valued)\b", candidate))


def _best_source_sentence(candidate_tokens: list[str], source_sentences: list[str]) -> str:
    candidate_bases = {_token_base(token) for token in candidate_tokens}
    best_sentence = ""
    best_overlap = 0
    for sentence in source_sentences:
        bases = _content_bases(sentence)
        overlap = len(candidate_bases & bases)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
    return best_sentence


def _candidate_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]


def _content_bases(text: str) -> set[str]:
    return {_token_base(token) for token in _content_tokens(text)}


def _content_tokens(text: str) -> list[str]:
    stop = {
        "about", "above", "after", "again", "against", "also", "being", "between", "could",
        "during", "each", "every", "from", "have", "into", "just", "know", "more", "most",
        "only", "other", "over", "quite", "rather", "should", "some", "still", "their",
        "there", "these", "those", "through", "under", "where", "which", "while", "will",
        "would", "than", "that", "this", "they", "with", "many", "because", "however",
        "therefore", "does", "what", "then", "today", "past", "system",
    }
    return [
        token.casefold()
        for token in re.findall(r"\b[A-Za-z][A-Za-z'-]{3,}\b", str(text or ""))
        if token.casefold() not in stop
    ]


def _token_base(token: str) -> str:
    value = str(token or "").casefold().strip("'")
    if len(value) > 5 and value.endswith("ing"):
        stem = value[:-3]
        return stem + "e" if not stem.endswith("e") else stem
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith("s"):
        return value[:-1]
    return value
