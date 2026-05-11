"""Document-level quality guard for rewrite controller candidates."""

from __future__ import annotations

import re
from typing import Any


_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_COMMON_LOWERCASE_STARTS = {
    "a",
    "an",
    "and",
    "as",
    "but",
    "for",
    "if",
    "in",
    "it",
    "its",
    "of",
    "or",
    "so",
    "the",
    "this",
    "to",
    "with",
}


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(str(text or "")) if match.group(0).strip()]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text or "")))


def _sentence_stem(sentence: str) -> str:
    return re.sub(r"^[\"'(\[]+", "", str(sentence or "").strip())


def evaluate_text_quality_regression(source_text: str, candidate_text: str, *, changed_sentence_ratio: float | None = None) -> dict[str, Any]:
    """Catch malformed or overly stitched rewrite artifacts before selection."""

    reject_reasons: list[str] = []
    warnings: list[str] = []
    source_sentences = _sentences(source_text)
    candidate_sentences = _sentences(candidate_text)
    source_sentence_set = {re.sub(r"\s+", " ", item).strip() for item in source_sentences}

    if changed_sentence_ratio is not None and changed_sentence_ratio > 0.22:
        reject_reasons.append(f"patchwork_budget_exceeded {changed_sentence_ratio:.3f}>0.220")

    for index, sentence in enumerate(candidate_sentences):
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if not normalized or normalized in source_sentence_set:
            continue
        stem = _sentence_stem(normalized)
        words = re.findall(r"\b[\w'-]+\b", stem)
        if not words:
            continue
        first = words[0]
        lower_first = first.lower()
        if first[:1].islower() and lower_first in _COMMON_LOWERCASE_STARTS:
            reject_reasons.append(f"lowercase_orphan_sentence_s{index + 1}")
        if _word_count(stem) <= 4 and lower_first in _COMMON_LOWERCASE_STARTS:
            reject_reasons.append(f"orphan_sentence_fragment_s{index + 1}")
        if len(words) >= 4 and words[0].lower() == words[-1].lower() and len(words[0]) > 3:
            warnings.append(f"circular_sentence_word_s{index + 1}")
        if len(words) >= 5 and words[0].lower() in {word.lower() for word in words[3:8]} and len(words[0]) > 3:
            reject_reasons.append(f"repeated_opening_word_artifact_s{index + 1}")
        if re.search(r"\b(\w{3,})\s+\1\b", stem, flags=re.I):
            reject_reasons.append(f"duplicate_adjacent_token_s{index + 1}")

    if re.search(r"[.!?]\s+[a-z]{2,}\b", str(candidate_text or "")):
        reject_reasons.append("lowercase_sentence_boundary_artifact")

    return {
        "passed": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
        "warnings": sorted(set(warnings)),
        "changed_sentence_ratio": changed_sentence_ratio,
        "source_sentence_count": len(source_sentences),
        "candidate_sentence_count": len(candidate_sentences),
    }
