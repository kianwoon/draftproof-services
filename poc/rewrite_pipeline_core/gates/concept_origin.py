"""Concept-origin validation helpers.

These helpers are domain-agnostic.  They check whether candidate concepts are
supported by nearby source context without hardcoding any specific subject.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from difflib import SequenceMatcher


_CONCEPT_ORIGIN_STOPWORDS = {
    "about", "above", "across", "after", "again", "against", "almost", "along",
    "also", "although", "always", "among", "another", "around", "because",
    "before", "being", "below", "between", "both", "cannot", "could", "does",
    "doing", "done", "each", "either", "else", "enough", "even", "every",
    "everything", "from", "further", "general", "have", "having", "here",
    "however", "into", "itself", "just", "like", "many", "might", "more",
    "most", "much", "must", "need", "needed", "needs", "only", "other",
    "others", "over", "same", "should", "since", "some", "still", "such",
    "than", "that", "their", "them", "then", "there", "these", "they",
    "thing", "things", "this", "those", "through", "under", "until", "very",
    "what", "when", "where", "which", "while", "with", "within", "without",
    "would", "important", "significant", "major", "modern", "strong",
    "global", "different", "various", "clear", "useful", "practical",
    "actual", "really", "simple", "simply", "point", "points", "issue",
    "issues", "question", "questions", "condition", "conditions", "case",
    "cases", "example", "examples", "process", "reason", "reasons",
    "claim", "claims", "general", "limit", "limited", "connect", "connected",
    "relate", "related", "paragraph", "statement", "stated", "wide", "wider",
    "treat", "treated",
}


def _concept_origin_normalize_term(term: str) -> str:
    value = re.sub(r"[^a-z0-9]", "", str(term or "").lower())
    if len(value) <= 3:
        return ""
    if value.endswith("ies") and len(value) > 5:
        value = value[:-3] + "y"
    elif value.endswith(("ing", "ers")) and len(value) > 6:
        value = value[:-3]
    elif value.endswith(("ed", "es")) and len(value) > 5:
        value = value[:-2]
    elif value.endswith("s") and len(value) > 5:
        value = value[:-1]
    if len(value) <= 3 or value in _CONCEPT_ORIGIN_STOPWORDS:
        return ""
    return value


def _concept_origin_terms(text: str) -> set[str]:
    """Return lightweight content terms for concept-origin validation."""
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'_-]{2,}", str(text or "")):
        normalized = _concept_origin_normalize_term(token)
        if normalized:
            terms.add(normalized)
    return terms


def _ordered_concept_origin_terms(text: str, *, limit: int = 6) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'_-]{2,}", str(text or "")):
        normalized = _concept_origin_normalize_term(token)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
        if len(ordered) >= limit:
            break
    return ordered


def _concept_origin_protected_terms(text: str) -> set[str]:
    protected = set()
    for token in re.findall(r"\b\d+(?:[.,]\d+)?%?\b", str(text or "")):
        normalized = _concept_origin_normalize_term(token)
        if normalized:
            protected.add(normalized)
    for match in re.findall(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4}\b", str(text or "")):
        for token in re.findall(r"[A-Za-z0-9]+", match):
            normalized = _concept_origin_normalize_term(token)
            if normalized:
                protected.add(normalized)
    for token in re.findall(r"\b[A-Z]{2,}[A-Z0-9-]*\b", str(text or "")):
        normalized = _concept_origin_normalize_term(token)
        if normalized:
            protected.add(normalized)
    return protected


def _best_source_paragraph_index(candidate_paragraph: str, source_paragraphs: list[str]) -> int:
    if not source_paragraphs:
        return 0
    best_index = 0
    best_ratio = -1.0
    candidate = str(candidate_paragraph or "").lower()
    for index, source_paragraph in enumerate(source_paragraphs):
        ratio = SequenceMatcher(None, candidate, str(source_paragraph or "").lower(), autojunk=False).ratio()
        if ratio > best_ratio:
            best_index = index
            best_ratio = ratio
    return best_index


def _candidate_concept_origin_reject_reason(
    source_text: str,
    candidate_text: str,
    *,
    logical_paragraphs: Callable[[str], list[str]],
    split_sentences: Callable[[str], list[str]],
    unsupported_term_limit: int = 4,
    unsupported_sentence_limit: int = 3,
) -> str:
    """Reject candidates that import unsupported concepts into changed regions."""
    source = str(source_text or "").strip()
    candidate = str(candidate_text or "").strip()
    if not source or not candidate or source == candidate:
        return ""
    source_paragraphs = logical_paragraphs(source)
    candidate_paragraphs = logical_paragraphs(candidate)
    if not source_paragraphs or not candidate_paragraphs:
        return ""
    protected_terms = _concept_origin_protected_terms(source)
    full_source_terms = _concept_origin_terms(source)
    for candidate_index, paragraph in enumerate(candidate_paragraphs):
        paragraph = str(paragraph or "").strip()
        if not paragraph:
            continue
        if any(SequenceMatcher(None, paragraph.lower(), src.lower(), autojunk=False).ratio() >= 0.96 for src in source_paragraphs):
            continue
        source_index = _best_source_paragraph_index(paragraph, source_paragraphs)
        support_parts = []
        for support_index in range(max(0, source_index - 1), min(len(source_paragraphs), source_index + 2)):
            support_parts.append(source_paragraphs[support_index])
        support_text = " ".join(support_parts)
        local_terms = _concept_origin_terms(support_text) | protected_terms
        candidate_terms = _concept_origin_terms(paragraph)
        unsupported = sorted(
            term for term in (candidate_terms - local_terms)
            if term not in protected_terms
        )
        if len(unsupported) >= unsupported_term_limit:
            return (
                "unsupported_concept_origin "
                f"paragraph={candidate_index} source_paragraph={source_index} "
                f"terms={','.join(unsupported[:8])}"
            )
        source_sentences = split_sentences(support_text)
        for sentence in split_sentences(paragraph):
            stripped_sentence = str(sentence or "").strip()
            if len(stripped_sentence.split()) < 7:
                continue
            if any(
                SequenceMatcher(None, stripped_sentence.lower(), src_sentence.lower(), autojunk=False).ratio() >= 0.82
                for src_sentence in source_sentences
            ):
                continue
            sentence_terms = _concept_origin_terms(stripped_sentence)
            unsupported_sentence_terms = sorted(
                term for term in (sentence_terms - local_terms)
                if term not in protected_terms and term not in full_source_terms
            )
            if len(unsupported_sentence_terms) >= unsupported_sentence_limit:
                return (
                    "unsupported_concept_origin_sentence "
                    f"paragraph={candidate_index} source_paragraph={source_index} "
                    f"terms={','.join(unsupported_sentence_terms[:8])}"
                )
    return ""
