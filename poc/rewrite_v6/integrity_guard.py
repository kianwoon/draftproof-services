from __future__ import annotations

import re


def candidate_integrity_blockers(text: str) -> list[str]:
    blockers: list[str] = []
    visible = _normalize(text)
    if _broken_citation_shape(visible):
        blockers.append("broken_citation_shape")
    if _citation_name_wrapper(visible):
        blockers.append("citation_name_wrapper")
    if _dangling_article_predicate(visible):
        blockers.append("dangling_article_predicate")
    if _subject_verb_agreement_error(visible):
        blockers.append("malformed_subject_verb_agreement")
    if _split_negation_fragment(visible):
        blockers.append("split_negation_fragment")
    if _malformed_verb_complement(visible):
        blockers.append("malformed_verb_complement")
    return blockers


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _broken_citation_shape(text: str) -> bool:
    if text.count("(") != text.count(")"):
        return True
    return bool(
        re.search(r"\([A-Z][A-Za-z]+(?:\s+et\s+al)?\.\s+[A-Z]|\([A-Z][A-Za-z]+(?:\s+et\s+al)?\.\s+(?:They|It|This|The)\b", text)
        or re.search(r"\([A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?\s+(?:19|20)\d{2}[a-z]?(?:\s*[;)])", text)
    )


def _citation_name_wrapper(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:from|by|with|according to)\s+[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?\s*\((?:19|20)\d{2}[a-z]?\)\s+"
            r"(?:aligns?|supports?|shows?|confirms?|proves?|highlights?|explains?)\b",
            text,
        )
    )


def _dangling_article_predicate(text: str) -> bool:
    return bool(re.search(r"(?:^|[.!?]\s+)The\s+(?:involves?|created|creates|helps?|shows?|supports?|requires?|guides?)\b", text, flags=re.I))


def _subject_verb_agreement_error(text: str) -> bool:
    plural_subject = r"(?:I|we|they|students|learners|teachers|educators|schools|teams|classes|people)"
    singular_s_verb = r"(?:carries|involves|creates|requires|supports|guides|explains|shows|makes|helps)"
    return bool(re.search(rf"\b{plural_subject}\s+(?:also\s+)?{singular_s_verb}\b", text, flags=re.I))


def _split_negation_fragment(text: str) -> bool:
    return bool(re.search(r"\b(?:not|no)\.\s+(?:They|It|This|The|Students|Learners)\b", text, flags=re.I))


def _malformed_verb_complement(text: str) -> bool:
    return bool(re.search(r"\b(?:learned|learn|guide|guides|guided|question)\s+(?:accepting|compare|develop|apply|create|created|creating)\b", text, flags=re.I))
