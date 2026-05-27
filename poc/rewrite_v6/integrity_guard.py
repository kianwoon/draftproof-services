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
    if _unsupported_learner_blame_shape(visible):
        blockers.append("unsupported_learner_blame_shape")
    if _bare_instruction_fragment(visible):
        blockers.append("bare_instruction_fragment")
    if _citation_report_sentence(visible):
        blockers.append("citation_report_sentence")
    if _stranded_thereby_citation(visible):
        blockers.append("stranded_thereby_citation")
    if _planner_language_leakage(visible):
        blockers.append("planner_language_leakage")
    if _malformed_serial_verb_chain(visible):
        blockers.append("malformed_serial_verb_chain")
    if _malformed_nominal_stack(visible):
        blockers.append("malformed_nominal_stack")
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


def _unsupported_learner_blame_shape(text: str) -> bool:
    human_group = r"(?:learners|students|clients|people|participants)"
    return bool(
        re.search(rf"\b{human_group}\s+(?:become|became|are|were)\s+difficult\b", text, flags=re.I)
        or re.search(rf"\b{human_group}\s+are\s+unwilling\s+to\s+learn\s+because\b", text, flags=re.I)
    )


def _bare_instruction_fragment(text: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[.!?]\s+)(?:Help|Promote|Enable|Teach|Apply|Use|Encourage)\s+"
            r"(?:them|students|learners|clients|inclusive|practical|the)\b",
            text,
        )
    )


def _citation_report_sentence(text: str) -> bool:
    return bool(
        re.search(r"(?:^|[.!?]\s+)The same outcome was observed in later studies\s*\([^)]+\)\.?", text)
        or re.search(r"(?:^|[.!?]\s+)Further evidence supports this finding\s*\([^)]+\)\.?", text)
    )


def _stranded_thereby_citation(text: str) -> bool:
    return bool(re.search(r"\bthereby\s*\([^)]+\)", text, flags=re.I))


def _planner_language_leakage(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:coverage\s+beat|source\s+slot|source_sentence_id|writer_execution_contract|"
            r"construction\s+recipe|route\s+question|active\s+variant|planner\s+decision|"
            r"relationship\s+sees|beat\s+plan|coverage\s+capsule|source\s+sentence)\b",
            text,
            flags=re.I,
        )
        or re.search(
            r"(?:^|[.!?]\s+)(?:guide|compare|develop|apply|focus|think|words)\s+"
            r"(?:prompts?|occurs?|expands?|shapes?|relationship|route|function)\b",
            text,
            flags=re.I,
        )
        or re.search(
            r"(?:^|[.!?]\s+)(?:shift|teacher|good\s+teacher)\s+"
            r"(?:has|is|helps?)\b",
            text,
            flags=re.I,
        )
        or re.search(r"\b(?:source\s+groups|education\s+today\s+emphasizes\s+the\s+words|words\s+students\s+learn)\b", text, flags=re.I)
    )


def _malformed_serial_verb_chain(text: str) -> bool:
    return bool(
        re.search(r"\bthink\s+deeply\s+solve\s+problems\b", text, flags=re.I)
        or re.search(r"\b(?:analyse|analyze)\s+adapt\s+communicate\b", text, flags=re.I)
    )


def _malformed_nominal_stack(text: str) -> bool:
    return bool(re.search(r"\busefulness\s+ethics\s+trustworthiness\b", text, flags=re.I))
