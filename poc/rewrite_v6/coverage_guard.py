from __future__ import annotations

import re

from .text import Paragraph, source_terms, strip_leading_heading as _strip_leading_heading


def coverage_ratio(text: str, paragraph: Paragraph) -> float:
    anchors = source_terms(paragraph.text, limit=24)
    lowered = str(text or "").casefold()
    return sum(1 for anchor in anchors if anchor.casefold() in lowered) / len(anchors) if anchors else 1.0


def missing_required_source_terms(text: str, paragraph: Paragraph) -> bool:
    return bool(missing_required_source_term_details(text, paragraph))


def missing_required_source_term_details(text: str, paragraph: Paragraph) -> list[str]:
    candidate_terms = _term_bases(source_terms(text, limit=160))
    candidate_numbers = set(_number_anchors(text))
    missing: list[str] = []
    missing.extend(anchor for anchor in _number_anchors(paragraph.text) if anchor not in candidate_numbers)
    for sentence in paragraph.sentences:
        terms = _required_list_terms(_strip_leading_heading(sentence.text))
        if not terms:
            continue
        covered = sum(1 for term in terms if _word_base(term) in candidate_terms)
        required_ratio = 1.0 if len(terms) <= 4 else 0.75
        if covered / len(terms) < required_ratio:
            missing.extend(term for term in terms if _word_base(term) not in candidate_terms)
    return list(dict.fromkeys(missing))


def _number_anchors(text: str) -> list[str]:
    pattern = r"\b\d+(?:\.\d+)?\b"
    return [match.group(0).casefold() for match in re.finditer(pattern, str(text or ""), flags=re.I)]


def _term_bases(terms: list[str]) -> set[str]:
    bases: set[str] = set()
    for term in terms:
        base = _word_base(term)
        if base:
            bases.add(base)
        for part in re.split(r"[-/]", _normalize_hyphen(str(term or ""))):
            part_base = _word_base(part)
            if part_base:
                bases.add(part_base)
    return bases


def _required_list_terms(text: str) -> list[str]:
    visible = str(text or "")
    normalized = re.sub(r",\s+(?:and|or)\s+", ", ", visible, flags=re.I)
    parts = [part.strip(" .,!?:;") for part in re.split(r",\s+|;\s+|\s+\band\b\s+", normalized, flags=re.I) if part.strip()]
    if len(parts) < 3:
        return []
    part_terms = [source_terms(part, limit=4) for part in parts]
    if len(part_terms) >= 4 and len(part_terms[0]) == 1 and len(part_terms[1]) > 1 and all(len(terms) == 1 for terms in part_terms[2:]):
        part_terms = [part_terms[1][-1:], *part_terms[2:]]
    if part_terms and len(part_terms[0]) > 1 and all(len(terms) == 1 for terms in part_terms[1:]):
        part_terms[0] = part_terms[0][-1:]
    if len(parts) < 4 and not all(len(terms) == 1 for terms in part_terms):
        return []
    return [term for terms in part_terms for term in terms if not _low_value_required_term(term)]


def _low_value_required_term(term: str) -> bool:
    return _word_base(term) in {
        "also",
        "another",
        "beginning",
        "call",
        "each",
        "know",
        "later",
        "one",
        "quite",
        "some",
        "still",
        "will",
    }

def _word_base(word: str) -> str:
    value = _normalize_hyphen(str(word or "")).casefold().removesuffix("'s").strip("-")
    irregular = {
        "found": "find",
        "learnt": "learn",
        "learned": "learn",
    }
    if value in irregular:
        return irregular[value]
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ied"):
        value = value[:-3] + "y"
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    return value.rstrip("s")


def _normalize_hyphen(text: str) -> str:
    return re.sub(r"[\u2010-\u2015\u2212]", "-", str(text or ""))
