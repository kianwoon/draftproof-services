from __future__ import annotations

import re

from .text import Paragraph, source_terms


def coverage_ratio(text: str, paragraph: Paragraph) -> float:
    anchors = source_terms(paragraph.text, limit=24)
    lowered = str(text or "").casefold()
    return sum(1 for anchor in anchors if anchor.casefold() in lowered) / len(anchors) if anchors else 1.0


def missing_required_source_terms(text: str, paragraph: Paragraph) -> bool:
    candidate_terms = {_word_base(term) for term in source_terms(text, limit=160)}
    for sentence in paragraph.sentences:
        terms = _required_list_terms(sentence.text)
        if not terms:
            continue
        covered = sum(1 for term in terms if _word_base(term) in candidate_terms)
        if covered / len(terms) < 0.75:
            return True
    return False


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
    return [term for terms in part_terms for term in terms]

def _word_base(word: str) -> str:
    return str(word or "").casefold().removesuffix("'s").rstrip("s")
