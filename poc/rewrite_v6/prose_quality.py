from __future__ import annotations

import re
from typing import Callable


def repair_generated_prose(text: str, source_text: str = "") -> str:
    return drop_redundant_adjacent_sentence_intent(
        _split_not_only_sentences(
            _split_identify_address_needs(
                _revoice_displayed_quality(
                    _restore_not_only_polarity(_repair_gerund_evidence_fragments(text), source_text)
                )
            )
        )
    )


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


def _repair_gerund_evidence_fragments(text: str) -> str:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    repaired: list[str] = []
    for block in paragraphs:
        heading, body = _split_heading(block)
        sentences = _sentences(body)
        rows: list[str] = []
        for sentence in sentences:
            if rows:
                sentence = _repair_gerund_sentence(sentence, rows[-1])
            rows.append(sentence)
        repaired.append("\n".join(part for part in [heading, " ".join(rows).strip()] if part).strip())
    return "\n\n".join(repaired).strip()


def _repair_gerund_sentence(sentence: str, previous: str) -> str:
    match = re.match(r"^(Demonstrating|Showing|Providing|Displaying)\s+(.+)$", sentence.strip(), flags=re.I)
    if not match:
        return sentence
    subject = _previous_actor(previous)
    if not subject:
        return sentence
    verb = {"demonstrating": "demonstrated", "showing": "showed", "providing": "provided", "displaying": "displayed"}[match.group(1).casefold()]
    return f"{subject} {verb} {match.group(2).strip()}"


def _previous_actor(sentence: str) -> str:
    if re.search(r"\b(?:he|his|him)\b", sentence, flags=re.I):
        return "He"
    if re.search(r"\b(?:she|her)\b", sentence, flags=re.I):
        return "She"
    match = re.search(r"\b([A-Z][A-Za-z'’-]{2,})\b", sentence)
    return match.group(1) if match else ""


def _restore_not_only_polarity(text: str, source_text: str) -> str:
    if "not only" not in str(source_text or "").casefold() or "not only" in str(text or "").casefold():
        return text
    source_sentences = [sentence for sentence in _sentences(source_text) if "not only" in sentence.casefold()]
    if not source_sentences:
        return text
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    repaired: list[str] = []
    for block in paragraphs:
        heading, body = _split_heading(block)
        rows = [_restore_not_only_sentence(sentence, source_sentences) for sentence in _sentences(body)]
        repaired.append("\n".join(part for part in [heading, " ".join(rows).strip()] if part).strip())
    return "\n\n".join(repaired).strip()


def _restore_not_only_sentence(sentence: str, source_sentences: list[str]) -> str:
    if not any(_shares_intent(sentence, source) for source in source_sentences):
        return sentence
    pattern = re.compile(r"^(.+?)\s+([A-Za-z]+s?)\s+as\s+(.+?)\s+and\s+(?:also\s+)?(.+)$", flags=re.I)
    match = pattern.match(sentence.strip())
    if not match:
        return sentence
    subject, verb, first_side, second_side = match.groups()
    helper, base = _not_only_helper_and_base(verb)
    return f"{subject} {helper} not only {base} as {first_side} but also {second_side}"


def _split_not_only_sentences(text: str) -> str:
    return _map_sentences(text, _split_not_only_sentence_density)


def _split_not_only_sentence_density(sentence: str) -> str:
    pattern = re.compile(r"^(.+?)\s+(do|does)\s+not\s+only\s+(.+?)\s+but\s+also\s+(.+)$", flags=re.I)
    match = pattern.match(sentence.strip())
    if not match or len(sentence.split()) < 16:
        return sentence
    subject, helper, first_side, second_side = match.groups()
    return f"{subject} {helper} not only {first_side}. {_followup_subject(subject)} also {second_side}"


def _revoice_displayed_quality(text: str) -> str:
    return _map_sentences(text, _revoice_displayed_quality_sentence)


def _revoice_displayed_quality_sentence(sentence: str) -> str:
    match = re.match(r"^During\s+(.+?)\s+(he|she|they)\s+displayed\s+a\s+high\s+degree\s+of\s+(.+)$", sentence.strip(), flags=re.I)
    if not match:
        return sentence
    context, actor, quality = match.groups()
    return f"{context[:1].upper() + context[1:]} showed {_possessive(actor)} {quality}"


def _split_identify_address_needs(text: str) -> str:
    return _map_sentences(text, _split_identify_address_sentence)


def _split_identify_address_sentence(sentence: str) -> str:
    pattern = re.compile(
        r"^(.+?)\s+(?:demonstrated\s+the\s+ability\s+to\s+)?(?:accurately\s+)?identify\s+and\s+gently\s+address\s+the\s+needs\s+of\s+(.+)$",
        flags=re.I,
    )
    match = pattern.match(sentence.strip())
    if not match:
        return sentence
    subject, target = match.groups()
    target = target.strip(" .")
    return f"{subject} identified the needs of {target}. {_subject_pronoun(subject)} gently addressed those needs."


def _map_sentences(text: str, mapper: Callable[[str], str]) -> str:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    rows: list[str] = []
    for block in paragraphs:
        heading, body = _split_heading(block)
        mapped = " ".join(mapper(sentence).strip() for sentence in _sentences(body)).strip()
        rows.append("\n".join(part for part in [heading, mapped] if part).strip())
    return "\n\n".join(rows).strip()


def _subject_pronoun(subject: str) -> str:
    lowered = subject.casefold()
    if re.search(r"\b(?:cards|they|learners|students|clients|teachers|educators|groups)\b", lowered):
        return "They"
    if re.search(r"\b(?:she|her)\b", lowered):
        return "She"
    if re.search(r"\b(?:he|his|him|johnny|learner|student|teacher|educator)\b", lowered):
        return "He"
    return "It"


def _followup_subject(subject: str) -> str:
    if re.search(r"\b(?:cards|they|learners|students|clients|teachers|educators|groups)\b", subject, flags=re.I):
        return "The same " + (re.findall(r"\b(cards|learners|students|clients|teachers|educators|groups)\b", subject, flags=re.I)[-1].casefold())
    return _subject_pronoun(subject)


def _possessive(actor: str) -> str:
    return {"he": "his", "she": "her", "they": "their"}.get(actor.casefold(), "their")


def _not_only_helper_and_base(verb: str) -> tuple[str, str]:
    lowered = verb.casefold()
    if lowered.endswith("s") and lowered not in {"is", "was", "has"}:
        return "does", verb[:-1]
    return "do", verb


def _shares_intent(sentence: str, source: str) -> bool:
    left, right = _content_words(sentence), _content_words(source)
    return bool(left and right) and len(left & right) / max(1, min(len(left), len(right))) >= 0.35


def has_fragment_or_trace_sentences(text: str) -> bool:
    sentences = _sentences(text)
    if not sentences:
        return True
    if any(_hard_fragment_like(sentence) for sentence in sentences):
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
    if re.match(r"^(?:which|where|that)\b", lowered) and len(words) <= 18:
        return True
    if re.match(r"^(?:and|but|or)\s+(?:need|needs|require|requires|must|can|could|should|would|is|are|was|were|has|have|had|do|does|did)\b", lowered):
        return True
    if re.match(r"^(?:when|while|whilst|although|though|because|if)\b", lowered) and "," not in value:
        return True
    if re.match(r"^as\b", lowered) and "," not in value:
        return True
    if re.match(r"^i\s+[a-z]+ly\s+[a-z]+ed$", lowered):
        return True
    if re.match(r"^(?:in|under|during|according to|by|with|for)\b", lowered) and len(words) <= 10 and not _has_finite_verb(lowered):
        return True
    if len(words) <= 5 and not _has_finite_verb(lowered):
        return True
    return bool(re.search(r"\b(?:the|a|an)\s+(?:includes|carries|applies)\b", lowered))


def _hard_fragment_like(sentence: str) -> bool:
    value = sentence.strip(" .!?")
    lowered = value.casefold()
    if re.match(r"^(?:which|where|that)\b", lowered):
        return True
    if re.match(r"^(?:and|but|or)\s+(?:need|needs|require|requires|must|can|could|should|would|is|are|was|were|has|have|had|do|does|did)\b", lowered):
        return True
    return bool(re.match(r"^as\b", lowered) and "," not in value)


def _repair_trace_like(sentence: str) -> bool:
    lowered = sentence.strip(" .!?").casefold()
    return bool(re.search(r"\b(?:is the context|carries the same point|keeps both sides visible|same point)\b", lowered))


def _has_finite_verb(text: str) -> bool:
    return bool(re.search(r"\b(?:am|are|is|was|were|be|being|been|have|has|had|do|does|did|must|should|would|could|can|will|may|might|[a-z]+ed|[a-z]+s)\b", text))
