from __future__ import annotations

import re
from typing import Callable


def repair_generated_prose(text: str, source_text: str = "") -> str:
    return drop_redundant_adjacent_sentence_intent(
        _repair_parenthetical_citation_wrappers(
            _remove_meta_reference_padding(
                _repair_pronoun_simplification(
                    _merge_example_when_fragments(
                        _split_thereby_consequence(
                            _split_not_only_sentences(
                                _split_help_and_consequence(
                                    _split_identify_address_needs(
                                        _revoice_displayed_quality(
                                            _revoice_guidance_understanding(
                                                _revoice_broad_belief_recommendations(
                                            _restore_not_only_polarity(_merge_source_term_fragments(_repair_gerund_evidence_fragments(text)), source_text),
                                            source_text,
                                        )
                                    )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            source_text,
        )
    )


def _merge_source_term_fragments(text: str) -> str:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    repaired: list[str] = []
    for block in paragraphs:
        heading, body = _split_heading(block)
        sentences = _sentences(body)
        rows: list[str] = []
        index = 0
        while index < len(sentences):
            current = sentences[index].strip()
            fragments: list[str] = []
            next_index = index + 1
            while next_index < len(sentences) and _source_term_fragment(sentences[next_index]):
                fragments.append(_normalise_fragment_item(sentences[next_index]))
                next_index += 1
            if fragments:
                rows.append(_append_fragment_list(current, fragments))
                index = next_index
                continue
            rows.append(current)
            index += 1
        repaired.append("\n".join(part for part in [heading, " ".join(rows).strip()] if part).strip())
    return "\n\n".join(repaired).strip()


def _source_term_fragment(sentence: str) -> bool:
    value = sentence.strip(" .!?")
    lowered = value.casefold()
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", value)
    if not words or len(words) > 5:
        return False
    if re.match(r"^(?:and|or)\s+", lowered):
        return True
    clear_predicate = re.search(
        r"\b(?:am|are|is|was|were|be|being|been|have|has|had|do|does|did|learn|learns|create|creates|created|make|makes|made|help|helps|support|supports|provide|provides|offer|offers)\b",
        lowered,
    )
    return not bool(clear_predicate)


def _normalise_fragment_item(sentence: str) -> str:
    value = sentence.strip(" .!?")
    value = re.sub(r"^(?:and|or)\s+", "", value, flags=re.I).strip()
    if value.isupper() or re.search(r"\b[A-Z]{2,}\b", value) or any(ch.isupper() for ch in value[1:]):
        return value
    return value[:1].lower() + value[1:]


def _append_fragment_list(sentence: str, fragments: list[str]) -> str:
    base = sentence.strip().rstrip(".!?")
    if not fragments:
        return sentence
    if len(fragments) == 1:
        return f"{base} and {fragments[0]}."
    return f"{base}, {', '.join(fragments[:-1])} and {fragments[-1]}."


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


def _remove_meta_reference_padding(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"\bThis\s+(example|experience|activity|event|result|outcome)\b", lambda match: "The " + match.group(1), value)
    value = re.sub(r"\bThese\s+(examples|experiences|activities|events|results|outcomes)\b", lambda match: "The " + match.group(1), value)
    value = re.sub(r"\bThe\s+earlier\s+(experience|example|activity|event)\b", r"The \1", value)
    value = re.sub(r"\bthe\s+earlier\s+(experience|example|activity|event)\b", r"the \1", value)
    value = re.sub(r"\b(?:described|mentioned|noted)\s+above\b", "", value, flags=re.I)
    return re.sub(r"\s{2,}", " ", value).strip()


def _merge_example_when_fragments(text: str) -> str:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    repaired: list[str] = []
    for block in paragraphs:
        heading, body = _split_heading(block)
        sentences = _sentences(body)
        rows: list[str] = []
        index = 0
        while index < len(sentences):
            current = sentences[index].strip()
            following = sentences[index + 1].strip() if index + 1 < len(sentences) else ""
            merged = _merge_example_when_pair(current, following)
            if merged:
                rows.append(merged)
                index += 2
                continue
            rows.append(current)
            index += 1
        repaired.append("\n".join(part for part in [heading, " ".join(rows).strip()] if part).strip())
    return "\n\n".join(repaired).strip()


def _merge_example_when_pair(current: str, following: str) -> str:
    if not following:
        return ""
    current = _normalize_example_prefix(current)
    match = re.match(r"^(when|while|if|because|although|though)\s+(.+)$", current.strip(" .!?"), flags=re.I)
    if not match:
        return ""
    conjunction, condition = match.groups()
    if "," in condition or _has_late_main_clause(f"{conjunction} {condition}".casefold()):
        return ""
    return f"{conjunction[:1].upper() + conjunction[1:].lower()} {condition.strip()}, {following[:1].lower() + following[1:]}"


def _repair_pronoun_simplification(text: str) -> str:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n+", str(text or "")) if block.strip()]
    repaired: list[str] = []
    for block in paragraphs:
        heading, body = _split_heading(block)
        rows: list[str] = []
        for sentence in _sentences(body):
            rows.append(_repair_pronoun_simplification_sentence(sentence, rows[-1] if rows else ""))
        repaired.append("\n".join(part for part in [heading, " ".join(rows).strip()] if part).strip())
    return "\n\n".join(repaired).strip()


def _repair_pronoun_simplification_sentence(sentence: str, previous: str) -> str:
    match = re.match(
        r"^It\s+applies\s+a\s+simplification\s+of\s+(.+?),\s*turning\s+(?:it|them|those|these)\s+into\s+(.+)$",
        sentence.strip(),
        flags=re.I,
    )
    if not match:
        return sentence
    source, target = match.groups()
    subject = _simple_sentence_subject(previous) or _sentence_subject(previous) or "The method"
    return f"{subject} simplifies {source.strip()} into {target.strip()}"


def _simple_sentence_subject(sentence: str) -> str:
    match = re.match(r"^((?:The|A|An|My|Our|His|Her|Their)\s+[A-Za-z][A-Za-z'’-]*)\b", sentence.strip(), flags=re.I)
    return match.group(1) if match else ""


def _revoice_broad_belief_recommendations(text: str, source_text: str) -> str:
    if not re.search(r"\bmy\s+teaching\s+method\b", str(source_text or ""), flags=re.I):
        return text
    return _map_sentences(text, _revoice_broad_belief_sentence)


def _revoice_broad_belief_sentence(sentence: str) -> str:
    match = re.match(
        r"^I\s+believe\s+(?:that\s+)?educators\s+should\s+(?:use|adopt)\s+(.+?)\s+to\s+support\s+(.+?)(?:,\s*in\s+practice)?\.?$",
        sentence.strip(),
        flags=re.I,
    )
    if not match:
        return sentence
    approach, target = match.groups()
    approach = approach.strip(" .")
    target = target.strip(" .")
    return f"My teaching method uses {approach} to support {target}."

def _revoice_guidance_understanding(text: str) -> str:
    return _map_sentences(text, _revoice_guidance_understanding_sentence)


def _revoice_guidance_understanding_sentence(sentence: str) -> str:
    sentence = _normalize_example_prefix(sentence)
    match = re.match(
        r"^(?:An\s+example\s+shows\s+that\s+)?When\s+I\s+guide\s+them\s+to\s+(?:a|the)?\s*(.+?)(?:,)?\s+they\s+(?:can\s+)?(?:readily\s+|easily\s+)?(?:understand|see)\s+(?:(where\s+to)\s+)?(.+?)(?:\s+(?:directly|straightaway))?(?:\s+as\s+an\s+example)?\.?$",
        sentence.strip(),
        flags=re.I,
    )
    if not match:
        return sentence
    anchor, where_to, action = match.groups()
    relation = "where to " if where_to else ""
    return f"The {anchor.strip()} shows {relation}{action.strip()}."


def _normalize_example_prefix(sentence: str) -> str:
    return re.sub(
        r"^\s*For\s+example\s*(?:[,;:\-–—]\s*|\s+)",
        "",
        str(sentence or "").strip(),
        count=1,
        flags=re.I,
    )


def _repair_parenthetical_citation_wrappers(text: str, source_text: str) -> str:
    source_citations = _citation_items(source_text)
    if not source_citations:
        return text
    sentences = _sentences(text)
    if not sentences:
        return text
    kept: list[str] = []
    removed: list[str] = []
    report_pattern = re.compile(
        r"^[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?\s*\((?:19|20)\d{2}\)\s+(?:reports?|reported|observed|documented|noted|supports?|supported|highlights?|highlighted|confirms?|proves?)\b",
        flags=re.I,
    )
    for sentence in sentences:
        if report_pattern.search(sentence.strip()):
            removed.extend(_citation_items(sentence))
        else:
            kept.append(sentence)
    if not removed or not kept:
        return text
    return _ensure_citation_span(" ".join(kept), source_citations)


def _citation_items(text: str) -> list[str]:
    rows: list[str] = []
    for span in re.findall(r"\(([^)]*(?:19|20)\d{2}[^)]*)\)", str(text or "")):
        for part in span.split(";"):
            item = part.strip()
            if re.search(r"(?:19|20)\d{2}", item):
                rows.append(item)
    return rows


def _ensure_citation_span(text: str, citations: list[str]) -> str:
    citation_span = "; ".join(dict.fromkeys(citations))
    if not citation_span:
        return text
    if re.search(r"\([^)]*(?:19|20)\d{2}[^)]*\)", text):
        return re.sub(r"\([^)]*(?:19|20)\d{2}[^)]*\)", f"({citation_span})", text, count=1)
    return re.sub(r"\.\s*$", f" ({citation_span}).", text.strip())


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


def _split_help_and_consequence(text: str) -> str:
    return _map_sentences(text, _split_help_and_consequence_sentence)


def _split_help_and_consequence_sentence(sentence: str) -> str:
    pattern = re.compile(
        r"^(.+?\b(?:also\s+)?(?:helps?|supports?|allows?|enables?|involves?)\s+.+?)\s+and\s+(carry|carries|carrying)\s+(?:that|this|the)\s+([A-Za-z][A-Za-z'’-]+)\s+(.+)$",
        flags=re.I,
    )
    match = pattern.match(sentence.strip())
    if not match or len(sentence.split()) < 14:
        return sentence
    first, verb, subject, rest = match.groups()
    carry = "carries" if verb.casefold() == "carries" else "can carry"
    return f"{first.strip()}. {subject[:1].upper() + subject[1:]} {carry} {rest.strip()}"


def _split_thereby_consequence(text: str) -> str:
    return _map_sentences(text, _split_thereby_consequence_sentence)


def _split_thereby_consequence_sentence(sentence: str) -> str:
    match = re.match(r"^(.+?),\s*thereby\s+([A-Za-z]+ing)\s+(.+)$", sentence.strip(), flags=re.I)
    if not match:
        return sentence
    first, gerund, rest = match.groups()
    subject = _sentence_subject(first)
    if not subject:
        return sentence
    return f"{first.strip()}. {subject} {_simple_past_from_gerund(gerund)} {rest.strip()}"


def _sentence_subject(sentence: str) -> str:
    subject_before_verb = re.match(
        r"^((?:The|A|An|My|Our|His|Her|Their)\s+.+?)\s+(?:also\s+)?(?:found|finds|learned|learns|created|creates|helped|helps|allowed|allows|made|makes|gave|gives|showed|shows|demonstrated|demonstrates|became|becomes)\b",
        sentence.strip(),
        flags=re.I,
    )
    if subject_before_verb:
        return subject_before_verb.group(1)
    match = re.match(r"^((?:The|A|An|My|Our|His|Her|Their)\s+[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,2}|[A-Z][A-Za-z'’-]{2,})\b", sentence.strip())
    return match.group(1) if match else ""


def _simple_past_from_gerund(word: str) -> str:
    lowered = word.casefold()
    if lowered.endswith("ying"):
        return word[:-4] + "ied"
    if lowered.endswith("ing"):
        stem = word[:-3]
        return stem + ("d" if stem.endswith("e") else "ed")
    return word


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


def robotic_sentence_chain(text: str) -> bool:
    sentences = _sentences(text)
    if len(sentences) < 5:
        return False
    starts: dict[str, int] = {}
    frames: dict[str, int] = {}
    for sentence in sentences:
        words = [part.strip(".,:;!?\"'“”’").casefold() for part in sentence.split() if part.strip(".,:;!?\"'“”’")]
        if not words:
            continue
        starts[words[0]] = starts.get(words[0], 0) + 1
        if len(words) >= 3:
            frame = " ".join(words[:3])
            frames[frame] = frames.get(frame, 0) + 1
    return max(starts.values(), default=0) >= 4 or max(frames.values(), default=0) >= 3


def catalogue_sentence_chain(text: str) -> bool:
    sentences = _sentences(text)
    if len(sentences) < 5:
        return False
    catalogue_run = 0
    for sentence in sentences:
        if _catalogue_sentence(sentence):
            catalogue_run += 1
            if catalogue_run >= 3:
                return True
        else:
            catalogue_run = 0
    return False


def _catalogue_sentence(sentence: str) -> bool:
    value = sentence.strip()
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", value)
    if not (4 <= len(words) <= 10):
        return False
    lowered = value.casefold()
    if re.match(r"^(?:knowledge|access|the\s+real\s+challenge|the\s+harder\s+task)\b", lowered):
        return True
    if re.match(r"^(?:online\s+courses|ai\s+tools|search\s+engines|social\s+media|peer\s+communities|youtube|tiktok)\b", lowered):
        return True
    return bool(
        re.match(r"^[A-Z][A-Za-z'’-]+(?:\s+(?:and|or)\s+[A-Z]?[A-Za-z'’-]+)?\s+(?:add|adds|aid|aids|broaden|broadens|complete|completes|enrich|enriches|expand|expands|further|help|helps|locate|offer|offers|provide|provides|support|supports)\b", value)
    )


def _sentences(text: str) -> list[str]:
    protected = re.sub(r"\bet\s+al\.", "et al<dot>", str(text or ""), flags=re.I)
    parts = [part.replace("et al<dot>", "et al.").strip() for part in re.split(r"(?<=[.!?])\s+", protected)]
    return [part for part in parts if part]


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
    if re.match(r"^(?:but\s+)?not\s+always\b", lowered):
        return True
    if re.match(r"^(?:and|but|or)\s+(?:solve|connect|analyse|analyze|adapt|communicate|create)\b", lowered):
        return True
    if re.match(r"^(?:solve|connect|analyse|analyze|adapt|communicate|create)\b", lowered) and len(words) <= 8:
        return True
    if re.match(r"^(?:when|while|whilst|although|though|because|if|since)\b", lowered) and "," not in value and not _has_late_main_clause(lowered):
        return True
    if re.match(r"^as\b", lowered) and "," not in value and not _has_late_main_clause(lowered):
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
    if re.match(r"^(?:but\s+)?not\s+always\b", lowered):
        return True
    if re.match(r"^(?:and|but|or)\s+(?:solve|connect|analyse|analyze|adapt|communicate|create)\b", lowered):
        return True
    if re.match(r"^(?:solve|connect|analyse|analyze|adapt|communicate|create)\b", lowered) and len(re.findall(r"[A-Za-z][A-Za-z'’-]*", value)) <= 8:
        return True
    return bool(re.match(r"^(?:as|since)\b", lowered) and "," not in value and not _has_late_main_clause(lowered))


def _repair_trace_like(sentence: str) -> bool:
    lowered = sentence.strip(" .!?").casefold()
    return bool(re.search(r"\b(?:is the context|carries the same point|keeps both sides visible|same point)\b", lowered))


def _has_finite_verb(text: str) -> bool:
    return bool(re.search(r"\b(?:am|are|is|was|were|be|being|been|have|has|had|do|does|did|must|should|would|could|can|will|may|might|[a-z]+ed|[a-z]+s)\b", text))


def _has_late_main_clause(text: str) -> bool:
    words = re.findall(r"[a-z][a-z'’-]*", text.casefold())
    if len(words) < 8:
        return False
    subject_pattern = r"\b(?:i|we|you|he|she|they|it|students?|learners?|teachers?|educators?|clients?|teams?|groups?|class|course|activity|experience|process|method|support|guidance|conversation)\s+(?:also\s+)?"
    verb_pattern = r"(?:am|are|is|was|were|have|has|had|do|does|did|must|should|would|could|can|will|may|might|got|became|came|made|took|gave|received|led|saw|knew|felt|[a-z]+ed|[a-z]+s)\b"
    matches = list(re.finditer(subject_pattern + verb_pattern, text))
    return len(matches) >= 2 and matches[-1].start() > max(10, len(text) // 3)
