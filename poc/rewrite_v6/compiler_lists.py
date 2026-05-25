from __future__ import annotations

import re

from .text import source_terms


def _has_packed_list(text: str) -> bool:
    visible = _without_parentheticals(text)
    lowered = visible.casefold()
    return visible.count(",") >= 2 or lowered.count(" and ") >= 2 or ";" in visible

def _unpack_sentence(text: str) -> list[str]:
    parts = _split_parts(text)
    parts = _attach_leading_context(parts)
    if len(parts) < 2:
        return []
    first = _clean_start(parts[0])
    reward_item = _reward_people_item(first)
    question_item = _question_item(first)
    subject = _leading_subject(first)
    prefix = _list_prefix(first)
    active_prefix = prefix
    active_question = question_item
    first_sentence = _reward_people_sentence(reward_item) if reward_item else first
    first_sentence = _question_sentence(question_item) if question_item else first_sentence
    sentences = [_period(_capitalize(first_sentence))]
    for part in parts[1:]:
        item = _clean_item(part)
        prefixed_item = _clean_prefixed_item(part)
        if not item:
            continue
        if reward_item:
            sentences.append(_period(_capitalize(_reward_people_sentence(item))))
        elif active_question:
            sentences.append(_period(_capitalize(_question_sentence(item))))
        elif item.casefold().startswith(("because ", "since ", "when ", "while ")):
            sentences.append(_period(_capitalize(item)))
        elif _starts_with_subject_clause(prefixed_item):
            sentences.append(_period(_capitalize(prefixed_item)))
            active_prefix = _list_prefix(prefixed_item) or active_prefix
        elif active_prefix and _is_active_verb_prefix(active_prefix):
            sentences.append(_period(_capitalize(f"{active_prefix} {prefixed_item}")))
        elif _looks_like_predicate(item) and subject:
            sentence = f"{subject} {item}"
            active_question = _question_item(sentence)
            sentences.append(_period(_capitalize(_question_sentence(active_question) if active_question else sentence)))
            active_prefix = _list_prefix(sentence) or active_prefix
        elif active_prefix:
            if _starts_with_modal_clause(prefixed_item):
                sentences.append(_period(_capitalize(prefixed_item)))
            else:
                item_sentence = _context_list_item_sentence(active_prefix, prefixed_item)
                sentences.append(_period(_capitalize(item_sentence)))
        elif item.casefold().startswith("not "):
            sentences.append(_period(_capitalize(f"The contrast is {item}")))
        elif _starts_with_modal_clause(item):
            sentences.append(_period(_capitalize(item)))
        else:
            sentences.append(_period(_capitalize(_article_phrase(item))))
    return sentences

def _attach_leading_context(parts: list[str]) -> list[str]:
    if len(parts) < 2:
        return parts
    first = parts[0].strip()
    if not _is_leading_context_fragment(first):
        return parts
    return [f"{first}, {parts[1].strip()}", *parts[2:]]

def _context_list_item_sentence(prefix: str, item: str) -> str:
    clean_prefix = _strip_leading_context(prefix)
    built_around = re.match(r"^(.+?)\s+(?:is|are|was|were)\s+(?:mostly\s+)?built\s+around$", clean_prefix, flags=re.I)
    if built_around:
        return f"That structure also included {item}"
    return f"{_prefix_for_item(clean_prefix, item)} {item}"

def _strip_leading_context(text: str) -> str:
    return re.sub(r"^(in|during|after|before|through|at|from|to|for|as)\s+[^,]{1,80},\s+", "", text.strip(), flags=re.I)

def _is_leading_context_fragment(text: str) -> bool:
    words = text.strip().split()
    if len(words) > 6:
        return False
    if not re.match(r"^(in|during|after|before|through|at|from|to|for|as)\b", text.strip(), flags=re.I):
        return False
    return not _has_finite_verb(text)

def _has_finite_verb(text: str) -> bool:
    return bool(
        re.search(r"\b(am|is|are|was|were|be|been|has|have|had|do|does|did|can|could|may|might|must|shall|should|will|would)\b", text, flags=re.I)
        or re.search(r"\b[A-Za-z]+(?:ed|es)\b", text)
    )

def _split_parts(text: str) -> list[str]:
    protected = _protect_parenthetical_commas(text)
    protected = _protect_quoted_commas(protected)
    protected = re.sub(r"\b([A-Z][A-Za-z'-]+)\s+and\s+([A-Z][A-Za-z'-]+)\s+(\()", r"\1 & \2 \3", protected)
    normalized = protected.replace(", and ", ", ").replace(", or ", ", ").replace(" and ", ", ").replace(";", ",")
    return [
        _restore_quoted_commas(_restore_parenthetical_commas(part.strip().replace(" & ", " and ")))
        for part in normalized.split(",")
        if part.strip()
    ]

def _split_existing_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()]
    return [_period(_capitalize(part)) for part in parts] if len(parts) > 1 else []

def _without_parentheticals(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text)

def _protect_parenthetical_commas(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return match.group(0).replace(",", "<comma>").replace(";", "<semi>").replace(" and ", " <and> ")

    return re.sub(r"\([^)]*\)", replace, text)

def _restore_parenthetical_commas(text: str) -> str:
    return text.replace("<comma>", ",").replace("<semi>", ";").replace(" <and> ", " and ")

def _protect_quoted_commas(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return match.group(0).replace(",", "<qcomma>").replace(" and ", " <qand> ")

    return re.sub(r"“[^”]*”|\"[^\"]*\"", replace, text)

def _restore_quoted_commas(text: str) -> str:
    return text.replace("<qcomma>", ",").replace(" <qand> ", " and ")

def _leading_subject(text: str) -> str:
    words = _clean_start(text).split()
    return words[0] if words else ""

def _list_prefix(text: str) -> str:
    words = _clean_start(text).split()
    if len(words) < 3:
        return ""
    lowered = [word.casefold() for word in words]
    connectors = {
        "is",
        "are",
        "was",
        "were",
        "on",
        "for",
        "from",
        "with",
        "through",
        "by",
        "as",
        "into",
        "about",
        "around",
        "become",
    }
    for index in range(len(words) - 2, -1, -1):
        if lowered[index] in connectors:
            return " ".join(words[: index + 1])
    modal_prefix = _modal_list_prefix(words)
    if modal_prefix:
        return modal_prefix
    if "can" in lowered and "help" in lowered:
        index = lowered.index("help")
        if index + 1 < len(words):
            return " ".join(words[: index + 2])
    if "to" in lowered:
        index = len(lowered) - lowered[::-1].index("to") - 1
        return " ".join(words[: min(len(words), index + 2)])
    if "can" in lowered:
        return " ".join(words[: lowered.index("can") + 1])
    if words[-3].casefold() in {"can", "to"}:
        return " ".join(words[:-1])
    active_prefix = _active_verb_prefix(words)
    if active_prefix:
        return active_prefix
    return ""

def _active_verb_prefix(words: list[str]) -> str:
    if len(words) < 3:
        return ""
    start = 2 if words[0].casefold() in {"the", "a", "an", "this", "that", "these", "those", "my", "our"} else 1
    for index in range(start, len(words) - 1):
        verb = words[index].casefold().strip(".,:;!?")
        if verb in {"is", "are", "was", "were", "has", "have", "had", "does"}:
            continue
        if re.match(r"^[A-Za-z][A-Za-z'-]*(?:s|es)$", verb):
            return " ".join(words[: index + 1])
    return ""

def _is_active_verb_prefix(text: str) -> bool:
    words = text.split()
    return bool(words and _active_verb_prefix([*words, "item"]) == text)

def _modal_list_prefix(words: list[str]) -> str:
    lowered = [word.casefold() for word in words]
    modals = {"can", "could", "may", "might", "must", "should", "would", "will"}
    skips = {"also", "not", "only", "still"}
    for index, word in enumerate(lowered[:-1]):
        if word not in modals:
            continue
        end = index + 1
        while end < len(words) - 1 and lowered[end] in skips:
            end += 1
        if end < len(words) - 1:
            return " ".join(words[: end + 1])
    return ""

def _prefix_for_item(prefix: str, item: str) -> str:
    words = prefix.split()
    if len(words) < 3:
        return prefix
    lowered = [word.casefold() for word in words]
    modals = {"can", "could", "may", "might", "must", "should", "would", "will"}
    helpers = {"help"}
    if words[-1].casefold() == "support" and len(item.split()) > 1:
        for index, word in enumerate(lowered):
            if word in modals:
                return " ".join(words[: index + 1])
    if not _looks_like_predicate(item):
        return prefix
    for index, word in enumerate(lowered):
        if word in modals and words[-1].casefold() not in helpers:
            return " ".join(words[: index + 1])
    return prefix

def _looks_like_predicate(text: str) -> bool:
    words = text.split()
    first = words[0].casefold() if words else ""
    return (
        first.endswith(("ed", "ing"))
        or (len(words) > 1 and first.endswith("ly"))
        or (len(words) > 1 and first.endswith("s"))
        or first in {"can", "could", "may", "might", "must", "should", "will", "would"}
    )

def _reward_people_item(text: str) -> str:
    match = re.match(r"^(?:it|this|that)\s+rewards\s+people\s+who\s+can\s+(.+)$", text, flags=re.I)
    return _clean_item(match.group(1)) if match else ""

def _reward_people_sentence(item: str) -> str:
    return f"People who can {_clean_item(item)} are rewarded"

def _question_item(text: str) -> str:
    match = re.match(r"^(?:it|this|that)\s+raises\s+questions\s+about\s+(.+)$", text, flags=re.I)
    return _clean_item(match.group(1)) if match else ""

def _question_sentence(item: str) -> str:
    return f"{_article_phrase(item)} becomes a question"

def _replace_initial_pronoun(text: str, previous: str) -> str:
    weak_terms = {"this", "that", "it", "they", "many", "some", "another", "other", "others", "the"}
    first_terms = [term for term in source_terms(previous, limit=4) if term.casefold() not in weak_terms]
    replacement = _article_phrase(first_terms[0]) if first_terms and not first_terms[0].casefold().endswith("ly") else "The point"
    return re.sub(r"^(it|this|that|they)\s+", replacement + " ", text, flags=re.I)

def _clean_start(text: str) -> str:
    return text.strip(" .")

def _clean_item(text: str) -> str:
    return re.sub(r"^(also|the|a|an)\s+", "", text.strip(" ."), flags=re.I)

def _clean_prefixed_item(text: str) -> str:
    return re.sub(r"^(also)\s+", "", text.strip(" ."), flags=re.I)

def _starts_with_modal_clause(text: str) -> bool:
    return bool(
        re.match(r"^(it|this|that|these|those|they|he|she|we)\s+\w+\s+(can|could|may|might|must|should|would|will)\s+", text, flags=re.I)
        or re.match(r"^(it|this|that|these|those|they|he|she|we)\s+(can|could|may|might|must|should|would|will)\s+", text, flags=re.I)
    )

def _starts_with_subject_clause(text: str) -> bool:
    return bool(
        re.match(r"^((it|this|that)'s|(i|we|you|he|she|it|this|that|they|students|learners|participants|clients|customers|users)\s+\w+(?:'\w+)?)\b", text, flags=re.I)
    )

def _article_phrase(text: str) -> str:
    phrase = _clean_item(text)
    if not phrase:
        return phrase
    first = phrase.split()[0].casefold()
    if first in {"the", "a", "an"}:
        return _capitalize(phrase)
    if first in {"this", "that", "it"}:
        return "This point"
    if first in {"they", "these"}:
        return "These points"
    if phrase[:1].isupper():
        return phrase
    return "The " + phrase

def _capitalize(text: str) -> str:
    stripped = text.strip()
    return stripped[:1].upper() + stripped[1:] if stripped else stripped

def _period(text: str) -> str:
    stripped = re.sub(r"\.\s+([\"'”’])$", r"\1", text.strip(" ,;:"))
    stripped = re.sub(r"\s+([\"'”’])$", r"\1", stripped)
    return stripped if stripped.rstrip("\"'”’").endswith((".", "!", "?")) else stripped + "."

def _dedupe(sentences: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        key = sentence.casefold()
        if sentence and key not in seen:
            out.append(sentence)
            seen.add(key)
    return out
