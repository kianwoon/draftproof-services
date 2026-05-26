from __future__ import annotations

import re
from typing import Any


def compile_sentence_rows(row: dict[str, Any]) -> str:
    """Compile writer sentence rows into paragraph text.

    The writer may still return a paragraph-level ``text`` field, but V6 should
    not let that field recombine carefully planned sentence rows into a smooth
    paragraph route. Rows are the execution unit; paragraph text is compiled.
    """
    rows = _row_list(row.get("sentence_rows")) or _row_list(row.get("coverage_map"))
    sentences: list[str] = []
    seen: set[str] = set()
    for item in rows:
        for sentence in _row_sentences(item):
            sentence = _repair_sentence(sentence, sentences[-1] if sentences else "")
            key = _sentence_key(sentence)
            if key in seen or _near_duplicate_sentence(sentence, sentences):
                continue
            sentences.append(sentence)
            seen.add(key)
    return " ".join(sentences).strip()


def compile_or_fallback_text(row: dict[str, Any]) -> str:
    compiled = compile_sentence_rows(row)
    return compiled or str(row.get("text") or "").strip()


def _row_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clean_sentence(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    text = _concretize_demonstrative_start(text)
    text = _plainify_smoothing(text)
    text = text[:1].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _concretize_demonstrative_start(text: str) -> str:
    text = re.sub(r"^\s*for\s+this\s+reason,?\s+", "", str(text or ""), flags=re.I)
    match = re.match(r"\s*(this|that|these|those)\s+([A-Za-z][A-Za-z'’-]{3,})(\b.*)$", str(text or ""), flags=re.I)
    if not match:
        return text
    if match.group(2).casefold() in {"applies", "allows", "shows", "means", "creates", "helps", "makes"}:
        return text
    return f"the {match.group(2)}{match.group(3)}"


def _plainify_smoothing(text: str) -> str:
    value = re.sub(r"\b(?:truly|strictly|inadvertently|extremely|highly)\s+", "", str(text or ""), flags=re.I)
    value = re.sub(r"\btime\s+for\s+metacognitive\s+monitoring\b", "time to monitor their thinking", value, flags=re.I)
    value = re.sub(r"\bmetacognitive\s+awareness\b", "awareness of how they learn", value, flags=re.I)
    return re.sub(r"\bmetacognitive\s+development\b", "learning awareness", value, flags=re.I)


def _row_sentences(item: dict[str, Any]) -> list[str]:
    sentence = _clean_sentence(item.get("sentence"))
    if not sentence:
        return []
    beat_count = len(item.get("coverage_beat_ids") or [])
    force = beat_count > 1 or bool(item.get("finding_contract_id"))
    if "regardless of" in sentence.casefold() and sentence.count(",") >= 2:
        return [_clean_sentence(part) for part in _split_regardless_list(sentence)]
    if sentence.count(",") >= 2 and not force:
        return [_clean_sentence(part) for part in _split_comma_scaffold(sentence)]
    for splitter in (_split_obligation_provide, _split_although_main_clause, _split_in_context_comma, _split_yet_result, _split_when_start_creates, _split_on_other_hand):
        parts = splitter(sentence)
        if len(parts) >= 2:
            return [_clean_sentence(part) for part in parts]
    if not force and _word_count(sentence) <= 24:
        return [sentence]
    return [_clean_sentence(part) for part in _split_overloaded_sentence(sentence, force=force)]


def _split_overloaded_sentence(sentence: str, *, force: bool = False) -> list[str]:
    text = str(sentence or "").strip().rstrip(".!?")
    if not force and _word_count(text) <= 24:
        return [_clean_sentence(text)]
    parts = _split_framework_working_row(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_evaluates_between_row(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_distinction_between(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_obligation_provide(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_under_obligation(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_according_to_start(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_must_not_undermine(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_such_as_strain(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_subject_remained_pair(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_demonstrated_ability_needs(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_context_delivery(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_and_instead_focus(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_although_comma_scaffold(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_although_main_clause(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_gerund_and_gerund_helps(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_by_replacing_then_subject(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_monitoring_regulation(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_do_not_lie_in(text)
    if len(parts) == 1 and parts[0] != text:
        return [_clean_sentence(parts[0])]
    parts = _split_more_valuable_than(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_equipping_enable_when(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_when_start_creates(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_not_by_expectation_when(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_in_setting_start(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_on_other_hand(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_yet_result(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_in_order_to(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_first_person_working_to(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_allow_gradual_build(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_once(text, r"\s+\bbecause\b\s+")
    if len(parts) == 2:
        left, right = parts
        return [_clean_sentence(left), *_split_overloaded_sentence(_ensure_subject(right, left), force=force)]
    parts = _split_once(text, r"\s+\bwhile\b\s+")
    if len(parts) == 2:
        left, right = parts
        return [_clean_sentence(left), _clean_sentence(_ensure_subject(right, left))]
    parts = _split_once(text, r"\s+\brather\s+than\b\s+")
    if len(parts) == 2:
        left, right = parts
        return [_clean_sentence(left), _clean_sentence(f"The contrast is not {right}")]
    parts = _split_once(text, r"\s+\bwhen\b\s+")
    if len(parts) == 2:
        left, right = parts
        return [_clean_sentence(left), _clean_sentence(_ensure_subject(right, left))]
    parts = _split_support_clause(text)
    if len(parts) == 2:
        return [_clean_sentence(parts[0]), _clean_sentence(parts[1])]
    parts = _split_regardless_list(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    parts = _split_comma_scaffold(text)
    if len(parts) >= 2:
        return [_clean_sentence(part) for part in parts]
    return [_clean_sentence(text)]


def _split_once(text: str, pattern: str) -> list[str]:
    parts = re.split(pattern, text, maxsplit=1, flags=re.I)
    return [part.strip(" ,;:") for part in parts if part.strip(" ,;:")]


def _split_first_person_working_to(text: str) -> list[str]:
    match = re.match(r"\s*((?:I|We)\s+(?:am|are)\s+[^.]{4,120}?)\s+working\s+to\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    subject = match.group(1).split()[0]
    auxiliary = "am" if subject.casefold() == "i" else "are"
    return [match.group(1), f"{subject} {auxiliary} working to {match.group(2)}"]


def _split_framework_working_row(text: str) -> list[str]:
    match = re.match(
        r"\s*((?:I|We)\s+(?:am|are)\s+[^.]{4,120}?)\s+working\s+to\s+design\s+and\s+implement\s+(.+?)\s+for\s+(.+?)\s+that\s+(.+)$",
        str(text or ""),
        flags=re.I,
    )
    if not match:
        return [text]
    role = match.group(1).strip(" ,;:")
    object_text = match.group(2).strip(" ,;:")
    context = match.group(3).strip(" ,;:")
    clause = match.group(4).strip(" ,;:")
    return [role, f"{context} uses {object_text}", f"{context} {clause}"]


def _split_evaluates_between_row(text: str) -> list[str]:
    match = re.match(
        r"\s*(.+?)\s+(?:critically\s+)?evaluates\s+(?:the\s+tension\s+)?between\s+(.+?)\s+and\s+(.+?)\s+required\s+by\s+(.+)$",
        str(text or ""),
        flags=re.I,
    )
    if match:
        subject = match.group(1).strip(" ,;:")
        first = _plain_compare_side(match.group(2))
        second = _plain_compare_side(match.group(3))
        requirement = match.group(4).strip(" ,;:")
        return [f"{subject} compares {first} with {second}", f"The second side is required by {requirement}"]
    match = re.match(
        r"\s*(.+?)\s+(?:critically\s+)?evaluates\s+(?:the\s+tension\s+)?between\s+(.+?)\s+and\s+(.+)$",
        str(text or ""),
        flags=re.I,
    )
    if not match:
        return [text]
    return [
        f"{match.group(1).strip(' ,;:')} compares {_plain_compare_side(match.group(2))} with {_plain_compare_side(match.group(3))}"
    ]


def _split_according_to_start(text: str) -> list[str]:
    match = re.match(r"\s*according\s+to\s+([^,]{3,80}),\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    source = match.group(1).strip(" ,;:")
    claim = match.group(2).strip(" ,;:")
    strain = _such_as_strain_parts(claim)
    if strain:
        category, example, pressure, object_text = strain
        return [f"{example} is part of {category}", f"{source} links that task to {pressure} on {object_text}"]
    return [f"{source} is the source frame", claim]


def _split_distinction_between(text: str) -> list[str]:
    match = re.match(r"\s*(.+?)\s+reveals\s+(?:a\s+)?(?:critical\s+)?distinction(?:\s+.+?)?\s+between\s+(.+?)\s+and\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    subject = match.group(1).strip(" ,;:")
    first = _plain_compare_side(match.group(2))
    second = _plain_compare_side(match.group(3))
    return [f"{subject} separates {first} from {second}", f"The distinction keeps both sides visible"]


def _split_obligation_provide(text: str) -> list[str]:
    match = re.match(r"\s*(.+?)\s+requires\s+(.+?)\s+to\s+have\s+a\s+legal\s+obligation\s+to\s+provide\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    source = match.group(1).strip(" ,;:")
    actor = match.group(2).strip(" ,;:")
    object_text = match.group(3).strip(" ,;:")
    return [f"{source} sets a legal obligation for {actor}", f"{actor[:1].upper()}{actor[1:]} must provide {object_text}"]


def _split_under_obligation(text: str) -> list[str]:
    match = re.match(r"\s*under\s+([^,]{3,100}),\s+(.+?)\s+have\s+a\s+legal\s+obligation\s+to\s+provide\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    source = match.group(1).strip(" ,;:")
    actor = match.group(2).strip(" ,;:")
    object_text = match.group(3).strip(" ,;:")
    return [f"{source} applies to {actor}", f"{actor[:1].upper()}{actor[1:]} must provide {object_text}"]


def _split_must_not_undermine(text: str) -> list[str]:
    match = re.match(r"\s*(.+?)\s+must\s+not\s+undermine\s+(.+?)\s+or\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    subject = match.group(1).strip(" ,;:")
    first = match.group(2).strip(" ,;:")
    second = match.group(3).strip(" ,;:")
    return [f"{subject} must not undermine {first}", f"The same limit also covers {second}"]


def _split_such_as_strain(text: str) -> list[str]:
    parts = _such_as_strain_parts(text)
    if not parts:
        return [text]
    category, example, pressure, object_text = parts
    return [f"{example} is part of {category}", f"The task places {pressure} on {object_text}"]


def _split_demonstrated_ability_needs(text: str) -> list[str]:
    match = re.match(r"\s*(.+?)\s+demonstrated\s+the\s+ability\s+to\s+(.+?)\s+and\s+(.+?)\s+the\s+needs\s+of\s+(.+?)\s+who\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    actor = match.group(1).strip(" ,;:")
    first = match.group(2).strip(" ,;:")
    second = match.group(3).strip(" ,;:")
    group = match.group(4).strip(" ,;:")
    limit = match.group(5).strip(" ,;:")
    return [f"{actor} {_past_action(first, 'identify')} the needs of {group}", f"{actor} {_past_action(second, 'address')} those needs", f"{group[:1].upper()}{group[1:]} {limit}"]


def _split_subject_remained_pair(text: str) -> list[str]:
    match = re.match(r"\s*(The\s+.+?),\s+(.+?\s+have\s+(?:still\s+)?remained\s+.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    first = match.group(1).strip(" ,;:")
    second = match.group(2).strip(" ,;:")
    predicate = re.sub(r"^.+?\s+have\s+", "have ", second, count=1, flags=re.I)
    return [f"{first} {predicate}", second]


def _past_action(text: str, verb: str) -> str:
    past = f"{verb[:-1]}ied" if verb.endswith("y") else (f"{verb}d" if verb.endswith("e") else f"{verb}ed")
    return re.sub(rf"\b{verb}\b", past, str(text or ""), count=1, flags=re.I)


def _such_as_strain_parts(text: str) -> tuple[str, str, str, str] | None:
    match = re.match(r"\s*(.+?)\s+such\s+as\s+(.+?)\s+place[s]?\s+(.+?)\s+on\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return None
    category = match.group(1).strip(" ,;:")
    example = match.group(2).strip(" ,;:")
    pressure = match.group(3).strip(" ,;:")
    object_text = match.group(4).strip(" ,;:")
    return category, example, pressure, object_text


def _split_context_delivery(text: str) -> list[str]:
    match = re.match(r"\s*in\s+(.+?)\s+([A-Za-z][^.]{12,180}?)\s+in\s+my\s+delivery\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    context = match.group(1).strip(" ,;:")
    middle = match.group(2).strip(" ,;:")
    delivery = match.group(3).strip(" ,;:")
    return [f"{context} is the context", f"{middle[:1].upper()}{middle[1:]} in my delivery {delivery}"]


def _split_and_instead_focus(text: str) -> list[str]:
    match = re.match(r"\s*(.+?)\s+must\s+move\s+beyond\s+(.+?)\s+and\s+instead\s+focus\s+on\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    subject = match.group(1).strip(" ,;:")
    rejected = match.group(2).strip(" ,;:")
    focus = match.group(3).strip(" ,;:")
    return [f"{subject} must move beyond {rejected}", f"{_compact_subject(subject)} should focus on {focus}"]


def _split_although_main_clause(text: str) -> list[str]:
    inner = re.sub(r"^\s*although\s+", "", str(text or ""), flags=re.I)
    replaced = _split_by_replacing_then_subject(inner)
    if len(replaced) == 2:
        pair = _split_subject_remained_pair(replaced[1])
        return [replaced[0], *pair] if len(pair) == 2 else replaced
    match = re.match(r"\s*although\s+(.+?)\s+(the\s+[^.]+?\s+remain[s]?\s+.+)$", str(text or ""), flags=re.I)
    if match:
        pair = _split_subject_remained_pair(match.group(2).strip(" ,;:"))
        return [match.group(1).strip(" ,;:"), *pair] if len(pair) == 2 else [match.group(1).strip(" ,;:"), match.group(2).strip(" ,;:")]
    match = re.match(r"\s*although\s+(.+?)\s+(the\s+.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    pair = _split_subject_remained_pair(match.group(2).strip(" ,;:"))
    return [match.group(1).strip(" ,;:"), *pair] if len(pair) == 2 else [match.group(1).strip(" ,;:"), match.group(2).strip(" ,;:")]


def _split_although_comma_scaffold(text: str) -> list[str]:
    value = str(text or "")
    return _split_comma_scaffold(value) if value.casefold().startswith("although ") and value.count(",") >= 2 else [text]


def _split_gerund_and_gerund_helps(text: str) -> list[str]:
    match = re.match(r"\s*(\w+ing\s+.+?)\s+and\s+(\w+ing\s+.+?)\s+(?:helps?|(?:explicitly\s+)?encourages?)\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    first = match.group(1).strip(" ,;:")
    second = match.group(2).strip(" ,;:")
    effect = match.group(3).strip(" ,;:")
    split_effect = _split_monitoring_regulation(f"{second} helps {effect}")
    if len(split_effect) == 2:
        return [f"{first} supports the teaching route", split_effect[0], split_effect[1]]
    return [f"{first} supports the teaching route", f"{second} helps {effect}"]


def _split_by_replacing_then_subject(text: str) -> list[str]:
    match = re.match(r"\s*(.+?\s+by\s+replacing\s+.+?\s+with\s+.+?)\s+(the\s+.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    right = match.group(2).strip(" ,;:")
    return [match.group(1).strip(" ,;:"), *_split_comma_scaffold(right)] if right.count(",") >= 2 else [match.group(1).strip(" ,;:"), right]


def _split_monitoring_regulation(text: str) -> list[str]:
    match = re.match(r"\s*(.+?)\s+helps?\s+(.+?)\s+engage\s+in\s+(.+?)\s+and\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    subject = match.group(1).strip(" ,;:")
    actor = _clean_actor(match.group(2))
    first = match.group(3).strip(" ,;:")
    second = match.group(4).strip(" ,;:")
    return [f"{subject} gives {actor} time for {first}", f"The same support also helps {actor} with {second}"]


def _clean_actor(text: str) -> str:
    value = re.sub(r"^(?:explicitly\s+)?encourage\s+", "", str(text or "").strip(" ,;:"), flags=re.I)
    return re.sub(r"\s+to$", "", value, flags=re.I).strip(" ,;:") or "learners"


def _split_more_valuable_than(text: str) -> list[str]:
    match = re.match(r"\s*for\s+(.+?),\s+(.+?)\s+(?:matters\s+more|is\s+more\s+valuable)\s+than\s+(.+)$", str(text or ""), flags=re.I)
    if match:
        context = match.group(1).strip(" ,;:")
        first = match.group(2).strip(" ,;:")
        second = match.group(3).strip(" ,;:")
        return [f"{first[:1].upper()}{first[1:]} matters for {context}", f"{second[:1].upper()}{second[1:]} is not enough by itself"]
    match = re.match(r"\s*(.+?)\s+is\s+more\s+valuable\s+than\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    first = match.group(1).strip(" ,;:")
    second = match.group(2).strip(" ,;:")
    return [f"{first} matters more than {second}", f"{second[:1].upper()}{second[1:]} is not enough by itself"]


def _split_do_not_lie_in(text: str) -> list[str]:
    match = re.match(r"\s*(?:The\s+demonstration\s+(?:shows\s+)?that\s+)?(.+?)\s+(do|does)\s+not\s+lie\s+in\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    subject = match.group(1).strip(" ,;:")
    right = match.group(3).strip(" ,;:")
    verb = "is" if match.group(2).casefold() == "does" else "are"
    based = re.match(r"(.+?)\s+based\s+on\s+(.+)$", right, flags=re.I)
    return [f"{subject[:1].upper()}{subject[1:]} {verb} not {based.group(1).strip(' ,;:')} or {based.group(2).strip(' ,;:')}" if based else f"{subject[:1].upper()}{subject[1:]} {verb} not based on {right}"]


def _split_equipping_enable_when(text: str) -> list[str]:
    match = re.match(r"\s*equipping\s+(.+?)\s+with\s+(.+?)\s+will\s+enable\s+(.+?)\s+to\s+(.+?)\s+when\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    actor = match.group(1).strip(" ,;:")
    skill = match.group(2).strip(" ,;:")
    action = match.group(4).strip(" ,;:")
    condition = match.group(5).strip(" ,;:")
    subject = _actor_subject(actor)
    return [f"{subject} need {skill}", f"The { _head_noun(skill) } help {actor} {action}", f"The same support matters when {condition}"]


def _actor_subject(text: str) -> str:
    value = str(text or "").strip()
    if value.casefold() == "them":
        return "They"
    return value[:1].upper() + value[1:]


def _split_when_start_creates(text: str) -> list[str]:
    match = re.match(r"\s*when\s+(.+?)\s+it\s+creates\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    condition = match.group(1).strip(" ,;:")
    result = match.group(2).strip(" ,;:")
    condition_parts = _split_in_order_to(condition)
    if len(condition_parts) == 2:
        return [condition_parts[0], condition_parts[1], f"The result creates {result}"]
    return [f"{condition[:1].upper()}{condition[1:]}", f"The result creates {result}"]


def _split_not_by_expectation_when(text: str) -> list[str]:
    match = re.match(r"\s*(.+?)\s+will\s+not\s+be\s+able\s+to\s+do\s+so\s+by\s+expecting\s+(.+?)\s+when\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    actor = match.group(1).strip(" ,;:")
    expectation = match.group(2).strip(" ,;:")
    condition = match.group(3).strip(" ,;:")
    subject = "The same people" if actor.casefold() == "they" else actor
    return [f"{subject} cannot rely on {expectation}", f"The same limit applies when {condition}"]


def _split_in_setting_start(text: str) -> list[str]:
    match = re.match(r"\s*in\s+(.+?)\s+(a|an|the)\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    setting = match.group(1).strip(" ,;:")
    rest = f"{match.group(2)} {match.group(3)}".strip(" ,;:")
    because = _split_once(rest, r"\s+\bbecause\b\s+")
    if len(because) == 2:
        return [f"{setting[:1].upper()}{setting[1:]} is the setting", because[0].strip(" ,;:")]
    return [f"{setting[:1].upper()}{setting[1:]} is the setting", f"{rest[:1].upper()}{rest[1:]}"]


def _split_in_context_comma(text: str) -> list[str]:
    match = re.match(r"\s*in\s+([^,]{4,100}),\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    context = match.group(1).strip(" ,;:")
    claim = match.group(2).strip(" ,;:")
    return [f"{context[:1].upper()}{context[1:]} is the context", f"{claim[:1].upper()}{claim[1:]}"]


def _split_on_other_hand(text: str) -> list[str]:
    parts = re.split(r"\s+on\s+the\s+other\s+hand,?\s+", str(text or ""), maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return [text]
    left = parts[0].strip(" ,;:")
    right = parts[1].strip(" ,;:")
    return [left, f"The other side is {right}"]


def _split_yet_result(text: str) -> list[str]:
    if re.search(r"\bnot\s+yet\s+competent\b", str(text or ""), flags=re.I):
        return [text]
    parts = re.split(r"\s+yet\s+", str(text or ""), maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return [text]
    return [parts[0].strip(" ,;:"), f"The result can {parts[1].strip(' ,;:')}"]


def _split_in_order_to(text: str) -> list[str]:
    parts = re.split(r"\s+in\s+order\s+to\s+", str(text or ""), maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return [text]
    return [parts[0].strip(" ,;:"), f"The purpose is to {parts[1].strip(' ,;:')}"]


def _split_allow_gradual_build(text: str) -> list[str]:
    match = re.match(
        r"\s*(.+?)\s+(should\s+)?allow\s+(.+?)\s+to\s+(.+?)\s+and\s+gradually\s+(.+)$",
        str(text or ""),
        flags=re.I,
    )
    if not match:
        return [text]
    subject = match.group(1).strip(" ,;:")
    modal = (match.group(2) or "").strip()
    actor = match.group(3).strip(" ,;:")
    first = match.group(4).strip(" ,;:")
    second = match.group(5).strip(" ,;:")
    allow_phrase = f"{modal} allow" if modal else "allow"
    return [f"{subject} {allow_phrase} {actor} to {first}", f"The same { _head_noun(subject) } help {actor} gradually {second}"]


def _head_noun(text: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", str(text or ""))
    if not words:
        return "support"
    if words[0].casefold() in {"the", "a", "an", "this", "that", "these", "those"} and len(words) >= 2:
        return words[1]
    return words[0]


def _compact_subject(text: str) -> str:
    subject = _first_subject(text)
    return subject if subject and subject != "The" else "The same point"


def _plain_compare_side(text: str) -> str:
    return re.sub(r"^\s*(?:providing|upholding|maintaining|using|applying)\s+", "", str(text or ""), flags=re.I).strip(" ,;:")


def _split_support_clause(text: str) -> list[str]:
    match = re.match(r"\s*(.{12,180}?)\s+\bto\s+support\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    return [match.group(1), f"The same support focused on {match.group(2)}"]


def _split_regardless_list(text: str) -> list[str]:
    match = re.match(r"\s*(.{8,120}?)\s+regardless\s+of\s+([^,.;]+),\s+([^,.;]+),\s+or\s+(.+)$", str(text or ""), flags=re.I)
    if not match:
        return [text]
    base = match.group(1).strip(" ,;:")
    first = match.group(2).strip(" ,;:")
    second = match.group(3).strip(" ,;:")
    third = match.group(4).strip(" ,;:")
    subject = re.split(r"\s+(?:applies|covers|includes|means|supports)\b", base, maxsplit=1, flags=re.I)[0].strip(" ,;:")
    subject = subject or _first_subject(base) or "The same point"
    if subject.casefold() in {"this", "that", "it", "these", "those"}:
        subject = "The same support"
    if first.casefold().startswith("whether they "):
        return [f"{base} regardless of {first}", f"{subject} still applies when they {second} or {third}"]
    return [f"{base} regardless of {first}", f"{subject} also covers {second} and {third}"]


def _split_comma_scaffold(text: str) -> list[str]:
    value = str(text or "").strip(" .!?")
    if value.count(",") < 2:
        return [text]
    value = re.sub(r"^(?:furthermore|however|therefore|overall),\s+", "", value, flags=re.I)
    if value.casefold().startswith("although "):
        left, right = value.split(",", 1)
        pair = _split_subject_remained_pair(right.strip(" ,;:"))
        return [re.sub(r"^although\s+", "", left, flags=re.I).strip(" ,;:"), *pair] if len(pair) == 2 else [left.strip(" ,;:"), right.strip(" ,;:")]
    match = re.match(r"(.+?),\s+([^,]+),\s+and\s+([^,]+)$", value, flags=re.I)
    if match:
        base = match.group(1).strip(" ,;:")
        first = match.group(2).strip(" ,;:")
        second = match.group(3).strip(" ,;:")
        return [base, f"{first[:1].upper()}{first[1:]} and {second} carry the same point"]
    match = re.match(r"((?:during|when|while)\s+.+?),\s+(.+)$", value, flags=re.I)
    if match:
        return [match.group(1).strip(" ,;:"), f"{match.group(2).strip(' ,;:')[:1].upper()}{match.group(2).strip(' ,;:')[1:]}"]
    left, right = value.split(",", 1)
    subject = _first_subject(left) or _head_noun(left)
    return [left.strip(" ,;:"), f"{subject} also carries {right.strip(' ,;:')}"]


def _ensure_subject(fragment: str, previous: str) -> str:
    text = str(fragment or "").strip()
    if not text:
        return text
    first = text.split()[0].casefold()
    subject = _first_subject(previous)
    if first in {"it", "this", "that"} and subject and subject.casefold() not in {"it", "this", "that"}:
        return re.sub(r"^\w+", subject, text, count=1)
    if first in {"i", "we", "the", "this", "that", "these", "those", "it", "they", "learners", "students"}:
        return text
    if subject and first.endswith("ing"):
        return f"{subject} involves {text}"
    return f"{subject} {text}" if subject else text


def _first_subject(text: str) -> str:
    match = re.match(r"\s*((?:I|We|This|That|These|Those|It|They)\b)", str(text or ""), flags=re.I)
    if match:
        return match.group(1).strip().capitalize()
    match = re.match(r"\s*The\s+([A-Za-z0-9'’-]+(?:\s+[A-Za-z0-9'’-]+){0,5})", str(text or ""), flags=re.I)
    if not match:
        return ""
    words: list[str] = []
    stop = {"is", "are", "was", "were", "has", "have", "had", "presents", "requires", "involves", "helps", "should", "must", "can", "could", "would", "will", "subsequently"}
    for word in match.group(1).split():
        if word.casefold() in stop:
            break
        words.append(word)
    return ("The " + " ".join(words)).strip() if words else "The"


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "")))


def _sentence_key(sentence: str) -> str:
    return re.sub(r"\W+", " ", sentence).strip().casefold()


def _near_duplicate_sentence(sentence: str, previous: list[str]) -> bool:
    words = _content_words(sentence)
    if len(words) < 3:
        return False
    for existing in previous[-3:]:
        other = _content_words(existing)
        overlap = len(words & other) / max(1, min(len(words), len(other))) if len(other) >= 3 else 0.0
        pair = f"{existing} {sentence}".casefold()
        if (
            "not enough by itself" in pair
            or "also covers" in pair
            or "still applies" in pair
            or "applies regardless" in pair
            or " combines " in pair
        ):
            continue
        if overlap >= 0.7:
            return True
        if overlap >= 0.45 and (_same_content_start(existing, sentence) or _starts_with_overlap_phrase(sentence, words & other)):
            return True
    return False


def _same_content_start(left: str, right: str) -> bool:
    left_words = _content_word_list(left)
    right_words = _content_word_list(right)
    return len(left_words) >= 2 and left_words[:2] == right_words[:2]


def _starts_with_overlap_phrase(sentence: str, overlap: set[str]) -> bool:
    words = _content_word_list(sentence)
    return len(words) >= 2 and words[0] in overlap and words[1] in overlap


def _content_word_list(text: str) -> list[str]:
    return [
        word.casefold().rstrip("s")
        for word in re.findall(r"[A-Za-z][A-Za-z'’-]{3,}", str(text or ""))
        if word.casefold() not in _STOP_WORDS
    ]


def _repair_sentence(sentence: str, previous: str) -> str:
    return _repair_fragment_start(_repair_context_start(sentence, previous), previous)


def _repair_fragment_start(sentence: str, previous: str) -> str:
    text = str(sentence or "").strip()
    match = re.match(r"^(Rather|Also|Thereby|Furthermore)\s+(.+)$", text, flags=re.I)
    if match:
        subject = _continuity_subject(previous)
        verb = _normalize_fragment_verb(match.group(2))
        return _clean_sentence(f"{subject} {verb}") if subject else text
    match = re.match(r"^(Displayed|Gently|Coupled)\s+(.+)$", text, flags=re.I)
    if match:
        subject = _continuity_subject(previous) or "The same work"
        verb = _normalize_fragment_verb(f"{match.group(1)} {match.group(2)}")
        return _clean_sentence(f"{subject} {verb}")
    if _word_count(text) <= 3 and previous:
        subject = _continuity_subject(previous)
        return _clean_sentence(f"{subject} includes {text}") if subject else text
    return text


def _continuity_subject(previous: str) -> str:
    text = str(previous or "")
    if re.search(r"\bresponsibility\b", text, flags=re.I):
        return "The same responsibility"
    if re.search(r"\bwork\b", text, flags=re.I):
        return "The same work"
    if re.search(r"\bdemonstration\b", text, flags=re.I):
        return "The same demonstration"
    subject = _first_subject(text)
    return "" if subject == "The" else subject


def _normalize_fragment_verb(text: str) -> str:
    value = str(text or "").strip(" ,;:")
    value = re.sub(r"^displayed\b", "displayed", value, flags=re.I)
    value = re.sub(r"^gently\s+address\b", "gently addresses", value, flags=re.I)
    value = re.sub(r"^coupled\s+provision\b", "is coupled with provision", value, flags=re.I)
    value = re.sub(r"^facilitating\b", "facilitates", value, flags=re.I)
    value = re.sub(r"^empowering\b", "empowers", value, flags=re.I)
    value = re.sub(r"^(stimulate|amplify|facilitate|attest|address|display|couple)\b", r"\1s", value, flags=re.I)
    value = re.sub(r"\band\s+amplify\b", "and amplifies", value, flags=re.I)
    return value


def _repair_context_start(sentence: str, previous: str) -> str:
    text = str(sentence or "").strip()
    if re.match(r"^They\s+need\b", text, flags=re.I) and re.search(r"\blearners\b", previous, flags=re.I):
        return re.sub(r"^They\b", "Learners", text, count=1, flags=re.I)
    match = re.match(r"^It is an?\s+(.+?)\s+that\s+(.+)$", text, flags=re.I)
    if match:
        return _clean_sentence(f"The {match.group(1).strip(' ,;:')} {match.group(2).strip(' ,;:')}")
    match = re.match(r"^It is an?\s+([A-Za-z][A-Za-z'’-]{3,})(\b.*)$", text, flags=re.I)
    if match:
        return _clean_sentence(f"The {match.group(1)}{match.group(2)}")
    match = re.match(r"^This applies\b(.*)$", text, flags=re.I)
    antecedent = _right_to_phrase(previous)
    if match and antecedent:
        return _clean_sentence(f"The {antecedent} applies{match.group(1)}")
    return text


def _right_to_phrase(text: str) -> str:
    match = re.search(r"\bright\s+to\s+[A-Za-z'’-]+", str(text or ""), flags=re.I)
    return match.group(0).strip().casefold() if match else ""


def _content_words(text: str) -> set[str]:
    return set(_content_word_list(text))


_STOP_WORDS = {"the", "and", "that", "this", "with", "from", "into", "their", "them", "they", "should", "would", "could", "also"}
