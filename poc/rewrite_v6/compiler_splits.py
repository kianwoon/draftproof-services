from __future__ import annotations

import re

from .compiler_lists import (
    _article_phrase,
    _capitalize,
    _clean_item,
    _clean_start,
    _has_finite_verb,
    _has_packed_list,
    _period,
    _replace_initial_pronoun,
    _split_parts,
    _unpack_sentence,
)
from .compiler_routes import _shift_predictable_start
from .text import source_terms


def _has_contrast(text: str) -> bool:
    return bool(re.search(r",\s*(but|though|while|yet)\s+", text, flags=re.I))

def _split_attribution(text: str) -> list[str]:
    text = re.sub(r"^putting\s+it\s+into\s+practice,\s+", "", text, flags=re.I)
    if not re.search(r"^(according to|as)\b", text, flags=re.I):
        return []
    parts = [part.strip() for part in text.split(",", 1)]
    if len(parts) != 2 or not parts[1]:
        return []
    claim = _shift_predictable_start(parts[1])
    return [_period(_capitalize(parts[0])), _period(_capitalize(claim))]

def _split_report_claim(text: str) -> list[str]:
    match = re.match(r"^(?:with\s+(.+?),\s+)?(.+?\(\d{4}\))\s+point(?:s)?\s+out\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    setup = match.group(1)
    source = match.group(2).strip()
    claim = re.sub(r"\bthe\s+main\s+", "a ", match.group(3).strip(), flags=re.I)
    rows = []
    if setup:
        rows.append(_period(_capitalize(f"{setup.strip()} sets the context")))
    where = re.match(r"^(.+?)\s+where\s+(.+)$", claim, flags=re.I)
    if where:
        rows.append(_period(_capitalize(f"{source} points out {where.group(1).strip()}")))
        rows.append(_period(_capitalize(where.group(2).strip())))
    else:
        unpacked_claim = _split_requirement_list(claim) or (_unpack_sentence(claim) if _has_packed_list(claim) else [])
        if unpacked_claim:
            rows.append(_period(_capitalize(f"{source} points out {unpacked_claim[0]}")))
            rows.extend(unpacked_claim[1:])
        else:
            rows.append(_period(_capitalize(f"{source} points out {claim}")))
    return rows

def _split_leading_result(text: str) -> list[str]:
    match = re.match(r"^(.+?),\s+leading\s+(.+?)\s+and\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    return [
        _period(_capitalize(match.group(1).strip())),
        _period(_capitalize(f"This leads {match.group(2).strip()}")),
        _period(_capitalize(f"This also leads {match.group(3).strip()}")),
    ]

def _split_if_then(text: str) -> list[str]:
    match = re.match(r"^if\s+(.+?)\s+then\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    return [_period(_capitalize(f"If {match.group(1).strip()}")), _period(_capitalize(match.group(2).strip()))]

def _split_if_comma(text: str) -> list[str]:
    match = re.match(r"^if\s+(.+?),\s+(.+)$", text, flags=re.I)
    if not match or len(text.split()) < 14:
        return []
    condition = match.group(1).strip()
    result = match.group(2).strip()
    cause = re.match(r"^(.+?)\s+causes?\s+(.+?)\s+to\s+be\s+(.+)$", condition, flags=re.I)
    if cause:
        subject = cause.group(2).strip()
        state = cause.group(3).strip()
        first = f"{cause.group(1).strip()} can leave {subject} {state}"
    else:
        first = f"The condition is {condition}"
    result = _split_that_result(result)
    return [_period(_capitalize(first)), *result]

def _split_that_result(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+that\s+(.+)$", text, flags=re.I)
    if not match:
        return [_period(_capitalize(text))]
    first = match.group(1).strip()
    anchor_terms = source_terms(first, limit=1)
    anchor = anchor_terms[0] if anchor_terms else "The result"
    if anchor.casefold() in {"it", "this", "that", "they", "these", "those", "he", "she"}:
        anchor = "The result"
    return [_period(_capitalize(first)), _period(_capitalize(f"{anchor} {match.group(2).strip()}"))]

def _split_not_but(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+not\s+(.+?)\s+but\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    subject = match.group(1).strip()
    if subject.casefold().startswith(("it", "this", "that")):
        subject = "The point"
    return [
        _period(_capitalize(f"{subject} not {match.group(2).strip()}")),
        _period(_capitalize(f"{subject} {match.group(3).strip()}")),
    ]

def _split_rather_than_no_comma(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+rather\s+than\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    return [
        _period(_capitalize(match.group(1).strip())),
        _period(_capitalize(f"The other route is {match.group(2).strip()}")),
    ]

def _split_and_has_begun(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+and\s+has\s+begun\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    subject_match = re.match(r"^([^,.;:!?]{1,40}?)\s+(?:no longer|has|is|was|can|may|will|would|should)\b", match.group(1), flags=re.I)
    subject = subject_match.group(1).strip() if subject_match else "The subject"
    return [_period(_capitalize(match.group(1).strip())), _period(_capitalize(f"{subject} has begun {match.group(2).strip()}"))]

def _split_and_then_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+and\s+then\s+([A-Za-z]+)\s+(.+)$", text, flags=re.I)
    if not match or len(text.split()) < 16:
        return []
    base = match.group(1).strip()
    actor = _actor_for_followup(base)
    verb = _third_person_present(match.group(2).strip())
    return [
        _period(_capitalize(base)),
        _period(_capitalize(f"{actor} then {verb} {match.group(3).strip()}")),
    ]

def _actor_for_followup(text: str) -> str:
    encouraged = re.match(r"^.+?\bencourage\s+(.+?)\s+to\s+.+$", text, flags=re.I)
    if encouraged:
        return _capitalize(encouraged.group(1).strip())
    return _simple_subject(text)

def _split_semicolon_claim(text: str) -> list[str]:
    if ";" not in re.sub(r"\([^)]*\)", "", text):
        return []
    parts = [part.strip() for part in text.split(";", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return []
    second = re.sub(r"\bshould\s+involve\b", "involves", parts[1], flags=re.I)
    return [_period(_capitalize(parts[0])), _period(_capitalize(second))]

def _split_colon_claim(text: str) -> list[str]:
    parts = [part.strip() for part in text.split(":", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return []
    if not _has_finite_verb(parts[1]) or re.match(r"^[A-Za-z]+ing\b", parts[1]):
        return []
    first = re.sub(r"\ban?\s+important\s+", "", parts[0], flags=re.I)
    contrast = re.split(r",\s+not\s+", parts[1], maxsplit=1, flags=re.I)
    if len(contrast) == 2:
        tail = [_period(_capitalize(contrast[0])), _period(_capitalize(f"The contrast is not {contrast[1]}"))]
    else:
        tail = _unpack_sentence(parts[1]) if _has_packed_list(parts[1]) else []
    return [_period(_capitalize(first)), *(tail or [_period(_capitalize(parts[1]))])]

def _split_requirement_list(text: str) -> list[str]:
    match = re.match(r"^(.+?\brequires)\s+(.+)$", text, flags=re.I)
    if not match or not _has_packed_list(match.group(2)):
        return []
    subject = re.sub(r"\s+often\s*$", "", match.group(1).rsplit(" requires", 1)[0].strip(), flags=re.I)
    subject = re.sub(r"^that\s+", "", subject, flags=re.I)
    items = [item.strip(" .") for item in _split_parts(match.group(2)) if item.strip(" .")]
    if not items:
        return []
    first = _period(_capitalize(f"{match.group(1).strip()} {items[0]}"))
    rest = [_period(_capitalize(f"{subject} also requires {item}")) for item in items[1:]]
    return [first, *rest]

def _split_not_only(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+should\s+not\s+only\s+(.+?)[,，]([\"'”’])?\s*but\s+also\s+(.+?)[.]?$", text, flags=re.I)
    if not match:
        return []
    subject = match.group(1).strip()
    first = _clean_quote_spacing((match.group(2).strip() + (match.group(3) or "")).strip())
    second = _clean_quote_spacing(match.group(4).strip())
    second = _expand_parallel_tail(first, second)
    return [
        _period(_capitalize(f"{subject} should not only {first}")),
        _period(_capitalize(f"{subject} should also {second}")),
    ]

def _split_is_not_only_about(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+is\s+not\s+only\s+about\s+(.+?),?\s+but\s+also\s+about\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    subject = re.sub(r"^the\s+example\s+demonstrates\s+that\s+", "", match.group(1).strip(), flags=re.I)
    first = match.group(2).strip()
    second = match.group(3).strip()
    return [
        _period(_capitalize(f"{subject} goes beyond {first}")),
        _period(_capitalize(f"{subject} also involves {second}")),
    ]

def _split_not_only_subject(text: str) -> list[str]:
    match = re.match(r"^not\s+only\s+(.+?)\s+but\s+(.+?)\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    first_subject = match.group(1).strip()
    second_subject = match.group(2).strip()
    predicate = match.group(3).strip()
    result = ""
    result_match = re.match(r"^(.+?),\s+thereby\s+(.+)$", predicate, flags=re.I)
    if result_match:
        predicate = result_match.group(1).strip()
        result = result_match.group(2).strip()
    rows = [
        _period(_capitalize(f"{first_subject} {predicate}")),
        _period(_capitalize(f"{second_subject} {predicate}")),
    ]
    if result:
        rows.append(_period(_capitalize(f"The result {result}")))
    return rows

def _expand_parallel_tail(first: str, second: str) -> str:
    first_words = first.split()
    second_words = second.split()
    if not first_words or not second_words:
        return second
    connectors = {"on", "in", "to", "for", "with", "about", "through", "from", "by", "as", "into", "around"}
    second_first = second_words[0].casefold().strip(".,:;!?\"'“”")
    lowered = [word.casefold().strip(".,:;!?\"'“”") for word in first_words]
    for index in range(len(first_words) - 1, 0, -1):
        if lowered[index] in connectors:
            prefix = " ".join(first_words[: index if second_first == lowered[index] else index + 1])
            if not second.casefold().startswith(prefix.casefold()):
                return f"{prefix} {second}"
    return second

def _clean_quote_spacing(text: str) -> str:
    text = re.sub(r"\s+([\"'”’])", r"\1", text.strip())
    return re.sub(r"\.\s+([\"'”’])$", r".\1", text)

def _split_contrast(text: str) -> list[str]:
    parts = re.split(r",\s*(?:but|though|while|yet)\s+", text, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return [_period(_capitalize(text))]
    first = _clean_start(parts[0])
    raw_second = parts[1].strip()
    second = _complete_contrast_fragment(raw_second, first)
    if second == raw_second:
        second = _replace_initial_pronoun(second, first)
    sentences: list[str] = []
    for part in (first, second):
        if _has_packed_list(part):
            sentences.extend(_unpack_sentence(part) or [_period(_capitalize(part))])
        else:
            sentences.append(_period(_capitalize(part)))
    return sentences

def _split_dash_clause(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\s*[—–]\s*", text, maxsplit=1) if part.strip()]
    if len(parts) != 2:
        return []
    return [_period(_capitalize(part)) for part in parts]

def _split_citation_relation(text: str) -> list[str]:
    if not re.search(r"\([^)]*\d{4}[^)]*\)", text):
        return []
    parts = re.split(r",\s+and\s+", text, maxsplit=1)
    if len(parts) != 2:
        return []
    return [_period(_capitalize(part)) for part in parts]

def _split_preposed_context_clause(text: str) -> list[str]:
    following = re.match(r"^following\s+completion\s+of\s+(.+?),\s+(.+)$", text, flags=re.I)
    if following:
        return [_period(_capitalize(f"{following.group(2).strip()} after {following.group(1).strip()} is complete"))]
    match = re.match(r"^((?:in|during|through|after|before|at|to)\s+[^,]{2,80}),\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    if not _has_finite_verb(match.group(1)):
        context = match.group(1).strip()
        rest = match.group(2).strip(" .")
        shifted_rest = _shift_predictable_start(rest)
        not_only_about = _split_is_not_only_about(shifted_rest)
        if not_only_about:
            return not_only_about
        combined = f"{context}, {rest}"
        nested_combined = _unpack_sentence(combined) if _has_packed_list(combined) else []
        if nested_combined:
            return nested_combined
        nested_rest = _split_from_who_clause(rest)
        if nested_rest:
            return [_period(f"{nested_rest[0].rstrip('.')} {_lower_first(context)}"), *nested_rest[1:]]
        moved = f"{_capitalize(shifted_rest)} {_lower_first(context)}"
        nested_moved = _split_from_who_clause(moved)
        return nested_moved or [_period(moved)]
    rest = _shift_predictable_start(match.group(2).strip())
    if len(rest.split()) < 6:
        return []
    nested = _split_comma_pronoun_result(rest) or _split_including_clause(rest) or (_unpack_sentence(rest) if _has_packed_list(rest) else [])
    return [_period(_capitalize(match.group(1).strip())), *(nested or [_period(_capitalize(rest))])]

def _split_prepositional_fragment(text: str) -> list[str]:
    match = re.match(r"^in\s+([A-Za-z][^.!?]{3,60})[.]?$", text, flags=re.I)
    if not match:
        return []
    if match.group(1).casefold().startswith("addition"):
        return []
    return [_period(_capitalize(match.group(1).strip()))]

def _split_during_gerund_context(text: str) -> list[str]:
    if " when " in f" {text.casefold()} ":
        return []
    match = re.match(r"^(.+?)\s+during\s+(.+?)\s+([A-Za-z]+)ing\s+(.+)$", text, flags=re.I)
    if not match or len(text.split()) < 16:
        return []
    verb = _third_person_present(match.group(3).strip())
    return [
        _period(_capitalize(f"{match.group(1).strip()} during {match.group(2).strip()}")),
        _period(_capitalize(f"The event {verb} {match.group(4).strip()}")),
    ]

def _split_heading_merge(text: str) -> list[str]:
    for match in re.finditer(r"\s+([A-Z][A-Za-z]+(?:\s+[a-z][A-Za-z'-]+){2,}.+)$", text):
        if match.start() < 10:
            continue
        first = text[: match.start()].strip()
        second = re.sub(r"\bneeds\s+to\s+be\b", "has to be", match.group(1).strip(), flags=re.I)
        return [_period(_capitalize(first)), _period(_capitalize(second))]
    return []

def _split_because_wrapper(text: str) -> list[str]:
    match = re.match(r"^(?:this|that|it|the result)\s+is\s+(?:also\s+)?because\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    claim = match.group(1).strip()
    if "," not in claim:
        return [_period(_capitalize(claim))]
    parts = [part.strip() for part in claim.split(",", 1)]
    return [_period(_capitalize(parts[0])), _period(_capitalize(parts[1]))]

def _split_where_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?),\s+where\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    rest_parts = _unpack_sentence(match.group(2).strip())
    return [_period(_capitalize(match.group(1).strip())), *(rest_parts or [_period(_capitalize(match.group(2).strip()))])]

def _split_rather_than_clause(text: str) -> list[str]:
    quoted = re.match(r"^rather\s+than\s+(.+?[\"”]),?\s+(.+)$", text, flags=re.I)
    if quoted:
        rest = quoted.group(2).strip()
        nested = _split_including_clause(rest) or _unpack_sentence(rest)
        return [_period(_capitalize(f"The first route is not {quoted.group(1).strip()}")), *(nested or [_period(_capitalize(rest))])]
    match = re.match(r"^rather\s+than\s+(.+?),\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    rest = match.group(2).strip()
    nested = _split_including_clause(rest) or _unpack_sentence(rest)
    return [_period(_capitalize(f"The first route is not {match.group(1).strip()}")), *(nested or [_period(_capitalize(rest))])]

def _split_comma_pronoun_result(text: str) -> list[str]:
    match = re.match(
        r"^(.+?),\s+(it|this|that|they)\s+(leads?|creates?|causes?|means?|shows?|suggests?|allows?|helps?)\s+(.+)$",
        text,
        flags=re.I,
    )
    if not match:
        return []
    first = match.group(1).strip()
    pronoun = match.group(2).casefold()
    result_subject = "Those points" if pronoun == "they" else "The result"
    result = f"{result_subject} {match.group(3)} {match.group(4).strip()}"
    return [_period(_capitalize(first)), _period(_capitalize(result))]

def _split_comma_subject_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?),\s+((?:the|a|an|this|that|these|those)\s+[^,]{1,80}\s+\w+s\b.+)$", text, flags=re.I)
    if not match:
        return []
    second = _shift_predictable_start(match.group(2).strip())
    return [_period(_capitalize(match.group(1).strip())), _period(_capitalize(second))]

def _split_which_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?),\s+which\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    first = _shift_predictable_start(match.group(1).strip())
    return [_period(_capitalize(first)), _period(_capitalize(f"The linked point {match.group(2).strip()}"))]

def _split_because_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?),?\s+because\s+(.+)$", text, flags=re.I)
    if not match or len(text.split()) < 18:
        return []
    reason = match.group(2).strip()
    reason_parts = _unpack_sentence(reason) if _has_packed_list(reason) else []
    return [_period(_capitalize(match.group(1).strip())), *(reason_parts or [_period(_capitalize(reason))])]

def _reframe_author_year_claim(text: str) -> list[str]:
    match = re.match(r"^([A-Z][A-Za-z'-]+(?:\s+et al\.)?)\s+\((\d{4})\)\s+(?:states|indicates|argues|argue|notes|describes)(?:\s+that|,)?\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    claim = _capitalize(match.group(3).strip())
    citation = f"({match.group(1)}, {match.group(2)})"
    return [_period(f"{claim} {citation}")]

def _split_when_clause(text: str) -> list[str]:
    parts = re.split(r"\s+when\s+", text, maxsplit=1, flags=re.I)
    if len(parts) != 2 or len(text.split()) < 20:
        return []
    return [_period(_capitalize(parts[0].strip())), _period(_capitalize(f"When {parts[1].strip()}"))]

def _split_while_clause(text: str) -> list[str]:
    parts = re.split(r"\s+while\s+", text, maxsplit=1, flags=re.I)
    if len(parts) != 2 or len(text.split()) < 8:
        return []
    if not _has_finite_verb(parts[1]):
        gerund_tail = _split_gerund_tail(parts[0].strip(), parts[1].strip())
        if gerund_tail:
            return gerund_tail
        return []
    return [_period(_capitalize(parts[0].strip())), _period(_capitalize(f"While {parts[1].strip()}"))]

def _split_gerund_tail(base: str, tail: str) -> list[str]:
    match = re.match(r"^([A-Za-z]+)ing\s+(.+)$", tail, flags=re.I)
    if not match:
        return []
    subject = _simple_subject(base)
    verb = _third_person_present(match.group(1))
    if not _has_packed_list(tail):
        return [_period(_capitalize(base)), _period(_capitalize(f"{subject} {verb} {match.group(2).strip()}"))]
    nested = _unpack_sentence(f"{subject} {verb} {match.group(2).strip()}")
    return [_period(_capitalize(base)), *(nested or [_period(_capitalize(f"{subject} {verb} {match.group(2).strip()}"))])]

def _simple_subject(text: str) -> str:
    match = re.match(r"^((?:the|a|an|this|that|these|those|my|our)\s+[^,.;:!?]{1,40}?)\s+\w+\b", text, flags=re.I)
    return _capitalize(match.group(1).strip()) if match else "The subject"

def _third_person_present(stem: str) -> str:
    base = {"creat": "create", "ensur": "ensure", "mak": "make", "us": "use", "tak": "take"}.get(stem.casefold(), stem)
    return base + ("es" if base.endswith(("s", "sh", "ch", "x", "z")) else "s")

def _split_showing_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?),\s+showing\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    return [
        _period(_capitalize(match.group(1).strip())),
        _period(_capitalize(f"The same evidence showed {match.group(2).strip()}")),
    ]

def _split_then_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?),\s+then\s+([A-Za-z]+)\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    subject = _simple_subject(match.group(1).strip())
    verb = _third_person_present(match.group(2).strip())
    return [
        _period(_capitalize(match.group(1).strip())),
        _period(_capitalize(f"{subject} then {verb} {match.group(3).strip()}")),
    ]

def _split_thereby_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?),\s+thereby\s+([A-Za-z]+)ing\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    verb = _third_person_present(match.group(2).strip())
    return [
        _period(_capitalize(match.group(1).strip())),
        _period(_capitalize(f"The result {verb} {match.group(3).strip()}")),
    ]

def _split_creating_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?),\s+creating(?:\s+and\s+developing)?\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    first = match.group(1).strip()
    effect = match.group(2).strip()
    return [
        _period(_capitalize(first)),
        _period(_capitalize(f"The result creates {effect}")),
    ]

def _split_from_who_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+from\s+(.+?)\s+who\s+(.+)$", text, flags=re.I)
    if not match or len(text.split()) < 10:
        return []
    group = match.group(2).strip()
    return [
        _period(_capitalize(f"{match.group(1).strip()} from {group}")),
        _period(_capitalize(f"{group} {match.group(3).strip()}")),
    ]

def _split_involves_represents(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+involves\s+(.+?)\s+and\s+represents\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    subject = match.group(1).strip()
    return [
        _period(_capitalize(f"{subject} involves {match.group(2).strip()}")),
        _period(_capitalize(f"{subject} represents {match.group(3).strip()}")),
    ]

def _split_provides_through(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+provides\s+(.+?)\s+through\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    subject = match.group(1).strip()
    return [
        _period(_capitalize(f"{subject} provides {match.group(2).strip()}")),
        _period(_capitalize(f"The route works through {match.group(3).strip()}")),
    ]

def _split_relative_used_to(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+that\s+can\s+be\s+used\s+to\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    subject = match.group(1).strip()
    active_subject = "The model" if subject.casefold().startswith(("it is", "this is", "that is")) else subject
    action = re.sub(r"^(measure|evaluate)\s+", "", match.group(2).strip(), flags=re.I)
    return [_period(_capitalize(subject)), _period(_capitalize(f"{active_subject} can measure {action}"))]

def _split_is_modal_predicate(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+is\s+(.+?)\s+(can|could|may|might|must|should|would|will)\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    if re.search(r"\b(that|where|which|who|because)\b|,", match.group(2), flags=re.I):
        return []
    subject_term = source_terms(match.group(1), limit=1)
    subject = subject_term[0] if subject_term else "The subject"
    predicate = match.group(4).strip()
    measure_scope = _split_measure_scope(subject, predicate)
    if measure_scope:
        return [_period(_capitalize(f"{match.group(1).strip()} is {match.group(2).strip()}")), *measure_scope]
    return [
        _period(_capitalize(f"{match.group(1).strip()} is {match.group(2).strip()}")),
        _period(_capitalize(f"{subject} {match.group(3).strip()} {predicate}")),
    ]

def _split_measure_scope(subject: str, predicate: str) -> list[str]:
    match = re.match(r"measure\s+(.+?)\s+regarding\s+(.+?)\s+or\s+(.+)$", predicate, flags=re.I)
    if not match:
        return []
    return [
        _period(_capitalize(f"{subject} can measure {match.group(1).strip()}")),
        _period(_capitalize(f"The first area is {match.group(2).strip()}")),
        _period(_capitalize(f"The second area is {match.group(3).strip()}")),
    ]

def _split_including_clause(text: str) -> list[str]:
    match = re.match(r"^(.+?)\s+including\s+(.+?)\s+and\s+(whether|how|if)\s+(.+)$", text, flags=re.I)
    if match:
        base = _clean_clause_end(match.group(1))
        first = _clean_clause_end(match.group(2))
        second = _clean_clause_end(f"{match.group(3)} {match.group(4)}")
        return [
            _period(_capitalize(base)),
            _period(_capitalize(f"The first check is {first}")),
            _period(_capitalize(f"The second check is {second}")),
        ]
    plain = re.match(r"^(.+?)\s+including\s+(.+?)\s+and\s+(.+)$", text, flags=re.I)
    if not plain:
        return []
    base = _clean_clause_end(plain.group(1))
    first = _clean_clause_end(plain.group(2))
    second = _clean_clause_end(plain.group(3))
    return [
        _period(_capitalize(base)),
        _period(_capitalize(f"One included issue is {first}")),
        _period(_capitalize(f"Another included issue is {second}")),
    ]

def _clean_clause_end(text: str) -> str:
    return text.strip(" .,;:")

def _lower_first(text: str) -> str:
    stripped = text.strip()
    return stripped[:1].lower() + stripped[1:] if stripped else stripped

def _split_or_modal(text: str) -> list[str]:
    parts = re.split(r",?\s+or\s+(it|this|that|they)\s+(can|could|may|might|must|should|would|will)\s+", text, maxsplit=1, flags=re.I)
    if len(parts) != 4:
        return []
    first, pronoun, modal, rest = parts
    return [_period(_capitalize(first.strip())), _period(_capitalize(f"{pronoun} {modal} {rest.strip()}"))]

def _complete_contrast_fragment(text: str, previous: str) -> str:
    lowered = text.casefold()
    match = re.match(r"^([A-Z][\w'-]*)\s+(may|might|can|could|should|would|will)\s+(\w+)\b", previous)
    if match and lowered.startswith("not always how "):
        subject, modal, verb = match.group(1), match.group(2), match.group(3)
        return f"{subject} {modal} not always {verb} {text[len('not always '):]}"
    match = re.match(r"^([A-Z][\w'-]*)\s+(\w+)\s+from\s+", previous)
    if match and lowered.startswith("also from "):
        return f"{match.group(1)} also {match.group(2)} from {text[len('also from '):]}"
    match = re.match(r"^(.+?)\s+should\s+include\s+not\s+only\s+.+$", previous, flags=re.I)
    if match and lowered.startswith("also "):
        return f"{match.group(1)} should also include {text[len('also '):]}"
    return text
