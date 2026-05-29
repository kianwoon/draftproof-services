from __future__ import annotations

import re

from .text import Paragraph


def scope_marker_reused_as_content(candidate_text: str, paragraph: Paragraph) -> bool:
    source = str(paragraph.text or "").casefold()
    candidate = str(candidate_text or "").casefold()
    if "no longer" not in source:
        return False
    for match in re.finditer(r"\blonger\b", candidate):
        before = candidate[max(0, match.start() - 4):match.start()]
        if not before.endswith("no "):
            return True
    return False


def unsupported_semantic_padding(candidate_text: str, paragraph: Paragraph) -> bool:
    source_bases = _content_bases(paragraph.text)
    if not source_bases:
        return False
    source_sentences = [sentence.text for sentence in paragraph.sentences if sentence.text.strip()]
    for sentence in _candidate_sentences(candidate_text):
        candidate_tokens = _content_tokens(sentence)
        if len(candidate_tokens) < 8:
            continue
        if _max_unsupported_run(candidate_tokens, source_bases) >= 5:
            return True
        matched_source = _best_source_sentence(candidate_tokens, source_sentences)
        if not matched_source:
            continue
        matched_bases = _content_bases(matched_source)
        if len(matched_bases) < 5:
            continue
        overlap = sum(1 for token in candidate_tokens if _token_base(token) in matched_bases)
        if overlap < 5:
            continue
        if _max_unsupported_run(candidate_tokens, source_bases) >= 5:
            return True
    return False


def _max_unsupported_run(tokens: list[str], source_bases: set[str]) -> int:
    unsupported_run = 0
    max_run = 0
    for token in tokens:
        base = _token_base(token)
        if base not in source_bases:
            unsupported_run += 1
            max_run = max(max_run, unsupported_run)
        else:
            unsupported_run = 0
    return max_run


def source_quality_blockers(candidate_text: str, paragraph: Paragraph) -> list[str]:
    blockers: list[str] = []
    if scope_marker_reused_as_content(candidate_text, paragraph):
        blockers.append("source_scope_marker_reused_as_content")
    if unsupported_source_channel_list(candidate_text, paragraph):
        blockers.append("unsupported_semantic_padding")
    if source_frame_reified_as_entity(candidate_text, paragraph):
        blockers.append("source_frame_reified_as_entity")
    if source_contrast_reframed(candidate_text, paragraph):
        blockers.append("source_contrast_reframed")
    if source_beat_reintroduced_after_closing(candidate_text, paragraph):
        blockers.append("source_beat_reintroduced_after_closing")
    return blockers


def source_frame_reified_as_entity(candidate_text: str, paragraph: Paragraph) -> bool:
    source_frames = _created_effect_frame_rows(paragraph.text)
    if not source_frames:
        return False
    candidate = str(candidate_text or "")
    for subject, verb, _object_text, source_objects in source_frames:
        pattern = rf"\b{re.escape(subject)}\s+(?:(?:has|have)\s+)?(?:also\s+)?{re.escape(verb)}\s+([^.!?;,]+)"
        for match in re.finditer(pattern, candidate, flags=re.I):
            candidate_object = match.group(1).strip()
            if not candidate_object:
                continue
            candidate_bases = _content_bases(candidate_object)
            if not candidate_bases:
                continue
            if candidate_bases.isdisjoint(source_objects):
                return True
    return False


def created_effect_frame_constraints(text: str) -> list[dict[str, object]]:
    return [
        {
            "subject": subject,
            "source_object_terms": sorted(source_objects),
            "writer_rule": "Preserve this as an abstract effect, tension, opportunity, risk, or condition. Do not rewrite it as the subject creating a concrete tool, platform, person, document, or entity.",
        }
        for subject, source_objects in _created_effect_frames(text)
    ]


def created_effect_frame_anchor_blockers(text: str) -> set[str]:
    blockers: set[str] = set()
    for subject, verb, _object_text, _source_objects in _created_effect_frame_rows(text):
        blockers.update(_content_bases(subject))
        if verb.strip():
            blockers.add(verb.casefold().strip())
    return blockers


def created_effect_frame_anchor_constraints(text: str) -> list[dict[str, object]]:
    return [
        {
            "subject": subject,
            "verb": verb,
            "object_text": object_text,
            "anchor_blockers": sorted({*_content_bases(subject), verb.casefold().strip()} - {""}),
            "safe_meanings": _created_effect_safe_meanings(object_text),
        }
        for subject, verb, object_text, _source_objects in _created_effect_frame_rows(text)
    ]


def created_effect_frame_meanings(text: str) -> list[str]:
    return [
        meaning
        for _subject, _verb, object_text, _source_objects in _created_effect_frame_rows(text)
        for meaning in _created_effect_safe_meanings(object_text)
    ]


def _created_effect_frames(text: str) -> list[tuple[str, set[str]]]:
    return [
        (subject, source_objects)
        for subject, _verb, _object_text, source_objects in _created_effect_frame_rows(text)
    ]


def _created_effect_frame_rows(text: str) -> list[tuple[str, str, str, set[str]]]:
    rows: list[tuple[str, str, str, set[str]]] = []
    for sentence in _candidate_sentences(text):
        frame = _split_source_frame(sentence)
        if not frame:
            continue
        subject, verb, object_text = frame
        object_bases = _content_bases(object_text)
        if subject and _frame_verb_shape(verb) and object_bases and _coordinated_frame_object(object_text):
            rows.append((subject, verb, object_text, object_bases))
    return rows


def _split_source_frame(sentence: str) -> tuple[str, str, str] | None:
    visible = re.sub(r"\s+", " ", str(sentence or "")).strip(" .")
    both_match = re.search(r"\bboth\s+(.+\b(?:and|or)\b.+)$", visible, flags=re.I)
    if both_match:
        prefix = visible[:both_match.start()].strip()
        object_text = both_match.group(1).strip()
    else:
        relation_match = re.search(r"\b(and|or)\b\s+([^.!?;,]+)$", visible, flags=re.I)
        if not relation_match:
            return None
        left = visible[:relation_match.start()].strip()
        if "," in left:
            return None
        right_object = relation_match.group(2).strip()
        if len(_content_tokens(right_object)) > 3:
            return None
        left_tokens = list(re.finditer(r"[A-Za-z][A-Za-z'’-]*", left))
        if len(left_tokens) < 2:
            return None
        first_object = left_tokens[-1].group(0)
        prefix = left[:left_tokens[-1].start()].strip()
        object_text = f"{first_object} {relation_match.group(1)} {right_object}"
    tokens = list(re.finditer(r"[A-Za-z][A-Za-z'’-]*", prefix))
    if len(tokens) < 2:
        return None
    verb = tokens[-1].group(0)
    subject = prefix[:tokens[-1].start()].strip()
    subject = re.sub(r"^(?:now|today|however|because|therefore|thus)\s+", "", subject, flags=re.I)
    subject = re.sub(r"\b(?:has|have|had|also)\b\s*$", "", subject, flags=re.I).strip()
    subject = re.sub(r"\b(?:has|have|had|also)\b\s*$", "", subject, flags=re.I).strip()
    subject = subject.split(",")[-1].strip()
    if not subject:
        return None
    return subject, verb, object_text


def _clean_frame_object(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" .,;:")
    value = re.sub(r"^both\s+", "", value, flags=re.I).strip()
    if value[:1].isupper():
        value = value[:1].lower() + value[1:]
    return value


def _created_effect_safe_meanings(object_text: str) -> list[str]:
    items = _coordinated_object_items(object_text)
    if len(items) >= 2:
        return [_join_effect_items(items[:4])]
    cleaned = _clean_frame_object(object_text)
    return [cleaned] if cleaned else []


def _join_effect_items(items: list[str]) -> str:
    visible = [item for item in items if item]
    if len(visible) <= 1:
        return visible[0] if visible else ""
    if len(visible) == 2:
        return f"{visible[0]} alongside {visible[1]}"
    return f"{', '.join(visible[:-1])}, alongside {visible[-1]}"


def _coordinated_object_items(text: str) -> list[str]:
    visible = _clean_frame_object(text)
    if not visible:
        return []
    parts = [
        re.sub(r"\s+", " ", part).strip(" .,;:")
        for part in re.split(r"\b(?:and|or)\b", visible, flags=re.I)
    ]
    return [part for part in parts if part]


def _coordinated_frame_object(text: str) -> bool:
    return bool(re.search(r"\b\S+\s+(?:and|or)\s+\S+\b", str(text or ""), flags=re.I))


def _frame_verb_shape(text: str) -> bool:
    verb = str(text or "").casefold().strip("'’")
    return len(verb) > 3 and (verb.endswith("s") or verb.endswith("ed"))



def unsupported_source_channel_list(candidate_text: str, paragraph: Paragraph) -> bool:
    source_bases = _content_bases(paragraph.text)
    if not source_bases:
        return False
    for sentence in _candidate_sentences(candidate_text):
        lowered = sentence.casefold()
        marker = re.search(r"\b(?:include|includes|including|such\s+as|range\s+from|ranges\s+from|from)\b", lowered)
        if not marker:
            continue
        tail = sentence[marker.end():]
        tokens = _content_tokens(tail)
        if len(tokens) < 5:
            continue
        unsupported = [token for token in tokens if _token_base(token) not in source_bases]
        supported = [token for token in tokens if _token_base(token) in source_bases]
        if len(unsupported) >= 4 and len(unsupported) > len(supported):
            return True
    return False


def source_contrast_reframed(candidate_text: str, paragraph: Paragraph) -> bool:
    source = str(paragraph.text or "").casefold()
    candidate = str(candidate_text or "").casefold()
    if "not less important" not in source:
        return False
    return bool(re.search(r"\bmore\s+important\s+and\s+less\s+(?:overlooked|ignored|visible|valued)\b", candidate))


def source_beat_reintroduced_after_closing(candidate_text: str, paragraph: Paragraph) -> bool:
    source_sentences = [sentence.text for sentence in paragraph.sentences if sentence.text.strip()]
    candidate_sentences = _candidate_sentences(candidate_text)
    if len(source_sentences) < 3 or len(candidate_sentences) < 3:
        return False
    source_bases = [_content_bases(sentence) for sentence in source_sentences]
    closing_bases = source_bases[-1]
    if len(closing_bases) < 3:
        return False
    closing_candidate_index = _first_candidate_covering_source(candidate_sentences, closing_bases)
    if closing_candidate_index is None or closing_candidate_index >= len(candidate_sentences) - 1:
        return False
    earlier_sources = source_bases[:-1]
    for sentence in candidate_sentences[closing_candidate_index + 1:]:
        candidate_bases = _content_bases(sentence)
        if len(candidate_bases) < 4:
            continue
        closing_overlap = len(candidate_bases & closing_bases)
        earlier_overlap = max((len(candidate_bases & bases) for bases in earlier_sources), default=0)
        if earlier_overlap >= 4 and earlier_overlap > closing_overlap:
            return True
    return False


def _first_candidate_covering_source(candidate_sentences: list[str], source_bases: set[str]) -> int | None:
    threshold = min(4, max(3, len(source_bases) // 2))
    for index, sentence in enumerate(candidate_sentences):
        candidate_bases = _content_bases(sentence)
        if len(candidate_bases & source_bases) >= threshold:
            return index
    return None


def _best_source_sentence(candidate_tokens: list[str], source_sentences: list[str]) -> str:
    candidate_bases = {_token_base(token) for token in candidate_tokens}
    best_sentence = ""
    best_overlap = 0
    for sentence in source_sentences:
        bases = _content_bases(sentence)
        overlap = len(candidate_bases & bases)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
    return best_sentence


def _candidate_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]


def _content_bases(text: str) -> set[str]:
    return {_token_base(token) for token in _content_tokens(text)}


def _content_tokens(text: str) -> list[str]:
    stop = {
        "about", "above", "after", "again", "against", "also", "being", "between", "could",
        "during", "each", "every", "from", "have", "into", "just", "know", "more", "most",
        "only", "other", "over", "quite", "rather", "should", "some", "still", "their",
        "there", "these", "those", "through", "under", "where", "which", "while", "will",
        "would", "than", "that", "this", "they", "with", "many", "because", "however",
        "therefore", "does", "what", "then", "today", "past", "system",
    }
    return [
        token.casefold()
        for token in re.findall(r"\b[A-Za-z][A-Za-z'-]{3,}\b", str(text or ""))
        if token.casefold() not in stop
    ]


def _token_base(token: str) -> str:
    value = str(token or "").casefold().strip("'")
    if len(value) > 5 and value.endswith("ing"):
        stem = value[:-3]
        return stem + "e" if not stem.endswith("e") else stem
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith("s"):
        return value[:-1]
    return value
