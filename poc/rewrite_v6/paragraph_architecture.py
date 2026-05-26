from __future__ import annotations

import re
from typing import Any

from .plan import Plan
from .text import Paragraph, source_terms, split_paragraphs, split_sentences, strip_leading_heading, word_count


def architecture_split_contract(paragraph: Paragraph, plan: Plan) -> dict[str, Any]:
    actions = plan.actions
    tags = [tag for action in actions for tag in action.tags]
    overload = sum(1 for tag in tags if tag in {"sentence_overload", "packed_list", "paragraph_rhythm"})
    anchor_gaps = sum(1 for tag in tags if tag in {"context_anchor_gap", "author_anchor_gap", "unsupported_claim_gap"})
    words = word_count(paragraph.text)
    active = (len(paragraph.sentences) >= 6 and overload >= 4 and (words >= 70 or anchor_gaps >= 2)) or (words >= 110 and overload >= 3)
    return {
        "active": active,
        "paragraph_count": "as many functional paragraphs as needed",
        "functional_groups": _functional_groups(paragraph, actions) if active else [],
        "split_rule": (
            "When active, divide the selected source paragraph into smaller paragraphs by function: "
            "source/support frame, author/context evidence, reasoning or consequence. "
            "Use functional_groups as the paragraph architecture map. "
            "Every required source anchor must survive across the new paragraphs; paragraph splitting is not summarization. "
            "Do not split by every sentence, do not add new facts, and keep the replacement inside the selected section."
        ),
    }


def _functional_groups(paragraph: Paragraph, actions: list[Any]) -> list[dict[str, Any]]:
    source_terms_by_id = {
        action.sentence_id: source_terms(strip_leading_heading(action.source_text), limit=48)
        for action in actions
    }
    source_text_by_id = {action.sentence_id: strip_leading_heading(action.source_text) for action in actions}
    groups: list[dict[str, Any]] = []
    current: list[str] = []
    current_role = ""
    for sentence in paragraph.sentences:
        role = _sentence_role(sentence.text)
        if current and role != current_role:
            groups.append(_group_row(current_role, current, source_terms_by_id, source_text_by_id))
            current = []
        current.append(sentence.id)
        current_role = role
    if current:
        groups.append(_group_row(current_role, current, source_terms_by_id, source_text_by_id))
    return _merge_small_groups(groups)


def _sentence_role(text: str) -> str:
    if _author_or_task_sentence(text):
        return "author/context evidence"
    if _reasoning_sentence(text):
        return "reasoning or consequence"
    return "source/support frame"


def _group_row(role: str, sentence_ids: list[str], terms_by_id: dict[str, list[str]], text_by_id: dict[str, str]) -> dict[str, Any]:
    terms: list[str] = []
    for sentence_id in sentence_ids:
        terms.extend(terms_by_id.get(sentence_id, []))
    return {"role": role, "source_sentence_ids": sentence_ids, "source_texts": [text_by_id.get(sentence_id, "") for sentence_id in sentence_ids], "must_survive_terms": _dedupe(terms)[:90]}


def _merge_small_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        if rows and len(group["source_sentence_ids"]) == 1 and len(rows[-1]["source_sentence_ids"]) == 1:
            rows[-1]["source_sentence_ids"].extend(group["source_sentence_ids"])
            rows[-1]["source_texts"].extend(group["source_texts"])
            rows[-1]["must_survive_terms"] = _dedupe(rows[-1]["must_survive_terms"] + group["must_survive_terms"])[:120]
            rows[-1]["role"] = rows[-1]["role"] + " + " + group["role"] if rows[-1]["role"] != group["role"] else rows[-1]["role"]
        else:
            rows.append(group)
    return rows


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value).casefold()
        if key and key not in seen:
            rows.append(str(value))
            seen.add(key)
    return rows


def apply_architecture_split_text(text: str, contract: dict[str, Any]) -> str:
    if not contract.get("active") or len(split_paragraphs(text)) != 1:
        return text
    sentences = [sentence.text for sentence in split_sentences(text, paragraph_id="candidate")]
    if len(sentences) < 6:
        return _decompress_comma_heavy_sentences(_backfill_missing_group_anchors(text, contract))
    first_break = _first_index(sentences, 2, _author_or_task_sentence) or max(2, len(sentences) // 3)
    second_break = _first_index(sentences, first_break + 2, _reasoning_sentence) or max(first_break + 2, (len(sentences) * 2) // 3)
    breaks = sorted({index for index in (first_break, second_break) if 1 < index < len(sentences)})
    if not breaks:
        return text
    chunks: list[str] = []
    start = 0
    for stop in [*breaks, len(sentences)]:
        chunks.append(" ".join(sentences[start:stop]).strip())
        start = stop
    return _decompress_comma_heavy_sentences(_backfill_missing_group_anchors("\n\n".join(chunk for chunk in chunks if chunk), contract))


def _backfill_missing_group_anchors(text: str, contract: dict[str, Any]) -> str:
    candidate_bases = {_word_base(term) for term in source_terms(text, limit=220)}
    additions: list[str] = []
    for group in contract.get("functional_groups") or []:
        terms = [str(term) for term in group.get("must_survive_terms", []) if str(term).strip()]
        missing = [term for term in terms if _word_base(term) not in candidate_bases]
        if len(missing) < 2:
            continue
        sentence = _source_anchor_sentence(group.get("source_texts") or [], missing)
        if sentence and not set(_word_base(term) for term in source_terms(sentence, limit=80)) <= candidate_bases:
            additions.append(sentence)
            candidate_bases.update(_word_base(term) for term in source_terms(sentence, limit=80))
    if not additions:
        return text
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return text
    parts = [paragraph.text for paragraph in paragraphs]
    parts[min(1, len(parts) - 1)] = parts[min(1, len(parts) - 1)].rstrip() + " " + " ".join(additions)
    return "\n\n".join(parts)


def _source_anchor_sentence(source_texts: list[str], missing_terms: list[str]) -> str:
    best = ""
    best_hits = 0
    missing_bases = {_word_base(term) for term in missing_terms}
    for source_text in source_texts:
        for fragment in re.split(r"[,;]", str(source_text or "")):
            hits = sum(1 for term in source_terms(fragment, limit=80) if _word_base(term) in missing_bases)
            if hits > best_hits:
                best, best_hits = fragment, hits
    if best_hits < 2:
        return ""
    value = re.sub(r"\s+", " ", best.strip(" .,:;"))
    value = re.sub(r"\bhave\s+still\s+remained\b", "remained", value, flags=re.I)
    value = re.sub(r"\bstill\s+remained\b", "remained", value, flags=re.I)
    return value[:1].upper() + value[1:] + "."


def _decompress_comma_heavy_sentences(text: str) -> str:
    paragraphs: list[str] = []
    for paragraph in split_paragraphs(text):
        sentences: list[str] = []
        for sentence in split_sentences(paragraph.text, paragraph_id="candidate"):
            sentences.extend(_split_comma_sentence(sentence.text))
        paragraphs.append(" ".join(sentences).strip())
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _split_comma_sentence(sentence: str) -> list[str]:
    value = str(sentence or "").strip()
    if value.count(",") < 3:
        return [value]
    parts = [part.strip(" .,:;") for part in value.rstrip(".!?").split(",") if part.strip(" .,:;")]
    if len(parts) < 3:
        return [value]
    return [_sentence(part) for part in parts]


def _sentence(fragment: str) -> str:
    value = re.sub(r"\s+", " ", str(fragment or "").strip(" .,:;"))
    return value[:1].upper() + value[1:] + "."


def _word_base(value: str) -> str:
    return str(value or "").casefold().removesuffix("'s").rstrip("s")


def _first_index(sentences: list[str], start: int, predicate) -> int | None:
    for index in range(start, len(sentences) - 1):
        if predicate(sentences[index]):
            return index
    return None


def _author_or_task_sentence(sentence: str) -> bool:
    return bool(re.search(r"\b(?:I|my|we|our|delivery|assessment|assessed|adapted|method|criteria|task)\b", sentence, re.I))


def _reasoning_sentence(sentence: str) -> bool:
    return bool(re.search(r"\b(?:when|result|therefore|reason|because|compromise|client|workplace|inclusive|expectation|obligation|conclusion)\b", sentence, re.I))
