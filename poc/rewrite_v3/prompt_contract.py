"""Compact scanner-derived prompt contracts for V3 executors."""

from __future__ import annotations

from typing import Any


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def unique_text(values: list[Any], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values or []:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def sentence_split(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(text or ""):
        if char not in ".!?":
            continue
        end = index + 1
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    tail = (text or "")[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def structural_fallback_spans(text: str, *, limit: int = 3) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for sentence in sentence_split(text):
        words = sentence.split()
        if len(words) < 6:
            continue
        punctuation_weight = sentence.count(",") * 1.5 + sentence.count(":") * 2.0 + sentence.count(";") * 2.0
        length_weight = min(len(words), 24) / 24
        candidates.append((punctuation_weight + length_weight, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return unique_text([_limit_text(sentence, 140) for _, sentence in candidates], limit=limit)


def _is_word_char(char: str) -> bool:
    return bool(char) and (char.isalnum() or char == "_")


def _word_boundary_span(text: str, start: int, end: int) -> bool:
    if start < 0 or end <= start or end > len(text):
        return False
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not _is_word_char(before) and not _is_word_char(after)


def _word_count(text: str) -> int:
    return len([part for part in str(text or "").replace("\n", " ").split(" ") if part.strip()])


def _trim_phrase(text: str) -> str:
    return str(text or "").strip(" \t\r\n,;:()[]{}")


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = start
    while left > 0 and text[left - 1] not in ".!?\n":
        left -= 1
    right = end
    while right < len(text) and text[right] not in ".!?\n":
        right += 1
    if right < len(text) and text[right] in ".!?":
        right += 1
    return left, right


def _word_window(text: str, start: int, end: int, *, before_words: int = 3, after_words: int = 2) -> str:
    word_spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and not _is_word_char(text[cursor]):
            cursor += 1
        if cursor >= len(text):
            break
        word_start = cursor
        while cursor < len(text) and _is_word_char(text[cursor]):
            cursor += 1
        word_spans.append((word_start, cursor))
    if not word_spans:
        return _trim_phrase(text[start:end])

    touched: list[int] = []
    for index, (word_start, word_end) in enumerate(word_spans):
        if not (word_end <= start or word_start >= end):
            touched.append(index)
    if not touched:
        return _trim_phrase(text[start:end])

    first = max(0, touched[0] - before_words)
    last = min(len(word_spans) - 1, touched[-1] + after_words)
    left = word_spans[first][0]
    right = word_spans[last][1]
    while right < len(text) and text[right] in ",;:":
        right += 1
    return _trim_phrase(text[left:right])


def _phrase_from_raw_span(raw_span: str, text: str) -> tuple[str, str]:
    span = _text(raw_span)
    if not span or span not in text:
        return "", "not_in_source"
    start = text.find(span)
    end = start + len(span)
    boundary_ok = _word_boundary_span(text, start, end)
    span_words = _word_count(span)
    if boundary_ok and span_words >= 2 and len(span) >= 8:
        return _trim_phrase(span), "scanner_exact_phrase"

    sentence_start, sentence_end = _sentence_bounds(text, start, end)
    sentence = text[sentence_start:sentence_end]
    local_start = max(0, start - sentence_start)
    local_end = max(local_start, end - sentence_start)
    window = _word_window(
        sentence,
        local_start,
        local_end,
        before_words=3,
        after_words=0 if boundary_ok else 2,
    )
    if _word_count(window) >= 2 and len(window) >= 8:
        return window, "expanded_word_window"

    fallback = _trim_phrase(sentence)
    if _word_count(fallback) >= 2:
        return _limit_text(fallback, 140), "expanded_sentence"
    return "", "subword_fragment"


def phrase_level_spans(
    raw_spans: list[Any],
    text: str,
    *,
    limit: int = 6,
) -> dict[str, Any]:
    value = str(text or "")
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    expanded: list[dict[str, str]] = []
    for raw in raw_spans or []:
        raw_text = _text(raw)
        phrase, reason = _phrase_from_raw_span(raw_text, value)
        if phrase:
            accepted.append(phrase)
            if phrase != raw_text:
                expanded.append({"raw_span": raw_text, "phrase_span": phrase, "reason": reason})
        else:
            rejected.append({"raw_span": raw_text, "reason": reason})
    return {
        "predictable_spans": unique_text(accepted, limit=limit),
        "rejected_predictable_spans": rejected,
        "expanded_predictable_spans": expanded,
    }


def span_rows(spans: list[str]) -> list[dict[str, str]]:
    return [
        {"id": f"ps{index:03d}", "text": span}
        for index, span in enumerate(spans or [], start=1)
    ]


def problem_tokens_from_spans(
    spans: list[str],
    *,
    extra_tokens: list[Any] | None = None,
    limit: int = 10,
) -> list[str]:
    tokens: list[Any] = []
    tokens.extend(extra_tokens or [])
    for span in spans:
        tokens.extend(str(span or "").replace(",", " ").replace(";", " ").replace(":", " ").split())
    return unique_text(tokens, limit=limit)


def brief_matches_group(brief: dict[str, Any], group: Any) -> bool:
    group_sentence_ids = set(_group_sentence_ids(group))
    group_paragraph_ids = set(_group_paragraph_ids(group))
    brief_sentence_ids = set(_brief_sentence_ids(brief))
    brief_paragraph_ids = set(_brief_paragraph_ids(brief))
    if group_sentence_ids and brief_sentence_ids and group_sentence_ids.intersection(brief_sentence_ids):
        return True
    if group_paragraph_ids and brief_paragraph_ids and group_paragraph_ids.intersection(brief_paragraph_ids):
        return True
    source_text = _group_source_text(group)
    target_sentence = _text(brief.get("target_sentence"))
    if target_sentence and target_sentence in source_text:
        return True
    return any(_text(span) and _text(span) in source_text for span in brief.get("predictable_token_spans") or [])


def topk_repair_contract_for_group(
    *,
    group: Any,
    replacement_text: str = "",
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    max_spans: int = 6,
) -> dict[str, Any]:
    spans: list[Any] = []
    problem_tokens: list[Any] = []
    source_sentences: list[Any] = []
    span_source = "scanner_exact"
    available_text = replacement_text or _group_source_text(group)
    for brief in predictability_briefs or []:
        if not isinstance(brief, dict) or not brief_matches_group(brief, group):
            continue
        spans.extend(brief.get("predictable_token_spans") or [])
        problem_tokens.extend(brief.get("problem_tokens") or [])
        sentence = _text(brief.get("target_sentence"))
        if sentence:
            source_sentences.append(_limit_text(sentence, 180))

    exact_spans = unique_text(spans)
    source_exact_spans = [span for span in exact_spans if str(span or "") and str(span) in available_text]
    phrase_payload = phrase_level_spans(source_exact_spans, available_text, limit=max_spans)
    predictable_spans = phrase_payload["predictable_spans"]
    if not predictable_spans:
        span_source = "structural_fallback"
        predictable_spans = structural_fallback_spans(available_text, limit=3)

    preferred_words = _preferred_words(group) or max(1, len(available_text.split()))
    max_changed_spans = max(1, min(3, len(predictable_spans) or 1))
    high_quality_spans = [
        span for span in predictable_spans
        if _word_count(span) >= 3 and len(span) >= 12
    ]
    required_modified_spans = min(2, len(high_quality_spans)) if span_source == "scanner_exact" else 1
    if span_source == "scanner_exact" and preferred_words < 60:
        required_modified_spans = min(1, len(high_quality_spans))
    required_modified_spans = max(1, required_modified_spans) if predictable_spans else 1
    return {
        "span_source": span_source,
        "raw_predictable_spans": source_exact_spans[:max_spans],
        "predictable_spans": predictable_spans,
        "predictable_spans_in_source": predictable_spans,
        "predictable_span_rows": span_rows(predictable_spans),
        "rejected_predictable_spans": phrase_payload["rejected_predictable_spans"],
        "expanded_predictable_spans": phrase_payload["expanded_predictable_spans"],
        "required_modified_spans": required_modified_spans,
        "source_sentences": unique_text(source_sentences, limit=3),
        "problem_tokens": problem_tokens_from_spans(predictable_spans, extra_tokens=problem_tokens),
        "avoid_phrases": predictable_spans[:max_spans],
        "locality_limits": {
            "max_changed_spans": max_changed_spans,
            "max_changed_words": max(10, min(24, round(preferred_words * 0.28))),
            "max_sentence_changes": min(2, max_changed_spans),
        },
        "allowed_operations": [
            "CLAUSE_ROUTE_CHANGE",
            "DELETE_EMPTY_PHRASE",
            "LIST_BREAK",
            "CONCRETE_SOURCE_WORDING",
            "TOPK_SPAN_REPATH",
        ],
    }


def target_action_contract(target: dict[str, Any], *, compact: bool = False) -> dict[str, Any]:
    payload = {
        "target_id": target.get("target_id"),
        "scope_level": target.get("scope_level"),
        "risk_level": target.get("risk_level"),
        "recommended_operation": target.get("recommended_operation"),
        "dominant_drivers": list(target.get("dominant_drivers") or [])[: 2 if compact else 4],
        "required_movement": target.get("required_movement") or {},
    }
    if compact:
        return payload
    payload.update({
        "operation_candidates": list(target.get("operation_candidates") or [])[:4],
        "anchor_pressure": target.get("target_anchor_pressure"),
        "semantic_edit_cost": target.get("semantic_edit_cost"),
        "reduction_allowed": target.get("reduction_allowed"),
        "reconstruction_allowed": target.get("reconstruction_allowed"),
        "rewrite_constraints": target.get("rewrite_constraints") or {},
    })
    return payload


def group_action_contract(
    *,
    group: Any,
    replacement_text: str = "",
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    compact: bool = False,
) -> dict[str, Any]:
    targets = [
        target_action_contract(target, compact=compact)
        for target in _group_targets(group)
        if isinstance(target, dict)
    ]
    citation_pressure_zone = any(
        _number(target.get("target_anchor_pressure")) >= 0.5
        or "citation_preserving_window_repair" in {str(item) for item in target.get("operation_candidates") or []}
        for target in _group_targets(group)
        if isinstance(target, dict)
    )
    topk_contract = topk_repair_contract_for_group(
        group=group,
        replacement_text=replacement_text,
        predictability_briefs=predictability_briefs,
        max_spans=3 if compact else 6,
    )
    if compact:
        topk_contract = {
            "span_source": topk_contract.get("span_source"),
            "predictable_spans_in_source": list(topk_contract.get("predictable_spans_in_source") or [])[:3],
            "predictable_span_rows": list(topk_contract.get("predictable_span_rows") or [])[:3],
            "required_modified_spans": topk_contract.get("required_modified_spans"),
            "locality_limits": topk_contract.get("locality_limits"),
        }
    return {
        "operation": _group_operation(group),
        "targets": targets,
        "citation_pressure_zone": bool(citation_pressure_zone),
        "citation_zone_instruction": (
            "Near citations or evidence anchors, keep source-like wording and avoid smoother academic paraphrase."
            if citation_pressure_zone
            else ""
        ),
        "topk_repair_contract": topk_contract,
        "allowed_rewrite_moves": _allowed_moves_for_group(group),
    }


def profile_action_contracts(
    *,
    rewrite_target_profile: dict[str, Any] | None,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    max_contracts: int = 8,
    compact: bool = False,
) -> list[dict[str, Any]]:
    profile = rewrite_target_profile if isinstance(rewrite_target_profile, dict) else {}
    contracts: list[dict[str, Any]] = []
    for target in profile.get("targets") or []:
        if not isinstance(target, dict):
            continue
        group = {
            "operation": target.get("recommended_operation"),
            "unit_id": target.get("unit_id") or target.get("paragraph_id"),
            "paragraph_id": target.get("paragraph_id"),
            "sentence_ids": target.get("sentence_ids") or [],
            "source_text": target.get("source_text") or target.get("source_excerpt") or "",
            "word_count_guide": target.get("word_count_guide") or {},
            "targets": [target],
        }
        contracts.append({
            "target_id": target.get("target_id"),
            "unit_id": target.get("unit_id"),
            "paragraph_id": target.get("paragraph_id"),
            "scope_level": target.get("scope_level"),
            "source_excerpt": _limit_text(group["source_text"], 220),
            "scanner_action_contract": group_action_contract(
                group=group,
                predictability_briefs=predictability_briefs,
                compact=compact,
            ),
        })
        if len(contracts) >= max(1, int(max_contracts or 1)):
            break
    return contracts


def _allowed_moves_for_group(group: Any) -> list[str]:
    operation = _group_operation(group)
    if operation == "unit_preserving_prune_bridge":
        return ["DELETE_EMPTY_PHRASE", "BRIDGE_NEARBY_MEANING", "CLAUSE_ROUTE_CHANGE"]
    if operation == "citation_preserving_window_repair":
        return ["CLAIM_FRAMING_REPAIR", "CLAUSE_ROUTE_CHANGE", "CONCRETE_SOURCE_WORDING"]
    if operation == "paragraph_preserving_broad_reconstruction":
        return ["BREAK_SURVEY_TEMPLATE", "BROAD_CLAIM_NARROWING", "CAUSE_EFFECT_OWNERSHIP", "TOPK_SPAN_REPATH"]
    return ["CLAUSE_ROUTE_CHANGE", "CONCRETE_SOURCE_WORDING", "TOPK_SPAN_REPATH"]


def _group_targets(group: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(group, dict):
        targets = group.get("targets")
    else:
        targets = getattr(group, "targets", ())
    return tuple(target for target in targets or () if isinstance(target, dict))


def _group_operation(group: Any) -> str:
    return _text(group.get("operation") if isinstance(group, dict) else getattr(group, "operation", ""))


def _group_source_text(group: Any) -> str:
    return _text(group.get("source_text") if isinstance(group, dict) else getattr(group, "source_text", ""))


def _group_sentence_ids(group: Any) -> list[str]:
    values: list[Any] = []
    if isinstance(group, dict):
        values.extend(group.get("sentence_ids") or [])
        targets = group.get("targets") or []
    else:
        values.extend(getattr(group, "sentence_ids", []) or [])
        targets = getattr(group, "targets", ()) or ()
    for target in targets:
        if isinstance(target, dict):
            values.extend(target.get("sentence_ids") or [])
    return unique_text(values)


def _group_paragraph_ids(group: Any) -> list[str]:
    values: list[Any] = []
    if isinstance(group, dict):
        values.extend([group.get("unit_id"), group.get("paragraph_id")])
        targets = group.get("targets") or []
    else:
        values.extend([getattr(group, "unit_id", None), getattr(group, "paragraph_id", None)])
        targets = getattr(group, "targets", ()) or ()
    for target in targets:
        if isinstance(target, dict):
            values.extend([target.get("paragraph_id"), target.get("parent_unit_id"), target.get("unit_id")])
    return unique_text(values)


def _brief_sentence_ids(brief: dict[str, Any]) -> list[str]:
    values = list(brief.get("sentence_ids") or [])
    values.extend([brief.get("sentence_id"), brief.get("target_sentence_id")])
    return unique_text(values)


def _brief_paragraph_ids(brief: dict[str, Any]) -> list[str]:
    values = list(brief.get("paragraph_ids") or [])
    values.extend([brief.get("paragraph_id"), brief.get("unit_id")])
    return unique_text(values)


def _preferred_words(group: Any) -> int:
    guide = group.get("word_count_guide") if isinstance(group, dict) else getattr(group, "word_count_guide", {})
    if not isinstance(guide, dict):
        return 0
    value = guide.get("preferred_words") or guide.get("source_words")
    return int(value) if isinstance(value, int) and value > 0 else 0


def _limit_text(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    clipped = value[:max(0, limit - 1)].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."
