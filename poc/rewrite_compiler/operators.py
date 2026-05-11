"""Deterministic compiler operators."""

from __future__ import annotations

import re
from typing import Any

from .signals import logical_paragraphs


OPERATOR_CATALOG = [
    {
        "operator": "REMOVE_GENERIC_CONNECTOR",
        "input_scope": "one_sentence",
        "allowed_change_ratio": 0.08,
        "must_preserve_anchors": True,
        "must_not_add_claims": True,
    },
    {
        "operator": "SHORTEN_EXPLANATORY_SENTENCE",
        "input_scope": "one_sentence",
        "allowed_change_ratio": 0.12,
        "must_preserve_anchors": True,
        "must_not_add_claims": True,
    },
    {
        "operator": "SPLIT_OVERLONG_SENTENCE",
        "input_scope": "one_sentence",
        "allowed_change_ratio": 0.12,
        "must_preserve_anchors": True,
        "must_not_add_claims": True,
    },
    {
        "operator": "REDUCE_SYMMETRIC_CADENCE",
        "input_scope": "one_sentence",
        "allowed_change_ratio": 0.12,
        "must_preserve_anchors": True,
        "must_not_add_claims": True,
    },
    {
        "operator": "COMPRESS_ABSTRACT_CLAIM",
        "input_scope": "one_sentence",
        "allowed_change_ratio": 0.14,
        "must_preserve_anchors": True,
        "must_not_add_claims": True,
    },
    {
        "operator": "BREAK_TEMPLATE_TRANSITION",
        "input_scope": "one_sentence_or_paragraph",
        "allowed_change_ratio": 0.12,
        "must_preserve_anchors": True,
        "must_not_add_claims": True,
    },
    {
        "operator": "REMOVE_LOW_VALUE_GENERIC_BLOCK",
        "input_scope": "one_paragraph",
        "allowed_change_ratio": 0.18,
        "must_preserve_anchors": True,
        "must_not_add_claims": True,
    },
]

CATALOG_BY_OPERATOR = {row["operator"]: row for row in OPERATOR_CATALOG}


def _clean_sentence(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    if value and value[0].islower():
        value = value[0].upper() + value[1:]
    return value


def _candidate_for_operator(sentence: str, operator: str, deps: Any) -> str:
    original = str(sentence or "").strip()
    if not original:
        return original
    if callable(getattr(deps, "is_canonical_fact_sentence", None)) and deps.is_canonical_fact_sentence(original):
        return original
    candidate = original
    if operator == "REMOVE_GENERIC_CONNECTOR":
        candidate = re.sub(
            r"^(?:Furthermore|Moreover|Additionally|In conclusion|Overall|Therefore|However|"
            r"At the same time|In addition|Despite this),?\s+",
            "",
            candidate,
            count=1,
            flags=re.I,
        )
    elif operator == "SHORTEN_EXPLANATORY_SENTENCE":
        replacements = (
            (r"\bIt is important to note that\s+", ""),
            (r"\bThis means that\s+", "That means "),
            (r"\bThis shows that\s+", "That shows "),
            (r"\bin many different ways\b", ""),
            (r"\bfor many decades\b", ""),
            (r"\bhas a (?:strong|significant|major) influence on\b", "influences"),
        )
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, candidate, count=1, flags=re.I)
            if updated != candidate:
                candidate = updated
                break
    elif operator == "SPLIT_OVERLONG_SENTENCE":
        if deps.text_word_count(candidate) >= 18:
            for splitter, lead in (
                (r"\s+because\s+", "Because "),
                (r"\s+but\s+", "But "),
                (r"\s+while\s+", "While "),
                (r"\s+which\s+", "That "),
            ):
                match = re.search(splitter, candidate, flags=re.I)
                if not match:
                    continue
                left = candidate[:match.start()].strip(" ,;")
                right = candidate[match.end():].strip(" ,;")
                if deps.text_word_count(left) >= 6 and deps.text_word_count(right) >= 5:
                    candidate = f"{left}. {lead}{right[0].lower() + right[1:] if len(right) > 1 else right}"
                    break
    elif operator == "REDUCE_SYMMETRIC_CADENCE":
        if "," in candidate:
            first, rest = candidate.split(",", 1)
            if 3 <= deps.text_word_count(first) <= 9 and deps.text_word_count(rest) >= 6:
                candidate = f"{rest.strip()} {first.strip().lower()}."
    elif operator == "COMPRESS_ABSTRACT_CLAIM":
        replacements = (
            (r"\bis often described as one of the most influential\b", "has wide influence"),
            (r"\bone of the biggest strengths of\b", "one strength of"),
            (r"\bAnother important feature of\b", "Another part of"),
            (r"\bplays? (?:a|an) (?:important|significant|major|crucial) role in\b", "matters in"),
            (r"\bhas shaped\b", "shapes"),
            (r"\bis known for\b", "is marked by"),
        )
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, candidate, count=1, flags=re.I)
            if updated != candidate:
                candidate = updated
                break
    elif operator == "BREAK_TEMPLATE_TRANSITION":
        replacements = (
            (r"^In conclusion,\s*", ""),
            (r"^Overall,\s*", ""),
            (r"^At the same time,\s*", "Still, "),
            (r"^Another important feature of\s+", "Another part of "),
            (r"^One of the biggest strengths of\s+", "One strength of "),
        )
        for pattern, replacement in replacements:
            updated = re.sub(pattern, replacement, candidate, count=1, flags=re.I)
            if updated != candidate:
                candidate = updated
                break
    candidate = _clean_sentence(candidate)
    return candidate if candidate != original else original


def _sentence_operators(row: dict) -> list[str]:
    reasons = set(row.get("reasons") or [])
    classification = str(row.get("classification") or "")
    operators: list[str] = []
    if "generic_connector" in reasons:
        operators.append("REMOVE_GENERIC_CONNECTOR")
    if "template_geometry" in reasons or classification == "template_geometry_target":
        operators.append("BREAK_TEMPLATE_TRANSITION")
    if "generic_expansion" in reasons or classification == "generic_expansion_target":
        operators.append("COMPRESS_ABSTRACT_CLAIM")
        operators.append("SHORTEN_EXPLANATORY_SENTENCE")
    if classification != "canonical_fact_preserve":
        operators.append("SPLIT_OVERLONG_SENTENCE")
        operators.append("REDUCE_SYMMETRIC_CADENCE")
    unique: list[str] = []
    for operator in operators:
        if operator not in unique and operator in CATALOG_BY_OPERATOR:
            unique.append(operator)
    return unique[:4]


def _splice_paragraph(text: str, paragraph_index: int, replacement: str) -> str:
    paragraphs = logical_paragraphs(text)
    if paragraph_index < 0 or paragraph_index >= len(paragraphs):
        return text
    if replacement.strip():
        paragraphs[paragraph_index] = replacement.strip()
    else:
        paragraphs.pop(paragraph_index)
    return "\n\n".join(paragraphs).strip()


def generate_candidates(text: str, plan: dict, deps: Any, *, limit: int = 14) -> list[dict]:
    candidates: list[dict] = []
    seen = {str(text or "").strip()}
    for row in plan.get("selected_windows") or []:
        if not isinstance(row, dict) or not row.get("rewrite_allowed"):
            continue
        sentence = str(row.get("sentence") or "")
        sentence_index = int(row.get("sentence_index") if isinstance(row.get("sentence_index"), int) else -1)
        if sentence_index < 0:
            continue
        for operator in _sentence_operators(row):
            replacement = _candidate_for_operator(sentence, operator, deps)
            if not replacement or replacement == sentence:
                continue
            candidate_text = deps.splice_sentence(text, sentence_index, replacement)
            normalized = candidate_text.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append({
                "strategy": f"compiler_{operator.lower()}_s{sentence_index + 1}",
                "text": candidate_text,
                "operator": operator,
                "operator_contract": CATALOG_BY_OPERATOR[operator],
                "scope": "sentence",
                "sentence_index": sentence_index,
                "classification": row.get("classification"),
                "risk_reasons": row.get("reasons") or [],
                "original_sentence": sentence,
                "replacement_sentence": replacement,
            })
            if len(candidates) >= limit:
                return candidates
    for block in plan.get("block_risk_map") or []:
        if len(candidates) >= limit:
            break
        if not isinstance(block, dict) or not block.get("remove_candidate") or block.get("protected"):
            continue
        block_index = int(block.get("block_index") if isinstance(block.get("block_index"), int) else -1)
        if block_index < 0:
            continue
        candidate_text = _splice_paragraph(text, block_index, "")
        normalized = candidate_text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append({
            "strategy": f"compiler_remove_low_value_block_b{block_index + 1}",
            "text": candidate_text,
            "operator": "REMOVE_LOW_VALUE_GENERIC_BLOCK",
            "operator_contract": CATALOG_BY_OPERATOR["REMOVE_LOW_VALUE_GENERIC_BLOCK"],
            "scope": "paragraph",
            "block_index": block_index,
            "classification": block.get("role"),
            "risk_reasons": ["low_value_generic_block"],
        })
    return candidates
