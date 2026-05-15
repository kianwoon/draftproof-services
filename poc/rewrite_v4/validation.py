"""Validation helpers for V4 normalizer and generator boundaries."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.text_integrity import minimal_replacement_text_integrity, raw_completion_integrity

from .models import CandidateVariant, RepairBrief


DISALLOWED_TASK_TERMS = {
    "add an example",
    "add a concrete example",
    "add a scenario",
    "personal anecdote",
    "personal experience",
    "fake experience",
    "make it casual",
    "slang",
    "random error",
    "typo",
}

RAW_DETECTOR_TERMS = {
    "ai",
    "detector",
    "scanner",
    "likelihood",
    "score",
    "risk",
    "bypass",
    "evade",
}


def parse_json_object(raw: str, *, required_keys: set[str] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    text = str(raw or "").strip()
    integrity = raw_completion_integrity(text)
    if not integrity.get("passed"):
        return None, {"status": "completion_corrupted", "raw_integrity": integrity, "raw_length": len(text)}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, {"status": "json_parse_failed", "error": str(exc), "raw_integrity": integrity, "raw_length": len(text)}
    if not isinstance(payload, dict):
        return None, {"status": "schema_failed", "reason": "root_not_object", "raw_integrity": integrity, "raw_length": len(text)}
    if required_keys is not None and set(payload.keys()) != required_keys:
        return None, {
            "status": "schema_failed",
            "reason": "root_keys_mismatch",
            "keys": sorted(payload.keys()),
            "expected_keys": sorted(required_keys),
            "raw_integrity": integrity,
            "raw_length": len(text),
        }
    return payload, {"status": "ok", "raw_integrity": integrity, "raw_length": len(text)}


def sanitize_repair_brief(
    *,
    normalizer: str,
    paragraph_role: Any,
    repair_tasks: Any,
    constraints: Any,
    avoid: Any,
    parse_diagnostics: dict[str, Any] | None = None,
) -> RepairBrief:
    rejected: list[dict[str, Any]] = []

    def clean_list(value: Any, *, max_items: int, field: str, reject_task_terms: bool = False) -> tuple[str, ...]:
        rows: list[str] = []
        items = value if isinstance(value, list) else []
        for index, item in enumerate(items, start=1):
            text = " ".join(str(item or "").split())
            if not text:
                continue
            lower = text.casefold()
            detector_terms = sorted(term for term in RAW_DETECTOR_TERMS if term in lower.split())
            if detector_terms:
                rejected.append({"field": field, "index": index, "text": text, "reason": "raw_detector_language", "terms": detector_terms})
                continue
            blocked = sorted(term for term in DISALLOWED_TASK_TERMS if term in lower)
            if reject_task_terms and blocked:
                rejected.append({"field": field, "index": index, "text": text, "reason": "unsupported_or_fake_human_task", "terms": blocked})
                continue
            rows.append(text)
            if len(rows) >= max_items:
                break
        return tuple(rows)

    safe_tasks = clean_list(repair_tasks, max_items=5, field="repair_tasks", reject_task_terms=True)
    safe_constraints = clean_list(constraints, max_items=8, field="constraints")
    safe_avoid = clean_list(avoid, max_items=10, field="avoid")
    role = " ".join(str(paragraph_role or "body paragraph").split()) or "body paragraph"
    return RepairBrief(
        normalizer=normalizer,
        paragraph_role=role,
        repair_tasks=safe_tasks,
        constraints=safe_constraints,
        avoid=safe_avoid,
        rejected_tasks=tuple(rejected),
        parse_diagnostics=parse_diagnostics or {},
    )


def parse_generator_variants(raw: str, *, min_words: int, max_words: int) -> tuple[list[CandidateVariant], dict[str, Any]]:
    payload, diagnostics = parse_json_object(raw, required_keys={"variants"})
    if payload is None:
        return [], diagnostics
    rows = payload.get("variants")
    if not isinstance(rows, list):
        return [], {**diagnostics, "status": "schema_failed", "reason": "variants_not_array"}

    variants: list[CandidateVariant] = []
    rejected: list[dict[str, Any]] = []
    allowed_keys = {"variant_id", "text"}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            rejected.append({"index": index, "reason": "variant_not_object"})
            continue
        keys = set(row.keys())
        if keys != allowed_keys:
            rejected.append({"index": index, "reason": "variant_keys_mismatch", "keys": sorted(keys)})
            continue
        variant_id = str(row.get("variant_id") or "").strip()
        text = str(row.get("text") or "").strip()
        if not variant_id or not text:
            rejected.append({"index": index, "reason": "empty_variant"})
            continue
        text_integrity = minimal_replacement_text_integrity(text)
        if not text_integrity.get("passed"):
            rejected.append({"index": index, "variant_id": variant_id, "reason": "text_integrity_failed", "text_integrity": text_integrity})
            continue
        if "\n\n" in text:
            rejected.append({"index": index, "variant_id": variant_id, "reason": "paragraph_split"})
            continue
        word_count = len(text.split())
        if word_count < min_words or word_count > max_words:
            rejected.append({
                "index": index,
                "variant_id": variant_id,
                "reason": "word_count_contract_failed",
                "word_count": word_count,
                "min_words": min_words,
                "max_words": max_words,
            })
            continue
        variants.append(CandidateVariant(variant_id=variant_id, text=text, word_count=word_count))

    status = "ok" if variants else "schema_failed"
    return variants, {
        **diagnostics,
        "status": status,
        "variant_count": len(variants),
        "rejected": rejected,
    }
