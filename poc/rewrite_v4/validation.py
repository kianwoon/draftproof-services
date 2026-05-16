"""Validation helpers for V4 normalizer and generator boundaries."""

from __future__ import annotations

import json
import os
from typing import Any

from rewrite_v3.text_integrity import minimal_replacement_text_integrity, raw_completion_integrity
from rewrite.guards import (
    _entity_present,
    _extract_named_entities,
    _extract_numbers,
    _keyword_cosine,
    check_semantic_drift,
)

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


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


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
    tutor_diagnosis: Any = "",
    student_explanation: Any = "",
    source_examples: Any = None,
    repair_assignment: Any = "",
    coverage_hint: Any = "paragraph",
    parse_diagnostics: dict[str, Any] | None = None,
) -> RepairBrief:
    rejected: list[dict[str, Any]] = []

    def clean_text(value: Any, *, field: str, max_chars: int) -> str:
        text = " ".join(str(value or "").split())
        if not text:
            return ""
        lower = text.casefold()
        detector_terms = sorted(term for term in RAW_DETECTOR_TERMS if term in lower.split())
        if detector_terms:
            rejected.append({"field": field, "text": text, "reason": "raw_detector_language", "terms": detector_terms})
            return ""
        return text[:max_chars]

    def clean_list(value: Any, *, max_items: int, field: str, reject_task_terms: bool = False) -> tuple[str, ...]:
        rows: list[str] = []
        items = value if isinstance(value, list) else []
        for index, item in enumerate(items, start=1):
            text = clean_text(item, field=field, max_chars=260)
            if not text:
                continue
            lower = text.casefold()
            blocked = sorted(term for term in DISALLOWED_TASK_TERMS if term in lower)
            if reject_task_terms and blocked:
                rejected.append({"field": field, "index": index, "text": text, "reason": "unsupported_or_fake_human_task", "terms": blocked})
                continue
            rows.append(text)
            if len(rows) >= max_items:
                break
        return tuple(rows)

    def clean_examples(value: Any) -> tuple[dict[str, Any], ...]:
        examples: list[dict[str, Any]] = []
        items = value if isinstance(value, list) else []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                rejected.append({"field": "source_examples", "index": index, "reason": "example_not_object"})
                continue
            example = {
                "excerpt": clean_text(item.get("excerpt"), field="source_examples.excerpt", max_chars=220),
                "issue": clean_text(item.get("issue"), field="source_examples.issue", max_chars=220),
                "repair_direction": clean_text(item.get("repair_direction"), field="source_examples.repair_direction", max_chars=260),
            }
            if not example["excerpt"] or not example["issue"] or not example["repair_direction"]:
                rejected.append({"field": "source_examples", "index": index, "reason": "example_incomplete"})
                continue
            examples.append(example)
            if len(examples) >= 4:
                break
        return tuple(examples)

    safe_tasks = clean_list(repair_tasks, max_items=5, field="repair_tasks", reject_task_terms=True)
    safe_constraints = clean_list(constraints, max_items=8, field="constraints")
    safe_avoid = clean_list(avoid, max_items=10, field="avoid")
    role = " ".join(str(paragraph_role or "body paragraph").split()) or "body paragraph"
    return RepairBrief(
        normalizer=normalizer,
        repair_mode="controlled_enrichment_repair" if "enrichment" in str(normalizer).casefold() else "source_preserving_repair",
        paragraph_role=role,
        tutor_diagnosis=clean_text(tutor_diagnosis, field="tutor_diagnosis", max_chars=500),
        student_explanation=clean_text(student_explanation, field="student_explanation", max_chars=600),
        source_examples=clean_examples(source_examples),
        repair_assignment=clean_text(repair_assignment, field="repair_assignment", max_chars=420),
        coverage_hint=clean_text(coverage_hint, field="coverage_hint", max_chars=80) or "paragraph",
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


def source_grounding_integrity(source_text: str, replacement_text: str, *, repair_mode: str = "source_preserving_repair") -> dict[str, Any]:
    source = str(source_text or "")
    replacement = str(replacement_text or "")
    failures: list[dict[str, Any]] = []
    mode = str(repair_mode or "source_preserving_repair")
    enriched = mode == "controlled_enrichment_repair"

    threshold_name = (
        "DRAFTPROOF_REWRITE_V4_ENRICHED_SOURCE_SIMILARITY_THRESHOLD"
        if enriched
        else "DRAFTPROOF_REWRITE_V4_SOURCE_SIMILARITY_THRESHOLD"
    )
    threshold = _float_env(threshold_name, 0.62 if enriched else 0.75, minimum=0.5, maximum=0.95)
    drift = check_semantic_drift(source, replacement, threshold=threshold)
    if not drift.accepted:
        failures.extend(
            {"reason": "semantic_drift", "detail": str(reason)}
            for reason in drift.reasons[:6]
        )

    source_entities = _extract_named_entities(source)
    replacement_entities = _extract_named_entities(replacement)
    unsupported_entities = sorted(
        entity
        for entity in replacement_entities
        if not _entity_present(entity, source)
    )
    if unsupported_entities:
        failures.append({
            "reason": "unsupported_named_entities",
            "values": unsupported_entities[:8],
        })

    source_numbers = _extract_numbers(source)
    replacement_numbers = _extract_numbers(replacement)
    unsupported_numbers = sorted(replacement_numbers - source_numbers)
    if unsupported_numbers:
        failures.append({
            "reason": "unsupported_numbers",
            "values": unsupported_numbers[:8],
        })

    claim_coverage = sentence_claim_coverage(source, replacement, repair_mode=mode)
    if not claim_coverage.get("passed"):
        failures.append({
            "reason": "source_sentence_claim_loss",
            "values": claim_coverage.get("failures", [])[:4],
        })

    return {
        "passed": not failures,
        "repair_mode": mode,
        "external_review_required": enriched,
        "failures": failures,
        "semantic_similarity": round(float(drift.similarity), 3),
        "semantic_similarity_threshold": threshold,
        "sentence_claim_coverage": claim_coverage,
    }


def sentence_claim_coverage(source_text: str, replacement_text: str, *, repair_mode: str = "source_preserving_repair") -> dict[str, Any]:
    enriched = str(repair_mode or "") == "controlled_enrichment_repair"
    threshold = _float_env(
        "DRAFTPROOF_REWRITE_V4_ENRICHED_SENTENCE_CLAIM_THRESHOLD" if enriched else "DRAFTPROOF_REWRITE_V4_SENTENCE_CLAIM_THRESHOLD",
        0.22 if enriched else 0.3,
        minimum=0.1,
        maximum=0.8,
    )
    source_sentences = _sentences(source_text)
    replacement_sentences = _sentences(replacement_text)
    failures: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    if not source_sentences or not replacement_sentences:
        return {"passed": True, "threshold": threshold, "rows": coverage_rows, "failures": failures}
    for index, sentence in enumerate(source_sentences, start=1):
        if len(sentence.split()) < 7:
            continue
        best = max((_keyword_cosine(sentence, candidate) for candidate in replacement_sentences), default=0.0)
        row = {
            "source_sentence_index": index,
            "best_similarity": round(float(best), 3),
            "source_excerpt": sentence[:180],
        }
        coverage_rows.append(row)
        if best < threshold:
            failures.append(row)
    return {
        "passed": not failures,
        "threshold": threshold,
        "rows": coverage_rows,
        "failures": failures,
    }


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for char in str(text or ""):
        current.append(char)
        if char in ".?!":
            sentence = "".join(current).strip()
            current = []
            if sentence:
                sentences.append(sentence)
    remainder = "".join(current).strip()
    if remainder:
        sentences.append(remainder)
    return sentences
