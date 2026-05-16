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
    mitigation_strategy: Any = None,
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

    def clean_strategy(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        allowed_scalars = {
            "scope",
            "strategy_id",
            "primary_problem",
            "rewrite_depth",
            "candidate_count_hint",
        }
        allowed_lists = {
            "target_zones",
            "current_route",
            "better_route",
            "route_moves",
            "route_must_preserve",
            "route_forbidden",
            "required_moves",
            "forbidden_moves",
            "must_preserve_claims",
            "success_checks",
        }
        strategy: dict[str, Any] = {}
        for key in allowed_scalars:
            if key not in value:
                continue
            text = clean_text(value.get(key), field=f"mitigation_strategy.{key}", max_chars=160)
            if text:
                strategy[key] = text
        for key in allowed_lists:
            if key not in value:
                continue
            rows = clean_list(value.get(key), max_items=8, field=f"mitigation_strategy.{key}")
            if rows:
                strategy[key] = list(rows)
        steps: list[dict[str, Any]] = []
        raw_steps = value.get("strategy_steps")
        if isinstance(raw_steps, list):
            for index, item in enumerate(raw_steps, start=1):
                if not isinstance(item, dict):
                    rejected.append({"field": "mitigation_strategy.strategy_steps", "index": index, "reason": "step_not_object"})
                    continue
                step = {
                    "op": clean_text(item.get("op"), field="mitigation_strategy.strategy_steps.op", max_chars=80),
                    "target": clean_text(item.get("target"), field="mitigation_strategy.strategy_steps.target", max_chars=120),
                    "instruction": clean_text(item.get("instruction"), field="mitigation_strategy.strategy_steps.instruction", max_chars=240),
                    "must_preserve": list(clean_list(item.get("must_preserve"), max_items=4, field="mitigation_strategy.strategy_steps.must_preserve")),
                    "avoid": list(clean_list(item.get("avoid"), max_items=4, field="mitigation_strategy.strategy_steps.avoid")),
                }
                if not step["op"] or not step["target"] or not step["instruction"]:
                    rejected.append({"field": "mitigation_strategy.strategy_steps", "index": index, "reason": "step_incomplete"})
                    continue
                steps.append(step)
                if len(steps) >= 5:
                    break
        if steps:
            strategy["strategy_steps"] = steps
        return strategy

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
        mitigation_strategy=clean_strategy(mitigation_strategy),
        repair_tasks=safe_tasks,
        constraints=safe_constraints,
        avoid=safe_avoid,
        rejected_tasks=tuple(rejected),
        parse_diagnostics=parse_diagnostics or {},
    )


def parse_generator_variants(
    raw: str,
    *,
    min_words: int,
    max_words: int,
    source_text: str = "",
) -> tuple[list[CandidateVariant], dict[str, Any]]:
    payload, diagnostics = parse_json_object(raw, required_keys={"variants"})
    if payload is None:
        if diagnostics.get("status") != "json_parse_failed":
            return [], diagnostics
        payload = _recover_variants_payload_from_schema_lines(raw)
        if payload is None:
            return [], diagnostics
        diagnostics = {
            **diagnostics,
            "status": "ok_after_local_json_repair",
            "first_parse_status": "json_parse_failed",
        }
    rows = payload.get("variants")
    if not isinstance(rows, list):
        return [], {**diagnostics, "status": "schema_failed", "reason": "variants_not_array"}

    variants: list[CandidateVariant] = []
    rejected: list[dict[str, Any]] = []
    allowed_keys = {"variant_id", "text"}
    source_structure = _block_structure(source_text)
    allow_multi_block = source_structure["block_count"] > 1
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
        text_structure = _block_structure(text)
        if text_structure["block_count"] > 1:
            if not allow_multi_block:
                rejected.append({"index": index, "variant_id": variant_id, "reason": "paragraph_split"})
                continue
            if text_structure["block_count"] != source_structure["block_count"]:
                rejected.append({
                    "index": index,
                    "variant_id": variant_id,
                    "reason": "block_count_contract_failed",
                    "block_count": text_structure["block_count"],
                    "source_block_count": source_structure["block_count"],
                })
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
        "source_structure": source_structure,
    }


def _block_structure(text: str) -> dict[str, Any]:
    lines = str(text or "").splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    blank_line_boundary_count = 0
    previous_nonempty = False
    for line in lines:
        if line.strip():
            current.append(line)
            previous_nonempty = True
            continue
        if current:
            blocks.append(current)
            current = []
        if previous_nonempty:
            blank_line_boundary_count += 1
        previous_nonempty = False
    if current:
        blocks.append(current)
    nonempty_line_count = sum(1 for line in lines if line.strip())
    return {
        "block_count": max(1, len(blocks)),
        "blank_line_boundary_count": blank_line_boundary_count,
        "nonempty_line_count": max(1, nonempty_line_count),
    }


def _recover_variants_payload_from_schema_lines(raw: str) -> dict[str, Any] | None:
    variants: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in str(raw or "").splitlines():
        stripped = line.strip()
        if stripped.startswith('"variant_id"'):
            value = _decode_schema_string_value(stripped)
            if value:
                current["variant_id"] = value
        elif stripped.startswith('"text"'):
            value = _decode_schema_string_value(stripped)
            if value:
                current["text"] = value
        if set(current.keys()) == {"variant_id", "text"}:
            variants.append(current)
            current = {}
    return {"variants": variants} if variants else None


def _decode_schema_string_value(line: str) -> str:
    colon = line.find(":")
    if colon < 0:
        return ""
    value = line[colon + 1:].strip()
    if value.endswith(","):
        value = value[:-1].rstrip()
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return ""
    inner = value[1:-1]
    escaped: list[str] = []
    slash_count = 0
    for char in inner:
        if char == "\\":
            escaped.append(char)
            slash_count += 1
            continue
        if char == '"' and slash_count % 2 == 0:
            escaped.append('\\"')
        else:
            escaped.append(char)
        slash_count = 0
    try:
        decoded = json.loads('"' + "".join(escaped) + '"')
    except json.JSONDecodeError:
        return ""
    return str(decoded)


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


def strategy_compliance_integrity(replacement_text: str, strategy: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    redundancy = repeated_sentence_claim_integrity(replacement_text)
    if not redundancy.get("passed"):
        failures.extend(redundancy.get("failures") or [])
    if not isinstance(strategy, dict) or not strategy:
        return {"passed": not failures, "failures": failures, "redundancy": redundancy}
    candidate = _strategy_phrase_normalize(replacement_text)

    forbidden_routes: list[str] = []
    for item in strategy.get("forbidden_moves") or []:
        forbidden_routes.extend(_forbidden_route_markers(str(item or "")))
    for item in strategy.get("route_forbidden") or []:
        forbidden_routes.extend(_forbidden_route_markers(str(item or "")))
    for step in strategy.get("strategy_steps") or []:
        if not isinstance(step, dict):
            continue
        for item in step.get("avoid") or []:
            forbidden_routes.extend(_forbidden_route_markers(str(item or "")))

    seen: set[str] = set()
    for marker in forbidden_routes:
        normalized = _strategy_phrase_normalize(marker)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in candidate:
            failures.append({
                "reason": "strategy_forbidden_route_present",
                "forbidden_route": marker,
            })

    for step in strategy.get("strategy_steps") or []:
        if not isinstance(step, dict):
            continue
        op = str(step.get("op") or "")
        must_preserve = [str(item or "").strip() for item in step.get("must_preserve") or [] if str(item or "").strip()]
        if op == "reaction_reason_link":
            missing = [
                item for item in must_preserve
                if not _loose_phrase_present(candidate, item)
            ]
            missing_limit = max(1, len(must_preserve) // 2)
            if len(missing) > missing_limit:
                failures.append({
                    "reason": "strategy_reaction_reason_claim_missing",
                    "missing": missing,
                })
    route_missing = [
        item for item in strategy.get("route_must_preserve") or []
        if str(item or "").strip() and not _loose_phrase_present(candidate, str(item))
    ]
    if len(route_missing) > max(1, len(strategy.get("route_must_preserve") or []) // 2):
        failures.append({
            "reason": "route_preserve_claim_missing",
            "missing": route_missing[:4],
        })

    return {
        "passed": not failures,
        "failures": failures,
        "redundancy": redundancy,
    }


def repeated_sentence_claim_integrity(replacement_text: str) -> dict[str, Any]:
    sentences = [
        sentence for sentence in _sentences(replacement_text)
        if len(sentence.split()) >= _float_env("DRAFTPROOF_REWRITE_V4_REDUNDANT_SENTENCE_MIN_WORDS", 9, minimum=5, maximum=30)
    ]
    threshold = _float_env("DRAFTPROOF_REWRITE_V4_REDUNDANT_SENTENCE_THRESHOLD", 0.68, minimum=0.55, maximum=0.95)
    overlap_threshold = _float_env("DRAFTPROOF_REWRITE_V4_REDUNDANT_SENTENCE_OVERLAP_THRESHOLD", 0.55, minimum=0.35, maximum=0.9)
    fallback_similarity = _float_env("DRAFTPROOF_REWRITE_V4_REDUNDANT_SENTENCE_FALLBACK_SIMILARITY", 0.5, minimum=0.35, maximum=0.9)
    failures: list[dict[str, Any]] = []
    for left_index, left in enumerate(sentences):
        left_terms = _claim_terms(left)
        if len(left_terms) < 4:
            continue
        for right_index, right in enumerate(sentences[left_index + 1:], start=left_index + 2):
            right_terms = _claim_terms(right)
            if len(right_terms) < 4:
                continue
            shared = left_terms & right_terms
            if len(shared) < max(3, min(len(left_terms), len(right_terms)) // 3):
                continue
            similarity = float(_keyword_cosine(left, right))
            shared_ratio = len(shared) / max(1, min(len(left_terms), len(right_terms)))
            if similarity >= threshold or (similarity >= fallback_similarity and shared_ratio >= overlap_threshold):
                failures.append({
                    "reason": "redundant_sentence_claim",
                    "left_sentence_index": left_index + 1,
                    "right_sentence_index": right_index,
                    "similarity": round(similarity, 3),
                    "shared_term_ratio": round(shared_ratio, 3),
                    "left_excerpt": left[:180],
                    "right_excerpt": right[:180],
                })
    return {
        "passed": not failures,
        "threshold": threshold,
        "failures": failures[:4],
    }


def _forbidden_route_markers(text: str) -> list[str]:
    source = _strategy_phrase_normalize(text)
    if not source:
        return []
    route_suffixes = (
        " route",
        " bridge",
        " opening",
        " frame",
    )
    for suffix in route_suffixes:
        if source.endswith(suffix):
            source = source[: -len(suffix)].strip()
            break
    if " as a standalone" in source:
        source = source.split(" as a standalone", 1)[0].strip()
    if " without " in source:
        source = source.split(" without ", 1)[0].strip()
    if " by itself" in source:
        source = source.split(" by itself", 1)[0].strip()
    if len(source.split()) < 2:
        return []
    return [source]


def _loose_phrase_present(candidate: str, phrase: str) -> bool:
    normalized = _strategy_phrase_normalize(phrase)
    if not normalized:
        return True
    if normalized in candidate:
        return True
    if _best_sentence_similarity(candidate, normalized) >= _float_env(
        "DRAFTPROOF_REWRITE_V4_PRESERVE_PHRASE_SEMANTIC_THRESHOLD",
        0.32,
        minimum=0.2,
        maximum=0.8,
    ):
        return True
    tokens = [token for token in normalized.split() if len(token) >= 4]
    if not tokens:
        return True
    hits = sum(1 for token in tokens if token in candidate)
    return hits >= max(1, len(tokens) - 1)


def _best_sentence_similarity(text: str, phrase: str) -> float:
    sentences = _sentences(text)
    if not sentences:
        return 0.0
    return max((float(_keyword_cosine(phrase, sentence)) for sentence in sentences), default=0.0)


def _strategy_phrase_normalize(value: Any) -> str:
    text = str(value or "").casefold()
    for char in "-_/":
        text = text.replace(char, " ")
    return " ".join(text.split())


def _claim_terms(text: str) -> set[str]:
    normalized = _strategy_phrase_normalize(text)
    return {
        token.strip(".,;:!?()[]{}\"'“”‘’")
        for token in normalized.split()
        if len(token.strip(".,;:!?()[]{}\"'“”‘’")) >= 4
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
