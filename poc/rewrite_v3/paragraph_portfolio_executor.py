"""Paragraph-portfolio executor for rewrite V3.

This module keeps broad paragraph reconstruction out of the main pipeline and
uses scanner-derived target groups as the execution contract. Prompt batches
are capped by rendered prompt size, not by source word count.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from .document_units import structural_shape_contract, structural_shape_failures, word_count
from .output_cleaning import clean_v3_candidate_output
from .prompt_templates.paragraph_portfolio import (
    PromptTemplatePayload,
    TEMPLATE_ID,
    build_paragraph_portfolio_ownership_prompt,
    build_paragraph_portfolio_planner_prompt,
    build_paragraph_portfolio_reconstruction_prompt,
    build_paragraph_portfolio_topk_prompt,
    fallback_paragraph_portfolio_plan,
    paragraph_portfolio_context,
    parse_paragraph_portfolio_plan,
    validate_paragraph_portfolio_plan,
)
from .prompt_contract import topk_repair_contract_for_group
from .target_executor import (
    TargetGroup,
    apply_target_replacements,
    batch_target_groups,
    target_execution_trace,
)


GatewayFactory = Callable[[int], Any]
TokenBudget = Callable[[int], int]


@dataclass(frozen=True)
class ParagraphPortfolioConfig:
    llm_planner_enabled: bool
    blind_topk_enabled: bool
    fallback_batch_size: int
    max_reconstruction_prompt_chars: int
    raw_preview_chars: int


def _bool_env(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _finish_reason(response: Any) -> str:
    return str(getattr(response, "finish_reason", "") or "").strip().lower()


def paragraph_portfolio_config(*, fallback_batch_size: int) -> ParagraphPortfolioConfig:
    explicit_batch_size = os.environ.get("DRAFTPROOF_REWRITE_V3_PORTFOLIO_BATCH_SIZE")
    if explicit_batch_size is not None:
        batch_size = _int_env("DRAFTPROOF_REWRITE_V3_PORTFOLIO_BATCH_SIZE", 1, low=1, high=3)
    else:
        batch_size = 1
    return ParagraphPortfolioConfig(
        llm_planner_enabled=_bool_env("DRAFTPROOF_REWRITE_V3_PORTFOLIO_LLM_PLANNER", False),
        blind_topk_enabled=_bool_env("DRAFTPROOF_REWRITE_V3_PORTFOLIO_BLIND_TOPK", False),
        fallback_batch_size=batch_size,
        max_reconstruction_prompt_chars=_int_env(
            "DRAFTPROOF_REWRITE_V3_PORTFOLIO_MAX_RECON_PROMPT_CHARS",
            9000,
            low=3500,
            high=18000,
        ),
        raw_preview_chars=_int_env(
            "DRAFTPROOF_REWRITE_V3_RAW_RESPONSE_PREVIEW_CHARS",
            1200,
            low=200,
            high=4000,
        ),
    )


def _strip_json_fences(raw: str) -> str:
    text = str(raw or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _preview(raw: str, limit: int) -> str:
    text = str(raw or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def parse_replacements_with_diagnostics(
    raw: str,
    *,
    preview_chars: int = 1200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = _strip_json_fences(raw)
    diagnostics: dict[str, Any] = {
        "raw_chars": len(str(raw or "")),
        "raw_preview": _preview(str(raw or ""), preview_chars),
        "parse_status": "ok",
        "top_level_keys": [],
        "replacement_count": 0,
    }
    if not text:
        diagnostics["parse_status"] = "empty_response"
        return [], diagnostics
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        diagnostics["parse_status"] = "invalid_json"
        diagnostics["json_error"] = str(exc)
        return [], diagnostics
    if not isinstance(payload, dict):
        diagnostics["parse_status"] = "invalid_json_root"
        return [], diagnostics
    diagnostics["top_level_keys"] = list(payload.keys())[:12]
    rows = payload.get("replacements")
    if not isinstance(rows, list):
        diagnostics["parse_status"] = "missing_replacements"
        return [], diagnostics
    replacements: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        if not isinstance(row, dict):
            skipped += 1
            continue
        group_id = str(row.get("group_id") or "").strip()
        replacement = str(row.get("replacement_text") or "").strip()
        if not group_id or not replacement:
            skipped += 1
            continue
        parsed_row: dict[str, Any] = {"group_id": group_id, "replacement_text": replacement}
        for key in (
            "changed_spans",
            "modified_span_ids",
            "predictable_spans_modified_count",
            "changed_word_estimate",
            "ownership_changes",
            "ownership_elements_supported",
            "new_claims_added",
            "hard_anchors_preserved",
        ):
            if key in row:
                parsed_row[key] = row.get(key)
        replacements.append(parsed_row)
    diagnostics["replacement_count"] = len(replacements)
    diagnostics["skipped_rows"] = skipped
    if not replacements:
        diagnostics["parse_status"] = "empty_replacements"
    return replacements, diagnostics


def validate_replacement_structure(
    *,
    target_groups: list[TargetGroup],
    replacements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups_by_id = {group.group_id: group for group in target_groups}
    valid: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for row in replacements:
        group_id = str(row.get("group_id") or "")
        replacement_text = str(row.get("replacement_text") or "")
        group = groups_by_id.get(group_id)
        if group is None:
            statuses.append({
                "group_id": group_id,
                "passed": False,
                "failures": ["unknown_group"],
            })
            continue
        failures = structural_shape_failures(group.source_text, replacement_text)
        statuses.append({
            "group_id": group_id,
            "passed": not failures,
            "failures": failures,
            "source_structure_contract": structural_shape_contract(group.source_text),
            "replacement_structure_contract": structural_shape_contract(replacement_text),
        })
        if not failures:
            valid.append(row)
    return valid, statuses


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def validate_topk_replacement_effect(
    *,
    target_groups: list[TargetGroup],
    current_replacements: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups_by_id = {group.group_id: group for group in target_groups}
    current_by_id = {
        str(row.get("group_id") or ""): str(row.get("replacement_text") or "")
        for row in current_replacements
        if isinstance(row, dict) and row.get("group_id")
    }
    valid: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for row in replacements:
        group_id = str(row.get("group_id") or "")
        replacement_text = str(row.get("replacement_text") or "")
        current_text = current_by_id.get(group_id, "")
        group = groups_by_id.get(group_id)
        failures: list[str] = []
        if group is None:
            failures.append("unknown_group")
        if not current_text:
            failures.append("missing_current_replacement")
        if current_text and replacement_text == current_text:
            failures.append("no_material_change")

        changed_rows = [item for item in row.get("changed_spans") or [] if isinstance(item, dict)]
        if not changed_rows:
            failures.append("missing_changed_spans")

        contract = (
            topk_repair_contract_for_group(
                group=group,
                replacement_text=current_text,
                predictability_briefs=predictability_briefs,
            )
            if group is not None
            else {}
        )
        span_rows = {
            str(item.get("id") or ""): str(item.get("text") or "")
            for item in contract.get("predictable_span_rows") or []
            if isinstance(item, dict) and item.get("id") and item.get("text")
        }
        actual_modified_ids: list[str] = []
        effect_failures: list[str] = []
        for changed in changed_rows:
            span_id = str(changed.get("span_id") or "")
            before = str(changed.get("before") or "").strip()
            after = str(changed.get("after") or "").strip()
            source_span = str(changed.get("source_span") or before).strip()
            expected_span = span_rows.get(span_id)
            if not span_id or span_id not in span_rows:
                effect_failures.append("unknown_span_id")
                continue
            if not before or not after or before == after:
                effect_failures.append("no_effect_span_patch")
                continue
            if current_text and before not in current_text:
                effect_failures.append("reported_before_not_in_current")
                continue
            if replacement_text and after not in replacement_text:
                effect_failures.append("reported_after_not_in_output")
                continue
            if expected_span and expected_span not in source_span and expected_span not in before:
                effect_failures.append("invalid_span_compliance")
                continue
            if span_id not in actual_modified_ids:
                actual_modified_ids.append(span_id)

        required_count = max(1, int(contract.get("required_modified_spans") or 1))
        modified_count = len(actual_modified_ids)
        if modified_count < required_count:
            failures.append("insufficient_span_movement")
        failures.extend(effect_failures)

        declared_count = _int_or_none(row.get("predictable_spans_modified_count"))
        if declared_count is not None and declared_count != modified_count:
            failures.append("self_report_mismatch")
        changed_word_estimate = _int_or_none(row.get("changed_word_estimate"))
        if changed_word_estimate == 0 and modified_count == 0:
            failures.append("zero_change_candidate")

        status = {
            "group_id": group_id,
            "passed": not failures,
            "failures": list(dict.fromkeys(failures)),
            "required_modified_spans": required_count,
            "declared_predictable_spans_modified_count": declared_count,
            "actual_predictable_spans_modified_count": modified_count,
            "actual_modified_span_ids": actual_modified_ids,
        }
        statuses.append(status)
        if status["passed"]:
            valid.append(row)
    return valid, statuses


def _ownership_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in row.get("ownership_changes") or []:
        if not isinstance(item, dict):
            continue
        before = str(item.get("before") or "").strip()
        after = str(item.get("after") or "").strip()
        if before and after and before != after:
            rows.append(item)
    return rows


def _ownership_elements(row: dict[str, Any]) -> list[str]:
    allowed = {"author_trace", "specific_context", "real_judgment"}
    elements: list[str] = []
    for item in row.get("ownership_elements_supported") or []:
        text = str(item or "").strip()
        if text in allowed and text not in elements:
            elements.append(text)
    return elements


def ownership_candidate_quality(
    *,
    group: TargetGroup,
    replacement: dict[str, Any],
) -> dict[str, Any]:
    rows = _ownership_rows(replacement)
    elements = _ownership_elements(replacement)
    source_words = max(1, word_count(group.source_text))
    replacement_words = max(1, word_count(str(replacement.get("replacement_text") or "")))
    ratio_delta = abs(replacement_words - source_words) / source_words
    preservation_score = max(0.0, 20.0 - ratio_delta * 80.0)
    ownership_score = min(10.0, len(rows) * 3.0 + len(elements) * 1.5)
    score = round(ownership_score * 4.0 + preservation_score, 3)
    return {
        "score": score,
        "ownership_score": round(ownership_score, 3),
        "ownership_change_count": len(rows),
        "ownership_elements_supported": elements,
        "source_words": source_words,
        "replacement_words": replacement_words,
        "preservation_score": round(preservation_score, 3),
        "operator_used": "CLAIM_OWNERSHIP_REPAIR",
    }


def validate_ownership_replacement_effect(
    *,
    target_groups: list[TargetGroup],
    replacements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups_by_id = {group.group_id: group for group in target_groups}
    valid: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for row in replacements:
        group_id = str(row.get("group_id") or "")
        replacement_text = str(row.get("replacement_text") or "")
        group = groups_by_id.get(group_id)
        failures: list[str] = []
        if group is None:
            failures.append("unknown_group")
        elif replacement_text.strip() == str(group.source_text or "").strip():
            failures.append("no_material_change")
        ownership_rows = _ownership_rows(row)
        ownership_elements = _ownership_elements(row)
        if not ownership_rows:
            failures.append("missing_ownership_changes")
        if not ownership_elements:
            failures.append("missing_ownership_elements")
        if group is not None and not failures:
            enriched = dict(row)
            enriched["candidate_quality"] = ownership_candidate_quality(group=group, replacement=row)
            valid.append(enriched)
        statuses.append({
            "group_id": group_id,
            "passed": not failures,
            "failures": list(dict.fromkeys(failures)),
            "ownership_change_count": len(ownership_rows),
            "ownership_elements_supported": ownership_elements,
        })
    return valid, statuses


def _context_for_batch(
    *,
    batch: list[TargetGroup],
    scan_contract: Any,
    content_mode: str,
    family: str,
) -> dict[str, Any]:
    return paragraph_portfolio_context(
        target_groups=batch,
        scan_contract=scan_contract,
        content_mode=content_mode,
        strategy_family=family,
    )


def build_reconstruction_batches_by_prompt(
    *,
    target_groups: list[TargetGroup],
    scan_contract: Any,
    content_mode: str,
    family: str,
    planner_output: dict[str, Any],
    max_prompt_chars: int,
    fallback_batch_size: int,
) -> list[list[TargetGroup]]:
    if not target_groups:
        return []
    batches: list[list[TargetGroup]] = []
    current: list[TargetGroup] = []
    for group in target_groups:
        trial = [*current, group]
        trial_context = _context_for_batch(
            batch=trial,
            scan_contract=scan_contract,
            content_mode=content_mode,
            family=family,
        )
        prompt_chars = len(
            _reconstruction_prompt_for_budget(
                trial_context,
                planner_output,
                max_prompt_chars=max_prompt_chars,
            )[0].prompt
        )
        too_many_groups = len(trial) > max(1, int(fallback_batch_size or 1))
        if current and (prompt_chars > max_prompt_chars or too_many_groups):
            batches.append(current)
            current = [group]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def _batch_word_budget(batch: list[TargetGroup], *, extra_words: int = 420) -> int:
    words = 0
    for group in batch:
        guide = group.word_count_guide if isinstance(group.word_count_guide, dict) else {}
        words += int(guide.get("preferred_words") or word_count(group.source_text))
    return max(220, words + extra_words)


def _auto_topk_repair_enabled(scan_contract: Any, target_groups: list[TargetGroup]) -> bool:
    if not _bool_env("DRAFTPROOF_REWRITE_V3_PORTFOLIO_AUTO_TOPK", True):
        return False
    summary = getattr(scan_contract, "target_driver_summary", {}) or {}
    if _number(summary.get("predictability_score")) >= 1.0:
        return True
    for group in target_groups:
        for target in getattr(group, "targets", ()) or ():
            if not isinstance(target, dict):
                continue
            for driver in target.get("dominant_drivers") or []:
                if (
                    isinstance(driver, dict)
                    and str(driver.get("key") or "") == "predictability_score"
                    and _number(driver.get("score")) >= 0.5
                ):
                    return True
    return False


def _reconstruction_prompt_for_budget(
    context: dict[str, Any],
    planner_output: dict[str, Any],
    *,
    max_prompt_chars: int,
) -> tuple[PromptTemplatePayload, bool]:
    template = build_paragraph_portfolio_reconstruction_prompt(context, planner_output)
    if len(template.prompt) <= max_prompt_chars:
        return template, False
    compact_template = build_paragraph_portfolio_reconstruction_prompt(
        context,
        planner_output,
        compact_for_budget=True,
    )
    return compact_template, True


def _call_reconstruction_batch(
    *,
    batch: list[TargetGroup],
    scan_contract: Any,
    content_mode: str,
    family: str,
    planner_output: dict[str, Any],
    gateway_factory: GatewayFactory,
    token_budget: TokenBudget,
    preview_chars: int,
    batch_index: int,
    max_prompt_chars: int,
    retry_of_batch: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], str | None]:
    batch_context = _context_for_batch(
        batch=batch,
        scan_contract=scan_contract,
        content_mode=content_mode,
        family=family,
    )
    reconstruction_template, budget_compacted = _reconstruction_prompt_for_budget(
        batch_context,
        planner_output,
        max_prompt_chars=max_prompt_chars,
    )
    reconstruction_error = None
    batch_replacements: list[dict[str, str]] = []
    diagnostics: dict[str, Any] = {
        "parse_status": "not_attempted",
        "raw_chars": 0,
        "replacement_count": 0,
    }
    try:
        reconstruction_response = gateway_factory(token_budget(_batch_word_budget(batch, extra_words=720))).chat(
            reconstruction_template.prompt,
            system="Return only valid JSON with a replacements array.",
            response_format={"type": "json_object"},
        )
        reconstruction_raw = reconstruction_response.content
        if _finish_reason(reconstruction_response) == "length":
            reconstruction_error = "llm_finish_reason_length"
            diagnostics = {
                "parse_status": "truncated_length",
                "finish_reason": "length",
                "raw_chars": len(str(reconstruction_raw or "")),
                "replacement_count": 0,
                "error": reconstruction_error,
            }
            reconstruction_raw = ""
        batch_replacements, diagnostics = parse_replacements_with_diagnostics(
            reconstruction_raw,
            preview_chars=preview_chars,
        ) if not reconstruction_error else ([], diagnostics)
        if not reconstruction_error:
            batch_replacements, structure_status = validate_replacement_structure(
                target_groups=batch,
                replacements=batch_replacements,
            )
            diagnostics["structure_status"] = structure_status
            diagnostics["structure_valid_count"] = len(batch_replacements)
            if not batch_replacements:
                reconstruction_error = "structure_contract_failed" if structure_status else str(diagnostics.get("parse_status") or "empty_replacements")
    except Exception as exc:
        reconstruction_error = str(exc)
        diagnostics = {
            "parse_status": "provider_error",
            "raw_chars": 0,
            "replacement_count": 0,
            "error": reconstruction_error,
        }
    trace = {
        **reconstruction_template.to_trace(),
        "batch_index": batch_index,
        "group_ids": [group.group_id for group in batch],
        "requested_groups": len(batch),
        "replacement_count": len(batch_replacements),
        "parse_diagnostics": diagnostics,
        "prompt_budget": {
            "max_prompt_chars": max_prompt_chars,
            "within_budget": len(reconstruction_template.prompt) <= max_prompt_chars,
            "compacted": budget_compacted,
        },
        "retry_of_batch": retry_of_batch,
        "error": reconstruction_error,
    }
    return batch_replacements, diagnostics, trace, reconstruction_error


def _ownership_prompt_for_budget(
    context: dict[str, Any],
    planner_output: dict[str, Any],
    *,
    max_prompt_chars: int,
) -> tuple[PromptTemplatePayload, bool]:
    template = build_paragraph_portfolio_ownership_prompt(context, planner_output)
    if len(template.prompt) <= max_prompt_chars:
        return template, False
    compact_template = build_paragraph_portfolio_ownership_prompt(
        context,
        planner_output,
        compact_for_budget=True,
    )
    return compact_template, True


def _call_ownership_batch(
    *,
    batch: list[TargetGroup],
    scan_contract: Any,
    content_mode: str,
    family: str,
    planner_output: dict[str, Any],
    gateway_factory: GatewayFactory,
    token_budget: TokenBudget,
    preview_chars: int,
    batch_index: int,
    max_prompt_chars: int,
    retry_of_batch: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str | None]:
    batch_context = _context_for_batch(
        batch=batch,
        scan_contract=scan_contract,
        content_mode=content_mode,
        family=family,
    )
    ownership_template, budget_compacted = _ownership_prompt_for_budget(
        batch_context,
        planner_output,
        max_prompt_chars=max_prompt_chars,
    )
    ownership_error = None
    batch_replacements: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "parse_status": "not_attempted",
        "raw_chars": 0,
        "replacement_count": 0,
    }
    try:
        ownership_response = gateway_factory(token_budget(_batch_word_budget(batch, extra_words=520))).chat(
            ownership_template.prompt,
            system="Return only valid JSON with a replacements array.",
            response_format={"type": "json_object"},
        )
        ownership_raw = ownership_response.content
        if _finish_reason(ownership_response) == "length":
            ownership_error = "llm_finish_reason_length"
            diagnostics = {
                "parse_status": "truncated_length",
                "finish_reason": "length",
                "raw_chars": len(str(ownership_raw or "")),
                "replacement_count": 0,
                "error": ownership_error,
            }
            ownership_raw = ""
        batch_replacements, diagnostics = parse_replacements_with_diagnostics(
            ownership_raw,
            preview_chars=preview_chars,
        ) if not ownership_error else ([], diagnostics)
        if not ownership_error:
            batch_replacements, structure_status = validate_replacement_structure(
                target_groups=batch,
                replacements=batch_replacements,
            )
            diagnostics["structure_status"] = structure_status
            diagnostics["structure_valid_count"] = len(batch_replacements)
            batch_replacements, ownership_status = validate_ownership_replacement_effect(
                target_groups=batch,
                replacements=batch_replacements,
            )
            diagnostics["ownership_status"] = ownership_status
            diagnostics["ownership_valid_count"] = len(batch_replacements)
            if not batch_replacements:
                ownership_error = "ownership_contract_failed" if ownership_status else str(diagnostics.get("parse_status") or "empty_replacements")
    except Exception as exc:
        ownership_error = str(exc)
        diagnostics = {
            "parse_status": "provider_error",
            "raw_chars": 0,
            "replacement_count": 0,
            "error": ownership_error,
        }
    trace = {
        **ownership_template.to_trace(),
        "batch_index": batch_index,
        "group_ids": [group.group_id for group in batch],
        "requested_groups": len(batch),
        "replacement_count": len(batch_replacements),
        "parse_diagnostics": diagnostics,
        "prompt_budget": {
            "max_prompt_chars": max_prompt_chars,
            "within_budget": len(ownership_template.prompt) <= max_prompt_chars,
            "compacted": budget_compacted,
        },
        "retry_of_batch": retry_of_batch,
        "error": ownership_error,
    }
    return batch_replacements, diagnostics, trace, ownership_error


def generate_paragraph_ownership_candidate(
    *,
    original_text: str,
    scan_contract: Any,
    content_mode: str,
    family: str,
    target_groups: list[TargetGroup],
    gateway_factory: GatewayFactory,
    token_budget: TokenBudget,
    config: ParagraphPortfolioConfig,
) -> tuple[str, dict[str, Any]]:
    stage_trace: list[dict[str, Any]] = []
    planner_output = fallback_paragraph_portfolio_plan(target_groups)
    planner_validation = {
        **validate_paragraph_portfolio_plan(planner_output, target_groups),
        "fallback_used": True,
        "fallback_reason": "claim_ownership_repair_uses_scanner_fallback_plan",
    }
    stage_trace.append({
        "prompt_template_id": TEMPLATE_ID,
        "strategy_id": "claim_ownership_repair",
        "prompt_stage": "planner",
        "prompt_chars": 0,
        "scanner_context_used": [
            "rewrite_target_profile.targets",
            "dominant_drivers",
            "hard_anchors",
            "soft_guidance_anchors",
            "word_count_guide",
        ],
        "validation": planner_validation,
        "error": None,
        "planner_mode": "scanner_fallback",
    })
    ownership_batches = build_reconstruction_batches_by_prompt(
        target_groups=target_groups,
        scan_contract=scan_contract,
        content_mode=content_mode,
        family=family,
        planner_output=planner_output,
        max_prompt_chars=config.max_reconstruction_prompt_chars,
        fallback_batch_size=config.fallback_batch_size,
    )
    replacements: list[dict[str, Any]] = []
    ownership_errors: list[str] = []
    parse_diagnostics: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(ownership_batches, start=1):
        batch_replacements, diagnostics, trace, ownership_error = _call_ownership_batch(
            batch=batch,
            scan_contract=scan_contract,
            content_mode=content_mode,
            family=family,
            planner_output=planner_output,
            gateway_factory=gateway_factory,
            token_budget=token_budget,
            preview_chars=config.raw_preview_chars,
            batch_index=batch_index,
            max_prompt_chars=config.max_reconstruction_prompt_chars,
        )
        if ownership_error and len(batch) > 1:
            retry_replacements: list[dict[str, Any]] = []
            retry_errors: list[str] = []
            for retry_offset, retry_group in enumerate(batch, start=1):
                retry_batch_index = (batch_index * 100) + retry_offset
                single_replacements, single_diagnostics, single_trace, single_error = _call_ownership_batch(
                    batch=[retry_group],
                    scan_contract=scan_contract,
                    content_mode=content_mode,
                    family=family,
                    planner_output=planner_output,
                    gateway_factory=gateway_factory,
                    token_budget=token_budget,
                    preview_chars=config.raw_preview_chars,
                    batch_index=retry_batch_index,
                    max_prompt_chars=config.max_reconstruction_prompt_chars,
                    retry_of_batch=batch_index,
                )
                stage_trace.append(single_trace)
                parse_diagnostics.append(single_diagnostics)
                retry_replacements.extend(single_replacements)
                if single_error:
                    retry_errors.append(single_error)
            if retry_replacements:
                batch_replacements = retry_replacements
                diagnostics = {
                    "parse_status": "retry_recovered" if not retry_errors else "retry_partial",
                    "replacement_count": len(retry_replacements),
                    "retry_error_count": len(retry_errors),
                    "original_error": ownership_error,
                }
                ownership_error = "; ".join(retry_errors) if retry_errors else None
        if ownership_error:
            ownership_errors.append(ownership_error)
        replacements.extend(batch_replacements)
        parse_diagnostics.append(diagnostics)
        stage_trace.append({**trace, "replacement_count": len(batch_replacements), "parse_diagnostics": diagnostics, "error": ownership_error})

    if not replacements:
        return "", target_execution_trace(
            attempted=True,
            target_groups=target_groups,
            replacements=[],
            batches=[
                {
                    "batch_index": index,
                    "group_ids": [group.group_id for group in batch],
                }
                for index, batch in enumerate(ownership_batches, start=1)
            ],
            error="; ".join(ownership_errors) if ownership_errors else "no_ownership_positive_replacement",
        ) | {
            "executor_engine": "paragraph_ownership_template",
            "prompt_template_id": "paragraph_portfolio.v1",
            "prompt_stage": "claim_ownership_repair",
            "ownership_repair_mode": True,
            "scanner_context_used": ["planner_output", "source_text", "ownership_contract", "hard_anchors", "word_count_guide"],
            "planner_output": planner_output,
            "prompt_stage_trace": stage_trace,
            "parse_diagnostics": parse_diagnostics,
            "strategy_stop_reason": "ownership_repair_failed",
        }

    text, apply_status = apply_target_replacements(
        original_text=original_text,
        target_groups=target_groups,
        replacements=replacements,
    )
    applied_group_ids = {
        str(row.get("group_id") or "")
        for row in apply_status
        if isinstance(row, dict) and row.get("applied")
    }
    applied_replacements = [
        row for row in replacements
        if str(row.get("group_id") or "") in applied_group_ids
    ]
    expected_group_ids = {group.group_id for group in target_groups}
    replaced_group_ids = {
        str(row.get("group_id") or "")
        for row in applied_replacements
        if str(row.get("group_id") or "")
    }
    missing_replacement_group_ids = sorted(expected_group_ids - replaced_group_ids)
    stop_reason = "ownership_repair_generated" if applied_replacements else "no_target_replacement_applied"
    trace = target_execution_trace(
        attempted=True,
        target_groups=target_groups,
        replacements=applied_replacements,
        apply_status=apply_status,
        batches=[
            {
                "batch_index": index,
                "group_ids": [group.group_id for group in batch],
            }
            for index, batch in enumerate(ownership_batches, start=1)
        ],
        error=None if applied_replacements else stop_reason,
    )
    trace.update({
        "executor_engine": "paragraph_ownership_template",
        "prompt_template_id": "paragraph_portfolio.v1",
        "prompt_stage": "claim_ownership_repair",
        "ownership_repair_mode": True,
        "scanner_context_used": list(dict.fromkeys([
            "rewrite_target_profile.targets",
            "planner_output",
            "source_text",
            "before_context",
            "after_context",
            "ownership_contract",
            "hard_anchors",
            "soft_guidance_anchors",
            "word_count_guide",
        ])),
        "planner_output": planner_output,
        "prompt_stage_trace": stage_trace,
        "parse_diagnostics": parse_diagnostics,
        "stage_apply_status": apply_status,
        "partial_candidate": bool(missing_replacement_group_ids),
        "missing_replacement_group_ids": missing_replacement_group_ids,
        "accepted_replacements": applied_replacements,
        "strategy_stop_reason": stop_reason if stop_reason != "ownership_repair_generated" else (
            "partial_ownership_repair_generated" if missing_replacement_group_ids else "ownership_repair_generated"
        ),
    })
    if text.strip() == original_text.strip():
        trace["error"] = "no_target_replacement_applied"
        trace["strategy_stop_reason"] = "no_target_replacement_applied"
    return clean_v3_candidate_output(text), trace


def generate_paragraph_portfolio_candidate(
    *,
    original_text: str,
    scan_contract: Any,
    content_mode: str,
    family: str,
    target_groups: list[TargetGroup],
    gateway_factory: GatewayFactory,
    token_budget: TokenBudget,
    config: ParagraphPortfolioConfig,
) -> tuple[str, dict[str, Any]]:
    stage_trace: list[dict[str, Any]] = []

    planner_error = None
    planner_mode = "scanner_fallback"
    planner_output = fallback_paragraph_portfolio_plan(target_groups)
    planner_validation = {
        **validate_paragraph_portfolio_plan(planner_output, target_groups),
        "fallback_used": True,
        "fallback_reason": "llm_planner_disabled",
    }
    if config.llm_planner_enabled:
        planner_mode = "llm_planner"
        context = _context_for_batch(
            batch=target_groups,
            scan_contract=scan_contract,
            content_mode=content_mode,
            family=family,
        )
        planner_template = build_paragraph_portfolio_planner_prompt(context)
        try:
            planner_raw = gateway_factory(token_budget(500)).chat(
                planner_template.prompt,
                system="Return only valid JSON with paragraph_plans.",
                response_format={"type": "json_object"},
            ).content
            planner_output = parse_paragraph_portfolio_plan(planner_raw)
        except Exception as exc:
            planner_error = str(exc)
            planner_output = {"paragraph_plans": []}
        planner_validation = validate_paragraph_portfolio_plan(planner_output, target_groups)
        if not planner_validation["passed"]:
            planner_output = fallback_paragraph_portfolio_plan(target_groups)
            planner_validation = {
                **validate_paragraph_portfolio_plan(planner_output, target_groups),
                "fallback_used": True,
                "fallback_reason": planner_error or "planner_validation_failed",
            }
        stage_trace.append({
            **planner_template.to_trace(),
            "validation": planner_validation,
            "error": planner_error,
            "planner_mode": planner_mode,
        })
    else:
        stage_trace.append({
            "prompt_template_id": "paragraph_portfolio.v1",
            "strategy_id": "paragraph_preserving_broad_reconstruction",
            "prompt_stage": "planner",
            "prompt_chars": 0,
            "scanner_context_used": [
                "rewrite_target_profile.targets",
                "dominant_drivers",
                "hard_anchors",
                "soft_guidance_anchors",
                "word_count_guide",
            ],
            "validation": planner_validation,
            "error": None,
            "planner_mode": planner_mode,
        })

    reconstruction_batches = build_reconstruction_batches_by_prompt(
        target_groups=target_groups,
        scan_contract=scan_contract,
        content_mode=content_mode,
        family=family,
        planner_output=planner_output,
        max_prompt_chars=config.max_reconstruction_prompt_chars,
        fallback_batch_size=config.fallback_batch_size,
    )
    replacements: list[dict[str, str]] = []
    reconstruction_errors: list[str] = []
    parse_diagnostics: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(reconstruction_batches, start=1):
        batch_replacements, diagnostics, trace, reconstruction_error = _call_reconstruction_batch(
            batch=batch,
            scan_contract=scan_contract,
            content_mode=content_mode,
            family=family,
            planner_output=planner_output,
            gateway_factory=gateway_factory,
            token_budget=token_budget,
            preview_chars=config.raw_preview_chars,
            batch_index=batch_index,
            max_prompt_chars=config.max_reconstruction_prompt_chars,
        )
        if reconstruction_error and len(batch) > 1:
            retry_replacements: list[dict[str, str]] = []
            retry_errors: list[str] = []
            for retry_offset, retry_group in enumerate(batch, start=1):
                retry_batch_index = (batch_index * 100) + retry_offset
                single_replacements, single_diagnostics, single_trace, single_error = _call_reconstruction_batch(
                    batch=[retry_group],
                    scan_contract=scan_contract,
                    content_mode=content_mode,
                    family=family,
                    planner_output=planner_output,
                    gateway_factory=gateway_factory,
                    token_budget=token_budget,
                    preview_chars=config.raw_preview_chars,
                    batch_index=retry_batch_index,
                    max_prompt_chars=config.max_reconstruction_prompt_chars,
                    retry_of_batch=batch_index,
                )
                stage_trace.append(single_trace)
                parse_diagnostics.append(single_diagnostics)
                retry_replacements.extend(single_replacements)
                if single_error:
                    retry_errors.append(single_error)
            if retry_replacements:
                batch_replacements = retry_replacements
                diagnostics = {
                    "parse_status": "retry_recovered" if not retry_errors else "retry_partial",
                    "replacement_count": len(retry_replacements),
                    "retry_error_count": len(retry_errors),
                    "original_error": reconstruction_error,
                }
                reconstruction_error = "; ".join(retry_errors) if retry_errors else None
        if reconstruction_error:
            reconstruction_errors.append(reconstruction_error)
        replacements.extend(batch_replacements)
        parse_diagnostics.append(diagnostics)
        stage_trace.append({**trace, "replacement_count": len(batch_replacements), "parse_diagnostics": diagnostics, "error": reconstruction_error})
    expected_group_ids = {group.group_id for group in target_groups}
    replaced_group_ids = {
        str(row.get("group_id") or "")
        for row in replacements
        if str(row.get("group_id") or "")
    }
    missing_replacement_group_ids = sorted(expected_group_ids - replaced_group_ids)
    if not replacements:
        return "", target_execution_trace(
            attempted=True,
            target_groups=target_groups,
            replacements=[],
            batches=[
                {
                    "batch_index": index,
                    "group_ids": [group.group_id for group in batch],
                }
                for index, batch in enumerate(reconstruction_batches, start=1)
            ],
            error="; ".join(reconstruction_errors) if reconstruction_errors else "generation_failed_empty_output",
        ) | {
            "executor_engine": "paragraph_portfolio_template",
            "prompt_template_id": "paragraph_portfolio.v1",
            "prompt_stage": "paragraph_reconstruction",
            "scanner_context_used": ["planner_output", "source_text", "dominant_drivers", "word_count_guide"],
            "planner_output": planner_output,
            "prompt_stage_trace": stage_trace,
            "parse_diagnostics": parse_diagnostics,
            "topk_repair_attempted": False,
            "strategy_stop_reason": "reconstruction_failed",
        }
    partial_candidate = bool(missing_replacement_group_ids)

    replacements_by_group = {
        row["group_id"]: row
        for row in replacements
        if row.get("group_id")
    }
    topk_replacements: list[dict[str, str]] = []
    topk_enabled = config.blind_topk_enabled or _auto_topk_repair_enabled(scan_contract, target_groups)
    if topk_enabled:
        for batch_index, batch in enumerate(reconstruction_batches, start=1):
            batch_ids = {group.group_id for group in batch}
            batch_replacements = [row for row in replacements if row.get("group_id") in batch_ids]
            if not batch_replacements:
                continue
            replacement_group_ids = {str(row.get("group_id") or "") for row in batch_replacements}
            topk_batch = [group for group in batch if group.group_id in replacement_group_ids]
            if not topk_batch:
                continue
            batch_context = _context_for_batch(
                batch=topk_batch,
                scan_contract=scan_contract,
                content_mode=content_mode,
                family=family,
            )
            topk_template = build_paragraph_portfolio_topk_prompt(
                context=batch_context,
                planner_output=planner_output,
                replacements=batch_replacements,
            )
            topk_error = None
            topk_diagnostics: dict[str, Any] = {"parse_status": "not_attempted", "replacement_count": 0}
            try:
                topk_response = gateway_factory(token_budget(_batch_word_budget(batch, extra_words=260))).chat(
                    topk_template.prompt,
                    system="Return only valid JSON with a replacements array.",
                    response_format={"type": "json_object"},
                )
                topk_raw = topk_response.content
                if _finish_reason(topk_response) == "length":
                    topk_error = "llm_finish_reason_length"
                    topk_diagnostics = {
                        "parse_status": "truncated_length",
                        "finish_reason": "length",
                        "raw_chars": len(str(topk_raw or "")),
                        "replacement_count": 0,
                        "error": topk_error,
                    }
                    batch_topk_replacements = []
                else:
                    batch_topk_replacements, topk_diagnostics = parse_replacements_with_diagnostics(
                        topk_raw,
                        preview_chars=config.raw_preview_chars,
                    )
                    batch_topk_replacements, topk_structure_status = validate_replacement_structure(
                        target_groups=topk_batch,
                        replacements=batch_topk_replacements,
                    )
                    topk_diagnostics["structure_status"] = topk_structure_status
                    topk_diagnostics["structure_valid_count"] = len(batch_topk_replacements)
                    batch_topk_replacements, topk_effect_status = validate_topk_replacement_effect(
                        target_groups=topk_batch,
                        current_replacements=batch_replacements,
                        replacements=batch_topk_replacements,
                        predictability_briefs=list(getattr(scan_contract, "predictability_briefs", ()) or ()),
                    )
                    topk_diagnostics["effect_status"] = topk_effect_status
                    topk_diagnostics["effect_valid_count"] = len(batch_topk_replacements)
            except Exception as exc:
                topk_error = str(exc)
                batch_topk_replacements = []
                topk_diagnostics = {
                    "parse_status": "provider_error",
                    "replacement_count": 0,
                    "error": topk_error,
                }
            for row in batch_topk_replacements:
                if row.get("group_id"):
                    replacements_by_group[row["group_id"]] = row
            topk_replacements.extend(batch_topk_replacements)
            stage_trace.append({
                **topk_template.to_trace(),
                "batch_index": batch_index,
                "group_ids": [group.group_id for group in topk_batch],
                "replacement_count": len(batch_topk_replacements),
                "parse_diagnostics": topk_diagnostics,
                "error": topk_error,
                "applied": bool(batch_topk_replacements),
                "topk_mode": "auto_enabled" if not config.blind_topk_enabled else "blind_enabled",
            })
    else:
        stage_trace.append({
            "prompt_template_id": "paragraph_portfolio.v1",
            "strategy_id": "paragraph_preserving_broad_reconstruction",
            "prompt_stage": "topk_repair",
            "prompt_chars": 0,
            "scanner_context_used": [],
            "replacement_count": 0,
            "error": None,
            "applied": False,
            "topk_mode": "deferred_until_rescan",
        })
    if topk_replacements:
        replacements = list(replacements_by_group.values())

    text, apply_status = apply_target_replacements(
        original_text=original_text,
        target_groups=target_groups,
        replacements=replacements,
    )
    stop_reason = "candidate_generated"
    if not any(row.get("applied") for row in apply_status):
        stop_reason = "no_target_replacement_applied"
    trace = target_execution_trace(
        attempted=True,
        target_groups=target_groups,
        replacements=replacements,
        apply_status=apply_status,
        batches=[
            {
                "batch_index": index,
                "group_ids": [group.group_id for group in batch],
            }
            for index, batch in enumerate(reconstruction_batches, start=1)
        ],
        error=None if stop_reason == "candidate_generated" else stop_reason,
    )
    trace.update({
        "executor_engine": "paragraph_portfolio_template",
        "prompt_template_id": "paragraph_portfolio.v1",
        "prompt_stage": "topk_repair" if topk_replacements else "paragraph_reconstruction",
        "scanner_context_used": list(dict.fromkeys([
            "rewrite_target_profile.targets",
            "planner_output",
            "source_text",
            "before_context",
            "after_context",
            "dominant_drivers",
            "required_movement",
            "hard_anchors",
            "soft_guidance_anchors",
            "word_count_guide",
            "current_replacements",
        ])),
        "planner_output": planner_output,
        "prompt_stage_trace": stage_trace,
        "parse_diagnostics": parse_diagnostics,
        "stage_apply_status": apply_status,
        "stage_rescan_delta": None,
        "partial_candidate": partial_candidate,
        "missing_replacement_group_ids": missing_replacement_group_ids,
        "topk_repair_attempted": bool(topk_enabled),
        "strategy_stop_reason": stop_reason if stop_reason != "candidate_generated" else (
            "partial_topk_repair_applied" if partial_candidate and topk_replacements else (
                "topk_repair_applied" if topk_replacements else (
                    "partial_reconstruction_generated_topk_deferred"
                    if partial_candidate
                    else "reconstruction_generated_topk_deferred"
                )
            )
        ),
    })
    if text.strip() == original_text.strip():
        trace["error"] = "no_target_replacement_applied"
        trace["strategy_stop_reason"] = "no_target_replacement_applied"
    return clean_v3_candidate_output(text), trace
