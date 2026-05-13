"""Scan-driven rewrite pipeline V2 entrypoint."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from detect.run import DetectionRunner
from llm.gateway import (
    LLMConfig,
    LLMGateway,
    model_supports_presence_frequency_penalties,
    model_supports_repetition_penalty,
)
from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from report.report import ReportBuilder, report_to_dict
from rewrite.guards import check_semantic_drift, detect_protected_spans, protected_spans_preserved

from .contracts import AnchorSeverity, anchor_present, build_rewrite_contract
from .diagnostics import annotate_candidate_diagnostics, summarize_candidate_diagnostics
from .goal_contract import RewriteGoalStatus, evaluate_rewrite_goal, needs_author_context
from .goal_contract import RewriteGoalEvaluation
from .layer_attempts import record_layer_attempt, summarize_layer_attempts
from .robustness import layer_failure_class_counts, normalize_strategy_layer, portfolio_limits, recommend_failure_policy
from .runtime_budget import RewriteV2RuntimeBudget
from .selection import (
    CandidateLane,
    decide_candidate,
    select_best_applicable_candidate,
    select_best_candidate,
    select_best_safe_progress_candidate,
)
from .layers.academic import (
    _academic_all_section_filter_failures,
    _academic_assignment_sections,
    _academic_section_filter_failures,
    _academic_section_targets,
    _all_section_compact_allowed,
    _compose_academic_sections,
    _generate_academic_anchor_repair_candidates,
    _generate_academic_all_section_candidates,
    _generate_academic_section_candidates,
    _normalize_academic_all_section_candidate,
    _parse_academic_all_section_variants,
)
from .strategy import (
    build_single_paragraph_reconstruction_prompt,
    build_strategy_prompt,
    classify_content_route,
    clean_candidate_output,
    route_strategies,
    targeted_paragraph_briefs,
)
from .structured_output import json_from_response, json_parse_diagnostics, structured_json_request_options


def _semantic_scan_allowed(strategy_kind: str | None, semantic_safe: bool) -> bool:
    if semantic_safe:
        return True
    if strategy_kind in {
        "full_rewrite",
        "entity_locked_full_reconstruction",
        "author_stance_thesis_reframe",
        "author_stance_texture_pass",
    }:
        return True
    return os.environ.get("DRAFTPROOF_REWRITE_V2_SCAN_REVIEW_CANDIDATES", "1").lower() not in {"0", "false", "no"}


def _local_filter_rejection_reason(generated_candidate: dict[str, Any]) -> str:
    layer = normalize_strategy_layer(generated_candidate)
    if layer == "targeted_paragraph_reconstruction":
        return "targeted_local_filter_rejected"
    if layer == "academic_anchor_repair_texture_pass":
        return "academic_repair_local_filter_rejected"
    if str(layer).startswith("academic_"):
        return "academic_local_filter_rejected"
    if layer in {
        "entity_locked_full_reconstruction",
        "keyword_locked_short_texture",
        "author_stance_thesis_reframe",
        "author_stance_texture_pass",
    }:
        return "full_document_local_filter_rejected"
    return "candidate_local_filter_rejected"


def _local_filter_rejected_candidate_row(generated_candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        **generated_candidate,
        "candidate_ai": None,
        "decision": {
            "lane": CandidateLane.REJECT.value,
            "reason": _local_filter_rejection_reason(generated_candidate),
            "rank": [],
        },
    }


def _empty_generated_candidate_row(generated_candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        **generated_candidate,
        "candidate_ai": None,
        "local_filter_passed": False,
        "local_filter_failures": list(generated_candidate.get("local_filter_failures") or []) or ["empty_candidate_text"],
        "decision": {
            "lane": CandidateLane.REJECT.value,
            "reason": "empty_generated_candidate_text",
            "rank": [],
        },
    }


def _close_partial_max_gap() -> float:
    return float(os.environ.get("DRAFTPROOF_REWRITE_V2_APPLY_PARTIAL_MAX_GAP", "2.0") or 2.0)


def _composition_partial_max_gap() -> float:
    return float(os.environ.get("DRAFTPROOF_REWRITE_V2_APPLY_COMPOSITION_MAX_GAP", "3.0") or 3.0)


def _composition_ai_penalty_max() -> float:
    return float(os.environ.get("DRAFTPROOF_REWRITE_V2_COMPOSITION_AI_PENALTY_MAX", "2.0") or 2.0)


def _llm_call_timeout_seconds(default: int = 30) -> int:
    raw = os.environ.get("DRAFTPROOF_REWRITE_V2_LLM_TIMEOUT_SECONDS", str(default))
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = default
    return max(5, min(60, value))


def _generation_budget_seconds(max_runtime_seconds: int) -> int:
    raw = os.environ.get("DRAFTPROOF_REWRITE_V2_GENERATION_BUDGET_SECONDS")
    if raw:
        try:
            configured = int(float(raw))
        except (TypeError, ValueError):
            configured = 0
        if configured > 0:
            return max(30, min(configured, max(30, int(max_runtime_seconds) - 30)))
    return max(30, min(180, int(max_runtime_seconds * 0.65), max(30, int(max_runtime_seconds) - 60)))


def _phase_start_margin_seconds(default: int = 5) -> int:
    raw = os.environ.get("DRAFTPROOF_REWRITE_V2_PHASE_START_MARGIN_SECONDS", str(default))
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = default
    return max(1, min(60, value))


def _portfolio_mode_enabled() -> bool:
    return os.environ.get("DRAFTPROOF_REWRITE_V2_PORTFOLIO_MODE", "1").lower() not in {"0", "false", "no"}


def _candidate_portfolio_allows(
    rows: list[dict[str, Any]],
    content_route: Any | None,
    candidate: dict[str, Any] | str | None = None,
) -> bool:
    limits = portfolio_limits(content_route)
    max_candidates = int(limits.get("max_generated_candidates") or 8)
    if len(rows) >= max_candidates:
        return False
    if candidate is None:
        return True
    layer = normalize_strategy_layer(candidate)
    cap = (limits.get("layer_candidate_caps") or {}).get(layer)
    if isinstance(cap, int) and cap >= 0:
        current = sum(1 for row in rows if normalize_strategy_layer(row) == layer)
        if current >= cap:
            return False
    return True


def _effective_config(
    *,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    max_runtime_seconds: int,
) -> dict[str, Any]:
    effective_model = model or os.environ.get("LLM_MODEL") or "openai/gpt-4.1-mini"
    effective_base_url = base_url or os.environ.get("LLM_BASE_URL", "")
    generation_budget = _generation_budget_seconds(max_runtime_seconds)
    return {
        "model": effective_model,
        "base_url": effective_base_url,
        "has_api_key": bool(api_key),
        "max_runtime_seconds": int(max_runtime_seconds),
        "llm_call_timeout_seconds": _llm_call_timeout_seconds(),
        "generation_budget_seconds": generation_budget,
        "phase_start_margin_seconds": _phase_start_margin_seconds(),
        "apply_partial_max_gap": _close_partial_max_gap(),
        "apply_composition_max_gap": _composition_partial_max_gap(),
        "composition_ai_penalty_max": _composition_ai_penalty_max(),
        "targeted_paragraphs": int(os.environ.get("DRAFTPROOF_REWRITE_V2_TARGETED_PARAGRAPHS", "4") or 4),
        "targeted_candidates": int(os.environ.get("DRAFTPROOF_REWRITE_V2_TARGETED_CANDIDATES", "4") or 4),
        "tactics": _paragraph_tactics(),
        "use_paragraph_local_score": _use_paragraph_local_score_gate(),
        "unsafe_cluster_rescue": os.environ.get("DRAFTPROOF_REWRITE_V2_UNSAFE_CLUSTER_RESCUE", "1").lower() not in {"0", "false", "no"},
        "allow_full_after_targeted": os.environ.get("DRAFTPROOF_REWRITE_V2_ALLOW_FULL_AFTER_TARGETED", "0").lower() in {"1", "true", "yes"},
        "entity_locked_full_reconstruction": os.environ.get("DRAFTPROOF_REWRITE_V2_ENTITY_LOCKED_FULL_RECONSTRUCTION", "1").lower() not in {"0", "false", "no"},
        "keyword_locked_short_texture": os.environ.get("DRAFTPROOF_REWRITE_V2_KEYWORD_LOCKED_SHORT_TEXTURE", "0").lower() not in {"0", "false", "no"},
        "author_stance_thesis_reframe": os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_THESIS_REFRAME", "1").lower() not in {"0", "false", "no"},
        "author_stance_candidates": int(os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_CANDIDATES", "3") or 3),
        "author_stance_texture_pass": os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_TEXTURE_PASS", "0").lower() not in {"0", "false", "no"},
        "academic_anchor_repair_texture_pass": os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_ANCHOR_REPAIR", "1").lower() not in {"0", "false", "no"},
        "academic_all_section_compact_reconstruction": os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_ALL_SECTION_COMPACT", "1").lower() not in {"0", "false", "no"},
        "academic_all_section_compact_candidates": int(os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_ALL_SECTION_CANDIDATES", "2") or 2),
        "academic_section_resolver": os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_SECTION_RESOLVER", "1").lower() not in {"0", "false", "no"},
        "academic_section_candidates": int(os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_SECTION_CANDIDATES", "1") or 1),
        "academic_section_max_sections": int(os.environ.get("DRAFTPROOF_REWRITE_V2_ACADEMIC_SECTION_MAX_SECTIONS", "2") or 2),
        "portfolio_mode": _portfolio_mode_enabled(),
        "global_max_generated_candidates": os.environ.get("DRAFTPROOF_REWRITE_V2_MAX_GENERATED_CANDIDATES"),
    }


def _strategy_family_allowed(content_route: Any, family: str) -> bool:
    if content_route is None:
        return True
    allowed = getattr(content_route, "allowed_strategy_families", None)
    if allowed is None and isinstance(content_route, dict):
        allowed = content_route.get("allowed_strategy_families")
    return family in set(allowed or [])


def _extract_original_text(detect_json: dict[str, Any]) -> str:
    for key in ("input_text", "original_text", "document_text", "text", "content"):
        value = detect_json.get(key)
        if isinstance(value, str) and value.strip():
            return value
    sentence_map = detect_json.get("sentence_map")
    if isinstance(sentence_map, dict):
        rows = [
            str((row or {}).get("text") or "").strip()
            for _, row in sorted(sentence_map.items())
            if isinstance(row, dict) and str(row.get("text") or "").strip()
        ]
        if rows:
            return " ".join(rows)
    raise ValueError("rewrite_v2 requires original text in detect JSON or sentence_map")


def _scan_report(text: str) -> dict[str, Any]:
    detect_report = DetectionRunner().run_all(text)
    builder = ReportBuilder()
    builder.add_detection_report(detect_report)
    if detect_report.postprocess_results:
        builder.add_postprocess_results(detect_report.postprocess_results)
    builder.set_meta(scan_time=0, original_text=text)
    return report_to_dict(builder.build())


def _badge_ai(report: dict | None) -> float | None:
    score = ((report or {}).get("ai_risk_badge") or {}).get("ai_likelihood_score")
    return float(score) if isinstance(score, (int, float)) else None


def _badge_wq(report: dict | None) -> float | None:
    score = ((report or {}).get("ai_risk_badge") or {}).get("writing_quality_score")
    return float(score) if isinstance(score, (int, float)) else None


def _rewrite_smoothness(report: dict | None) -> float | None:
    value = (((report or {}).get("ai_risk_badge") or {}).get("ai_components") or {}).get("rewrite_smoothness")
    return float(value) if isinstance(value, (int, float)) else None


def _first_applied_paragraph_patch(candidate: dict[str, Any]) -> dict[str, Any] | None:
    for patch in candidate.get("patches") or []:
        if isinstance(patch, dict) and patch.get("applied") and patch.get("target_paragraph") and patch.get("rewritten_paragraph"):
            return patch
    return None


def _paragraph_local_score(candidate: dict[str, Any]) -> dict[str, Any] | None:
    patch = _first_applied_paragraph_patch(candidate)
    if not patch:
        return None
    original_report = _scan_report(str(patch["target_paragraph"]))
    rewritten_report = _scan_report(str(patch["rewritten_paragraph"]))
    original_ai = _badge_ai(original_report)
    rewritten_ai = _badge_ai(rewritten_report)
    original_wq = _badge_wq(original_report)
    rewritten_wq = _badge_wq(rewritten_report)
    ai_drop = (
        round(float(original_ai) - float(rewritten_ai), 3)
        if isinstance(original_ai, (int, float)) and isinstance(rewritten_ai, (int, float))
        else None
    )
    return {
        "paragraph_id": patch.get("paragraph_id"),
        "original_ai": original_ai,
        "rewritten_ai": rewritten_ai,
        "ai_drop": ai_drop,
        "original_wq": original_wq,
        "rewritten_wq": rewritten_wq,
        "improved": bool(isinstance(ai_drop, (int, float)) and ai_drop > 0.0),
    }


def _compose_local_winners(original_text: str, rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    winners: dict[str, dict[str, Any]] = {}
    for row in rows:
        local = row.get("paragraph_local_score")
        if not isinstance(local, dict) or not local.get("improved"):
            continue
        paragraph_id = str(local.get("paragraph_id") or row.get("paragraph_id") or "")
        if not paragraph_id:
            continue
        current = winners.get(paragraph_id)
        if current is None or float(local.get("ai_drop") or 0.0) > float((current.get("paragraph_local_score") or {}).get("ai_drop") or 0.0):
            winners[paragraph_id] = row
    text = original_text
    applied: list[dict[str, Any]] = []
    for paragraph_id, row in winners.items():
        patch = _first_applied_paragraph_patch(row)
        if not patch:
            continue
        target = str(patch.get("target_paragraph") or "")
        replacement = str(patch.get("rewritten_paragraph") or "")
        if target and replacement and target in text:
            text = text.replace(target, replacement, 1)
            applied.append({
                "paragraph_id": paragraph_id,
                "local_ai_drop": (row.get("paragraph_local_score") or {}).get("ai_drop"),
                "candidate_ai": row.get("candidate_ai"),
                "candidate_number": row.get("candidate_number"),
                "strategy": row.get("strategy"),
            })
    return text, applied


def _compose_full_doc_delta_winners(
    original_text: str,
    rows: list[dict[str, Any]],
    reference_ai: float | None,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(reference_ai, (int, float)):
        return original_text, []
    winners: dict[str, dict[str, Any]] = {}
    for row in rows:
        decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
        if not decision.get("quality_safe") or not decision.get("semantic_safe"):
            continue
        if row.get("protected_anchors_safe") is False or row.get("semantic_safe") is False:
            continue
        paragraph_id = str(row.get("paragraph_id") or "")
        candidate_ai = row.get("candidate_ai")
        if not paragraph_id or not isinstance(candidate_ai, (int, float)):
            continue
        full_doc_delta = float(reference_ai) - float(candidate_ai)
        if full_doc_delta <= 0.0:
            continue
        patch = _first_applied_paragraph_patch(row)
        if not patch:
            continue
        current = winners.get(paragraph_id)
        if current is None or full_doc_delta > float(current.get("full_doc_delta") or 0.0):
            winners[paragraph_id] = {
                "row": row,
                "patch": patch,
                "full_doc_delta": round(full_doc_delta, 3),
            }
    text = original_text
    applied: list[dict[str, Any]] = []
    for paragraph_id, winner in winners.items():
        patch = winner["patch"]
        target = str(patch.get("target_paragraph") or "")
        replacement = str(patch.get("rewritten_paragraph") or "")
        if target and replacement and target in text:
            row = winner["row"]
            text = text.replace(target, replacement, 1)
            applied.append({
                "paragraph_id": paragraph_id,
                "full_doc_delta": winner["full_doc_delta"],
                "local_ai_drop": (row.get("paragraph_local_score") or {}).get("ai_drop"),
                "candidate_ai": row.get("candidate_ai"),
                "candidate_number": row.get("candidate_number"),
                "strategy": row.get("strategy"),
                "tactic": row.get("tactic"),
            })
    return text, applied


def _candidate_patch_coverage(candidate: dict[str, Any] | None) -> int:
    if not isinstance(candidate, dict):
        return 0
    composed = candidate.get("composed_patches")
    if isinstance(composed, list):
        return len(composed)
    count = candidate.get("applied_patch_count")
    if isinstance(count, (int, float)):
        return int(count)
    patches = candidate.get("patches")
    if isinstance(patches, list):
        return sum(1 for patch in patches if isinstance(patch, dict) and patch.get("applied"))
    return 0


def _is_safe_partial_candidate(candidate: dict[str, Any] | None, *, max_gap: float) -> bool:
    if not isinstance(candidate, dict):
        return False
    decision = candidate.get("decision") if isinstance(candidate.get("decision"), dict) else {}
    if decision.get("lane") != CandidateLane.PARTIAL_DIAGNOSTIC.value:
        return False
    if not decision.get("quality_safe") or not decision.get("semantic_safe"):
        return False
    gap = decision.get("ai_target_gap")
    return isinstance(gap, (int, float)) and float(gap) <= float(max_gap)


def _content_mode_value(content_route: Any | None) -> str:
    if content_route is None:
        return ""
    if isinstance(content_route, dict):
        return str(content_route.get("content_mode") or "")
    return str(getattr(content_route, "content_mode", "") or "")


def _candidate_lane(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    decision = candidate.get("decision") if isinstance(candidate.get("decision"), dict) else {}
    return str(decision.get("lane") or "")


def _prefer_author_stance_frontier(
    best: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    *,
    content_route: Any | None = None,
) -> dict[str, Any]:
    if _content_mode_value(content_route) != "broad_explanatory_essay":
        return best
    if _candidate_lane(best) == CandidateLane.GOAL_MET.value:
        return best
    if str(best.get("strategy") or "") == "scan_author_stance_thesis_reframe":
        return best
    author_candidates = [
        row for row in candidate_rows
        if str(row.get("strategy") or "") == "scan_author_stance_thesis_reframe"
        and _candidate_lane(row) in {CandidateLane.SAFE_NEAR_MISS.value, CandidateLane.GOAL_MET.value}
        and ((row.get("decision") or {}).get("quality_safe"))
        and ((row.get("decision") or {}).get("semantic_safe"))
    ]
    if not author_candidates:
        return best
    preferred = max(author_candidates, key=lambda row: tuple((row.get("decision") or {}).get("rank") or ()))
    if _candidate_lane(preferred) == CandidateLane.GOAL_MET.value:
        return preferred
    preferred["strategy_preferred_over"] = {
        "strategy": best.get("strategy"),
        "strategy_kind": best.get("strategy_kind"),
        "candidate_ai": best.get("candidate_ai"),
        "reason": "broad_explanatory_essay_prefers_author_stance_over_rescue_or_reconstruction",
    }
    return preferred


def _select_best_v2_frontier(
    candidate_rows: list[dict[str, Any]],
    *,
    content_route: Any | None = None,
) -> dict[str, Any] | None:
    close_gap = _close_partial_max_gap()
    best = (
        select_best_applicable_candidate(candidate_rows, close_partial_max_gap=close_gap)
        or select_best_safe_progress_candidate(candidate_rows)
    )
    if not best:
        return None
    best = _prefer_author_stance_frontier(best, candidate_rows, content_route=content_route)
    best_ai = best.get("candidate_ai")
    if not isinstance(best_ai, (int, float)):
        return best
    best_coverage = _candidate_patch_coverage(best)
    composition_candidates = [
        row for row in candidate_rows
        if row.get("strategy") == "scan_targeted_composed_full_doc_delta_winners"
        and _candidate_patch_coverage(row) >= max(2, best_coverage + 2)
        and isinstance(row.get("candidate_ai"), (int, float))
        and _is_safe_partial_candidate(row, max_gap=_composition_partial_max_gap())
    ]
    if not composition_candidates:
        return best
    composition = min(composition_candidates, key=lambda row: float(row.get("candidate_ai") or 999.0))
    ai_penalty = float(composition.get("candidate_ai") or 999.0) - float(best_ai)
    if ai_penalty <= _composition_ai_penalty_max():
        composition["coverage_preferred_over"] = {
            "strategy": best.get("strategy"),
            "paragraph_id": best.get("paragraph_id"),
            "candidate_ai": best.get("candidate_ai"),
            "coverage": best_coverage,
            "ai_penalty": round(ai_penalty, 3),
            "reason": "safe_composition_covers_more_paragraphs_with_bounded_ai_penalty",
        }
        return composition
    return best


def _paragraph_target_map(scan_report: dict | None, original_text: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", str(original_text or ""))
        if paragraph.strip()
    ]
    for index, paragraph in enumerate(paragraphs, start=1):
        result[f"p{index:03d}"] = paragraph
    for brief in (scan_report or {}).get("rewrite_edit_briefs") or []:
        if not isinstance(brief, dict):
            continue
        paragraph_id = str(brief.get("paragraph_id") or "").strip()
        paragraph = str(brief.get("paragraph_excerpt") or "").strip()
        if paragraph_id and paragraph and paragraph_id not in result:
            result[paragraph_id] = paragraph
    return result


def _entities_from_target_text(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9&'-]{2,}(?:\s+[A-Z][A-Za-z0-9&'-]{2,}){0,4}\b", text or ""):
        entity = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:")
        if entity and entity not in values:
            values.append(entity)
    for match in re.finditer(r"\b(?:\d{4}|\d+(?:\.\d+)?%?)\b", text or ""):
        value = match.group(0)
        if value not in values:
            values.append(value)
    return values[:80]


def _keywords_from_target_text(text: str) -> list[str]:
    values: list[str] = []
    stop = {
        "about", "after", "also", "because", "between", "could", "from", "have", "into",
        "more", "most", "over", "that", "their", "there", "these", "this", "those",
        "through", "under", "when", "where", "which", "while", "with", "would",
    }
    for token in re.findall(r"\b[A-Za-z][A-Za-z'-]{3,}\b", text or ""):
        lowered = token.lower()
        if lowered not in stop and lowered not in values:
            values.append(lowered)
    return values[:40]


def _required_entities_for_full_reconstruction(text: str) -> list[str]:
    stop = {
        "another", "although", "also", "because", "cities", "critics", "despite", "education",
        "healthcare", "however", "immigration", "many", "millions", "one", "organizations",
        "political", "some", "sports", "supporters", "technology", "the", "this", "throughout",
        "understanding", "university", "while",
    }
    values: list[str] = []
    seen: set[str] = set()
    source = str(text or "")
    pattern = re.compile(r"\b[A-Z][A-Za-z0-9&'-]{2,}(?:\s+[A-Z][A-Za-z0-9&'-]{2,}){0,4}\b")
    for match in pattern.finditer(source):
        entity = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:")
        if "." in entity:
            continue
        lowered = entity.lower()
        first_token = lowered.split()[0] if lowered.split() else ""
        if first_token in {"another", "many", "millions", "some", "throughout", "different", "several", "various"}:
            continue
        if lowered in stop:
            continue
        before = source[:match.start()].rstrip()
        starts_sentence = not before or before[-1:] in {".", "!", "?", "\n"}
        single_token = len(entity.split()) == 1
        is_acronym = entity.isupper() and len(entity) > 1
        has_internal_capital = any(char.isupper() for char in entity[1:])
        if single_token and starts_sentence and not is_acronym and not has_internal_capital:
            continue
        key = lowered
        if key not in seen:
            values.append(entity)
            seen.add(key)
    for match in re.finditer(r"\b(?:\d{4}|\d+(?:\.\d+)?%?)\b", source):
        entity = match.group(0)
        if entity not in seen:
            values.append(entity)
            seen.add(entity)
    return values[:48]


def _paragraph_inventory_for_full_reconstruction(scan_report: dict | None, original_text: str) -> list[dict[str, Any]]:
    targets = _paragraph_target_map(scan_report, original_text)
    rows: list[dict[str, Any]] = []
    for paragraph_id in sorted(targets):
        paragraph = targets[paragraph_id]
        if not paragraph.strip():
            continue
        rows.append({
            "paragraph_id": paragraph_id,
            "required_entities": _required_entities_for_full_reconstruction(paragraph)[:10],
            "keywords": _keywords_from_target_text(paragraph)[:16],
            "approx_words": len(paragraph.split()),
        })
    return rows


def _entity_acronym(entity: str) -> str:
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9&'-]*\b", str(entity or ""))
    skip = {"the", "and", "of", "for", "in", "on", "at", "to"}
    letters = [word[0].upper() for word in words if word.lower() not in skip]
    return "".join(letters) if len(letters) >= 2 else ""


def _strip_rewrite_meta_text(text: str) -> str:
    value = clean_candidate_output(text)
    value = re.sub(r"^\s*(?:here(?:'s| is)[^\n]*|below is[^\n]*|rewritten (?:essay|version)[^\n]*)\n+", "", value, flags=re.I)
    if "---" in value:
        parts = [part.strip() for part in value.split("---") if part.strip()]
        if len(parts) >= 2:
            value = max(parts, key=len)
    value = _strip_generated_paragraph_labels(value)
    value = re.sub(r"\n+(?:changes made|kept your|notes?|explanation)\s*:\s*[\s\S]*$", "", value, flags=re.I).strip()
    return value


def _strip_generated_paragraph_labels(text: str) -> str:
    value = str(text or "")
    label_prefix = r"(?:#+\s*)?(?:\*\*)?Paragraph\s+\d+(?:\*\*)?\s*(?:[:.\-–](?:\*\*)?)?"
    value = re.sub(rf"(?im)^[^\S\n]*{label_prefix}[^\S\n]*\n+", "", value)
    value = re.sub(rf"(?im)^[^\S\n]*{label_prefix}[^\S\n]+", "", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def _restore_required_anchor_forms(candidate_text: str, required_entities: list[str]) -> str:
    text = _strip_rewrite_meta_text(candidate_text)
    for entity in required_entities:
        entity = str(entity or "").strip()
        if not entity:
            continue
        if re.search(rf"\b{re.escape(entity)}\b", text, flags=re.I):
            # Preserve exact casing/article form when the candidate only differs by case.
            text = re.sub(rf"\b{re.escape(entity)}\b", entity, text, count=1, flags=re.I)
            continue
        alternate = entity[4:] if entity.startswith("The ") else ""
        if alternate and re.search(rf"\b{re.escape(alternate)}\b", text, flags=re.I):
            text = re.sub(rf"\b{re.escape(alternate)}\b", entity, text, count=1, flags=re.I)
            continue
        acronym = _entity_acronym(entity)
        if acronym and re.search(rf"\b{re.escape(acronym)}\b", text):
            text = re.sub(rf"\b{re.escape(acronym)}\b", entity, text, count=1)
    return text.strip()


def _expected_full_reconstruction_paragraph_count(scan_report: dict | None, original_text: str) -> int:
    ids: set[str] = set()
    for brief in (scan_report or {}).get("rewrite_edit_briefs") or []:
        if isinstance(brief, dict) and str(brief.get("paragraph_id") or "").strip():
            ids.add(str(brief.get("paragraph_id")).strip())
    if ids:
        return len(ids)
    return _paragraph_count(original_text)


def _enrich_paragraph_brief_with_target(brief: dict[str, Any], paragraph_targets: dict[str, str]) -> dict[str, Any]:
    paragraph_id = str(brief.get("paragraph_id") or "")
    target = paragraph_targets.get(paragraph_id) or ""
    if not target:
        return brief
    enriched = {**brief}
    required_entities = list(enriched.get("required_entities") or [])
    for entity in _entities_from_target_text(target):
        if entity not in required_entities:
            required_entities.append(entity)
    context_keywords = list(enriched.get("context_keywords") or [])
    for keyword in _keywords_from_target_text(target):
        if keyword not in context_keywords:
            context_keywords.append(keyword)
    enriched["required_entities"] = required_entities[:24]
    enriched["context_keywords"] = context_keywords[:48]
    word_count = len(target.split())
    if word_count:
        enriched["target_word_range"] = {
            "min": max(35, int(word_count * 0.75)),
            "max": max(65, int(word_count * 1.15)),
        }
    return enriched


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        if sentence.strip()
    ]


def _cluster_text_from_gate(text: str, cluster: dict[str, Any]) -> str:
    sentences = _split_sentences(text)
    start = cluster.get("start_sentence")
    end = cluster.get("end_sentence")
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    start = max(0, start)
    end = min(len(sentences) - 1, end)
    if start > end:
        return ""
    return " ".join(sentences[start:end + 1]).strip()


def _replace_once_flexible(text: str, target: str, replacement: str) -> tuple[str, bool]:
    if not target or not replacement or target.strip() == replacement.strip():
        return text, False
    if target in text:
        return text.replace(target, replacement, 1), True
    pattern = re.sub(r"\\\s+", r"\\s+", re.escape(target.strip()))
    rewritten, count = re.subn(pattern, replacement.strip(), text, count=1)
    return rewritten, bool(count)


def _cluster_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "unsafe_cluster_rescue",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "replacement_text": {
                        "type": "string",
                        "description": "Replacement text for the exact unsafe cluster.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Short reason tied to reducing density, top-k predictability, and smoothness.",
                    },
                },
                "required": ["replacement_text", "rationale"],
                "additionalProperties": False,
            },
        },
    }


def _structured_json_request_options(model: str | None, response_format: dict[str, Any]) -> dict[str, Any]:
    return structured_json_request_options(model, response_format)


def _build_unsafe_cluster_rescue_prompt(
    *,
    cluster_text: str,
    cluster: dict[str, Any],
    goal: dict[str, Any] | None,
    variant: int,
) -> str:
    payload = {
        "variant": variant,
        "task": "Rewrite this unsafe detector cluster only.",
        "cluster": {
            "start_sentence": cluster.get("start_sentence"),
            "end_sentence": cluster.get("end_sentence"),
            "word_count": cluster.get("word_count"),
            "risk_score": cluster.get("risk_score"),
            "generic_hits": cluster.get("generic_hits"),
            "transition_count": cluster.get("transition_count"),
        },
        "remaining_blockers": {
            "texture_blockers": ((goal or {}).get("ai_footprint_gate") or {}).get("texture_blockers"),
            "remaining_ai_footprint_drivers": ((goal or {}).get("ai_footprint_gate") or {}).get("remaining_ai_footprint_drivers"),
            "turnitin_component_drops": ((goal or {}).get("turnitin_like_gate") or {}).get("component_drops"),
        },
        "unsafe_cluster_text": cluster_text,
    }
    return (
        "DraftProof unsafe-cluster rescue.\n"
        "Rewrite only the provided cluster text. Do not rewrite the full document.\n"
        "Goal: reduce remaining AI detector flags, especially top-k predictability, unsafe eligible density, and smoothness/bypasser texture.\n"
        "Use natural uneven student prose, not fragment lists and not polished encyclopedia prose.\n"
        "Keep paragraph meaning, names, dates, numbers, and claims. Do not add new facts, citations, examples, sources, or personal experience.\n"
        "Avoid generic transitions such as 'another important feature', 'at the same time', 'in conclusion', 'plays a major role', and 'one of the biggest'.\n"
        "Prefer concrete sentence routes, varied sentence openings, and a few moderate-length sentences mixed with shorter ones.\n"
        "Return valid JSON only with keys: replacement_text, rationale.\n\n"
        f"UNSAFE_CLUSTER_RESCUE_JSON:\n{json.dumps(payload, indent=2, default=str)}"
    )


def _generate_unsafe_cluster_rescue_candidates(
    *,
    frontier: dict[str, Any],
    original_text: str,
    original_report: dict[str, Any],
    reference_ai: float | None,
    required_ai_drop: float,
    target_ai_score: float | None,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    timeout_seconds: int,
    starting_cost: int,
    max_rows: int | None = None,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    if not api_key or not frontier:
        return []
    frontier_text = str(frontier.get("text") or "")
    frontier_report = frontier.get("report") if isinstance(frontier.get("report"), dict) else {}
    frontier_goal = frontier.get("goal") if isinstance(frontier.get("goal"), dict) else {}
    density = (frontier_goal.get("eligible_span_density_gate") or {})
    clusters = [
        cluster for cluster in (density.get("top_unsafe_clusters") or [])
        if isinstance(cluster, dict)
    ][:max(1, int(os.environ.get("DRAFTPROOF_REWRITE_V2_RESCUE_CLUSTERS", "1") or 1))]
    if not frontier_text or not clusters:
        return []
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model or os.environ.get("LLM_MODEL") or "openai/gpt-4.1-mini",
        base_url=base_url or os.environ.get("LLM_BASE_URL", ""),
        timeout=timeout_seconds,
        max_retries=1,
        max_tokens=3000,
        temperature=0.55,
    ))
    protected = detect_protected_spans(original_text)
    rows: list[dict[str, Any]] = []
    variants = max(1, min(4, int(os.environ.get("DRAFTPROOF_REWRITE_V2_RESCUE_CANDIDATES", "2") or 2)))
    original_smoothness = _rewrite_smoothness(original_report)
    max_smoothness_regression = float(os.environ.get("DRAFTPROOF_REWRITE_V2_MAX_SMOOTHNESS_REGRESSION", "4.0") or 4.0)
    for cluster_index, cluster in enumerate(clusters, start=1):
        if max_rows is not None and len(rows) >= max_rows:
            break
        cluster_text = _cluster_text_from_gate(frontier_text, cluster)
        if not cluster_text:
            continue
        for variant in range(1, variants + 1):
            if max_rows is not None and len(rows) >= max_rows:
                break
            if deadline is not None and time.time() + timeout_seconds + 2.0 >= deadline:
                break
            prompt = _build_unsafe_cluster_rescue_prompt(
                cluster_text=cluster_text,
                cluster=cluster,
                goal=frontier_goal,
                variant=variant,
            )
            structured_options = _structured_json_request_options(model, _cluster_response_format())
            response = gateway.chat(
                prompt,
                system="You are DraftProof's unsafe cluster rescue engine.",
                max_tokens=2200,
                temperature=0.62,
                top_p=0.9,
                presence_penalty=0.25 if _supports_openai_penalties(model) else None,
                frequency_penalty=0.35 if _supports_openai_penalties(model) else None,
                repetition_penalty=1.05 if _supports_repetition_penalty(model) else None,
                seed=2300 + (cluster_index * 10) + variant,
                response_format=structured_options["response_format"],
                provider=structured_options["provider"],
            )
            parse_diagnostics = _json_parse_diagnostics(response.content)
            payload = parse_diagnostics["payload"]
            replacement = str(payload.get("replacement_text") or "").strip()
            rescue_text, applied = _replace_once_flexible(frontier_text, cluster_text, replacement)
            if not applied:
                rows.append({
                    "strategy": "unsafe_cluster_rescue",
                    "strategy_kind": "unsafe_cluster_rescue",
                    "candidate_number": variant,
                    "cluster_index": cluster_index,
                    "candidate_ai": None,
                    "candidate_response": payload,
                    "decision": {
                        "lane": CandidateLane.REJECT.value,
                        "reason": "unsafe_cluster_target_not_found",
                        "rank": [],
                    },
                })
                continue
            filter_failures = _patch_filter_failures([{
                "target_paragraph": cluster_text,
                "rewritten_paragraph": replacement,
            }])
            if filter_failures:
                rows.append({
                    "strategy": "unsafe_cluster_rescue",
                    "strategy_kind": "unsafe_cluster_rescue",
                    "candidate_number": variant,
                    "cluster_index": cluster_index,
                    "candidate_ai": None,
                    "candidate_response": payload,
                    "local_filter_failures": filter_failures,
                    "decision": {
                        "lane": CandidateLane.REJECT.value,
                        "reason": "unsafe_cluster_local_filter_rejected",
                        "rank": [],
                    },
                })
                continue
            semantic = check_semantic_drift(original_text, rescue_text, threshold=0.15)
            anchors_safe = protected_spans_preserved(original_text, rescue_text, protected)
            if not anchors_safe or not semantic.accepted:
                rows.append({
                    "strategy": "unsafe_cluster_rescue",
                    "strategy_kind": "unsafe_cluster_rescue",
                    "candidate_number": variant,
                    "cluster_index": cluster_index,
                    "decision": {
                        "lane": CandidateLane.REJECT.value,
                        "reason": "unsafe_cluster_semantic_or_anchor_rejected",
                        "rank": [],
                    },
                    "semantic_safe": bool(semantic.accepted),
                    "protected_anchors_safe": bool(anchors_safe),
                    "semantic_similarity": getattr(semantic, "similarity", None),
                    "semantic_reasons": getattr(semantic, "reasons", None),
                })
                continue
            candidate_report = _scan_report(rescue_text)
            goal = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=rescue_text,
                original_report=original_report,
                candidate_report=candidate_report,
            )
            decision = decide_candidate(
                goal=goal,
                original_report=original_report,
                candidate_report=candidate_report,
                reference_ai=reference_ai,
                required_ai_drop=required_ai_drop,
                target_ai_score=target_ai_score,
                semantic_safe=bool(semantic.accepted),
                quality_safe=anchors_safe,
                cost=starting_cost + len(rows) + 1,
            ).to_dict()
            smoothness = _rewrite_smoothness(candidate_report)
            smoothness_regression = (
                round(float(smoothness) - float(original_smoothness), 3)
                if isinstance(smoothness, (int, float)) and isinstance(original_smoothness, (int, float))
                else None
            )
            if (
                isinstance(smoothness_regression, (int, float))
                and smoothness_regression > max_smoothness_regression
                and decision.get("lane") != CandidateLane.GOAL_MET.value
            ):
                decision = {
                    **decision,
                    "lane": CandidateLane.REJECT.value,
                    "reason": "unsafe_cluster_smoothness_regression_rejected",
                }
            rows.append({
                "strategy": "unsafe_cluster_rescue",
                "strategy_kind": "unsafe_cluster_rescue",
                "candidate_number": variant,
                "cluster_index": cluster_index,
                "cluster": cluster,
                "cluster_text": cluster_text,
                "replacement_text": replacement,
                "candidate_response": payload,
                "structured_output_mode": structured_options["structured_output_mode"],
                "structured_output_parse": {
                    key: value
                    for key, value in parse_diagnostics.items()
                    if key != "payload"
                },
                "candidate_ai": _badge_ai(candidate_report),
                "candidate_wq": _badge_wq(candidate_report),
                "rewrite_smoothness": smoothness,
                "smoothness_regression": smoothness_regression,
                "goal": goal.to_dict(),
                "decision": decision,
                "semantic_safe": bool(semantic.accepted),
                "protected_anchors_safe": bool(anchors_safe),
                "semantic_similarity": getattr(semantic, "similarity", None),
                "semantic_reasons": getattr(semantic, "reasons", None),
                "report": candidate_report,
                "text": rescue_text,
            })
    return rows


def _json_parse_diagnostics(raw: str) -> dict[str, Any]:
    return json_parse_diagnostics(raw)


def _json_from_response(raw: str) -> dict[str, Any]:
    return json_from_response(raw)



def _apply_targeted_patches(original_text: str, payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text = original_text
    applied: list[dict[str, Any]] = []
    patches = payload.get("patches") if isinstance(payload, dict) else []
    if not isinstance(patches, list):
        return original_text, applied
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        target = str(patch.get("target_paragraph") or patch.get("target_sentence") or "").strip()
        replacement = str(patch.get("rewritten_paragraph") or patch.get("rewritten_sentence") or "").strip()
        if not target or not replacement or target == replacement:
            continue
        if target not in text:
            applied.append({
                "finding_id": patch.get("finding_id"),
                "finding_ids": patch.get("finding_ids"),
                "paragraph_id": patch.get("paragraph_id"),
                "applied": False,
                "reason": "target_paragraph_not_found",
                "target_paragraph": target,
                "rewritten_paragraph": replacement,
            })
            continue
        text = text.replace(target, replacement, 1)
        applied.append({
            "finding_id": patch.get("finding_id"),
            "finding_ids": patch.get("finding_ids"),
            "paragraph_id": patch.get("paragraph_id"),
            "applied": True,
            "target_paragraph": target,
            "rewritten_paragraph": replacement,
            "rationale": patch.get("rationale"),
        })
    return text, applied


def _targeted_candidate_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if isinstance(candidates, list) and candidates:
        return [row for row in candidates if isinstance(row, dict)]
    patches = payload.get("patches") if isinstance(payload, dict) else None
    if isinstance(patches, list):
        return [{"candidate_id": "variant_1", "patches": patches}]
    return []


def _surface_quality_failures(text: str) -> list[str]:
    failures: list[str] = []
    sentences = _split_sentences(text)
    if not sentences:
        return ["empty_sentence_sequence"]
    word_counts = [len(re.findall(r"\b[\w'-]+\b", sentence)) for sentence in sentences]
    if len(sentences) >= 3:
        short_count = sum(1 for count in word_counts if count <= 4)
        if short_count / len(sentences) > 0.2:
            failures.append("fragment_sentence_ratio_high")
    if sum(1 for count in word_counts if count <= 2) >= 2:
        failures.append("one_or_two_word_sentence_count_high")
    fragment_openers = {"from", "in", "between", "through", "across", "after", "before", "with", "without", "like", "also"}
    opener_hits = 0
    for sentence in sentences:
        first = (re.findall(r"\b[A-Za-z']+\b", sentence.lower()) or [""])[0]
        if first in fragment_openers and len(re.findall(r"\b[\w'-]+\b", sentence)) <= 8:
            opener_hits += 1
    if opener_hits:
        failures.append("prepositional_fragment_sentence")
    return failures


def _patch_filter_failures(patches: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    banned = [
        "is often described",
        "often described as",
        "one of the most",
        "in modern history",
        "highly influential",
        "key role",
        "played a key role",
        "shaping modern history",
        "significantly affected",
        "influence has extended",
        "worldwide",
        "for decades",
        "emerging from",
        "stands as",
        "significant global entity",
        "various sectors",
        "notable over time",
        "swiftly ascended",
        "melting pot",
        "grapples with",
        "pressing",
        "complex relationship",
        "complex interplay",
        "deeply intertwined",
        "multifaceted",
        "dynamic landscape",
        "continues to shape",
        "evaluating",
        "judging",
        "coexist",
        "persistent challenges",
    ]
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        target = str(patch.get("target_paragraph") or patch.get("target_sentence") or "").strip()
        rewrite = str(patch.get("rewritten_paragraph") or patch.get("rewritten_sentence") or "").strip()
        lowered = rewrite.lower()
        if not rewrite:
            failures.append("empty_rewrite")
            continue
        if target and rewrite == target:
            failures.append("unchanged_target_sentence")
        for failure in _surface_quality_failures(rewrite):
            failures.append(f"surface_quality:{failure}")
        for phrase in banned:
            if phrase in lowered:
                failures.append(f"banned_phrase:{phrase}")
    return failures


def _paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", str(text or "")) if part.strip()])


def _has_rewrite_meta_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    return bool(
        re.search(r"^\s*(?:here(?:'s| is)\b|below is\b|rewritten (?:essay|version)\b|this version\b)", value, re.I)
        or re.search(r"\n\s*(?:changes made|notes?|explanation)\s*:\s*", value, re.I)
        or re.search(r"\blet me know\b", value, re.I)
        or re.search(r"^\s*```", value)
    )


def _full_reconstruction_filter_failures(
    *,
    original_text: str,
    candidate_text: str,
    required_entities: list[str],
    expected_paragraph_count: int | None = None,
) -> list[str]:
    failures: list[str] = []
    if not candidate_text.strip():
        return ["empty_rewrite"]
    original_paragraphs = expected_paragraph_count or _paragraph_count(original_text)
    candidate_paragraphs = _paragraph_count(candidate_text)
    if original_paragraphs and candidate_paragraphs != original_paragraphs:
        failures.append(f"paragraph_count:{candidate_paragraphs}_expected:{original_paragraphs}")
    lowered = candidate_text.lower()
    if _has_rewrite_meta_text(candidate_text):
        failures.append("meta_text_leak")
    if os.environ.get("DRAFTPROOF_REWRITE_V2_REJECT_SURVEY_FULL_RECONSTRUCTION", "1").lower() not in {"0", "false", "no"}:
        failures.extend(_survey_style_failures(candidate_text))
    for entity in required_entities:
        alternate = entity[4:] if entity.startswith("The ") else ""
        if entity and entity not in candidate_text and (not alternate or alternate not in candidate_text):
            failures.append(f"missing_required_entity:{entity}")
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", candidate_text) if part.strip()]:
        for failure in _surface_quality_failures(paragraph):
            failures.append(f"surface_quality:{failure}")
    return failures


def _author_stance_thesis_filter_failures(
    *,
    candidate_text: str,
    required_entities: list[str],
    min_paragraphs: int = 4,
    max_paragraphs: int = 6,
    require_author_stance_marker: bool = True,
    reject_survey_style: bool = False,
) -> list[str]:
    failures: list[str] = []
    if not candidate_text.strip():
        return ["empty_rewrite"]
    paragraph_count = _paragraph_count(candidate_text)
    if paragraph_count < min_paragraphs or paragraph_count > max_paragraphs:
        failures.append(f"paragraph_count:{paragraph_count}_expected_between:{min_paragraphs}_{max_paragraphs}")
    if _has_rewrite_meta_text(candidate_text):
        failures.append("meta_text_leak")
    missing_entities: list[str] = []
    for entity in required_entities:
        alternate = entity[4:] if entity.startswith("The ") else ""
        if entity and entity not in candidate_text and (not alternate or alternate not in candidate_text):
            missing_entities.append(entity)
    if required_entities:
        coverage = (len(required_entities) - len(missing_entities)) / len(required_entities)
        if coverage < 0.8:
            failures.append(f"required_entity_coverage:{coverage:.2f}_missing:{len(missing_entities)}")
    lowered = candidate_text.lower()
    if require_author_stance_marker and not re.search(r"\b(i think|i find|i do not|i don't|i see|i would)\b", lowered):
        failures.append("missing_author_stance_marker")
    if reject_survey_style:
        failures.extend(_survey_style_failures(candidate_text))
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", candidate_text) if part.strip()]:
        for failure in _surface_quality_failures(paragraph):
            failures.append(f"surface_quality:{failure}")
    return failures


def _survey_style_failures(text: str) -> list[str]:
    failures: list[str] = []
    banned_openers = {
        "economically",
        "culturally",
        "globally",
        "internationally",
        "technologically",
        "politically",
        "socially",
    }
    for paragraph in [part.strip() for part in re.split(r"\n\s*\n", str(text or "")) if part.strip()]:
        first_words = " ".join(re.findall(r"\b[A-Za-z']+\b", paragraph.lower())[:4])
        first_word = first_words.split(" ")[0] if first_words else ""
        if first_word in banned_openers:
            failures.append(f"survey_opening:{first_word}")
        if first_words.startswith("on the global") or first_words.startswith("in conclusion"):
            failures.append(f"survey_opening:{first_words}")
    lowered = str(text or "").lower()
    banned_phrases = [
        "undeniably powerful",
        "undeniably strong",
        "influence is undeniable",
        "wields immense influence",
        "massive impact",
        "global trends",
        "stabilizing force",
        "layer of complexity",
        "unresolved challenges",
        "internal tensions",
        "powerhouse",
        "melting pot",
        "beacon of opportunity",
        "remarkable achievements",
        "significant contradictions",
        "significant challenges",
        "dominant role",
        "complex and influential",
        "complex force",
        "shaped the world",
        "global affairs",
        "groundbreaking technology",
        "purely successful or flawed",
        "pivotal moment",
        "coexists with",
        "this duality",
    ]
    for phrase in banned_phrases:
        if phrase in lowered:
            failures.append(f"survey_phrase:{phrase}")
    return failures


def _required_anchor_coverage(candidate_text: str, required_entities: list[str]) -> float:
    if not required_entities:
        return 1.0
    preserved = 0
    for entity in required_entities:
        entity = str(entity or "").strip()
        if not entity:
            continue
        alternate = entity[4:] if entity.startswith("The ") else ""
        if entity in candidate_text or (alternate and alternate in candidate_text):
            preserved += 1
    return preserved / len([entity for entity in required_entities if str(entity or "").strip()] or [""])


def _author_strategy_semantic_override_allowed(
    *,
    strategy_kind: str,
    generated_candidate: dict[str, Any],
    candidate_text: str,
    semantic_similarity: float | None,
    anchors_safe: bool,
) -> bool:
    if strategy_kind not in {"author_stance_thesis_reframe", "author_stance_texture_pass"}:
        return False
    if not anchors_safe:
        return False
    if not isinstance(semantic_similarity, (int, float)) or float(semantic_similarity) < 0.75:
        return False
    required_entities = generated_candidate.get("required_entities")
    if not isinstance(required_entities, list):
        required_entities = []
    return _required_anchor_coverage(candidate_text, required_entities) >= 0.85


def _academic_contract_semantic_override_allowed(
    *,
    strategy_kind: str,
    original_text: str,
    original_report: dict[str, Any],
    candidate_text: str,
    semantic_similarity: float | None,
    anchors_safe: bool,
    semantic_reasons: list[str] | None,
) -> bool:
    if not str(strategy_kind or "").startswith("academic_"):
        return False
    if not anchors_safe:
        return False
    if not isinstance(semantic_similarity, (int, float)) or float(semantic_similarity) < 0.45:
        return False
    reasons = [str(reason or "") for reason in (semantic_reasons or []) if str(reason or "").strip()]
    if not reasons or any(not reason.startswith("lost_named_entity:") for reason in reasons):
        return False
    sections = _academic_assignment_sections(original_text, original_report)
    contract = build_rewrite_contract(
        original_text,
        content_mode="academic_cited_text",
        sections=sections or [{"section_id": "document", "text": original_text}],
    )
    required = [
        anchor for anchor in contract.anchors
        if anchor.severity in {AnchorSeverity.HARD_EXACT, AnchorSeverity.HARD_NORMALIZED, AnchorSeverity.SOFT_REQUIRED}
    ]
    missing_required = [anchor.text for anchor in required if not anchor_present(anchor, candidate_text)]
    return not missing_required


def _attach_hidden_paragraph_targets(candidate_payload: dict[str, Any], target_map: dict[str, str]) -> dict[str, Any]:
    patches = candidate_payload.get("patches")
    if not isinstance(patches, list):
        return candidate_payload
    enriched = {**candidate_payload, "patches": []}
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        row = dict(patch)
        paragraph_id = str(row.get("paragraph_id") or "").strip()
        if paragraph_id and not row.get("target_paragraph") and paragraph_id in target_map:
            row["target_paragraph"] = target_map[paragraph_id]
        enriched["patches"].append(row)
    return enriched


def _paragraph_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "paragraph_reconstruction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "paragraph_id": {
                        "type": "string",
                        "description": "The paragraph id from the input brief.",
                    },
                    "rewritten_paragraph": {
                        "type": "string",
                        "description": "Replacement paragraph only, no commentary.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Short reason tied to rhythm, predictable spans, and preservation.",
                    },
                },
                "required": ["paragraph_id", "rewritten_paragraph", "rationale"],
                "additionalProperties": False,
            },
        },
    }


def _targeted_batch_response_format() -> dict[str, Any]:
    patch_schema = {
        "type": "object",
        "properties": {
            "paragraph_id": {"type": "string"},
            "rewritten_paragraph": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["paragraph_id", "rewritten_paragraph", "rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "targeted_paragraph_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "string"},
                                "patches": {"type": "array", "items": patch_schema},
                            },
                            "required": ["candidate_id", "patches"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["candidates"],
                "additionalProperties": False,
            },
        },
    }


def _build_targeted_batch_prompt(briefs: list[dict[str, Any]], strategy: Any, tactics: list[str]) -> str:
    payload = {
        "strategy": strategy.to_dict() if hasattr(strategy, "to_dict") else {},
        "task": "Create candidate patch sets for all paragraph briefs.",
        "candidate_count": max(1, len(tactics)),
        "candidate_tactics": tactics,
        "required_patch_count_per_candidate": len(briefs),
        "paragraph_rewrite_briefs": briefs,
        "output_schema": {
            "candidates": [{
                "candidate_id": "variant_1",
                "patches": [{
                    "paragraph_id": "string",
                    "rewritten_paragraph": "replacement paragraph only",
                    "rationale": "short reason tied to rhythm, predictable spans, and preservation",
                }],
            }]
        },
    }
    return (
        "DraftProof targeted paragraph batch reconstruction.\n"
        f"Create exactly {max(1, len(tactics))} candidate patch sets.\n"
        f"Each candidate must include exactly {len(briefs)} patches, one for every paragraph_brief.\n"
        "Regenerate each paragraph from structured context only; original paragraph prose is intentionally not provided.\n"
        "Every required entity in each paragraph brief must appear verbatim in that paragraph's rewrite.\n"
        "Break predictable token paths, repeated openings, transition rhythm, and paragraph-level uniformity.\n"
        "Use the candidate_tactics to make the variants meaningfully different.\n"
        "Keep plain student analytical prose. Avoid ornate, promotional, encyclopedic, or marketing language.\n"
        "Use complete sentences. Do not create sentence fragments or one-word list sentences.\n"
        "Keep most sentences between 8 and 24 words unless a required entity forces a longer sentence.\n"
        "Do not add unsupported facts, examples, citations, author experience, or commentary.\n"
        "Return valid JSON only, matching the schema.\n\n"
        f"TARGETED_BATCH_JSON:\n{json.dumps(payload, indent=2, default=str)[:30000]}"
    )


def _build_entity_locked_full_reconstruction_prompt(
    *,
    original_text: str,
    required_entities: list[str],
    paragraph_inventory: list[dict[str, Any]],
    expected_paragraph_count: int,
    variant: int,
) -> str:
    variant_instruction = {
        1: "Keep the original ten paragraph topics in order. Remove broad filler and make examples do more work.",
        2: "Rewrite from structured memory rather than sentence-by-sentence paraphrase. Keep enough detail from each original paragraph.",
    }.get(variant, "Prefer grounded, limited claims over sweeping claims while preserving the original topics.")
    return (
        "Output only the rewritten essay. No title, no notes, no markdown, no explanation.\n"
        f"Return exactly {expected_paragraph_count} paragraphs, separated by blank lines. Do not merge paragraph topics.\n"
        "Treat the original as a fact inventory, not prose to paraphrase sentence by sentence.\n"
        "Use the original facts only. Do not add new named people, places, organizations, dates, statistics, sources, or events.\n"
        f"Every required entity must appear at least once: {json.dumps(required_entities, ensure_ascii=False)}\n"
        f"Follow this paragraph inventory in order: {json.dumps(paragraph_inventory, ensure_ascii=False)}\n"
        "Use complete sentences only. No fragments. No bullet lists.\n"
        "Avoid these phrases: often described, one of the most, in modern history, key role, significant, significantly, complex and influential, shaped the modern world, at the same time, land of opportunity, cultural powerhouse.\n"
        "Use varied paragraph openings. Avoid starting most paragraphs with \"The United States\".\n"
        "Rebuild each paragraph with 2 to 4 sentences. Prefer concrete noun-verb sentences over broad summary claims.\n"
        "Change sentence openings, clause order, and rhythm across the essay while preserving meaning.\n"
        "Keep a careful student essay tone: plain, specific, readable, not polished marketing copy.\n"
        f"{variant_instruction}\n\n"
        f"ORIGINAL:\n{original_text}"
    )


def _build_keyword_locked_short_texture_prompt(
    *,
    required_entities: list[str],
    paragraph_inventory: list[dict[str, Any]],
    expected_paragraph_count: int,
    original_text: str,
) -> str:
    global_keywords = _keywords_from_target_text(original_text)[:55]
    return (
        "Output only the rewritten essay. No title, notes, markdown, bullets, numbering, or explanation.\n"
        f"Return exactly {expected_paragraph_count} paragraphs separated by blank lines.\n"
        "Generate from structured anchors, not from original sentence phrasing.\n"
        f"Preserve required entities and numbers exactly where natural: {json.dumps(required_entities, ensure_ascii=False)}\n"
        f"Use this paragraph inventory in order: {json.dumps(paragraph_inventory, ensure_ascii=False)}\n"
        f"Keep these topic keywords represented naturally across the essay: {json.dumps(global_keywords, ensure_ascii=False)}\n"
        "Use 2 to 3 complete sentences per paragraph. Keep semantic coverage, but avoid long generic explanation chains.\n"
        "Use varied sentence rhythm and concrete subjects. Do not add new facts, examples, named entities, citations, dates, or events.\n"
        "Avoid stock essay phrases such as in conclusion, often described, one of the most, key role, shaped the modern world, and significant.\n"
        "The result should read like concise student prose, not notes and not a marketing summary."
    )


def _build_author_stance_thesis_reframe_prompt(
    *,
    original_text: str,
    required_entities: list[str],
    paragraph_inventory: list[dict[str, Any]],
    target_paragraph_count: int = 4,
    variant: int = 1,
) -> str:
    global_keywords = _keywords_from_target_text(original_text)[:45]
    variant_instruction = {
        1: "Use a skeptical student voice with clear judgment lines. Keep the argument direct.",
        2: "Start paragraphs with concrete claims, not category labels. Use shorter sentences where the point is simple.",
        3: "Make the essay feel like a writer thinking through a contradiction, not covering topics evenly.",
    }.get(variant, "Prioritize plain judgment and uneven rhythm over balanced survey coverage.")
    return (
        "Output only the rewritten essay. No title, notes, markdown, bullets, numbering, or explanation.\n"
        f"Rewrite as a narrow thesis-driven essay in {target_paragraph_count} paragraphs separated by blank lines.\n"
        "Use first-person analytical stance where natural, such as I think, I find, or I do not see. Do not invent personal experience.\n"
        "Do not preserve the broad one-topic-per-paragraph survey shape. Merge related topics and prioritize an argument.\n"
        "Core thesis pattern: this subject is difficult to judge cleanly because its power or importance comes from several places and has contradictions.\n"
        f"Preserve these required anchors exactly where relevant: {json.dumps(required_entities, ensure_ascii=False)}\n"
        f"Use this source inventory for factual coverage, but do not copy its structure: {json.dumps(paragraph_inventory, ensure_ascii=False)}\n"
        f"Represent these topic keywords naturally, without forcing all of them: {json.dumps(global_keywords, ensure_ascii=False)}\n"
        "Use only source facts. Do not add new named people, places, organizations, dates, statistics, examples, events, or citations.\n"
        "Do not open paragraphs with category labels such as Economically, Culturally, Technologically, Politically, or On the global stage.\n"
        "Avoid polished survey phrases and academic nouns such as in conclusion, global impact, key role, significant influence, shaped the modern world, complex and influential, global influence is undeniable, powerhouse, melting pot, beacon of opportunity, systemic issues, narrative of progress, central to understanding, exerts immense influence, deep-seated, coexists, interventionism, societal concerns, duality, remarkable achievements, significant contradictions, significant challenges, global affairs, and defying simple judgment.\n"
        "Use plain judgment lines instead of formal topic sentences. Prefer sentences like: That picture is incomplete. The harder question is who benefits from it. I do not see that as a simple success story.\n"
        "Keep the voice readable, slightly uneven, and mildly opinionated, not encyclopedic, not balanced category coverage, and not over-compressed.\n\n"
        f"{variant_instruction}\n\n"
        f"SOURCE:\n{original_text}"
    )


def _build_author_stance_texture_pass_prompt(
    *,
    source_text: str,
    draft_text: str,
    required_entities: list[str],
    target_paragraph_count: int = 4,
) -> str:
    return (
        "Output only the revised essay. No title, notes, markdown, bullets, numbering, or explanation.\n"
        f"Keep exactly {target_paragraph_count} paragraphs separated by blank lines.\n"
        "This is a texture pass, not a new essay. Preserve the same facts, claims, sequence of ideas, and overall stance.\n"
        f"Keep required anchors where natural and preserve high anchor coverage: {json.dumps(required_entities, ensure_ascii=False)}\n"
        "Use only facts already present in SOURCE or CURRENT_DRAFT. Do not add new named people, places, organizations, dates, statistics, examples, events, or citations.\n"
        "Hard rule: no paragraph may begin with a category label or scope label. Banned paragraph openings include Economically, Culturally, Technologically, Politically, Globally, Internationally, On the global stage, and In conclusion.\n"
        "Use natural claim openings instead. Paragraph 1 should open with a judgment by the writer. Paragraph 2 should open with a concrete contrast. Paragraph 3 should open with public image or culture as part of the argument, not as a category label. Paragraph 4 should open with power outside the subject or wider consequences, not a scope label.\n"
        "Replace polished survey wording with plain judgment. Avoid phrases such as undeniably powerful, undeniably strong, wields immense influence, massive impact, global trends, stabilizing force, layer of complexity, unresolved challenges, internal tensions, powerhouse, remarkable achievements, significant contradictions, and dominant role.\n"
        "Vary sentence length naturally. Include 2 to 3 short complete judgment sentences where they fit. Do not create fragments or choppy notes.\n"
        "Keep first-person analytical stance where natural. Do not invent personal experience or pretend to have lived events.\n"
        "The result should sound like a careful student revising their own draft, not a balanced encyclopedia summary.\n\n"
        f"SOURCE:\n{source_text}\n\n"
        f"CURRENT_DRAFT:\n{draft_text}"
    )


def _supports_openai_penalties(model: str | None) -> bool:
    return model_supports_presence_frequency_penalties(model)


def _supports_repetition_penalty(model: str | None) -> bool:
    return model_supports_repetition_penalty(model)


def _author_stance_target_paragraph_count() -> int:
    raw = os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_PARAGRAPHS", "4")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 4
    return max(3, min(6, value))


def _paragraph_tactics() -> list[str]:
    raw = os.environ.get(
        "DRAFTPROOF_REWRITE_V2_TACTICS",
        "minimal_carrier,compressed_power,choppy_analytic,simple_subject_stack,specific_noun_action",
    )
    tactics = [item.strip() for item in raw.split(",") if item.strip()]
    if os.environ.get("DRAFTPROOF_REWRITE_V2_ALLOW_FRAGMENT_TACTICS", "0").lower() not in {"1", "true", "yes"}:
        tactics = [item for item in tactics if item not in {"broken_choppy"}]
    return tactics or ["plain_student_draft"]


def _topk_calibrated_risk(report: dict | None) -> float:
    value = (((report or {}).get("ai_risk_badge") or {}).get("ai_components") or {}).get("topk_calibrated_risk")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _density_risk(report: dict | None) -> float:
    value = (((report or {}).get("ai_risk_badge") or {}).get("ai_components") or {}).get("qualifying_text_ai_density")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _should_entity_locked_full_reconstruction(scan_report: dict | None) -> bool:
    if os.environ.get("DRAFTPROOF_REWRITE_V2_ENTITY_LOCKED_FULL_RECONSTRUCTION", "1").lower() in {"0", "false", "no"}:
        return False
    return _topk_calibrated_risk(scan_report) >= 90.0 or _density_risk(scan_report) >= 70.0


def _use_paragraph_local_score_gate() -> bool:
    return os.environ.get("DRAFTPROOF_REWRITE_V2_USE_PARAGRAPH_LOCAL_SCORE", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _sentence_comparison(original_text: str, final_text: str) -> list[dict[str, Any]]:
    return [{
        "index": 1,
        "original": original_text,
        "rewritten": final_text,
        "changed": original_text.strip() != final_text.strip(),
    }]


def _candidate_rows_from_replay(
    replay_candidate_records: list[dict[str, Any]],
    *,
    original_text: str,
    original_report: dict[str, Any],
    required_ai_drop: float,
    target_ai_score: float | None,
) -> list[dict[str, Any]]:
    rows = []
    reference_ai = _badge_ai(original_report)
    for index, record in enumerate(replay_candidate_records, start=1):
        candidate_report = record.get("report") if isinstance(record.get("report"), dict) else dict(record)
        if isinstance(record.get("ai"), (int, float)) and "ai_risk_badge" not in candidate_report:
            candidate_report = {
                **candidate_report,
                "ai_risk_badge": {
                    "ai_likelihood_score": record.get("ai"),
                    "writing_quality_score": record.get("writing_quality"),
                    "ai_components": (
                        ((record.get("ai_footprint_gate") or {}).get("after") or {}).get("authorship_footprint")
                        or {}
                    ),
                },
            }
        candidate_text = str(record.get("text") or original_text)
        if isinstance(record.get("ai_footprint_gate"), dict) and isinstance(record.get("turnitin_like_ai_gate"), dict):
            density_gate = record.get("eligible_span_density_gate") if isinstance(record.get("eligible_span_density_gate"), dict) else {}
            strict_safe = bool(record["ai_footprint_gate"].get("safe_band"))
            turnitin_target = bool(
                record["turnitin_like_ai_gate"].get("target_met")
                or record["turnitin_like_ai_gate"].get("safe_band")
            )
            density_safe = bool(density_gate.get("safe"))
            detector_safe = bool(strict_safe and turnitin_target and density_safe)
            goal = RewriteGoalEvaluation(
                status=RewriteGoalStatus.AI_MITIGATED if detector_safe else RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE,
                goal_met=detector_safe,
                detector_safe=detector_safe,
                strict_ai_safe_band_achieved=strict_safe,
                turnitin_like_target_met=turnitin_target,
                eligible_span_density_safe=density_safe,
                reason="replay_candidate_goal_met" if detector_safe else "replay_candidate_failed_strict_goal",
                ai_footprint_gate=record["ai_footprint_gate"],
                turnitin_like_gate=record["turnitin_like_ai_gate"],
                eligible_span_density_gate=density_gate,
            )
        elif "report" in record:
            goal = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=candidate_text,
                original_report=original_report,
                candidate_report=candidate_report,
            )
        else:
            goal = RewriteGoalEvaluation(
                status=RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE,
                goal_met=False,
                detector_safe=False,
                strict_ai_safe_band_achieved=False,
                turnitin_like_target_met=False,
                eligible_span_density_safe=False,
                reason="replay_record_without_rescan_report_failed_strict_goal",
                ai_footprint_gate={},
                turnitin_like_gate={},
                eligible_span_density_gate={},
            )
        decision = decide_candidate(
            goal=goal,
            original_report=original_report,
            candidate_report=candidate_report,
            reference_ai=reference_ai,
            required_ai_drop=required_ai_drop,
            target_ai_score=target_ai_score,
            semantic_safe=True,
            quality_safe=True,
            cost=index,
        )
        if (
            "report" not in record
            and not isinstance(record.get("ai_footprint_gate"), dict)
            and not isinstance(record.get("turnitin_like_ai_gate"), dict)
        ):
            decision_payload = {
                "lane": CandidateLane.REJECT.value,
                "selected_as_success": False,
                "goal_met": False,
                "ai_target_gap": None,
                "required_drop_met": False,
                "quality_safe": False,
                "semantic_safe": True,
                "reason": "replay_record_missing_rescan_gates",
                "rank": [],
            }
        else:
            decision_payload = decision.to_dict()
        rows.append({
            "strategy": record.get("strategy") or f"replay_candidate_{index}",
            "strategy_kind": record.get("strategy_kind"),
            "paragraph_id": record.get("paragraph_id"),
            "candidate_number": record.get("candidate_number"),
            "tactic": record.get("tactic"),
            "candidate_ai": _badge_ai(candidate_report),
            "candidate_wq": _badge_wq(candidate_report),
            "composed_patches": record.get("composed_patches"),
            "applied_patch_count": record.get("applied_patch_count"),
            "patch_count": record.get("patch_count"),
            "patches": record.get("patches"),
            "goal": goal.to_dict(),
            "decision": decision_payload,
            "report": candidate_report,
            "text": candidate_text,
        })
    return rows


def _generate_candidates(
    *,
    original_text: str,
    scan_report: dict[str, Any],
    strategies: list[Any],
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    timeout_seconds: int,
    deadline: float | None = None,
    content_route: Any | None = None,
    layer_attempts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not api_key:
        record_layer_attempt(
            layer_attempts,
            layer="llm_gateway",
            status="blocked",
            reason="missing_api_key",
            allowed=False,
            applicable=False,
        )
        return []
    paragraph_targets = _paragraph_target_map(scan_report, original_text)
    gateway = LLMGateway(LLMConfig(
        api_key=api_key,
        model=model or os.environ.get("LLM_MODEL") or "openai/gpt-4.1-mini",
        base_url=base_url or os.environ.get("LLM_BASE_URL", ""),
        timeout=timeout_seconds,
        max_retries=1,
        max_tokens=6500,
        temperature=0.45,
    ))
    candidates = []
    full_reconstruction_allowed = _strategy_family_allowed(content_route, "entity_locked_full_reconstruction")
    author_stance_allowed = _strategy_family_allowed(content_route, "author_stance_thesis_reframe")
    keyword_texture_allowed = _strategy_family_allowed(content_route, "keyword_locked_short_texture")
    author_texture_allowed = _strategy_family_allowed(content_route, "author_stance_texture_pass")
    limits = portfolio_limits(content_route)
    layer_caps = limits.get("layer_candidate_caps") or {}
    max_generated_candidates = int(limits.get("max_generated_candidates") or 8)

    def record_attempt(layer: str, status: str, reason: str, before: int, applicable: bool | None = None, allowed: bool | None = True) -> None:
        record_layer_attempt(
            layer_attempts,
            layer=layer,
            status=status,
            reason=reason,
            allowed=allowed,
            applicable=applicable,
            generated_count=max(0, len(candidates) - before),
            candidate_count_before=before,
            candidate_count_after=len(candidates),
        )

    def budget_exhausted() -> bool:
        return len(candidates) >= max_generated_candidates

    def append_candidate(row: dict[str, Any]) -> bool:
        if budget_exhausted():
            return False
        layer = normalize_strategy_layer(row)
        cap = layer_caps.get(layer)
        if isinstance(cap, int) and cap >= 0:
            current = sum(1 for candidate in candidates if normalize_strategy_layer(candidate) == layer)
            if current >= cap:
                return False
        candidates.append(row)
        return True

    def extend_candidates(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            append_candidate(row)

    academic_all_allowed = _strategy_family_allowed(content_route, "academic_all_section_compact_reconstruction")
    if academic_all_allowed:
        before = len(candidates)
        academic_all_section_candidates = _generate_academic_all_section_candidates(
            original_text=original_text,
            scan_report=scan_report,
            gateway=gateway,
            model=model,
            deadline=deadline,
            timeout_seconds=timeout_seconds,
        )
        if academic_all_section_candidates:
            extend_candidates(academic_all_section_candidates)
            record_attempt("academic_all_section_compact_reconstruction", "ran", "generated_candidates", before, applicable=True)
            if not _portfolio_mode_enabled() and any(candidate.get("local_filter_passed") for candidate in academic_all_section_candidates):
                return candidates
            if budget_exhausted():
                return candidates
        else:
            record_attempt("academic_all_section_compact_reconstruction", "skipped", "no_candidates_or_not_applicable", before, applicable=False)
    else:
        record_layer_attempt(
            layer_attempts,
            layer="academic_all_section_compact_reconstruction",
            status="skipped",
            reason="strategy_family_blocked",
            allowed=False,
            applicable=False,
        )
    academic_section_allowed = _strategy_family_allowed(content_route, "academic_cited_section_density_resolver")
    if academic_section_allowed:
        before = len(candidates)
        academic_candidates = _generate_academic_section_candidates(
            original_text=original_text,
            scan_report=scan_report,
            gateway=gateway,
            model=model,
            deadline=deadline,
            timeout_seconds=timeout_seconds,
        )
        if academic_candidates:
            extend_candidates(academic_candidates)
            record_attempt("academic_cited_section_density_resolver", "ran", "generated_candidates", before, applicable=True)
            if not _portfolio_mode_enabled() and any(candidate.get("local_filter_passed") for candidate in academic_candidates):
                return candidates
            if budget_exhausted():
                return candidates
        else:
            record_attempt("academic_cited_section_density_resolver", "skipped", "no_candidates_or_not_applicable", before, applicable=False)
    else:
        record_layer_attempt(
            layer_attempts,
            layer="academic_cited_section_density_resolver",
            status="skipped",
            reason="strategy_family_blocked",
            allowed=False,
            applicable=False,
        )
    full_doc_driver_applicable = _should_entity_locked_full_reconstruction(scan_report)
    if (
        full_doc_driver_applicable
        and (full_reconstruction_allowed or author_stance_allowed or keyword_texture_allowed)
    ):
        required_entities = _required_entities_for_full_reconstruction(original_text)
        expected_paragraph_count = _expected_full_reconstruction_paragraph_count(scan_report, original_text)
        paragraph_inventory = _paragraph_inventory_for_full_reconstruction(scan_report, original_text)
        variants = max(1, min(2, int(os.environ.get("DRAFTPROOF_REWRITE_V2_FULL_RECONSTRUCTION_CANDIDATES", "2") or 2)))
        if full_reconstruction_allowed:
            before = len(candidates)
            for number in range(1, variants + 1):
                if deadline is not None and time.time() + timeout_seconds + 2.0 >= deadline:
                    record_attempt("entity_locked_full_reconstruction", "skipped", "runtime_budget_preflight", before, applicable=True)
                    return candidates
                prompt = _build_entity_locked_full_reconstruction_prompt(
                    original_text=original_text,
                    required_entities=required_entities,
                    paragraph_inventory=paragraph_inventory,
                    expected_paragraph_count=expected_paragraph_count,
                    variant=number,
                )
                response = gateway.chat(
                    prompt,
                    system="You rewrite essays naturally while preserving required entities and avoiding unsupported additions.",
                    max_tokens=7000,
                    temperature=0.58 + (number * 0.03),
                    top_p=0.9,
                    presence_penalty=0.05 if _supports_openai_penalties(model) else None,
                    frequency_penalty=0.15 if _supports_openai_penalties(model) else None,
                    repetition_penalty=1.02 if _supports_repetition_penalty(model) else None,
                    seed=4100 + number,
                )
                raw_text = clean_candidate_output(response.content)
                candidate_text = _restore_required_anchor_forms(raw_text, required_entities)
                filter_failures = _full_reconstruction_filter_failures(
                    original_text=original_text,
                    candidate_text=candidate_text,
                    required_entities=required_entities,
                    expected_paragraph_count=expected_paragraph_count,
                )
                append_candidate({
                    "strategy": "scan_entity_locked_full_reconstruction",
                    "strategy_kind": "entity_locked_full_reconstruction",
                    "candidate_number": number,
                    "text": candidate_text,
                    "candidate_response": candidate_text,
                    "raw_candidate_response": raw_text,
                    "local_filter_passed": not filter_failures,
                    "local_filter_failures": filter_failures,
                    "required_entities": required_entities,
                    "paragraph_count": _paragraph_count(candidate_text),
                    "expected_paragraph_count": expected_paragraph_count,
                })
                if budget_exhausted():
                    record_attempt("entity_locked_full_reconstruction", "ran", "generated_candidates_budget_exhausted", before, applicable=True)
                    return candidates
            record_attempt("entity_locked_full_reconstruction", "ran", "generated_candidates", before, applicable=True)
        else:
            variants = 0
            record_layer_attempt(
                layer_attempts,
                layer="entity_locked_full_reconstruction",
                status="skipped",
                reason="strategy_family_blocked",
                allowed=False,
                applicable=False,
            )
        if (
            os.environ.get("DRAFTPROOF_REWRITE_V2_KEYWORD_LOCKED_SHORT_TEXTURE", "0").lower()
            not in {"0", "false", "no"}
            and keyword_texture_allowed
            and (deadline is None or time.time() + timeout_seconds + 2.0 < deadline)
        ):
            before = len(candidates)
            prompt = _build_keyword_locked_short_texture_prompt(
                required_entities=required_entities,
                paragraph_inventory=paragraph_inventory,
                expected_paragraph_count=expected_paragraph_count,
                original_text=original_text,
            )
            response = gateway.chat(
                prompt,
                system="You generate semantically faithful, lower-predictability rewrites from structured anchors.",
                max_tokens=5000,
                temperature=0.66,
                top_p=0.9,
                presence_penalty=0.08 if _supports_openai_penalties(model) else None,
                frequency_penalty=0.18 if _supports_openai_penalties(model) else None,
                repetition_penalty=1.04 if _supports_repetition_penalty(model) else None,
                seed=11702,
            )
            raw_text = clean_candidate_output(response.content)
            candidate_text = _restore_required_anchor_forms(raw_text, required_entities)
            filter_failures = _full_reconstruction_filter_failures(
                original_text=original_text,
                candidate_text=candidate_text,
                required_entities=required_entities,
                expected_paragraph_count=expected_paragraph_count,
            )
            append_candidate({
                "strategy": "scan_keyword_locked_short_texture",
                "strategy_kind": "entity_locked_full_reconstruction",
                "candidate_number": variants + 1,
                "text": candidate_text,
                "candidate_response": candidate_text,
                "raw_candidate_response": raw_text,
                "local_filter_passed": not filter_failures,
                "local_filter_failures": filter_failures,
                "required_entities": required_entities,
                "paragraph_count": _paragraph_count(candidate_text),
                "expected_paragraph_count": expected_paragraph_count,
            })
            if budget_exhausted():
                record_attempt("keyword_locked_short_texture", "ran", "generated_candidates_budget_exhausted", before, applicable=True)
                return candidates
            record_attempt("keyword_locked_short_texture", "ran", "generated_candidates", before, applicable=True)
        elif keyword_texture_allowed:
            keyword_reason = (
                "disabled_by_config"
                if os.environ.get("DRAFTPROOF_REWRITE_V2_KEYWORD_LOCKED_SHORT_TEXTURE", "0").lower() in {"0", "false", "no"}
                else "runtime_budget_preflight"
            )
            record_layer_attempt(
                layer_attempts,
                layer="keyword_locked_short_texture",
                status="skipped",
                reason=keyword_reason,
                allowed=True,
                applicable=keyword_reason != "disabled_by_config",
            )
        else:
            record_layer_attempt(
                layer_attempts,
                layer="keyword_locked_short_texture",
                status="skipped",
                reason="strategy_family_blocked",
                allowed=False,
                applicable=False,
            )
        if (
            os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_THESIS_REFRAME", "1").lower()
            not in {"0", "false", "no"}
            and author_stance_allowed
            and (deadline is None or time.time() + timeout_seconds + 2.0 < deadline)
        ):
            before = len(candidates)
            texture_attempt_recorded = False
            target_paragraph_count = _author_stance_target_paragraph_count()
            author_variants = max(1, min(4, int(os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_CANDIDATES", "3") or 3)))
            for author_number in range(1, author_variants + 1):
                if deadline is not None and time.time() + timeout_seconds + 2.0 >= deadline:
                    record_attempt("author_stance_thesis_reframe", "skipped", "runtime_budget_preflight", before, applicable=True)
                    break
                prompt = _build_author_stance_thesis_reframe_prompt(
                    original_text=original_text,
                    required_entities=required_entities,
                    paragraph_inventory=paragraph_inventory,
                    target_paragraph_count=target_paragraph_count,
                    variant=author_number,
                )
                response = gateway.chat(
                    prompt,
                    system=(
                        "You rewrite broad essays into faithful, thesis-driven student prose. "
                        "Preserve source anchors and do not invent facts."
                    ),
                    max_tokens=5000,
                    temperature=0.68 + (author_number * 0.04),
                    top_p=0.9 + (0.01 * min(author_number, 3)),
                    presence_penalty=0.12 if _supports_openai_penalties(model) else None,
                    frequency_penalty=0.25 if _supports_openai_penalties(model) else None,
                    repetition_penalty=1.06 if _supports_repetition_penalty(model) else None,
                    seed=19000 + author_number,
                )
                raw_text = clean_candidate_output(response.content)
                candidate_text = _restore_required_anchor_forms(raw_text, required_entities)
                filter_failures = _author_stance_thesis_filter_failures(
                    candidate_text=candidate_text,
                    required_entities=required_entities,
                    min_paragraphs=target_paragraph_count,
                    max_paragraphs=target_paragraph_count,
                )
                append_candidate({
                    "strategy": "scan_author_stance_thesis_reframe",
                    "strategy_kind": "author_stance_thesis_reframe",
                    "candidate_number": variants + 1 + author_number,
                    "author_variant": author_number,
                    "text": candidate_text,
                    "candidate_response": candidate_text,
                    "raw_candidate_response": raw_text,
                    "local_filter_passed": not filter_failures,
                    "local_filter_failures": filter_failures,
                    "required_entities": required_entities,
                    "paragraph_count": _paragraph_count(candidate_text),
                    "expected_paragraph_count": target_paragraph_count,
                })
                if budget_exhausted():
                    record_attempt("author_stance_thesis_reframe", "ran", "generated_candidates_budget_exhausted", before, applicable=True)
                    return candidates
                if (
                    not filter_failures
                    and os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_TEXTURE_PASS", "0").lower()
                    not in {"0", "false", "no"}
                    and author_texture_allowed
                    and (deadline is None or time.time() + timeout_seconds + 2.0 < deadline)
                ):
                    texture_before = len(candidates)
                    prompt = _build_author_stance_texture_pass_prompt(
                        source_text=original_text,
                        draft_text=candidate_text,
                        required_entities=required_entities,
                        target_paragraph_count=target_paragraph_count,
                    )
                    response = gateway.chat(
                        prompt,
                        system=(
                            "You revise essay texture while preserving facts. "
                            "Reduce formal survey rhythm without adding information."
                        ),
                        max_tokens=5000,
                        temperature=0.78,
                        top_p=0.94,
                        presence_penalty=0.18 if _supports_openai_penalties(model) else None,
                        frequency_penalty=0.32 if _supports_openai_penalties(model) else None,
                        repetition_penalty=1.08 if _supports_repetition_penalty(model) else None,
                        seed=19100 + author_number,
                    )
                    raw_text = clean_candidate_output(response.content)
                    texture_text = _restore_required_anchor_forms(raw_text, required_entities)
                    texture_failures = _author_stance_thesis_filter_failures(
                        candidate_text=texture_text,
                        required_entities=required_entities,
                        min_paragraphs=target_paragraph_count,
                        max_paragraphs=target_paragraph_count,
                        require_author_stance_marker=False,
                        reject_survey_style=True,
                    )
                    append_candidate({
                        "strategy": "scan_author_stance_texture_pass",
                        "strategy_kind": "author_stance_texture_pass",
                        "candidate_number": variants + 1 + author_variants + author_number,
                        "author_variant": author_number,
                        "text": texture_text,
                        "candidate_response": texture_text,
                        "raw_candidate_response": raw_text,
                        "local_filter_passed": not texture_failures,
                        "local_filter_failures": texture_failures,
                        "required_entities": required_entities,
                        "paragraph_count": _paragraph_count(texture_text),
                        "expected_paragraph_count": target_paragraph_count,
                        "source_strategy": "scan_author_stance_thesis_reframe",
                    })
                    record_attempt("author_stance_texture_pass", "ran", "generated_candidates", texture_before, applicable=True)
                    texture_attempt_recorded = True
                    if budget_exhausted():
                        return candidates
            if len(candidates) > before:
                record_attempt("author_stance_thesis_reframe", "ran", "generated_candidates", before, applicable=True)
            if author_texture_allowed and not texture_attempt_recorded:
                texture_enabled = os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_TEXTURE_PASS", "0").lower() not in {"0", "false", "no"}
                record_layer_attempt(
                    layer_attempts,
                    layer="author_stance_texture_pass",
                    status="skipped",
                    reason="disabled_by_config" if not texture_enabled else "no_valid_author_stance_source",
                    allowed=True,
                    applicable=texture_enabled,
                )
        elif author_stance_allowed:
            author_reason = (
                "disabled_by_config"
                if os.environ.get("DRAFTPROOF_REWRITE_V2_AUTHOR_STANCE_THESIS_REFRAME", "1").lower() in {"0", "false", "no"}
                else "runtime_budget_preflight"
            )
            record_layer_attempt(
                layer_attempts,
                layer="author_stance_thesis_reframe",
                status="skipped",
                reason=author_reason,
                allowed=True,
                applicable=author_reason != "disabled_by_config",
            )
        else:
            record_layer_attempt(
                layer_attempts,
                layer="author_stance_thesis_reframe",
                status="skipped",
                reason="strategy_family_blocked",
                allowed=False,
                applicable=False,
            )
        if candidates and not _portfolio_mode_enabled():
            return candidates
        if budget_exhausted():
            return candidates
    else:
        full_skip_reason = "not_applicable" if not full_doc_driver_applicable else "strategy_family_blocked"
        for layer, allowed in (
            ("entity_locked_full_reconstruction", full_reconstruction_allowed),
            ("keyword_locked_short_texture", keyword_texture_allowed),
            ("author_stance_thesis_reframe", author_stance_allowed),
            ("author_stance_texture_pass", author_texture_allowed),
        ):
            record_layer_attempt(
                layer_attempts,
                layer=layer,
                status="skipped",
                reason=full_skip_reason if allowed else "strategy_family_blocked",
                allowed=allowed,
                applicable=False,
            )
    for strategy in strategies:
        if getattr(strategy, "kind", None).value == "targeted":
            if budget_exhausted():
                return candidates
            targeted_before = len(candidates)
            paragraph_briefs = [
                _enrich_paragraph_brief_with_target(brief, paragraph_targets)
                for brief in targeted_paragraph_briefs(scan_report)
            ]
            if not paragraph_briefs:
                record_attempt("targeted_paragraph_reconstruction", "skipped", "no_targets", targeted_before, applicable=False)
                continue
            tactics = _paragraph_tactics()
            for brief in paragraph_briefs:
                paragraph_id = str(brief.get("paragraph_id") or "")
                variant_limit = max(1, int(strategy.max_candidates or 1))
                for number, tactic in enumerate(tactics[:variant_limit], start=1):
                    if deadline is not None and time.time() + timeout_seconds + 2.0 >= deadline:
                        return candidates
                    prompt = build_single_paragraph_reconstruction_prompt(brief, strategy, tactic=tactic)
                    structured_options = _structured_json_request_options(model, _paragraph_response_format())
                    response = gateway.chat(
                        prompt,
                        system="You are DraftProof's paragraph reconstruction engine.",
                        max_tokens=1800,
                        temperature=0.65,
                        top_p=0.9,
                        presence_penalty=0.35 if _supports_openai_penalties(model) else None,
                        frequency_penalty=0.45 if _supports_openai_penalties(model) else None,
                        repetition_penalty=1.08 if _supports_repetition_penalty(model) else None,
                        seed=(1701 + number),
                        response_format=structured_options["response_format"],
                        provider=structured_options["provider"],
                    )
                    parse_diagnostics = _json_parse_diagnostics(response.content)
                    payload = parse_diagnostics["payload"]
                    patch = {
                        "paragraph_id": str(payload.get("paragraph_id") or paragraph_id),
                        "rewritten_paragraph": str(payload.get("rewritten_paragraph") or "").strip(),
                        "rationale": payload.get("rationale"),
                    }
                    candidate_payload = _attach_hidden_paragraph_targets(
                        {"candidate_id": f"{paragraph_id or 'paragraph'}_variant_{number}", "patches": [patch]},
                        paragraph_targets,
                    )
                    patch_payload = {"patches": candidate_payload.get("patches") or []}
                    candidate_text, applied_patches = _apply_targeted_patches(original_text, patch_payload)
                    filter_failures = _patch_filter_failures(patch_payload["patches"])
                    if not parse_diagnostics["ok"]:
                        filter_failures.insert(0, f"structured_output_invalid:{parse_diagnostics['reason']}")
                    append_candidate({
                        "strategy": strategy.strategy_id,
                        "strategy_kind": strategy.kind.value,
                        "candidate_number": number,
                        "paragraph_id": paragraph_id,
                        "tactic": tactic,
                        "text": candidate_text,
                        "candidate_response": {key: value for key, value in payload.items() if key != "target_paragraph"} or payload,
                        "structured_output_mode": structured_options["structured_output_mode"],
                        "structured_output_parse": {
                            key: value
                            for key, value in parse_diagnostics.items()
                            if key != "payload"
                        },
                        "local_filter_passed": not filter_failures,
                        "local_filter_failures": filter_failures,
                        "applied_patch_count": sum(1 for row in applied_patches if row.get("applied")),
                        "patch_count": len(applied_patches),
                        "patches": applied_patches,
                    })
                    if budget_exhausted():
                        record_attempt("targeted_paragraph_reconstruction", "ran", "generated_candidates_budget_exhausted", targeted_before, applicable=True)
                        return candidates
            record_attempt("targeted_paragraph_reconstruction", "ran", "generated_candidates", targeted_before, applicable=True)
            continue
        generic_before = len(candidates)
        for number in range(1, max(1, int(strategy.max_candidates or 1)) + 1):
            if deadline is not None and time.time() + timeout_seconds + 2.0 >= deadline:
                record_attempt(normalize_strategy_layer(getattr(strategy, "strategy_id", "") or getattr(strategy, "kind", "")), "skipped", "runtime_budget_preflight", generic_before, applicable=True)
                return candidates
            prompt = build_strategy_prompt(original_text, scan_report, strategy)
            response = gateway.chat(
                prompt,
                system="You are DraftProof's scan-driven AI-risk mitigation rewrite engine.",
                max_tokens=6500,
                temperature=0.45,
                top_p=0.82,
                presence_penalty=0.15,
                frequency_penalty=0.25,
            )
            candidate_text = clean_candidate_output(response.content)
            filter_failures: list[str] = []
            if getattr(strategy, "kind", None).value == "full_rewrite":
                filter_failures = _full_reconstruction_filter_failures(
                    original_text=original_text,
                    candidate_text=candidate_text,
                    required_entities=_required_entities_for_full_reconstruction(original_text),
                    expected_paragraph_count=_expected_full_reconstruction_paragraph_count(scan_report, original_text),
                )
            append_candidate({
                "strategy": strategy.strategy_id,
                "strategy_kind": strategy.kind.value,
                "candidate_number": number,
                "text": candidate_text,
                "candidate_response": candidate_text,
                "local_filter_passed": not filter_failures,
                "local_filter_failures": filter_failures,
            })
            if budget_exhausted():
                record_attempt(normalize_strategy_layer({
                    "strategy": strategy.strategy_id,
                    "strategy_kind": strategy.kind.value,
                }), "ran", "generated_candidates_budget_exhausted", generic_before, applicable=True)
                return candidates
        record_attempt(normalize_strategy_layer({
            "strategy": strategy.strategy_id,
            "strategy_kind": strategy.kind.value,
        }), "ran", "generated_candidates", generic_before, applicable=True)
    return candidates


def run_rewrite_pipeline_v2(
    *,
    detect_json: dict[str, Any],
    output_dir: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    replay_candidate_records: list[dict[str, Any]] | None = None,
    max_runtime_seconds: int = 300,
    required_ai_drop: float = 5.0,
    full_rewrite_allowed: bool = True,
) -> dict[str, Any]:
    started = time.time()
    runtime_budget = RewriteV2RuntimeBudget(
        started_at=started,
        max_runtime_seconds=max_runtime_seconds,
        generation_budget_seconds=_generation_budget_seconds(max_runtime_seconds),
    )

    def progress(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    progress(62, "Starting scan-driven rewrite V2")
    original_text = _extract_original_text(detect_json)
    original_report = detect_json
    reference_ai = _badge_ai(original_report)
    effective_config = _effective_config(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_runtime_seconds=max_runtime_seconds,
    )
    target_ai_score = (
        float(reference_ai) - float(required_ai_drop)
        if isinstance(reference_ai, (int, float))
        else None
    )
    content_route = classify_content_route(original_text, original_report)
    strategies = route_strategies(
        original_report,
        full_rewrite_allowed=full_rewrite_allowed,
        content_route=content_route,
    )
    effective_config = {
        **effective_config,
        "content_mode": content_route.content_mode,
        "content_mode_confidence": content_route.confidence,
        "allowed_strategy_families": content_route.allowed_strategy_families,
        "blocked_strategy_families": content_route.blocked_strategy_families,
        "portfolio_limits": portfolio_limits(content_route),
    }
    author_context_blocked = (
        needs_author_context(original_report)
        and os.environ.get("DRAFTPROOF_REWRITE_V2_FAIL_FAST_AUTHOR_CONTEXT", "0").lower() in {"1", "true", "yes"}
    )
    if author_context_blocked and replay_candidate_records is None:
        elapsed = time.time() - started
        final_goal_eval = evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=original_text,
            original_report=original_report,
            candidate_report=original_report,
        )
        final_goal = {
            **final_goal_eval.to_dict(),
            "status": RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT.value,
            "goal_met": False,
            "reason": "scan_requires_author_context_before_rewrite_budget",
        }
        summary = {
            "rewrite_pipeline_version": "rewrite_v2_scan_driven",
            "outcome": RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT.value,
            "rewrite_goal_status": final_goal,
            "reference_ai": reference_ai,
            "required_ai_drop": required_ai_drop,
            "target_ai_score": target_ai_score,
            "rewrite_effective_config": effective_config,
            "runtime_budget": runtime_budget.to_dict(),
            "candidate_generation_status": {
                "generated_count": 0,
                "candidate_rows": 0,
                "reason": "needs_author_context",
                "layer_attempts": [],
                "layer_attempt_summary": summarize_layer_attempts([]),
            },
            "content_router_trace": content_route.to_dict(),
            "strategy_trace": [strategy.to_dict() for strategy in strategies],
            "candidate_trace": [],
            "selected_candidate": None,
            "stage_timings": [{
                "stage": "rewrite_v2_scan_driven",
                "seconds": round(elapsed, 3),
                "candidates": 0,
                "selected": False,
                "stop_reason": "needs_author_context",
            }],
            "detect_scan_original": original_report,
            "detect_scan_rewritten": original_report,
            "final_text": original_text,
        }
        sentence_comparison = _sentence_comparison(original_text, original_text)
        result_obj = SimpleNamespace(
            summary=summary,
            sentence_comparison=sentence_comparison,
            rewrite_plan=None,
            mp_result=SimpleNamespace(
                original_text=original_text,
                final_text=original_text,
                converged=False,
                convergence_reason="rewrite_v2_needs_author_context",
                passes=[],
            ),
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        md_path = out_dir / f"draftproof_rewrite_v2_{ts}.md"
        pdf_path = out_dir / f"draftproof_rewrite_v2_{ts}.pdf"
        json_path = out_dir / f"draftproof_rewrite_v2_{ts}.json"
        md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
        md_path.write_text(md_text, encoding="utf-8")
        render_pdf(md_text, str(pdf_path))
        json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        progress(88, "Scan-driven rewrite V2 stopped: author context required")
        return {
            "status": RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT.value,
            "md_path": str(md_path),
            "pdf_path": str(pdf_path),
            "json_path": str(json_path),
            "result": result_obj,
            "elapsed": elapsed,
        }
    candidate_rows: list[dict[str, Any]]
    if replay_candidate_records is not None:
        generated_count = len(replay_candidate_records)
        generation_reason = "replay_candidates"
        layer_attempts: list[dict[str, Any]] = []
        candidate_rows = _candidate_rows_from_replay(
            replay_candidate_records,
            original_text=original_text,
            original_report=original_report,
            required_ai_drop=required_ai_drop,
            target_ai_score=target_ai_score,
        )
    else:
        layer_attempts = []
        generated = _generate_candidates(
            original_text=original_text,
            scan_report=original_report,
            strategies=strategies,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=_llm_call_timeout_seconds(),
            deadline=runtime_budget.generation_deadline,
            content_route=content_route,
            layer_attempts=layer_attempts,
        )
        generated_count = len(generated)
        generation_reason = "generated_candidates" if generated_count else "candidate_generation_budget_exhausted_no_candidates"
        candidate_rows = []
        post_layer_trace: list[dict[str, Any]] = []
        protected = detect_protected_spans(original_text)

        def record_post_layer(
            *,
            layer: str,
            status: str,
            reason: str,
            before_count: int | None = None,
            generated_count: int = 0,
            allowed: bool | None = True,
            applicable: bool | None = None,
            extra: dict[str, Any] | None = None,
        ) -> None:
            trace_row = {"layer": layer, "status": status, "reason": reason}
            if extra:
                trace_row.update(extra)
            post_layer_trace.append(trace_row)
            record_layer_attempt(
                layer_attempts,
                layer=layer,
                status=status,
                reason=reason,
                allowed=allowed,
                applicable=applicable,
                generated_count=generated_count,
                candidate_count_before=before_count,
                candidate_count_after=len(candidate_rows),
            )

        for index, generated_candidate in enumerate(generated, start=1):
            if not runtime_budget.can_start(_phase_start_margin_seconds()):
                break
            candidate_text = str(generated_candidate.get("text") or "").strip()
            if generated_candidate.get("local_filter_passed") is False:
                candidate_rows.append(_local_filter_rejected_candidate_row(generated_candidate))
                continue
            if not candidate_text:
                candidate_rows.append(_empty_generated_candidate_row(generated_candidate))
                continue
            local_score = None
            if (
                str(generated_candidate.get("strategy_kind") or "") == "targeted"
                and _use_paragraph_local_score_gate()
            ):
                local_score = _paragraph_local_score(generated_candidate)
                if isinstance(local_score, dict) and not local_score.get("improved"):
                    candidate_rows.append({
                        **generated_candidate,
                        "candidate_ai": None,
                        "paragraph_local_score": local_score,
                        "decision": {
                            "lane": CandidateLane.REJECT.value,
                            "reason": "paragraph_local_score_not_improved",
                            "rank": [],
                        },
                    })
                    continue
            semantic = check_semantic_drift(original_text, candidate_text, threshold=0.15)
            anchors_safe = protected_spans_preserved(original_text, candidate_text, protected)
            strategy_kind = str(generated_candidate.get("strategy_kind") or "")
            semantic_similarity = getattr(semantic, "similarity", None)
            semantic_safe = bool(semantic.accepted)
            semantic_override = False
            if not semantic_safe and _author_strategy_semantic_override_allowed(
                strategy_kind=strategy_kind,
                generated_candidate=generated_candidate,
                candidate_text=candidate_text,
                semantic_similarity=semantic_similarity,
                anchors_safe=bool(anchors_safe),
            ):
                semantic_safe = True
                semantic_override = True
            if not semantic_safe and _academic_contract_semantic_override_allowed(
                strategy_kind=strategy_kind,
                original_text=original_text,
                original_report=original_report,
                candidate_text=candidate_text,
                semantic_similarity=semantic_similarity,
                anchors_safe=bool(anchors_safe),
                semantic_reasons=getattr(semantic, "reasons", None),
            ):
                semantic_safe = True
                semantic_override = True
            if not anchors_safe or not _semantic_scan_allowed(strategy_kind, semantic_safe):
                candidate_rows.append({
                    **generated_candidate,
                    "decision": {
                        "lane": CandidateLane.REJECT.value,
                        "reason": "protected_anchor_or_semantic_scan_guard_rejected",
                        "rank": [],
                    },
                    "semantic_safe": semantic_safe,
                    "protected_anchors_safe": bool(anchors_safe),
                    "semantic_review_required": not semantic_safe,
                    "semantic_similarity": semantic_similarity,
                    "semantic_reasons": getattr(semantic, "reasons", None),
                    "semantic_override_applied": semantic_override,
                })
                continue
            progress(min(80, 64 + index * 4), f"Scanning V2 candidate {index}")
            candidate_report = _scan_report(candidate_text)
            goal = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=candidate_text,
                original_report=original_report,
                candidate_report=candidate_report,
            )
            decision = decide_candidate(
                goal=goal,
                original_report=original_report,
                candidate_report=candidate_report,
                reference_ai=reference_ai,
                required_ai_drop=required_ai_drop,
                target_ai_score=target_ai_score,
                semantic_safe=semantic_safe,
                quality_safe=anchors_safe,
                cost=index,
            )
            candidate_rows.append({
                **generated_candidate,
                "candidate_ai": _badge_ai(candidate_report),
                "candidate_wq": _badge_wq(candidate_report),
                "paragraph_local_score": local_score,
                "goal": goal.to_dict(),
                "decision": decision.to_dict(),
                "semantic_safe": semantic_safe,
                "protected_anchors_safe": bool(anchors_safe),
                "semantic_review_required": not semantic_safe,
                "semantic_similarity": semantic_similarity,
                "semantic_reasons": getattr(semantic, "reasons", None),
                "semantic_override_applied": semantic_override,
                "report": candidate_report,
                "text": candidate_text,
            })
            if decision.lane == CandidateLane.GOAL_MET:
                break
        composed_text, composed_patches = _compose_full_doc_delta_winners(original_text, candidate_rows, reference_ai)
        if composed_patches and composed_text.strip() != original_text.strip():
            index = len(candidate_rows) + 1
            before_count = len(candidate_rows)
            composition_budget_ok = runtime_budget.can_start(_phase_start_margin_seconds())
            if (
                composition_budget_ok
                and _candidate_portfolio_allows(candidate_rows, content_route, "targeted_composition")
            ):
                progress(min(84, 64 + index * 4), "Scanning V2 composed local winners")
                candidate_report = _scan_report(composed_text)
                goal = evaluate_rewrite_goal(
                    original_text=original_text,
                    candidate_text=composed_text,
                    original_report=original_report,
                    candidate_report=candidate_report,
                )
                semantic = check_semantic_drift(original_text, composed_text, threshold=0.15)
                anchors_safe = protected_spans_preserved(original_text, composed_text, protected)
                decision = decide_candidate(
                    goal=goal,
                    original_report=original_report,
                    candidate_report=candidate_report,
                    reference_ai=reference_ai,
                    required_ai_drop=required_ai_drop,
                    target_ai_score=target_ai_score,
                    semantic_safe=bool(semantic.accepted),
                    quality_safe=anchors_safe,
                    cost=index,
                )
                candidate_rows.append({
                    "strategy": "scan_targeted_composed_full_doc_delta_winners",
                    "strategy_kind": "targeted_composition",
                    "candidate_number": 1,
                    "candidate_ai": _badge_ai(candidate_report),
                    "candidate_wq": _badge_wq(candidate_report),
                    "composed_patches": composed_patches,
                    "goal": goal.to_dict(),
                    "decision": decision.to_dict(),
                    "semantic_safe": bool(semantic.accepted),
                    "protected_anchors_safe": bool(anchors_safe),
                    "semantic_review_required": not bool(semantic.accepted),
                    "semantic_similarity": getattr(semantic, "similarity", None),
                    "semantic_reasons": getattr(semantic, "reasons", None),
                    "report": candidate_report,
                    "text": composed_text,
                })
                record_post_layer(
                    layer="targeted_composition",
                    status="ran",
                    reason="candidate_scored",
                    before_count=before_count,
                    generated_count=1,
                    applicable=True,
                )
            else:
                record_post_layer(
                    layer="targeted_composition",
                    status="skipped",
                    reason="runtime_budget_exhausted" if not composition_budget_ok else "portfolio_budget_exhausted",
                    before_count=before_count,
                    applicable=True,
                )
        else:
            record_post_layer(
                layer="targeted_composition",
                status="skipped",
                reason="no_composable_local_winners",
                before_count=len(candidate_rows),
                applicable=False,
            )
        frontier = _select_best_v2_frontier(candidate_rows, content_route=content_route)
        frontier_lane = ((frontier or {}).get("decision") or {}).get("lane")
        academic_repair_frontier = frontier or select_best_candidate([
            row for row in candidate_rows
            if str(row.get("strategy") or "").startswith("academic_")
            and isinstance(row.get("candidate_ai"), (int, float))
        ])
        academic_repair_budget_ok = runtime_budget.can_start(_llm_call_timeout_seconds() + _phase_start_margin_seconds())
        academic_repair_allowed = _strategy_family_allowed(content_route, "academic_anchor_repair_texture_pass")
        if (
            academic_repair_frontier
            and frontier_lane != CandidateLane.GOAL_MET.value
            and academic_repair_allowed
            and str(academic_repair_frontier.get("strategy") or "").startswith("academic_")
            and academic_repair_budget_ok
            and _candidate_portfolio_allows(candidate_rows, content_route, "academic_anchor_repair_texture_pass")
        ):
            before_count = len(candidate_rows)
            progress(82, "Running V2 academic anchor repair")
            repair_rows = _generate_academic_anchor_repair_candidates(
                frontier=academic_repair_frontier,
                original_text=original_text,
                scan_report=original_report,
                gateway=LLMGateway(LLMConfig(
                    api_key=api_key,
                    model=model or os.environ.get("LLM_MODEL") or "openai/gpt-4.1-mini",
                    base_url=base_url or os.environ.get("LLM_BASE_URL", ""),
                    timeout=_llm_call_timeout_seconds(),
                    max_retries=1,
                    max_tokens=7600,
                    temperature=0.45,
                )),
                model=model,
                deadline=runtime_budget.absolute_deadline,
                timeout_seconds=_llm_call_timeout_seconds(),
            )
            repair_start = len(candidate_rows) + 1
            for offset, generated_candidate in enumerate(repair_rows):
                index = repair_start + offset
                if not runtime_budget.can_start(_phase_start_margin_seconds()):
                    break
                candidate_text = str(generated_candidate.get("text") or "").strip()
                if generated_candidate.get("local_filter_passed") is False:
                    candidate_rows.append(_local_filter_rejected_candidate_row(generated_candidate))
                    continue
                if not candidate_text:
                    candidate_rows.append(_empty_generated_candidate_row(generated_candidate))
                    continue
                semantic = check_semantic_drift(original_text, candidate_text, threshold=0.15)
                anchors_safe = protected_spans_preserved(original_text, candidate_text, protected)
                semantic_safe = bool(semantic.accepted)
                semantic_similarity = getattr(semantic, "similarity", None)
                semantic_override = False
                if not semantic_safe and _academic_contract_semantic_override_allowed(
                    strategy_kind=str(generated_candidate.get("strategy_kind") or ""),
                    original_text=original_text,
                    original_report=original_report,
                    candidate_text=candidate_text,
                    semantic_similarity=semantic_similarity,
                    anchors_safe=bool(anchors_safe),
                    semantic_reasons=getattr(semantic, "reasons", None),
                ):
                    semantic_safe = True
                    semantic_override = True
                if not anchors_safe or not _semantic_scan_allowed(str(generated_candidate.get("strategy_kind") or ""), semantic_safe):
                    candidate_rows.append({
                        **generated_candidate,
                        "decision": {
                            "lane": CandidateLane.REJECT.value,
                            "reason": "protected_anchor_or_semantic_scan_guard_rejected",
                            "rank": [],
                        },
                        "semantic_safe": semantic_safe,
                        "protected_anchors_safe": bool(anchors_safe),
                        "semantic_review_required": not semantic_safe,
                        "semantic_similarity": semantic_similarity,
                        "semantic_reasons": getattr(semantic, "reasons", None),
                        "semantic_override_applied": semantic_override,
                    })
                    continue
                progress(min(84, 64 + index * 4), f"Scanning V2 academic repair candidate {offset + 1}")
                candidate_report = _scan_report(candidate_text)
                goal = evaluate_rewrite_goal(
                    original_text=original_text,
                    candidate_text=candidate_text,
                    original_report=original_report,
                    candidate_report=candidate_report,
                )
                decision = decide_candidate(
                    goal=goal,
                    original_report=original_report,
                    candidate_report=candidate_report,
                    reference_ai=reference_ai,
                    required_ai_drop=required_ai_drop,
                    target_ai_score=target_ai_score,
                    semantic_safe=semantic_safe,
                    quality_safe=anchors_safe,
                    cost=index,
                )
                candidate_rows.append({
                    **generated_candidate,
                    "candidate_ai": _badge_ai(candidate_report),
                    "candidate_wq": _badge_wq(candidate_report),
                    "paragraph_local_score": None,
                    "goal": goal.to_dict(),
                    "decision": decision.to_dict(),
                    "semantic_safe": semantic_safe,
                    "protected_anchors_safe": bool(anchors_safe),
                    "semantic_review_required": not semantic_safe,
                    "semantic_similarity": semantic_similarity,
                    "semantic_reasons": getattr(semantic, "reasons", None),
                    "semantic_override_applied": semantic_override,
                    "report": candidate_report,
                    "text": candidate_text,
                })
                if decision.lane == CandidateLane.GOAL_MET:
                    break
            frontier = _select_best_v2_frontier(candidate_rows, content_route=content_route)
            frontier_lane = ((frontier or {}).get("decision") or {}).get("lane")
            generated_rows = max(0, len(candidate_rows) - before_count)
            record_post_layer(
                layer="academic_anchor_repair_texture_pass",
                status="ran" if generated_rows else "skipped",
                reason="repair_candidates_evaluated" if generated_rows else "no_repair_candidates",
                before_count=before_count,
                generated_count=generated_rows,
                applicable=True,
            )
        else:
            academic_skip_reasons = []
            if not academic_repair_frontier:
                academic_skip_reasons.append("no_academic_frontier")
            if frontier_lane == CandidateLane.GOAL_MET.value:
                academic_skip_reasons.append("goal_met")
            if not academic_repair_allowed:
                academic_skip_reasons.append("strategy_family_blocked")
            if academic_repair_frontier and not str(academic_repair_frontier.get("strategy") or "").startswith("academic_"):
                academic_skip_reasons.append("frontier_not_academic")
            if not academic_repair_budget_ok:
                academic_skip_reasons.append("runtime_budget_exhausted")
            if not _candidate_portfolio_allows(candidate_rows, content_route, "academic_anchor_repair_texture_pass"):
                academic_skip_reasons.append("portfolio_budget_exhausted")
            record_post_layer(
                layer="academic_anchor_repair_texture_pass",
                status="skipped",
                reason=",".join(academic_skip_reasons) or "not_applicable",
                before_count=len(candidate_rows),
                allowed=academic_repair_allowed,
                applicable=bool(academic_repair_frontier and str(academic_repair_frontier.get("strategy") or "").startswith("academic_")),
            )
        rescue_enabled = (
            os.environ.get("DRAFTPROOF_REWRITE_V2_UNSAFE_CLUSTER_RESCUE", "1").lower() not in {"0", "false", "no"}
            and _strategy_family_allowed(content_route, "unsafe_cluster_rescue")
        )
        rescue_budget_ok = runtime_budget.can_start(_llm_call_timeout_seconds() + _phase_start_margin_seconds())
        if (
            rescue_enabled
            and frontier
            and frontier_lane != CandidateLane.GOAL_MET.value
            and rescue_budget_ok
            and _candidate_portfolio_allows(candidate_rows, content_route, "unsafe_cluster_rescue")
        ):
            before_count = len(candidate_rows)
            progress(84, "Running V2 unsafe-cluster rescue")
            rescue_budget_remaining = max(
                0,
                int((portfolio_limits(content_route).get("max_generated_candidates") or 0)) - len(candidate_rows),
            )
            rescue_layer_cap = (portfolio_limits(content_route).get("layer_candidate_caps") or {}).get("unsafe_cluster_rescue")
            if isinstance(rescue_layer_cap, int) and rescue_layer_cap >= 0:
                rescue_budget_remaining = min(
                    rescue_budget_remaining,
                    max(0, rescue_layer_cap - sum(1 for row in candidate_rows if normalize_strategy_layer(row) == "unsafe_cluster_rescue")),
                )
            rescue_rows = _generate_unsafe_cluster_rescue_candidates(
                frontier=frontier,
                original_text=original_text,
                original_report=original_report,
                reference_ai=reference_ai,
                required_ai_drop=required_ai_drop,
                target_ai_score=target_ai_score,
                api_key=api_key,
                model=model,
                base_url=base_url,
                timeout_seconds=_llm_call_timeout_seconds(),
                starting_cost=len(candidate_rows),
                max_rows=rescue_budget_remaining,
                deadline=runtime_budget.absolute_deadline,
            )
            for row in rescue_rows:
                if not _candidate_portfolio_allows(candidate_rows, content_route, row):
                    break
                candidate_rows.append(row)
            trace_extra = {
                "candidate_rows": len(rescue_rows),
            }
            record_post_layer(
                layer="unsafe_cluster_rescue",
                status="ran" if rescue_rows else "skipped",
                reason="rescue_candidates_evaluated" if rescue_rows else "no_unsafe_clusters_or_budget",
                before_count=before_count,
                generated_count=max(0, len(candidate_rows) - before_count),
                applicable=True,
                extra=trace_extra,
            )
        else:
            rescue_skip_reasons = []
            if not rescue_enabled:
                rescue_skip_reasons.append("disabled_or_strategy_family_blocked")
            if not frontier:
                rescue_skip_reasons.append("no_frontier")
            if frontier_lane == CandidateLane.GOAL_MET.value:
                rescue_skip_reasons.append("goal_met")
            if not rescue_budget_ok:
                rescue_skip_reasons.append("runtime_budget_exhausted")
            if not _candidate_portfolio_allows(candidate_rows, content_route, "unsafe_cluster_rescue"):
                rescue_skip_reasons.append("portfolio_budget_exhausted")
            record_post_layer(
                layer="unsafe_cluster_rescue",
                status="skipped",
                reason=",".join(rescue_skip_reasons) or "not_applicable",
                before_count=len(candidate_rows),
                allowed=rescue_enabled,
                applicable=bool(frontier and frontier_lane != CandidateLane.GOAL_MET.value),
            )
    if replay_candidate_records is not None:
        post_layer_trace = []
    close_partial_max_gap = _close_partial_max_gap()
    diagnostic_best = select_best_candidate(candidate_rows)
    best = _select_best_v2_frontier(candidate_rows, content_route=content_route) or diagnostic_best
    best_decision = best.get("decision") if isinstance(best, dict) else {}
    best_lane = (best_decision or {}).get("lane")
    best_applicable_near_miss = bool(
        best
        and best_lane == CandidateLane.SAFE_NEAR_MISS.value
        and (best_decision or {}).get("required_drop_met")
        and (best_decision or {}).get("quality_safe")
        and (best_decision or {}).get("semantic_safe")
    )
    close_partial_gap = (best_decision or {}).get("ai_target_gap")
    best_applicable_close_partial = bool(
        best
        and os.environ.get("DRAFTPROOF_REWRITE_V2_APPLY_CLOSE_PARTIAL", "1").lower() not in {"0", "false", "no"}
        and best_lane == CandidateLane.PARTIAL_DIAGNOSTIC.value
        and isinstance(close_partial_gap, (int, float))
        and (
            float(close_partial_gap) <= close_partial_max_gap
            or (
                _candidate_patch_coverage(best) >= 2
                and float(close_partial_gap) <= _composition_partial_max_gap()
            )
        )
        and (best_decision or {}).get("quality_safe")
        and (best_decision or {}).get("semantic_safe")
    )
    if best and best_lane == CandidateLane.GOAL_MET.value:
        final_text = str(best.get("text") or original_text)
        final_report = best.get("report") if isinstance(best.get("report"), dict) else original_report
        final_goal = best.get("goal")
        public_status = RewriteGoalStatus.AI_MITIGATED.value
        converged = True
        convergence_reason = "rewrite_v2_strict_goal_met"
    elif best_applicable_near_miss:
        final_text = str(best.get("text") or original_text)
        final_report = best.get("report") if isinstance(best.get("report"), dict) else original_report
        near_miss_goal = best.get("goal") if isinstance(best.get("goal"), dict) else {}
        final_goal = {
            **near_miss_goal,
            "status": RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value,
            "goal_met": False,
            "applied_candidate_lane": CandidateLane.SAFE_NEAR_MISS.value,
            "reason": "score_target_met_but_strict_detector_safe_goal_not_met",
        }
        public_status = "safe_near_miss_applied"
        converged = False
        convergence_reason = "rewrite_v2_score_target_candidate_applied_strict_goal_not_met"
    elif best_applicable_close_partial:
        final_text = str(best.get("text") or original_text)
        final_report = best.get("report") if isinstance(best.get("report"), dict) else original_report
        close_partial_goal = best.get("goal") if isinstance(best.get("goal"), dict) else {}
        final_goal = {
            **close_partial_goal,
            "status": RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value,
            "goal_met": False,
            "applied_candidate_lane": CandidateLane.PARTIAL_DIAGNOSTIC.value,
            "reason": "close_score_frontier_applied_but_target_not_met",
        }
        public_status = "safe_partial_mitigation_applied"
        converged = False
        convergence_reason = "rewrite_v2_close_score_frontier_applied_target_not_met"
    else:
        final_text = original_text
        final_report = original_report
        preserved_goal = evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=original_text,
            original_report=original_report,
            candidate_report=original_report,
            no_text_change=True,
        )
        if candidate_rows:
            preserved_goal = evaluate_rewrite_goal(
                original_text=original_text,
                candidate_text=str((best or {}).get("text") or original_text),
                original_report=original_report,
                candidate_report=(best or {}).get("report") if isinstance((best or {}).get("report"), dict) else original_report,
            )
        status = (
            RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT
            if preserved_goal.status == RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT
            else RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE
        )
        failure_reason = (
            "candidate_generation_failed_no_candidates"
            if not candidate_rows and generated_count == 0
            else "candidate_evaluation_skipped_runtime_budget_exhausted"
            if not candidate_rows and generated_count > 0
            else "no_safe_rewrite_applied"
        )
        final_goal = {
            **preserved_goal.to_dict(),
            "status": status.value,
            "goal_met": False,
            "reason": failure_reason,
        }
        public_status = status.value
        converged = False
        convergence_reason = f"rewrite_v2_{failure_reason}"
    elapsed = time.time() - started
    diagnostic_candidate_rows = [
        annotate_candidate_diagnostics(row)
        for row in candidate_rows
    ]
    candidate_diagnostics = summarize_candidate_diagnostics(candidate_rows, generated_count=generated_count)
    candidate_diagnostics = {
        **candidate_diagnostics,
        "failure_class_counts_by_layer": layer_failure_class_counts(candidate_rows),
    }
    robustness_policy = recommend_failure_policy(
        candidate_rows,
        generated_count=generated_count,
        content_route=content_route,
        layer_attempts=layer_attempts,
    )
    summary = {
        "rewrite_pipeline_version": "rewrite_v2_scan_driven",
        "outcome": public_status,
        "strict_goal_status": final_goal.get("status") if isinstance(final_goal, dict) else public_status,
        "rewrite_goal_status": final_goal,
        "reference_ai": reference_ai,
        "required_ai_drop": required_ai_drop,
        "target_ai_score": target_ai_score,
        "rewrite_effective_config": effective_config,
        "runtime_budget": runtime_budget.to_dict(),
        "candidate_generation_status": {
            "generated_count": generated_count,
            "candidate_rows": len(candidate_rows),
            "scored_candidates": sum(1 for row in candidate_rows if row.get("candidate_ai") is not None),
            "rejected_candidates": sum(
                1
                for row in candidate_rows
                if ((row.get("decision") or {}).get("lane") == CandidateLane.REJECT.value)
            ),
            "reason": generation_reason,
            "diagnostics": candidate_diagnostics,
            "robustness_policy": robustness_policy,
            "layer_attempts": layer_attempts,
            "layer_attempt_summary": summarize_layer_attempts(layer_attempts),
            "post_layer_trace": post_layer_trace,
        },
        "content_router_trace": content_route.to_dict(),
        "strategy_trace": [strategy.to_dict() for strategy in strategies],
        "candidate_trace": [
            {key: value for key, value in row.items() if key not in {"text", "report"}}
            for row in diagnostic_candidate_rows
        ],
        "selected_candidate": {
            key: value for key, value in (best or {}).items() if key not in {"text", "report"}
        } if best else None,
        "diagnostic_candidate_text": (
            str(best.get("text") or "")
            if best and os.environ.get("DRAFTPROOF_REWRITE_V2_EXPOSE_DIAGNOSTIC_TEXT", "0").lower() in {"1", "true", "yes"}
            else None
        ),
        "stage_timings": [{
            "stage": "rewrite_v2_scan_driven",
            "seconds": round(elapsed, 3),
            "candidates": len(candidate_rows),
            "selected": bool(
                public_status == RewriteGoalStatus.AI_MITIGATED.value
                or public_status in {"safe_near_miss_applied", "safe_partial_mitigation_applied"}
                or best_applicable_near_miss
                or best_applicable_close_partial
            ),
            "strict_selected": public_status == RewriteGoalStatus.AI_MITIGATED.value,
            "stop_reason": convergence_reason,
        }],
        "detect_scan_original": original_report,
        "detect_scan_rewritten": final_report,
        "final_text": final_text,
    }
    sentence_comparison = _sentence_comparison(original_text, final_text)
    result_obj = SimpleNamespace(
        summary=summary,
        sentence_comparison=sentence_comparison,
        rewrite_plan=None,
        mp_result=SimpleNamespace(
            original_text=original_text,
            final_text=final_text,
            converged=converged,
            convergence_reason=convergence_reason,
            passes=[],
        ),
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"draftproof_rewrite_v2_{ts}.md"
    pdf_path = out_dir / f"draftproof_rewrite_v2_{ts}.pdf"
    json_path = out_dir / f"draftproof_rewrite_v2_{ts}.json"
    md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
    md_path.write_text(md_text, encoding="utf-8")
    render_pdf(md_text, str(pdf_path))
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    progress(88, "Scan-driven rewrite V2 complete")
    return {
        "status": public_status,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "result": result_obj,
        "elapsed": elapsed,
    }
